"""Prepare LINQS WebKB (Cornell, Texas, Washington) and GraphSAINT Reddit2 for forwardgnn.

Every dataset goes through the same pipeline (DATASET_PREPROCESSING.md explains each choice):

  1. download the released raw files and check their size and SHA-256
  2. parse node ids, features, labels and directed links exactly as released
  3. drop links whose endpoints are not in the node list, drop duplicate links, remove self-loops
  4. make the graph undirected: add every reverse edge, then deduplicate again
  5. keep every node (isolated ones included) and every class; labels become 0..K-1
  6. store the features as float32 without rescaling
  7. when loaded for training: standardise every feature column over all nodes (z-score; no labels
     are used, so this is legitimate in the transductive setting); constant columns become 0

`install()` makes forwardgnn's `load_dataset` return these graphs, and forwardgnn's `DataSplit` builds
the same five random 64/16/20 train/val/test splits it uses for every dataset. Step 7 can be switched
to forwardgnn's own `NormalizeFeatures()` (FGNN_FEATURE_NORM=row_l1) for comparability with runs on
its built-in datasets; DATASET_PREPROCESSING.md explains why that is not the default.

CLI, run from forwardgnn/src inside the training environment:
    python -m fgnn_bench.datasets_ext WebKB-Cornell WebKB-Texas WebKB-Washington Reddit2 \
        --raw-dir /content/fgnn_raw --splits --report /path/to/dataset_report.md
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import coalesce, is_undirected, to_undirected

import settings

WEBKB_URL = 'https://linqs-data.soe.ucsc.edu/public/lbc/WebKB.tgz'
WEBKB_SIZE = 132209
WEBKB_SHA256 = '62630a2427e5aeffc1158ad4e6ee275a0dacdcff9ef8b4506aef9c4805e75ab4'
WEBKB_CLASSES = ['course', 'faculty', 'project', 'staff', 'student']  # label id = list index

# GraphSAINT's release of Reddit, the same Google Drive files torch_geometric.datasets.Reddit2 uses.
# PyG 2.2.0 downloads them from docs.google.com/uc, which now answers the two large files with an HTML
# virus-scan page; drive.usercontent.google.com serves the bytes directly.
GDRIVE_URL = 'https://drive.usercontent.google.com/download?id={}&confirm=t'
REDDIT2_FILES = {  # name: (Drive id, size in bytes, SHA-256)
    'adj_full.npz': ('1sncK996BM5lpuDf75lDFqCiDZyErc1c2', 64866285,
                     '2e190c5216ead7018989fd26502c1455782abb39d3b316b714b0cbc8ced2a2dd'),
    'feats.npy': ('1ZsHaJ0ussP1W722krmEIp_8pwKAoi5b3', 1121959520,
                  '5a8a381f5bf80b06d5555cbf0aa5521958fed191c00400e02f4a9ea7d4bb2f41'),
    'class_map.json': ('1JF3Pjv9OboMNYs2aXRQGbJbc4t_nDd5u', 3089824,
                       'b15bb00caee03ee3b83dd382ce3470f5c677f53e75c53af2b1231fe6376a8487'),
    'role.json': ('1nJIKd77lcAGU4j-kVNx_AIGEkveIKz3A', 1752634,
                  '5efd0e9e75be6ddfc8411236e9b22d3b241c992d6311060f93f72091075ef518'),
}

# benchmark name -> WebKB university (None for Reddit2)
DATASETS = {
    'WebKB-Cornell': 'cornell',
    'WebKB-Texas': 'texas',
    'WebKB-Washington': 'washington',
    'Reddit2': None,
}

PIPELINE = [
    'download the released raw files; verify size and SHA-256',
    'parse nodes, features, labels and directed links as released',
    'drop links with unknown endpoints and duplicate links; remove self-loops',
    'symmetrise (add reverse edges) and deduplicate',
    'keep all nodes (incl. isolated) and all classes; labels mapped to 0..K-1',
    'features stored as float32, unscaled',
    'at load time: column-wise z-score over all nodes (FGNN_FEATURE_NORM, default standardize)',
    'splits: forwardgnn DataSplit, KFold(5, shuffle, random_state=101) -> 64/16/20 train/val/test',
]


class StandardizeFeatures:
    """Column-wise z-score over all nodes, (x - mean) / std with the population std; columns that are
    constant over the graph become 0. Uses no labels and no split information."""

    def __call__(self, data):
        mean = data.x.mean(dim=0, keepdim=True)
        std = data.x.std(dim=0, unbiased=False, keepdim=True)
        data.x = (data.x - mean) / torch.where(std > 0, std, torch.ones_like(std))
        return data

    def __repr__(self):
        return 'StandardizeFeatures()'


FEATURE_NORMS = {  # FGNN_FEATURE_NORM value -> transform applied every time the graph is loaded
    'standardize': StandardizeFeatures,
    'row_l1': NormalizeFeatures,  # forwardgnn's transform for CitationFull / Amazon / GitHub
    'none': None,
}


def feature_norm():
    name = os.environ.get('FGNN_FEATURE_NORM', 'standardize')
    if name not in FEATURE_NORMS:
        raise ValueError(f'FGNN_FEATURE_NORM={name!r}; choose from {sorted(FEATURE_NORMS)}')
    return name


def dataset_root(name):
    return Path(settings.DATA_ROOT) / 'fgnn_bench' / name


# ----------------------------------------------------------------------------------------- download

def _sha256(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def _fetch(url, dest, size, sha256, retries=3):
    """Download `url` to `dest` unless a verified copy is already there; return the path."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size == size and _sha256(dest) == sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.part')
    for attempt in range(1, retries + 1):
        try:
            print(f'downloading {dest.name} ({size / 2 ** 20:.1f} MiB)', flush=True)
            request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(request, timeout=300) as response, open(part, 'wb') as f:
                shutil.copyfileobj(response, f, 1 << 22)
            got_size, got_sha = part.stat().st_size, _sha256(part)
            if got_size != size or got_sha != sha256:
                raise IOError(f'{dest.name}: expected {size} bytes with sha256 {sha256[:12]}..., '
                              f'got {got_size} bytes with sha256 {got_sha[:12]}...')
            os.replace(part, dest)
            return dest
        except Exception as exc:  # network hiccups, Drive quota pages, truncated files
            print(f'  attempt {attempt}/{retries} failed: {exc}', flush=True)
            if attempt == retries:
                raise
            time.sleep(10 * attempt)


# ------------------------------------------------------------------------------------------- parse

def _read_webkb(raw_dir, university):
    archive = _fetch(WEBKB_URL, Path(raw_dir) / 'WebKB.tgz', WEBKB_SIZE, WEBKB_SHA256)
    with tarfile.open(archive, 'r:gz') as tar:  # read members in memory; nothing is extracted
        content = tar.extractfile(f'webkb/{university}.content').read().decode('utf-8')
        cites = tar.extractfile(f'webkb/{university}.cites').read().decode('utf-8')

    # .content: "<page url> <1703 binary word indicators> <class>", tab separated
    node_ids, rows, labels = [], [], []
    for line in content.splitlines():
        if line.strip():
            fields = line.split('\t')
            node_ids.append(fields[0])
            rows.append(fields[1:-1])
            labels.append(fields[-1].strip())
    index = {node: i for i, node in enumerate(node_ids)}
    if len(index) != len(node_ids):
        raise ValueError(f'{university}: duplicate page ids')
    x = np.asarray(rows, dtype=np.float32)
    y = np.asarray([WEBKB_CLASSES.index(label) for label in labels], dtype=np.int64)

    # .cites: "<cited page> <citing page>"; the hyperlink points from the citing to the cited page
    src, dst, n_lines, n_dangling = [], [], 0, 0
    for line in cites.splitlines():
        if line.strip():
            cited, citing = line.split()
            n_lines += 1
            if cited in index and citing in index:
                src.append(index[citing])
                dst.append(index[cited])
            else:
                n_dangling += 1
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    source = {
        'origin': 'WebKB, CMU World Wide Knowledge Base project (Craven et al., AAAI 1998)',
        'release': f'LINQS WebKB release, files webkb/{university}.content and webkb/{university}.cites',
        'files': {'WebKB.tgz': {'url': WEBKB_URL, 'bytes': WEBKB_SIZE, 'sha256': WEBKB_SHA256}},
    }
    raw = {'nodes': len(node_ids), 'links': n_lines, 'links_with_unknown_endpoint': n_dangling}
    return x, y, edge_index, list(WEBKB_CLASSES), source, raw


def _read_reddit2(raw_dir):
    paths = {name: _fetch(GDRIVE_URL.format(file_id), Path(raw_dir) / name, size, sha256)
             for name, (file_id, size, sha256) in REDDIT2_FILES.items()}

    adj = np.load(paths['adj_full.npz'])  # scipy CSR arrays: data, indices, indptr, shape
    n = int(adj['shape'][0])
    indptr, indices = adj['indptr'].astype(np.int64), adj['indices'].astype(np.int64)
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    stored = adj['data'] != 0
    edge_index = torch.from_numpy(np.stack([rows[stored], indices[stored]]))

    x = np.load(paths['feats.npy']).astype(np.float32)  # released as float64
    with open(paths['class_map.json']) as f:
        class_map = json.load(f)
    y = np.full(n, -1, dtype=np.int64)
    for node, label in class_map.items():
        y[int(node)] = int(label)
    if (y < 0).any():
        raise ValueError('Reddit2: some nodes have no label')
    with open(paths['role.json']) as f:
        role = json.load(f)

    source = {
        'origin': 'Reddit posts, Sept. 2014 (Hamilton et al., GraphSAGE, NeurIPS 2017)',
        'release': 'GraphSAINT sparsified version (Zeng et al., ICLR 2020) = torch_geometric.datasets.Reddit2',
        'files': {name: {'url': GDRIVE_URL.format(file_id), 'bytes': size, 'sha256': sha256}
                  for name, (file_id, size, sha256) in REDDIT2_FILES.items()},
    }
    raw = {'nodes': n, 'links': int(stored.sum()), 'links_with_unknown_endpoint': 0,
           'official_split_not_used': {part: len(ids) for part, ids in role.items()}}
    label_names = [f'class_{k}' for k in range(int(y.max()) + 1)]
    return x, y, edge_index, label_names, source, raw


# ------------------------------------------------------------------------------------------- clean

def clean_edges(edge_index, num_nodes):
    """Steps 3-4 of the pipeline; returns the undirected edge_index and what was removed."""
    stats = {'links_parsed': int(edge_index.size(1))}
    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    stats['duplicate_links'] = stats['links_parsed'] - int(edge_index.size(1))
    loops = edge_index[0] == edge_index[1]
    stats['self_loops'] = int(loops.sum())
    edge_index = edge_index[:, ~loops]
    directed = int(edge_index.size(1))
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    undirected = int(edge_index.size(1)) // 2
    stats['reciprocal_link_pairs'] = directed - undirected
    stats['undirected_edges_incl_self_loops'] = undirected + stats['self_loops']
    stats['undirected_edges'] = undirected
    stats['edge_index_entries'] = int(edge_index.size(1))
    assert is_undirected(edge_index, num_nodes=num_nodes)
    assert not bool((edge_index[0] == edge_index[1]).any())
    return edge_index, stats


def _feature_stats(x):
    nonzero = x != 0
    row_sums = x.sum(1)
    stats = {
        'dim': int(x.shape[1]),
        'binary': bool(np.isin(x, (0.0, 1.0)).all()),
        'min': float(x.min()), 'max': float(x.max()), 'mean': float(x.mean()),
        'nonzero_fraction': float(nonzero.mean()),
        'all_zero_rows': int((~nonzero.any(1)).sum()),
        'constant_columns': int((x.max(0) == x.min(0)).sum()),
        'row_sum_min': float(row_sums.min()), 'row_sum_max': float(row_sums.max()),
    }
    if not stats['binary']:
        widest = np.argsort(-np.abs(x).max(0))[:5]
        stats['largest_magnitude_columns'] = [
            {'column': int(c), 'min': float(x[:, c].min()), 'max': float(x[:, c].max())} for c in widest]
    return stats


def _transformed_feature_stats(x):
    """What each FEATURE_NORMS option hands the model. `median_top_column_share` is the median over
    nodes of max_j |x_ij| / ||x_i||_2: how much a single column dominates a row's direction (the
    single-forward layers L2-normalise every row before using it)."""
    out = {}
    for name, transform in FEATURE_NORMS.items():
        if transform is None:
            continue
        z = transform()(Data(x=torch.from_numpy(x).clone())).x
        share = z.abs().max(dim=1).values / z.norm(dim=1).clamp(min=1e-12)
        out[name] = {'min': float(z.min()), 'max': float(z.max()), 'mean_abs': float(z.abs().mean()),
                     'median_top_column_share': float(share.median())}
    return out


def graph_stats(x, y, edge_index, label_names):
    n = x.shape[0]
    degree = torch.bincount(edge_index[0], minlength=n).numpy()
    same_label = (torch.from_numpy(y)[edge_index[0]] == torch.from_numpy(y)[edge_index[1]])
    return {
        'nodes': int(n),
        'undirected_edges': int(edge_index.size(1)) // 2,
        'edge_index_entries': int(edge_index.size(1)),
        'isolated_nodes': int((degree == 0).sum()),
        'degree_mean': float(degree.mean()), 'degree_median': float(np.median(degree)),
        'degree_max': int(degree.max()),
        'edge_homophily': float(same_label.float().mean()) if edge_index.size(1) else float('nan'),
        'num_classes': len(label_names),
        'label_names': label_names,
        'class_counts': np.bincount(y, minlength=len(label_names)).tolist(),
        'features': _feature_stats(x),
        'features_after_transform': _transformed_feature_stats(x),
    }


# ------------------------------------------------------------------------------------- prepare/load

def prepare(name, raw_root=None, force=False):
    """Run the pipeline for `name`; writes <root>/processed/data.pt and <root>/stats.json.

    Raw downloads go to <raw_root>/<name> (raw_root defaults to $FGNN_RAW_DIR, else <root>/raw is
    used directly). They are not needed again once the processed file exists."""
    if name not in DATASETS:
        raise ValueError(f'unknown dataset {name!r}; choose from {sorted(DATASETS)}')
    root = dataset_root(name)
    processed, stats_path = root / 'processed' / 'data.pt', root / 'stats.json'
    if processed.exists() and stats_path.exists() and not force:
        return json.loads(stats_path.read_text())

    raw_root = raw_root or os.environ.get('FGNN_RAW_DIR')
    raw_dir = Path(raw_root) / name if raw_root else root / 'raw'
    if DATASETS[name] is None:
        x, y, edge_index, label_names, source, raw = _read_reddit2(raw_dir)
    else:
        x, y, edge_index, label_names, source, raw = _read_webkb(raw_dir, DATASETS[name])
    if not (y.min() == 0 and len(np.unique(y)) == len(label_names)):
        raise ValueError(f'{name}: labels must cover 0..{len(label_names) - 1}')

    edge_index, cleaning = clean_edges(edge_index, num_nodes=x.shape[0])
    stats = {
        'dataset': name,
        'source': source,
        'raw': raw,
        'cleaning': cleaning,
        'processed': graph_stats(x, y, edge_index, label_names),
        'pipeline': PIPELINE,
        'versions': {'torch': torch.__version__, 'torch_geometric': torch_geometric.__version__,
                     'numpy': np.__version__, 'python': sys.version.split()[0]},
        'prepared_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }

    data = Data(x=torch.from_numpy(x), edge_index=edge_index, y=torch.from_numpy(y))
    processed.parent.mkdir(parents=True, exist_ok=True)
    tmp = processed.with_name('data.pt.tmp')
    torch.save(InMemoryDataset.collate([data]), tmp)
    os.replace(tmp, processed)
    stats_path.write_text(json.dumps(stats, indent=2))
    return stats


class PreparedNodeDataset(InMemoryDataset):
    """The single graph written by prepare(). It defines no download()/process(), so PyG never
    looks for raw files once the processed file exists."""

    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']


_LOADED = {}


def load(name):
    """The prepared dataset with the FGNN_FEATURE_NORM transform, cached per process.

    Indexing the dataset returns a shallow copy with a freshly transformed `x`, so the per-run masks
    and device moves done by forwardgnn never leak into the cached graph."""
    key = (name, feature_norm())
    if key not in _LOADED:
        root = dataset_root(name)
        if not (root / 'processed' / 'data.pt').exists():
            prepare(name)
        transform = FEATURE_NORMS[feature_norm()]
        dataset = PreparedNodeDataset(str(root), transform=transform() if transform else None)
        assert len(dataset) == 1, len(dataset)
        assert is_undirected(dataset.data.edge_index, num_nodes=dataset.data.num_nodes)
        _LOADED[key] = dataset
    return _LOADED[key]


def install():
    """Let forwardgnn's load_dataset (and so DataSplit and the training scripts) see these datasets."""
    import datasets
    import datasets.dataloader as dataloader

    original = dataloader.load_dataset
    if getattr(original, 'fgnn_bench', False):
        return

    def load_dataset(dataset):
        return load(dataset) if dataset in DATASETS else original(dataset)

    load_dataset.fgnn_bench = True
    dataloader.load_dataset = load_dataset
    datasets.load_dataset = load_dataset


def split_report(name, num_splits=5):
    """Sizes and class counts of forwardgnn's node splits (generated on first use, then reused)."""
    install()
    from datasets.datasplit import DataSplit

    y = load(name).data.y
    num_classes = int(y.max()) + 1
    splitter = DataSplit(dataset_name=name, num_splits=num_splits)
    report = []
    for split_i in range(num_splits):
        split = splitter.load_node_split(split_i=split_i)
        row = {'split': split_i}
        for part in ('train', 'val', 'test'):
            idx = split[f'{part}_node_index']
            row[part] = int(idx.numel())
            row[f'{part}_class_counts'] = torch.bincount(y[idx], minlength=num_classes).tolist()
        report.append(row)
    return report


# ------------------------------------------------------------------------------------------ report

def _fmt(v):
    return f'{v:,}' if isinstance(v, int) else (f'{v:.3f}' if isinstance(v, float) else str(v))


def markdown_report(names, splits=None):
    """Markdown tables of the raw -> processed numbers (and split sizes when `splits` is given)."""
    stats = {name: prepare(name) for name in names}
    out = ['## Raw files -> cleaned graph', '',
           '| | ' + ' | '.join(names) + ' |', '|---|' + '---|' * len(names)]
    rows = [
        ('nodes (raw = processed)', lambda s: s['processed']['nodes']),
        ('links in the release', lambda s: s['raw']['links']),
        ('links with unknown endpoint (dropped)', lambda s: s['raw']['links_with_unknown_endpoint']),
        ('duplicate links (dropped)', lambda s: s['cleaning']['duplicate_links']),
        ('self-loops (dropped)', lambda s: s['cleaning']['self_loops']),
        ('reciprocal link pairs (merged)', lambda s: s['cleaning']['reciprocal_link_pairs']),
        ('undirected edges incl. self-loops (survey count)',
         lambda s: s['cleaning']['undirected_edges_incl_self_loops']),
        ('**undirected edges used**', lambda s: s['processed']['undirected_edges']),
        ('edge_index entries (both directions)', lambda s: s['processed']['edge_index_entries']),
        ('isolated nodes', lambda s: s['processed']['isolated_nodes']),
        ('mean / max degree', lambda s: f"{s['processed']['degree_mean']:.2f} / "
                                        f"{s['processed']['degree_max']:,}"),
        ('edge homophily', lambda s: s['processed']['edge_homophily']),
        ('features (binary?)', lambda s: f"{s['processed']['features']['dim']} "
                                         f"({'yes' if s['processed']['features']['binary'] else 'no'})"),
        ('classes', lambda s: s['processed']['num_classes']),
        ('smallest / largest class', lambda s: f"{min(s['processed']['class_counts']):,} / "
                                               f"{max(s['processed']['class_counts']):,}"),
    ]
    for label, get in rows:
        out.append(f'| {label} | ' + ' | '.join(_fmt(get(stats[n])) for n in names) + ' |')

    out += ['', '## Class counts', '']
    for name in names:
        p = stats[name]['processed']
        pairs = ', '.join(f'{k}: {c:,}' for k, c in zip(p['label_names'], p['class_counts']))
        out.append(f'- **{name}**: {pairs}')

    out += ['', '## Raw features', '',
            '| dataset | dim | binary | min | max | nonzero frac. | constant columns | row-sum range |',
            '|---|---|---|---|---|---|---|---|']
    for name in names:
        f = stats[name]['processed']['features']
        out.append(f"| {name} | {f['dim']} | {'yes' if f['binary'] else 'no'} | {f['min']:.4g} | "
                   f"{f['max']:.4g} | {f['nonzero_fraction']:.4f} | {f.get('constant_columns', 'n/a')} | "
                   f"{f['row_sum_min']:.4g} to {f['row_sum_max']:.4g} |")
        for c in f.get('largest_magnitude_columns', [])[:2]:
            out.append(f"|  column {c['column']} | | | {c['min']:.4g} | {c['max']:.4g} | | | |")

    out += ['', '## Features after each load-time transform', '',
            '`median_top_column_share` = median over nodes of max_j |x_ij| / ||x_i||_2 '
            '(how much one column dominates a row; single-forward layers L2-normalise rows).', '',
            '| dataset | transform | min | max | mean abs | median top-column share |', '|---|---|---|---|---|---|']
    for name in names:
        for transform, g in stats[name]['processed']['features_after_transform'].items():
            out.append(f"| {name} | {transform} | {g['min']:.4g} | {g['max']:.4g} | {g['mean_abs']:.4g} | "
                       f"{g['median_top_column_share']:.3f} |")

    if splits:
        out += ['', '## forwardgnn node splits (train / val / test)', '']
        for name in names:
            label_names = stats[name]['processed']['label_names']
            sizes = ', '.join(f"{r['train']}/{r['val']}/{r['test']}" for r in splits[name])
            out.append(f'- **{name}** splits 0-4: {sizes}')
            for r in splits[name]:
                missing = [label_names[k] for k, c in enumerate(r['train_class_counts']) if c == 0]
                if missing:
                    out.append(f"  - split {r['split']}: no training node of class {', '.join(missing)}")
    return '\n'.join(out) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('names', nargs='+', choices=sorted(DATASETS))
    parser.add_argument('--raw-dir', default=None,
                        help='raw downloads go to <raw-dir>/<name> (default: $FGNN_RAW_DIR, else <root>/raw)')
    parser.add_argument('--force', action='store_true', help='re-run the pipeline even if processed')
    parser.add_argument('--splits', action='store_true', help='also generate/report the 5 node splits')
    parser.add_argument('--report', default=None, help='write a markdown report to this path')
    args = parser.parse_args(argv)

    for name in args.names:
        start = time.time()
        stats = prepare(name, raw_root=args.raw_dir, force=args.force)
        p = stats['processed']
        print(f"{name}: {p['nodes']:,} nodes, {p['undirected_edges']:,} undirected edges, "
              f"{p['features']['dim']} features, {p['num_classes']} classes "
              f"({time.time() - start:.0f}s) -> {dataset_root(name)}", flush=True)

    splits = {name: split_report(name) for name in args.names} if args.splits else None
    report = markdown_report(args.names, splits)
    print()
    print(report)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(report)


if __name__ == '__main__':
    main()