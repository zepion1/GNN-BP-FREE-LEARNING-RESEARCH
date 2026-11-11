#!/usr/bin/env python3
"""Emit the job grid as a TSV for a scheduler array, skipping configurations that are already done.

    $FGNN_PY gen_jobs.py --records-root $FGNN_ROOT/bench_records --out jobs.tsv

Three groups:
  reddit  tier 1  Reddit2              -> bench_records/webkb-reddit-sparse-standardize/  (your notebook's folder)
  pubmed  tier 1  CitationFull-PubMed  -> bench_records/pubmed-sparse-row_l1/              (row_l1: see note)
  webkb   tier 2  Cornell/Texas/Wash.  -> bench_records/webkb-reddit-sparse-standardize/

Every group lists depths 1-4 AND 6/8/12/16. A configuration whose 5 runs already have "ok" records is
skipped, so depths 1-4 you finished on Colab are not re-run, while the unfinished GAT Reddit2 runs are.
Copy your Colab bench_records/ to the cluster BEFORE running this.

PubMed note: forwardgnn's built-in datasets always use NormalizeFeatures (row-L1), whatever
--feature-norm says (run_bench.py prints a warning). Passing row_l1 makes the records say so truthfully,
and a separate folder keeps the notebook's one-feature-norm-per-folder check valid. Your earlier PubMed
results came from a different harness (no macro-F1, memory counter reset per layer), so PubMed is run at
all depths here; it is cheap.
"""
import argparse
import itertools
import json
from collections import Counter, OrderedDict
from pathlib import Path

FORWARD_FLAGS = ['--append-label', 'none', '--aug-edge-direction', 'bidirection']
METHOD_SPECS = OrderedDict([
    ('BP',           ('train_backprop.py', 'GNN-{arch}', [])),
    ('SF',           ('train_forward.py', 'GNN_SingleForward-{arch}', FORWARD_FLAGS)),
    ('SF-Top2Input', ('train_forward.py', 'GNN_Top2Input_SingleForward-{arch}',
                      FORWARD_FLAGS + ['--alternating-update'])),
    ('SF-Top2Loss',  ('train_forward.py', 'GNN_Top2Loss_SingleForward-{arch}',
                      FORWARD_FLAGS + ['--alternating-update'])),
])
HPARAMS = dict(epochs=1000, val_every=2, lr=0.001, patience=100, num_hidden=128, seed=100)
ARCHS = ['GCN', 'SAGE', 'GAT']
ALL_DEPTHS = [1, 2, 3, 4, 6, 8, 12, 16]

GROUPS = OrderedDict([
    ('reddit', dict(tier=1, datasets=['Reddit2'],
                    exp_setting='webkb-reddit-sparse-standardize', feature_norm='standardize')),
    ('pubmed', dict(tier=1, datasets=['CitationFull-PubMed'],
                    exp_setting='pubmed-sparse-row_l1', feature_norm='row_l1')),
    ('webkb',  dict(tier=2, datasets=['WebKB-Cornell', 'WebKB-Texas', 'WebKB-Washington'],
                    exp_setting='webkb-reddit-sparse-standardize', feature_norm='standardize')),
])

# resource bands: walltime, host RAM. Check your partition's max walltime and adjust.
CLASSES = OrderedDict([
    ('small',  ('02:00:00', '16G')),
    ('medium', ('08:00:00', '32G')),
    ('large',  ('24:00:00', '64G')),
    ('xlarge', ('48:00:00', '96G')),
])


def est_class(dataset, arch, method, depth):
    if dataset.startswith('WebKB'):
        return 'small'
    if dataset == 'CitationFull-PubMed':
        if depth <= 8:
            return 'small'
        return 'medium'
    # Reddit2
    if arch == 'GAT' or method == 'SF-Top2Input':
        return 'xlarge'
    return 'large' if depth <= 8 else 'xlarge'


def config_status(records_root, exp_setting, dataset, model, depth, num_runs):
    d = Path(records_root) / exp_setting / dataset / model / f'L{depth}'
    sts = []
    for i in range(num_runs):
        p = d / f'run{i}.json'
        if not p.exists():
            sts.append(None)
            continue
        try:
            sts.append(json.loads(p.read_text()).get('status'))
        except (ValueError, OSError):
            sts.append('unreadable')
    if all(s == 'ok' for s in sts):
        return 'complete'
    if any(s == 'oom' for s in sts):
        return 'oom'
    if any(s in ('error', 'unreadable') for s in sts):
        return 'error'
    if any(s == 'ok' for s in sts):
        return 'partial'
    return 'missing'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--records-root', required=True,
                    help='the bench_records/ directory (holding one folder per exp-setting)')
    ap.add_argument('--out', default='jobs.tsv')
    ap.add_argument('--groups', nargs='*', default=list(GROUPS), choices=list(GROUPS))
    ap.add_argument('--depths', nargs='*', type=int, default=ALL_DEPTHS)
    ap.add_argument('--num-runs', type=int, default=5)
    ap.add_argument('--mp-backend', default='sparse')
    ap.add_argument('--retry-failed', action='store_true',
                    help='re-emit configurations whose last attempt was OOM or error')
    a = ap.parse_args()

    root = Path(a.records_root)
    if not root.is_dir():
        raise SystemExit(f'{root} does not exist. Copy your Colab bench_records/ there first, '
                         'or create it empty if you really want every configuration re-run.')

    jobs, seen = [], Counter()
    for gname in a.groups:
        g = GROUPS[gname]
        for ds, arch, method, depth in itertools.product(g['datasets'], ARCHS, METHOD_SPECS, a.depths):
            script, model, extra = METHOD_SPECS[method]
            model = model.format(arch=arch)
            st = config_status(root, g['exp_setting'], ds, model, depth, a.num_runs)
            seen[(gname, st)] += 1
            if st == 'complete' or (st in ('oom', 'error') and not a.retry_failed):
                continue
            cmd = ['-m', 'fgnn_bench.run_bench',
                   '--records-dir', f"RECORDS_ROOT/{g['exp_setting']}",
                   '--mp-backend', a.mp_backend, '--feature-norm', g['feature_norm'],
                   '--', script,
                   '--exp-setting', g['exp_setting'], '--dataset', ds, '--task', 'node-class',
                   '--model', model, '--num-layers', str(depth),
                   '--num-runs', str(a.num_runs), '--seed', str(HPARAMS['seed']),
                   '--epochs', str(HPARAMS['epochs']), '--val-every', str(HPARAMS['val_every']),
                   '--lr', str(HPARAMS['lr']), '--patience', str(HPARAMS['patience']),
                   '--num-hidden', str(HPARAMS['num_hidden'])] + extra
            jobs.append(dict(tier=g['tier'], group=gname, dataset=ds, arch=arch, method=method,
                             depth=depth, est=est_class(ds, arch, method, depth), status=st,
                             cmd=' '.join(cmd)))

    rank = {c: i for i, c in enumerate(CLASSES)}
    jobs.sort(key=lambda j: (j['tier'], rank[j['est']], j['dataset'], j['arch'], j['method'], j['depth']))

    with open(a.out, 'w') as f:
        f.write('idx\ttier\tdataset\tarch\tmethod\tdepth\test_class\tcommand\n')
        for i, j in enumerate(jobs, 1):
            j['idx'] = i
            f.write(f"{i}\t{j['tier']}\t{j['dataset']}\t{j['arch']}\t{j['method']}\t"
                    f"{j['depth']}\t{j['est']}\t{j['cmd']}\n")

    print('Existing records found:')
    for gname in a.groups:
        parts = ', '.join(f'{st} {n}' for (g, st), n in sorted(seen.items()) if g == gname)
        print(f'  {gname:<7} {parts}')
    print(f'\n{len(jobs)} configurations to run (x {a.num_runs} runs) -> {a.out}')
    if not jobs:
        return

    print('\nSubmission plan (one line per tier/resource band; indices are contiguous):')
    k = 0
    while k < len(jobs):
        t, c = jobs[k]['tier'], jobs[k]['est']
        e = k
        while e + 1 < len(jobs) and jobs[e + 1]['tier'] == t and jobs[e + 1]['est'] == c:
            e += 1
        wall, mem = CLASSES[c]
        lo, hi = jobs[k]['idx'], jobs[e]['idx']
        print(f'  tier {t} {c:<6} {hi - lo + 1:>3} jobs: '
              f'sbatch --array={lo}-{hi}%8 --time={wall} --mem={mem} run_array.sbatch')
        k = e + 1


if __name__ == '__main__':
    main()
