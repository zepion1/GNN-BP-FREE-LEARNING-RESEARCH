#!/usr/bin/env python3
"""Load every benchmark dataset through forwardgnn's own loader and check its splits.
Run on a COMPUTE node (Reddit2 needs several GB of RAM):

    source $FGNN_ROOT/activate.sh && $FGNN_PY $FGNN_ROOT/check_data.py

For each dataset: nodes, edges, features, classes, and for each of the 5 splits whether the split file
was already on disk or generated now (both fine: forwardgnn's KFold uses random_state=101, and the raw
files are SHA-256-verified, so regenerated splits equal the Colab ones), plus a fingerprint of its test nodes.
"""
import hashlib
import os
import sys

src = os.environ['FGNN_SRC']
os.chdir(src)
sys.path.insert(0, src)
os.environ.setdefault('FGNN_FEATURE_NORM', 'standardize')

from fgnn_bench import datasets_ext          # noqa: E402
datasets_ext.install()
import datasets.dataloader as dl              # noqa: E402
from datasets.datasplit import DataSplit      # noqa: E402

NAMES = sys.argv[1:] or ['CitationFull-PubMed', 'Reddit2',
                         'WebKB-Cornell', 'WebKB-Texas', 'WebKB-Washington']
problems = 0
for name in NAMES:
    print(f'\n== {name}')
    try:
        d = dl.load_dataset(name)[0]
    except Exception as e:
        problems += 1
        print(f'   LOAD FAILED: {type(e).__name__}: {e}')
        if 'urlopen' in str(e) or 'URLError' in type(e).__name__:
            print('   -> needs a download; copy its data/ folder from Drive, or load it once on the login node')
        continue
    print(f'   nodes {d.num_nodes:,}  directed edges {d.edge_index.size(1):,}  '
          f'features {d.num_features:,}  classes {int(d.y.max()) + 1}')
    split = DataSplit(dataset_name=name, num_splits=5)
    root = split.node_split_root
    existed = root.exists() and any(root.iterdir())
    for i in range(5):
        s = split.load_node_split(split_i=i)
        test = sorted(int(x) for x in s['test_node_index'])
        fp = hashlib.sha1(','.join(map(str, test)).encode()).hexdigest()[:10]
        print(f'   split {i}: train {len(s["train_node_index"]):>7,}  val {len(s["val_node_index"]):>6,}  '
              f'test {len(test):>6,}  test-fingerprint {fp}')
    print(f'   splits: {"already on disk" if existed else "generated now"}')

print('\nALL DATASETS OK' if not problems else f'\n{problems} problem(s) -- see above')
sys.exit(1 if problems else 0)
