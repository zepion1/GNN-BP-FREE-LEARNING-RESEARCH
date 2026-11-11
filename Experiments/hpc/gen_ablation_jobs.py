#!/usr/bin/env python3
"""Job list for the virtual-node ablation of single forward, in the same TSV format run_array.sbatch reads.

    $FGNN_PY gen_ablation_jobs.py --records-root $FGNN_RECORDS_ROOT --out jobs_ablation.tsv

Variants (GCN and SAGE; depths 1, 2, 3, 4, 6, 8, 12, 16; 5 runs; same hyperparameters as the main runs):
  SF-uni   GNN_SingleForward-<arch>      --aug-edge-direction unidirection  (no label flow back into the graph)
  SF-noVN  GNN_SingleForwardNoVN-<arch>  no virtual nodes; learnable class prototypes per layer

Backprop and single forward are NOT re-run: you already have them in the main folders.

Results go to separate folders so nothing collides with the main runs (SF-uni has the same model name as SF,
so in the main folder its runs would be skipped as already done):
  PubMed   bench_records/pubmed-sparse-row_l1-ablation             (row_l1, as the main PubMed runs)
  Reddit2  bench_records/webkb-reddit-sparse-standardize-ablation  (standardize, as the main Reddit2 runs)

Configurations whose 5 runs are already 'ok' are skipped, so re-running this after a time-out gives round 2.
"""
import argparse
import itertools
import json
from collections import OrderedDict
from pathlib import Path

HP = dict(epochs=1000, val_every=2, lr=0.001, patience=100, num_hidden=128, seed=100)
DEPTHS = [1, 2, 3, 4, 6, 8, 12, 16]
ARCHS = ['GCN', 'SAGE']
VARIANTS = OrderedDict([
    ('SF-uni',  ('GNN_SingleForward-{a}',     'unidirection')),
    ('SF-noVN', ('GNN_SingleForwardNoVN-{a}', 'bidirection')),  # direction is irrelevant: no virtual edges
])
DATASETS = OrderedDict([  # dataset: (exp-setting / records folder, feature-norm, partition, --time, --mem)
    ('CitationFull-PubMed', ('pubmed-sparse-row_l1-ablation', 'row_l1', 'gpu-day-long', '08:00:00', '32G')),
    ('Reddit2', ('webkb-reddit-sparse-standardize-ablation', 'standardize', 'gpu-day-long', '1-00:00:00', '96G')),
])


def complete(records_root, exp, ds, model, depth, runs):
    d = Path(records_root) / exp / ds / model / f'L{depth}'
    for i in range(runs):
        p = d / f'run{i}.json'
        try:
            if json.loads(p.read_text()).get('status') != 'ok':
                return False
        except (OSError, ValueError):
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--records-root', required=True)
    ap.add_argument('--out', default='jobs_ablation.tsv')
    ap.add_argument('--num-runs', type=int, default=5)
    ap.add_argument('--throttle', type=int, default=4)
    a = ap.parse_args()
    if not Path(a.records_root).is_dir():
        raise SystemExit(f'{a.records_root} does not exist')

    jobs, skipped = [], 0
    for ds, (exp, fnorm, part, wall, mem) in DATASETS.items():
        for arch, (variant, (tmpl, direction)), depth in itertools.product(ARCHS, VARIANTS.items(), DEPTHS):
            model = tmpl.format(a=arch)
            if complete(a.records_root, exp, ds, model, depth, a.num_runs):
                skipped += 1
                continue
            cmd = ['-m', 'fgnn_bench.run_bench', '--records-dir', f'RECORDS_ROOT/{exp}',
                   '--mp-backend', 'sparse', '--feature-norm', fnorm, '--', 'train_forward.py',
                   '--exp-setting', exp, '--dataset', ds, '--task', 'node-class', '--model', model,
                   '--num-layers', str(depth), '--num-runs', str(a.num_runs), '--seed', str(HP['seed']),
                   '--epochs', str(HP['epochs']), '--val-every', str(HP['val_every']), '--lr', str(HP['lr']),
                   '--patience', str(HP['patience']), '--num-hidden', str(HP['num_hidden']),
                   '--append-label', 'none', '--aug-edge-direction', direction]
            jobs.append((ds, arch, variant, depth, ' '.join(cmd)))

    with open(a.out, 'w') as f:
        f.write('idx\ttier\tdataset\tarch\tmethod\tdepth\test_class\tcommand\n')
        for i, (ds, arch, variant, depth, cmd) in enumerate(jobs, 1):
            f.write(f'{i}\t1\t{ds}\t{arch}\t{variant}\t{depth}\tablation\t{cmd}\n')

    print(f'{len(jobs)} configurations to run (x {a.num_runs} runs) -> {a.out}; {skipped} already complete')
    print('\nSubmission commands (run from $FGNN_ROOT):')
    for ds, (exp, fnorm, part, wall, mem) in DATASETS.items():
        idx = [i for i, j in enumerate(jobs, 1) if j[0] == ds]
        if idx:
            print(f'\n# {ds}: {len(idx)} config(s) -> bench_records/{exp}')
            print(f'FGNN_JOBS=$PWD/{a.out} sbatch --partition={part} --array={idx[0]}-{idx[-1]}%{a.throttle} '
                  f'--time={wall} --mem={mem} run_array.sbatch')


if __name__ == '__main__':
    main()
