#!/usr/bin/env python3
"""Layer-by-layer view of single-forward training, read from the job logs. No GPU and no new runs needed.

    $FGNN_PY extract_layers.py --logs logs --out reports/layers

For every single-forward-style run (SF, SF-uni and SF-noVN by default) it records, for each layer:
  stop_epoch  epoch at which early stopping ended that layer's training (blank: it ran all epochs)
  best_epoch  epoch whose weights the layer kept = stop_epoch - patience * val_every
  reset       best_epoch < --reset-before (default 10): the layer was restored to its weights from its first
              few epochs, i.e. it never improved the combined prediction and is essentially untrained
  cum_acc     test accuracy of the combined prediction after this layer (the class scores of layers 0..k
              summed), as printed by forwardgnn in "[Layer-k] Test Accuracy"

Writes to --out:
  layers.csv          one row per run and layer
  summary.csv / .md   per configuration: complete runs, mean number of later layers reset, accuracy after the
                      first and after the last layer, and the gain (mean +- std over runs)
  cumulative_acc.png  for the deepest models: accuracy after each layer, mean +- std over runs
  reset_share.png     for the deepest models: share of runs in which each layer was reset

Which logs are used: the job header names the configuration ("=== [i] <dataset>-<arch>-<method>-L<depth>"),
and only logs whose run_bench line reports --num-runs runs (default 5) count, which leaves out the 1-run
probes. If a run appears in several logs (a resubmission), the latest complete copy is used.
"""
import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np

HEADER = re.compile(r'=== \[\d+\] (?P<tag>\S+)\s+class=')
TAG = re.compile(r'^(?P<dataset>.+)-(?P<arch>GCN|SAGE|GAT)-(?P<method>.+)-L(?P<depth>\d+)$')
BENCH = re.compile(r'\[bench\] \S+ \| \S+ \| \d+ layer\(s\) \| (?P<runs>\d+) runs')
START = re.compile(r'Starting run-(\d+) of ')
STOP = re.compile(r'\[Layer (\d+): Epoch-(\d+)\] Early stop')
ACC = re.compile(r'\[Layer-(\d+)\] Test Accuracy : ([\d.]+)%')
FINAL = re.compile(r'\[bench\] run (\d+) (\w+): acc ([\d.]+)%')

STYLE = {'SF': ('#d62728', 's'), 'SF-uni': ('#ff7f0e', 'v'), 'SF-noVN': ('#8c564b', 'P'),
         'SF-Top2Input': ('#2ca02c', '^'), 'SF-Top2Loss': ('#9467bd', 'D')}
ARCH_ORDER = ['GCN', 'SAGE', 'GAT']


def parse_log(path, num_runs):
    with open(path, 'rb') as f:
        text = f.read().decode('utf-8', 'replace')
    h = HEADER.search(text)
    m = TAG.match(h.group('tag')) if h else None
    b = BENCH.search(text)
    if m is None or b is None or int(b.group('runs')) != num_runs:
        return None
    cfg = dict(dataset=m.group('dataset'), arch=m.group('arch'), method=m.group('method'),
               depth=int(m.group('depth')))
    starts = [(s.start(), int(s.group(1))) for s in START.finditer(text)]
    runs = []
    for k, (pos, run_i) in enumerate(starts):
        seg = text[pos: starts[k + 1][0] if k + 1 < len(starts) else len(text)]
        final = FINAL.search(seg)
        runs.append(dict(run=run_i,
                         stops={int(l): int(e) for l, e in STOP.findall(seg)},
                         accs={int(l): float(a) for l, a in ACC.findall(seg)},
                         final_status=final.group(2) if final else None,
                         final_acc=float(final.group(3)) if final else None))
    return cfg, runs


def mean_std(values):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float('nan'), float('nan')
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--logs', default='logs')
    ap.add_argument('--out', default='reports/layers')
    ap.add_argument('--methods', nargs='*', default=['SF', 'SF-uni', 'SF-noVN'])
    ap.add_argument('--archs', nargs='*', default=['GCN', 'SAGE'])
    ap.add_argument('--num-runs', type=int, default=5)
    ap.add_argument('--patience', type=int, default=100)
    ap.add_argument('--val-every', type=int, default=2)
    ap.add_argument('--reset-before', type=int, default=10,
                    help='a layer counts as reset if the epoch it kept is earlier than this')
    ap.add_argument('--depth', type=int, help='depth for the figures (default: deepest found)')
    a = ap.parse_args()
    window = a.patience * a.val_every

    best = {}  # (dataset, arch, method, depth, run) -> (complete, mtime, run dict, log)
    n_logs = n_used = 0
    for path in glob.glob(os.path.join(a.logs, '*.out')):
        n_logs += 1
        parsed = parse_log(path, a.num_runs)
        if parsed is None:
            continue
        cfg, runs = parsed
        if cfg['method'] not in a.methods or cfg['arch'] not in a.archs:
            continue
        n_used += 1
        mtime = os.path.getmtime(path)
        for r in runs:
            key = (cfg['dataset'], cfg['arch'], cfg['method'], cfg['depth'], r['run'])
            cand = (len(r['accs']) == cfg['depth'], mtime, r, os.path.basename(path))
            if key not in best or cand[:2] > best[key][:2]:
                best[key] = cand
    if not best:
        raise SystemExit(f'no single-forward runs with {a.num_runs} runs found in {a.logs}/')
    os.makedirs(a.out, exist_ok=True)

    rows, per_cfg = [], defaultdict(list)
    incomplete, mismatches, no_output = [], [], 0
    for (ds, arch, method, depth, run_i), (complete, _, r, log) in sorted(best.items()):
        if not r['accs']:  # stopped before its first layer finished, e.g. out of memory
            no_output += 1
            continue
        if not complete:
            incomplete.append(f'{ds}-{arch}-{method}-L{depth} run {run_i} ({len(r["accs"])}/{depth} layers, {log})')
            continue
        if r['final_acc'] is not None and abs(r['final_acc'] - r['accs'][depth - 1]) > 0.011:
            mismatches.append(f'{ds}-{arch}-{method}-L{depth} run {run_i}: last layer {r["accs"][depth - 1]:.2f}% '
                              f'vs recorded {r["final_acc"]:.2f}%')
        resets = []
        for layer in range(depth):
            stop = r['stops'].get(layer)
            kept = stop - window if stop is not None else None
            reset = kept is not None and kept < a.reset_before
            resets.append(reset)
            rows.append(dict(dataset=ds, arch=arch, method=method, depth=depth, run=run_i, layer=layer,
                             stop_epoch='' if stop is None else stop, best_epoch='' if kept is None else kept,
                             reset=int(reset), cum_acc=r['accs'][layer], log=log))
        per_cfg[(ds, arch, method, depth)].append(dict(
            later_reset=sum(resets[1:]), first=r['accs'][0], last=r['accs'][depth - 1],
            resets=resets, accs=[r['accs'][k] for k in range(depth)]))

    with open(os.path.join(a.out, 'layers.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = []
    for (ds, arch, method, depth), runs in sorted(per_cfg.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][3], kv[0][2])):
        lr = mean_std([x['later_reset'] for x in runs])
        fa, la = mean_std([x['first'] for x in runs]), mean_std([x['last'] for x in runs])
        ga = mean_std([x['last'] - x['first'] for x in runs])
        summary.append(dict(dataset=ds, arch=arch, method=method, depth=depth, runs=len(runs),
                            later_layers=depth - 1, later_reset_mean=lr[0], later_reset_std=lr[1],
                            acc_first_mean=fa[0], acc_first_std=fa[1], acc_last_mean=la[0], acc_last_std=la[1],
                            gain_mean=ga[0], gain_std=ga[1]))
    with open(os.path.join(a.out, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    def pm(m, s, d=2):
        return f'{m:.{d}f}' + ('' if s != s else f' ± {s:.{d}f}')

    md = ['| dataset | arch | method | depth | runs | later layers reset | acc after layer 1 (%) '
          '| acc after last layer (%) | gain (points) |', '|' + '---|' * 9]
    for s in summary:
        md.append(f"| {s['dataset'].replace('CitationFull-', '')} | {s['arch']} | {s['method']} | {s['depth']} "
                  f"| {s['runs']} | {pm(s['later_reset_mean'], s['later_reset_std'], 1)} of {s['later_layers']} "
                  f"| {pm(s['acc_first_mean'], s['acc_first_std'])} | {pm(s['acc_last_mean'], s['acc_last_std'])} "
                  f"| {pm(s['gain_mean'], s['gain_std'])} |")
    with open(os.path.join(a.out, 'summary.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')

    # ------------------------------------------------------------------ figures for the deepest models
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    depth = a.depth or max(k[3] for k in per_cfg)
    deep = {k: v for k, v in per_cfg.items() if k[3] == depth}
    datasets = sorted({k[0] for k in deep})
    archs = [x for x in ARCH_ORDER if any(k[1] == x for k in deep)]
    xs = np.arange(1, depth + 1)

    for kind, ylabel, fname in [('acc', 'Test accuracy after this layer (%)', 'cumulative_acc.png'),
                                ('reset', 'Share of runs where this layer was reset', 'reset_share.png')]:
        fig, axes = plt.subplots(len(datasets), len(archs), figsize=(5.2 * len(archs), 3.8 * len(datasets)),
                                 squeeze=False, sharex=True)
        for i, ds in enumerate(datasets):
            for j, arch in enumerate(archs):
                ax = axes[i][j]
                for method in a.methods:
                    runs = deep.get((ds, arch, method, depth))
                    if not runs:
                        continue
                    color, marker = STYLE.get(method, ('#333333', 'o'))
                    if kind == 'acc':
                        mat = np.array([x['accs'] for x in runs])
                        mu = mat.mean(0)
                        sd = mat.std(0, ddof=1) if len(runs) > 1 else np.zeros_like(mu)
                        ax.plot(xs, mu, marker=marker, ms=4, lw=1.6, color=color, label=f'{method} (n={len(runs)})')
                        ax.fill_between(xs, mu - sd, mu + sd, color=color, alpha=0.15, lw=0)
                    else:
                        share = np.array([x['resets'] for x in runs], dtype=float).mean(0)
                        ax.plot(xs, share, marker=marker, ms=4, lw=1.4, color=color, label=method)
                        ax.set_ylim(-0.05, 1.05)
                ax.set_title(f"{ds.replace('CitationFull-', '')} / {arch}", fontweight='bold')
                ax.grid(alpha=0.3)
                ax.set_xticks(xs[::max(1, depth // 8)])
                if j == 0:
                    ax.set_ylabel(ylabel)
                if i == len(datasets) - 1:
                    ax.set_xlabel(f'Layer (of a {depth}-layer model)')
                if i == 0 and j == 0:
                    ax.legend(fontsize=8)
        title = ('Combined prediction after each layer: mean over runs (band = ±1 std)' if kind == 'acc'
                 else f'Layers restored to their first {a.reset_before} epochs by early stopping')
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(os.path.join(a.out, fname), dpi=110)
        plt.close(fig)

    # ------------------------------------------------------------------ console report
    print(f'{n_logs} logs read, {n_used} used; {len(rows)} layer records from '
          f'{sum(len(v) for v in per_cfg.values())} complete runs -> {a.out}/')
    if no_output:
        print(f'{no_output} run(s) ended before their first layer finished (e.g. out of memory): left out')
    if incomplete:
        print(f'{len(incomplete)} incomplete run(s) left out (e.g. a job that hit its time limit mid-run):')
        for x in incomplete[:10]:
            print('   ', x)
    print(('cross-check: last-layer accuracy matches the recorded accuracy in every run' if not mismatches
           else f'cross-check: {len(mismatches)} run(s) where the last layer differs from the record:'))
    for x in mismatches[:10]:
        print('   ', x)
    print(f'\nDepth {depth}: later layers reset (of {depth - 1}), accuracy after layer 1 and after layer {depth}, '
          f'mean over runs')
    for s in summary:
        if s['depth'] == depth:
            print(f"  {s['dataset'].replace('CitationFull-', ''):<8} {s['arch']:<5} {s['method']:<8} "
                  f"n={s['runs']}  reset {s['later_reset_mean']:4.1f}  "
                  f"{s['acc_first_mean']:6.2f}% -> {s['acc_last_mean']:6.2f}%  "
                  f"gain {s['gain_mean']:+5.2f} ± {s['gain_std']:.2f}")


if __name__ == '__main__':
    main()
