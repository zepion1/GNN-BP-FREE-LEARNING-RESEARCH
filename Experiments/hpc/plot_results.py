#!/usr/bin/env python3
"""Summarise and plot benchmark records: accuracy, macro-F1, peak GPU memory and training time vs depth.

    $FGNN_PY plot_results.py --records bench_records/pubmed-sparse-row_l1 --out reports/pubmed

Several folders can be combined, e.g. the virtual-node ablation next to the main runs:
    $FGNN_PY plot_results.py --records bench_records/pubmed-sparse-row_l1 bench_records/pubmed-sparse-row_l1-ablation \
        --methods BP SF SF-uni SF-noVN --archs GCN SAGE --out reports/ablation-pubmed

Writes to --out:
    overview_<dataset>.png          all four metrics x three architectures in one image
    <metric>_<dataset>.png          one image per metric (for slides)
    summary.csv / summary.md        mean, std and run counts per configuration
Needs only numpy and matplotlib; no GPU.
"""
import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')                      # no display on the cluster
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

METHODS = ['BP', 'SF', 'SF-uni', 'SF-noVN', 'SF-Top2Input', 'SF-Top2Loss']
ARCHS = ['GCN', 'SAGE', 'GAT']
STYLE = {'BP': ('#1f77b4', 'o'), 'SF': ('#d62728', 's'),
         'SF-uni': ('#ff7f0e', 'v'), 'SF-noVN': ('#8c564b', 'P'),
         'SF-Top2Input': ('#2ca02c', '^'), 'SF-Top2Loss': ('#9467bd', 'D')}
METRICS = [  # key, label, log-scale y
    ('acc', 'Test accuracy (%)', False),
    ('f1_macro', 'Macro-F1 (%)', False),
    ('gpu_peak_alloc_train_mb', 'Peak GPU memory in training (MB)', True),
    ('train_time_excl_test_s', 'Training time per run (s)', True),
]
TABLE_KEYS = ['acc', 'f1_macro', 'f1_micro', 'f1_weighted',
              'gpu_peak_alloc_train_mb', 'train_time_excl_test_s']


def load(records_dirs):
    runs, seen = [], {}
    for records_dir in records_dirs:
        for p in glob.glob(os.path.join(records_dir, '**', 'run*.json'), recursive=True):
            try:
                with open(p) as f:
                    r = json.load(f)
            except (OSError, ValueError):
                print(f'  skipped unreadable {p}')
                continue
            key = (r.get('dataset'), r.get('model'), r.get('method'), r.get('num_layers'), r.get('run_i'))
            if key in seen:  # same run in two folders: keep the first folder's copy
                print(f'  duplicate run ignored: {p} (already read from {seen[key]})')
                continue
            seen[key] = p
            runs.append(r)
    return runs


def summarise(runs):
    groups = defaultdict(list)
    for r in runs:
        groups[(r['dataset'], r['arch'], r['method'], int(r['num_layers']))].append(r)
    rows = []
    for (ds, arch, method, depth), rs in sorted(groups.items()):
        ok = [r for r in rs if r.get('status') == 'ok']
        row = dict(dataset=ds, arch=arch, method=method, depth=depth,
                   n_ok=len(ok), n_oom=sum(r.get('status') == 'oom' for r in rs),
                   n_error=sum(r.get('status') == 'error' for r in rs),
                   devices=','.join(sorted({str(r.get('device')) for r in ok})))
        for k in TABLE_KEYS:
            vals = [r[k] for r in ok if isinstance(r.get(k), (int, float))]
            row[f'{k}_mean'] = float(np.mean(vals)) if vals else math.nan
            row[f'{k}_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else math.nan
        rows.append(row)
    return rows


def plot_metric(ax, rows, ds, arch, key, logy, depths):
    for m in METHODS:
        pts = sorted((r['depth'], r[f'{key}_mean'], r[f'{key}_std']) for r in rows
                     if r['dataset'] == ds and r['arch'] == arch and r['method'] == m
                     and not math.isnan(r[f'{key}_mean']))
        if not pts:
            continue
        x, y, s = (np.array(v, dtype=float) for v in zip(*pts))
        s = np.nan_to_num(s)
        color, marker = STYLE[m]
        ax.plot(x, y, marker=marker, color=color, label=m, lw=1.6, ms=5)
        if not logy:
            ax.fill_between(x, y - s, y + s, color=color, alpha=0.15, lw=0)
        oom = [r['depth'] for r in rows if r['dataset'] == ds and r['arch'] == arch
               and r['method'] == m and r['n_ok'] == 0 and r['n_oom'] > 0]
        for d in oom:
            ax.annotate('OOM', (d, 0), xycoords=('data', 'axes fraction'), xytext=(0, 3),
                        textcoords='offset points', ha='center', fontsize=7, color=color)
    ax.set_xscale('log', base=2)
    ax.set_xticks(depths)
    ax.set_xticklabels([str(d) for d in depths])
    ax.minorticks_off()
    if logy:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}' if v >= 1 else f'{v:g}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=0.3, which='major')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', required=True, nargs='+',
                    help='one or more exp-setting folders, e.g. bench_records/pubmed-sparse-row_l1')
    ap.add_argument('--methods', nargs='*', help=f'only these methods (default: all), from {METHODS}')
    ap.add_argument('--archs', nargs='*', help='only these architectures (default: all)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    runs = load(a.records)
    if a.methods:
        runs = [r for r in runs if r.get('method') in a.methods]
    if a.archs:
        runs = [r for r in runs if r.get('arch') in a.archs]
    if not runs:
        raise SystemExit(f'no matching run*.json under {a.records}')
    rows = summarise(runs)
    os.makedirs(a.out, exist_ok=True)

    with open(os.path.join(a.out, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def fmt(r, k, d=2):
        m, s = r[f'{k}_mean'], r[f'{k}_std']
        if math.isnan(m):
            return '-'
        return f'{m:.{d}f}' + (f' ± {s:.{d}f}' if not math.isnan(s) else '')

    lines = ['| dataset | arch | method | depth | runs ok | accuracy (%) | macro-F1 (%) | micro-F1 (%) '
             '| peak GPU (MB) | train time (s) |', '|' + '---|' * 10]
    for r in rows:
        runs_txt = f"{r['n_ok']}" + (f" (+{r['n_oom']} OOM)" if r['n_oom'] else '') + \
                   (f" (+{r['n_error']} err)" if r['n_error'] else '')
        lines.append(f"| {r['dataset']} | {r['arch']} | {r['method']} | {r['depth']} | {runs_txt} | "
                     f"{fmt(r, 'acc')} | {fmt(r, 'f1_macro')} | {fmt(r, 'f1_micro')} | "
                     f"{fmt(r, 'gpu_peak_alloc_train_mb', 0)} | {fmt(r, 'train_time_excl_test_s', 1)} |")
    with open(os.path.join(a.out, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    datasets = sorted({r['dataset'] for r in rows})
    for ds in datasets:
        depths = sorted({r['depth'] for r in rows if r['dataset'] == ds})
        archs = [x for x in ARCHS if any(r['arch'] == x and r['dataset'] == ds for r in rows)]
        tag = ds.replace('CitationFull-', '')

        fig, axes = plt.subplots(len(METRICS), len(archs), figsize=(5 * len(archs), 3.6 * len(METRICS)),
                                 squeeze=False, sharex=True)
        for i, (key, label, logy) in enumerate(METRICS):
            for j, arch in enumerate(archs):
                plot_metric(axes[i][j], rows, ds, arch, key, logy, depths)
                if i == 0:
                    axes[i][j].set_title(arch, fontsize=13, fontweight='bold')
                if j == 0:
                    axes[i][j].set_ylabel(label)
                if i == len(METRICS) - 1:
                    axes[i][j].set_xlabel('Number of layers')
            if not logy:  # same y-range across architectures for accuracy/F1
                lo = min(ax.get_ylim()[0] for ax in axes[i]); hi = max(ax.get_ylim()[1] for ax in axes[i])
                for ax in axes[i]:
                    ax.set_ylim(lo, hi)
        h, l = axes[0][0].get_legend_handles_labels()
        fig.suptitle(f'{tag}: mean over runs (shaded band = ±1 std)', y=0.995, fontsize=13)
        fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.978), ncol=len(l),
                   frameon=False, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.955))
        fig.savefig(os.path.join(a.out, f'overview_{tag}.png'), dpi=110)
        plt.close(fig)

        for key, label, logy in METRICS:
            fig, axes = plt.subplots(1, len(archs), figsize=(5 * len(archs), 4), squeeze=False, sharey=not logy)
            for j, arch in enumerate(archs):
                plot_metric(axes[0][j], rows, ds, arch, key, logy, depths)
                axes[0][j].set_title(arch, fontweight='bold'); axes[0][j].set_xlabel('Number of layers')
            axes[0][0].set_ylabel(label)
            axes[0][0].legend(fontsize=8)
            fig.suptitle(f'{tag}: {label}')
            fig.tight_layout()
            fig.savefig(os.path.join(a.out, f'{key}_{tag}.png'), dpi=110)
            plt.close(fig)

    ok = [r for r in runs if r.get('status') == 'ok']
    gap = max((abs(r['acc'] - r['f1_micro']) for r in ok
               if isinstance(r.get('acc'), (int, float)) and isinstance(r.get('f1_micro'), (int, float))),
              default=0.0)
    devs = sorted({str(r.get('device')) for r in ok})
    print(f'{len(runs)} records, {len(ok)} ok, {len(rows)} configurations -> {a.out}/')
    print(f'devices: {devs}' + ('   *** some runs were NOT on a GPU' if any(not d.startswith('cuda') for d in devs) else ''))
    if gap < 0.01:
        print('note: micro-F1 equals accuracy in every run (true for single-label classification), '
              'so it is in the tables but not plotted.')
    for fn in sorted(os.listdir(a.out)):
        print('  ', fn)


if __name__ == '__main__':
    main()
