#!/usr/bin/env python3
"""Split a job list into submission groups using the 6-epoch probe results.

    $FGNN_PY plan_from_probe.py --jobs jobs_reddit_deep.tsv --probe-root bench_records/probe-reddit --prefix jobs_reddit

For every configuration in --jobs it reads <probe-root>/<dataset>/<model>/L<depth>/run0.json and estimates the
worst case: every layer trains the full 1000 epochs, for all 5 runs. Real runs usually stop far earlier
(patience 100), so these are time limits to request, not expected costs.

Writes <prefix>_<group>.tsv for each non-empty group and prints the exact sbatch command for it:
  oom     probe ran out of GPU memory  -> short job that records the OOM properly in the real results
  day     worst case fits in a day     -> gpu-day-long
  week    worst case fits in a week    -> gpu-week-long
  month   a single run may exceed a week -> printed, but decide before submitting
  noprobe no probe record (timed out at 2 h, or failed) -> not submitted; look at its log
"""
import argparse
import json
import os
import shlex

EPOCHS_FULL, EPOCHS_PROBE, RUNS = 1000, 6, 5
GROUPS = {  # name: (partition, --time, --mem, worst-case hours for all runs it may hold)
    'oom':   ('gpu-short',      '02:00:00',    '64G'),
    'day':   ('gpu-day-long',   '1-00:00:00',  '96G'),
    'week':  ('gpu-week-long',  '7-00:00:00',  '96G'),
    'month': ('gpu-month-long', '31-00:00:00', '96G'),
}
DAY_H, WEEK_H = 22.0, 160.0          # leave headroom under the 24 h / 168 h partition limits


def arg_of(cmd, flag):
    parts = shlex.split(cmd)
    return parts[parts.index(flag) + 1] if flag in parts else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--probe-root', required=True, help='e.g. bench_records/probe-reddit')
    ap.add_argument('--prefix', required=True, help='output files are <prefix>_<group>.tsv')
    ap.add_argument('--throttle', type=int, default=4, help='max jobs at once per array')
    a = ap.parse_args()

    with open(a.jobs) as f:
        header, *lines = [l.rstrip('\n') for l in f if l.strip()]
    groups = {g: [] for g in list(GROUPS) + ['noprobe']}
    table = []
    for line in lines:
        idx, tier, ds, arch, method, depth, est, cmd = line.split('\t')
        model, depth = arg_of(cmd, '--model'), int(depth)
        p = os.path.join(a.probe_root, ds, model, f'L{depth}', 'run0.json')
        rec = json.load(open(p)) if os.path.exists(p) else None
        st = rec.get('status') if rec else 'missing'
        t, e = (rec or {}).get('train_time_s'), (rec or {}).get('epochs_total')
        mb = (rec or {}).get('gpu_peak_alloc_train_mb')
        run_h = None
        if st == 'ok' and t:
            est1 = t * EPOCHS_FULL / EPOCHS_PROBE                       # scale the whole probe
            layers = depth if method == 'SF' else 1   # top-down methods: 1 time step updates every layer
            est2 = (t / e) * EPOCHS_FULL * layers if e else 0          # per-epoch x max epochs
            run_h = max(est1, est2) / 3600
        if st == 'oom':
            g = 'oom'
        elif st == 'ok' and run_h is not None:
            if run_h * RUNS <= DAY_H:
                g = 'day'
            elif run_h <= WEEK_H:      # every single run fits in a week (all 5 may need a 2nd round)
                g = 'week'
            else:
                g = 'month'
        else:
            g = 'noprobe'
        groups[g].append(line)
        note = ''
        if mb and mb > 35 * 1024:
            note = 'peak > 35 GB: may OOM'
        if g == 'week' and run_h * RUNS > WEEK_H:
            note = (note + '; ' if note else '') + 'may need a 2nd round'
        table.append((ds, model, depth, st, mb, run_h, g, note))

    print(f"{'model':<34}{'L':>3}  {'probe':<8}{'peak GB':>8}{'worst h/run':>12}  {'group':<8}note")
    for ds, model, depth, st, mb, run_h, g, note in table:
        gb = f'{mb / 1024:8.1f}' if mb else '       -'
        rh = f'{run_h:12.1f}' if run_h is not None else '           -'
        print(f'{model:<34}{depth:>3}  {st:<8}{gb}{rh}  {g:<8}{note}')

    print('\nSubmission commands (run from $FGNN_ROOT):')
    for g, rows in groups.items():
        if not rows:
            continue
        out = f'{a.prefix}_{g}.tsv'
        with open(out, 'w') as f:
            f.write(header + '\n' + '\n'.join(rows) + '\n')
        if g == 'noprobe':
            print(f'\n# {len(rows)} config(s) without a probe result -> {out}. NOT submitted: check their probe logs.')
            continue
        part, wall, mem = GROUPS[g]
        warn = '   # a single run may take over a week: decide before submitting' if g == 'month' else ''
        print(f'\n# {g}: {len(rows)} config(s){warn}')
        print(f'FGNN_JOBS=$PWD/{out} sbatch --partition={part} --array=1-{len(rows)}%{a.throttle} '
              f'--time={wall} --mem={mem} run_array.sbatch')


if __name__ == '__main__':
    main()
