#!/usr/bin/env python3
"""Find what keeps GPU memory alive between layers of a forward-learning run (diagnostic, not part of the sweeps).

    cd $FGNN_SRC && $FGNN_PY $FGNN_ROOT/diag_memory.py --records-dir $FGNN_ROOT/smoke_records/diag \
        --mp-backend sparse --feature-norm row_l1 -- train_forward.py --exp-setting smoke-diag \
        --dataset CitationFull-PubMed --task node-class --model GNN_SingleForwardNoVN-GCN --num-layers 6 \
        --num-runs 1 --seed 100 --epochs 30 --val-every 2 --lr 0.001 --patience 5 --num-hidden 128 \
        --append-label none --aug-edge-direction bidirection

Runs run_bench as usual, and after every layer prints the GPU memory in use and every large CUDA tensor still
alive (at least DIAG_MIN_ROWS rows, default 10000), with what references each new one. Written to explain why
SF-noVN's peak memory grows with depth while SF's does not; not yet run at the time of writing.
"""
import collections, gc, os, sys
src = os.environ['FGNN_SRC']; os.chdir(src); sys.path.insert(0, src)
import torch
from utils.train_utils import ResultManager

MIN_ROWS = int(os.environ.get('DIAG_MIN_ROWS', '10000'))
prev = collections.Counter()
orig = ResultManager.save_run_result


def census():
    found = collections.defaultdict(list)
    for o in gc.get_objects():
        try:
            if isinstance(o, torch.Tensor) and o.is_cuda and o.dim() >= 1 and o.size(0) >= MIN_ROWS:
                found[(tuple(o.shape), str(o.dtype)[6:], 'grad' if o.requires_grad else '')].append(o)
        except Exception:
            pass
    return found


def holders_of(t, skip):
    out = set()
    for r in gc.get_referrers(t):
        if any(r is s for s in skip):
            continue
        if hasattr(r, 'f_code'):
            if r.f_code.co_name not in ('census', 'holders_of', 'save_run_result', '<listcomp>'):
                out.add('frame ' + r.f_code.co_name)
        elif isinstance(r, dict):
            owners = [type(x).__name__ for x in gc.get_referrers(r) if getattr(x, '__dict__', None) is r]
            out.add('attribute of ' + (owners[0] if owners else 'a dict'))
        else:
            out.add(type(r).__name__)
    return ', '.join(sorted(out)) or 'no Python holder'


def save_run_result(self, run_i, perf_dict, num_layers=None):
    global prev
    if num_layers is not None and torch.cuda.is_available():
        before = torch.cuda.memory_allocated() / 2 ** 20
        n = gc.collect()
        after = torch.cuda.memory_allocated() / 2 ** 20
        found = census()
        print(f'[diag] layer {num_layers} done: {before:,.0f} MB in use; {after:,.0f} MB after garbage collection ({n} objects)', flush=True)
        for key, ts in sorted(found.items(), key=lambda kv: (-len(kv[1]), str(kv[0]))):
            delta = len(ts) - prev.get(key, 0)
            print(f'[diag]   {len(ts):>3} x {key[0]} {key[1]} {key[2]}' + (f'  ({delta:+d})' if delta and num_layers > 1 else ''), flush=True)
            if delta > 0 and num_layers > 1:
                for t in ts[:4]:
                    print(f'[diag]        held by: {holders_of(t, (ts, found))}', flush=True)
        prev = collections.Counter({k: len(v) for k, v in found.items()})
    return orig(self, run_i, perf_dict, num_layers)


ResultManager.save_run_result = save_run_result
from fgnn_bench import run_bench  # noqa: E402
run_bench.main()
