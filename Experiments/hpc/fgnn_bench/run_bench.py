"""Run forwardgnn's train_backprop.py / train_forward.py and record benchmark metrics per run.

    cd forwardgnn/src
    python -m fgnn_bench.run_bench --records-dir DIR [--mp-backend sparse|edge_index] \\
        [--feature-norm standardize|row_l1|none] -- \\
        train_forward.py --dataset WebKB-Cornell --model GNN_SingleForward-GCN --num-layers 2 ...

The training script runs unmodified and in-process; everything after `--` is its own command line.
A run is one (data split, seed) pair, exactly as in the script. For every run the launcher writes
DIR/<dataset>/<model>/L<num_layers>/run<i>.json containing

  acc, f1_macro, f1_micro, f1_weighted   in %, from the script's own final test-set predictions
  train_time_s                           the script's own training timer (what the paper reports)
  train_time_excl_test_s                 the same minus test-set evaluations made during training
                                         (the forward-learning scripts score the test set whenever
                                         validation improves; backprop does not) and minus the
                                         per-layer result files single-forward writes while training
  gpu_peak_alloc_train_mb                torch.cuda.max_memory_allocated() from the start of the run
                                         to the end of training; counters are reset for every run
  gpu_peak_alloc_mb, gpu_peak_reserved_mb  the same over the whole run, final evaluation included

and status "ok", "oom" or "error". Runs that already have an "ok" record are skipped, so running a
configuration again resumes it. Exit code: 0 ok, 3 out of memory, 1 any other failure.
"""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # before CUDA initialises, as the scripts do

import argparse
import gc
import importlib
import inspect
import json
import platform
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent  # forwardgnn/src
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
import torch_geometric

SCRIPTS = ('train_backprop.py', 'train_forward.py')
METHODS = (  # model-name prefix -> method label; longest prefixes first
    ('GNN_SingleForwardNoVN-', 'SF-noVN'),  # ablation: no virtual nodes (fgnn_bench/sf_novn.py)
    ('GNN_Top2Input_SingleForward-', 'SF-Top2Input'),
    ('GNN_Top2Loss_SingleForward-', 'SF-Top2Loss'),
    ('GNN_SingleForward-', 'SF'),
    ('GNN-', 'BP'),
)
HPARAMS = ('lr', 'epochs', 'patience', 'val_every', 'num_hidden', 'num_runs', 'seed', 'temperature',
           'append_label', 'aug_edge_direction', 'alternating_update', 'test_time_steps',
           'storable_time_steps', 'grad_max_norm', 'val_from')
MB = 2 ** 20


def method_of(model):
    for prefix, method in METHODS:
        if model.startswith(prefix):
            return method
    raise ValueError(f'unsupported model {model!r} (expected one of the BP/SF/Top2Input/Top2Loss names)')


def record_path(records_dir, dataset, model, num_layers, run_i):
    return Path(records_dir) / dataset / model / f'L{num_layers}' / f'run{run_i}.json'


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


class Recorder:
    """Collects one run's measurements from the hooks and turns them into a record."""

    def __init__(self, records_dir, mp_backend, feature_norm):
        self.records_dir = Path(records_dir)
        self.mp_backend = mp_backend
        self.feature_norm = feature_norm
        self.args = None
        self.run_i = None
        self._new_run()

    def bind(self, args):
        self.args = args
        self.method = method_of(args.model)
        if self.method == 'SF' and getattr(args, 'aug_edge_direction', None) == 'unidirection':
            self.method = 'SF-uni'  # ablation: virtual nodes do not send messages back to real nodes
        self.arch = args.model.rsplit('-', 1)[-1]
        self.cuda = args.device.type == 'cuda'

    def _new_run(self):
        self.t_start = time.perf_counter()
        self.in_training = False
        self.in_final_test = False
        self.train_data = None
        self.test_eval_s = 0.0
        self.n_test_evals = 0
        self.layer_save_s = 0.0
        self.train_peak = None
        self.mem_at_start = None
        self.y_true = self.y_pred = None
        self.num_classes = None
        self.run_recorded = False

    # ------------------------------------------------------------------------------ records on disk

    def path(self, run_i, num_layers=None):
        num_layers = self.args.num_layers if num_layers is None else num_layers
        return record_path(self.records_dir, self.args.dataset, self.args.model, num_layers, run_i)

    def load(self, run_i, num_layers=None):
        try:
            return json.loads(self.path(run_i, num_layers).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def write(self, record):
        if record['run_i'] is None:  # failed before the first run started
            write_json(self.path(0).with_name('setup-error.json'), record)
        else:
            write_json(self.path(record['run_i']), record)
            self.run_recorded = True

    # ---------------------------------------------------------------------------- hook entry points

    def start_run(self, run_i):
        self._new_run()
        self.run_i = run_i
        if self.cuda:
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.mem_at_start = torch.cuda.memory_allocated()
        self.t_start = time.perf_counter()

    def end_training(self):
        if self.cuda and self.train_peak is None:
            torch.cuda.synchronize()
            self.train_peak = (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())

    def capture(self, y_true, y_pred, num_classes):
        self.y_true = y_true.detach().cpu().numpy().astype(np.int64)
        self.y_pred = y_pred.detach().cpu().numpy().astype(np.int64)
        self.num_classes = int(num_classes)

    # ------------------------------------------------------------------------------------- records

    def _identity(self, run_i):
        from utils.train_utils import SeedManager

        a = self.args
        return {
            'dataset': a.dataset, 'model': a.model, 'arch': self.arch, 'method': self.method,
            'num_layers': a.num_layers, 'run_i': run_i, 'split_i': run_i,
            'seed': SeedManager(a.seed).get_run_seed(run_i) if run_i is not None else None,
            'exp_setting': a.exp_setting, 'mp_backend': self.mp_backend, 'feature_norm': self.feature_norm,
            'hparams': {k: getattr(a, k) for k in HPARAMS if hasattr(a, k)},
            'device': str(a.device), 'python': platform.python_version(), 'torch': torch.__version__,
            'torch_geometric': torch_geometric.__version__,
            'finished_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        }

    def _memory(self):
        if not self.cuda:
            return {'gpu_peak_alloc_train_mb': None, 'gpu_peak_reserved_train_mb': None,
                    'gpu_peak_alloc_mb': None, 'gpu_peak_reserved_mb': None,
                    'gpu_alloc_at_run_start_mb': None, 'gpu_name': None, 'gpu_total_mb': None}
        torch.cuda.synchronize()
        alloc, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        train_alloc, train_reserved = self.train_peak or (alloc, reserved)
        props = torch.cuda.get_device_properties(self.args.device)
        return {
            'gpu_peak_alloc_train_mb': train_alloc / MB, 'gpu_peak_reserved_train_mb': train_reserved / MB,
            'gpu_peak_alloc_mb': alloc / MB, 'gpu_peak_reserved_mb': reserved / MB,
            'gpu_alloc_at_run_start_mb': (self.mem_at_start or 0) / MB,
            'gpu_name': props.name, 'gpu_total_mb': props.total_memory / MB,
        }

    def ok_record(self, run_i, perf_dict):
        from sklearn.metrics import f1_score, precision_recall_fscore_support

        if self.y_true is None:
            raise RuntimeError('the final test predictions were not captured, so F1 cannot be computed')
        y_true, y_pred, k = self.y_true, self.y_pred, self.num_classes
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(k)), zero_division=0)
        acc = 100.0 * float(np.mean(y_true == y_pred))
        perf = float(perf_dict['perf'])
        acc_script = 100.0 * perf if self.method == 'BP' else perf  # BP reports a fraction, the others %
        train_time = float(perf_dict['train_time'])
        train_epochs = [int(e) for e in perf_dict['train_epochs']]
        record = {
            'status': 'ok',
            **self._identity(run_i),
            'acc': acc,
            'f1_macro': 100.0 * f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_micro': 100.0 * f1_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_weighted': 100.0 * f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'perf_reported_by_script': perf,
            'n_test': int(y_true.size),
            'per_class': {
                'f1': (100.0 * f1).tolist(), 'precision': (100.0 * precision).tolist(),
                'recall': (100.0 * recall).tolist(), 'test_support': support.tolist(),
                'predicted': np.bincount(y_pred, minlength=k).tolist(),
            },
            'train_time_s': train_time,
            'test_eval_during_training_s': self.test_eval_s,
            'n_test_evals_during_training': self.n_test_evals,
            'layer_result_files_s': self.layer_save_s,
            'train_time_excl_test_s': train_time - self.test_eval_s - self.layer_save_s,
            'run_wall_time_s': time.perf_counter() - self.t_start,
            'train_epochs': train_epochs,
            'best_val_epochs': [int(e) for e in perf_dict['best_val_epochs']],
            'epochs_total': sum(e + 1 for e in train_epochs),
            **self._memory(),
        }
        if abs(acc - acc_script) > 1e-3:
            record['status'] = 'error'
            record['error'] = f'recomputed accuracy {acc:.4f}% != script accuracy {acc_script:.4f}%'
        return record

    def failure_record(self, exc):
        message = f'{type(exc).__name__}: {exc}'
        oom_type = getattr(torch.cuda, 'OutOfMemoryError', None)
        oom = (oom_type is not None and isinstance(exc, oom_type)) or any(
            s in message.lower() for s in ('out of memory', 'not enough memory', "can't allocate memory"))
        record = {
            'status': 'oom' if oom else 'error',
            **self._identity(self.run_i),
            'error': message[:2000],
            'traceback': traceback.format_exc()[-6000:],
            'failed_during_training': self.in_training,
            'run_wall_time_s': time.perf_counter() - self.t_start,
        }
        if self.args is not None and self.cuda:
            record.update(self._memory())  # peak reached before the failure
        return record


# ------------------------------------------------------------------------------------------- hooks

def install_hooks(rec, train_module, script):
    from utils.train_utils import ResultManager, SeedManager

    set_run_i = SeedManager.set_run_i

    def set_run_i_hook(self, run_i):  # both scripts call this first thing in every run
        set_run_i(self, run_i)
        rec.start_run(run_i)

    SeedManager.set_run_i = set_run_i_hook

    def load_run_result(self, run_i, num_layers=None):  # the scripts skip runs for which this is not None
        record = rec.load(run_i, num_layers)  # ok, or finished with an accuracy cross-check error
        if record is not None and 'perf_reported_by_script' in record:
            return {'perf': record['perf_reported_by_script']}
        return None

    ResultManager.load_run_result = load_run_result

    save_run_result = ResultManager.save_run_result

    def save_run_result_hook(self, run_i, perf_dict, num_layers=None):
        if num_layers is not None:  # per-layer file written by single-forward *during* training
            start = time.perf_counter()
            try:
                return save_run_result(self, run_i, perf_dict, num_layers)
            finally:
                rec.layer_save_s += time.perf_counter() - start
        record = rec.ok_record(run_i, perf_dict)  # the final save of the run
        result = save_run_result(self, run_i, perf_dict, num_layers)
        rec.write(record)
        mem = record['gpu_peak_alloc_train_mb']
        print(f"[bench] run {run_i} {record['status']}: acc {record['acc']:.2f}% | "
              f"macro-F1 {record['f1_macro']:.2f}% | train {record['train_time_s']:.1f}s "
              f"(excl. test evals {record['train_time_excl_test_s']:.1f}s) | "
              f"peak GPU {'n/a' if mem is None else f'{mem:,.0f} MB'}"
              + (f" | {record['error']}" if record['status'] != 'ok' else ''), flush=True)
        return result

    ResultManager.save_run_result = save_run_result_hook

    if script == 'train_backprop.py':
        _hook_backprop(rec, train_module)
    else:
        from forward_learning.nodeclass import GNNSingleForward, GNNTop2InputSF, GNNTop2LossSF

        for cls in (GNNSingleForward, GNNTop2InputSF, GNNTop2LossSF):
            _hook_forward_model(rec, cls)


def _hook_backprop(rec, train_module):
    trainer = train_module.NodeClassificationTrainer
    train, test = trainer.train, trainer.test

    def train_hook(self):
        rec.in_training = True
        try:
            return train(self)
        finally:
            rec.in_training = False
            rec.end_training()

    def test_hook(self):
        rec.in_final_test, rec.num_classes = True, int(self.data.num_classes)
        try:
            return test(self)
        finally:
            rec.in_final_test = False

    trainer.train, trainer.test = train_hook, test_hook

    evaluate = train_module.eval_node_classification  # validation (in train) and test (in test)

    def eval_hook(y_true, y_pred, mask):
        result = evaluate(y_true, y_pred, mask)
        if rec.in_final_test:
            rec.capture(y_true[mask], y_pred[mask], rec.num_classes)
        return result

    train_module.eval_node_classification = eval_hook


def _hook_forward_model(rec, cls):
    forward_train, eval_model = cls.forward_train, cls.eval_model
    signature = inspect.signature(eval_model)

    def forward_train_hook(self, data, *args, **kwargs):
        rec.train_data, rec.in_training = data, True  # forward_train moves `data` to the device in place
        try:
            return forward_train(self, data, *args, **kwargs)
        finally:
            rec.in_training = False
            rec.end_training()

    def eval_model_hook(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        eval_mask = bound.arguments['eval_mask']
        start = time.perf_counter()
        acc, probs = eval_model(self, *args, **kwargs)
        if rec.in_training and rec.train_data is not None and eval_mask is rec.train_data.test_mask:
            rec.test_eval_s += time.perf_counter() - start
            rec.n_test_evals += 1
        if bound.arguments.get('print_eval', False):  # only the script's final test evaluation
            pred = sum(probs).argmax(dim=1)  # the prediction eval_model itself scored
            rec.capture(self.graph_augmenter.y[eval_mask], pred, self.num_classes)
        return acc, probs

    cls.forward_train, cls.eval_model = forward_train_hook, eval_model_hook


# -------------------------------------------------------------------------------------------- main

def _exit_on_interrupt(signum, frame):
    # forwardgnn's forward trainers treat KeyboardInterrupt as "stop this layer early" and would then
    # save a truncated run; exit instead so the run stays unrecorded and is redone next time.
    print('\n[bench] interrupted: exiting without recording the current run', flush=True)
    os._exit(130)


def main():
    argv = sys.argv[1:]
    if '--' not in argv or argv.index('--') == len(argv) - 1:
        raise SystemExit(__doc__)
    cut = argv.index('--')
    parser = argparse.ArgumentParser(prog='run_bench')
    parser.add_argument('--records-dir', required=True, type=Path)
    parser.add_argument('--mp-backend', choices=('sparse', 'edge_index'), default='sparse')
    parser.add_argument('--feature-norm', choices=('standardize', 'row_l1', 'none'), default='standardize')
    bench = parser.parse_args(argv[:cut])
    script, script_args = argv[cut + 1], argv[cut + 2:]
    if script not in SCRIPTS:
        raise SystemExit(f'script must be one of {SCRIPTS}, got {script!r}')

    signal.signal(signal.SIGINT, _exit_on_interrupt)
    os.environ['FGNN_FEATURE_NORM'] = bench.feature_norm  # read by datasets_ext.load()

    from fgnn_bench import datasets_ext

    datasets_ext.install()
    if bench.mp_backend == 'sparse':
        from fgnn_bench import sparse_backend

        sparse_backend.install()

    sys.argv = [str(SRC_DIR / script)] + script_args
    train_module = importlib.import_module(script[:-len('.py')])
    rec = Recorder(bench.records_dir, bench.mp_backend, bench.feature_norm)
    install_hooks(rec, train_module, script)
    if script == 'train_forward.py':
        from fgnn_bench import sf_novn

        sf_novn.install(train_module)  # routes GNN_SingleForwardNoVN-* to the no-virtual-node ablation

    args = train_module.populate_args(train_module.parse_args())
    if args.task != 'node-class':
        raise SystemExit('run_bench only supports --task node-class')
    if args.dataset not in datasets_ext.DATASETS and bench.feature_norm != 'row_l1':
        print(f'[bench] note: {args.dataset} is a built-in forwardgnn dataset; it always uses '
              f'NormalizeFeatures, whatever --feature-norm says', flush=True)
    rec.bind(args)
    print(f'[bench] {args.dataset} | {args.model} | {args.num_layers} layer(s) | {args.num_runs} runs | '
          f'device {args.device} | message passing: {bench.mp_backend} | features: {bench.feature_norm}',
          flush=True)
    try:
        train_module.main(args)
    except Exception as exc:  # CUDA out-of-memory included
        traceback.print_exc()
        if rec.run_recorded:  # failed after the current run was already saved; keep that record
            print(f'[bench] failed after run {rec.run_i} was recorded: {type(exc).__name__}: {exc}', flush=True)
            sys.exit(1)
        record = rec.failure_record(exc)
        rec.write(record)
        print(f"[bench] run {record['run_i']} {record['status'].upper()}: {record['error'][:300]}", flush=True)
        sys.exit(3 if record['status'] == 'oom' else 1)


if __name__ == '__main__':
    main()