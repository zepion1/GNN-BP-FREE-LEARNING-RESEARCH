"""Benchmark helpers for running facebookresearch/forwardgnn on WebKB (LINQS) and Reddit2.

Modules (all run inside the Python 3.10 / torch 1.13.1 / PyG 2.2.0 training environment):
    datasets_ext    download, clean and register the datasets with forwardgnn's loader
    sparse_backend  optional SparseTensor message passing (same math, no per-edge feature tensors)
    run_bench       launcher that runs train_backprop.py / train_forward.py and records metrics per run
"""