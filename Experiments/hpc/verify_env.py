#!/usr/bin/env python3
"""Verify the forwardgnn environment. Run this on a COMPUTE node (srun/sbatch),
not the login node: the GPU checks are the point.

    source $FGNN_ROOT/activate.sh && $FGNN_PY $FGNN_ROOT/verify_env.py
"""
import os
import sys

ok = True


def check(label, fn):
    global ok
    try:
        print(f"  {label:<34} {fn()}")
    except Exception as e:
        ok = False
        print(f"  {label:<34} FAILED: {type(e).__name__}: {e}")


print("= versions")
import numpy, scipy, sklearn, torch, torch_geometric
check("python", lambda: sys.version.split()[0])
check("torch", lambda: torch.__version__)
check("torch_geometric", lambda: torch_geometric.__version__)
check("numpy / scipy / sklearn",
      lambda: f"{numpy.__version__} / {scipy.__version__} / {sklearn.__version__}")

print("= gpu")
check("cuda available", lambda: torch.cuda.is_available())
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    check("device", lambda: torch.cuda.get_device_name(0))
    check("compute capability", lambda: f"sm_{cap[0]}{cap[1]}")
    check("total memory",
          lambda: f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    check("arch list in this build", lambda: ' '.join(torch.cuda.get_arch_list()))
    supported = f"sm_{cap[0]}{cap[1]}" in torch.cuda.get_arch_list()
    if not supported:
        ok = False
        print("  *** this GPU is NOT in the build's arch list -- kernels will fail or")
        print("  *** fall back to slow PTX JIT. Rebuild with a newer torch/CUDA.")
else:
    print("  *** no GPU. Run this under srun with a GPU allocation.")

print("= forwardgnn imports (vendored PyG 2.2 internals)")
src = os.environ.get("FGNN_SRC")
if src:
    sys.path.insert(0, src)
check("gnn.GNNConv", lambda: __import__("gnn", fromlist=["GNNConv"]).GNNConv.__name__)
check("torch_sparse.SparseTensor",
      lambda: __import__("torch_sparse", fromlist=["SparseTensor"]).SparseTensor.__name__)
check("fgnn_bench", lambda: __import__("fgnn_bench").__name__)

print("= compiled kernels")
if torch.cuda.is_available():
    import torch_scatter
    from torch_sparse import SparseTensor
    x = torch.randn(5, 3, device="cuda")
    idx = torch.tensor([0, 0, 1, 2, 2], device="cuda")
    check("torch_scatter",
          lambda: tuple(torch_scatter.scatter(x, idx, dim=0, reduce="mean").shape))
    adj = SparseTensor(row=idx, col=torch.arange(5, device="cuda"), sparse_sizes=(3, 5))
    check("torch_sparse spmm", lambda: tuple((adj @ x).shape))

print("\n" + ("ENVIRONMENT OK" if ok else "ENVIRONMENT HAS PROBLEMS -- see FAILED lines"))
sys.exit(0 if ok else 1)
