#!/usr/bin/env bash
# Bootstrap the forwardgnn training environment on an HPC cluster with no root
# and no modern system Python.
#
#   bash setup_hpc_env.sh /path/to/project
#
# Run this ON A LOGIN NODE: it needs outbound internet. Compute nodes usually
# have none, so everything is installed into the project directory first.
#
# Nothing is installed system-wide and no CUDA module is required: the pip
# wheels bundle their own CUDA runtime, so only the NVIDIA *driver* matters.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FGNN_ROOT="${1:-$PWD/fgnn}"
mkdir -p "$FGNN_ROOT"
FGNN_ROOT="$(cd "$FGNN_ROOT" && pwd)"
VENV="$FGNN_ROOT/env"
export UV_INSTALL_DIR="$FGNN_ROOT/bin"
export UV_CACHE_DIR="$FGNN_ROOT/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$FGNN_ROOT/python"   # keep uv's Python out of a small $HOME quota
export PIP_CACHE_DIR="$FGNN_ROOT/.pip-cache"
UV="$UV_INSTALL_DIR/uv"

echo "=== target: $FGNN_ROOT"
echo "=== free space here:"; df -h "$FGNN_ROOT" | tail -1
echo "    (the environment needs roughly 6 GB; put it on scratch/project space, not a quota'd \$HOME)"

# ---------------------------------------------------------------- glibc check
GLIBC=$(ldd --version | head -1 | grep -oE '[0-9]+\.[0-9]+$')
echo "=== glibc: $GLIBC  (standalone Python needs >= 2.17)"
awk -v v="$GLIBC" 'BEGIN{ if (v+0 < 2.17) { print "    FATAL: glibc too old"; exit 1 } }'

# ------------------------------------------------------------------- GPU check
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version \
               --format=csv,noheader || true
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')
    CC="${CC:-0}"
    if [ "$CC" -ge 90 ] 2>/dev/null; then
        echo "    *** compute capability >= 9.0 (H100/H200/B200)."
        echo "    *** torch 1.13.1+cu117 has NO kernels for this GPU and will fail."
        echo "    *** Set FGNN_TORCH=2.0.1 FGNN_CUDA=cu118 and re-run; see notes."
    elif [ "$CC" -lt 37 ] 2>/dev/null; then
        echo "    *** GPU too old for cu117."
    fi
else
    echo "    no GPU visible here (normal on a login node); the GPU check runs in verify_env.py"
fi

# -------------------------------------------------------------------- versions
TORCH="${FGNN_TORCH:-1.13.1}"
CUDA="${FGNN_CUDA:-cu117}"
PYVER="${FGNN_PYTHON:-3.10}"
echo "=== installing python $PYVER + torch $TORCH+$CUDA + torch_geometric 2.2.0"

# ------------------------------------------------------------------------- uv
if [ ! -x "$UV" ]; then
    echo "=== installing uv into $UV_INSTALL_DIR"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
"$UV" --version

# ----------------------------------------------------------------- virtualenv
[ -x "$VENV/bin/python" ] || "$UV" venv --seed --python "$PYVER" "$VENV"
PY="$VENV/bin/python"
"$PY" -m ensurepip --upgrade >/dev/null
"$PY" -m pip install -q --upgrade pip wheel "setuptools<70"   # old enough for PyG 2.2.0's 2022 setup.py
echo "=== venv python: $($PY -V)"

PIP=("$PY" -m pip install)

# numpy<2 first: torch 1.x is built against the numpy 1.x ABI
"${PIP[@]}" --only-binary=:all: 'numpy==1.24.4' 'scipy==1.10.1'
"${PIP[@]}" "torch==${TORCH}+${CUDA}" --index-url "https://download.pytorch.org/whl/${CUDA}"

# torch_scatter / torch_sparse: resolve exact wheel URLs so pip never builds from source
"$PY" - "$TORCH" "$CUDA" << 'EOF' > "$FGNN_ROOT/.wheels.txt"
import re, sys, urllib.request
from urllib.parse import urljoin
torch_v, cuda = sys.argv[1], sys.argv[2]
tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
index = f"https://data.pyg.org/whl/torch-{torch_v}+{cuda}.html"
html = urllib.request.urlopen(index).read().decode()
for pkg in ("torch_scatter", "torch_sparse"):
    hits = re.findall(rf'href="([^"]*{pkg}-[^"]*{tag}-{tag}-linux_x86_64\.whl[^"]*)"', html)
    if not hits:
        have = sorted(set(re.findall(rf'{pkg}-[^"]*?-(cp\d+)-', html)))
        sys.exit(f"No {pkg} wheel for {tag} at {index}; index has {have}")
    print(urljoin(index, hits[-1]))
EOF
cat "$FGNN_ROOT/.wheels.txt"
# shellcheck disable=SC2046
"${PIP[@]}" $(cat "$FGNN_ROOT/.wheels.txt")

# --only-binary: fail loudly ("No matching distribution") instead of trying to compile on an old system.
"${PIP[@]}" --only-binary=:all: 'scikit-learn==1.3.2' tqdm jinja2 requests pyparsing 'psutil>=5.8.0'
# PyG 2.2.0 exists only as source. Build it with the env's (pinned) setuptools, deps already installed above.
"${PIP[@]}" --no-build-isolation --no-deps 'torch-geometric==2.2.0'
# plotting for plot_results.py / extract_layers.py; numpy named so pip cannot upgrade it past torch 1.13's ABI
"${PIP[@]}" --only-binary=:all: 'matplotlib<3.9' 'numpy==1.24.4'
# pandas is NOT installed: the training code never imports it (analysis runs elsewhere).

# ------------------------------------------------------------------ repo + code
cd "$FGNN_ROOT"
if [ ! -d forwardgnn ]; then
    git clone -q https://github.com/facebookresearch/forwardgnn.git
    (cd forwardgnn && git checkout -q 73fe25720afd0f854e49c099f9c2517c0549617d)   # no `git -C`: git < 1.8.5
fi
mkdir -p "$FGNN_ROOT"/{data,datasplits,results,bench_records,logs,reports,smoke_records}
for d in data datasplits results; do
    rm -rf "forwardgnn/$d"
    ln -s "$FGNN_ROOT/$d" "forwardgnn/$d"
done

BENCH_SRC=""
for d in "$SCRIPT_DIR/fgnn_bench" "$FGNN_ROOT/fgnn_bench_src/fgnn_bench"; do   # repository layout, bundle layout
    [ -f "$d/run_bench.py" ] && { BENCH_SRC="$d"; break; }
done
if [ -n "$BENCH_SRC" ]; then
    rm -rf forwardgnn/src/fgnn_bench
    cp -r "$BENCH_SRC" forwardgnn/src/
    echo "=== fgnn_bench installed into forwardgnn/src/ (from $BENCH_SRC)"
    if [ "$SCRIPT_DIR" != "$FGNN_ROOT" ]; then   # run from a clone of the repository: put the tools in FGNN_ROOT
        cp -n "$SCRIPT_DIR"/*.py "$SCRIPT_DIR"/*.sbatch "$FGNN_ROOT"/ 2>/dev/null || true
        echo "=== job and analysis scripts copied into $FGNN_ROOT (existing files kept)"
    fi
else
    echo "=== NOTE: copy fgnn_bench_src/ here, then re-run, or copy the package manually to forwardgnn/src/"
fi

# --------------------------------------------------------------- env file
cat > "$FGNN_ROOT/activate.sh" << ACT
# source this in job scripts
export FGNN_ROOT="$FGNN_ROOT"
export FGNN_PY="$VENV/bin/python"
export FGNN_SRC="$FGNN_ROOT/forwardgnn/src"
export FGNN_RECORDS_ROOT="$FGNN_ROOT/bench_records"
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=30
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-4}
ACT

echo
echo "=== done"
echo "    source $FGNN_ROOT/activate.sh"
echo "    \$FGNN_PY $FGNN_ROOT/verify_env.py"
