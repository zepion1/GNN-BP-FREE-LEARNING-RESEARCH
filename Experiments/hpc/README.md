# ForwardGNN experiments on an HPC cluster

Code for the deep-layer sweeps (depths up to 16) and the virtual-node ablation of single forward, run on a
SLURM cluster with NVIDIA A100 GPUs. It builds on
[facebookresearch/forwardgnn](https://github.com/facebookresearch/forwardgnn) (Park et al., *Forward Learning of
Graph Neural Networks*, ICLR 2024), pinned at commit `73fe25720afd0f854e49c099f9c2517c0549617d`. forwardgnn is
not included here: `setup_hpc_env.sh` clones it. It is licensed CC BY-NC 4.0 by its authors.

The benchmark harness (`fgnn_bench/`) began in the Colab notebook
`../forwardgnn_webkb_reddit_benchmark.ipynb` and was moved here when the work migrated to the cluster.

## Files

| File | Purpose |
|---|---|
| `fgnn_bench/run_bench.py` | Runs a forwardgnn training script unchanged and records, per run: accuracy, macro/micro/weighted F1, per-class scores, peak GPU memory during training, and training time with and without test evaluations |
| `fgnn_bench/datasets_ext.py` | Adds Reddit2 and WebKB (Cornell, Texas, Washington) to forwardgnn's loader; downloads are verified by size and SHA-256 |
| `fgnn_bench/sparse_backend.py` | Optional sparse-matrix message passing (needed to fit Reddit2 on a 40 GB GPU) |
| `fgnn_bench/check_backend.py` | Confirms the sparse backend computes the same function as forwardgnn's default |
| `fgnn_bench/sf_novn.py` | **SF-noVN** ablation: single forward with the virtual nodes removed |
| `setup_hpc_env.sh` | Builds the Python environment without root on a cluster whose system Python is too old |
| `verify_env.py`, `check_data.py` | Check the environment on a GPU node, and the datasets and splits |
| `gen_jobs.py` | Job list for the main sweeps (backprop and the forward methods) |
| `plan_from_probe.py` | Sizes the Reddit2 jobs from a short probe run of every configuration |
| `gen_ablation_jobs.py`, `check_ablation.py` | Job list for the ablation (SF-uni, SF-noVN), and a self-test of both variants |
| `run_array.sbatch` | SLURM array job: runs one line of a job list per array task |
| `plot_results.py` | Accuracy, macro-F1, memory and time against depth, plus summary tables |
| `extract_layers.py` | Layer-by-layer analysis of single-forward runs, read from the job logs |
| `diag_memory.py` | Diagnostic for SF-noVN's memory growth (pending) |
| `save_results_to_repo.sh` | Copies results from the cluster into `../../Results/` |

## Environment

Tested on a cluster with CentOS 7 (system Python 3.6, glibc 2.17, git 1.8.3), SLURM, and NVIDIA
A100-PCIE-40GB GPUs, without root access. `setup_hpc_env.sh` uses [uv](https://github.com/astral-sh/uv) to install a
standalone Python 3.10 inside the project folder, then installs:

| Package | Version |
|---|---|
| torch | 1.13.1+cu117 |
| torch_geometric | 2.2.0 (forwardgnn vendors PyG 2.2 internals, so newer versions do not work) |
| torch_scatter / torch_sparse | 2.1.1 / 0.6.17 (prebuilt wheels for torch 1.13 + cu117) |
| numpy / scipy / scikit-learn | 1.24.4 / 1.10.1 / 1.3.2 |
| matplotlib | < 3.9 |

The CUDA 11.7 build has kernels up to `sm_86` (A100 is `sm_80`). It does not support H100 (`sm_90`).

## Setup

Run on a login node, which needs internet access:

```bash
git clone https://github.com/zepion1/GNN-BP-FREE-LEARNING-RESEARCH.git
bash GNN-BP-FREE-LEARNING-RESEARCH/Experiments/hpc/setup_hpc_env.sh $HOME/SF_Experiments/fgnn
```

This creates the environment, clones forwardgnn at the pinned commit, installs `fgnn_bench`, copies the job and
analysis scripts into the project folder, and writes `activate.sh`. Then, on a GPU node:

```bash
source $HOME/SF_Experiments/fgnn/activate.sh
$FGNN_PY $FGNN_ROOT/verify_env.py        # must end with ENVIRONMENT OK
```

## Data

| Dataset | Nodes | Directed edges | Features | Classes | Source |
|---|---|---|---|---|---|
| CitationFull-PubMed | 19,717 | 88,648 | 500 | 3 | PyG, downloaded by forwardgnn |
| Reddit2 | 232,965 | 23,213,838 | 602 | 41 | GraphSAINT release (Google Drive) |
| WebKB Cornell / Texas / Washington | 195 / 187 / 230 | small | 1,703 | 5 | LINQS |

```bash
cd $FGNN_SRC
FGNN_FEATURE_NORM=standardize $FGNN_PY -m fgnn_bench.datasets_ext Reddit2 WebKB-Cornell WebKB-Texas WebKB-Washington \
    --raw-dir $FGNN_ROOT/raw --splits --report $FGNN_ROOT/reports/dataset_report.md
$FGNN_PY $FGNN_ROOT/check_data.py CitationFull-PubMed Reddit2
```

Splits are forwardgnn's own: 5-fold `KFold` with `random_state=101`, 20% of the training part held out for
validation. Because the raw files are checksum-verified and processing is deterministic, regenerated splits equal
the ones used in Colab.

If the WebKB host (`linqs-data.soe.ucsc.edu`) is unreachable from the cluster, place `WebKB.tgz` (132,209 bytes,
SHA-256 `62630a24...5ab4`) in `$FGNN_ROOT/raw/WebKB-<university>/` and run the command again. The GitHub dataset
used in earlier plans could not be obtained: PyG's source host (`graphmining.ai`) was unreachable.

## Settings used in every run

`--epochs 1000 --val-every 2 --lr 0.001 --patience 100 --num-hidden 128 --num-runs 5 --seed 100`,
`--mp-backend sparse`. Forward methods add `--append-label none --aug-edge-direction bidirection`
(top-down methods also `--alternating-update`). Features: Reddit2 and WebKB `standardize`; PubMed always gets
forwardgnn's row-L1 `NormalizeFeatures`, whatever `--feature-norm` says, so its runs are recorded as `row_l1`.

## Running the sweeps

All commands run from `$FGNN_ROOT`. Each job list is a TSV; `run_array.sbatch` runs line *i* as array task *i*,
writes `logs/fgnn-<job>_<task>.out`, refuses to run without a GPU, and records an out-of-memory run as a result
rather than a failure. Runs already recorded as `ok` are skipped, so resubmitting a list resumes it.

**Main sweeps.** PubMed at depths 1-16, Reddit2 at depths 6-16 (depths 1-4 were run in Colab):

```bash
$FGNN_PY gen_jobs.py --records-root $FGNN_RECORDS_ROOT --groups pubmed --out jobs_pubmed.tsv
$FGNN_PY gen_jobs.py --records-root $FGNN_RECORDS_ROOT --groups reddit --depths 6 8 12 16 --out jobs_reddit_deep.tsv
```

**Reddit2 sizing.** Probe every configuration with 6 epochs and 1 run into a separate folder, then let the
planner group the configurations by out-of-memory, about a day, or about a week, and print the `sbatch` commands:

```bash
sed -e 's/--epochs 1000/--epochs 6/' -e 's/--patience 100/--patience 5/' -e 's/--num-runs 5/--num-runs 1/' \
    -e 's/webkb-reddit-sparse-standardize/probe-reddit/g' jobs_reddit_deep.tsv > jobs_reddit_probe.tsv
FGNN_JOBS=$PWD/jobs_reddit_probe.tsv sbatch --partition=gpu-short --array=1-48%6 --time=02:00:00 --mem=64G run_array.sbatch
$FGNN_PY plan_from_probe.py --jobs jobs_reddit_deep.tsv --probe-root bench_records/probe-reddit --prefix jobs_reddit
```

Partition names (`gpu-short`, `gpu-day-long`, `gpu-week-long`) are those of the cluster used; adjust them for another site.

**Virtual-node ablation.** GCN and SAGE, PubMed and Reddit2, depths 1-16:

| Variant | What changes relative to single forward (SF) |
|---|---|
| SF-uni | `--aug-edge-direction unidirection`: virtual nodes still aggregate their class members but send nothing back, so no label information flows into the real nodes |
| SF-noVN | No virtual nodes in the graph; each layer gets one learnable vector per class (initialised like a bias-free `nn.Linear`) in their place, with forwardgnn's own loss, early stopping and summed-softmax prediction unchanged |

```bash
$FGNN_PY check_ablation.py            # self-test on the real forwardgnn code; must print ALL CHECKS PASSED
$FGNN_PY gen_ablation_jobs.py --records-root $FGNN_RECORDS_ROOT --out jobs_ablation.tsv
```

## Analysis

```bash
$FGNN_PY plot_results.py --records bench_records/pubmed-sparse-row_l1 bench_records/pubmed-sparse-row_l1-ablation \
    --methods BP SF SF-uni SF-noVN --archs GCN SAGE --out reports/ablation-pubmed
$FGNN_PY extract_layers.py --logs logs --out reports/layers
```

For Reddit2 at all depths, first combine the Colab records (depths 1-4) and the cluster records, cluster last:

```bash
mkdir -p bench_records/reddit2-merged
cp -r bench_records/colab/webkb-reddit-sparse-standardize/Reddit2 bench_records/reddit2-merged/
cp -r bench_records/webkb-reddit-sparse-standardize/Reddit2/. bench_records/reddit2-merged/Reddit2/
```

## Points to keep in mind when reading the results

- **Training time across hardware.** Reddit2 depths 1-4 ran on Colab's A100-SXM4-40GB and depths 6-16 on the
  cluster's slower A100-PCIE-40GB. Accuracy, F1 and memory are comparable across the two; time is not.
- **Depth 1 is not the same model in backprop and single forward.** Backprop at depth 1 is one graph convolution
  straight to the classes; single forward at depth 1 has a hidden layer plus a per-layer class-scoring step.
- **Single forward predicts with all its layers.** Each layer has its own class scores and the prediction sums them,
  while backprop predicts from its last layer. Early stopping also resets layers that never improve the combined
  validation accuracy. `extract_layers.py` measures how often that happens.
- **GAT on Reddit2** runs out of memory on a 40 GB GPU for every method at depths 6-16: one layer's per-edge
  attention tensors already exceed it.
- **SF-noVN's memory figures are unverified.** Its peak memory grows with depth while SF's does not; `diag_memory.py`
  is meant to find out why. Its accuracy is unaffected.
