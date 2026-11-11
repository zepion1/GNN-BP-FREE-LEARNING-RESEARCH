# GNN-BP-FREE-LEARNING-RESEARCH
Master Project with Dr. Chao Huang and Anower Zihad (GNN BP-Free Learning Research/Project)

Reproduction and extension of *Forward Learning of Graph Neural Networks* (ForwardGNN, Park et al., ICLR 2024):
backpropagation against forward learning (single forward and its top-down variants) for GCN, GraphSAGE and GAT,
from 1 to 16 layers, measuring accuracy, macro/micro-F1, peak GPU memory and training time, plus an ablation of
single forward's virtual nodes.

## Repository structure

| Path | Contents |
|---|---|
| `Notes.md`, `Notes_FORWARDGNN.md`, `ForwardGNN_Additional_Notes.md` | Study notes on BP-free learning and ForwardGNN |
| `PAPERS/` | Papers and progress reports |
| `Artifacts/` | Interactive explainer of single forward |
| `Experiments/forwardgnn_pubmed_colab.ipynb` | Colab: GCN on PubMed (backprop, single forward, top-down) |
| `Experiments/forwardgnn_gat_sage_memory.ipynb` | Colab: GAT and SAGE, with GPU memory recording |
| `Experiments/forwardgnn_webkb_reddit_benchmark.ipynb` | Colab: WebKB and Reddit2 benchmark, depths 1-4; origin of the `fgnn_bench` harness |
| `Experiments/hpc/` | Cluster code: environment setup, job generation, SLURM array job, virtual-node ablation, analysis. See its README |
| `Results/` | Run records, figures and tables from the cluster. See its README |

## Experiments

1. **GCN reproduction (Colab).** Backprop vs single forward vs single forward top-down on PubMed.
2. **GAT and SAGE (Colab).** Added architectures and peak GPU memory.
3. **Benchmark harness (Colab).** `fgnn_bench`: F1 scores, per-run records, sparse message passing, Reddit2 and WebKB, depths 1-4.
4. **Deep sweeps (cluster).** PubMed at depths 1-16, Reddit2 at depths 6-16, for all methods and architectures.
5. **Virtual-node ablation (cluster).** Single forward without label flow back into the graph (SF-uni) and without
   virtual nodes (SF-noVN), GCN and SAGE, PubMed and Reddit2, depths 1-16.
6. **Layer-by-layer analysis.** How much each single-forward layer contributes, read from the training logs.

## Credits

ForwardGNN code: [facebookresearch/forwardgnn](https://github.com/facebookresearch/forwardgnn), licensed
CC BY-NC 4.0. It is not included in this repository; the setup script clones it at commit `73fe257`.
