# Results

Filled from the cluster by `Experiments/hpc/save_results_to_repo.sh`. Every figure and table here can be
regenerated from `bench_records/` with the scripts in `Experiments/hpc/`.

## bench_records/

One JSON file per run, at `<folder>/<dataset>/<model>/L<depth>/run<i>.json`, holding the run's status (`ok`, `oom`
or `error`), accuracy, macro/micro/weighted F1, per-class precision/recall/F1, peak GPU memory during training,
training time, epochs per layer, GPU name and the settings used.

| Folder | Contents |
|---|---|
| `pubmed-sparse-row_l1/` | PubMed main runs, depths 1-16: backprop, SF, SF-Top2Input, SF-Top2Loss (GCN, SAGE, GAT) |
| `pubmed-sparse-row_l1-ablation/` | PubMed ablation, depths 1-16: SF-uni and SF-noVN (GCN, SAGE) |
| `webkb-reddit-sparse-standardize/` | Reddit2 main runs on the cluster, depths 6-16 (GAT recorded as out of memory) |
| `webkb-reddit-sparse-standardize-ablation/` | Reddit2 ablation, depths 1-16: SF-uni and SF-noVN (GCN, SAGE) |
| `probe-reddit/` | 6-epoch, 1-run probes of every Reddit2 configuration, used only to size the cluster jobs |
| `colab/` | Records from Colab (Reddit2 and WebKB, depths 1-4), if copied to the cluster |

Combining Colab depths 1-4 with the cluster's depths 6-16 for Reddit2 (cluster copies win):

```bash
mkdir -p bench_records/reddit2-merged
cp -r bench_records/colab/webkb-reddit-sparse-standardize/Reddit2 bench_records/reddit2-merged/
cp -r bench_records/webkb-reddit-sparse-standardize/Reddit2/. bench_records/reddit2-merged/Reddit2/
```

## reports/

Figures (`overview_<dataset>.png` and one PNG per metric), `summary.md` / `summary.csv` per analysis, and
`layers/` (layer-by-layer analysis of single-forward runs: `summary.md`, `layers.csv`, `cumulative_acc.png`,
`reset_share.png`).

## job_lists/

The `jobs*.tsv` files that defined each sweep. Each line is one configuration and the exact command it ran.
