#!/usr/bin/env bash
# Copy the results worth keeping from the cluster working folder into a clone of the repository.
#
#   bash save_results_to_repo.sh <FGNN_ROOT> <repository clone>
#
# Into <repo>/Results/ it copies
#   bench_records/  every run record (JSON): main runs, ablation, Reddit2 probe, Colab depth 1-4 records
#                   (left out: reddit2-merged/, which is rebuilt from the others, and uploaded .zip files)
#   reports/        figures, summary tables, the layer-by-layer analysis
#   job_lists/      the jobs*.tsv files that defined each sweep
# It never copies data, raw downloads, splits, the Python environment, caches, logs or forwardgnn itself.
# Absolute paths inside the copied text files (error tracebacks) are replaced by $FGNN_ROOT / $HOME.
# Safe to re-run: files are added or updated, nothing in the repository is deleted.
set -euo pipefail
SRC="$(cd "${1:?usage: save_results_to_repo.sh <FGNN_ROOT> <repository clone>}" && pwd)"
REPO="$(cd "${2:?usage: save_results_to_repo.sh <FGNN_ROOT> <repository clone>}" && pwd)"
[ -d "$SRC/bench_records" ] || { echo "no bench_records/ in $SRC"; exit 1; }
[ -d "$REPO/.git" ] || { echo "$REPO is not a git clone"; exit 1; }
DEST="$REPO/Results"
mkdir -p "$DEST/bench_records" "$DEST/reports" "$DEST/job_lists"

copy_tree() {  # copy_tree <from> <to> [tar exclude patterns...]
    local from="$1" to="$2"; shift 2
    local ex=(--exclude=.git); for p in "$@"; do ex+=(--exclude="$p"); done
    (cd "$from" && tar cf - "${ex[@]}" .) | (cd "$to" && tar xf -)
}
copy_tree "$SRC/bench_records" "$DEST/bench_records" './reddit2-merged' '*.zip'
[ -d "$SRC/reports" ] && copy_tree "$SRC/reports" "$DEST/reports"
cp "$SRC"/jobs*.tsv "$DEST/job_lists/" 2>/dev/null || true

# scrub absolute paths (they appear in tracebacks of failed / out-of-memory runs)
n=0
while IFS= read -r -d '' f; do
    if grep -q -e "$SRC" -e "$HOME" "$f" 2>/dev/null; then
        sed -i -e "s|$SRC|\$FGNN_ROOT|g" -e "s|$HOME|\$HOME|g" "$f"; n=$((n + 1))
    fi
done < <(find "$DEST" -type f \( -name '*.json' -o -name '*.md' -o -name '*.csv' -o -name '*.txt' -o -name '*.tsv' \) -print0)

echo "copied into $DEST:"
echo "  run records : $(find "$DEST/bench_records" -name 'run*.json' | wc -l)"
for d in "$DEST"/bench_records/*/; do echo "      $(basename "$d"): $(find "$d" -name 'run*.json' | wc -l)"; done
echo "  reports     : $(find "$DEST/reports" -type f | wc -l) files"
echo "  job lists   : $(ls "$DEST/job_lists" | wc -l) files"
echo "  paths scrubbed in $n file(s)"
du -sh "$DEST"
big=$(find "$DEST" -type f -size +50M)
[ -n "$big" ] && { echo "WARNING: files over 50 MB (GitHub rejects files over 100 MB):"; echo "$big"; } || true
