#!/usr/bin/env bash
#SBATCH --job-name=fgnn
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
# --- SITE-SPECIFIC: uncomment and fill in from your cluster's docs ----------
##SBATCH --partition=gpu
##SBATCH --account=your_project
##SBATCH --qos=normal
# ---------------------------------------------------------------------------
# --array, --time and --mem are given on the command line (gen_jobs.py prints them).
# Submit from $FGNN_ROOT so logs/ resolves:  cd $FGNN_ROOT && sbatch --array=... run_array.sbatch
set -uo pipefail

FGNN_ROOT="${FGNN_ROOT:-$SLURM_SUBMIT_DIR}"
source "$FGNN_ROOT/activate.sh"
JOBS="${FGNN_JOBS:-$FGNN_ROOT/jobs.tsv}"

LINE=$(awk -v n="$SLURM_ARRAY_TASK_ID" -F'\t' 'NR==n+1' "$JOBS")
if [ -z "$LINE" ]; then echo "no job at index $SLURM_ARRAY_TASK_ID in $JOBS"; exit 0; fi
IFS=$'\t' read -r IDX TIER DATASET ARCH METHOD DEPTH ESTCLASS COMMAND <<< "$LINE"
COMMAND="${COMMAND//RECORDS_ROOT/$FGNN_RECORDS_ROOT}"
TAG="${DATASET}-${ARCH}-${METHOD}-L${DEPTH}"

echo "=== [$SLURM_ARRAY_TASK_ID] $TAG  class=$ESTCLASS"
echo "=== $(hostname)  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "=== start $(date -Is)"

if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "=== no GPU on $(hostname): refusing to run (wrong partition, or --gres missing?)"; exit 1
fi
cd "$FGNN_SRC" || exit 1
RC=0
# shellcheck disable=SC2086
"$FGNN_PY" $COMMAND || RC=$?

echo "=== end $(date -Is)"
case $RC in
  0) echo "=== $TAG OK" ;;
  3) echo "=== $TAG OOM -- recorded as a result" ;;
  *) echo "=== $TAG FAILED rc=$RC -- see output above" ;;
esac
exit 0   # OOM/failure are recorded in bench_records; don't let one element mark the array failed
