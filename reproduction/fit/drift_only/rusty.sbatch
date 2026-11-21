#!/usr/bin/env bash
# Submit from the repository root after pulling the intended commit.
# Pilot: 4 fits, one seed, 20 epochs, <=2 allocated CPU-hours in total.
# Full seed-1 run, after measuring pilot cost:
#   sbatch --time=02:00:00 --export=ALL,SNAPMMD_EPOCHS=1000,SNAPMMD_STAGE=full reproduction/fit/drift_only/rusty.sbatch
# Only explicitly selecting --array=0-39%2 expands to ten seeds (not a default).
#SBATCH --job-name=drift-fixed
#SBATCH --partition=genx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00
#SBATCH --array=0-3%2
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
ROOT=${SNAPMMD_REPO:-$SLURM_SUBMIT_DIR}
ROOT=$(cd "$ROOT" && pwd -P)
PY=${SNAPMMD_PYTHON:-$HOME/venvs/snapmmd-drift/bin/python}
RUNS=${SNAPMMD_RUNS:-/mnt/ceph/users/$USER/snapmmd-runs}
EPOCHS=${SNAPMMD_EPOCHS:-20}
STAGE=${SNAPMMD_STAGE:-pilot}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
SEEDS=(1 2 3 4 5 40 41 42 43 44)
TASKS=(LV LV ReprParam ReprParam)
DRIFTS=(structured mlp structured mlp)
ID=${SLURM_ARRAY_TASK_ID:?submit as an array}
[[ "$ID" =~ ^[0-9]+$ ]] && (( ID < 40 )) || { echo "array id must be 0..39" >&2; exit 1; }
INDEX=$((ID % 4))
SEED=${SEEDS[$((ID / 4))]}
TASK=${TASKS[$INDEX]}
DRIFT=${DRIFTS[$INDEX]}
OUT="$RUNS/$STAGE/${SLURM_ARRAY_JOB_ID}/${TASK}_${DRIFT}_s${SEED}"
"$PY" -c 'import torch, torchsde, numpy, ot; print("torch", torch.__version__, "torchsde", torchsde.__version__, "numpy", numpy.__version__, "POT", ot.__version__)'
echo "start=$(date -u +%FT%TZ) host=$(hostname) job=${SLURM_JOB_ID} task=$TASK drift=$DRIFT seed=$SEED output=$OUT"
git -C "$ROOT" rev-parse HEAD
git -C "$ROOT" status --short
exec "$PY" "$ROOT/reproduction/fit/drift_only/train.py" \
    --task "$TASK" --drift "$DRIFT" --seed "$SEED" --epochs "$EPOCHS" \
    --threads 1 --device cpu --output "$OUT"
