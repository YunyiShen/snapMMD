#!/usr/bin/env bash
# Submit from the repository root after pulling the intended commit.
# Pilot: 4 fits, one seed, 20 epochs, <=2 allocated CPU-hours in total.
# Full seed-1 run, after measuring pilot cost:
#   sbatch --time=02:00:00 --export=ALL,SNAPMMD_EPOCHS=1000,SNAPMMD_STAGE=full reproduction/fit/drift_only/rusty.sbatch
# Only explicitly selecting --array=0-39%2 expands to ten seeds (not a default).
# rusty_gpu.sbatch runs this body inside its GPU allocation with SNAPMMD_DEVICE=cuda.
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
DEVICE=${SNAPMMD_DEVICE:-cpu}
case "$DEVICE" in
    cpu) DEFAULT_PY="$HOME/venvs/snapmmd-drift/bin/python" ;;
    cuda) DEFAULT_PY="$HOME/venvs/snapmmd-drift-cuda/bin/python" ;;
    *) echo "SNAPMMD_DEVICE must be cpu or cuda" >&2; exit 2 ;;
esac
PY=${SNAPMMD_PYTHON:-$DEFAULT_PY}
RUNS=${SNAPMMD_RUNS:-/mnt/ceph/users/$USER/snapmmd-runs}
EPOCHS=${SNAPMMD_EPOCHS:-20}
STAGE=${SNAPMMD_STAGE:-pilot}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
if [[ "$DEVICE" == cuda ]]; then
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    "$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise RuntimeError("GPU job has no usable CUDA device; refusing a CPU fallback")
# Test an actual CUDA operation, including backward, before any training/data work.
torch.use_deterministic_algorithms(True)
x = torch.ones(4, 4, device="cuda", requires_grad=True)
(x @ x).sum().backward()
torch.cuda.synchronize()
print("CUDA preflight OK:", torch.cuda.get_device_name(), "runtime", torch.version.cuda)
PY
fi
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
if command -v lscpu >/dev/null 2>&1; then lscpu; fi
echo "start=$(date -u +%FT%TZ) host=$(hostname) job=${SLURM_JOB_ID} task=$TASK drift=$DRIFT seed=$SEED output=$OUT"
git -C "$ROOT" rev-parse HEAD
git -C "$ROOT" status --short
exec "$PY" "$ROOT/reproduction/fit/drift_only/train.py" \
    --task "$TASK" --drift "$DRIFT" --seed "$SEED" --epochs "$EPOCHS" \
    --threads 1 --device "$DEVICE" --output "$OUT"
