#!/usr/bin/env bash
# Same 4-task, 20-epoch pilot as rusty.sbatch; one GPU task at a time.
# Maximum total allocation: 40 GPU-minutes and 40 CPU-minutes.
# Requires setup_rusty.sh cuda. Submit from the repository root.
#SBATCH --job-name=drift-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --array=0-3%1
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
export SNAPMMD_DEVICE=cuda
export SNAPMMD_STAGE=${SNAPMMD_STAGE:-pilot-gpu}
ROOT=${SNAPMMD_REPO:-$SLURM_SUBMIT_DIR}
# Direct shell execution reuses the task mapping; the CPU #SBATCH header is ignored.
exec bash "$ROOT/reproduction/fit/drift_only/rusty.sbatch"
