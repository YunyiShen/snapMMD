#!/bin/bash
# Re-run JKOnet* fits whose outputs predate the one-step-ahead arrays and the parameter pickle (2026-10-04 09:35).
# Usage: bash run_jkonet_redo.sh [N_PAR] [pattern]  — reruns the jobs of joblist_jkonet.txt matching pattern whose log is absent.
N_PAR=${1:-2}; PATTERN=${2:-potential}
cd "$(dirname "$0")"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate jkonet-star
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
run_one() {
  set -- $1
  log=results/logs/${1}_${2}_${3}_${4}.log
  [ -f "$log" ] && exit 0
  python run_baselines.py --method $1 --variant $2 --task $3 --seed $4 > "$log" 2>&1
}
export -f run_one
grep "$PATTERN" results/logs/joblist_jkonet.txt | xargs -P $N_PAR -I{} bash -c 'run_one "{}"'
