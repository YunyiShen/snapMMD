#!/bin/bash
# Campaign 4, JKOnet* block (added 2026-10-04): N_PAR processes in parallel in the `jkonet-star` conda env (JAX).
# Usage: bash run_jkonet.sh [N_PAR]. Skips jobs whose log file already exists.
# Order: the authors' default solver (potential only) on every task, PBMC first; then the full model (potential +
# diffusion + interaction) on the four synthetic tasks. The full model is not queued on PBMC: its Gaussian-mixture
# density estimate has no usable component there (30 dimensions, 500 cells; see RUNLOG), which would leave the
# diffusion term untrained and beta at its random initial value.
N_PAR=${1:-3}
cd "$(dirname "$0")"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate jkonet-star
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
mkdir -p results/logs
jobs=results/logs/joblist_jkonet.txt; : > $jobs
for task in PBMC LV ReprParam ReprProtein GoM; do
  for seed in 1 2 3 4 5 40 41 42 43 44; do echo "jkonet potential $task $seed" >> $jobs; done
done
for task in LV ReprParam ReprProtein GoM; do
  for seed in 1 2 3 4 5 40 41 42 43 44; do echo "jkonet full $task $seed" >> $jobs; done
done
run_one() {
  set -- $1
  log=results/logs/${1}_${2}_${3}_${4}.log
  [ -f "$log" ] && exit 0
  python run_baselines.py --method $1 --variant $2 --task $3 --seed $4 > "$log" 2>&1
}
export -f run_one
cat $jobs | xargs -P $N_PAR -I{} bash -c 'run_one "{}"'
