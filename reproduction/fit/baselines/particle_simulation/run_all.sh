#!/bin/bash
# Campaign 4: all fits, N_PAR processes in parallel. Usage: bash run_all.sh [N_PAR]
# Job list: method variant task seed. Skips jobs whose log file already exists.
N_PAR=${1:-10}
cd "$(dirname "$0")"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate snapmmd
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results/logs
# Order: the authors' default variant of each method, PBMC first within each block (the slowest).
jobs=results/logs/joblist.txt; : > $jobs
# The two non-default noise variants (prescient sd0.1, pisde const0.5) are not run; PBMC first.
for block in "scnode default" "prescient sd0.5" "pisde const0.1" "pisde mlp"; do
  for task in PBMC LV ReprParam ReprProtein GoM; do
    for seed in 1 2 3 4 5 40 41 42 43 44; do
      echo "$block $task $seed" >> $jobs
    done
  done
done
run_one() {
  set -- $1
  log=results/logs/${1}_${2}_${3}_${4}.log
  [ -f "$log" ] && exit 0      # finished or in progress (a fit started by an earlier launcher)
  python run_baselines.py --method $1 --variant $2 --task $3 --seed $4 > "$log" 2>&1
}
export -f run_one
cat $jobs | xargs -P $N_PAR -I{} bash -c 'run_one "{}"'
