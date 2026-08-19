#!/bin/bash

# Create necessary directories
mkdir -p ../models/GoM ../models/LV ../models/pbmc ../models/repressilator ../models/repressilator_missing_obs
mkdir -p ../results/GoM ../results/LV ../results/pbmc ../results/repressilator ../results/repressilator_missing_obs

# Define the seeds to use
seeds=(1 2 3 4 5 40 41 42 43 44)

# Function to run a script with all seeds
run_script() {
    script=$1
    echo "Running $script with all seeds..."
    for seed in "${seeds[@]}"; do
        echo "Running $script with seed $seed"
        python "$script" --seed "$seed"
        echo "Completed $script with seed $seed"
    done
    echo "Finished all seeds for $script"
    echo ""
}

# Run all scripts with all seeds
echo "Starting all runs with seeds: ${seeds[*]}"
echo ""

run_script "GoM.py"
run_script "LV.py"
run_script "pbmc.py"
run_script "repressilator.py"
run_script "repressilator_missing_obs.py"

echo "All runs completed successfully!"
