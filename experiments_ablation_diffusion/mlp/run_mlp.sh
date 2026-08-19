#!/bin/bash
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 0-06:00
#SBATCH -p mit_normal
#SBATCH --mem=32G
#SBATCH -o ./logs/repres_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./logs/repres_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
#SBATCH --mail-user=yshen99@mit.edu
#SBATCH --array=0-9  # four jobs (0,1,2,3)

module load miniforge/24.3.0-0; 
conda activate torch
python MLP.py --id $SLURM_ARRAY_TASK_ID