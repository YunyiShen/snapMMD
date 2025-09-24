#!/usr/bin/env bash
#!/bin/bash
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 0-06:00
#SBATCH --mem=32G
#SBATCH -o ./logs/classic_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./logs/classic_%j.err  # File to which STDERR will be written, %j inserts jobid

#SBATCH --array=0-19  # four jobs (0,1,2,3)

module load miniforge/24.3.0-0; 
conda activate torch
python classic_sde.py --id $SLURM_ARRAY_TASK_ID
