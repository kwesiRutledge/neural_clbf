#!/bin/bash

#SBATCH -J USERNAME
#SBATCH -o outputs/%j.stdout
#SBATCH -e outputs/%j.stderr
#SBATCH -c 20
#SBATCH --gres=gpu:volta:2
#SBATCH --time=24:00:00

# Write your commands here

# Load Anaconda
module load anaconda/2023a

# Activate Environment
source activate neural_clbf

# Enter Training Directory for scalar_demo_capa2_system
cd /home/gridsan/krutledge/neural_clbf/neural_clbf/training/adaptive/
python train_scalar_demo_capa2_system.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT