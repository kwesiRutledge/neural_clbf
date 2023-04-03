#!/bin/bash

#SBATCH -J scalar_demo_capa2_system_training
#SBATCH -o outputs/scalar-%j.stdout
#SBATCH -e outputs/scalar-%j.stderr
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1
#SBATCH --time=24:00:00

# Write your commands here

# Load Anaconda
module load anaconda/2023a

# Activate Environment
source activate neural_clbf

# Enter Training Directory for scalar_demo_capa2_system
cd /home/gridsan/krutledge/neural_clbf/neural_clbf/training/adaptive/
python train_scalar_demo_capa2_system.py --max_epochs 41 --np_random_seed 11 --include_oracle_loss True --barrier True