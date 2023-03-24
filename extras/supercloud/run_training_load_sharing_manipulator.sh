#!/bin/bash

#SBATCH -J load_sharing_manipulator_training
#SBATCH -o outputs/loaded-manipulator-%j.stdout
#SBATCH -e outputs/loaded-manipulator-%j.stderr
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
python train_load_sharing_manipulator.py \
  --max_epochs 31 \
  --use_oracle True --barrier True