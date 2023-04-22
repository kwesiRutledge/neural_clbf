#!/bin/bash

#SBATCH -J load_sharing_manipulator_training
#SBATCH -o outputs/loaded-manipulator-%j.stdout
#SBATCH -e outputs/loaded-manipulator-%j.stderr
#SBATCH -c 40
#SBATCH --gres=gpu:volta:4
#SBATCH --time=24:00:00

# Write your commands here

# Load Anaconda
module load anaconda/2023a

# Activate Environment
source activate neural_clbf

# Enter Training Directory for scalar_demo_capa2_system
cd /home/gridsan/krutledge/neural_clbf/neural_clbf/training/adaptive/
python train_load_sharing_manipulator.py \
  --max_epochs 201 --clf_lambda 0.4 \
  --num_cpu_cores 20 --number_of_gpus 4 \
  --include_oracle_loss True --barrier True \
  --gradient_clip_val 10000.0