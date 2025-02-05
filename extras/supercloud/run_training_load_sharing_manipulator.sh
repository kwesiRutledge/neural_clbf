#!/bin/bash

#SBATCH -J load_sharing_manipulator_training
#SBATCH -o outputs/loaded-manipulator-%j.stdout
#SBATCH -e outputs/loaded-manipulator-%j.stderr
#SBATCH -c 40
#SBATCH --gres=gpu:volta:2
#SBATCH --time=30:00:00

# Write your commands here

# Load Anaconda
module load anaconda/2023a

# Activate Environment
source activate neural_clbf

# Enter Training Directory for scalar_demo_capa2_system
cd /home/gridsan/krutledge/neural_clbf/neural_clbf/training/adaptive/
python train_load_sharing_manipulator.py \
  --max_epochs 111 --clf_lambda 1.0 \
  --safe_level 10.0 \
  --num_cpu_cores 20 --number_of_gpus 2 \
  --include_oracle_loss True --barrier True \
  --gradient_clip_val 5000.0 --clf_relaxation_penalty 1e3 \
  --learn_shape_epochs 55 --learn_boundary_epochs 30 \
  --diff_qp_layer_to_use cvxpylayer

