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
python train_scalar_demo_capa2_system.py --max_epochs 61 \
  --np_random_seed 11 --include_oracle_loss True --barrier True \
  --include_estimation_error_loss True --gradient_clip_val 1000.0

  /home/gridsan/krutledge/neural_clbf/neural_clbf/training/adaptive/logs/pusher_slider_sticking_force_input/commit_6797883/version_0/checkpoints/epoch=50-step=71757.ckpt
