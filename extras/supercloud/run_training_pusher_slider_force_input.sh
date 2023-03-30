#!/bin/bash

#SBATCH -J pusher_slider_training
#SBATCH -o outputs/pusher-slider-%j.stdout
#SBATCH -e outputs/pusher-slider-%j.stderr
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
python train_pusher_slider_force_input.py \
  --max_epochs 71 --clf_lambda 1.0 \
  --use_oracle True --barrier True --gradient_clip_val 100.0