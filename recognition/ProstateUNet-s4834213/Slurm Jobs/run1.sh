#!/bin/bash
#SBATCH --job-name=imporved_unet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=unet_out_%j.log
#SBATCH --error=unet_error_%j.log

# Load conda
source ~/miniconda3/etc/profile.d/conda.sh
# Activate your environment
conda activate myenv

python predict.py
