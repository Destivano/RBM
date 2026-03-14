#!/bin/bash
#SBATCH --job-name=mnist_compare
#SBATCH --output=mnist_compare_%j.out
#SBATCH --error=mnist_compare_%j.err
#SBATCH --partition=ENSTA-l40s
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dl2

# Go to project root
cd ~/dl2/tp_dnn

# Run experiment
python -m src.cli mnist_compare --seed 0