#!/bin/bash
#SBATCH --job-name=fig1_layers
#SBATCH --output=fig1_layers_%j.out
#SBATCH --error=fig1_layers_%j.err
#SBATCH --partition=ENSTA-l40s
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

source $(conda info --base)/etc/profile.d/conda.sh
conda activate dl2

cd ~/dl2/tp_dnn

python -m src.cli fig1_layers --seed 0