#!/bin/bash
#SBATCH --job-name=fig3_datasize
#SBATCH --output=fig3_datasize_%j.out
#SBATCH --error=fig3_datasize_%j.err
#SBATCH --partition=ENSTA-l40s
#SBATCH --nodelist=ensta-l40s02.r2.enst.fr
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

source $(conda info --base)/etc/profile.d/conda.sh
conda activate dl2

cd ~/dl2/tp_dnn

python -m src.cli fig3_datasize --seed 0
