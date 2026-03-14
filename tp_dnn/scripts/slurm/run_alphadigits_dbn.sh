#!/usr/bin/env bash
#SBATCH --job-name=alphadigits_dbn
#SBATCH --partition=ENSTA-l40s
#SBATCH --nodelist=ensta-l40s02.r2.enst.fr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/logs/alphadigits_dbn_%j.out
#SBATCH --error=outputs/logs/alphadigits_dbn_%j.err

source $(conda info --base)/etc/profile.d/conda.sh
conda activate dl2

cd ~/dl2/tp_dnn

python -m src.cli alphadigits_dbn --seed 0
