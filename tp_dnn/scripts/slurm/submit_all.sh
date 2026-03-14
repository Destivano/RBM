#!/usr/bin/env bash
# submit_all.sh – Submit the four main MNIST experiments to Slurm.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Submitting MNIST experiments..."

sbatch "$SCRIPT_DIR/run_mnist_compare.sbatch"
sbatch "$SCRIPT_DIR/run_fig1_layers.sbatch"
sbatch "$SCRIPT_DIR/run_fig2_width.sbatch"
sbatch "$SCRIPT_DIR/run_fig3_datasize.sbatch"

echo "All MNIST jobs submitted."
