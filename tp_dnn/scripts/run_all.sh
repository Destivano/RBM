#!/usr/bin/env bash
# run_all.sh – Run MNIST experiments sequentially.
set -euo pipefail

cd "$(dirname "$0")/.."

echo ">>> mnist_compare"
python -m src.cli mnist_compare

echo ">>> fig1_layers"
python -m src.cli fig1_layers

echo ">>> fig2_width"
python -m src.cli fig2_width

echo ">>> fig3_datasize"
python -m src.cli fig3_datasize

echo "=== All experiments complete ==="
