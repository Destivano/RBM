# TP Deep Neural Networks

Deep Learning lab: RBM → DBN → DNN pipeline on binarized MNIST and Binary AlphaDigits.

## Project structure

```
tp_dnn/
├─ data/            # Raw and processed datasets (not tracked)
├─ src/             # All source code
│  ├─ utils/        # Seeding, device, logging, visualization, batching
│  ├─ datasets/     # MNIST and AlphaDigits loaders
│  ├─ models/       # RBM, DBN, DNN implementations
│  ├─ training/     # Training loops and evaluation
│  ├─ experiments/  # Experiment scripts
│  └─ cli.py        # Command-line interface
├─ scripts/         # Shell and Slurm job scripts
├─ outputs/         # Logs, figures, samples (not tracked)
└─ report/          # Final report PDF and notes
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

All experiments are launched via the CLI:

```bash
# Run a single experiment
python -m src.cli alphadigits_rbm
python -m src.cli alphadigits_dbn
python -m src.cli mnist_compare
python -m src.cli fig1_layers
python -m src.cli fig2_width
python -m src.cli fig3_datasize

# Run all experiments sequentially
bash scripts/run_all.sh
```

### Slurm (GPU cluster)

```bash
bash scripts/slurm/submit_all.sh
```

## Dependencies

- Python 3.9+
- PyTorch, torchvision
- numpy, matplotlib, scipy, tqdm
