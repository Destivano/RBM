# RBM

Deep-learning lab (TP) implementing and comparing **Restricted Boltzmann Machines (RBM)**, **Deep Belief Networks (DBN)**, and **Deep Neural Networks (DNN)** in PyTorch, applied to the **Binary AlphaDigits** and **MNIST** datasets.

The name of the repository comes from the Restricted Boltzmann Machine, which is the core building block of the project: DBNs here are stacks of RBMs trained greedily layer-by-layer, and the resulting weights are used to pretrain (initialize) a supervised DNN classifier.

All the actual project code lives in the [`tp_dnn/`](tp_dnn) subdirectory.

## What's implemented

- **RBM** (`tp_dnn/src/models/rbm.py`): a Bernoulli-Bernoulli Restricted Boltzmann Machine built as a PyTorch `nn.Module`, with visible→hidden and hidden→visible probability maps, Bernoulli sampling, and a single-step **Contrastive Divergence (CD-1)** pass used for unsupervised training.
- **DBN** (`tp_dnn/src/models/dbn.py`): a Deep Belief Network built as a stack of RBMs, trained greedily layer-by-layer (each RBM trained on the hidden activations produced by the previous one).
- **DNN** (`tp_dnn/src/models/dnn.py`): a feed-forward MLP classifier (sigmoid hidden layers, linear/logit output layer) that can either be randomly initialized or have its weights initialized from a pretrained DBN (`DNN.from_dbn`), to study the effect of unsupervised pretraining on supervised classification.
- **Datasets**:
  - **Binary AlphaDigits** (`tp_dnn/src/datasets/alphadigits.py`) — loads the `binaryalphadigs.mat` file (36 classes: digits 0-9 and letters A-Z, 39 examples each, 20×16 binary images) and is used to train/evaluate the RBM and DBN as generative models (via reconstruction error and sample generation).
  - **MNIST** (`tp_dnn/src/datasets/mnist.py`) — loaded via `torchvision`, binarized at a 0.5 threshold and flattened to 784-d vectors, used for the supervised DBN-pretrained-vs-random-init DNN comparison and for several sweep experiments.
- **Training loops** (`tp_dnn/src/training/`): `rbm_trainer.py` and `dbn_trainer.py` for unsupervised CD-1 pretraining, `dnn_trainer.py` for supervised fine-tuning with cross-entropy loss, and `eval.py` for computing classification error.
- **Experiments** (`tp_dnn/src/experiments/`), each runnable via the CLI:
  - `alpha_rbm` — train an RBM on Binary AlphaDigits, track reconstruction error, and generate samples.
  - `alpha_dbn` — train a DBN on Binary AlphaDigits, track per-layer reconstruction error, and generate samples.
  - `mnist_compare` — pretrain a DBN on MNIST, initialize one DNN from it (`DNN.from_dbn`) and one DNN randomly, fine-tune both with the same architecture/hyperparameters, and compare train/test classification error.
  - `fig1_layers` — sweep over number of hidden layers and measure the effect on test error.
  - `fig2_width` — sweep over hidden layer width (neurons per layer) and measure the effect on test error.
  - `fig3_datasize` — sweep over training-set size and measure the effect on test error.
- **Utilities** (`tp_dnn/src/utils/`): reproducible seeding, device selection (CPU/GPU), run-directory logging, metrics/CSV saving, and plotting of training curves and generated samples.
- A command-line entry point (`tp_dnn/src/cli.py`, runnable as `python -m src.cli <experiment>`) that dispatches to any of the experiments above.
- Shell scripts (`tp_dnn/scripts/`) to run all MNIST experiments sequentially, plus Slurm job scripts (`tp_dnn/scripts/slurm/`) for running the same experiments on a GPU cluster.

## Key result (from included logs)

From a logged run in `tp_dnn/outputs/logs/mnist_compare_*/metrics.json` (784→500→500→10 architecture, 100 RBM epochs, 200 DNN epochs, lr=0.1, batch size 128):

| Initialization | Train error | Test error |
|---|---|---|
| DBN-pretrained | 0.00065 | 0.018 |
| Random init | 0.00025 | 0.0264 |

i.e. in this run, DBN/RBM pretraining produced a lower test error than random initialization at the same architecture and training budget, consistent with the classic "greedy layer-wise pretraining helps generalization" result this lab is built to demonstrate.

Reconstruction-error logs for the Binary AlphaDigits RBM/DBN runs, generated sample images, and sweep results (`results.csv`) for the `fig1`/`fig2`/`fig3` experiments are also saved under `tp_dnn/outputs/`.

A written report (`tp_dnn/report/noms-TP-DNN.pdf`) summarizing the methodology and results is included in the repository.

## Repository structure

```
RBM/
└── tp_dnn/
    ├── data/            # Raw and processed datasets (not tracked; must be downloaded)
    │   ├── raw/mnist/
    │   └── raw/binary_alpha_digits/
    ├── src/
    │   ├── utils/        # Seeding, device selection, logging, visualization, batching
    │   ├── datasets/      # MNIST and Binary AlphaDigits loaders
    │   ├── models/        # RBM, DBN, DNN implementations
    │   ├── training/       # Training loops and evaluation
    │   ├── experiments/    # Experiment entry points
    │   └── cli.py          # Command-line interface
    ├── scripts/          # Shell and Slurm job scripts
    ├── outputs/          # Logs, figures, samples (generated by running experiments)
    └── report/            # Final report PDF and notes
```

## Prerequisites and dependencies

- Python 3.9+
- PyTorch and torchvision
- numpy, scipy, matplotlib, tqdm

Declared in [`tp_dnn/requirements.txt`](tp_dnn/requirements.txt):

```
torch
torchvision
numpy
matplotlib
scipy
tqdm
```

## Installation

```bash
git clone https://github.com/destivano/RBM.git
cd RBM/tp_dnn
pip install -r requirements.txt
```

### Data

The datasets are not tracked in the repository and must be provided locally:

- **MNIST**: place the standard `MNIST/` folder (as produced by `torchvision.datasets.MNIST`) under `tp_dnn/data/raw/mnist/`. The loader uses `download=False`, so it expects the data to already be present there.
- **Binary AlphaDigits**: place `binaryalphadigs.mat` under `tp_dnn/data/raw/binary_alpha_digits/`.

## Usage

All commands below are run from the `tp_dnn/` directory.

Run a single experiment via the CLI:

```bash
python -m src.cli alphadigits_rbm    # RBM on Binary AlphaDigits
python -m src.cli alphadigits_dbn    # DBN on Binary AlphaDigits
python -m src.cli mnist_compare      # Pretrained vs random DNN on MNIST
python -m src.cli fig1_layers        # Error vs number of hidden layers
python -m src.cli fig2_width         # Error vs neurons per layer
python -m src.cli fig3_datasize      # Error vs training-set size
```

An optional `--seed` flag overrides the default seed defined in `src/config.py`:

```bash
python -m src.cli mnist_compare --seed 123
```

Run all MNIST experiments sequentially:

```bash
bash scripts/run_all.sh
```

### Slurm (GPU cluster)

```bash
bash scripts/slurm/submit_all.sh
```

Each experiment writes logs, metrics (`metrics.json`), CSV results, and figures/sample images to a timestamped directory under `tp_dnn/outputs/`.

## Notes

- Default hyperparameters (epochs, learning rate, batch size, layer widths, experiment sweep grids, random seed) are centralized in `tp_dnn/src/config.py`.
- `RBM.generate` and `DBN.generate` (defined directly on the model classes) are unimplemented stubs (`raise NotImplementedError`) in this snapshot of the code. Sample generation for the `alpha_rbm`/`alpha_dbn` experiments is instead performed via free-running Gibbs sampling helper functions (`sample_rbm` in `tp_dnn/src/training/rbm_trainer.py`, and the equivalent in `dbn_trainer.py`), which start from random binary visible vectors and alternate v→h→v for a fixed number of steps. Generated sample images from these runs are saved under `tp_dnn/outputs/samples/`.
