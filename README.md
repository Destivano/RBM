# 🧠 RBM — Boltzmann Machines Go Brrr

> A PyTorch deep-learning lab where RBMs stack into DBNs, DBNs pretrain DNNs, and we find out whether "unsupervised pretraining helps" still holds up.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=plotly&logoColor=white)](https://matplotlib.org/)
[![Dataset](https://img.shields.io/badge/Datasets-MNIST%20%7C%20Binary%20AlphaDigits-8A2BE2?style=flat)](tp_dnn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](LICENSE)

Deep-learning lab (TP) implementing and comparing **Restricted Boltzmann Machines (RBM)**, **Deep Belief Networks (DBN)**, and **Deep Neural Networks (DNN)** in PyTorch, applied to the **Binary AlphaDigits** and **MNIST** datasets.

The repo's name comes from the Restricted Boltzmann Machine — it's the core building block here: DBNs are stacks of RBMs trained greedily layer-by-layer, and the resulting weights are used to pretrain (initialize) a supervised DNN classifier.

All the actual project code lives in the [`tp_dnn/`](tp_dnn) subdirectory.

---

## 🧩 What's implemented

- **RBM** (`tp_dnn/src/models/rbm.py`) — a Bernoulli-Bernoulli Restricted Boltzmann Machine built as a PyTorch `nn.Module`, with visible→hidden and hidden→visible probability maps, Bernoulli sampling, and a single-step **Contrastive Divergence (CD-1)** pass for unsupervised training.
- **DBN** (`tp_dnn/src/models/dbn.py`) — a Deep Belief Network built as a stack of RBMs, trained greedily layer-by-layer (each RBM trained on the hidden activations produced by the previous one).
- **DNN** (`tp_dnn/src/models/dnn.py`) — a feed-forward MLP classifier (sigmoid hidden layers, linear/logit output layer) that can be randomly initialized or have its weights initialized from a pretrained DBN (`DNN.from_dbn`), to study the effect of unsupervised pretraining on supervised classification.

<details>
<summary><b>📦 Datasets</b></summary>

- **Binary AlphaDigits** (`tp_dnn/src/datasets/alphadigits.py`) — loads `binaryalphadigs.mat` (36 classes: digits 0-9 and letters A-Z, 39 examples each, 20×16 binary images), used to train/evaluate the RBM and DBN as generative models (reconstruction error + sample generation).
- **MNIST** (`tp_dnn/src/datasets/mnist.py`) — loaded via `torchvision`, binarized at a 0.5 threshold and flattened to 784-d vectors, used for the supervised DBN-pretrained-vs-random-init DNN comparison and several sweep experiments.

</details>

<details>
<summary><b>🏋️ Training & experiments</b></summary>

- **Training loops** (`tp_dnn/src/training/`) — `rbm_trainer.py` and `dbn_trainer.py` for unsupervised CD-1 pretraining, `dnn_trainer.py` for supervised fine-tuning with cross-entropy loss, and `eval.py` for classification error.
- **Experiments** (`tp_dnn/src/experiments/`), each runnable via the CLI:
  - `alpha_rbm` — train an RBM on Binary AlphaDigits, track reconstruction error, generate samples.
  - `alpha_dbn` — train a DBN on Binary AlphaDigits, track per-layer reconstruction error, generate samples.
  - `mnist_compare` — pretrain a DBN on MNIST, initialize one DNN from it (`DNN.from_dbn`) and one DNN randomly, fine-tune both identically, and compare train/test classification error.
  - `fig1_layers` — sweep number of hidden layers vs. test error.
  - `fig2_width` — sweep hidden layer width (neurons/layer) vs. test error.
  - `fig3_datasize` — sweep training-set size vs. test error.
- **Utilities** (`tp_dnn/src/utils/`) — reproducible seeding, device selection (CPU/GPU), run-directory logging, metrics/CSV saving, and plotting of training curves and generated samples.
- CLI entry point (`tp_dnn/src/cli.py`, runnable as `python -m src.cli <experiment>`) that dispatches to any experiment above.
- Shell scripts (`tp_dnn/scripts/`) to run all MNIST experiments sequentially, plus Slurm job scripts (`tp_dnn/scripts/slurm/`) for a GPU cluster.

</details>

---

## 📊 Results (from included logs)

From a logged run in `tp_dnn/outputs/logs/mnist_compare_*/metrics.json` — architecture `784→500→500→10`, 100 RBM epochs, 200 DNN epochs, lr=0.1, batch size 128:

| Initialization    | Train error | Test error |
|--------------------|:-----------:|:----------:|
| DBN-pretrained      | 0.00065     | **0.0180** |
| Random init         | 0.00025     | 0.0264     |

Pretraining wins on generalization: in this run, DBN/RBM pretraining produced a lower test error than random initialization at the same architecture and training budget — the classic "greedy layer-wise pretraining helps generalization" result this lab is built to demonstrate. (Random init actually edges out on train error, which tracks — pretraining acts as a regularizer, not a memorization trick.)

Reconstruction-error logs for the Binary AlphaDigits RBM/DBN runs, generated sample images, and sweep results (`results.csv`) for the `fig1`/`fig2`/`fig3` experiments are also saved under `tp_dnn/outputs/`.

A written report (`tp_dnn/report/noms-TP-DNN.pdf`) covering methodology and results is included in the repo.

---

## 🏗️ Repository structure

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

---

## ⚙️ Setup

**Prerequisites:** Python 3.9+, PyTorch + torchvision, numpy, scipy, matplotlib, tqdm.

Declared in [`tp_dnn/requirements.txt`](tp_dnn/requirements.txt):

```
torch
torchvision
numpy
matplotlib
scipy
tqdm
```

**Install:**

```bash
git clone https://github.com/destivano/RBM.git
cd RBM/tp_dnn
pip install -r requirements.txt
```

<details>
<summary><b>📁 Getting the data</b></summary>

The datasets are not tracked in the repository and must be provided locally:

- **MNIST** — place the standard `MNIST/` folder (as produced by `torchvision.datasets.MNIST`) under `tp_dnn/data/raw/mnist/`. The loader uses `download=False`, so it expects the data to already be present there.
- **Binary AlphaDigits** — place `binaryalphadigs.mat` under `tp_dnn/data/raw/binary_alpha_digits/`.

</details>

---

## 🚀 Usage

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

<details>
<summary><b>🖥️ Slurm (GPU cluster)</b></summary>

```bash
bash scripts/slurm/submit_all.sh
```

</details>

Each experiment writes logs, metrics (`metrics.json`), CSV results, and figures/sample images to a timestamped directory under `tp_dnn/outputs/`.

---

## 📝 Notes

- Default hyperparameters (epochs, learning rate, batch size, layer widths, experiment sweep grids, random seed) are centralized in `tp_dnn/src/config.py`.
- `RBM.generate` and `DBN.generate` (defined directly on the model classes) are unimplemented stubs (`raise NotImplementedError`) in this snapshot of the code. Sample generation for the `alpha_rbm`/`alpha_dbn` experiments is instead performed via free-running Gibbs sampling helper functions (`sample_rbm` in `tp_dnn/src/training/rbm_trainer.py`, and the equivalent in `dbn_trainer.py`), which start from random binary visible vectors and alternate v→h→v for a fixed number of steps. Generated sample images from these runs are saved under `tp_dnn/outputs/samples/`.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
