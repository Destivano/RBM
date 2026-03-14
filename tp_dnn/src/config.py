"""
config.py – Default hyperparameters and experiment grids.

All constants used across the project are centralised here so that
experiments and training loops can import them directly.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Paths (relative to project root tp_dnn/)
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
FIG_DIR = OUTPUT_DIR / "figures"
SAMPLE_DIR = OUTPUT_DIR / "samples"

# ──────────────────────────────────────────────
# Training hyperparameters (defaults)
# ──────────────────────────────────────────────
RBM_EPOCHS = 100
DBN_EPOCHS = 100
DNN_EPOCHS = 200
BATCH_SIZE = 128
LR = 0.1

# ──────────────────────────────────────────────
# Model defaults
# ──────────────────────────────────────────────
VISIBLE_SIZE = 784          # 28×28 MNIST images
HIDDEN_SIZE = 500           # default hidden layer width
NUM_CLASSES = 10            # MNIST digits 0-9

# ──────────────────────────────────────────────
# Experiment grids
# ──────────────────────────────────────────────
FIG1_LAYERS = [2, 3, 4, 5]
FIG2_WIDTHS = [100, 300, 700]
FIG3_TRAIN_SIZES = [1000, 3000, 7000, 10000, 30000, 60000]

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
SEED = 42
