"""
alpha_rbm.py – RBM experiment on Binary AlphaDigits.

Train an RBM on selected characters, visualise generated samples.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    RBM_EPOCHS, LR, BATCH_SIZE, SEED, SAMPLE_DIR,
)
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.viz import save_image_grid, plot_training_curve
from src.utils.logging import create_run_directory, save_metrics
from src.datasets.alphadigits import load_alphadigits
from src.models.rbm import RBM
from src.training.rbm_trainer import train_rbm, sample_rbm


def run(seed=None):
    """Run the AlphaDigits RBM experiment."""
    seed = seed if seed is not None else SEED
    set_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    run_dir = create_run_directory("alphadigits_rbm")

    # Load data (first 10 characters = digits 0-9)
    X = load_alphadigits(chars=list(range(10)))
    print(f"AlphaDigits data: {X.shape}")

    loader = DataLoader(TensorDataset(X), batch_size=BATCH_SIZE, shuffle=True)

    # Train RBM
    rbm = RBM(n_visible=X.shape[1], n_hidden=200)
    errors = train_rbm(rbm, loader, epochs=RBM_EPOCHS, lr=LR, device=device)

    # Save training curve
    plot_training_curve(
        errors, ylabel="Reconstruction MSE",
        title="RBM on AlphaDigits – reconstruction error",
        save_path=str(run_dir / "alpha_rbm_error.png"),
    )

    # Generate samples
    samples = sample_rbm(rbm, gibbs_steps=1000, n_samples=20, device=device)
    save_image_grid(
        samples, height=20, width=16,
        out_path=str(SAMPLE_DIR / "alpha_rbm_samples.png"), nrow=10,
    )

    # Save original data grid for comparison
    save_image_grid(
        X[:20], height=20, width=16,
        out_path=str(SAMPLE_DIR / "alpha_rbm_data.png"), nrow=10,
    )

    # Save metrics
    metrics = {
        "experiment": "alphadigits_rbm",
        "seed": seed,
        "architecture": {"n_visible": int(X.shape[1]), "n_hidden": 200},
        "hyperparameters": {
            "epochs": RBM_EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
        },
        "final_reconstruction_mse": errors[-1],
    }
    save_metrics(metrics, run_dir / "metrics.json")

    print(f"Final reconstruction MSE: {errors[-1]:.4f}")
    print(f"Outputs saved to {run_dir}")


if __name__ == "__main__":
    run()
