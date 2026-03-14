"""
alpha_dbn.py – DBN experiment on Binary AlphaDigits.

Pretrain a DBN on selected characters, visualise generated samples.
"""

import torch

from src.config import (
    DBN_EPOCHS, LR, BATCH_SIZE, SEED, SAMPLE_DIR,
)
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.viz import save_image_grid, plot_training_curve
from src.utils.logging import create_run_directory, save_metrics
from src.datasets.alphadigits import load_alphadigits
from src.training.dbn_trainer import pretrain_dbn
from src.training.rbm_trainer import sample_rbm


def run(seed=None):
    """Run the AlphaDigits DBN experiment."""
    seed = seed if seed is not None else SEED
    set_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    run_dir = create_run_directory("alphadigits_dbn")

    # Load data (first 10 characters = digits 0-9)
    X = load_alphadigits(chars=list(range(10)))
    n_visible = X.shape[1]  # 320
    print(f"AlphaDigits data: {X.shape}")

    # Pretrain DBN: 320 → 200 → 100
    layer_sizes = [n_visible, 200, 100]
    dbn, all_errors = pretrain_dbn(
        X, layer_sizes,
        epochs=DBN_EPOCHS, lr=LR, batch_size=BATCH_SIZE, device=device,
    )

    # Save training curves per layer
    for i, errors in enumerate(all_errors):
        plot_training_curve(
            errors, ylabel="Reconstruction MSE",
            title=f"DBN AlphaDigits – RBM layer {i + 1}",
            save_path=str(run_dir / f"alpha_dbn_layer{i + 1}_error.png"),
        )

    # Generate samples from the top RBM and propagate down
    top_rbm = dbn.rbms[-1]
    top_samples = sample_rbm(top_rbm, gibbs_steps=1000, n_samples=20, device=device)

    # Propagate down through lower RBMs
    with torch.no_grad():
        v = top_samples.to(device)
        for rbm in reversed(dbn.rbms[:-1]):
            rbm.to(device)
            v = rbm.h_to_v_prob(v)

    save_image_grid(
        v.cpu(), height=20, width=16,
        out_path=str(SAMPLE_DIR / "alpha_dbn_samples.png"), nrow=10,
    )

    # Save original data grid for comparison
    save_image_grid(
        X[:20], height=20, width=16,
        out_path=str(SAMPLE_DIR / "alpha_dbn_data.png"), nrow=10,
    )

    # Save metrics
    final_errors = [errs[-1] for errs in all_errors]
    metrics = {
        "experiment": "alphadigits_dbn",
        "seed": seed,
        "architecture": {"layer_sizes": layer_sizes},
        "hyperparameters": {
            "epochs": DBN_EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
        },
        "final_reconstruction_mse_per_layer": final_errors,
    }
    save_metrics(metrics, run_dir / "metrics.json")

    print(f"Outputs saved to {run_dir}")


if __name__ == "__main__":
    run()
