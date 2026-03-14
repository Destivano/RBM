"""
fig1_layers.py – Figure 1: Classification error vs number of hidden layers.

Sweeps over different network depths (config.FIG1_LAYERS) and compares
pretrained vs random initialisation.
"""

import torch

from src.config import (
    FIG1_LAYERS, RBM_EPOCHS, DNN_EPOCHS, LR, BATCH_SIZE, SEED,
    VISIBLE_SIZE, HIDDEN_SIZE, NUM_CLASSES, FIG_DIR,
)
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.viz import plot_error_curve
from src.utils.logging import create_run_directory, save_metrics, save_csv
from src.datasets.mnist import load_mnist_binarized
from src.models.dnn import DNN
from src.training.dbn_trainer import pretrain_dbn
from src.training.dnn_trainer import train_dnn
from src.training.eval import evaluate_model


def run(seed=None):
    """Run the layers sweep experiment and produce Figure 1."""
    seed = seed if seed is not None else SEED
    device = get_device()
    print(f"Device: {device}")

    run_dir = create_run_directory("fig1_layers")

    train_loader, test_loader = load_mnist_binarized(
        batch_size=BATCH_SIZE, num_workers=2,
    )

    # Collect training data for DBN pretraining
    all_x = []
    for batch in train_loader:
        all_x.append(batch[0])
    X_train = torch.cat(all_x, dim=0)

    pre_errors = []
    rand_errors = []

    for n_layers in FIG1_LAYERS:
        print(f"\n{'='*50}")
        print(f"  Hidden layers: {n_layers}")
        print(f"{'='*50}")

        set_seed(seed)
        layer_sizes = [VISIBLE_SIZE] + [HIDDEN_SIZE] * n_layers

        # Pretrained
        dbn, _ = pretrain_dbn(
            X_train, layer_sizes,
            epochs=RBM_EPOCHS, lr=LR, batch_size=BATCH_SIZE, device=device,
        )
        dnn_pre = DNN.from_dbn(dbn, num_classes=NUM_CLASSES)
        train_dnn(dnn_pre, train_loader, epochs=DNN_EPOCHS, lr=LR, device=device)
        err_pre = evaluate_model(dnn_pre, test_loader, device)

        # Random
        set_seed(seed)
        dnn_rand = DNN(layer_sizes + [NUM_CLASSES])
        train_dnn(dnn_rand, train_loader, epochs=DNN_EPOCHS, lr=LR, device=device)
        err_rand = evaluate_model(dnn_rand, test_loader, device)

        pre_errors.append(err_pre)
        rand_errors.append(err_rand)
        print(f"  Pretrained test error: {err_pre:.4f}")
        print(f"  Random     test error: {err_rand:.4f}")

    # Plot
    plot_error_curve(
        FIG1_LAYERS, pre_errors, rand_errors,
        title="Fig 1 – Error vs number of hidden layers",
        xlabel="Number of hidden layers",
        out_path=str(FIG_DIR / "fig1_layers.png"),
    )

    # Save metrics
    metrics = {
        "experiment": "fig1_layers",
        "seed": seed,
        "architecture": {"hidden_size": HIDDEN_SIZE, "num_classes": NUM_CLASSES},
        "hyperparameters": {
            "rbm_epochs": RBM_EPOCHS, "dnn_epochs": DNN_EPOCHS,
            "lr": LR, "batch_size": BATCH_SIZE,
        },
        "layers": FIG1_LAYERS,
        "pretrained_test_errors": pre_errors,
        "random_test_errors": rand_errors,
    }
    save_metrics(metrics, run_dir / "metrics.json")

    # Save CSV
    save_csv(
        {
            "layers": FIG1_LAYERS,
            "pretrained_test_error": pre_errors,
            "random_test_error": rand_errors,
        },
        run_dir / "results.csv",
    )

    print(f"\nFigure saved to outputs/figures/fig1_layers.png")
    print(f"Outputs saved to {run_dir}")


if __name__ == "__main__":
    run()
