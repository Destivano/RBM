"""
fig2_width.py – Figure 2: Classification error vs number of neurons per layer.

Sweeps over different hidden-layer widths (config.FIG2_WIDTHS) and compares
pretrained vs random initialisation.  Uses a fixed 2-hidden-layer architecture.
"""

import torch

from src.config import (
    FIG2_WIDTHS, RBM_EPOCHS, DNN_EPOCHS, LR, BATCH_SIZE, SEED,
    VISIBLE_SIZE, NUM_CLASSES, FIG_DIR,
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
    """Run the width sweep experiment and produce Figure 2."""
    seed = seed if seed is not None else SEED
    device = get_device()
    print(f"Device: {device}", flush=True)

    run_dir = create_run_directory("fig2_width")

    train_loader, test_loader = load_mnist_binarized(
        batch_size=BATCH_SIZE, num_workers=2,
    )

    all_x = []
    for batch in train_loader:
        all_x.append(batch[0])
    X_train = torch.cat(all_x, dim=0)

    pre_errors = []
    rand_errors = []

    for width in FIG2_WIDTHS:
        print(f"\n{'='*50}", flush=True)
        print(f"  Hidden width: {width}", flush=True)
        print(f"{'='*50}", flush=True)

        set_seed(seed)
        layer_sizes = [VISIBLE_SIZE, width, width]

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
        print(f"  Pretrained test error: {err_pre:.4f}", flush=True)
        print(f"  Random     test error: {err_rand:.4f}", flush=True)

    # Plot
    plot_error_curve(
        FIG2_WIDTHS, pre_errors, rand_errors,
        title="Fig 2 – Error vs hidden layer width",
        xlabel="Neurons per hidden layer",
        out_path=str(FIG_DIR / "fig2_width.png"),
    )

    # Save metrics
    metrics = {
        "experiment": "fig2_width",
        "seed": seed,
        "architecture": {"n_hidden_layers": 2, "num_classes": NUM_CLASSES},
        "hyperparameters": {
            "rbm_epochs": RBM_EPOCHS, "dnn_epochs": DNN_EPOCHS,
            "lr": LR, "batch_size": BATCH_SIZE,
        },
        "widths": FIG2_WIDTHS,
        "pretrained_test_errors": pre_errors,
        "random_test_errors": rand_errors,
    }
    save_metrics(metrics, run_dir / "metrics.json")

    # Save CSV
    save_csv(
        {
            "width": FIG2_WIDTHS,
            "pretrained_test_error": pre_errors,
            "random_test_error": rand_errors,
        },
        run_dir / "results.csv",
    )

    print(f"\nFigure saved to outputs/figures/fig2_width.png", flush=True)
    print(f"Outputs saved to {run_dir}", flush=True)


if __name__ == "__main__":
    run()
