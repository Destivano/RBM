"""
mnist_compare.py – Compare pretrained (DBN→DNN) vs randomly initialised DNN on MNIST.

Steps:
1. Load binarized MNIST
2. Pretrain a DBN stack
3. Create DNN_pretrained from DBN, and DNN_random with same architecture
4. Train both with cross-entropy
5. Evaluate and log results
"""

import torch

from src.config import (
    RBM_EPOCHS, DNN_EPOCHS, LR, BATCH_SIZE, SEED,
    VISIBLE_SIZE, HIDDEN_SIZE, NUM_CLASSES, FIG_DIR,
)
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.viz import plot_training_curve
from src.utils.logging import create_run_directory, save_metrics, save_csv
from src.datasets.mnist import load_mnist_binarized
from src.models.dnn import DNN
from src.training.dbn_trainer import pretrain_dbn
from src.training.dnn_trainer import train_dnn
from src.training.eval import evaluate_model


def run(seed=None):
    """Run the MNIST pretrained vs random comparison experiment."""
    seed = seed if seed is not None else SEED
    set_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    run_dir = create_run_directory("mnist_compare")

    # Load MNIST
    train_loader, test_loader = load_mnist_binarized(
        batch_size=BATCH_SIZE, num_workers=2,
    )

    # Collect flattened training data for DBN pretraining
    all_x = []
    for batch in train_loader:
        all_x.append(batch[0])
    X_train = torch.cat(all_x, dim=0)

    # Architecture
    layer_sizes = [VISIBLE_SIZE, HIDDEN_SIZE, HIDDEN_SIZE]

    # --- Pretrained path ---
    print("\n=== DBN Pretraining ===")
    dbn, rbm_errors = pretrain_dbn(
        X_train, layer_sizes,
        epochs=RBM_EPOCHS, lr=LR, batch_size=BATCH_SIZE, device=device,
    )
    dnn_pre = DNN.from_dbn(dbn, num_classes=NUM_CLASSES)

    # --- Random init path ---
    dnn_rand = DNN(layer_sizes + [NUM_CLASSES])

    # --- Supervised fine-tuning ---
    print("\n=== Training pretrained DNN ===")
    set_seed(seed)
    losses_pre = train_dnn(dnn_pre, train_loader, epochs=DNN_EPOCHS, lr=LR, device=device)

    print("\n=== Training random DNN ===")
    set_seed(seed)
    losses_rand = train_dnn(dnn_rand, train_loader, epochs=DNN_EPOCHS, lr=LR, device=device)

    # --- Evaluate ---
    err_pre_train = evaluate_model(dnn_pre, train_loader, device)
    err_pre_test = evaluate_model(dnn_pre, test_loader, device)
    err_rand_train = evaluate_model(dnn_rand, train_loader, device)
    err_rand_test = evaluate_model(dnn_rand, test_loader, device)

    print(f"\nPretrained  – train error: {err_pre_train:.4f}, test error: {err_pre_test:.4f}")
    print(f"Random init – train error: {err_rand_train:.4f}, test error: {err_rand_test:.4f}")

    # Save training curves
    plot_training_curve(
        losses_pre, ylabel="Cross-entropy loss",
        title="Pretrained DNN – training loss",
        save_path=str(run_dir / "mnist_pretrained_loss.png"),
    )
    plot_training_curve(
        losses_rand, ylabel="Cross-entropy loss",
        title="Random DNN – training loss",
        save_path=str(run_dir / "mnist_random_loss.png"),
    )

    # Save metrics
    metrics = {
        "experiment": "mnist_compare",
        "seed": seed,
        "architecture": {"layer_sizes": layer_sizes, "num_classes": NUM_CLASSES},
        "hyperparameters": {
            "rbm_epochs": RBM_EPOCHS, "dnn_epochs": DNN_EPOCHS,
            "lr": LR, "batch_size": BATCH_SIZE,
        },
        "pretrained": {"train_error": err_pre_train, "test_error": err_pre_test},
        "random": {"train_error": err_rand_train, "test_error": err_rand_test},
    }
    save_metrics(metrics, run_dir / "metrics.json")

    # Save CSV
    save_csv(
        {
            "model": ["pretrained", "random"],
            "train_error": [err_pre_train, err_rand_train],
            "test_error": [err_pre_test, err_rand_test],
        },
        run_dir / "results.csv",
    )

    print(f"Outputs saved to {run_dir}")


if __name__ == "__main__":
    run()


if __name__ == "__main__":
    run()
