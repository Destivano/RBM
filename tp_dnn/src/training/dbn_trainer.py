"""
dbn_trainer.py – Greedy layer-wise pretraining for a DBN.

Pretrains each RBM in the stack sequentially. After each RBM is trained,
the training data is transformed through it to obtain input for the next layer.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training.rbm_trainer import train_rbm


def pretrain_dbn(X, layer_sizes, epochs, lr, batch_size, device):
    """Greedy layer-wise pretraining that builds and trains a DBN.

    Parameters
    ----------
    X : torch.Tensor
        Training data, shape (N, visible_size).
    layer_sizes : list[int]
        Layer sizes including visible, e.g. [784, 500, 500].
    epochs : int
        Number of CD-1 epochs per RBM.
    lr : float
        Learning rate.
    batch_size : int
        Mini-batch size.
    device : torch.device
        Device to run on.

    Returns
    -------
    dbn : DBN
        DBN with trained RBMs.
    all_errors : list[list[float]]
        Reconstruction errors per epoch, per layer.
    """
    from src.models.dbn import DBN

    dbn = DBN(layer_sizes)
    all_errors = []
    current_data = X.clone()

    for idx, rbm in enumerate(dbn.rbms):
        print(f"\n>>> Pretraining RBM layer {idx + 1}/{len(dbn.rbms)}  "
              f"({rbm.n_visible} -> {rbm.n_hidden})")

        loader = DataLoader(
            TensorDataset(current_data),
            batch_size=batch_size,
            shuffle=True,
        )

        errors = train_rbm(rbm, loader, epochs=epochs, lr=lr, device=device)
        all_errors.append(errors)

        # Transform data through trained RBM for next layer
        rbm.to(device)
        with torch.no_grad():
            current_data = rbm.v_to_h_prob(current_data.to(device)).cpu()

    return dbn, all_errors
