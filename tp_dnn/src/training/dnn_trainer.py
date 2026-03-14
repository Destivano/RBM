"""
dnn_trainer.py – Supervised training loop for a DNN classifier.

Uses CrossEntropyLoss + SGD.
"""

import torch
import torch.nn as nn
from tqdm import trange


def train_dnn(model, train_loader, epochs, lr, device):
    """Train a DNN classifier with SGD + cross-entropy.

    Parameters
    ----------
    model : DNN (nn.Module)
        The network to train.
    train_loader : DataLoader
        Yields (data, labels) batches.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate for SGD.
    device : torch.device
        Device to run on.

    Returns
    -------
    list[float]
        Mean training loss per epoch.
    """
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    epoch_losses = []

    for epoch in trange(epochs, desc="DNN training"):
        loss_sum = 0.0
        n_samples = 0

        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            bs = x.size(0)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * bs
            n_samples += bs

        epoch_losses.append(loss_sum / n_samples)

    return epoch_losses
