"""
eval.py – Evaluation utilities.
"""

import torch


def evaluate_model(model, loader, device):
    """Compute classification error rate over a DataLoader.

    Parameters
    ----------
    model : nn.Module
        Network with a standard forward returning logits.
    loader : DataLoader
        Yields (data, labels) batches.
    device : torch.device
        Device to run on.

    Returns
    -------
    float
        Error rate in [0, 1].
    """
    model.to(device)
    model.eval()

    wrong = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            preds = model(x).argmax(dim=1)
            wrong += (preds != y).sum().item()
            total += y.size(0)

    return wrong / total if total > 0 else 0.0
