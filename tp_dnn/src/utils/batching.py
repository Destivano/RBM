"""
batching.py – Mini-batch iteration utilities.

Provides helpers to wrap plain tensors into PyTorch DataLoaders
and to iterate over tensors in mini-batches.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset


def create_tensor_dataloader(X: torch.Tensor, batch_size: int,
                             shuffle: bool = True) -> DataLoader:
    """Wrap a tensor in a TensorDataset and return a DataLoader.

    Parameters
    ----------
    X : torch.Tensor
        Data tensor of shape (N, D).
    batch_size : int
        Mini-batch size.
    shuffle : bool
        Whether to shuffle each epoch.

    Returns
    -------
    DataLoader
    """
    dataset = TensorDataset(X)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def iter_batches(data: torch.Tensor, batch_size: int, shuffle: bool = True):
    """Yield mini-batches from a tensor.

    Parameters
    ----------
    data : torch.Tensor
        Dataset tensor of shape (N, ...).
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        Whether to shuffle indices each pass.

    Yields
    ------
    torch.Tensor
        A batch of shape (<=batch_size, ...).
    """
    raise NotImplementedError("Implemented in a later prompt")


def iter_batches_with_labels(data: torch.Tensor, labels: torch.Tensor,
                             batch_size: int, shuffle: bool = True):
    """Yield (batch_data, batch_labels) mini-batches.

    Parameters
    ----------
    data : torch.Tensor
        Dataset tensor of shape (N, D).
    labels : torch.Tensor
        Labels tensor of shape (N,).
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        Whether to shuffle indices each pass.

    Yields
    ------
    tuple[torch.Tensor, torch.Tensor]
        (batch_data, batch_labels).
    """
    raise NotImplementedError("Implemented in a later prompt")
