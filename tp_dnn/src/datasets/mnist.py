"""
mnist.py – MNIST dataset loading and binarization.

Loads MNIST from data/raw/mnist/ (already downloaded),
binarizes pixels, flattens to 784-d vectors, and returns DataLoaders.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as T
import numpy as np

from src.config import DATA_RAW


def _binarize_transform():
    """Return a torchvision transform that converts to tensor, binarizes, and flattens."""
    return T.Compose([
        T.ToTensor(),                          # [0,1] float, shape (1,28,28)
        T.Lambda(lambda x: (x >= 0.5).float()),  # 127/255 ≈ 0.498 → threshold at 0.5
        T.Lambda(lambda x: x.view(-1)),        # flatten to 784
    ])


def load_mnist_binarized(
    data_dir=None,
    batch_size=128,
    num_workers=2,
    train_subset=None,
    seed=0,
):
    """Load binarized MNIST and return train / test DataLoaders.

    Parameters
    ----------
    data_dir : str or Path, optional
        Root directory containing the ``MNIST/`` folder.
        Defaults to ``data/raw/mnist``.
    batch_size : int
        Mini-batch size.
    num_workers : int
        Number of DataLoader workers.
    train_subset : int or None
        If not None, randomly sample this many training examples.
    seed : int
        Random seed used when sub-sampling training data.

    Returns
    -------
    train_loader : DataLoader
    test_loader  : DataLoader
    """
    if data_dir is None:
        data_dir = str(DATA_RAW / "mnist")

    transform = _binarize_transform()

    train_set = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=False, transform=transform,
    )
    test_set = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=False, transform=transform,
    )

    # Optional subset of training data
    if train_subset is not None:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(train_set), size=train_subset, replace=False)
        train_set = Subset(train_set, indices.tolist())

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    return train_loader, test_loader
