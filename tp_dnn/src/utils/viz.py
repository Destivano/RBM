"""
viz.py – Visualization helpers.

Contains functions to save image grids, plot training curves,
and comparison figures.
"""

import os
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for cluster
import matplotlib.pyplot as plt


def save_image_grid(X_flat, height, width, out_path, nrow=10):
    """Reshape flat vectors into images and save as a PNG grid.

    Parameters
    ----------
    X_flat : torch.Tensor or np.ndarray
        Batch of flattened images, shape (N, D).
    height : int
        Image height in pixels.
    width : int
        Image width in pixels.
    out_path : str or Path
        Destination file path for the saved PNG.
    nrow : int
        Number of images per row in the grid.
    """
    if isinstance(X_flat, torch.Tensor):
        X_flat = X_flat.detach().cpu().numpy()

    N = X_flat.shape[0]
    ncol = nrow
    nrow_grid = math.ceil(N / ncol)

    fig, axes = plt.subplots(nrow_grid, ncol,
                             figsize=(ncol * 1.2, nrow_grid * 1.2))
    axes = np.atleast_2d(axes)

    for i in range(nrow_grid):
        for j in range(ncol):
            idx = i * ncol + j
            ax = axes[i, j]
            ax.axis("off")
            if idx < N:
                img = X_flat[idx].reshape(height, width)
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)

    plt.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".",
                exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_images(images, title: str = "", save_path: str = None):
    """Display a grid of images.

    Parameters
    ----------
    images : array-like
        Batch of images (N, H, W) or (N, H*W).
    title : str
        Figure title.
    save_path : str, optional
        If provided, save figure to this path.
    """
    raise NotImplementedError("Implemented in a later prompt")


def plot_training_curve(values, ylabel: str, title: str = "", save_path: str = None):
    """Plot a single metric over epochs and save to file.

    Parameters
    ----------
    values : list[float]
        Metric value per epoch.
    ylabel : str
        Y-axis label.
    title : str
        Figure title.
    save_path : str, optional
        If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(values) + 1), values)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
                    exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(results: dict, xlabel: str, ylabel: str,
                    title: str = "", save_path: str = None):
    """Plot comparison curves (e.g., pretrained vs random init).

    Parameters
    ----------
    results : dict
        label -> (list_of_x, list_of_y).
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    title : str
        Figure title.
    save_path : str, optional
        If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, (xs, ys) in results.items():
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
                    exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_curve(x_values, pretrained_errors, random_errors, title, xlabel, out_path):
    """Plot pretrained vs random error curves for comparison figures.

    Parameters
    ----------
    x_values : list
        X-axis values (e.g. number of layers, width, or train size).
    pretrained_errors : list[float]
        Test error rates for pretrained network.
    random_errors : list[float]
        Test error rates for randomly initialised network.
    title : str
        Figure title.
    xlabel : str
        X-axis label.
    out_path : str or Path
        Destination file path for the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_values, pretrained_errors, marker="o", label="Pretrained (DBN)")
    ax.plot(x_values, random_errors, marker="s", label="Random init")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test error rate")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".",
                exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
