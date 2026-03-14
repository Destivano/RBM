"""
alphadigits.py – Binary AlphaDigits dataset loading.

Loads the Binary AlphaDigits .mat file and returns selected characters
as a float32 tensor with values in {0, 1}.
"""

import torch
import numpy as np
import scipy.io as sio

from src.config import DATA_RAW


def load_alphadigits(mat_path=None, chars=None):
    """Load Binary AlphaDigits dataset from a .mat file.

    Parameters
    ----------
    mat_path : str or Path, optional
        Full path to ``binaryalphadigs.mat``.
        Defaults to ``data/raw/binary_alpha_digits/binaryalphadigs.mat``.
    chars : list[int] or None
        List of character indices to extract (0-35).
        If None, load all 36 characters.

    Returns
    -------
    torch.Tensor
        Image data of shape (N, D) with D = 20 * 16 = 320,
        dtype float32, values in {0.0, 1.0}.
    """
    if mat_path is None:
        mat_path = str(DATA_RAW / "binary_alpha_digits" / "binaryalphadigs.mat")

    mat = sio.loadmat(mat_path)
    # The .mat file stores a (36, 39) object array of (20, 16) binary images.
    # 36 characters (0-9 then A-Z), 39 examples each.
    dat = mat["dat"]  # shape (36, 39), each cell is (20, 16)

    if chars is None:
        chars = list(range(dat.shape[0]))

    images = []
    for c in chars:
        for j in range(dat.shape[1]):
            img = np.array(dat[c, j], dtype=np.float32).flatten()
            images.append(img)

    X = torch.tensor(np.stack(images), dtype=torch.float32)
    return X
