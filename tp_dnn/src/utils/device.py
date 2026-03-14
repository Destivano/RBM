"""
device.py – Device selection helper.

Returns a torch.device pointing to CUDA if available, else CPU.
"""

import torch


def get_device() -> torch.device:
    """Return the best available torch device (CUDA > CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
