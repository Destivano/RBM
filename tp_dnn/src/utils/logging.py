"""
logging.py – Simple logging utilities for training runs.

Provides helpers to create run directories, save metrics (JSON),
and save sweep results (CSV) under outputs/logs/.
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path

from src.config import LOG_DIR


def create_run_directory(experiment_name: str) -> Path:
    """Create a timestamped run directory under outputs/logs/.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.

    Returns
    -------
    Path
        Path to the created directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOG_DIR / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_metrics(metrics_dict: dict, path) -> None:
    """Save a metrics dictionary as JSON.

    Parameters
    ----------
    metrics_dict : dict
        Metric name → value pairs.
    path : str or Path
        Destination file path (e.g. run_dir / "metrics.json").
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics_dict, f, indent=2)


def save_csv(results, path) -> None:
    """Save experiment sweep results to CSV.

    Parameters
    ----------
    results : dict or list[dict]
        If dict, keys are column names, values are lists of equal length.
        If list of dicts, each dict is one row.
    path : str or Path
        Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(results, dict):
        keys = list(results.keys())
        rows = [dict(zip(keys, vals)) for vals in zip(*results.values())]
    else:
        rows = results
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
