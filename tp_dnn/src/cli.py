"""
cli.py – Command-line interface for all experiments.

Usage:
    python -m src.cli <experiment>

Subcommands:
    alphadigits_rbm   – RBM on Binary AlphaDigits
    alphadigits_dbn   – DBN on Binary AlphaDigits
    mnist_compare     – Pretrained vs random DNN on MNIST
    fig1_layers       – Error vs number of hidden layers
    fig2_width        – Error vs neurons per layer
    fig3_datasize     – Error vs training-set size
"""

import argparse
import sys

from src.experiments import (
    alpha_rbm,
    alpha_dbn,
    mnist_compare,
    fig1_layers,
    fig2_width,
    fig3_datasize,
)


EXPERIMENTS = {
    "alphadigits_rbm": alpha_rbm.run,
    "alphadigits_dbn": alpha_dbn.run,
    "mnist_compare": mnist_compare.run,
    "fig1_layers": fig1_layers.run,
    "fig2_width": fig2_width.run,
    "fig3_datasize": fig3_datasize.run,
}


def main():
    parser = argparse.ArgumentParser(
        prog="src.cli",
        description="TP DNN – run deep-learning experiments.",
    )
    parser.add_argument(
        "experiment",
        choices=EXPERIMENTS.keys(),
        help="Name of the experiment to run.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed (default: config.SEED).",
    )
    args = parser.parse_args()

    print(f"=== Running experiment: {args.experiment} ===", flush=True)
    EXPERIMENTS[args.experiment](seed=args.seed)
    print(f"=== Finished: {args.experiment} ===", flush=True)


if __name__ == "__main__":
    main()
