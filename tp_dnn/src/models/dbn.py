"""
dbn.py – Deep Belief Network.

A DBN is a stack of RBMs trained greedily layer-by-layer.
After pretraining, the learned weights can initialise a DNN.
"""

from src.models.rbm import RBM


class DBN:
    """Deep Belief Network built from stacked RBMs.

    Parameters
    ----------
    layer_sizes : list[int]
        Sizes of each layer including the visible layer.
        e.g. [784, 500, 500] → two RBMs: (784→500) and (500→500).
    """

    def __init__(self, layer_sizes: list):
        self.layer_sizes = layer_sizes
        self.rbms = []
        for i in range(len(layer_sizes) - 1):
            self.rbms.append(RBM(layer_sizes[i], layer_sizes[i + 1]))

    def generate(self, n_gibbs: int = 1000, n_samples: int = 1):
        """Generate samples from the top-level RBM and propagate down."""
        raise NotImplementedError("Implemented in a later prompt")
