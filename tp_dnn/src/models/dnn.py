"""
dnn.py – Deep Neural Network classifier (MLP).

Sigmoid hidden layers + linear output (logits).
Softmax is handled by the loss function (nn.CrossEntropyLoss).
Supports optional weight initialisation from a pretrained DBN.
"""

import torch
import torch.nn as nn


class DNN(nn.Module):
    """Feed-forward neural network for classification.

    Parameters
    ----------
    layer_sizes : list[int]
        Sizes of each layer including input and output.
        e.g. [784, 500, 500, 10].
    """

    def __init__(self, layer_sizes: list):
        super().__init__()
        self.layer_sizes = layer_sizes

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: sigmoid on hidden layers, raw logits on output.

        Parameters
        ----------
        x : torch.Tensor
            Input data, shape (batch, input_size).

        Returns
        -------
        torch.Tensor
            Logits, shape (batch, num_classes).
        """
        for layer in self.layers[:-1]:
            x = torch.sigmoid(layer(x))
        x = self.layers[-1](x)  # output logits (no activation)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class labels.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Predicted class indices, shape (batch,).
        """
        logits = self.forward(x)
        return logits.argmax(dim=1)

    @classmethod
    def from_dbn(cls, dbn, num_classes: int = 10):
        """Create a DNN initialised with weights from a pretrained DBN.

        The hidden layers take W and biases from the DBN's RBMs.
        The final classification layer is randomly initialised.

        Parameters
        ----------
        dbn : DBN
            Pretrained Deep Belief Network.
        num_classes : int
            Number of output classes.

        Returns
        -------
        DNN
        """
        layer_sizes = dbn.layer_sizes + [num_classes]
        dnn = cls(layer_sizes)

        # Copy RBM weights into hidden layers
        with torch.no_grad():
            for i, rbm in enumerate(dbn.rbms):
                dnn.layers[i].weight.copy_(rbm.W.t())
                dnn.layers[i].bias.copy_(rbm.b)

        return dnn
