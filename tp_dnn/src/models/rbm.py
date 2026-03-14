"""
rbm.py – Restricted Boltzmann Machine (Bernoulli-Bernoulli).

Implements an RBM as an nn.Module with:
- v_to_h_prob / h_to_v_prob  (deterministic probability maps)
- sample_bernoulli            (stochastic activation)
- cd1_step                    (one CD-1 pass, returns statistics only)
"""

import torch
import torch.nn as nn


class RBM(nn.Module):
    """Bernoulli-Bernoulli Restricted Boltzmann Machine.

    Parameters
    ----------
    n_visible : int
        Number of visible units.
    n_hidden : int
        Number of hidden units.
    """

    def __init__(self, n_visible: int, n_hidden: int):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # Weight matrix W ~ N(0, 0.01)
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        # Visible bias
        self.a = nn.Parameter(torch.zeros(n_visible))
        # Hidden bias
        self.b = nn.Parameter(torch.zeros(n_hidden))

    # ── probability maps ──────────────────────────────

    def v_to_h_prob(self, v: torch.Tensor) -> torch.Tensor:
        """Compute P(h=1 | v) = sigmoid(v @ W + b).

        Parameters
        ----------
        v : torch.Tensor
            Visible activations, shape (batch, n_visible).

        Returns
        -------
        torch.Tensor
            Hidden probabilities, shape (batch, n_hidden).
        """
        return torch.sigmoid(v @ self.W + self.b)

    def h_to_v_prob(self, h: torch.Tensor) -> torch.Tensor:
        """Compute P(v=1 | h) = sigmoid(h @ W^T + a).

        Parameters
        ----------
        h : torch.Tensor
            Hidden activations, shape (batch, n_hidden).

        Returns
        -------
        torch.Tensor
            Visible probabilities, shape (batch, n_visible).
        """
        return torch.sigmoid(h @ self.W.t() + self.a)

    # ── sampling ──────────────────────────────────────

    @staticmethod
    def sample_bernoulli(p: torch.Tensor) -> torch.Tensor:
        """Draw Bernoulli samples from probabilities.

        Parameters
        ----------
        p : torch.Tensor
            Probabilities in [0, 1].

        Returns
        -------
        torch.Tensor
            Binary samples with same shape as p.
        """
        return (torch.rand_like(p) < p).float()

    # ── CD-1 step (statistics only) ──────────────────

    def cd1_step(self, v0: torch.Tensor):
        """Run one CD-1 pass and return phase statistics.

        Does NOT update parameters — that is the trainer's job.

        Parameters
        ----------
        v0 : torch.Tensor
            Mini-batch of visible data, shape (batch, n_visible).

        Returns
        -------
        dict
            Keys:
            - "v0"      : original visible input
            - "h0_prob" : P(h|v0)
            - "h0"      : sampled hidden from h0_prob
            - "v1_prob" : P(v|h0)  (reconstruction probabilities)
            - "v1"      : sampled visible from v1_prob
            - "h1_prob" : P(h|v1)
        """
        # Positive phase
        h0_prob = self.v_to_h_prob(v0)
        h0 = self.sample_bernoulli(h0_prob)

        # Negative phase (one Gibbs step)
        v1_prob = self.h_to_v_prob(h0)
        v1 = self.sample_bernoulli(v1_prob)
        h1_prob = self.v_to_h_prob(v1)

        return {
            "v0": v0,
            "h0_prob": h0_prob,
            "h0": h0,
            "v1_prob": v1_prob,
            "v1": v1,
            "h1_prob": h1_prob,
        }

    # ── stubs for later prompts ──────────────────────

    def generate(self, n_gibbs: int = 1000, n_samples: int = 1):
        """Generate samples via Gibbs sampling."""
        raise NotImplementedError("Implemented in a later prompt")
