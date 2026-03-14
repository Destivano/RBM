"""
rbm_trainer.py – CD-1 training loop for a single RBM and Gibbs sampling.
"""

import torch
from tqdm import trange


def train_rbm(rbm, train_loader, epochs, lr, device):
    """Train an RBM with Contrastive Divergence (CD-1).

    Parameters
    ----------
    rbm : RBM
        The RBM instance (nn.Module) to train.
    train_loader : DataLoader
        Yields batches; each element is either a tensor or a (tensor, ...) tuple.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate for the CD-1 update.
    device : torch.device
        Device to run on.

    Returns
    -------
    list[float]
        Mean reconstruction MSE per epoch.
    """
    rbm.to(device)
    epoch_errors = []

    for epoch in trange(epochs, desc="RBM training"):
        mse_sum = 0.0
        n_samples = 0

        for batch in train_loader:
            # Handle DataLoader yielding (data,) or (data, labels)
            v0 = batch[0] if isinstance(batch, (list, tuple)) else batch
            v0 = v0.to(device)
            bs = v0.size(0)

            # --- CD-1 forward pass (no autograd needed) ---
            with torch.no_grad():
                stats = rbm.cd1_step(v0)

                v1 = stats["v1"]
                h0_prob = stats["h0_prob"]
                h1_prob = stats["h1_prob"]

                # Parameter updates
                # W += lr * (v0^T h0_prob - v1^T h1_prob) / batch_size
                rbm.W.add_((v0.t() @ h0_prob - v1.t() @ h1_prob) * (lr / bs))
                # a += lr * mean(v0 - v1)
                rbm.a.add_((v0 - v1).mean(dim=0) * lr)
                # b += lr * mean(h0_prob - h1_prob)
                rbm.b.add_((h0_prob - h1_prob).mean(dim=0) * lr)

                # Reconstruction MSE
                mse_sum += ((v0 - stats["v1_prob"]) ** 2).sum().item()
                n_samples += bs

        epoch_errors.append(mse_sum / n_samples)

    return epoch_errors


def sample_rbm(rbm, gibbs_steps, n_samples, device):
    """Generate samples from a trained RBM via Gibbs sampling.

    Starts from random visible vectors and alternates
    v → h → v for *gibbs_steps* iterations.

    Parameters
    ----------
    rbm : RBM
        Trained RBM.
    gibbs_steps : int
        Number of Gibbs sampling iterations.
    n_samples : int
        Number of samples to generate.
    device : torch.device
        Device to run on.

    Returns
    -------
    torch.Tensor
        Generated visible samples, shape (n_samples, n_visible).
    """
    rbm.to(device)

    with torch.no_grad():
        v = (torch.rand(n_samples, rbm.n_visible, device=device) > 0.5).float()

        for _ in range(gibbs_steps):
            h_prob = rbm.v_to_h_prob(v)
            h = rbm.sample_bernoulli(h_prob)
            v_prob = rbm.h_to_v_prob(h)
            v = rbm.sample_bernoulli(v_prob)

    return v
