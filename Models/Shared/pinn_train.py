
"""
Shared.pinn_train
=================
Training utilities for PINNs (residual-only training loop for 1D Poisson).
Now with a pretty tqdm progress bar (no per-epoch prints).
"""

import time
import torch
import numpy as np
from tqdm.auto import tqdm

from .pinn_ops import physics_loss, u_of_x_builder, l2_error

torch.set_default_dtype(torch.float64)

def train_model(model, x, weights, f_grid, k: float, featurizer=None, steps: int = 2000, lr: float = 1e-3,
                save_path: str = None, progress: bool = True):
    """
    Train a model on the physics-informed residual loss.

    Args:
        model: torch.nn.Module
        x, weights, f_grid: tensors defining the domain quadrature and forcing
        k: frequency for exact solution (used only for rel-L2 reporting)
        featurizer: optional callable z(x) for input mapping
        steps: number of optimization iterations
        lr: Adam learning rate
        save_path: optional path to save a checkpoint (pt file) with losses and relerrs
        progress: if True, show a tqdm progress bar with live loss/relL2

    Returns:
        (losses: np.ndarray, relerrs: np.ndarray)
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses, relerrs = [], []

    iterator = range(steps)
    if progress:
        iterator = tqdm(iterator, desc="Training", leave=True, dynamic_ncols=True)

    for t in iterator:
        opt.zero_grad()
        u_of_x = u_of_x_builder(model, featurizer)
        loss = physics_loss(u_of_x, x, weights, f_grid)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

        with torch.no_grad():
            up = u_of_x(x)
            rel = l2_error(up, x, weights, k)

        losses.append(float(loss))
        relerrs.append(rel)

        if progress:
            # Update progress bar postfix with compact metrics
            iterator.set_postfix({'loss': f"{float(loss):.2e}", 'relL2': f"{rel:.2e}"})

    # Save checkpoint if path given
    if save_path is not None:
        state = {
            "model_state": model.state_dict(),
            "losses": np.array(losses),
            "relerrs": np.array(relerrs),
        }
        torch.save(state, save_path)

    return np.array(losses), np.array(relerrs)
