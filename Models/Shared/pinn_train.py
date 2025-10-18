
"""
Shared.pinn_train
=================
Training utilities for PINNs (residual-only training loop for 1D Poisson).
"""

import time
import torch
import numpy as np
from .pinn_ops import physics_loss, u_of_x_builder, l2_error

torch.set_default_dtype(torch.float64)

def train_model(model, x, weights, f_grid, k: float, featurizer=None, steps: int = 2000, lr: float = 1e-3,
                verbose_every: int = 200, save_path: str = None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses, relerrs = [], []
    start = time.time()

    for t in range(steps):
        opt.zero_grad()
        u_of_x = u_of_x_builder(model, featurizer)
        loss = physics_loss(u_of_x, x, weights, f_grid)
        loss.backward()
        opt.step()

        with torch.no_grad():
            up = u_of_x(x)
            rel = l2_error(up, x, weights, k)

        if (t+1) % verbose_every == 0 or t == 0:
            elapsed = time.time() - start
            print(f"[{t+1:5d}/{steps}] loss={float(loss):.3e} | relL2={rel:.3e} | {elapsed:.1f}s")

        losses.append(float(loss))
        relerrs.append(rel)

    # Save checkpoint if path given
    if save_path is not None:
        state = {
            "model_state": model.state_dict(),
            "losses": np.array(losses),
            "relerrs": np.array(relerrs),
        }
        torch.save(state, save_path)

    return np.array(losses), np.array(relerrs)
