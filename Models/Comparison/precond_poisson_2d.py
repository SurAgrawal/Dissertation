"""
precond_poisson_2d.py
---------------------

Eigenvalue-scaled Dirichlet-sine feature preconditioning for the 2-D Poisson equation
(using boundary penalty, not hard enforcement).

We solve on Ω=(0,1)^2:
    -Δu = f,    u|∂Ω = 0,
with manufactured solution
    u*(x,y) = sin(πx) sin(πy),   f(x,y) = 2π^2 u*(x,y).

This script matches the interface expected by run_all_methods.py:
  train_preconditioned_poisson(..., outdir=...)
and writes metrics to:
  <outdir>/<timestamp>-precond/precond_metrics.npz
containing iteration/loss/l2_error/h1_error.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Dict

import numpy as np
import torch
from torch import nn
from torch.autograd import grad as torch_grad

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Exact solution + RHS + grad
# -----------------------------
def exact_solution(x: torch.Tensor) -> torch.Tensor:
    """Analytical solution sin(pi x) sin(pi y)."""
    return torch.sin(np.pi * x[:, [0]]) * torch.sin(np.pi * x[:, [1]])


def rhs(x: torch.Tensor) -> torch.Tensor:
    """Right hand side f for -Δu = f given the analytic solution."""
    return 2.0 * (np.pi ** 2) * exact_solution(x)


def exact_grad(x: torch.Tensor) -> torch.Tensor:
    """Analytical gradient of the exact solution."""
    gx = np.pi * torch.cos(np.pi * x[:, [0]]) * torch.sin(np.pi * x[:, [1]])
    gy = np.pi * torch.sin(np.pi * x[:, [0]]) * torch.cos(np.pi * x[:, [1]])
    return torch.cat([gx, gy], dim=1)


# -----------------------------------------
# Eigenvalue-scaled Dirichlet sine features
# -----------------------------------------
def dirichlet_sine_features_scaled(x: torch.Tensor, k_max: int) -> torch.Tensor:
    """
    φ_nm(x,y) = sin(nπx) sin(mπy) / λ_nm
    with λ_nm = (n^2+m^2)π^2, n,m=1..k_max.
    Returns shape (N, k_max^2).
    """
    N = x.shape[0]
    xx = x[:, 0].reshape(N, 1, 1)
    yy = x[:, 1].reshape(N, 1, 1)

    n = torch.arange(1, k_max + 1, device=x.device, dtype=x.dtype).reshape(1, k_max, 1)
    m = torch.arange(1, k_max + 1, device=x.device, dtype=x.dtype).reshape(1, 1, k_max)

    lam = (n**2 + m**2) * (np.pi ** 2)  # (1,k,k)
    feats = torch.sin(np.pi * n * xx) * torch.sin(np.pi * m * yy) / lam  # (N,k,k)
    return feats.reshape(N, k_max * k_max)


# -----------------------------
# MLP (tanh as in your script)
# -----------------------------
class MLP(nn.Module):
    """Simple fully connected network with configurable depth and width."""

    def __init__(self, in_dim: int, width: int, depth: int, out_dim: int = 1) -> None:
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(dim, width))
            layers.append(nn.Tanh())
            dim = width
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Residuals: -Δu - f = 0
# -----------------------------
def pde_residual(model: nn.Module, x_interior: torch.Tensor,
                 rhs_func: Callable[[torch.Tensor], torch.Tensor],
                 k_max: int) -> torch.Tensor:
    """Compute squared residual of -Δu - f = 0 at interior points."""
    x_interior = x_interior.clone().detach().requires_grad_(True)
    u = model(dirichlet_sine_features_scaled(x_interior, k_max))  # (N,1)

    grads = torch_grad(u, x_interior, torch.ones_like(u), create_graph=True)[0]  # (N,2)
    lap_parts = []
    for i in range(2):
        grad_i = grads[:, i]
        second = torch_grad(grad_i, x_interior, torch.ones_like(grad_i), create_graph=True)[0][:, i]
        lap_parts.append(second)
    laplacian = (lap_parts[0] + lap_parts[1]).unsqueeze(-1)  # (N,1)

    res = -laplacian - rhs_func(x_interior)
    return res.pow(2)


def boundary_residual(model: nn.Module, x_boundary: torch.Tensor, k_max: int) -> torch.Tensor:
    """Squared residual of boundary condition u=0 at boundary points (penalty)."""
    u = model(dirichlet_sine_features_scaled(x_boundary, k_max))
    return u.pow(2)


# -----------------------------
# Train entrypoint (run_all_methods-compatible)
# -----------------------------
def train_preconditioned_poisson(
    steps: int = 50_000,
    log_every: int = 1_000,
    batch_interior: int = 1_024,
    batch_boundary: int = 256,
    k_max: int = 10,
    width: int = 64,
    depth: int = 3,
    seed: int = 0,
    outdir: str = "./",
) -> None:
    """Train PINN with eigenvalue-scaled Dirichlet-sine features and record metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    in_dim = k_max * k_max
    model = MLP(in_dim, width, depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    metrics: Dict[str, list] = {
        "iteration": [],
        "loss": [],
        "l2_error": [],
        "h1_error": [],
    }

    os.makedirs(outdir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(outdir, f"{ts}-precond")
    os.makedirs(run_dir, exist_ok=True)

    # evaluation grid for error measurement
    eval_coords = torch.stack(torch.meshgrid(
        torch.linspace(0.0, 1.0, 100, device=device),
        torch.linspace(0.0, 1.0, 100, device=device),
        indexing="ij"
    ), dim=-1).reshape(-1, 2)

    for it in tqdm(range(steps + 1), desc="Preconditioned Training", unit="it"):
        # sample interior points uniformly
        x_int = torch.rand(batch_interior, 2, device=device)

        # sample boundary points (four edges); keep same style as your other scripts
        bb = (batch_boundary // 4) * 4
        s = torch.rand(bb // 4, 1, device=device)
        left = torch.cat([torch.zeros_like(s), s], dim=1)
        right = torch.cat([torch.ones_like(s), s], dim=1)
        bottom = torch.cat([s, torch.zeros_like(s)], dim=1)
        top = torch.cat([s, torch.ones_like(s)], dim=1)
        x_bdry = torch.cat([left, right, bottom, top], dim=0)

        optimizer.zero_grad(set_to_none=True)
        res_int = pde_residual(model, x_int, rhs, k_max)
        res_bdry = boundary_residual(model, x_bdry, k_max)
        loss = res_int.mean() + res_bdry.mean()
        loss.backward()
        optimizer.step()

        if it % log_every == 0:
            # L2 error without grads
            with torch.no_grad():
                pred = model(dirichlet_sine_features_scaled(eval_coords, k_max))
                true = exact_solution(eval_coords)
                err = pred - true
                l2_error = torch.sqrt(torch.mean(err ** 2)).item()

            # H1 error (needs grads wrt coords)
            eval_coords.requires_grad_(True)
            out = model(dirichlet_sine_features_scaled(eval_coords, k_max))
            grads = torch_grad(out, eval_coords, torch.ones_like(out), create_graph=False)[0]
            grad_true = exact_grad(eval_coords)
            grad_err = grads - grad_true
            h1_error = l2_error + torch.sqrt(torch.mean(torch.sum(grad_err ** 2, dim=1))).item()
            eval_coords.requires_grad_(False)

            metrics["iteration"].append(int(it))
            metrics["loss"].append(float(loss.item()))
            metrics["l2_error"].append(float(l2_error))
            metrics["h1_error"].append(float(h1_error))

            tqdm.write(
                f"Precond Iter {it}: loss={loss.item():.6e}, L2={l2_error:.6e}, H1={h1_error:.6e}"
            )

    out_path = os.path.join(run_dir, "precond_metrics.npz")
    np.savez(out_path, **metrics)
    print(f"Metrics saved to {out_path}")

    # -------------------------------------------------------------------
    # Compute and save the predicted solution on a regular grid.
    #
    # After training, evaluate the PINN on a dense grid of points to
    # visualise the learned solution.  We generate a mesh of
    # ``grid_res`` points in each dimension, compute the network
    # prediction at each point using the same eigenvalue‑scaled
    # Dirichlet‑sine features as during training, reshape the
    # predictions into a 2‑D array, and store them in the run
    # directory.  This file can then be loaded by external scripts
    # to compare against the exact solution and visualise errors.
    # -------------------------------------------------------------------
    grid_res = 100
    xs = torch.linspace(0.0, 1.0, grid_res, device=device)
    xv, yv = torch.meshgrid(xs, xs, indexing="ij")
    grid_coords = torch.stack([xv, yv], dim=-1).reshape(-1, 2)
    with torch.no_grad():
        pred = model(dirichlet_sine_features_scaled(grid_coords, k_max))
    pred_np = pred.detach().cpu().numpy().reshape(grid_res, grid_res)
    sol_path = os.path.join(run_dir, "precond_solution.npy")
    np.save(sol_path, pred_np)
    print(f"Solution saved to {sol_path}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train preconditioned 2D Poisson PINN (eigenvalue-scaled sine features) and log metrics.")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--batch_interior", type=int, default=1024)
    parser.add_argument("--batch_boundary", type=int, default=256)
    parser.add_argument("--k_max", type=int, default=10)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="./poisson_logs")
    args = parser.parse_args()
    train_preconditioned_poisson(
        args.steps,
        args.log_every,
        args.batch_interior,
        args.batch_boundary,
        args.k_max,
        args.width,
        args.depth,
        args.seed,
        args.outdir,
    )


if __name__ == "__main__":
    main()