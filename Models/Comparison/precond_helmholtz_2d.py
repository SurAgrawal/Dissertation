"""
precond_helmholtz_2d.py
----------------------

Eigenvalue-scaled Dirichlet-sine feature preconditioning for the 2-D Helmholtz equation
(using boundary penalty, not hard enforcement).

We solve on Ω=(0,1)^2:
    -Δu + k^2 u = f,    u|∂Ω = 0,
with manufactured solution u*(x,y)=sin(πx)sin(πy), hence f=(2π²+k²)u*.

This script matches the interface expected by run_all_methods_helmholtz.py:
  train_preconditioned_helmholtz(..., outdir=...)
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
device = torch.device("cpu")  # keep your current choice


def exact_solution(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(np.pi * x[:, [0]]) * torch.sin(np.pi * x[:, [1]])


def rhs(x: torch.Tensor, helmholtz_k: float) -> torch.Tensor:
    k2 = float(helmholtz_k) ** 2
    return (2.0 * (np.pi ** 2) + k2) * exact_solution(x)


def exact_grad(x: torch.Tensor) -> torch.Tensor:
    gx = np.pi * torch.cos(np.pi * x[:, [0]]) * torch.sin(np.pi * x[:, [1]])
    gy = np.pi * torch.sin(np.pi * x[:, [0]]) * torch.cos(np.pi * x[:, [1]])
    return torch.cat([gx, gy], dim=1)


def dirichlet_sine_features_scaled(x: torch.Tensor, k_max: int, helmholtz_k: float) -> torch.Tensor:
    """
    φ_nm(x,y) = sin(nπx) sin(mπy) / (λ_nm + k^2)
    with λ_nm = (n^2+m^2)π^2 and k = helmholtz_k.
    """
    N = x.shape[0]
    xx = x[:, 0].reshape(N, 1, 1)
    yy = x[:, 1].reshape(N, 1, 1)

    n = torch.arange(1, k_max + 1, device=x.device, dtype=x.dtype).reshape(1, k_max, 1)
    m = torch.arange(1, k_max + 1, device=x.device, dtype=x.dtype).reshape(1, 1, k_max)

    lam = (n**2 + m**2) * (np.pi ** 2)  # (1,k,k)
    k2 = (float(helmholtz_k) ** 2)
    denom = lam + k2
    feats = torch.sin(np.pi * n * xx) * torch.sin(np.pi * m * yy) / denom
    return feats.reshape(N, k_max * k_max)


class MLP(nn.Module):
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


def pde_residual(
    model: nn.Module,
    x_interior: torch.Tensor,
    rhs_func: Callable[[torch.Tensor, float], torch.Tensor],
    k_max: int,
    helmholtz_k: float,
) -> torch.Tensor:
    """
    Residual for -Δu + k^2 u = f:
        -Δu + k^2 u - f = 0.
    """
    x_interior = x_interior.clone().detach().requires_grad_(True)
    u = model(dirichlet_sine_features_scaled(x_interior, k_max, helmholtz_k))  # (N,1)

    grads = torch_grad(u, x_interior, torch.ones_like(u), create_graph=True)[0]  # (N,2)
    lap_parts = []
    for i in range(2):
        grad_i = grads[:, i]
        second = torch_grad(grad_i, x_interior, torch.ones_like(grad_i), create_graph=True)[0][:, i]
        lap_parts.append(second)
    laplacian = (lap_parts[0] + lap_parts[1]).unsqueeze(-1)  # (N,1)

    k2 = float(helmholtz_k) ** 2
    res = (-laplacian + k2 * u - rhs_func(x_interior, helmholtz_k))
    return res.pow(2)


def boundary_residual(model: nn.Module, x_boundary: torch.Tensor, k_max: int, helmholtz_k: float) -> torch.Tensor:
    u = model(dirichlet_sine_features_scaled(x_boundary, k_max, helmholtz_k))
    return u.pow(2)


def train_preconditioned_helmholtz(
    steps: int = 50_000,
    log_every: int = 1_000,
    batch_interior: int = 1_024,
    batch_boundary: int = 256,
    k_max: int = 10,
    width: int = 64,
    depth: int = 3,
    seed: int = 0,
    outdir: str = "./",
    tol: float = 0.0,
    helmholtz_k: float = 1.0,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    in_dim = k_max * k_max
    model = MLP(in_dim, width, depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    metrics: Dict[str, list] = {"iteration": [], "loss": [], "l2_error": [], "h1_error": []}

    os.makedirs(outdir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(outdir, f"{ts}-precond")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        f.write(f"helmholtz_k={float(helmholtz_k)}\n")

    eval_coords = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 1.0, 100, device=device),
            torch.linspace(0.0, 1.0, 100, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)

    for it in tqdm(range(steps + 1), desc="Preconditioned Training", unit="it"):
        x_int = torch.rand(batch_interior, 2, device=device)

        bb = (batch_boundary // 4) * 4
        s = torch.rand(bb // 4, 1, device=device)
        left = torch.cat([torch.zeros_like(s), s], dim=1)
        right = torch.cat([torch.ones_like(s), s], dim=1)
        bottom = torch.cat([s, torch.zeros_like(s)], dim=1)
        top = torch.cat([s, torch.ones_like(s)], dim=1)
        x_bdry = torch.cat([left, right, bottom, top], dim=0)

        optimizer.zero_grad(set_to_none=True)
        res_int = pde_residual(model, x_int, rhs, k_max, helmholtz_k)
        res_bdry = boundary_residual(model, x_bdry, k_max, helmholtz_k)
        loss = res_int.mean() + res_bdry.mean()
        loss.backward()
        optimizer.step()

        if it % log_every == 0:
            with torch.no_grad():
                pred = model(dirichlet_sine_features_scaled(eval_coords, k_max, helmholtz_k))
                true = exact_solution(eval_coords)
                err = pred - true
                l2_error = torch.sqrt(torch.mean(err ** 2)).item()

            eval_coords.requires_grad_(True)
            out = model(dirichlet_sine_features_scaled(eval_coords, k_max, helmholtz_k))
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
                f"Precond Iter {it}: loss={loss.item():.6e}, L2={l2_error:.6e}, H1={h1_error:.6e}, k={float(helmholtz_k)}"
            )

            if tol > 0.0 and len(metrics["loss"]) >= 2:
                if abs(metrics["loss"][-1] - metrics["loss"][-2]) < tol:
                    tqdm.write(f"Early stopping at iter {it} (tol={tol}).")
                    break

    np.savez(os.path.join(run_dir, "precond_metrics.npz"), **metrics)
    print(f"Metrics saved to {os.path.join(run_dir, 'precond_metrics.npz')}")

    # save solution grid
    grid_res = 100
    xs = torch.linspace(0.0, 1.0, grid_res, device=device)
    xv, yv = torch.meshgrid(xs, xs, indexing="ij")
    grid_coords = torch.stack([xv, yv], dim=-1).reshape(-1, 2)
    with torch.no_grad():
        pred = model(dirichlet_sine_features_scaled(grid_coords, k_max, helmholtz_k))
    pred_np = pred.detach().cpu().numpy().reshape(grid_res, grid_res)
    sol_path = os.path.join(run_dir, "precond_solution.npy")
    np.save(sol_path, pred_np)
    print(f"Solution saved to {sol_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Train preconditioned 2D Helmholtz PINN and log metrics.")
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--batch_interior", type=int, default=1024)
    p.add_argument("--batch_boundary", type=int, default=256)
    p.add_argument("--k_max", type=int, default=10)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="./helmholtz_logs")
    p.add_argument("--tol", type=float, default=0.0)
    p.add_argument("--helmholtz_k", type=float, default=1.0)
    args = p.parse_args()
    train_preconditioned_helmholtz(
        args.steps, args.log_every, args.batch_interior, args.batch_boundary,
        args.k_max, args.width, args.depth, args.seed, args.outdir, args.tol, args.helmholtz_k
    )