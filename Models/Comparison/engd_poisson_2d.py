"""
engd_poisson_2d.py
------------------

Energy Natural Gradient (ENGD) implementation for the 2‑D Poisson
equation.  This script modifies the original example from the
Natural‑Gradient‑PINNs ICML23 codebase to record training metrics at
regular intervals and store them for later analysis.  The problem is
identical to the one solved by the Adam variant: we seek ``u`` on the
unit square satisfying −Δu = f with homogeneous Dirichlet boundary
conditions and exact solution ``u(x,y) = sin(πx) sin(πy)``.  The
natural gradient is constructed via gramians of the boundary and
Laplace operators, and a simple grid line search is used to select
the optimal step size.  Metrics such as the loss, L2 error and H1
error are recorded every ``log_every`` iterations and saved in
``engd_metrics.npz``.
"""

import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit

from ngrad.models import init_params, mlp
from ngrad.domains import Square, SquareBoundary
from ngrad.integrators import DeterministicIntegrator
from ngrad.utility import laplace, grid_line_search_factory
from ngrad.inner import model_laplace, model_identity
from ngrad.gram import gram_factory, nat_grad_factory

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

jax.config.update("jax_enable_x64", True)


def main(
    steps: int = 50,
    log_every: int = 5,
    hidden_width: int = 32,
    hidden_depth: int = 1,
    seed: int = 0,
    outdir: str = "./",
) -> None:
    """Train the 2‑D Poisson PINN with energy natural gradient descent.

    Args:
        steps: Number of natural gradient updates.  The original
            example runs for 51 iterations; this argument allows
            modification.
        log_every: Interval at which to compute and record metrics.
        hidden_width: Width of each hidden layer.
        hidden_depth: Number of hidden layers.  A value of 1 matches the
            original implementation (single hidden layer), while larger values
            create a deeper network of equal‑width hidden layers.
        seed: Random seed.
        outdir: Directory to which metrics are written.
    """

    # domains
    interior = Square(1.0)
    boundary = SquareBoundary(1.0)

    # integrators
    n_interior = 32
    n_boundary = 32
    interior_integrator = DeterministicIntegrator(interior, n_interior)
    boundary_integrator = DeterministicIntegrator(boundary, n_boundary)
    eval_integrator = DeterministicIntegrator(interior, 200)

    # model
    activation = lambda x: jnp.tanh(x)
    layer_sizes = [2] + [hidden_width] * hidden_depth + [1]
    params = init_params(layer_sizes, random.PRNGKey(seed))
    model = mlp(activation)
    v_model = vmap(model, (None, 0))

    # exact solution and rhs
    @jit
    def u_star(x):
        return jnp.prod(jnp.sin(jnp.pi * x))

    @jit
    def f(x):
        return 2.0 * jnp.pi**2 * u_star(x)

    # gramian factories
    gram_bdry = gram_factory(model=model, trafo=model_identity, integrator=boundary_integrator)
    gram_laplace = gram_factory(model=model, trafo=model_laplace, integrator=interior_integrator)

    @jit
    def gram(prm):
        return gram_laplace(prm) + gram_bdry(prm)

    nat_grad = nat_grad_factory(gram)

    # residual and loss
    laplace_model = lambda prm: laplace(lambda x: model(prm, x))
    residual = lambda prm, x: (laplace_model(prm)(x) + f(x))**2
    v_residual = jit(vmap(residual, (None, 0)))

    @jit
    def interior_loss(prm):
        return interior_integrator(lambda x: v_residual(prm, x))

    @jit
    def boundary_loss(prm):
        return boundary_integrator(lambda x: v_model(prm, x)**2)

    @jit
    def loss(prm):
        return interior_loss(prm) + boundary_loss(prm)

    # grid line search
    grid = jnp.linspace(0, 30, 31)
    steps_grid = 0.5**grid
    ls_update = grid_line_search_factory(loss, steps_grid)

    # error functions
    err = lambda x: model(params, x) - u_star(x)
    v_error = vmap(err, (0))
    v_error_abs_grad = vmap(lambda x: jnp.dot(grad(err)(x), grad(err)(x))**0.5)

    def l2_norm(func, integrator):
        return integrator(lambda x: (func(x))**2)**0.5

    # metrics storage
    metrics = {
        "iteration": [],
        "loss": [],
        "l2_error": [],
        "h1_error": [],
        "step": []
    }

    # ensure output directory exists and create a unique run directory
    os.makedirs(outdir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(outdir, f"{ts}-engd")
    os.makedirs(run_dir, exist_ok=True)

    # training loop with progress bar
    for iteration in tqdm(range(steps + 1), desc="ENGD Training", unit="it"):
        grads = grad(loss)(params)
        nat_grads = nat_grad(params, grads)
        params, actual_step = ls_update(params, nat_grads)

        if iteration % log_every == 0:
            l_val = loss(params)
            l2_err = l2_norm(v_error, eval_integrator)
            h1_err = l2_err + l2_norm(v_error_abs_grad, eval_integrator)
            metrics["iteration"].append(int(iteration))
            metrics["loss"].append(float(l_val))
            metrics["l2_error"].append(float(l2_err))
            metrics["h1_error"].append(float(h1_err))
            metrics["step"].append(float(actual_step))
            tqdm.write(
                f"ENGD Iteration {iteration}: loss={l_val:.6e}, L2={l2_err:.6e}, H1={h1_err:.6e}, step={actual_step}"
            )

    # save metrics
    out_path = os.path.join(run_dir, "engd_metrics.npz")
    np.savez(out_path, **metrics)
    print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train 2D Poisson PINN with energy natural gradient and log metrics.")
    parser.add_argument("--steps", type=int, default=50, help="Number of ENGD iterations")
    parser.add_argument("--log_every", type=int, default=5, help="Logging interval")
    parser.add_argument("--hidden_width", type=int, default=32, help="Width of each hidden layer")
    parser.add_argument(
        "--hidden_depth", type=int, default=1,
        help="Number of hidden layers (default: 1).  Higher values create a deeper"
             " MLP with uniform hidden width."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--outdir", type=str, default="./poisson_logs", help="Output directory")
    args = parser.parse_args()
    main(
        args.steps,
        args.log_every,
        args.hidden_width,
        args.hidden_depth,
        args.seed,
        args.outdir,
    )