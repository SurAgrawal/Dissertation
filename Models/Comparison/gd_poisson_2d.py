"""
gd_poisson_2d.py
----------------

Gradient descent with line search for the 2‑D Poisson equation.  This
script replicates the gradient descent example from the
Natural‑Gradient‑PINNs ICML23 repository and adds persistent logging
of key training metrics.  The PDE and exact solution are the same as
in the Adam and ENGD variants, and a simple geometric grid is used
for the line search.  At regular intervals controlled by
``log_every`` the script records the current iteration, loss, L2
error, H1 error and line‑search step, writing them to
``gd_metrics.npz`` in the specified output directory when training
concludes.
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

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

jax.config.update("jax_enable_x64", True)


def main(
    steps: int = 200_000,
    log_every: int = 1_000,
    hidden_width: int = 32,
    hidden_depth: int = 1,
    seed: int = 0,
    outdir: str = "./",
) -> None:
    """Train a PINN for the 2‑D Poisson problem using gradient descent.

    Args:
        steps: Number of gradient descent updates to perform.
        log_every: Interval at which to record metrics.
        hidden_width: Width of each hidden layer.
        hidden_depth: Number of hidden layers.  The default (1) retains the
            original single‑hidden‑layer architecture; larger values build a
            deeper network with repeated hidden layers of equal width.
        seed: Random seed.
        outdir: Directory where metrics will be saved.
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
    run_dir = os.path.join(outdir, f"{ts}-gd")
    os.makedirs(run_dir, exist_ok=True)

    # training loop with progress bar
    for iteration in tqdm(range(steps + 1), desc="GD Training", unit="it"):
        grads = grad(loss)(params)
        params, actual_step = ls_update(params, grads)

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
                f"GD Iteration {iteration}: loss={l_val:.6e}, L2={l2_err:.6e}, H1={h1_err:.6e}, step={actual_step}"
            )

    # save metrics
    out_path = os.path.join(run_dir, "gd_metrics.npz")
    np.savez(out_path, **metrics)
    print(f"Metrics saved to {out_path}")

    # -------------------------------------------------------------------
    # Compute and save the predicted solution on a regular grid.
    #
    # To assess the learned solution visually, evaluate the network
    # on a dense grid of evaluation points spanning [0,1]^2.  The
    # resulting prediction matrix is stored in the run directory and
    # can later be compared to the analytical solution.
    # -------------------------------------------------------------------
    grid_res = 100
    xs = jnp.linspace(0.0, 1.0, grid_res)
    xv, yv = jnp.meshgrid(xs, xs)
    grid = jnp.stack([xv, yv], axis=-1).reshape(-1, 2)
    preds = v_model(params, grid)
    preds_np = np.array(preds).reshape(grid_res, grid_res)
    sol_path = os.path.join(run_dir, "gd_solution.npy")
    np.save(sol_path, preds_np)
    print(f"Solution saved to {sol_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train 2D Poisson PINN with gradient descent and log metrics.")
    parser.add_argument("--steps", type=int, default=50_000, help="Number of GD iterations")
    parser.add_argument("--log_every", type=int, default=1000, help="Logging interval")
    parser.add_argument("--hidden_width", type=int, default=32, help="Width of each hidden layer")
    parser.add_argument(
        "--hidden_depth", type=int, default=1,
        help="Number of hidden layers (default: 1).  Increase this to build a deeper"
             " network with the same width on each hidden layer."
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