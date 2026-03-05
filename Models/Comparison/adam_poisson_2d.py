"""
adam_poisson_2d.py
-------------------

Modified implementation of the 2‑D Poisson example using the Adam
optimizer.  This script closely follows the original reference from
the Natural‑Gradient‑PINNs ICML23 repository but augments it to
record detailed training metrics for analysis.  The PDE under
consideration is −Δu = f on the unit square with u = 0 on the
boundary and exact solution ``u(x,y) = sin(πx) sin(πy)`` which
implies ``f(x,y) = 2π²u(x,y)``.  During training the code logs the
current loss and the L2 and H1 errors against the known analytical
solution every ``log_every`` iterations and stores these values in
``metrics.npz`` upon completion.
"""

import json
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
import optax
try:
    # tqdm provides a nice progress bar for loops
    from tqdm import tqdm
except ImportError:
    # fallback if tqdm is not installed; define a dummy wrapper
    def tqdm(iterable, *args, **kwargs):
        return iterable

from ngrad.models import init_params, mlp
from ngrad.domains import Square, SquareBoundary
from ngrad.integrators import DeterministicIntegrator
from ngrad.utility import laplace

jax.config.update("jax_enable_x64", True)


def main(
    steps: int = 200_000,
    log_every: int = 1_000,
    hidden_width: int = 32,
    hidden_depth: int = 1,
    seed: int = 0,
    outdir: str = "./",
) -> None:
    """Train a PINN for the 2‑D Poisson problem with Adam and log metrics.

    Args:
        steps: Number of training iterations to perform.
        log_every: Frequency (in iterations) at which to record metrics.
        hidden_width: Width of each hidden layer.
        hidden_depth: Number of hidden layers.  The default (1) reproduces the
            original example with a single hidden layer, while larger values
            construct a deeper multilayer perceptron by repeating the
            hidden layer ``hidden_depth`` times.
        seed: Random seed for reproducibility.
        outdir: Directory where metrics will be written.

    The resulting metrics are saved as ``metrics.npz`` in ``outdir``.
    """

    # domains
    interior = Square(1.0)
    boundary = SquareBoundary(1.0)

    # integrators (increase points proportional to network width)
    n_interior = 32
    n_boundary = 32
    interior_integrator = DeterministicIntegrator(interior, n_interior)
    boundary_integrator = DeterministicIntegrator(boundary, n_boundary)
    eval_integrator = DeterministicIntegrator(interior, 200)

    # model and parameters
    activation = lambda x: jnp.tanh(x)
    # build a list of layer sizes: input dimension 2, ``hidden_depth`` hidden
    # layers each of size ``hidden_width``, and a single output unit
    layer_sizes = [2] + [hidden_width] * hidden_depth + [1]
    params = init_params(layer_sizes, random.PRNGKey(seed))
    model = mlp(activation)
    v_model = vmap(model, (None, 0))

    # exact solution and source term
    @jit
    def u_star(x):
        return jnp.prod(jnp.sin(jnp.pi * x))

    @jit
    def f(x):
        return 2.0 * jnp.pi**2 * u_star(x)

    # residual computation
    laplace_model = lambda prm: laplace(lambda x: model(prm, x))
    residual = lambda prm, x: (laplace_model(prm)(x) + f(x))**2
    v_residual = jit(vmap(residual, (None, 0)))

    # loss functions
    @jit
    def interior_loss(prm):
        return interior_integrator(lambda x: v_residual(prm, x))

    @jit
    def boundary_loss(prm):
        return boundary_integrator(lambda x: v_model(prm, x)**2)

    @jit
    def loss(prm):
        return interior_loss(prm) + boundary_loss(prm)

    # error norms
    err = lambda x: model(params, x) - u_star(x)
    v_error = vmap(err, (0))
    v_error_abs_grad = vmap(lambda x: jnp.dot(grad(err)(x), grad(err)(x))**0.5)

    def l2_norm(func, integrator):
        return integrator(lambda x: (func(x))**2)**0.5

    # optimizer
    exponential_decay = optax.exponential_decay(
        init_value=0.001,
        transition_steps=10_000,
        transition_begin=15_000,
        decay_rate=0.1,
        end_value=1e-7,
    )
    optimizer = optax.adam(learning_rate=exponential_decay)
    opt_state = optimizer.init(params)

    # metrics storage
    metrics = {
        "iteration": [],
        "loss": [],
        "l2_error": [],
        "h1_error": []
    }

    # ensure output directory exists and create a unique run directory
    os.makedirs(outdir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(outdir, f"{ts}-adam")
    os.makedirs(run_dir, exist_ok=True)

    # training loop with progress bar
    for iteration in tqdm(range(steps + 1), desc="Adam Training", unit="it"):
        grads = grad(loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if iteration % log_every == 0:
            # compute metrics lazily (outside jit to avoid compilation overhead)
            l_val = loss(params)
            l2_err = l2_norm(v_error, eval_integrator)
            h1_err = l2_err + l2_norm(v_error_abs_grad, eval_integrator)
            metrics["iteration"].append(int(iteration))
            metrics["loss"].append(float(l_val))
            metrics["l2_error"].append(float(l2_err))
            metrics["h1_error"].append(float(h1_err))
            # print progress information alongside tqdm bar
            tqdm.write(
                f"Adam Iteration {iteration}: loss={l_val:.6e}, L2={l2_err:.6e}, H1={h1_err:.6e}"
            )

    # save metrics
    out_path = os.path.join(run_dir, "adam_metrics.npz")
    np.savez(out_path, **metrics)
    print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train PINN for 2D Poisson with Adam and log metrics.")
    parser.add_argument("--steps", type=int, default=50_000, help="Number of training iterations")
    parser.add_argument("--log_every", type=int, default=1000, help="Logging frequency in iterations")
    parser.add_argument("--hidden_width", type=int, default=32, help="Width of each hidden layer")
    parser.add_argument(
        "--hidden_depth", type=int, default=1,
        help="Number of hidden layers (default: 1).  Setting this to a value >1"
             " creates a deeper network with the same width on each hidden layer."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--outdir", type=str, default="./poisson_logs", help="Output directory for metrics")
    args = parser.parse_args()
    main(
        args.steps,
        args.log_every,
        args.hidden_width,
        args.hidden_depth,
        args.seed,
        args.outdir,
    )