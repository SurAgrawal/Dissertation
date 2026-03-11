"""
adam_helmholtz_2d.py
--------------------

Modified implementation of the 2-D Helmholtz example using the Adam
optimizer.

We solve on the unit square Ω=(0,1)^2:
    -Δu + k^2 u = f,   u|∂Ω = 0,
with manufactured solution u*(x,y)=sin(πx)sin(πy), which implies
    f(x,y) = (2π² + k²) u*(x,y).

The script logs loss and L2/H1 errors against u* every ``log_every`` steps,
and saves metrics and the predicted solution grid into a timestamped run
directory under ``outdir``.
"""

import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
import optax

try:
    from tqdm import tqdm
except ImportError:
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
    tol: float = 0.0,
    helmholtz_k: float = 1.0,
) -> None:
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

    k2 = jnp.asarray(float(helmholtz_k) ** 2)

    # exact solution and rhs
    @jit
    def u_star(x):
        return jnp.prod(jnp.sin(jnp.pi * x))

    @jit
    def f_rhs(x):
        return (2.0 * jnp.pi**2 + k2) * u_star(x)

    # Helmholtz residual using laplace_model = Δu
    # PDE: -Δu + k^2 u = f  <=>  Δu - k^2 u + f = 0
    laplace_model = lambda prm: laplace(lambda x: model(prm, x))
    residual = lambda prm, x: (laplace_model(prm)(x) - k2 * model(prm, x) + f_rhs(x))**2
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

    # error norms
    err = lambda x: model(params, x) - u_star(x)
    v_error = vmap(err, (0))
    v_error_abs_grad = vmap(lambda x: jnp.dot(grad(err)(x), grad(err)(x))**0.5)

    def l2_norm(func, integrator):
        return integrator(lambda x: (func(x))**2)**0.5

    # optimizer (same schedule)
    exponential_decay = optax.exponential_decay(
        init_value=0.001,
        transition_steps=10_000,
        transition_begin=15_000,
        decay_rate=0.1,
        end_value=1e-7,
    )
    optimizer = optax.adam(learning_rate=exponential_decay)
    opt_state = optimizer.init(params)

    metrics = {"iteration": [], "loss": [], "l2_error": [], "h1_error": []}

    os.makedirs(outdir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(outdir, f"{ts}-adam")
    os.makedirs(run_dir, exist_ok=True)

    # save config
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        f.write(f"helmholtz_k={float(helmholtz_k)}\n")

    for iteration in tqdm(range(steps + 1), desc="Adam Training", unit="it"):
        grads = grad(loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if iteration % log_every == 0:
            l_val = loss(params)
            l2_err = l2_norm(v_error, eval_integrator)
            h1_err = l2_err + l2_norm(v_error_abs_grad, eval_integrator)
            metrics["iteration"].append(int(iteration))
            metrics["loss"].append(float(l_val))
            metrics["l2_error"].append(float(l2_err))
            metrics["h1_error"].append(float(h1_err))
            tqdm.write(
                f"Adam Iter {iteration}: loss={l_val:.6e}, L2={l2_err:.6e}, H1={h1_err:.6e}, k={float(helmholtz_k)}"
            )

            if tol > 0.0 and len(metrics["loss"]) >= 2:
                if abs(metrics["loss"][-1] - metrics["loss"][-2]) < tol:
                    tqdm.write(f"Early stopping at iter {iteration} (tol={tol}).")
                    break

    np.savez(os.path.join(run_dir, "adam_metrics.npz"), **metrics)
    print(f"Metrics saved to {os.path.join(run_dir, 'adam_metrics.npz')}")

    # save solution grid
    grid_res = 100
    xs = jnp.linspace(0.0, 1.0, grid_res)
    xv, yv = jnp.meshgrid(xs, xs)
    grid = jnp.stack([xv, yv], axis=-1).reshape(-1, 2)
    preds = v_model(params, grid)
    preds_np = np.array(preds).reshape(grid_res, grid_res)
    sol_path = os.path.join(run_dir, "adam_solution.npy")
    np.save(sol_path, preds_np)
    print(f"Solution saved to {sol_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Train PINN for 2D Helmholtz with Adam and log metrics.")
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--hidden_width", type=int, default=32)
    p.add_argument("--hidden_depth", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="./helmholtz_logs")
    p.add_argument("--tol", type=float, default=0.0)
    p.add_argument("--helmholtz_k", type=float, default=1.0)
    args = p.parse_args()
    main(
        args.steps, args.log_every, args.hidden_width, args.hidden_depth,
        args.seed, args.outdir, args.tol, args.helmholtz_k
    )