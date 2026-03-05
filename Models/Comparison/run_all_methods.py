"""
run_all_methods.py
------------------

This script automates running all four Poisson solvers (Adam, energy
natural gradient, gradient descent with line search, and Fourier‑feature
preconditioned optimization) on the same 2‑D Poisson equation.  It
invokes each solver with user‑specified hyperparameters, collects the
training loss and error metrics recorded during optimisation, and
produces a comparison plot of the training loss versus iteration.  A
summary of final errors is also saved as JSON for quick inspection.

The script assumes that the solvers live in the ``poisson_comparison``
package next to this file and that each solver writes its metrics
into a timestamped run directory under the provided ``--outdir``.

Example usage::

    python run_all_methods.py --steps 2000 --log-every 200 --outdir experiments

This will run each solver for 2000 optimisation steps (logging every
200 iterations) and place all run artefacts under ``experiments``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Ensure the poisson_comparison package can be imported.  Add the
# directory containing this script to the Python path and append the
# subfolder.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POISSON_DIR = os.path.join(SCRIPT_DIR, "poisson_comparison")
if POISSON_DIR not in sys.path:
    sys.path.append(POISSON_DIR)

from .adam_poisson_2d import main as adam_solver  # type: ignore
from .engd_poisson_2d import main as engd_solver  # type: ignore
from .gd_poisson_2d import main as gd_solver      # type: ignore
from .precond_poisson_2d import train_preconditioned_poisson as pre_solver  # type: ignore


def run_solver(
    name: str,
    steps: int,
    log_every: int,
    base_outdir: str,
    seed: int,
    width: int,
    depth: int,
    pre_params: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Execute a solver by name and return its recorded metrics.

    Parameters
    ----------
    name : str
        Name of the solver (adam, engd, gd, precond).
    steps : int
        Number of optimisation steps to run.
    log_every : int
        Interval in iterations at which to record metrics.
    base_outdir : str
        Directory under which run subfolders are created.
    seed : int
        Random seed passed to the solvers.
    hidden_width : int
        Hidden layer width for the JAX solvers (Adam/ENGD/GD).
    pre_params : Dict[str, int]
        Additional parameters for the preconditioned solver.

    Returns
    -------
    Tuple of numpy arrays: (iterations, loss, l2_error, h1_error)

    Raises
    ------
    ValueError
        If the name is unrecognised.
    """
    # Ensure base_outdir exists
    os.makedirs(base_outdir, exist_ok=True)
    # Choose solver function and run
    # Capture wall‑clock training time
    start_time = time.perf_counter()
    if name == "adam":
        # Run Adam solver
        adam_solver(
            steps=steps,
            log_every=log_every,
            hidden_width=width,
            hidden_depth=depth,
            seed=seed,
            outdir=base_outdir,
        )
    elif name == "engd":
        # Run energy natural gradient solver
        engd_solver(
            steps=steps,
            log_every=log_every,
            hidden_width=width,
            hidden_depth=depth,
            seed=seed,
            outdir=base_outdir,
        )
    elif name == "gd":
        # Run gradient descent solver
        gd_solver(
            steps=steps,
            log_every=log_every,
            hidden_width=width,
            hidden_depth=depth,
            seed=seed,
            outdir=base_outdir,
        )
    elif name == "precond":
        # Run preconditioned solver.  Provide batch sizes, width and depth.
        pre_solver(
            steps=steps,
            log_every=log_every,
            batch_interior=pre_params.get("batch_interior", 1024),
            batch_boundary=pre_params.get("batch_boundary", 256),
            k_max=pre_params.get("k_max", 10),
            width=pre_params.get("width", width),
            depth=pre_params.get("depth", depth),
            seed=seed,
            outdir=base_outdir,
        )
    else:
        raise ValueError(f"Unknown solver name '{name}'")

    # Record end time once solver has finished
    end_time = time.perf_counter()
    train_time = end_time - start_time
    # After running the solver, locate the most recent run directory for
    # this solver.  Subfolders are named like "YYYYMMDD-HHMMSS-adam".
    subdirs = [
        d
        for d in os.listdir(base_outdir)
        if d.endswith(f"-{name}") and os.path.isdir(os.path.join(base_outdir, d))
    ]
    if not subdirs:
        raise RuntimeError(f"Could not find run directory for solver {name}")
    # Select the newest directory by modification time
    subdirs.sort(key=lambda d: os.path.getmtime(os.path.join(base_outdir, d)))
    run_dir = os.path.join(base_outdir, subdirs[-1])
    # Construct metrics file name
    metrics_file = os.path.join(run_dir, f"{name}_metrics.npz")
    if not os.path.exists(metrics_file):
        raise RuntimeError(f"Metrics file not found for solver {name} at {metrics_file}")
    data = np.load(metrics_file)
    return data["iteration"], data["loss"], data["l2_error"], data["h1_error"], train_time


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all Poisson solvers and plot training loss comparisons")
    parser.add_argument("--steps", type=int, default=2000, help="Number of optimisation steps for each solver")
    parser.add_argument("--log-every", type=int, default=100, help="Interval at which solvers log metrics")
    parser.add_argument("--outdir", type=str, default="./Comparison/logs/comparison_runs", help="Base directory for run outputs")
    # Solver‑specific width and depth parameters.  Each solver can have a
    # distinct architecture.  For JAX solvers, ``width`` and ``depth``
    # correspond to the size and number of hidden layers.  The default
    # values reproduce the original single‑layer architecture.
    parser.add_argument("--adam-width", type=int, default=32, help="Hidden layer width for the Adam solver")
    parser.add_argument("--adam-depth", type=int, default=1, help="Hidden layer depth for the Adam solver")
    parser.add_argument("--engd-width", type=int, default=32, help="Hidden layer width for the ENGD solver")
    parser.add_argument("--engd-depth", type=int, default=1, help="Hidden layer depth for the ENGD solver")
    parser.add_argument("--gd-width", type=int, default=32, help="Hidden layer width for the GD solver")
    parser.add_argument("--gd-depth", type=int, default=1, help="Hidden layer depth for the GD solver")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    # Preconditioner specific arguments
    parser.add_argument("--pre-batch-interior", type=int, default=512, help="Interior batch size for preconditioned solver")
    parser.add_argument("--pre-batch-boundary", type=int, default=128, help="Boundary batch size for preconditioned solver")
    parser.add_argument("--pre-k-max", type=int, default=10, help="Number of Fourier frequencies for preconditioned solver")
    parser.add_argument("--pre-width", type=int, default=64, help="Hidden layer width for preconditioned solver")
    parser.add_argument("--pre-depth", type=int, default=3, help="Depth of MLP for preconditioned solver")
    args = parser.parse_args()

    # Base directory for this comparison run: append timestamp
    os.makedirs(args.outdir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    comparison_dir = os.path.join(args.outdir, f"{timestamp}-comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    # Parameters for preconditioned solver
    pre_params = {
        "batch_interior": args.pre_batch_interior,
        "batch_boundary": args.pre_batch_boundary,
        "k_max": args.pre_k_max,
        "width": args.pre_width,
        "depth": args.pre_depth,
    }

    # Define per‑solver architectural parameters.  For JAX solvers,
    # width and depth are taken from solver‑specific CLI arguments.  The
    # preconditioned solver uses its own width/depth provided via
    # ``pre_params``.
    solver_params = {
        "adam": {"width": args.adam_width, "depth": args.adam_depth},
        "engd": {"width": args.engd_width, "depth": args.engd_depth},
        "gd": {"width": args.gd_width, "depth": args.gd_depth},
        "precond": {"width": args.pre_width, "depth": args.pre_depth},
    }

    solvers = ["adam", "engd", "gd", "precond"]
    results = {}
    for name in solvers:
        print(f"Running solver {name}...")
        # For the preconditioned solver, width/depth from solver_params are
        # passed in but overridden inside ``run_solver`` by ``pre_params``.
        iters, losses, l2, h1, train_time = run_solver(
            name,
            steps=args.steps,
            log_every=args.log_every,
            base_outdir=comparison_dir,
            seed=args.seed,
            width=solver_params[name]["width"],
            depth=solver_params[name]["depth"],
            pre_params=pre_params,
        )
        results[name] = {
            "iteration": iters,
            "loss": losses,
            "l2_error": l2,
            "h1_error": h1,
            "training_time": train_time,
        }

    # Plot training loss comparisons
    plt.figure(figsize=(8, 5))
    for name, data in results.items():
        plt.plot(data["iteration"], data["loss"], label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training loss vs iteration for different solvers")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    loss_plot_path = os.path.join(comparison_dir, "training_loss_comparison.png")
    plt.savefig(loss_plot_path, dpi=200)
    plt.close()
    print(f"Saved training loss comparison plot to {loss_plot_path}")

    # Save summary metrics
    summary = {}
    for name, data in results.items():
        summary[name] = {
            "final_loss": float(data["loss"][-1]),
            "final_l2_error": float(data["l2_error"][-1]),
            "final_h1_error": float(data["h1_error"][-1]),
            "training_time_seconds": float(data["training_time"]),
        }
    summary_path = os.path.join(comparison_dir, "summary_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary metrics to {summary_path}")

    # Also save raw metrics for convenience
    for name, data in results.items():
        np.savez(os.path.join(comparison_dir, f"{name}_metrics.npz"), **data)


if __name__ == "__main__":
    main()