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

# Ensure the directory containing this script is on the Python path so
# that the local solver modules can be imported.  In the user's
# workflow all solver files reside alongside this script in the same
# directory rather than inside a separate package.  Adding the script
# directory to ``sys.path`` allows statements like ``import
# adam_poisson_2d`` to succeed when run as a standalone script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Import solver entrypoints from local modules.  Each solver module
# defines a ``main`` function for training.  The preconditioned solver
# exports ``train_preconditioned_poisson`` as its entrypoint.
import adam_poisson_2d as adam_solver  # type: ignore
import engd_poisson_2d as engd_solver  # type: ignore
import gd_poisson_2d as gd_solver      # type: ignore
import precond_poisson_2d as pre_solver  # type: ignore


def run_solver(
    name: str,
    steps: int,
    log_every: int,
    base_outdir: str,
    seed: int,
    width: int,
    depth: int,
    pre_params: Dict[str, int],
    tol: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str]:
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
        # Run Adam solver via its main entrypoint
        adam_solver.main(
            steps=steps,
            log_every=log_every,
            hidden_width=width,
            hidden_depth=depth,
            seed=seed,
            outdir=base_outdir,
            tol=tol,
        )
    elif name == "engd":
        # Run energy natural gradient solver via its main entrypoint
        engd_solver.main(
            steps=steps,
            log_every=log_every,
            hidden_width=width,
            hidden_depth=depth,
            seed=seed,
            outdir=base_outdir,
            tol=tol,
        )
    elif name == "gd":
        # Run gradient descent solver via its main entrypoint
        gd_solver.main(
            steps=steps,
            log_every=log_every,
            hidden_width=width,
            hidden_depth=depth,
            seed=seed,
            outdir=base_outdir,
            tol=tol,
        )
    elif name == "precond":
        # Run the preconditioned solver.  The module ``pre_solver``
        # defines ``train_preconditioned_poisson`` as its training entrypoint.
        pre_solver.train_preconditioned_poisson(
            steps=steps,
            log_every=log_every,
            batch_interior=pre_params.get("batch_interior", 1024),
            batch_boundary=pre_params.get("batch_boundary", 256),
            k_max=pre_params.get("k_max", 10),
            width=pre_params.get("width", width),
            depth=pre_params.get("depth", depth),
            seed=seed,
            outdir=base_outdir,
            tol=tol,
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
    return data["iteration"], data["loss"], data["l2_error"], data["h1_error"], train_time, run_dir


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
    # Early stopping tolerance: shared across all solvers.  If positive,
    # training of each solver will cease when the absolute change in loss
    # between successive logging events drops below this value.  Set to 0
    # to disable early stopping.
    parser.add_argument(
        "--tol", type=float, default=0.0,
        help="Early stopping tolerance for loss stagnation across solvers"
    )
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
        iters, losses, l2, h1, train_time, run_dir = run_solver(
            name,
            steps=args.steps,
            log_every=args.log_every,
            base_outdir=comparison_dir,
            seed=args.seed,
            width=solver_params[name]["width"],
            depth=solver_params[name]["depth"],
            pre_params=pre_params,
            tol=args.tol,
        )
        results[name] = {
            "iteration": iters,
            "loss": losses,
            "l2_error": l2,
            "h1_error": h1,
            "training_time": train_time,
            "run_dir": run_dir,
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

    # ------------------------------------------------------------------
    # Generate per‑solver solution comparison images with unified colour
    # scales.  To make differences across solvers immediately
    # comparable, we determine a global minimum and maximum for the
    # predicted and exact solutions across all solvers and a global
    # maximum for the absolute error.  These values are then used as
    # ``vmin`` and ``vmax`` when plotting the first two panels and
    # ``vmax`` when plotting the error.  Each figure comprises three
    # panels: the exact solution, the solver’s prediction, and the
    # absolute error.
    # ------------------------------------------------------------------
    # First load all solution arrays and determine global ranges
    solver_solutions: Dict[str, np.ndarray] = {}
    global_min = float("inf")
    global_max = float("-inf")
    global_error_max = float("-inf")
    # Build the exact solution grid lazily; we will compute it on demand
    true_solution_cache: Dict[int, np.ndarray] = {}
    for name, data in results.items():
        run_dir = data.get("run_dir")
        if name == "precond":
            sol_file = "precond_solution.npy"
        else:
            sol_file = f"{name}_solution.npy"
        sol_path = os.path.join(run_dir, sol_file)
        if not os.path.exists(sol_path):
            print(f"Warning: expected solution file {sol_path} not found; skipping plot.")
            continue
        sol = np.load(sol_path)
        solver_solutions[name] = sol
        global_min = min(global_min, float(sol.min()))
        global_max = max(global_max, float(sol.max()))
        # compute true solution for this grid size if not cached
        N = sol.shape[0]
        if N not in true_solution_cache:
            xs = np.linspace(0.0, 1.0, N)
            X, Y = np.meshgrid(xs, xs)
            true_solution_cache[N] = np.sin(np.pi * X) * np.sin(np.pi * Y)
        true_sol = true_solution_cache[N]
        err = np.abs(sol - true_sol)
        global_error_max = max(global_error_max, float(err.max()))
    # Now create the figures using the global scales
    for name, sol in solver_solutions.items():
        N = sol.shape[0]
        true_sol = true_solution_cache[N]
        error = np.abs(sol - true_sol)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        # panel 1: exact solution
        im0 = axes[0].imshow(true_sol, extent=[0, 1, 0, 1], origin='lower',
                             vmin=global_min, vmax=global_max, cmap='viridis')
        axes[0].set_title('Exact solution')
        fig.colorbar(im0, ax=axes[0])
        # panel 2: solver prediction
        im1 = axes[1].imshow(sol, extent=[0, 1, 0, 1], origin='lower',
                             vmin=global_min, vmax=global_max, cmap='viridis')
        axes[1].set_title(f'{name.capitalize()} solution')
        fig.colorbar(im1, ax=axes[1])
        # panel 3: absolute error
        im2 = axes[2].imshow(error, extent=[0, 1, 0, 1], origin='lower',
                             cmap='magma', vmin=0.0, vmax=global_error_max)
        axes[2].set_title('Absolute error')
        fig.colorbar(im2, ax=axes[2])
        for ax in axes:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        fig.tight_layout()
        img_path = os.path.join(comparison_dir, f"{name}_solution_comparison.png")
        plt.savefig(img_path, dpi=200)
        plt.close(fig)
        print(f"Saved solution comparison plot for {name} to {img_path}")


if __name__ == "__main__":
    main()