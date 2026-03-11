"""
run_all_methods_helmholtz.py
----------------------------

Run all four Helmholtz solvers (Adam, ENGD, GD+line-search, preconditioned)
for the 2-D Helmholtz equation:
    -Δu + k^2 u = f,   u|∂Ω=0,
with u*(x,y)=sin(πx)sin(πy), f=(2π²+k²)u*.

Produces loss comparison plots, per-solver solution comparison images,
a single combined solution-comparison image (GD, Adam, Precond, ENGD),
and a summary JSON of final metrics and training time.

Quality-of-life: Press Ctrl+C during a solver run to SKIP it and move on
to the next solver (instead of terminating the whole script).
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import adam_helmholtz_2d as adam_solver  # type: ignore
import engd_helmholtz_2d as engd_solver  # type: ignore
import gd_helmholtz_2d as gd_solver      # type: ignore
import precond_helmholtz_2d as pre_solver  # type: ignore


def run_solver(
    name: str,
    steps: int,
    log_every: int,
    base_outdir: str,
    seed: int,
    width: int,
    depth: int,
    pre_params: Dict[str, int],
    tol: float,
    helmholtz_k: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str]:
    os.makedirs(base_outdir, exist_ok=True)
    start_time = time.perf_counter()

    if name == "adam":
        adam_solver.main(
            steps=steps, log_every=log_every, hidden_width=width, hidden_depth=depth,
            seed=seed, outdir=base_outdir, tol=tol, helmholtz_k=helmholtz_k
        )
    elif name == "engd":
        engd_solver.main(
            steps=steps, log_every=log_every, hidden_width=width, hidden_depth=depth,
            seed=seed, outdir=base_outdir, tol=tol, helmholtz_k=helmholtz_k
        )
    elif name == "gd":
        gd_solver.main(
            steps=steps, log_every=log_every, hidden_width=width, hidden_depth=depth,
            seed=seed, outdir=base_outdir, tol=tol, helmholtz_k=helmholtz_k
        )
    elif name == "precond":
        pre_solver.train_preconditioned_helmholtz(
            steps=steps, log_every=log_every,
            batch_interior=pre_params.get("batch_interior", 1024),
            batch_boundary=pre_params.get("batch_boundary", 256),
            k_max=pre_params.get("k_max", 10),
            width=pre_params.get("width", width),
            depth=pre_params.get("depth", depth),
            seed=seed, outdir=base_outdir, tol=tol, helmholtz_k=helmholtz_k
        )
    else:
        raise ValueError(f"Unknown solver name '{name}'")

    train_time = time.perf_counter() - start_time

    subdirs = [
        d for d in os.listdir(base_outdir)
        if d.endswith(f"-{name}") and os.path.isdir(os.path.join(base_outdir, d))
    ]
    if not subdirs:
        raise RuntimeError(f"Could not find run directory for solver {name}")
    subdirs.sort(key=lambda d: os.path.getmtime(os.path.join(base_outdir, d)))
    run_dir = os.path.join(base_outdir, subdirs[-1])

    metrics_file = os.path.join(run_dir, f"{name}_metrics.npz")
    if not os.path.exists(metrics_file):
        raise RuntimeError(f"Metrics file not found for solver {name} at {metrics_file}")
    data = np.load(metrics_file)
    return data["iteration"], data["loss"], data["l2_error"], data["h1_error"], train_time, run_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Run all Helmholtz solvers and plot comparisons")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--outdir", type=str, default="./Comparison/logs/helmholtz_runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=0.0)

    p.add_argument("--helmholtz-k", type=float, default=1.0, help="k in -Δu + k^2 u = f")

    p.add_argument("--adam-width", type=int, default=64)
    p.add_argument("--adam-depth", type=int, default=3)
    p.add_argument("--engd-width", type=int, default=64)
    p.add_argument("--engd-depth", type=int, default=1)
    p.add_argument("--gd-width", type=int, default=64)
    p.add_argument("--gd-depth", type=int, default=3)

    p.add_argument("--pre-batch-interior", type=int, default=512)
    p.add_argument("--pre-batch-boundary", type=int, default=128)
    p.add_argument("--pre-k-max", type=int, default=10)
    p.add_argument("--pre-width", type=int, default=64)
    p.add_argument("--pre-depth", type=int, default=3)

    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    comparison_dir = os.path.join(args.outdir, f"{timestamp}-comparison-k{args.helmholtz_k:g}")
    os.makedirs(comparison_dir, exist_ok=True)

    pre_params = {
        "batch_interior": args.pre_batch_interior,
        "batch_boundary": args.pre_batch_boundary,
        "k_max": args.pre_k_max,
        "width": args.pre_width,
        "depth": args.pre_depth,
    }

    solver_params = {
        "adam": {"width": args.adam_width, "depth": args.adam_depth},
        "engd": {"width": args.engd_width, "depth": args.engd_depth},
        "gd": {"width": args.gd_width, "depth": args.gd_depth},
        "precond": {"width": args.pre_width, "depth": args.pre_depth},
    }

    solvers = ["adam", "engd", "gd", "precond"]
    results: Dict[str, dict] = {}

    for name in solvers:
        print(f"Running solver {name} (k={args.helmholtz_k})...  (Ctrl+C to skip)")
        try:
            iters, losses, l2, h1, train_time, run_dir = run_solver(
                name=name,
                steps=args.steps,
                log_every=args.log_every,
                base_outdir=comparison_dir,
                seed=args.seed,
                width=solver_params[name]["width"],
                depth=solver_params[name]["depth"],
                pre_params=pre_params,
                tol=args.tol,
                helmholtz_k=args.helmholtz_k,
            )
        except KeyboardInterrupt:
            print(f"\nSkipped solver {name} (KeyboardInterrupt). Moving to next.\n")
            continue
        except Exception as e:
            print(f"\nSolver {name} failed with error: {e}\nMoving to next.\n")
            continue

        results[name] = {
            "iteration": iters,
            "loss": losses,
            "l2_error": l2,
            "h1_error": h1,
            "training_time": train_time,
            "run_dir": run_dir,
        }

    if not results:
        print("No solver finished successfully; exiting.")
        return

    # plot loss comparison
    plt.figure(figsize=(8, 5))
    for name, data in results.items():
        plt.plot(data["iteration"], data["loss"], label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Helmholtz training loss vs iteration (k={args.helmholtz_k:g})")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    loss_plot_path = os.path.join(comparison_dir, "training_loss_comparison.png")
    plt.savefig(loss_plot_path, dpi=200)
    plt.close()
    print(f"Saved training loss comparison plot to {loss_plot_path}")

    # summary json
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

    # save raw metrics
    for name, data in results.items():
        np.savez(os.path.join(comparison_dir, f"{name}_metrics.npz"), **data)

    # ------------------------------------------------------------------
    # Per-solver solution comparison images with unified scales
    # ------------------------------------------------------------------
    solver_solutions: Dict[str, np.ndarray] = {}
    global_min = float("inf")
    global_max = float("-inf")
    global_error_max = float("-inf")
    true_solution_cache: Dict[int, np.ndarray] = {}

    for name, data in results.items():
        run_dir = data["run_dir"]
        sol_file = "precond_solution.npy" if name == "precond" else f"{name}_solution.npy"
        sol_path = os.path.join(run_dir, sol_file)
        if not os.path.exists(sol_path):
            print(f"Warning: expected solution file {sol_path} not found; skipping plot for {name}.")
            continue
        sol = np.load(sol_path)
        solver_solutions[name] = sol
        global_min = min(global_min, float(sol.min()))
        global_max = max(global_max, float(sol.max()))

        N = sol.shape[0]
        if N not in true_solution_cache:
            xs = np.linspace(0.0, 1.0, N)
            X, Y = np.meshgrid(xs, xs)
            true_solution_cache[N] = np.sin(np.pi * X) * np.sin(np.pi * Y)

        true_sol = true_solution_cache[N]
        err = np.abs(sol - true_sol)
        global_error_max = max(global_error_max, float(err.max()))

    for name, sol in solver_solutions.items():
        N = sol.shape[0]
        true_sol = true_solution_cache[N]
        error = np.abs(sol - true_sol)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        im0 = axes[0].imshow(true_sol, extent=[0, 1, 0, 1], origin="lower",
                             vmin=global_min, vmax=global_max, cmap="viridis")
        axes[0].set_title("Exact solution")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(sol, extent=[0, 1, 0, 1], origin="lower",
                             vmin=global_min, vmax=global_max, cmap="viridis")
        axes[1].set_title(f"{name.capitalize()} solution")
        fig.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(error, extent=[0, 1, 0, 1], origin="lower",
                             vmin=0.0, vmax=global_error_max, cmap="magma")
        axes[2].set_title("Absolute error")
        fig.colorbar(im2, ax=axes[2])

        for ax in axes:
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        fig.tight_layout()
        img_path = os.path.join(comparison_dir, f"{name}_solution_comparison.png")
        plt.savefig(img_path, dpi=200)
        plt.close(fig)
        print(f"Saved solution comparison plot for {name} to {img_path}")

    # ------------------------------------------------------------------
    # Single combined figure: one row per solver in order:
    # GD, Adam, Preconditioning, ENGD
    # ------------------------------------------------------------------
    order = ["gd", "adam", "precond", "engd"]
    pretty = {"gd": "Gradient Descent", "adam": "Adam", "precond": "Preconditioning", "engd": "Energy Natural GD"}

    rows = [n for n in order if n in solver_solutions]
    if rows:
        fig, axes = plt.subplots(len(rows), 3, figsize=(12, 4 * len(rows)))

        if len(rows) == 1:
            axes = np.expand_dims(axes, axis=0)

        axes[0, 0].set_title("Exact solution")
        axes[0, 1].set_title("Model prediction")
        axes[0, 2].set_title("Absolute error")

        im_exact = im_pred = im_err = None
        for r, name in enumerate(rows):
            sol = solver_solutions[name]
            N = sol.shape[0]
            true_sol = true_solution_cache[N]
            error = np.abs(sol - true_sol)

            im_exact = axes[r, 0].imshow(
                true_sol, extent=[0, 1, 0, 1], origin="lower",
                vmin=global_min, vmax=global_max, cmap="viridis"
            )
            im_pred = axes[r, 1].imshow(
                sol, extent=[0, 1, 0, 1], origin="lower",
                vmin=global_min, vmax=global_max, cmap="viridis"
            )
            im_err = axes[r, 2].imshow(
                error, extent=[0, 1, 0, 1], origin="lower",
                vmin=0.0, vmax=global_error_max, cmap="magma"
            )

            axes[r, 0].set_ylabel(f"{pretty.get(name, name)}\n$y$")
            for c in range(3):
                axes[r, c].set_xlabel("$x$")

        fig.colorbar(im_exact, ax=axes[:, 0], fraction=0.02, pad=0.02)
        fig.colorbar(im_pred,  ax=axes[:, 1], fraction=0.02, pad=0.02)
        fig.colorbar(im_err,   ax=axes[:, 2], fraction=0.02, pad=0.02)

        fig.tight_layout()
        combined_path = os.path.join(comparison_dir, "all_solutions_comparison.png")
        plt.savefig(combined_path, dpi=200)
        plt.close(fig)
        print(f"Saved combined solution comparison plot to {combined_path}")
    else:
        print("No solution arrays found; skipping combined solution plot.")


if __name__ == "__main__":
    main()