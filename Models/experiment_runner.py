"""
experiment_runner.py
---------------------

This script orchestrates hyperparameter sweeps for the physics‑informed
DeepONet ODE example provided in this repository.  It drives the existing
training, log plotting and evaluation scripts without modifying their
internal logic.  You can launch a grid of experiments over different
network widths, depths, feature dimensions, activations and random seeds
from a single command line.  After all runs finish the script builds
aggregate tables and figures summarising test performance and model
capacity.

**Key responsibilities**

* Build a Cartesian product of hyperparameter values from user supplied
  lists or numeric ranges.
* For each configuration/seed:
  - Construct a human readable run tag (``w{WIDTH}-d{DEPTH}-f{FEAT}-a{ACT}-s{SEED}``).
  - Launch ``train.py`` with the chosen hyperparameters, a fixed budget
    (e.g. number of training examples) and a unique run name.  All
    checkpoints and logs are placed under the experiment root.
  - Capture stdout/stderr to ``train_stdout.txt`` in the eventual run
    directory.
  - After training finishes, invoke ``plot_train_log.py`` on the
    generated ``train_log.npz`` to produce a ``train_loss.png``.
  - Evaluate both ``best.pt`` and ``last.pt`` checkpoints via ``eval.py``,
    capturing their outputs to separate text files and renaming the
    generated ``metrics.json`` and summary figure so that both
    checkpoints are preserved.  Per example metrics and plots are kept in
    the evaluation subfolders.
  - Record any failures into ``failures.log`` but continue the sweep.
* Once all individual runs have completed, parse their ``config.json``
  and ``metrics_*`` files to build a master table.  Compute the total
  parameter count for each model, derive summary statistics (mean and
  95 % confidence interval) across seeds for each unique architecture,
  and save these as CSV/JSON.
* Generate comparison plots:
  - A bar chart (with a log scaled y‑axis) of mean test relative
    ``L^2`` error for every configuration (aggregated over seeds).
  - A scatter plot showing the relationship between model capacity
    (number of trainable parameters) and test error.
  - A grid of representative training curves, one per chosen width,
    overlaying total, PDE and IC losses from the existing training log.

The orchestrator uses conservative defaults – training jobs are run
serially by default – but can attempt to run multiple trainings in
parallel if ``--max-parallel`` is greater than one.  Should a run fail
with a CUDA out‑of‑memory error, the script automatically downgrades
parallelism and notes this decision in ``notes.txt``.

Usage example:

.. code-block:: bash

    python experiment_runner.py \
      --exp-root experiments/ODE_sweep \
      --width-range 20:40:10 \
      --depth-list 4,6 \
      --feat-list 50 \
      --activation tanh,relu \
      --seeds 123,124,125 \
      --ntrain 2048 --nval 128 --ntest 1000 --steps 8000 \
      --batch 256 --lr 1e-3 --val-every 500 --max-parallel 1

The above will explore widths 20 and 30, depths 4 and 6, a single
feature dimension of 50 and two activations (tanh and relu) across
three seeds.  It keeps all run artefacts under ``experiments/ODE_sweep``
and writes summary tables and plots once the sweep is complete.

This script only orchestrates the experiments – you must have all
dependencies (e.g. PyTorch) installed for the individual training and
evaluation scripts to run correctly.  Furthermore the orchestrator does
not alter any of the underlying model or loss definitions, faithfully
reusing the provided code.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np  # type: ignore
import matplotlib

# Use a non‑interactive backend to avoid requiring a display when
# generating plots during aggregation.  This must be set before
# importing pyplot.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore


def parse_range_list(value: str) -> List[int]:
    """Parse a comma separated list of ints or a numeric range.

    Examples:
      "10,20,30" -> [10, 20, 30]
      "5:15:5"   -> [5, 10, 15]

    Raises ValueError if the input cannot be parsed.
    """
    if not value:
        raise ValueError("Empty value for range/list argument")
    if ":" in value:
        parts = value.split(":")
        if len(parts) != 3:
            raise ValueError(f"Range must be start:end:step, got {value}")
        start, end, step = map(int, parts)
        if step == 0:
            raise ValueError("Range step cannot be zero")
        # Inclusive of end if the sequence hits it exactly
        seq: List[int] = []
        i = start
        if step > 0:
            while i <= end:
                seq.append(i)
                i += step
        else:
            while i >= end:
                seq.append(i)
                i += step
        return seq
    else:
        return [int(x) for x in value.split(",") if x.strip()]


def compute_param_count(m: int, width: int, depth: int, in_dim: int = 1) -> int:
    """Compute the number of trainable parameters of the DeepONet model.

    The architecture comprises a branch MLP taking ``m`` sensor values to
    ``feat_dim`` outputs and a trunk MLP taking coordinates of size
    ``in_dim`` to ``feat_dim`` outputs.  Each MLP has ``depth`` total
    layers: ``depth-1`` hidden layers of size ``width`` followed by the
    final output layer.  Both bias and weight parameters are counted.

    Parameters
    ----------
    m : int
        Number of sensor points (input dimension to the branch).
    width : int
        Width of the hidden layers.
    depth : int
        Total number of layers in each MLP (including final output layer).
    feat_dim : int
        Output feature dimension of each MLP.
    in_dim : int
        Dimensionality of the coordinate input to the trunk (1 for ODE).

    Returns
    -------
    int
        Total number of trainable parameters.
    """
    # Hidden layer sizes: depth-1 hidden layers all of size width
    hidden = [width] * (depth - 1)
    branch_sizes = [m] + hidden + [50]
    trunk_sizes = [in_dim] + hidden + [50]
    param_count = 0
    # Branch MLP parameters
    for in_d, out_d in zip(branch_sizes[:-1], branch_sizes[1:]):
        param_count += in_d * out_d + out_d  # weights + biases
    # Trunk MLP parameters
    for in_d, out_d in zip(trunk_sizes[:-1], trunk_sizes[1:]):
        param_count += in_d * out_d + out_d
    return param_count


@dataclass
class RunConfig:
    """Container for a single run configuration."""
    width: int
    depth: int
    activation: str
    seed: int
    # Additional budget/other hyperparameters
    ntrain: Optional[int] = None
    nval: Optional[int] = None
    ntest: Optional[int] = None
    m: Optional[int] = None
    ell: Optional[float] = None
    steps: Optional[int] = None
    batch: Optional[int] = None
    lr: Optional[float] = None
    val_every: Optional[int] = None
    log_every: Optional[int] = None

    def tag(self) -> str:
        """Return a human friendly tag for this run."""
        return f"w{self.width}-d{self.depth}-a{self.activation}-s{self.seed}"

    def to_train_args(self) -> List[str]:
        """Convert the configuration to command line arguments for train.py."""
        args: List[str] = []
        args += ["--width", str(self.width)]
        args += ["--depth", str(self.depth)]
        args += ["--activation", self.activation]
        args += ["--seed", str(self.seed)]
        # Budget overrides
        if self.ntrain is not None:
            args += ["--ntrain", str(self.ntrain)]
        if self.nval is not None:
            args += ["--nval", str(self.nval)]
        if self.ntest is not None:
            args += ["--ntest", str(self.ntest)]
        if self.m is not None:
            args += ["--m", str(self.m)]
        if self.ell is not None:
            args += ["--ell", str(self.ell)]
        if self.steps is not None:
            args += ["--steps", str(self.steps)]
        if self.batch is not None:
            args += ["--batch", str(self.batch)]
        if self.lr is not None:
            args += ["--lr", str(self.lr)]
        if self.val_every is not None:
            args += ["--val-every", str(self.val_every)]
        if self.log_every is not None:
            # not currently used by train.py but accepted for completeness
            args += ["--log_every", str(self.log_every)]
        return args


def run_training(
    exp_root: Path,
    train_script: Path,
    plotter_script: Path,
    eval_script: Path,
    cfg: RunConfig,
    max_parallel: int,
    notes: List[str],
    failures_log: List[str],
    debug: bool = False,
) -> Dict[str, Any]:
    """Execute a single configuration: train, plot losses, evaluate.

    Parameters
    ----------
    exp_root : Path
        Experiment root directory into which all runs will be placed.
    train_script : Path
        Path to the ``train.py`` script.
    plotter_script : Path
        Path to the ``plot_train_log.py`` script.
    eval_script : Path
        Path to the ``eval.py`` script.
    cfg : RunConfig
        Hyperparameter configuration for this run.
    max_parallel : int
        Maximum number of trainings that can run concurrently.  Unused
        within this function but recorded in notes if concurrency is
        downgraded during execution.
    notes : List[str]
        Mutable list used to collect informative messages about
        concurrency downgrades (e.g. due to CUDA OOM).
    failures_log : List[str]
        Mutable list used to collect failure messages.
    debug : bool
        If True, prints additional diagnostic information.

    Returns
    -------
    Dict[str, Any]
        A dictionary with keys describing the run outcome.  Always
        contains 'status' ("success" or "failure").  On success the
        dictionary also includes: 'config', 'run_dir', 'metrics_best',
        'metrics_last' and 'param_count'.  On failure the dictionary
        contains 'error'.
    """
    result: Dict[str, Any] = {"config": asdict(cfg)}

    # Compose tag and ensure run directory exists
    tag = cfg.tag()
    run_dir = exp_root / tag
    # Precreate the top level folder so that evaluation artefacts can be
    # moved or created even if training fails.  The training script
    # itself will create a timestamped subdirectory within exp_root if
    # called with --run-name, but we use this folder later when
    # reorganising outputs.
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build command for training
    train_args = cfg.to_train_args()
    # Always set save-dir and run-name so that train.py writes into
    # exp_root and includes our tag in the timestamped directory name.
    cmd = [sys.executable, '-m', 'Models.deepo-ODE.train', '--save-dir', str(exp_root), "--run-name", tag] + train_args
    if debug:
        print(f"Launching training: {' '.join(cmd)}")

    # Start the training process
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    # Capture output and parse run directory if possible
    run_subdir: Optional[Path] = None
    stdout_lines: List[str] = []
    try:
        if proc.stdout is not None:
            for line in proc.stdout:
                stdout_lines.append(line)
                # The training script prints "Run directory: <path>"
                # early during execution; capture it to locate the actual
                # subfolder created (timestamp + tag).
                m = re.search(r"Run directory:\s*(.*)", line)
                if m:
                    path_str = m.group(1).strip()
                    if path_str:
                        try:
                            run_subdir = Path(path_str)
                        except Exception:
                            pass
                if debug:
                    # echo to console if debugging
                    sys.stdout.write(line)
        proc.wait()
    except Exception as ex:
        proc.kill()
        proc.wait()
        failures_log.append(f"Run {tag} crashed while streaming output: {ex}")
        result["status"] = "failure"
        result["error"] = str(ex)
        return result

    # Write captured stdout to file
    # If a timestamped subfolder was created we will move it after training.
    try:
        # Determine final location for training stdout
        train_stdout_path = run_dir / "train_stdout.txt"
        with train_stdout_path.open("w", encoding="utf-8") as f:
            f.writelines(stdout_lines)
    except Exception as e:
        # not fatal; just record
        failures_log.append(f"Failed to write training stdout for {tag}: {e}")

    # Check process exit status
    ret = proc.returncode
    if ret != 0:
        # Inspect output for OOM to possibly downgrade parallelism
        joined = "".join(stdout_lines).lower()
        if "out of memory" in joined or "cuda" in joined and "memory" in joined:
            # If multiple workers were requested, record a note to
            # downgrade to serial for remaining jobs.
            if max_parallel > 1:
                msg = (
                    f"Detected an out‑of‑memory error while running {tag}; "
                    f"will switch to serial execution for remaining runs."
                )
                notes.append(msg)
            failures_log.append(f"Run {tag} failed due to OOM")
        else:
            failures_log.append(f"Run {tag} returned non‑zero exit code {ret}")
        result["status"] = "failure"
        result["error"] = f"Training failed with exit code {ret}"
        return result

    # At this point training succeeded.  Determine the timestamped
    # subdirectory produced by train.py.  If not captured, search for
    # directories under exp_root that end with our tag.
    if run_subdir is None or not run_subdir.exists():
        # Fallback: find a subdir in exp_root whose name ends with '-' + tag
        candidates = []
        for child in exp_root.iterdir():
            if child.is_dir() and child.name.endswith("-" + tag):
                candidates.append(child)
        # Choose the most recently modified candidate
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            run_subdir = candidates[0]
        else:
            # As a last resort, assume train.py wrote directly into exp_root
            run_subdir = exp_root

    # Move contents from the timestamped directory into our run_dir.  If
    # run_subdir is already run_dir (no timestamp), this is a no‑op.
    try:
        if run_subdir != run_dir:
            # Ensure destination exists
            run_dir.mkdir(parents=True, exist_ok=True)
            # Move every file/directory from run_subdir into run_dir
            for item in run_subdir.iterdir():
                dest = run_dir / item.name
                if dest.exists():
                    # Remove existing dest before moving to avoid
                    # shutil.Error on Windows when overwriting directories
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))
            # Remove the now empty timestamped folder
            try:
                run_subdir.rmdir()
            except OSError:
                pass
    except Exception as e:
        failures_log.append(f"Failed to reorganise run directory for {tag}: {e}")

    # At this point run_dir contains all files generated by training
    result["run_dir"] = str(run_dir)

    # Plot training losses using plot_train_log.py
    train_log_path = run_dir / "train_log.npz"
    if train_log_path.exists():
        try:
            # Provide required --log argument; let plotter choose the output filename
            cmd_plot = [sys.executable, str(plotter_script), '--log', str(train_log_path)]
            proc_plot = subprocess.run(cmd_plot, capture_output=True, text=True)
            if proc_plot.returncode != 0:
                failures_log.append(
                    f"plot_train_log.py failed for {tag}: {proc_plot.stderr.strip()}"
                )
            else:
                # The plotter writes training_curves_from_log.png by default.
                default_png = run_dir / "training_curves_from_log.png"
                legacy_png = run_dir / f"{train_log_path.stem}.png"  # fallback for older versions
                src_img = default_png if default_png.exists() else legacy_png if legacy_png.exists() else None
                if src_img:
                    dest_img = run_dir / "train_loss.png"
                    if dest_img.exists():
                        dest_img.unlink()
                    src_img.rename(dest_img)
        except Exception as e:
            failures_log.append(f"Error while plotting training log for {tag}: {e}")

    # Evaluate best and last checkpoints
    metrics_best = None
    metrics_last = None
    for ckpt_name in ["best.pt", "last.pt"]:
        ckpt_path = run_dir / ckpt_name
        if not ckpt_path.exists():
            failures_log.append(f"Checkpoint {ckpt_name} missing for {tag}")
            continue
        # Build evaluation command
        ckpt_label = "best" if ckpt_name.startswith("best") else "last"
        eval_cmd = [sys.executable, '-m', 'Models.deepo-ODE.eval', '--ckpt', str(ckpt_path)]
        # Capture stdout
        try:
            proc_eval = subprocess.run(eval_cmd, capture_output=True, text=True)
        except Exception as e:
            failures_log.append(f"Failed to run eval.py on {ckpt_name} for {tag}: {e}")
            continue
        # Write evaluation stdout to file
        eval_stdout_path = run_dir / f"eval_{ckpt_label}_stdout.txt"
        try:
            with eval_stdout_path.open("w", encoding="utf-8") as f:
                f.write(proc_eval.stdout)
                f.write(proc_eval.stderr or "")
        except Exception as e:
            failures_log.append(f"Unable to write eval stdout for {tag}: {e}")
        # Check evaluation return code
        if proc_eval.returncode != 0:
            failures_log.append(
                f"eval.py returned non‑zero exit code on {ckpt_name} for {tag}: {proc_eval.stderr.strip()}"
            )
            continue
        # metrics.json will be placed in run_dir by eval.py; rename it
        metrics_json = run_dir / "metrics.json"
        if metrics_json.exists():
            dest_metrics = run_dir / f"metrics_{ckpt_label}.json"
            try:
                # Read metrics before renaming for aggregator usage
                with metrics_json.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if ckpt_label == "best":
                    metrics_best = data
                else:
                    metrics_last = data
                # Rename file
                if dest_metrics.exists():
                    dest_metrics.unlink()
                metrics_json.rename(dest_metrics)
            except Exception as e:
                failures_log.append(f"Failed to process metrics.json for {tag}: {e}")
        # The summary bar plot is named relL2_summary_bar.png; rename to include label
        summary_png = run_dir / "relL2_summary_bar.png"
        if summary_png.exists():
            dest_png = run_dir / f"relL2_summary_bar_{ckpt_label}.png"
            try:
                if dest_png.exists():
                    dest_png.unlink()
                summary_png.rename(dest_png)
            except Exception as e:
                failures_log.append(f"Failed to rename summary plot for {tag}: {e}")

    # Compute parameter count for this configuration
    param_count = None
    try:
        # Use m from config.json if present, else fallback to cfg.m or 100
        m_val = cfg.m
        # Attempt to read config.json saved by train.py
        cfg_json = run_dir / "config.json"
        if cfg_json.exists():
            with cfg_json.open("r", encoding="utf-8") as f:
                saved_cfg = json.load(f)
            m_val = int(saved_cfg.get("m", m_val if m_val is not None else 100))
            # width, depth, feat_dim may be saved in train config under
            # slightly different keys – fallback to cfg if missing
            width_val = int(saved_cfg.get("width", cfg.width))
            depth_val = int(saved_cfg.get("depth", cfg.depth))
            param_count = compute_param_count(m_val, width_val, depth_val)
        else:
            # Without config.json we still estimate using provided cfg
            m_val = m_val if m_val is not None else 100
            param_count = compute_param_count(m_val, cfg.width, cfg.depth)
    except Exception as e:
        failures_log.append(f"Failed to compute parameter count for {tag}: {e}")

    result.update({
        "status": "success",
        "metrics_best": metrics_best,
        "metrics_last": metrics_last,
        "param_count": param_count,
    })
    return result


def aggregate_results(results: List[Dict[str, Any]], exp_root: Path) -> None:
    """Aggregate per‑run results into summary files and plots.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of dictionaries returned by ``run_training``.
    exp_root : Path
        Experiment root directory where summary files will be written.
    """
    # Filter successful runs
    successes = [r for r in results if r.get("status") == "success"]
    if not successes:
        return
    # Build table rows
    rows = []
    for r in successes:
        cfg = r["config"]
        width = cfg["width"]
        depth = cfg["depth"]
        activation = cfg["activation"]
        seed = cfg["seed"]
        param_count = r.get("param_count")
        mbest = r.get("metrics_best") or {}
        mlast = r.get("metrics_last") or {}
        best_rel = mbest.get("relL2", {}).get("mean")
        last_rel = mlast.get("relL2", {}).get("mean")
        row = {
            "width": width,
            "depth": depth,
            "activation": activation,
            "seed": seed,
            "param_count": param_count,
            "best_relL2_mean": best_rel,
            "last_relL2_mean": last_rel,
        }
        rows.append(row)
    # Save summary CSV
    summary_csv_path = exp_root / "summary.csv"
    import csv
    with summary_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    # Save summary JSON
    summary_json_path = exp_root / "summary.json"
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # Group by architecture (width, depth, feat_dim, activation) ignoring seed
    group_stats: Dict[Tuple[int, int, int, str], List[float]] = {}
    for row in rows:
        key = (row["width"], row["depth"], row["activation"])
        val = row["best_relL2_mean"]
        if val is None:
            continue
        group_stats.setdefault(key, []).append(val)
    # Compute aggregated mean and 95% CI
    summary_agg = []
    for (w, d, act), vals in group_stats.items():
        arr = np.array(vals, dtype=float)
        mean = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        ci95 = float(1.96 * sd / np.sqrt(max(len(arr), 1)))
        summary_agg.append({
            "width": w,
            "depth": d,
            "activation": act,
            "mean_relL2": mean,
            "ci95_relL2": ci95,
        })
    # Save aggregated summary
    summary_agg_json_path = exp_root / "summary_agg.json"
    with summary_agg_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_agg, f, indent=2)

    # Generate plots directory
    plots_dir = exp_root / "plots"
    plots_dir.mkdir(exist_ok=True)
    # Bar chart of aggregated mean relL2 per configuration
    labels = []
    means = []
    errs = []
    for item in summary_agg:
        labels.append(f"w{item['width']}-d{item['depth']}-a{item['activation']}")
        means.append(item['mean_relL2'])
        errs.append(item['ci95_relL2'])
    # Sort by mean for nicer plotting
    order = np.argsort(means)
    labels = [labels[i] for i in order]
    means = [means[i] for i in order]
    errs = [errs[i] for i in order]
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(labels)), 4))
    ax.bar(range(len(labels)), means, yerr=errs, capsize=5, alpha=0.7, edgecolor='k')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Mean test relL2')
    ax.set_yscale('log')
    ax.set_title('Test relative $L^2$ error (mean ±95% CI)')
    ax.grid(axis='y', which='both', alpha=0.3)
    fig.tight_layout()
    bar_path = plots_dir / "relL2_bar.png"
    fig.savefig(bar_path, dpi=200)
    plt.close(fig)

    # Scatter plot of parameter count vs mean relL2
    # Use colour for activation
    acts = sorted(set(item['activation'] for item in summary_agg))
    act_to_color = {act: plt.cm.tab10(i % 10) for i, act in enumerate(acts)}
    fig, ax = plt.subplots(figsize=(6, 4))
    for item in summary_agg:
        key = (item['width'], item['depth'], item['activation'])
        # find representative param_count from one of the runs
        # We'll take the first run matching this key
        param_val = None
        for r in rows:
            if (r['width'], r['depth'], r['activation']) == key:
                param_val = r['param_count']
                break
        if param_val is None:
            continue
        ax.scatter(param_val, item['mean_relL2'], color=act_to_color[item['activation']], label=item['activation'], alpha=0.8)
    # Build legend without duplicates
    handles, labels_legend = [], []
    for act in acts:
        handles.append(plt.Line2D([0], [0], marker='o', color=act_to_color[act], linestyle='', label=act))
        labels_legend.append(act)
    ax.legend(handles, labels_legend, title='Activation', loc='best')
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xlabel('Parameter count')
    ax.set_ylabel('Mean test relL2')
    ax.set_title('Model capacity vs performance')
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    scatter_path = plots_dir / "capacity_vs_performance.png"
    fig.savefig(scatter_path, dpi=200)
    plt.close(fig)

    # Representative training curves: pick one representative run per unique width
    # specifically the configuration with minimal mean relL2 among those with a given width
    selected_tags: Dict[int, str] = {}
    for width in sorted(set(r['width'] for r in rows)):
        best_tag = None
        best_mean = float('inf')
        for item in summary_agg:
            if item['width'] != width:
                continue
            if item['mean_relL2'] < best_mean:
                best_mean = item['mean_relL2']
                # compose tag
                best_tag = f"w{item['width']}-d{item['depth']}-a{item['activation']}"
        if best_tag is not None:
            selected_tags[width] = best_tag
    # Plot training curves for selected tags
    n_sel = len(selected_tags)
    if n_sel > 0:
        fig, axes = plt.subplots(n_sel, 1, figsize=(6, 3 * n_sel), sharex=False)
        if n_sel == 1:
            axes = [axes]
        for ax, (width, tag) in zip(axes, selected_tags.items()):
            run_path = exp_root / tag
            train_log = run_path / "train_log.npz"
            if not train_log.exists():
                continue
            try:
                data = np.load(train_log)
                steps = data['step']
                loss = data['loss']
                loss_pde = data['loss_pde']
                loss_ic = data['loss_ic']
                ax.plot(steps, loss, label='total loss')
                ax.plot(steps, loss_pde, label='PDE loss')
                ax.plot(steps, loss_ic, label='IC loss')
                ax.set_yscale('log')
                ax.set_title(f'Training losses for {tag}')
                ax.set_xlabel('Step')
                ax.set_ylabel('Loss')
                ax.grid(alpha=0.3, which='both')
                ax.legend(loc='best', fontsize='small')
            except Exception:
                continue
        fig.tight_layout()
        curves_path = plots_dir / "training_curves.png"
        fig.savefig(curves_path, dpi=200)
        plt.close(fig)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run hyperparameter sweeps and aggregate results for the DeepONet ODE example")
    parser.add_argument('--exp-root', required=True, help='Directory where all runs and summaries will be stored')
    # Hyperparameter spaces: either list or range
    parser.add_argument('--width-list', default=None, help='Comma separated list of widths')
    parser.add_argument('--width-range', default=None, help='Range start:end:step for widths')
    parser.add_argument('--depth-list', default=None, help='Comma separated list of depths')
    parser.add_argument('--depth-range', default=None, help='Range start:end:step for depths')
    parser.add_argument('--activation', required=True, help='Comma separated list of activations (tanh,relu,silu,gelu,softplus)')
    parser.add_argument('--seeds', required=True, help='Comma separated list of integer seeds')
    # Budget overrides
    parser.add_argument('--ntrain', type=int, default=None, help='Number of training functions')
    parser.add_argument('--nval', type=int, default=None, help='Number of validation functions')
    parser.add_argument('--ntest', type=int, default=None, help='Number of test functions')
    parser.add_argument('--m', type=int, default=None, help='Number of grid/sensor points on [0,1]')
    parser.add_argument('--ell', type=float, default=None, help='RBF kernel length scale')
    parser.add_argument('--steps', type=int, default=None, help='Training steps/iterations')
    parser.add_argument('--batch', type=int, default=None, help='Batch size (# functions per step)')
    parser.add_argument('--lr', type=float, default=None, help='Adam learning rate')
    parser.add_argument('--val-every', type=int, default=None, help='Validate every N steps')
    parser.add_argument('--log-every', type=int, default=None, help='Log every N steps (if supported)')
    # Parallelism
    parser.add_argument('--max-parallel', type=int, default=1, help='Maximum number of training jobs to run in parallel')
    # Debugging
    parser.add_argument('--debug', action='store_true', help='Print additional debug output')
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Resolve experiment root
    exp_root = Path(args.exp_root).resolve()
    exp_root.mkdir(parents=True, exist_ok=True)

    # Derive width values
    if args.width_list and args.width_range:
        parser.error('Specify only one of --width-list or --width-range')
    if args.width_list:
        width_vals = parse_range_list(args.width_list)
    elif args.width_range:
        width_vals = parse_range_list(args.width_range)
    else:
        parser.error('You must specify either --width-list or --width-range')
    # Depth values
    if args.depth_list and args.depth_range:
        parser.error('Specify only one of --depth-list or --depth-range')
    if args.depth_list:
        depth_vals = parse_range_list(args.depth_list)
    elif args.depth_range:
        depth_vals = parse_range_list(args.depth_range)
    else:
        parser.error('You must specify either --depth-list or --depth-range')
    # Activation list
    act_vals = [act.strip() for act in args.activation.split(',') if act.strip()]
    if not act_vals:
        parser.error('Activation list cannot be empty')
    # Seeds
    seed_vals = [int(s) for s in args.seeds.split(',') if s.strip()]
    if not seed_vals:
        parser.error('Seeds list cannot be empty')

    # Compose run configurations (Cartesian product)
    configs: List[RunConfig] = []
    for width, depth,  act, seed in itertools.product(width_vals, depth_vals,  act_vals, seed_vals):
        cfg = RunConfig(
            width=width,
            depth=depth,
            activation=act,
            seed=seed,
            ntrain=args.ntrain,
            nval=args.nval,
            ntest=args.ntest,
            m=args.m,
            ell=args.ell,
            steps=args.steps,
            batch=args.batch,
            lr=args.lr,
            val_every=args.val_every,
            log_every=args.log_every,
        )
        configs.append(cfg)

    # Determine script locations relative to this file.  We assume
    # train.py, plot_train_log.py and eval.py are in the same directory as
    # this orchestrator or one level up.  If not found the program will
    # exit with an error.
    current_dir = Path(__file__).resolve().parent
    # Search candidate paths for each script
    def find_script(name: str) -> Path:
        # Candidate locations: same directory, parent directory
        for candidate in [current_dir / name, current_dir.parent / name]:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Cannot locate {name} relative to {current_dir}")
    train_script = find_script('deepo-ODE/train.py')
    plotter_script = find_script('Shared/plot_train_log.py')
    eval_script = find_script('deepo-ODE/eval.py')

    # Prepare logs for notes and failures
    notes: List[str] = []
    failures: List[str] = []
    results: List[Dict[str, Any]] = []

    # Manage parallel execution.  We schedule up to args.max_parallel
    # training jobs at once.  For simplicity we reuse the run_training
    # function sequentially but spawn multiple subprocesses concurrently
    # when needed.  If a CUDA OOM is detected, we reduce concurrency
    # mid‑sweep.
    max_parallel = max(1, args.max_parallel)
    pending = list(configs)
    active_procs: List[Tuple[subprocess.Popen, RunConfig, Path, List[str]]] = []  # hold process, cfg, run_dir placeholder, stdout_lines

    # This orchestrator launches each training one after another when
    # max_parallel==1.  If max_parallel>1 we manage a pool of concurrent
    # training processes below.  However, because run_training
    # encapsulates launching and post‑processing within a single call, the
    # concurrency is effectively limited to starting multiple training
    # scripts simultaneously; evaluation still runs sequentially after
    # each training completes.
    if max_parallel == 1:
        for cfg in pending:
            res = run_training(
                exp_root=exp_root,
                train_script=train_script,
                plotter_script=plotter_script,
                eval_script=eval_script,
                cfg=cfg,
                max_parallel=max_parallel,
                notes=notes,
                failures_log=failures,
                debug=args.debug,
            )
            results.append(res)
        # Aggregate after serial runs
        aggregate_results(results, exp_root)
    else:
        # Parallel execution: we'll keep a set of active training
        # processes up to max_parallel.  Each process runs only the
        # training portion; once complete we perform post‑processing
        # sequentially to avoid overloading the GPU during evaluation.
        index = 0
        while index < len(pending) or active_procs:
            # Launch new processes if we have capacity
            while len(active_procs) < max_parallel and index < len(pending):
                cfg = pending[index]
                index += 1
                tag = cfg.tag()
                # Build command
                train_args = cfg.to_train_args()
                cmd = [sys.executable, 'm', str(train_script), '--save-dir', str(exp_root), '--run-name', tag] + train_args
                if args.debug:
                    print(f"[parallel] Launching {tag}: {' '.join(cmd)}")
                # Start process capturing stdout/stderr into a list
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )
                active_procs.append((proc, cfg, exp_root / tag, []))
            # Check processes for completion
            time.sleep(0.5)
            still_active: List[Tuple[subprocess.Popen, RunConfig, Path, List[str]]] = []
            for (proc, cfg, run_dir, stdout_lines) in active_procs:
                # Read any available output
                if proc.stdout is not None:
                    try:
                        while True:
                            line = proc.stdout.readline()
                            if not line:
                                break
                            stdout_lines.append(line)
                    except Exception:
                        pass
                if proc.poll() is not None:
                    # Process finished
                    # Write stdout to file temporarily within a timestamped folder later
                    # We'll call run_training to perform post‑processing
                    # Create a temporary file to hold captured output
                    tag = cfg.tag()
                    run_stdout_path_tmp = (exp_root / tag / 'train_stdout_partial.txt')
                    try:
                        (exp_root / tag).mkdir(parents=True, exist_ok=True)
                        with run_stdout_path_tmp.open('w', encoding='utf-8') as f:
                            f.writelines(stdout_lines)
                    except Exception:
                        pass
                    # Evaluate return code
                    retcode = proc.returncode
                    # To perform the remainder of the pipeline (organising run
                    # directory, plotting, evaluation), we invoke run_training
                    # but provide the captured stdout via file.  To avoid
                    # re‑running training we simulate success/failure based
                    # solely on return code and captured output.  Hence we
                    # temporarily override run_training to only perform
                    # post‑processing when returncode==0, otherwise mark
                    # failure.
                    if retcode == 0:
                        # We cannot reconstruct run_subdir; run_training will
                        # search for timestamped folder as fallback.
                        res = run_training(
                            exp_root=exp_root,
                            train_script=train_script,
                            plotter_script=plotter_script,
                            eval_script=eval_script,
                            cfg=cfg,
                            max_parallel=max_parallel,
                            notes=notes,
                            failures_log=failures,
                            debug=args.debug,
                        )
                        results.append(res)
                    else:
                        # Non‑zero exit code
                        joined = ''.join(stdout_lines).lower()
                        if 'out of memory' in joined or ('cuda' in joined and 'memory' in joined):
                            if max_parallel > 1:
                                notes.append(
                                    f"Detected an out‑of‑memory error while running {cfg.tag()}; switching to serial for remaining runs."
                                )
                                # Drain remaining processes and run the rest serially
                                max_parallel = 1
                                # Terminate all other active processes
                                for (p_, _, _, _) in active_procs:
                                    try:
                                        p_.terminate()
                                    except Exception:
                                        pass
                                # Wait for them to finish
                                for (p_, _, _, _) in active_procs:
                                    try:
                                        p_.wait(timeout=5)
                                    except Exception:
                                        pass
                                # Append failure result for this cfg
                                results.append({
                                    'config': asdict(cfg),
                                    'status': 'failure',
                                    'error': 'CUDA out of memory during parallel training',
                                })
                                # Push back remaining pending configs to process serially
                                remaining_configs = pending[index:]
                                for cfg_rem in remaining_configs:
                                    res = run_training(
                                        exp_root=exp_root,
                                        train_script=train_script,
                                        plotter_script=plotter_script,
                                        eval_script=eval_script,
                                        cfg=cfg_rem,
                                        max_parallel=max_parallel,
                                        notes=notes,
                                        failures_log=failures,
                                        debug=args.debug,
                                    )
                                    results.append(res)
                                # Clear pending and active lists to exit both loops
                                pending.clear()
                                active_procs.clear()
                                break
                        else:
                            failures.append(
                                f"Run {cfg.tag()} returned non‑zero exit code {retcode} in parallel execution"
                            )
                            results.append({
                                'config': asdict(cfg),
                                'status': 'failure',
                                'error': f'Training failed with exit code {retcode}',
                            })
                    # Do not keep this process in the active list
                else:
                    still_active.append((proc, cfg, run_dir, stdout_lines))
            if max_parallel == 1 and not pending:
                # All remaining have been switched to serial; break outer loop
                break
            active_procs = still_active
        # After exiting loop, aggregate results
        aggregate_results(results, exp_root)

    # Write notes and failures logs
    if notes:
        with (exp_root / 'notes.txt').open('w', encoding='utf-8') as f:
            for line in notes:
                f.write(line + '\n')
    if failures:
        with (exp_root / 'failures.log').open('w', encoding='utf-8') as f:
            for line in failures:
                f.write(line + '\n')


if __name__ == '__main__':
    main()