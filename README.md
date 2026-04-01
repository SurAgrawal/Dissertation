# Dissertation Numerical Experiments Repository

This repository contains the **experimental codebase** for a dissertation studying *Physics‑Informed Neural Networks (PINNs)* and *Deep Operator Networks (DeepONets)*.  The experiments explore PINNs on canonical partial differential equations (PDEs) in one and two dimensions, compare first‑order optimisation (Adam, gradient descent with line search) against **Energy Natural Gradient Descent (ENGD)**, and investigate *preconditioning strategies* such as Fourier features and eigenvalue scaling.  Additional modules implement DeepONets for 1‑D ordinary differential equation (ODE) operator learning and diffusion–reaction PDEs.

The code is organised into several subdirectories under `Models/`.  Each subdirectory corresponds to a family of experiments.  This README explains the purpose of each folder, outlines the key scripts, and shows **copy‑paste commands** to run the experiments.  Throughout this document citations link back to the code for traceability.

## Table of Contents

1. [Shared Components](#shared-components)
2. [Comparison Experiments (2‑D Poisson & Helmholtz)](#comparison-experiments)
3. [DeepONet ODE Experiments](#deep-onet-ode-experiments)
4. [Diffusion–Reaction DeepONet](#diffusion–reaction-deeponet)
5. [Burgers Preconditioning Experiments](#burger-preconditioning)
6. [Natural Gradient Methods](#natural-gradient-methods)
7. [Preconditioning Attempt (1‑D Poisson)](#preconditioning-attempt)
8. [Additional Notes](#additional-notes)

---

## Shared Components

The `Models/Shared` folder provides utilities used across experiments:

| File | Purpose |
| --- | --- |
| `models.py` | Defines a generic multi‑layer perceptron (`mlp`) and a **DeepONet** class implementing the branch‑and‑trunk architecture.  The branch network processes input functions and the trunk network processes spatial coordinates; their outputs are combined and summed to produce the operator output. |
| `pinn_models.py` | Provides a dataclass `MLPConfig` and an `MLP` class for building fully‑connected networks with configurable width/depth/activation.  It also includes a `fourier_features` function to generate periodic Fourier feature vectors scaled by `1/k^2` . |
| `data.py` | Supplies helper functions for PINN experiments: `make_grid` generates collocation grids, `rbf_cov` defines radial basis function kernels, `sample_grf` samples functions from Gaussian random fields, and `antiderivative_batch` computes antiderivatives . |
| `utils.py` | Utility functions to set random seeds, create timestamped run directories, and save/load checkpoints or JSON configs. |

These modules are imported by the scripts in other subdirectories.

---

## Comparison Experiments

`Models/Comparison` contains **baseline and preconditioned PINN solvers** for two‑dimensional Poisson and Helmholtz equations.  Four solvers are implemented: **Adam**, **Gradient Descent (GD) with line search**, **Energy Natural Gradient Descent (ENGD)**, and **Eigenvalue‑scaled sine preconditioning (Dirichlet‑sine features)**.  Each solver logs training curves, relative errors, and saves predicted solution grids.

### Individual solver scripts

The following scripts solve the 2‑D Poisson equation `-Δu = f` with Dirichlet boundary conditions:

| Script | Description | Run command |
| --- | --- | --- |
| `adam_poisson_2d.py` | Implements the Adam optimiser for a 2‑D Poisson PINN.  It defines the exact solution `sin(π x) sin(π y)`, residuals, constructs a JAX MLP, and runs a training loop with optional early stopping; metrics and predicted solutions are logged to `adam_poisson_metrics.npz` . | `python -m Models.Comparison.adam_poisson_2d --steps 2000 --log_every 100 --hidden_width 64 --hidden_depth 3 --seed 0 --outdir /path/to/output` |
| `precond_poisson_2d.py` | Applies **eigenvalue‑scaled Dirichlet‑sine feature preconditioning**.  The script computes sine features scaled by eigenvalues, builds an MLP, defines PDE and boundary residuals, trains with Adam, and logs loss, relative L2 and H1 errors. | `python -m Models.Comparison.precond_poisson_2d --steps 2000 --k_max 10 --hidden_width 64 --hidden_depth 3 --seed 0 --outdir /path/to/output` |
| `gd_poisson_2d.py` | Gradient descent with **geometric line search** for the Poisson equation.  It sets up integrators and an MLP, defines the loss, uses line search to pick step sizes, logs loss, L2/H1 errors, and saves metrics and the solution grid. | `python -m Models.Comparison.gd_poisson_2d --steps 2000 --log_every 100 --hidden_width 64 --hidden_depth 3 --seed 0 --outdir /path/to/output` |
| `engd_poisson_2d.py` | Implements **Energy Natural Gradient Descent** for Poisson.  The script constructs Gramian matrices using natural gradients, performs a grid line search and records metrics (loss, L2 and H1 errors); it supports early stopping and logs predicted solutions. | `python -m Models.Comparison.engd_poisson_2d --steps 2000 --log_every 100 --hidden_width 64 --hidden_depth 3 --seed 0 --outdir /path/to/output --tol 1e-8` |

Analogous scripts solve the 2‑D **Helmholtz equation** `-Δu + k²u = f` with parameter `k`:

| Script | Description | Run command |
| --- | --- | --- |
| `adam_helmholtz_2d.py` | Adam optimiser for Helmholtz.  It constructs the MLP, defines residuals and loss, uses an exponential decay learning rate, and logs metrics. | `python -m Models.Comparison.adam_helmholtz_2d --steps 2000 --log_every 100 --hidden_width 64 --hidden_depth 3 --seed 0 --helmholtz_k 1.0 --outdir /path/to/output` |
| `precond_helmholtz_2d.py` | Eigenvalue‑scaled sine features for Helmholtz; similar to the Poisson preconditioning but features scaled by `(λ_nm + k²)`. | `python -m Models.Comparison.precond_helmholtz_2d --steps 2000 --k_max 10 --hidden_width 64 --hidden_depth 3 --helmholtz_k 1.0 --seed 0 --outdir /path/to/output` |
| `gd_helmholtz_2d.py` | Gradient descent with grid line search for Helmholtz. | `python -m Models.Comparison.gd_helmholtz_2d --steps 2000 --log_every 100 --hidden_width 64 --hidden_depth 3 --helmholtz_k 1.0 --seed 0 --outdir /path/to/output` |
| `engd_helmholtz_2d.py` | ENGD for Helmholtz; uses Gramian matrices and a natural gradient update with line search, logs metrics and saves solution grids. | `python -m Models.Comparison.engd_helmholtz_2d --steps 2000 --log_every 100 --hidden_width 64 --hidden_depth 3 --helmholtz_k 1.0 --seed 0 --outdir /path/to/output --tol 1e-8` |

### Aggregated comparison scripts

Two scripts orchestrate runs of all methods and produce comparative plots and summary statistics:

| Script | Purpose | Run command |
| --- | --- | --- |
| `run_all_methods.py` | Executes all Poisson solvers (Adam, ENGD, GD+line search, preconditioned) with a common set of hyperparameters, aggregates training loss curves, writes a summary JSON of final errors and training times, and generates comparison plots. | `python -m Models.Comparison.run_all_methods --steps 2000 --log-every 100 --hidden_width 64 --hidden_depth 3 --k_max 10 --seed 0 --outdir /path/to/output` |
| `run_all_methods_helmholtz.py` | Similar to `run_all_methods.py` but for Helmholtz; accepts an extra `--helmholtz-k` parameter and produces unified scales across solvers. | `python -m Models.Comparison.run_all_methods_helmholtz --steps 2000 --log-every 100 --hidden_width 64 --hidden_depth 3 --k_max 10 --helmholtz_k 1.0 --seed 0 --outdir /path/to/output` |

### Experiment runner (hyperparameter sweeps)

The `experiment_runner.py` script performs **hyperparameter sweeps** for the DeepONet ODE experiments.  It builds a Cartesian product of hyperparameters (widths, depths, activations, seeds), launches `deepo-ODE/train.py` for each, evaluates the best and last checkpoints using `eval.py`, aggregates results into CSV/JSON summaries, and plots comparisons.  Use it as:

```bash
python -m Models.Comparison.experiment_runner --widths 50 100 --depths 4 6 --acts relu tanh --seeds 0 1 2 --outdir /path/to/output
```

This will create subdirectories for each run, log training curves and metrics, and write summary plots.

---

## DeepONet ODE Experiments

The `Models/deepo-ODE` subdirectory contains experiments on **Deep Operator Networks** (DeepONets) learning solution operators of 1‑D ODEs.  Two main scripts are provided:

| Script | Purpose | Run command |
| --- | --- | --- |
| `train.py` | Trains a physics‑informed DeepONet on a family of ODEs with random forcing.  It samples training functions from a Gaussian random field, builds a DeepONet (branch and trunk networks) with configurable width/depth and feature dimension, and runs an Adam training loop.  Checkpoints (`best.pt` and `last.pt`), JSON config, and training logs (`train_log.npz`) are saved. | `python -m Models.deepo-ODE.train --ntrain 50 --ntest 20 --m 50 --ell 0.5 --steps 2000 --batch-size 16 --width 100 --depth 4 --act relu --lr 1e-3 --outdir /path/to/output` |
| `eval.py` | Evaluates a trained DeepONet.  It loads a checkpoint, reconstructs the model, regenerates or loads test functions, computes predictions, and calculates metrics such as relative L2, MAE, RMSE, max absolute error and R².  It prints summary statistics and saves a per‑example CSV and bar plot of relative errors. | `python -m Models.deepo-ODE.eval --ckpt /path/to/best.pt --num-examples 5 --outdir /path/to/eval` |

Running `train.py` will create a run directory with a timestamp (using `make_run_dir` from `utils.py`).  Use `eval.py` to assess performance on held‑out functions.

---

## Diffusion–Reaction DeepONet

`Models/DiffusionReaction` implements DeepONets for a **diffusion–reaction PDE** in one spatial dimension.  The diffusion coefficient `D` and reaction rate `k` are configurable.  The main scripts are:

| Script | Purpose | Run command |
| --- | --- | --- |
| `train.py` | Generates training and test datasets using Gaussian random fields, constructs a DeepONet with specified width, depth and feature dimension, and trains using an Adam optimiser.  It logs PDE, initial and boundary losses separately and saves `best.pt`, `last.pt`, and training curves (NPZ and PNG). | `python -m Models.DiffusionReaction.train --ntrain 50 --ntest 20 --m 50 --ell 0.5 --steps 2000 --batch-size 16 --width 100 --depth 4 --feat-dim 50 --D 0.1 --k 1.0 --w-pde 1.0 --w-ic 1.0 --w-bc 1.0 --lr 1e-3 --outdir /path/to/output` |
| `eval.py` | Given a checkpoint, reconstructs the DeepONet, loads saved test functions and grid, and computes predicted solutions.  It evaluates PDE residual maps and boundary/initial errors, optionally compares to a finite‑difference reference, and saves heatmaps and error plots. | `python -m Models.DiffusionReaction.eval --ckpt /path/to/best.pt --ntest 20 --num-examples 5 --with-reference --outdir /path/to/eval` |

---

## Burger Preconditioning

`Models/BurgerPreCondition` explores **preconditioning for Burgers’ equation** using Fourier features versus a plain MLP.  The `eval_burgers.py` script loads trained checkpoints (`plain_last.pt` and `ff_last.pt`), reconstructs models, plots physics loss curves across epochs, and compares time evolution of the solution against a finite‑difference reference.  It can also compute eigenvalue spectra of the operator matrix.  Use it as:

```bash
python -m Models.BurgerPreCondition.eval_burgers --run /path/to/run_dir --outdir /path/to/eval --compute-spectrum --spectral_Nx 256 --spectral_Nt 100 --num-slices 5
```

The script reads `config.json` from the run directory to know the model widths, depths and Fourier feature counts, then plots the training loss and solution snapshots.

---

## Natural Gradient Methods

`Models/NaturalGradient` demonstrates **natural gradient optimisation** on Poisson and Burgers’ equations.  These scripts use Gramian matrices to approximate the Fisher information and perform natural gradient updates with line search:

| Script | Description |
| --- | --- |
| `GitPoisson2D.py` | Runs an ENGD PINN for 2‑D Poisson.  It constructs the domain and integrators, builds an MLP, computes Gramian matrices using natural gradient, trains via line search, records the L² error over iterations, and produces a plot. |
| `EngdBurgers1D.py` | Implements ENGD for the 1‑D viscous Burgers equation.  It defines the domain in space and time, sets up interior, initial and boundary residuals, constructs Gramian matrices, and performs natural gradient updates.  A `visualize` function compares the PINN to a finite‑difference reference and produces heatmaps.  Run with `python -m Models.NaturalGradient.EngdBurgers1D --steps 2000 --log-every 100 --hidden_width 64 --hidden_depth 3 --seed 0 --outdir /path/to/output`. |
| `Visualise.py` | A more sophisticated Burgers’ experiment with periodic boundary conditions.  It uses a hyperrectangle domain `[0, 2π] × [0,1]`, defines interior, initial and periodic losses, constructs the Gramian using Laplacian transform only, runs natural gradient training and visualises the solution against a finite‑difference reference. |

These scripts are largely demonstrations; adjust CLI parameters (steps, width, depth, seed, etc.) as needed.

---

## Preconditioning Attempt

`Models/PreconditioningAttempt` contains the **initial experiments** on Fourier‑feature preconditioning for a 1‑D Poisson problem.  The folder stores checkpoints and evaluation scripts used to generate figures for the dissertation appendix.

### Training script

The `train.py` script trains two models on the same 1‑D Poisson problem: a **plain MLP** and an **FF‑MLP** with Fourier features.  It uses functions from `pinn_models.py` and `pinn_ops.py` to set up the PDE and training loop.  The script saves the final checkpoints (`poisson_plain_last.pt` and `poisson_ff_last.pt`), logs the physics loss and relative error, and writes `config.json` summarising the run.  To run:

```bash
python -m Models.PreconditioningAttempt.train --steps 2000 --width 64 --depth 3 --K 20 --lr 1e-3 --outdir Models/PreconditioningAttempt/checkpoints
```

This command creates a timestamped run directory under `checkpoints` and stores loss curves as NumPy files.  The run directory name is of the form `YYYYMMDD-HHMMSS-precond`.

### Evaluation and eigenvalue analysis

`eval.py` analyses a run directory to generate figures like those in the dissertation.  It can compute the spectrum of the operator matrix `A` for both models (plain and Fourier‑feature), plot eigenvalue distributions, training loss curves, relative errors, and other diagnostic plots.  Key options include:

* `--spectral_N` (number of grid points for eigenvalue computation)
* `--run` (path to the run directory)
* `--outdir` (where to save plots)
* `--skip-spectrum` (use previously computed eigenvalues)
* `--use-evals-from` (reuse eigenvalues from another run)

An example usage:

```bash
python -m Models.PreconditioningAttempt.eval --run Models/PreconditioningAttempt/checkpoints/20251019-144711-precond \ 
    --outdir Models/PreconditioningAttempt/checkpoints/20251019-144711-precond/eval --spectral_N 256
```

### Stand‑alone plotting utilities

To visualise eigenvalues saved by `eval.py`, two helper scripts are provided:

| Script | Description | Example |
| --- | --- | --- |
| `plot_eig_scatter.py` | Creates a scatter plot (or Argand diagram) of eigenvalues and reports the **condition number** κ.  It expects `evals_plain.npy` and `evals_ff.npy` in the eval directory.  The plot distinguishes the plain MLP and Fourier‑feature model and saves `eig_scatter.png`. | `python -m Models.PreconditioningAttempt.plot_eig_scatter --evaldir /path/to/eval` |
| `plot_eig_distribution.py` | Plots histograms of the eigenvalue distribution.  By default it compares the plain MLP and Fourier‑feature model; a third set of eigenvalues can be supplied.  It saves `eig_distribution.png`. | `python -m Models.PreconditioningAttempt.plot_eig_distribution --evaldir /path/to/eval` |

These scripts print the condition numbers and produce plots in the given evaluation directory.

---

## Additional Notes

* **Dependencies:** Most scripts require Python ≥3.8, PyTorch, JAX (for Poisson/Helmholtz PINNs), NumPy and Matplotlib.  Ensure CUDA is available if GPU training is desired.
* **Run directories:** Each training script creates a timestamped run directory (e.g., `20260305-191808-deepoODE` or `20251019-144711-precond`) and stores checkpoints, JSON configs, and metrics.  Use the `eval.py` scripts to analyse these runs.
* **Hyperparameter sweeps:** When exploring multiple widths, depths or activation functions, use the `experiment_runner.py` script.  The summarised metrics and plots will help identify the best configurations.

This README aims to provide a comprehensive guide for reproducing and extending the dissertation’s numerical experiments.  Feel free to explore the scripts, adjust parameters, and contribute additional experiments.
