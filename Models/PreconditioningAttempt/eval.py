
"""
Evaluation & figure generation for C.4-style analysis.
Generates:
  - Eigenvalue spectra of A at initialization for both models
  - Training curves (loss and rel-L2) for a selected run directory
Usage:
    python -m Models.PreconditioningAttempt.eval --run <run_dir> --K 20 --width 64 --depth 3 --spectral_N 512
If --run is omitted, the latest run in --outdir is used.
"""

import os, math, argparse, json, time, numpy as np, torch, matplotlib.pyplot as plt
from glob import glob
from ..Shared.pinn_models import MLP, MLPConfig, fourier_features
from ..Shared.pinn_ops import PDEConfig, make_grid, forcing, A_matrix
from ..Shared.plot_spectrum import plot_spectrum, plot_losses, plot_relerrs

torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"

def latest_run(outdir: str):
    runs = sorted([d for d in glob(os.path.join(outdir, "*-precond")) if os.path.isdir(d)])
    return runs[-1] if runs else None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--spectral_N", type=int, default=512, help="Grid size for A-matrix (keep modest)")
    p.add_argument("--outdir", type=str, default=os.path.join("Models", "PreconditioningAttempt", "checkpoints"))
    p.add_argument("--run", type=str, default=None, help="Specific run directory under outdir")
    p.add_argument("--prefix", type=str, default="poisson")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    run_dir = args.run or latest_run(args.outdir)
    if run_dir is None:
        raise SystemExit("No run directory found. Train first or pass --run.")
    print("Using run:", run_dir)

    # Create an eval output folder inside the run
    eval_dir = os.path.join(run_dir, f"eval-{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(eval_dir, exist_ok=True)

    # Smaller grid for spectral construction (A is expensive)
    cfg = PDEConfig(N=args.spectral_N, k=1.0, lam_bc=0.0, domain_left=-math.pi, domain_right=math.pi)
    x, w = make_grid(cfg, device=device)
    f_grid = forcing(x, cfg.k)

    # Instantiate models at initialization
    plain = MLP(MLPConfig(in_dim=1, out_dim=1, width=args.width, depth=args.depth)).to(device)
    ff = MLP(MLPConfig(in_dim=2*args.K, out_dim=1, width=args.width, depth=args.depth)).to(device)

    # Compute A and spectra (normalize by max eigenvalue)
    print("Building A for Plain MLP (may take time)...")
    A_p = A_matrix(plain, x, w, cfg.k, featurizer=None, lambda_bc=0.0)
    evals_p = torch.linalg.eigvalsh(A_p).clamp_min(0)
    evals_p = (evals_p / evals_p.max()).cpu().numpy()

    print("Building A for FF-MLP (may take time)...")
    A_f = A_matrix(ff, x, w, cfg.k, featurizer=lambda xin: fourier_features(xin, args.K), lambda_bc=0.0)
    evals_f = torch.linalg.eigvalsh(A_f).clamp_min(0)
    evals_f = (evals_f / evals_f.max()).cpu().numpy()

    # Plots
    plot_spectrum(evals_p, "Normalized eigenvalues of A (Plain MLP)")
    plot_spectrum(evals_f, "Normalized eigenvalues of A (FF-MLP)")

    # Training curves if available in run_dir
    loss_p_path = os.path.join(run_dir, f"{args.prefix}_loss_plain.npy")
    loss_f_path = os.path.join(run_dir, f"{args.prefix}_loss_ff.npy")
    rel_p_path  = os.path.join(run_dir, f"{args.prefix}_rel_plain.npy")
    rel_f_path  = os.path.join(run_dir, f"{args.prefix}_rel_ff.npy")
    if all(os.path.exists(pth) for pth in [loss_p_path, loss_f_path, rel_p_path, rel_f_path]):
        loss_p = np.load(loss_p_path, allow_pickle=True)
        loss_f = np.load(loss_f_path, allow_pickle=True)
        rel_p  = np.load(rel_p_path, allow_pickle=True)
        rel_f  = np.load(rel_f_path, allow_pickle=True)
        plot_losses(loss_p, loss_f)
        plot_relerrs(rel_p, rel_f)
    else:
        print("Training logs not found in run_dir; run the train script first to produce loss/rel-L2 curves.")

    # Save figures into eval_dir
    for i, f in enumerate(plt.get_fignums(), start=1):
        plt.figure(f)
        outp = os.path.join(eval_dir, f"figure_{i:02d}.png")
        plt.savefig(outp, dpi=160, bbox_inches="tight")
        print("Saved", outp)

    # Save spectra arrays for later reuse
    np.save(os.path.join(eval_dir, "evals_plain.npy"), evals_p)
    np.save(os.path.join(eval_dir, "evals_ff.npy"), evals_f)

    print("Done.")

if __name__ == "__main__":
    main()
