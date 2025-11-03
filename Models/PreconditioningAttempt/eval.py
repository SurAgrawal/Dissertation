
"""
Evaluation & figure generation for C.4-style analysis.

Auto-reads width/depth/K from run's config.json.
Supports:
  --spectral_N          : grid size for A (only if computing spectra)
  --skip-spectrum       : do not compute A/eigenvalues; only plot curves (and reuse old spectra if found)
  --use-evals-from DIR  : reuse eigenvalues from another eval directory
  --run / --outdir      : select run folder (auto-picks latest if omitted)
"""

import os, math, argparse, json, time, numpy as np, torch, matplotlib.pyplot as plt
from glob import glob
from tqdm.auto import tqdm

from ..Shared.pinn_models import MLP, MLPConfig, fourier_features
from ..Shared.pinn_ops import PDEConfig, make_grid, forcing, A_matrix
from ..Shared.plot_spectrum import plot_spectrum, plot_losses, plot_relerrs

torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"

def latest_run(outdir: str):
    runs = sorted([d for d in glob(os.path.join(outdir, "*-precond")) if os.path.isdir(d)])
    return runs[-1] if runs else None

def list_evaldirs(run_dir: str):
    return sorted([d for d in glob(os.path.join(run_dir, "eval-*")) if os.path.isdir(d)])

def latest_evaldir_with_evals(run_dir: str):
    for d in reversed(list_evaldirs(run_dir)):
        if os.path.exists(os.path.join(d, "evals_plain.npy")) and os.path.exists(os.path.join(d, "evals_ff.npy")):
            return d
    return None

def load_evals(evaldir: str):
    eplain = os.path.join(evaldir, "evals_plain.npy")
    eff    = os.path.join(evaldir, "evals_ff.npy")
    if not (os.path.exists(eplain) and os.path.exists(eff)):
        raise FileNotFoundError(f"Could not find evals_plain.npy and evals_ff.npy in {evaldir}")
    return np.load(eplain), np.load(eff)

def read_model_cfg(run_dir: str):
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.json not found in run directory: {run_dir}")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    extra = cfg.get("extra", {})
    mp = extra.get("model_plain", {})
    mf = extra.get("model_ff", {})
    width_plain = int(mp.get("width", 64))
    depth_plain = int(mp.get("depth", 3))
    width_ff    = int(mf.get("width", 64))
    depth_ff    = int(mf.get("depth", 3))
    K           = int(mf.get("K", 20))
    return dict(width=width_plain, depth=depth_plain), dict(width=width_ff, depth=depth_ff, K=K)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spectral_N", type=int, default=512, help="Grid size for A-matrix (only if computing spectra)")
    p.add_argument("--outdir", type=str, default=os.path.join("Models", "PreconditioningAttempt", "checkpoints"))
    p.add_argument("--run", type=str, default=None, help="Specific run directory under outdir")
    p.add_argument("--prefix", type=str, default="poisson")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    p.add_argument("--skip-spectrum", action="store_true", help="Do not compute eigenvalues/A; only plot curves")
    p.add_argument("--use-evals-from", type=str, default=None, help="Directory that already contains evals_plain.npy and evals_ff.npy")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    run_dir = args.run or latest_run(args.outdir)
    if run_dir is None:
        raise SystemExit("No run directory found. Train first or pass --run.")
    print("Using run:", run_dir)

    # Read model configuration from run's config.json
    model_plain_cfg, model_ff_cfg = read_model_cfg(run_dir)
    width_plain, depth_plain = model_plain_cfg["width"], model_plain_cfg["depth"]
    width_ff, depth_ff, K = model_ff_cfg["width"], model_ff_cfg["depth"], model_ff_cfg["K"]

    # Progress bar planning
    do_compute_spectrum = (not args.skip_spectrum) and (args.use_evals_from is None)
    steps_total = (2 if do_compute_spectrum else 0) + 1 + 1  # compute (2) + plot (1) + save (1)
    iterator = tqdm(total=steps_total, desc="Eval", dynamic_ncols=True) if not args.no_progress else None

    evals_p = evals_f = None

    # Spectrum logic
    if args.use_evals_from is not None:
        if iterator: iterator.set_postfix_str("Loading spectra from --use-evals-from")
        evals_p, evals_f = load_evals(args.use_evals_from)
        if iterator: iterator.update(1)
    elif args.skip_spectrum:
        prev_eval = latest_evaldir_with_evals(run_dir)
        if prev_eval is not None:
            if iterator: iterator.set_postfix_str("Reusing spectra from latest eval in run")
            evals_p, evals_f = load_evals(prev_eval)
            if iterator: iterator.update(1)
        else:
            print("No previous spectra found for this run; skipping spectrum plots.")
    else:
        # Build grid and models only if we actually compute spectra
        cfg = PDEConfig(N=args.spectral_N, k=1.0, lam_bc=0.0, domain_left=-math.pi, domain_right=math.pi)
        x, w = make_grid(cfg, device=device)
        f_grid = forcing(x, cfg.k)

        plain = MLP(MLPConfig(in_dim=1, out_dim=1, width=width_plain, depth=depth_plain)).to(device)
        ff    = MLP(MLPConfig(in_dim=2*K, out_dim=1, width=width_ff, depth=depth_ff)).to(device)

        if iterator: iterator.set_postfix_str("Building A: Plain MLP")
        A_p = A_matrix(plain, x, w, cfg.k, featurizer=None, lambda_bc=0.0)
        ep = torch.linalg.eigvalsh(A_p).clamp_min(0)
        evals_p = (ep / (ep.max() if ep.max() > 0 else 1)).cpu().numpy()
        if iterator: iterator.update(1)

        if iterator: iterator.set_postfix_str("Building A: FF-MLP")
        A_f = A_matrix(ff, x, w, cfg.k, featurizer=lambda xin: fourier_features(xin, K), lambda_bc=0.0)
        ef = torch.linalg.eigvalsh(A_f).clamp_min(0)
        evals_f = (ef / (ef.max() if ef.max() > 0 else 1)).cpu().numpy()
        if iterator: iterator.update(1)

    # Plot spectra (if present)
    if iterator: iterator.set_postfix_str("Plotting")
    if evals_p is not None and evals_f is not None:
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
    if iterator: iterator.update(1)

    # Create the eval output folder inside the run *now* (after we know what we plotted)
    eval_dir = os.path.join(run_dir, f"eval-{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(eval_dir, exist_ok=True)

    # Save figures into eval_dir
    if iterator: iterator.set_postfix_str("Saving figures")
    figs = list(plt.get_fignums())
    if len(figs) == 0:
        print("No figures were created (likely because spectra were skipped and no curves found).")
    for i, fnum in enumerate(figs, start=1):
        plt.figure(fnum)
        outp = os.path.join(eval_dir, f"figure_{i:02d}.png")
        plt.savefig(outp, dpi=160, bbox_inches="tight")
        if iterator: tqdm.write(f"Saved {outp}")

    # If we computed spectra this run, save them now
    if (not args.skip_spectrum) and (args.use_evals_from is None) and (evals_p is not None and evals_f is not None):
        np.save(os.path.join(eval_dir, "evals_plain.npy"), evals_p)
        np.save(os.path.join(eval_dir, "evals_ff.npy"),    evals_f)

    if iterator: iterator.update(1); iterator.close()
    print("Done. Figures saved to", eval_dir)

if __name__ == "__main__":
    main()
