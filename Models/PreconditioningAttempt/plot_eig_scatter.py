# Models/PreconditioningAttempt/plot_eig_scatter.py
"""
Scatter plot of eigenvalues saved by eval.py (+ condition number).

- Loads evals_plain.npy / evals_ff.npy from --evaldir
  * These are normalized in the current eval.py (λ / λ_max).
  * If files evals_plain_raw.npy / evals_ff_raw.npy exist too, we use those
    to plot raw eigenvalues and compute κ from raw.
  * If raw files are absent, we use normalized arrays; κ is still valid
    (scale-invariant), and the y-label notes "normalized".

- Prints condition number κ = λ_max / λ_min_pos (ignoring values <= --eig-threshold).
- If eigenvalues are complex, draws an Argand diagram instead of 1D scatter.

Usage:
  python -m Models.PreconditioningAttempt.plot_eig_scatter \
    --evaldir Models/PreconditioningAttempt/checkpoints/<run>/eval-<timestamp> \
    [--eig-threshold 1e-12] [--alpha 0.8] [--markers o ^] [--jitter 0.0] \
    [--complex-eps 1e-12] [--no-grid]
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_evals(evaldir: str):
    # Normalized (always expected)
    p_norm = os.path.join(evaldir, "evals_plain.npy")
    f_norm = os.path.join(evaldir, "evals_ff.npy")
    if not (os.path.exists(p_norm) and os.path.exists(f_norm)):
        raise FileNotFoundError(f"Need evals_plain.npy and evals_ff.npy in {evaldir}.")
    ep_norm = np.load(p_norm)
    ef_norm = np.load(f_norm)
    # Optional raw
    p_raw = os.path.join(evaldir, "evals_plain_raw.npy")
    f_raw = os.path.join(evaldir, "evals_ff_raw.npy")
    ep_raw = np.load(p_raw) if os.path.exists(p_raw) else None
    ef_raw = np.load(f_raw) if os.path.exists(f_raw) else None
    return (ep_norm, ef_norm, ep_raw, ef_raw)

def cond_number(evals: np.ndarray, eig_threshold: float):
    # κ = λ_max / λ_min_pos ; ignore <= threshold
    if evals.size == 0:
        return np.nan
    vals = np.abs(evals) if np.iscomplexobj(evals) else evals
    pos = vals[vals > eig_threshold]
    if pos.size == 0:
        return float("inf")
    return float(pos.max() / pos.min())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaldir", required=True)
    ap.add_argument("--labels", nargs=2, default=["MLP","FF-MLP"])
    ap.add_argument("--eig-threshold", type=float, default=1e-12)
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--markers", nargs=2, default=["o","^"])
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--complex-eps", type=float, default=1e-12)
    ap.add_argument("--no-grid", action="store_true")
    args = ap.parse_args()

    ep_norm, ef_norm, ep_raw, ef_raw = load_evals(args.evaldir)
    use_raw = (ep_raw is not None) and (ef_raw is not None)
    ep = (ep_raw if use_raw else ep_norm).reshape(-1)
    ef = (ef_raw if use_raw else ef_norm).reshape(-1)

    # Condition numbers (scale-invariant; fine on normalized too)
    kappa_p = cond_number(ep, args.eig_threshold)
    kappa_f = cond_number(ef, args.eig_threshold)

    # Stats
    zeros_p = int((np.abs(ep) <= args.eig_threshold).sum()) if not np.iscomplexobj(ep) else 0
    zeros_f = int((np.abs(ef) <= args.eig_threshold).sum()) if not np.iscomplexobj(ef) else 0
    print(f"plain: total={ep.size}, ~zeros<={args.eig_threshold}: {zeros_p}, κ≈{kappa_p:.3e}")
    print(f"ff   : total={ef.size}, ~zeros<={args.eig_threshold}: {zeros_f}, κ≈{kappa_f:.3e}")

    # If anything is complex, plot Argand diagram
    complex_mode = np.iscomplexobj(ep) or np.iscomplexobj(ef)
    if complex_mode:
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        ax.scatter(np.real(ep), np.imag(ep), s=10, alpha=args.alpha, marker=args.markers[0],
                   label=f"{args.labels[0]} (κ≈{kappa_p:.3e})")
        ax.scatter(np.real(ef), np.imag(ef), s=10, alpha=args.alpha, marker=args.markers[1],
                   label=f"{args.labels[1]} (κ≈{kappa_f:.3e})")
        ax.axhline(0, linewidth=0.6); ax.axvline(0, linewidth=0.6)
        ax.set_xlabel("Re(λ)"); ax.set_ylabel("Im(λ)")
        ax.set_title("Eigenvalues (Argand diagram)")
        if not args.no_grid: ax.grid(True, linewidth=0.3, alpha=0.6)
        ax.legend(frameon=False)
        plt.tight_layout()
        out = os.path.join(args.evaldir, "eig_scatter_argand.png")
        plt.savefig(out, dpi=180)
        print(f"Saved {out}")
        return

    # Otherwise 1D scatter (index vs eigenvalue)
    idx_p = np.arange(ep.size); idx_f = np.arange(ef.size)
    jp = (np.random.rand(ep.size)-0.5)*2*args.jitter if args.jitter>0 else 0.0
    jf = (np.random.rand(ef.size)-0.5)*2*args.jitter if args.jitter>0 else 0.0

    fig = plt.figure(figsize=(8,5)); ax = fig.add_subplot(111)
    ylab = "Eigenvalue (raw)" if use_raw else "Normalized eigenvalue (λ / λ_max)"
    ax.scatter(idx_p + (jp if isinstance(jp, np.ndarray) else jp), ep,
               s=10, alpha=args.alpha, marker=args.markers[0],
               label=f"{args.labels[0]} (κ≈{kappa_p:.3e})")
    ax.scatter(idx_f + (jf if isinstance(jf, np.ndarray) else jf), ef,
               s=10, alpha=args.alpha, marker=args.markers[1],
               label=f"{args.labels[1]} (κ≈{kappa_f:.3e})")
    ax.set_xlabel("Eigenvalue index (ascending)")
    ax.set_ylabel(ylab)
    ax.set_title("Eigenvalue scatter (A-matrix)")
    if not args.no_grid: ax.grid(True, linewidth=0.3, alpha=0.6)
    ax.legend(frameon=False)
    plt.tight_layout()
    out = os.path.join(args.evaldir, "eig_scatter.png")
    plt.savefig(out, dpi=180)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
