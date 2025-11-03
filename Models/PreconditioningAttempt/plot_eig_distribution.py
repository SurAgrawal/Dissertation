
"""
Plot eigenvalue distributions from saved .npy files in an eval directory.
Usage:
    python -m Models.PreconditioningAttempt.plot_eig_distribution --evaldir <path_to_eval_dir> \
        --labels "MLP" "Fourier Features + Preconditioning" ["MLP + Preconditioned Fourier Features"]
By default, it looks for:
    evals_plain.npy (plain MLP)
    evals_ff.npy    (FF-MLP)
Optionally, you can supply a third file via --third evals_third.npy  (e.g., linear model on preconditioned Fourier features).
The plot is saved alongside as 'eig_distribution.png'.
"""

import os, argparse, numpy as np, matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evaldir", type=str, required=True, help="Directory containing evals_*.npy from eval run")
    p.add_argument("--bins", type=int, default=30)
    p.add_argument("--labels", nargs="+", default=["MLP", "Fourier Features + Preconditioning"])
    p.add_argument("--third", type=str, default=None, help="Optional path to a third npy (e.g., evals_linear_ff.npy)")
    args = p.parse_args()

    evals_plain_path = os.path.join(args.evaldir, "evals_plain.npy")
    evals_ff_path = os.path.join(args.evaldir, "evals_ff.npy")

    if not os.path.exists(evals_plain_path) or not os.path.exists(evals_ff_path):
        raise SystemExit("Could not find evals_plain.npy and evals_ff.npy in evaldir. Run eval.py first.")

    evals_plain = np.load(evals_plain_path)
    evals_ff    = np.load(evals_ff_path)

    datasets = [evals_plain, evals_ff]
    labels = args.labels

    if args.third is not None and os.path.exists(args.third):
        evals_third = np.load(args.third)
        datasets.append(evals_third)

    plt.figure(figsize=(6,3))
    for i, data in enumerate(datasets):
        plt.hist(data, bins=args.bins, alpha=0.6, label=labels[i] if i < len(labels) else f"set {i+1}",
                 density=False, log=True)
    plt.title(r"Distribution of eigenvalues of $A$")
    plt.xlabel("normalized eigenvalue")
    plt.ylabel("count (log scale)")
    plt.legend()
    plt.tight_layout()

    out = os.path.join(args.evaldir, "eig_distribution.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print("Saved", out)

if __name__ == "__main__":
    main()
