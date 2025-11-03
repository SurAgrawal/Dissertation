
"""
Evaluation and (optional) A-matrix spectrum for Burgers experiments.
Usage examples:
  python -m Models.BurgerPreCondition.eval_burgers --outdir ./Models/BurgerPreCondition/checkpoints --skip-spectrum
  python -m Models.BurgerPreCondition.eval_burgers --run <path> --compute-spectrum
"""

import os, json, argparse, numpy as np, torch, matplotlib.pyplot as plt
from glob import glob

from .burger_ops import BurgersConfig, make_grid_2d, A_matrix_linearized
from .burger_models import FeatureConfig, build_u_of_xt, fourier_features_2d
from ..Shared.pinn_models import MLP, MLPConfig

torch.set_default_dtype(torch.float64)

def latest_run(outdir):
    runs = sorted(glob(os.path.join(outdir, "*") ))
    return runs[-1] if runs else None


def read_config(run_dir):
    with open(os.path.join(run_dir, "config.json"), "r") as f:
        cfg = json.load(f)
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="./Models/BurgerPreCondition/checkpoints")
    p.add_argument("--run", type=str, default=None)
    p.add_argument("--compute-spectrum", action="store_true")
    p.add_argument("--skip-spectrum", action="store_true")
    p.add_argument("--spectral_Nx", type=int, default=128)
    p.add_argument("--spectral_Nt", type=int, default=64)
    args = p.parse_args()

    run_dir = args.run or latest_run(args.outdir)
    if run_dir is None:
        raise SystemExit("No run directory found. Train first or pass --run.")
    print("Using run:", run_dir)

    cfg_json = read_config(run_dir)
    phys = cfg_json["physics"]
    feats = cfg_json["features"]
    m_plain = cfg_json["model_plain"]
    m_ff = cfg_json["model_ff"]

    # Rebuild models
    plain = MLP(MLPConfig(in_dim=m_plain["in_dim"], width=m_plain["width"], depth=m_plain["depth"]))
    ff = MLP(MLPConfig(in_dim=m_ff["in_dim"], width=m_ff["width"], depth=m_ff["depth"]))

    # Load weights (PyTorch>=2.6 default weights_only=True breaks for our saved dict; set False)
    s_plain = torch.load(os.path.join(run_dir, "plain_last.pt"), map_location="cpu", weights_only=False)
    s_ff = torch.load(os.path.join(run_dir, "ff_last.pt"), map_location="cpu", weights_only=False)
    plain.load_state_dict(s_plain["model_state"]) ; ff.load_state_dict(s_ff["model_state"]) 

    # Build featurizer
    feat_cfg = FeatureConfig(Kx=feats["Kx"], Mt=feats["Mt"], T=phys["T"], nu=phys["nu"], hard_periodic=phys["hard_periodic"]) 
    def zmap(x,t):
        return fourier_features_2d(x,t,feat_cfg)

    # Plot training loss (per-epoch) from npy if present
    for tag in ["plain","ff"]:
        loss_path = os.path.join(run_dir, f"{tag}_losses.npy")
        if os.path.exists(loss_path):
            losses = np.load(loss_path)
        else:
            ck = torch.load(os.path.join(run_dir, f"{tag}_last.pt"), map_location="cpu", weights_only=False)
            losses = ck.get("losses", [])
        if len(losses):
            plt.plot(losses, label=f"{tag} loss")
    plt.yscale("log")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("physics loss")
    plt.title("Training loss (per-epoch)")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curves.png"), dpi=160)
    plt.close()

    # === Final prediction vs FD reference (t = T) — side-by-side panels ===
    try:
        from .fdm_reference_burgers import solve_burgers_1d
        x_ref, u_ref_T = solve_burgers_1d(T=phys["T"], Nx=1024, Nt=2048, nu=phys["nu"], boundary=phys["boundary"])
        x_eval = torch.tensor(x_ref, dtype=torch.get_default_dtype()).view(-1,1)
        t_eval = torch.full_like(x_eval, phys["T"])

        # Predictions
        with torch.no_grad():
            u_pred_plain = build_u_of_xt(plain, None)(x_eval, t_eval).cpu().numpy().ravel()
            u_pred_ff = build_u_of_xt(ff, zmap)(x_eval, t_eval).cpu().numpy().ravel()

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True, sharey=True)

        # Panel 1: FD reference
        axes[0].plot(x_ref, u_ref_T)
        axes[0].set_title("FD reference")
        axes[0].set_xlabel("x"); axes[0].set_ylabel("u(x,T)")

        # Panel 2: Plain MLP
        axes[1].plot(x_ref, u_pred_plain)
        axes[1].set_title("Plain MLP")
        axes[1].set_xlabel("x")

        # Panel 3: Preconditioned (FF-MLP)
        axes[2].plot(x_ref, u_pred_ff)
        axes[2].set_title("Preconditioned FF-MLP")
        axes[2].set_xlabel("x")

        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "final_profiles_side_by_side.png"), dpi=160)
        plt.close()
    except Exception as e:
        print("Skipping final side-by-side profiles due to error:", repr(e))

    # Optional spectra
    if (not args.skip_spectrum) and args.compute_spectrum:
        print("Computing A spectra (linearized)... this can be slow.")
        bx = BurgersConfig(Nx=args.spectral_Nx, Nt=args.spectral_Nt, a=phys["a"], b=phys["b"], T=phys["T"], nu=phys["nu"], device="cpu")
        x, t, w = make_grid_2d(bx)
        w = w / w.mean()
        A_plain = A_matrix_linearized(plain, x, t, w, bx, featurizer=None)
        A_ff = A_matrix_linearized(ff, x, t, w, bx, featurizer=zmap)
        evals_plain = torch.linalg.eigvalsh(A_plain).cpu().numpy()
        evals_ff = torch.linalg.eigvalsh(A_ff).cpu().numpy()
        np.save(os.path.join(run_dir, "evals_plain.npy"), evals_plain)
        np.save(os.path.join(run_dir, "evals_ff.npy"), evals_ff)
        print("Saved eigenvalues to run dir.")

    print("Done. Artifacts saved in:", run_dir)

if __name__ == "__main__":
    main()
