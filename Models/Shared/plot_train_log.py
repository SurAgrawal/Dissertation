# Models/DiffusionReaction/plot_train_log.py
from __future__ import annotations
import argparse, os, numpy as np
import matplotlib
matplotlib.use("Agg")                  # safe non-GUI backend
import matplotlib.pyplot as plt

def _to_array(x):
    # hist was saved from Python lists; make sure we get real float arrays
    return np.asarray(x, dtype=float)

def _smooth(y: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or k > len(y): return y
    kern = np.ones(k, dtype=float) / k
    return np.convolve(y, kern, mode="valid")

def main():
    ap = argparse.ArgumentParser("Plot training losses from train_log.npz")
    ap.add_argument("--log", required=True, help="path to train_log.npz")
    ap.add_argument("--out", default="", help="output PNG (default: alongside .npz)")
    ap.add_argument("--smooth", type=int, default=1, help="moving-average window (steps)")
    args = ap.parse_args()

    data = np.load(args.log, allow_pickle=True)
    keys = set(data.files)
    required = {"step","loss","loss_pde","loss_ic","loss_bc","lr"}
    missing = required - keys
    if missing:
        raise RuntimeError(f"train_log missing keys: {sorted(missing)}. Found: {sorted(keys)}")

    step     = _to_array(data["step"])
    loss     = _to_array(data["loss"])
    loss_pde = _to_array(data["loss_pde"])
    loss_ic  = _to_array(data["loss_ic"])
    loss_bc  = _to_array(data["loss_bc"])
    lr       = _to_array(data["lr"])

    # optional smoothing
    sstep = step
    if args.smooth > 1:
        # align x with the 'valid' convolution length
        offset = args.smooth - 1
        sstep  = step[offset:]                    # simple alignment
        loss   = _smooth(loss,     args.smooth)
        loss_pde=_smooth(loss_pde, args.smooth)
        loss_ic =_smooth(loss_ic,  args.smooth)
        loss_bc =_smooth(loss_bc,  args.smooth)
        lr      =_smooth(lr,       args.smooth)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.semilogy(sstep, loss,     label="total")
    ax.semilogy(sstep, loss_pde, label="PDE")
    ax.semilogy(sstep, loss_ic,  label="IC")
    ax.semilogy(sstep, loss_bc,  label="BC")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title("Training losses (semilog)")

    # ax.plot(sstep, loss,     label="total")
    # ax.plot(sstep, loss_pde, label="PDE")
    # ax.plot(sstep, loss_ic,  label="IC")
    # ax.plot(sstep, loss_bc,  label="BC")
    # ax.set_yscale("symlog", linthresh=1e-2, base=10, linscale=1)

    ax.grid(True, which="both", alpha=0.3); ax.legend()



    # # plot LR on a twin y-axis for context
    # ax2 = ax.twinx()
    # ax2.plot(sstep, lr, color="gray", alpha=0.6, linewidth=1.0, label='LR')
    # ax2.set_ylabel("learning rate")

    fig.tight_layout()
    out = args.out or os.path.join(os.path.dirname(args.log), "training_curves_from_log.png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
