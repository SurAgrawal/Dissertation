
"""
Shared.plot_spectrum
====================
Plot helpers for spectra and training curves.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_spectrum(evals, title):
    ev = np.sort(np.maximum(evals, 1e-16))
    plt.figure()
    plt.semilogy(ev)
    plt.title(title)
    plt.xlabel("index")
    plt.ylabel("normalized eigenvalue")
    plt.tight_layout()

def plot_losses(loss_plain, loss_ff):
    plt.figure()
    plt.semilogy(loss_plain, label="Plain")
    plt.semilogy(loss_ff, label="FF-MLP")
    plt.title("Physics loss vs. step")
    plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()

def plot_relerrs(rel_plain, rel_ff):
    plt.figure()
    plt.semilogy(rel_plain, label="Plain")
    plt.semilogy(rel_ff, label="FF-MLP")
    plt.title("Relative L2 error vs. step")
    plt.xlabel("step"); plt.ylabel("rel. L2"); plt.legend(); plt.tight_layout()
