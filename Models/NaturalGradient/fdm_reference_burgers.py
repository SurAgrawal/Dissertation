"""
Finite-difference reference solver for viscous Burgers on x∈[0,2π], t∈[0,T].
Default: periodic BC in x. (Dirichlet/Neumann hooks are placeholders.)
Scheme: semi-implicit in diffusion (FFT for periodic), explicit central for convection.
Lightweight and aimed for evaluation, not production CFD.

Extras:
- times: array-like of snapshot times in [0,T] (optional).
- return_series: if True, returns u(x,t) at those times (or 21 equally spaced if times=None).
"""

import math
import numpy as np

def solve_burgers_1d(
    T=1.0, Nx=512, Nt=1024, nu=1e-2, boundary="periodic",
    dirichlet_vals=(0.0, 0.0), neumann_vals=(0.0, 0.0),
    u0_fn=lambda x: -np.sin(x), a=0.0, b=2*math.pi,
    times=None, return_series=False
):
    if boundary != "periodic":
        raise NotImplementedError("Non-periodic BCs not yet implemented in the reference FD solver. Use periodic.")

    x = np.linspace(a, b, Nx, endpoint=False)
    dx = (b - a) / Nx
    dt = T / Nt

    u = u0_fn(x).astype(np.float64)

    # FFT diagonalization for diffusion: denominator is (1 + dt*nu*k^2)
    ks = 2*math.pi*np.fft.fftfreq(Nx, d=dx)
    denom = 1.0 + dt*nu*(ks**2)

    # Optional series storage
    series = None
    t_targets = None
    if return_series:
        if times is None:
            t_targets = np.linspace(0.0, T, 21)
        else:
            t_targets = np.asarray(times, dtype=np.float64)
        series = np.zeros((t_targets.size, Nx), dtype=np.float64)
        t_cur = 0.0
        idx = 0
        # store initial snapshot (t=0) if requested
        while idx < t_targets.size and abs(t_targets[idx] - t_cur) < 1e-12:
            series[idx] = u
            idx += 1

    for step in range(Nt):
        # explicit convection (central derivative)
        ux = (np.roll(u, -1) - np.roll(u, 1)) / (2*dx)
        rhs = u - dt * (u * ux)

        # implicit diffusion via FFT
        uh = np.fft.fft(rhs)
        u = np.fft.ifft(uh / denom).real

        if return_series:
            t_cur = (step + 1) * dt
            # record any snapshots that fall at/under this step
            while idx < t_targets.size and t_cur + 1e-12 >= t_targets[idx]:
                series[idx] = u
                idx += 1

    if return_series:
        return x, series, t_targets
    else:
        return x, u
