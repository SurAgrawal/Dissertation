
"""
Finite-difference reference solver for viscous Burgers on x∈[0,2π], t∈[0,T].
Default: periodic BC in x. (Dirichlet/Neumann hooks are placeholders.)
Scheme: semi-implicit in diffusion (FFT for periodic), explicit central for convection.
This is lightweight and aimed for evaluation, not production CFD.
"""

import math
import numpy as np

def solve_burgers_1d(T=1.0, Nx=512, Nt=1024, nu=1e-2, boundary="periodic",
                      dirichlet_vals=(0.0, 0.0), neumann_vals=(0.0, 0.0),
                      u0_fn=lambda x: -np.sin(x), a=0.0, b=2*math.pi):
    x = np.linspace(a, b, Nx, endpoint=False)
    dx = (b - a) / Nx
    dt = T / Nt

    u = u0_fn(x).astype(np.float64)
    u_next = u.copy()

    if boundary != "periodic":
        # For this reference implementation we keep periodic as the robust default.
        # Non-periodic BCs can be added later with a banded solver.
        raise NotImplementedError("Non-periodic BCs not yet implemented in the reference FD solver. Use periodic.")

    # FFT diagonalization for diffusion: (I + dt*nu*k^2) in denominator
    ks = 2*math.pi*np.fft.fftfreq(Nx, d=dx)
    denom = 1.0 + dt*nu*(ks**2)

    for _ in range(Nt):
        # central derivative for ux
        ux = (np.roll(u, -1) - np.roll(u, 1)) / (2*dx)
        rhs = u - dt * (u * ux)  # explicit convection

        uh = np.fft.fft(rhs)
        u = np.fft.ifft(uh / denom).real  # implicit diffusion

    return x, u
