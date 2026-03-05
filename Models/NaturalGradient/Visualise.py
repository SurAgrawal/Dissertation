import math

import jax
import jax.numpy as jnp
from jax import grad, vmap, jit, random

import matplotlib.pyplot as plt
import numpy as np

# --- your helpers ---
from ngrad.models import init_params, mlp
from ngrad.domains import Hyperrectangle   # from domains.py
from ngrad.integrators import DeterministicIntegrator
from ngrad.utility import del_i, grid_line_search_factory
from ngrad.inner import model_laplace
from ngrad.gram import gram_factory, nat_grad_factory

# FD reference solver (your file)
from fdm_reference_burgers import solve_burgers_1d

jax.config.update("jax_enable_x64", True)

# -------------------------------------------------------------------
# Problem settings
# -------------------------------------------------------------------
T = 1.0
nu = 0.01 / jnp.pi
a, b = 0.0, 2.0 * jnp.pi  # x-domain [0, 2π]

# -------------------------------------------------------------------
# Domains & integrators
#   Hyperrectangle in (x, t): [0, 2π] x [0, 1]
# -------------------------------------------------------------------
from ngrad.domains import Rectangle

interior_domain = Rectangle(
    intervals=[[a, b], [0.0, T]]  # or [[0.0, 2*jnp.pi], [0.0, T]]
)
interior_integrator = DeterministicIntegrator(interior_domain, N=40)


# For IC (integrate over x at t=0) and periodic BC (integrate over t)
Nx_ic = 80
Nt_per = 80

# -------------------------------------------------------------------
# Model: uθ(x,t)
# -------------------------------------------------------------------
activation = lambda x: jnp.tanh(x)
layer_sizes = [2, 8, 8, 8, 1]

key = random.PRNGKey(0)
params = init_params(layer_sizes, key)
model = mlp(activation)

def u_theta(params, z):
    return model(params, z)

v_u_theta = vmap(u_theta, (None, 0))


# -------------------------------------------------------------------
# Initial condition u0(x)
# -------------------------------------------------------------------
def u0_np(x):
    # numpy version for FD solver (matches its default)
    return -np.sin(x)

@jit
def u0(z):
    # z = [x, t], but we only use x here
    return -jnp.sin(z[0])


# -------------------------------------------------------------------
# Burgers PDE residual: r = u_t + u u_x - ν u_xx
# -------------------------------------------------------------------
def burgers_residual(params, z):
    """
    z = [x, t]
    """
    def g(z_):
        return u_theta(params, z_)

    # derivatives in x and t
    u_x = del_i(g, argnum=0)
    u_t = del_i(g, argnum=1)
    u_xx = del_i(u_x, argnum=0)

    val = g(z)
    return u_t(z) + val * u_x(z) - nu * u_xx(z)

@jit
def residual_sq(params, z):
    r = burgers_residual(params, z)
    return r**2

v_residual_sq = jit(vmap(residual_sq, (None, 0)))


# -------------------------------------------------------------------
# Loss terms
# -------------------------------------------------------------------
@jit
def interior_loss(params):
    # integrate residual^2 over [0,2π] x [0,T]
    return interior_integrator(lambda pts: v_residual_sq(params, pts))


@jit
def ic_loss(params):
    """
    IC: u(x,0) = -sin(x) on [0, 2π]
    approximate integral via uniform grid in x
    """
    xs = jnp.linspace(a, b, Nx_ic, endpoint=False)
    ts = jnp.zeros_like(xs)
    pts = jnp.stack([xs, ts], axis=1)   # shape (Nx_ic, 2)

    diff = v_u_theta(params, pts) - vmap(u0)(pts)
    # mean over grid is enough; constant scaling doesn’t matter too much
    return jnp.mean(diff**2)


@jit
def periodic_loss(params):
    """
    Enforce full periodicity:
        u(0,t) = u(2π,t)
        u_x(0,t) = u_x(2π,t)
    approximate integrals over t via uniform grid
    """
    ts = jnp.linspace(0.0, T, Nt_per)
    xs_left = jnp.zeros_like(ts)       # x = 0
    xs_right = jnp.ones_like(ts) * b   # x = 2π

    pts_left = jnp.stack([xs_left, ts], axis=1)
    pts_right = jnp.stack([xs_right, ts], axis=1)

    # value periodicity
    vals_left = v_u_theta(params, pts_left)
    vals_right = v_u_theta(params, pts_right)
    val_mis = (vals_left - vals_right)**2

    # derivative periodicity
    def g(z):  # scalar function u(x,t)
        return u_theta(params, z)

    u_x = del_i(g, argnum=0)
    v_u_x = vmap(u_x)

    ux_left = v_u_x(pts_left)
    ux_right = v_u_x(pts_right)
    grad_mis = (ux_left - ux_right)**2

    return jnp.mean(val_mis + grad_mis)


@jit
def total_loss(params):
    return interior_loss(params) + ic_loss(params) + periodic_loss(params)


# -------------------------------------------------------------------
# ENGD: Gramian & natural gradient
#   use Laplace transform on parameter-tangent fields, as in the paper
# -------------------------------------------------------------------
gram_laplace = gram_factory(
    model=u_theta,
    trafo=model_laplace,
    integrator=interior_integrator,
)

@jit
def gram(params):
    return gram_laplace(params)

nat_grad = nat_grad_factory(gram)

# line search over step sizes 0.5^k, k=0..30
grid = jnp.linspace(0, 30, 31)
steps = 0.5 ** grid
ls_update = grid_line_search_factory(total_loss, steps)


# -------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------
def train(num_iterations=200):
    global params

    for it in range(num_iterations + 1):
        g_params = grad(total_loss)(params)
        ng = nat_grad(params, g_params)

        params_new, actual_step = ls_update(params, ng)
        params = params_new

        if it % 10 == 0:
            print(
                f"[Iter {it:4d}] "
                f"loss={float(total_loss(params)):.3e}  "
                f"step={float(actual_step):.3e}"
            )


# -------------------------------------------------------------------
# Visualization: FD reference vs PINN
# -------------------------------------------------------------------
def visualize():
    # 1. finite-difference reference
    x_ref, series_ref, times_ref = solve_burgers_1d(
        T=float(T),
        Nx=512,
        Nt=2048,
        nu=float(nu),
        u0_fn=u0_np,   # same IC
        a=float(a),
        b=float(b),
        return_series=True
    )

    x_ref = np.asarray(x_ref)
    times_ref = np.asarray(times_ref)
    series_ref = np.asarray(series_ref)

    # 2. PINN prediction on same grid
    def pinn_series_for_times(times_np):
        times_j = jnp.array(times_np, dtype=jnp.float64)
        xs_j = jnp.array(x_ref, dtype=jnp.float64)

        def eval_at_time(t):
            xs = xs_j
            ts = jnp.ones_like(xs) * t
            pts = jnp.stack([xs, ts], axis=1)
            return np.array(v_u_theta(params, pts)).reshape(-1)

        return np.stack([eval_at_time(t) for t in times_j], axis=0)

    series_pinn = pinn_series_for_times(times_ref)

    # 3. snapshot comparison at t ≈ 0.5
    t_vis = 0.5
    k = int(np.argmin(np.abs(times_ref - t_vis)))
    u_ref = series_ref[k]
    u_pinn = series_pinn[k]

    plt.figure(figsize=(10, 4))
    plt.plot(x_ref, u_ref, label="FD reference")
    plt.plot(x_ref, u_pinn, "--", label="PINN (ENGD)")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title(f"Burgers solution at t ≈ {times_ref[k]:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4. heatmaps of reference vs PINN (space-time)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    im0 = axs[0].imshow(
        series_ref,
        aspect="auto",
        extent=[a, b, T, 0],
        origin="upper"
    )
    axs[0].set_title("FD reference")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("t")
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(
        series_pinn,
        aspect="auto",
        extent=[a, b, T, 0],
        origin="upper"
    )
    axs[1].set_title("PINN (ENGD)")
    axs[1].set_xlabel("x")
    fig.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(
        np.abs(series_ref - series_pinn),
        aspect="auto",
        extent=[a, b, T, 0],
        origin="upper"
    )
    axs[2].set_title("|error|")
    axs[2].set_xlabel("x")
    fig.colorbar(im2, ax=axs[2])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train(num_iterations=200)
    visualize()
