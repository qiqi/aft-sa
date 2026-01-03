"""Aerodynamic force computation: CL, CD, Cp, Cf.

JAX-based implementation.
"""

import numpy as np
import numpy.typing as npt
from typing import NamedTuple

from src.constants import NGHOST
from src.physics.jax_config import jax, jnp

NDArrayFloat = npt.NDArray[np.floating]


class AeroForces(NamedTuple):
    """Aerodynamic force coefficients."""
    CL: float
    CD: float
    CD_p: float
    CD_f: float
    CL_p: float
    CL_f: float
    Fx: float
    Fy: float


class SurfaceData(NamedTuple):
    """Surface distribution data."""
    x: NDArrayFloat
    y: NDArrayFloat
    Cp: NDArrayFloat
    Cf: NDArrayFloat


def compute_aerodynamic_forces(
    Q: np.ndarray,
    metrics,
    mu_laminar: float,
    mu_turb: np.ndarray = None,
    alpha_deg: float = 0.0,
    chord: float = 1.0,
    rho_inf: float = 1.0,
    V_inf: float = 1.0,
    n_wake: int = 0,
) -> AeroForces:
    """Compute CL, CD, CD_p, CD_f by integrating surface forces.
    
    This function wraps the JAX implementation for compatibility with NumPy arrays.
    """
    # Convert to JAX arrays
    Q_jax = jnp.asarray(Q)
    Sj_x = jnp.asarray(metrics.Sj_x)
    Sj_y = jnp.asarray(metrics.Sj_y)
    volume = jnp.asarray(metrics.volume)
    
    return compute_aerodynamic_forces_jax(
        Q_jax, Sj_x, Sj_y, volume,
        mu_laminar, mu_turb, alpha_deg, chord, rho_inf, V_inf, n_wake, NGHOST
    )


def compute_surface_distributions(
    Q: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    metrics,
    mu_laminar: float,
    mu_turb: np.ndarray = None,
    p_inf: float = 0.0,
    rho_inf: float = 1.0,
    V_inf: float = 1.0,
) -> SurfaceData:
    """Compute surface Cp and Cf distributions.
    
    This function wraps the JAX implementation for compatibility with NumPy arrays.
    """
    NI = Q.shape[0] - 2 * NGHOST
    NJ = Q.shape[1] - 2 * NGHOST
    
    if mu_turb is None:
        mu_eff = jnp.full((NI, NJ), mu_laminar)
    else:
        mu_eff = jnp.asarray(mu_laminar + np.maximum(0.0, mu_turb))
    
    q_inf = 0.5 * rho_inf * V_inf**2
    if q_inf < 1e-14:
        q_inf = 1e-14
    
    Q_jax = jnp.asarray(Q)
    Sj_x = jnp.asarray(metrics.Sj_x)
    Sj_y = jnp.asarray(metrics.Sj_y)
    volume = jnp.asarray(metrics.volume)
    
    Cp, Cf = compute_surface_cp_cf_jax(Q_jax, Sj_x, Sj_y, volume, mu_eff, p_inf, q_inf, NGHOST)
    
    x_surf = 0.5 * (X[:-1, 0] + X[1:, 0])
    y_surf = 0.5 * (Y[:-1, 0] + Y[1:, 0])
    
    return SurfaceData(x=x_surf, y=y_surf, Cp=np.asarray(Cp), Cf=np.asarray(Cf))


def create_surface_vtk_fields(
    Q: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    metrics,
    mu_laminar: float,
    mu_turb: np.ndarray = None,
    p_inf: float = 0.0,
    rho_inf: float = 1.0,
    V_inf: float = 1.0,
) -> dict:
    """Create Cp and Cf fields for VTK output (values meaningful only at j=0)."""
    NI = Q.shape[0] - 2 * NGHOST
    NJ = Q.shape[1] - 2 * NGHOST
    
    surf = compute_surface_distributions(
        Q, X, Y, metrics, mu_laminar, mu_turb, p_inf, rho_inf, V_inf
    )
    
    Cp_field = np.full((NI, NJ), np.nan)
    Cf_field = np.full((NI, NJ), np.nan)
    
    Cp_field[:, 0] = surf.Cp
    Cf_field[:, 0] = surf.Cf
    
    return {'SurfaceCp': Cp_field, 'SurfaceCf': Cf_field}


# =============================================================================
# JAX Implementation
# =============================================================================

@jax.jit
def _compute_surface_forces_jax_kernel(p_wall, u_int, v_int, Sx, Sy, vol, mu, mask):
    """JIT-compiled kernel for surface forces."""
    area = jnp.sqrt(Sx**2 + Sy**2)
    
    # Pressure forces (negative because outward normal)
    Fx_p_arr = -p_wall * Sx
    Fy_p_arr = -p_wall * Sy
    
    # Viscous forces
    dy = vol / (area + 1e-14)
    
    # Wall shear (assuming zero wall velocity)
    dudn = 2.0 * u_int / dy
    dvdn = 2.0 * v_int / dy
    
    tau_x = mu * dudn
    tau_y = mu * dvdn
    
    Fx_v_arr = tau_x * area
    Fy_v_arr = tau_y * area
    
    # Also exclude cells with negligible area
    mask = jnp.logical_and(mask, area > 1e-14)
    
    Fx_p = jnp.sum(jnp.where(mask, Fx_p_arr, 0.0))
    Fy_p = jnp.sum(jnp.where(mask, Fy_p_arr, 0.0))
    Fx_v = jnp.sum(jnp.where(mask, Fx_v_arr, 0.0))
    Fy_v = jnp.sum(jnp.where(mask, Fy_v_arr, 0.0))
    
    return Fx_p, Fy_p, Fx_v, Fy_v


def compute_surface_forces_jax(Q, Sj_x, Sj_y, volume, mu_eff, n_wake, nghost):
    """
    JAX: Compute pressure and viscous forces on airfoil surface (j=0).
    
    Parameters
    ----------
    Q : jnp.ndarray
        State array (NI+2*nghost, NJ+2*nghost, 4).
    Sj_x, Sj_y : jnp.ndarray
        J-face normal vectors (NI, NJ+1).
    volume : jnp.ndarray
        Cell volumes (NI, NJ).
    mu_eff : jnp.ndarray
        Effective viscosity (NI, NJ).
    n_wake : int
        Number of wake points to exclude.
    nghost : int
        Number of ghost cells.
        
    Returns
    -------
    Fx_p, Fy_p, Fx_v, Fy_v : float
        Pressure and viscous force components.
    """
    NI = Q.shape[0] - 2 * nghost
    
    # Extract arrays with concrete indices
    Sx = Sj_x[:, 0]
    Sy = Sj_y[:, 0]
    p_wall = Q[nghost:nghost+NI, nghost, 0]
    u_int = Q[nghost:nghost+NI, nghost, 1]
    v_int = Q[nghost:nghost+NI, nghost, 2]
    vol = volume[:, 0]
    mu = mu_eff[:, 0]
    
    # Create mask with concrete n_wake
    mask = jnp.ones(NI, dtype=bool)
    if n_wake > 0:
        mask = mask.at[:n_wake].set(False)
        mask = mask.at[-n_wake:].set(False)
    
    return _compute_surface_forces_jax_kernel(
        p_wall, u_int, v_int, Sx, Sy, vol, mu, mask
    )


def compute_aerodynamic_forces_jax(
    Q,
    Sj_x, Sj_y, volume,
    mu_laminar: float,
    mu_turb=None,
    alpha_deg: float = 0.0,
    chord: float = 1.0,
    rho_inf: float = 1.0,
    V_inf: float = 1.0,
    n_wake: int = 0,
    nghost: int = NGHOST,
) -> AeroForces:
    """
    JAX: Compute CL, CD, CD_p, CD_f by integrating surface forces.
    """
    NI = Q.shape[0] - 2 * nghost
    NJ = Q.shape[1] - 2 * nghost
    
    if mu_turb is None:
        mu_eff = jnp.full((NI, NJ), mu_laminar)
    else:
        mu_eff = mu_laminar + jnp.maximum(0.0, mu_turb)
    
    Fx_p, Fy_p, Fx_v, Fy_v = compute_surface_forces_jax(
        Q, Sj_x, Sj_y, volume, mu_eff, n_wake, nghost
    )
    
    Fx = Fx_p + Fx_v
    Fy = Fy_p + Fy_v
    
    alpha = jnp.deg2rad(alpha_deg)
    cos_a = jnp.cos(alpha)
    sin_a = jnp.sin(alpha)
    
    D_total = Fx * cos_a + Fy * sin_a
    L_total = Fy * cos_a - Fx * sin_a
    D_p = Fx_p * cos_a + Fy_p * sin_a
    L_p = Fy_p * cos_a - Fx_p * sin_a
    D_f = Fx_v * cos_a + Fy_v * sin_a
    L_f = Fy_v * cos_a - Fx_v * sin_a
    
    q_inf = 0.5 * rho_inf * V_inf**2
    ref_area = chord * 1.0
    
    CL = float(L_total / (q_inf * ref_area))
    CD = float(D_total / (q_inf * ref_area))
    CD_p = float(D_p / (q_inf * ref_area))
    CD_f = float(D_f / (q_inf * ref_area))
    CL_p = float(L_p / (q_inf * ref_area))
    CL_f = float(L_f / (q_inf * ref_area))
    
    return AeroForces(CL=CL, CD=CD, CD_p=CD_p, CD_f=CD_f, 
                      CL_p=CL_p, CL_f=CL_f, Fx=float(Fx), Fy=float(Fy))


@jax.jit
def compute_surface_cp_cf_jax(Q, Sj_x, Sj_y, volume, mu_eff, p_inf, q_inf, nghost):
    """
    JAX: Compute surface Cp and Cf distributions.
    
    Returns
    -------
    Cp, Cf : jnp.ndarray
        Pressure and skin friction coefficients (NI,).
    """
    NI = Q.shape[0] - 2 * nghost
    
    Sx = Sj_x[:, 0]
    Sy = Sj_y[:, 0]
    area = jnp.sqrt(Sx**2 + Sy**2)
    
    p_wall = Q[nghost:nghost+NI, nghost, 0]
    Cp = (p_wall - p_inf) / q_inf
    
    vol = volume[:, 0]
    dy = vol / (area + 1e-14)
    
    u_int = Q[nghost:nghost+NI, nghost, 1]
    v_int = Q[nghost:nghost+NI, nghost, 2]
    
    dudn = 2.0 * u_int / dy
    dvdn = 2.0 * v_int / dy
    
    mu = mu_eff[:, 0]
    tau_mag = mu * jnp.sqrt(dudn**2 + dvdn**2)
    Cf = tau_mag / q_inf
    
    # Handle degenerate cells
    Cp = jnp.where(area < 1e-14, 0.0, Cp)
    Cf = jnp.where(area < 1e-14, 0.0, Cf)
    
    return Cp, Cf
