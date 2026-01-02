"""Aerodynamic force computation: CL, CD, Cp, Cf."""

import numpy as np
from numba import njit
from typing import NamedTuple

from src.constants import NGHOST


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
    x: np.ndarray
    y: np.ndarray
    Cp: np.ndarray
    Cf: np.ndarray


@njit(cache=True, fastmath=True)
def _compute_surface_forces_kernel(
    Q: np.ndarray,
    Sj_x: np.ndarray,
    Sj_y: np.ndarray,
    volume: np.ndarray,
    mu_eff: np.ndarray,
    n_wake: int = 0,
) -> tuple:
    """Compute pressure and viscous forces on airfoil surface (j=0)."""
    NI = Q.shape[0] - 2 * NGHOST
    
    Fx_p = 0.0
    Fy_p = 0.0
    Fx_v = 0.0
    Fy_v = 0.0
    
    i_start = n_wake if n_wake > 0 else 0
    i_end = NI - n_wake if n_wake > 0 else NI
    
    for i in range(i_start, i_end):
        Sx = Sj_x[i, 0]
        Sy = Sj_y[i, 0]
        area = np.sqrt(Sx*Sx + Sy*Sy)
        
        if area < 1e-14:
            continue
            
        nx = Sx / area
        ny = Sy / area
        
        p_wall = Q[i + NGHOST, NGHOST, 0]
        Fx_p += -p_wall * Sx
        Fy_p += -p_wall * Sy
        
        vol = volume[i, 0]
        dy = vol / area
        
        u_int = Q[i + NGHOST, NGHOST, 1]
        v_int = Q[i + NGHOST, NGHOST, 2]
        
        dudn = 2.0 * u_int / dy
        dvdn = 2.0 * v_int / dy
        
        mu = mu_eff[i, 0]
        tau_x = mu * dudn
        tau_y = mu * dvdn
        
        Fx_v += tau_x * area
        Fy_v += tau_y * area
    
    return Fx_p, Fy_p, Fx_v, Fy_v


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
    """Compute CL, CD, CD_p, CD_f by integrating surface forces."""
    NI = Q.shape[0] - 2 * NGHOST
    NJ = Q.shape[1] - 2 * NGHOST
    
    if mu_turb is None:
        mu_eff = np.full((NI, NJ), mu_laminar)
    else:
        mu_eff = mu_laminar + np.maximum(0.0, mu_turb)
    
    Fx_p, Fy_p, Fx_v, Fy_v = _compute_surface_forces_kernel(
        Q, metrics.Sj_x, metrics.Sj_y, metrics.volume, mu_eff, n_wake
    )
    
    Fx = Fx_p + Fx_v
    Fy = Fy_p + Fy_v
    
    alpha = np.deg2rad(alpha_deg)
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)
    
    D_total = Fx * cos_a + Fy * sin_a
    L_total = Fy * cos_a - Fx * sin_a
    D_p = Fx_p * cos_a + Fy_p * sin_a
    L_p = Fy_p * cos_a - Fx_p * sin_a
    D_f = Fx_v * cos_a + Fy_v * sin_a
    L_f = Fy_v * cos_a - Fx_v * sin_a
    
    q_inf = 0.5 * rho_inf * V_inf**2
    ref_area = chord * 1.0
    
    CL = L_total / (q_inf * ref_area)
    CD = D_total / (q_inf * ref_area)
    CD_p = D_p / (q_inf * ref_area)
    CD_f = D_f / (q_inf * ref_area)
    CL_p = L_p / (q_inf * ref_area)
    CL_f = L_f / (q_inf * ref_area)
    
    return AeroForces(CL=CL, CD=CD, CD_p=CD_p, CD_f=CD_f, CL_p=CL_p, CL_f=CL_f, Fx=Fx, Fy=Fy)


@njit(cache=True, fastmath=True)
def _compute_surface_cp_cf_kernel(
    Q: np.ndarray,
    Sj_x: np.ndarray,
    Sj_y: np.ndarray,
    volume: np.ndarray,
    mu_eff: np.ndarray,
    p_inf: float,
    q_inf: float,
    Cp_out: np.ndarray,
    Cf_out: np.ndarray,
) -> None:
    """Compute surface Cp and Cf distributions."""
    NI = Q.shape[0] - 2 * NGHOST
    
    for i in range(NI):
        Sx = Sj_x[i, 0]
        Sy = Sj_y[i, 0]
        area = np.sqrt(Sx*Sx + Sy*Sy)
        
        if area < 1e-14:
            Cp_out[i] = 0.0
            Cf_out[i] = 0.0
            continue
        
        p_wall = Q[i + NGHOST, NGHOST, 0]
        Cp_out[i] = (p_wall - p_inf) / q_inf
        
        vol = volume[i, 0]
        dy = vol / area
        
        u_int = Q[i + NGHOST, NGHOST, 1]
        v_int = Q[i + NGHOST, NGHOST, 2]
        
        dudn = 2.0 * u_int / dy
        dvdn = 2.0 * v_int / dy
        
        mu = mu_eff[i, 0]
        tau_mag = mu * np.sqrt(dudn*dudn + dvdn*dvdn)
        Cf_out[i] = tau_mag / q_inf


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
    """Compute surface Cp and Cf distributions."""
    NI = Q.shape[0] - 2 * NGHOST
    NJ = Q.shape[1] - 2 * NGHOST
    
    if mu_turb is None:
        mu_eff = np.full((NI, NJ), mu_laminar)
    else:
        mu_eff = mu_laminar + np.maximum(0.0, mu_turb)
    
    q_inf = 0.5 * rho_inf * V_inf**2
    if q_inf < 1e-14:
        q_inf = 1e-14
    
    Cp = np.zeros(NI)
    Cf = np.zeros(NI)
    
    _compute_surface_cp_cf_kernel(
        Q, metrics.Sj_x, metrics.Sj_y, metrics.volume,
        mu_eff, p_inf, q_inf, Cp, Cf
    )
    
    x_surf = 0.5 * (X[:-1, 0] + X[1:, 0])
    y_surf = 0.5 * (Y[:-1, 0] + Y[1:, 0])
    
    return SurfaceData(x=x_surf, y=y_surf, Cp=Cp, Cf=Cf)


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
    from src.constants import NGHOST
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
