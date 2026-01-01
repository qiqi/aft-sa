"""
Diagnostic quantities for CFD solution analysis.

This module provides functions to compute diagnostic quantities
useful for solution verification and debugging.
"""

import numpy as np
from typing import NamedTuple


class TotalPressureLoss(NamedTuple):
    """Total pressure loss coefficient field and statistics."""
    field: np.ndarray    # C_pt at each cell
    min: float
    max: float
    mean: float


def compute_total_pressure_loss(Q: np.ndarray, 
                                 p_inf: float = 0.0,
                                 u_inf: float = 1.0,
                                 v_inf: float = 0.0) -> np.ndarray:
    """
    Compute Total Pressure Loss Coefficient (C_pt) for entropy check.
    
    For inviscid flow, C_pt should be zero everywhere (no entropy generation).
    Non-zero values indicate spurious entropy production (numerical error).
    
    Physics (Incompressible, ρ=1):
        P_0 = p + 0.5 * (u² + v²)           (local total pressure)
        P_0_inf = p_inf + 0.5 * V_inf²      (freestream total pressure)
        q_inf = 0.5 * V_inf²                 (dynamic pressure)
        C_pt = (P_0_inf - P_0_local) / q_inf (total pressure loss coefficient)
    
    Parameters
    ----------
    Q : ndarray, shape (NI, NJ, 4) or (NI+2, NJ+2, 4)
        State vector [p, u, v, nu_t]. Can include ghost cells.
    p_inf : float
        Freestream pressure (default 0.0 for AC formulation).
    u_inf : float
        Freestream x-velocity.
    v_inf : float
        Freestream y-velocity.
        
    Returns
    -------
    C_pt : ndarray
        Total pressure loss coefficient at each cell.
        Positive = entropy production (energy loss)
        Negative = entropy destruction (unphysical)
    """
    # Extract state components
    if Q.ndim != 3 or Q.shape[2] != 4:
        raise ValueError(f"Unexpected Q shape: {Q.shape}")
    
    p = Q[:, :, 0]
    u = Q[:, :, 1]
    v = Q[:, :, 2]
    
    # Dynamic pressure (assume rho = 1.0 for AC formulation)
    V_inf_sq = u_inf**2 + v_inf**2
    q_inf = 0.5 * V_inf_sq
    
    if q_inf < 1e-12:
        return np.zeros_like(p)
    
    # Freestream total pressure
    P0_inf = p_inf + 0.5 * V_inf_sq
    
    # Local total pressure
    P0_local = p + 0.5 * (u**2 + v**2)
    
    # Total pressure loss coefficient
    C_pt = (P0_inf - P0_local) / q_inf
    
    return C_pt


def compute_solution_bounds(Q: np.ndarray) -> dict:
    """
    Check solution for physical bounds and anomalies.
    
    Parameters
    ----------
    Q : ndarray, shape (NI+2, NJ+2, 4) or (NI, NJ, 4)
        State vector [p, u, v, nu_t].
        
    Returns
    -------
    info : dict
        Dictionary with bounds and anomaly flags.
    """
    # Strip ghost cells if present (heuristic: check if shape suggests ghosts)
    if Q.shape[0] > 2 and Q.shape[1] > 2:
        Q_int = Q[1:-1, 1:-1, :] if Q.shape[0] > Q.shape[1] else Q
    else:
        Q_int = Q
    
    p = Q_int[:, :, 0]
    u = Q_int[:, :, 1]
    v = Q_int[:, :, 2]
    nu_t = Q_int[:, :, 3]
    vel_mag = np.sqrt(u**2 + v**2)
    
    info = {
        'has_nan': np.any(np.isnan(Q_int)),
        'has_inf': np.any(np.isinf(Q_int)),
        'p_min': float(p.min()),
        'p_max': float(p.max()),
        'u_min': float(u.min()),
        'u_max': float(u.max()),
        'v_min': float(v.min()),
        'v_max': float(v.max()),
        'nu_min': float(nu_t.min()),
        'nu_max': float(nu_t.max()),
        'vel_max': float(vel_mag.max()),
        'vel_max_loc': tuple(int(x) for x in np.unravel_index(np.argmax(vel_mag), vel_mag.shape)),
    }
    
    return info


def compute_residual_statistics(residual: np.ndarray) -> dict:
    """
    Compute statistics of residual field.
    
    Parameters
    ----------
    residual : ndarray, shape (NI, NJ, 4)
        Residual for each variable.
        
    Returns
    -------
    stats : dict
        Statistics for each variable and overall.
    """
    n_cells = residual.shape[0] * residual.shape[1]
    
    # RMS for each variable
    rms = [np.sqrt(np.sum(residual[:, :, i]**2) / n_cells) for i in range(4)]
    
    # Max residual location (using continuity/pressure)
    R_p = np.abs(residual[:, :, 0])
    max_idx = np.unravel_index(np.argmax(R_p), R_p.shape)
    
    return {
        'rms_p': rms[0],
        'rms_u': rms[1],
        'rms_v': rms[2],
        'rms_nu': rms[3],
        'rms_total': np.sqrt(sum(r**2 for r in rms)),
        'max_p': float(R_p.max()),
        'max_loc': (int(max_idx[0]), int(max_idx[1])),
    }

