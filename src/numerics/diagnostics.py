"""Diagnostic quantities for CFD solution analysis."""

import numpy as np
import numpy.typing as npt
from src.constants import NGHOST
from typing import NamedTuple, Dict, Tuple, Any

NDArrayFloat = npt.NDArray[np.floating]


class TotalPressureLoss(NamedTuple):
    """Total pressure loss coefficient field and statistics."""
    field: NDArrayFloat
    min: float
    max: float
    mean: float


def compute_total_pressure_loss(Q: NDArrayFloat, 
                                 p_inf: float = 0.0,
                                 u_inf: float = 1.0,
                                 v_inf: float = 0.0) -> NDArrayFloat:
    """
    Compute Total Pressure Loss Coefficient C_pt = (P0_inf - P0_local) / q_inf.
    
    For inviscid flow, C_pt should be zero (no entropy generation).
    """
    if Q.ndim != 3 or Q.shape[2] != 4:
        raise ValueError(f"Unexpected Q shape: {Q.shape}")
    
    p: NDArrayFloat = Q[:, :, 0]
    u: NDArrayFloat = Q[:, :, 1]
    v: NDArrayFloat = Q[:, :, 2]
    
    V_inf_sq: float = u_inf**2 + v_inf**2
    q_inf: float = 0.5 * V_inf_sq
    
    if q_inf < 1e-12:
        return np.zeros_like(p)
    
    P0_inf: float = p_inf + 0.5 * V_inf_sq
    P0_local: NDArrayFloat = p + 0.5 * (u**2 + v**2)
    C_pt: NDArrayFloat = (P0_inf - P0_local) / q_inf
    
    return C_pt


def compute_solution_bounds(Q: NDArrayFloat) -> Dict[str, Any]:
    """Check solution for physical bounds and anomalies."""
    Q_int: NDArrayFloat
    if Q.shape[0] > 2 and Q.shape[1] > 3:
        Q_int = Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :] if Q.shape[0] > Q.shape[1] else Q
    else:
        Q_int = Q
    
    p: NDArrayFloat = Q_int[:, :, 0]
    u: NDArrayFloat = Q_int[:, :, 1]
    v: NDArrayFloat = Q_int[:, :, 2]
    nu_t: NDArrayFloat = Q_int[:, :, 3]
    vel_mag: NDArrayFloat = np.sqrt(u**2 + v**2)
    
    return {
        'has_nan': bool(np.any(np.isnan(Q_int))),
        'has_inf': bool(np.any(np.isinf(Q_int))),
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


def compute_residual_statistics(residual: NDArrayFloat) -> Dict[str, Any]:
    """Compute RMS and max residual statistics for each variable."""
    n_cells: int = residual.shape[0] * residual.shape[1]
    rms: list[float] = [float(np.sqrt(np.sum(residual[:, :, i]**2) / n_cells)) for i in range(4)]
    
    R_p: NDArrayFloat = np.abs(residual[:, :, 0])
    max_idx: Tuple[int, ...] = np.unravel_index(np.argmax(R_p), R_p.shape)
    
    return {
        'rms_p': rms[0],
        'rms_u': rms[1],
        'rms_v': rms[2],
        'rms_nu': rms[3],
        'rms_total': float(np.sqrt(sum(r**2 for r in rms))),
        'max_p': float(R_p.max()),
        'max_loc': (int(max_idx[0]), int(max_idx[1])),
    }
