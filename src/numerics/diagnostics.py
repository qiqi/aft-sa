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


class WakeCutDiscontinuity(NamedTuple):
    """Wake cut discontinuity statistics."""
    delta_p: NDArrayFloat  # Pressure difference across cut for each cell pair
    delta_p_max: float      # Maximum pressure discontinuity
    delta_nu: NDArrayFloat  # NuHat difference across cut
    delta_nu_max: float     # Maximum NuHat discontinuity
    delta_u_max: float      # Maximum U-velocity discontinuity
    delta_v_max: float      # Maximum V-velocity discontinuity
    delta_x_max: float      # Maximum Grid X-coordinate discontinuity
    i_max: int              # Index of max discontinuity (in wake region)
    x_max: float            # x-coordinate of max discontinuity (if coords provided)


def compute_wake_cut_discontinuity(
    Q: NDArrayFloat,
    n_wake: int,
    X: NDArrayFloat = None,
    nghost: int = NGHOST
) -> WakeCutDiscontinuity:
    """
    Compute solution and grid discontinuity across the C-mesh wake cut.
    """
    NI = Q.shape[0] - 2 * nghost
    j_int = nghost  # First interior j-index (wall-adjacent)
    
    # Lower wake: cells [nghost:nghost+n_wake] at j=j_int
    # Upper wake: cells [NI+nghost-n_wake:NI+nghost] at j=j_int (reversed pairing)
    i_lower_start = nghost
    i_lower_end = nghost + n_wake
    i_upper_start = NI + nghost - n_wake
    i_upper_end = NI + nghost
    
    # 1. Pressure Analysis
    p_lower = Q[i_lower_start:i_lower_end, j_int, 0]
    p_upper = Q[i_upper_start:i_upper_end, j_int, 0][::-1]
    delta_p = np.abs(p_lower - p_upper)
    i_max_local_p = int(np.argmax(delta_p))
    delta_p_max = float(delta_p[i_max_local_p])
    
    # 2. NuHat Analysis
    nu_lower = Q[i_lower_start:i_lower_end, j_int, 3]
    nu_upper = Q[i_upper_start:i_upper_end, j_int, 3][::-1]
    delta_nu = np.abs(nu_lower - nu_upper)
    delta_nu_max = float(np.max(delta_nu))
    
    # 3. Velocity Analysis
    u_lower = Q[i_lower_start:i_lower_end, j_int, 1]
    u_upper = Q[i_upper_start:i_upper_end, j_int, 1][::-1]
    delta_u_max = float(np.max(np.abs(u_lower - u_upper)))
    
    v_lower = Q[i_lower_start:i_lower_end, j_int, 2]
    v_upper = Q[i_upper_start:i_upper_end, j_int, 2][::-1]
    delta_v_max = float(np.max(np.abs(v_lower - v_upper)))
    
    # 4. Grid Coordinate Analysis (Interior Nodes)
    # Check if the grid itself is symmetric/matched at the cut
    delta_x_max = 0.0
    if X is not None:
        # Check node symmetry directly using n_wake
        # Lower Wake Nodes: 0 to n_wake (inclusive)
        # Upper Wake Nodes: NI to NI-n_wake (inclusive, reversed)
        
        # We need N (number of nodes in X)
        N_nodes = X.shape[0] # NI + 1 usually
        if N_nodes > 2 * n_wake:
            x_lower = X[0 : n_wake + 1, j_int - nghost]
            # Upper: NI, NI-1 ... NI-n_wake
            x_upper = X[N_nodes - 1 - n_wake : N_nodes, j_int - nghost][::-1]
            
            if x_lower.shape == x_upper.shape:
                delta_x_max = float(np.max(np.abs(x_lower - x_upper)))
    
    # Find global location of max PRESSURE discontinuity
    i_max_global = i_lower_start + i_max_local_p
    
    # Get x-coordinate if available
    x_max = 0.0
    if X is not None:
        if X.shape[0] > i_max_global - nghost + 1:
            x_max = float(0.5 * (X[i_max_global - nghost, j_int - nghost] + 
                                  X[i_max_global - nghost + 1, j_int - nghost]))
    
    return WakeCutDiscontinuity(
        delta_p=delta_p,
        delta_p_max=delta_p_max,
        delta_nu=delta_nu,
        delta_nu_max=delta_nu_max,
        delta_u_max=delta_u_max,
        delta_v_max=delta_v_max,
        delta_x_max=delta_x_max,
        i_max=i_max_local_p,
        x_max=x_max,
    )


class NuHatDiagnostic(NamedTuple):
    """Detailed diagnostic for nuHat residual hotspot."""
    max_res: float
    i_max: int
    j_max: int
    x_loc: float
    y_loc: float
    nu_hat: float
    p_res: float # Pressure residual at same location

def analyze_nuhat_residual(
    residual: NDArrayFloat,
    Q: NDArrayFloat,
    X: NDArrayFloat,
    Y: NDArrayFloat,
    nghost: int = NGHOST
) -> NuHatDiagnostic:
    """
    Analyze the nuHat residual field to find the hotspot and local flow properties.
    """
    # Handle ghost cells in residual if present
    if residual.shape[0] == Q.shape[0]:
        R_int = residual[nghost:-nghost, nghost:-nghost, :]
    else:
        R_int = residual

    # Handle Q
    Q_int = Q[nghost:-nghost, nghost:-nghost, :]
    
    # 1. Find Max NuHat Residual Location
    R_nu = np.abs(R_int[:, :, 3])
    max_idx = np.unravel_index(np.argmax(R_nu), R_nu.shape)
    i_max, j_max = max_idx
    max_res = float(R_nu[i_max, j_max])
    
    # 2. Get Coordinates
    x_loc = 0.0
    y_loc = 0.0
    # X is usually node coordinates (NI+1, NJ+1)
    if X is not None and X.shape[0] > i_max + nghost:
        # Approximate cell center
        x_loc = float(0.25 * (X[i_max, j_max] + X[i_max+1, j_max] + X[i_max, j_max+1] + X[i_max+1, j_max+1]))
        y_loc = float(0.25 * (Y[i_max, j_max] + Y[i_max+1, j_max] + Y[i_max, j_max+1] + Y[i_max+1, j_max+1]))

    # 3. Local Flow State
    q_loc = Q_int[i_max, j_max, :]
    nu_hat_loc = float(q_loc[3])
    
    # Pressure residual at same location
    p_res = float(np.abs(R_int[i_max, j_max, 0]))

    return NuHatDiagnostic(
        max_res=max_res,
        i_max=int(i_max),
        j_max=int(j_max),
        x_loc=x_loc,
        y_loc=y_loc,
        nu_hat=nu_hat_loc,
        p_res=p_res
    )
