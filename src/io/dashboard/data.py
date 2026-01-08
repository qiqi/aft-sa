"""
Data structures and management for the CFD Dashboard.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Any, Union
from ...constants import NGHOST
from .._array_utils import sanitize_array, safe_minmax, safe_absmax

@dataclass
class Snapshot:
    """Stores a single timestep's solution data (interior cells only)."""
    iteration: int
    residual: np.ndarray  # 4-component: (p, u, v, nuHat) RMS residuals
    cfl: float
    p: np.ndarray  # Relative pressure (p - p_inf)
    u: np.ndarray  # Relative u-velocity (u - u_inf)
    v: np.ndarray  # Relative v-velocity (v - v_inf)
    nu: np.ndarray
    C_pt: Optional[np.ndarray] = None
    residual_field: Optional[np.ndarray] = None
    vel_max: float = 0.0
    p_min: float = 0.0
    p_max: float = 0.0
    # AFT diagnostic fields (optional)
    Re_Omega: Optional[np.ndarray] = None  # Vorticity Reynolds number
    Gamma: Optional[np.ndarray] = None     # Shape factor for AFT
    is_turb: Optional[np.ndarray] = None   # Turbulent fraction (0=laminar, 1=turbulent)


class DataManager:
    """Manages collection and processing of snapshots and residual history."""

    def __init__(self, p_inf: float = 0.0, u_inf: float = 1.0, v_inf: float = 0.0):
        self.snapshots: List[Snapshot] = []
        self.residual_history: List[np.ndarray] = []
        self.iteration_history: List[int] = []
        self.divergence_snapshots: List[Snapshot] = []
        
        self.p_inf = p_inf
        self.u_inf = u_inf
        self.v_inf = v_inf

    def update_freestream(self, p_inf: float, u_inf: float, v_inf: float) -> None:
        """Update freestream reference values."""
        self.p_inf = p_inf
        self.u_inf = u_inf
        self.v_inf = v_inf

    def store_snapshot(
        self, 
        Q: np.ndarray, 
        iteration: int, 
        residual_history: List,
        cfl: float = 0.0,
        C_pt: Optional[np.ndarray] = None,
        residual_field: Optional[np.ndarray] = None,
        iteration_history: Optional[List[int]] = None,
        is_divergence_dump: bool = False,
        Re_Omega: Optional[np.ndarray] = None,
        Gamma: Optional[np.ndarray] = None,
        is_turb: Optional[np.ndarray] = None,
    ) -> None:
        """Store current solution state with diagnostic data."""
        Q_int = Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
        
        # Get residual
        residual = self._parse_residual(residual_history)
        
        # Create snapshot
        snapshot = self._create_snapshot(
            Q_int, iteration, residual, cfl, C_pt, residual_field,
            Re_Omega, Gamma, is_turb
        )
        
        if is_divergence_dump:
            self.divergence_snapshots.append(snapshot)
        else:
            self.snapshots.append(snapshot)
        
        # Store residual history
        self._update_residual_history(residual_history, iteration_history)

    def _parse_residual(self, residual_history: List) -> np.ndarray:
        """Parse residual from history list."""
        if not residual_history:
            return np.zeros(4)
        
        last_res = residual_history[-1]
        if isinstance(last_res, (tuple, list, np.ndarray)):
            return np.array([r if np.isfinite(r) else 1e10 for r in last_res])
        else:
            val = last_res if np.isfinite(last_res) else 1e10
            return np.array([val] * 4)

    def _create_snapshot(
        self,
        Q_int: np.ndarray,
        iteration: int,
        residual: np.ndarray,
        cfl: float,
        C_pt: Optional[np.ndarray],
        residual_field: Optional[np.ndarray],
        Re_Omega: Optional[np.ndarray] = None,
        Gamma: Optional[np.ndarray] = None,
        is_turb: Optional[np.ndarray] = None,
    ) -> Snapshot:
        """Create a Snapshot from solution data."""
        max_safe_vel = 1e10
        
        # Relative values
        p_rel = sanitize_array(Q_int[:, :, 0] - self.p_inf, fill_value=0.0)
        u_rel = sanitize_array(Q_int[:, :, 1] - self.u_inf, fill_value=0.0)
        v_rel = sanitize_array(Q_int[:, :, 2] - self.v_inf, fill_value=0.0)
        nu = sanitize_array(Q_int[:, :, 3], fill_value=0.0)
        
        # Velocity magnitude
        u_abs = np.clip(sanitize_array(Q_int[:, :, 1], fill_value=self.u_inf), -max_safe_vel, max_safe_vel)
        v_abs = np.clip(sanitize_array(Q_int[:, :, 2], fill_value=self.v_inf), -max_safe_vel, max_safe_vel)
        vel_mag = np.sqrt(u_abs**2 + v_abs**2)
        
        # C_pt computation
        if C_pt is None:
            C_pt = self._compute_cpt(Q_int, u_abs, v_abs)
        else:
            C_pt = sanitize_array(C_pt, fill_value=0.0)
        
        # Residual field (already scaled RMS if from get_scaled_residual_field)
        res_field = None
        if residual_field is not None:
            # Handle both old-style 4-component arrays and new-style 2D scalar arrays
            if residual_field.ndim == 3:
                # Legacy: 4-component array, compute RMS across equations
                res_clipped = np.clip(residual_field, -max_safe_vel, max_safe_vel)
                res_field = sanitize_array(np.sqrt(np.mean(res_clipped**2, axis=2)), fill_value=1e-12)
            else:
                # New: already a 2D scalar field (scaled RMS)
                res_clipped = np.clip(residual_field, 0, max_safe_vel)
                res_field = sanitize_array(res_clipped, fill_value=1e-12)
        
        # AFT diagnostic fields
        Re_Omega_arr = None
        if Re_Omega is not None:
            Re_Omega_arr = sanitize_array(Re_Omega, fill_value=1.0).copy()
        Gamma_arr = None
        if Gamma is not None:
            Gamma_arr = sanitize_array(Gamma, fill_value=0.0).copy()
        is_turb_arr = None
        if is_turb is not None:
            is_turb_arr = sanitize_array(is_turb, fill_value=0.0).copy()
        
        return Snapshot(
            iteration=iteration,
            residual=residual,
            cfl=cfl,
            p=p_rel.copy(),
            u=u_rel.copy(),
            v=v_rel.copy(),
            nu=nu.copy(),
            C_pt=C_pt.copy() if C_pt is not None else None,
            residual_field=res_field,
            vel_max=float(safe_absmax(vel_mag, default=1.0)),
            p_min=float(safe_minmax(p_rel)[0]),
            p_max=float(safe_minmax(p_rel)[1]),
            Re_Omega=Re_Omega_arr,
            Gamma=Gamma_arr,
            is_turb=is_turb_arr,
        )

    def _compute_cpt(self, Q_int: np.ndarray, u_abs: np.ndarray, v_abs: np.ndarray) -> Optional[np.ndarray]:
        """Compute total pressure loss coefficient."""
        V_inf_sq = self.u_inf**2 + self.v_inf**2
        if V_inf_sq < 1e-14:
            return None
        
        p = sanitize_array(Q_int[:, :, 0], fill_value=self.p_inf)
        p = np.clip(p, -1e10, 1e10)
        p_total = p + 0.5 * (u_abs**2 + v_abs**2)
        p_total_inf = self.p_inf + 0.5 * V_inf_sq
        C_pt = (p_total_inf - p_total) / (0.5 * V_inf_sq)
        return sanitize_array(C_pt, fill_value=0.0)

    def _update_residual_history(self, residual_history: List, iteration_history: Optional[List[int]]) -> None:
        """Update stored residual history."""
        self.residual_history = []
        for r in residual_history:
            if isinstance(r, (tuple, list, np.ndarray)):
                self.residual_history.append(np.array([v if np.isfinite(v) else 1e10 for v in r]))
            else:
                val = r if np.isfinite(r) else 1e10
                self.residual_history.append(np.array([val, val, val, val]))
        
        if iteration_history is not None:
            self.iteration_history = list(iteration_history)
        elif len(self.iteration_history) != len(self.residual_history):
            self.iteration_history = list(range(len(self.residual_history)))
    
    def get_all_snapshots(self) -> List[Snapshot]:
        """Get all snapshots sorted by iteration."""
        all_snapshots = list(self.snapshots)
        if self.divergence_snapshots:
            all_snapshots.extend(self.divergence_snapshots)
            all_snapshots.sort(key=lambda s: s.iteration)
        return all_snapshots
