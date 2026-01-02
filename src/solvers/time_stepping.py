"""
Local time stepping for 2D incompressible RANS solver.

Δt_{i,j} = CFL * Ω_{i,j} / (Λ^I_{i,j} + Λ^J_{i,j})

Reference: Jameson, Schmidt, Turkel (1981), AIAA 81-1259.
"""

import numpy as np
from typing import NamedTuple, Optional
from dataclasses import dataclass

from src.constants import NGHOST


@dataclass
class TimeStepConfig:
    """Configuration for time stepping."""
    cfl: float = 0.8
    min_dt: float = 0.0
    max_dt: float = 1e10
    use_global_dt: bool = False


class SpectralRadius(NamedTuple):
    """Spectral radii in each direction."""
    lambda_i: np.ndarray
    lambda_j: np.ndarray


def compute_spectral_radii(Q: np.ndarray, 
                           Si_x: np.ndarray, Si_y: np.ndarray,
                           Sj_x: np.ndarray, Sj_y: np.ndarray,
                           beta: float) -> SpectralRadius:
    """Compute spectral radii Λ = |U_n| + c * |S| in I and J directions."""
    NI = Q.shape[0] - 2
    NJ = Q.shape[1] - 3
    
    int_slice = slice(NGHOST, -NGHOST)
    u = Q[int_slice, int_slice, 1]
    v = Q[int_slice, int_slice, 2]
    
    c_art = np.sqrt(u**2 + v**2 + beta)
    
    Si_mag = np.sqrt(Si_x**2 + Si_y**2)
    Si_L = Si_mag[:-1, :]
    Si_R = Si_mag[1:, :]
    Si_avg = 0.5 * (Si_L + Si_R)
    
    Si_nx_avg = 0.5 * (Si_x[:-1, :] + Si_x[1:, :]) / (Si_avg + 1e-12)
    Si_ny_avg = 0.5 * (Si_y[:-1, :] + Si_y[1:, :]) / (Si_avg + 1e-12)
    
    U_I = np.abs(u * Si_nx_avg + v * Si_ny_avg) * Si_avg
    lambda_i = U_I + c_art * Si_avg
    
    Sj_mag = np.sqrt(Sj_x**2 + Sj_y**2)
    Sj_B = Sj_mag[:, :-1]
    Sj_T = Sj_mag[:, 1:]
    Sj_avg = 0.5 * (Sj_B + Sj_T)
    
    Sj_nx_avg = 0.5 * (Sj_x[:, :-1] + Sj_x[:, 1:]) / (Sj_avg + 1e-12)
    Sj_ny_avg = 0.5 * (Sj_y[:, :-1] + Sj_y[:, 1:]) / (Sj_avg + 1e-12)
    
    U_J = np.abs(u * Sj_nx_avg + v * Sj_ny_avg) * Sj_avg
    lambda_j = U_J + c_art * Sj_avg
    
    return SpectralRadius(lambda_i=lambda_i, lambda_j=lambda_j)


def compute_local_timestep(Q: np.ndarray,
                           Si_x: np.ndarray, Si_y: np.ndarray,
                           Sj_x: np.ndarray, Sj_y: np.ndarray,
                           volume: np.ndarray,
                           beta: float,
                           cfg: Optional[TimeStepConfig] = None,
                           nu: float = 0.0) -> np.ndarray:
    """Compute local time step for each cell."""
    if cfg is None:
        cfg = TimeStepConfig()
    
    spec_rad = compute_spectral_radii(Q, Si_x, Si_y, Sj_x, Sj_y, beta)
    
    lambda_sum = spec_rad.lambda_i + spec_rad.lambda_j
    lambda_sum = np.maximum(lambda_sum, 1e-12)
    dt_conv = cfg.cfl * volume / lambda_sum
    
    if nu > 1e-12:
        Si_avg = 0.5 * (np.sqrt(Si_x[:-1, :]**2 + Si_y[:-1, :]**2) + 
                        np.sqrt(Si_x[1:, :]**2 + Si_y[1:, :]**2))
        Sj_avg = 0.5 * (np.sqrt(Sj_x[:, :-1]**2 + Sj_y[:, :-1]**2) + 
                        np.sqrt(Sj_x[:, 1:]**2 + Sj_y[:, 1:]**2))
        
        dx = volume / (Sj_avg + 1e-12)
        dy = volume / (Si_avg + 1e-12)
        
        dx_min = np.minimum(dx, dy)
        dt_visc = 0.25 * dx_min**2 / nu * cfg.cfl
        
        dt = np.minimum(dt_conv, dt_visc)
    else:
        dt = dt_conv
    
    dt = np.clip(dt, cfg.min_dt, cfg.max_dt)
    
    if cfg.use_global_dt:
        dt_global = np.min(dt)
        dt = np.full_like(dt, dt_global)
    
    return dt


def compute_global_timestep(Q: np.ndarray,
                            Si_x: np.ndarray, Si_y: np.ndarray,
                            Sj_x: np.ndarray, Sj_y: np.ndarray,
                            volume: np.ndarray,
                            beta: float,
                            cfl: float = 0.8) -> float:
    """Compute global (minimum) time step."""
    cfg = TimeStepConfig(cfl=cfl, use_global_dt=False)
    dt_local = compute_local_timestep(Q, Si_x, Si_y, Sj_x, Sj_y, volume, beta, cfg)
    return float(np.min(dt_local))


class ExplicitEuler:
    """Explicit Euler time integration with local time stepping."""
    
    def __init__(self, beta: float, cfg: Optional[TimeStepConfig] = None):
        self.beta = beta
        self.cfg = cfg if cfg is not None else TimeStepConfig()
    
    def step(self, Q: np.ndarray, residual: np.ndarray,
             Si_x: np.ndarray, Si_y: np.ndarray,
             Sj_x: np.ndarray, Sj_y: np.ndarray,
             volume: np.ndarray) -> np.ndarray:
        """Perform one explicit Euler step."""
        dt = compute_local_timestep(Q, Si_x, Si_y, Sj_x, Sj_y, 
                                    volume, self.beta, self.cfg)
        
        Q_new = Q.copy()
        Q_new[NGHOST:-NGHOST, NGHOST:-NGHOST, :] += (dt / volume)[:, :, np.newaxis] * residual
        
        return Q_new


class RungeKutta5:
    """
    5-stage Runge-Kutta (Jameson scheme).
    
    Coefficients: α = [1/4, 1/6, 3/8, 1/2, 1]
    """
    
    ALPHA = [0.25, 0.166666667, 0.375, 0.5, 1.0]
    
    def __init__(self, beta: float, cfg: Optional[TimeStepConfig] = None,
                 irs_epsilon: float = 0.0):
        self.beta = beta
        self.cfg = cfg if cfg is not None else TimeStepConfig()
        self.irs_epsilon = irs_epsilon
    
    def step(self, Q: np.ndarray, 
             compute_residual,
             apply_bc,
             Si_x: np.ndarray, Si_y: np.ndarray,
             Sj_x: np.ndarray, Sj_y: np.ndarray,
             volume: np.ndarray) -> np.ndarray:
        """Perform one RK5 step."""
        dt = compute_local_timestep(Q, Si_x, Si_y, Sj_x, Sj_y, 
                                    volume, self.beta, self.cfg)
        
        Q0 = Q.copy()
        Qk = Q.copy()
        
        if self.irs_epsilon > 0.0:
            from ..numerics.smoothing import apply_residual_smoothing
        
        for alpha in self.ALPHA:
            Qk = apply_bc(Qk)
            R = compute_residual(Qk)
            
            if self.irs_epsilon > 0.0:
                apply_residual_smoothing(R, self.irs_epsilon)
            
            Qk = Q0.copy()
            Qk[NGHOST:-NGHOST, NGHOST:-NGHOST, :] += alpha * (dt / volume)[:, :, np.newaxis] * R
        
        return Qk


RungeKutta4 = RungeKutta5
