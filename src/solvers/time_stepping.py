"""
Local time stepping for 2D incompressible RANS solver.

Δt_{i,j} = CFL * Ω_{i,j} / (Λ^I_{i,j} + Λ^J_{i,j})

Reference: Jameson, Schmidt, Turkel (1981), AIAA 81-1259.
"""

import numpy as np
import numpy.typing as npt
from typing import NamedTuple, Optional, Callable
from dataclasses import dataclass

from src.constants import NGHOST

NDArrayFloat = npt.NDArray[np.floating]


@dataclass
class TimeStepConfig:
    """Configuration for time stepping."""
    cfl: float = 0.8
    min_dt: float = 0.0
    max_dt: float = 1e10
    use_global_dt: bool = False


class SpectralRadius(NamedTuple):
    """Spectral radii in each direction."""
    lambda_i: NDArrayFloat
    lambda_j: NDArrayFloat


def compute_spectral_radii(Q: NDArrayFloat, 
                           Si_x: NDArrayFloat, Si_y: NDArrayFloat,
                           Sj_x: NDArrayFloat, Sj_y: NDArrayFloat,
                           beta: float) -> SpectralRadius:
    """Compute spectral radii Λ = |U_n| + c * |S| in I and J directions."""
    NI: int = Q.shape[0] - 2
    NJ: int = Q.shape[1] - 3
    
    int_slice: slice = slice(NGHOST, -NGHOST)
    u: NDArrayFloat = Q[int_slice, int_slice, 1]
    v: NDArrayFloat = Q[int_slice, int_slice, 2]
    
    c_art: NDArrayFloat = np.sqrt(u**2 + v**2 + beta)
    
    Si_mag: NDArrayFloat = np.sqrt(Si_x**2 + Si_y**2)
    Si_L: NDArrayFloat = Si_mag[:-1, :]
    Si_R: NDArrayFloat = Si_mag[1:, :]
    Si_avg: NDArrayFloat = 0.5 * (Si_L + Si_R)
    
    Si_nx_avg: NDArrayFloat = 0.5 * (Si_x[:-1, :] + Si_x[1:, :]) / (Si_avg + 1e-12)
    Si_ny_avg: NDArrayFloat = 0.5 * (Si_y[:-1, :] + Si_y[1:, :]) / (Si_avg + 1e-12)
    
    U_I: NDArrayFloat = np.abs(u * Si_nx_avg + v * Si_ny_avg) * Si_avg
    lambda_i: NDArrayFloat = U_I + c_art * Si_avg
    
    Sj_mag: NDArrayFloat = np.sqrt(Sj_x**2 + Sj_y**2)
    Sj_B: NDArrayFloat = Sj_mag[:, :-1]
    Sj_T: NDArrayFloat = Sj_mag[:, 1:]
    Sj_avg: NDArrayFloat = 0.5 * (Sj_B + Sj_T)
    
    Sj_nx_avg: NDArrayFloat = 0.5 * (Sj_x[:, :-1] + Sj_x[:, 1:]) / (Sj_avg + 1e-12)
    Sj_ny_avg: NDArrayFloat = 0.5 * (Sj_y[:, :-1] + Sj_y[:, 1:]) / (Sj_avg + 1e-12)
    
    U_J: NDArrayFloat = np.abs(u * Sj_nx_avg + v * Sj_ny_avg) * Sj_avg
    lambda_j: NDArrayFloat = U_J + c_art * Sj_avg
    
    return SpectralRadius(lambda_i=lambda_i, lambda_j=lambda_j)


def compute_local_timestep(Q: NDArrayFloat,
                           Si_x: NDArrayFloat, Si_y: NDArrayFloat,
                           Sj_x: NDArrayFloat, Sj_y: NDArrayFloat,
                           volume: NDArrayFloat,
                           beta: float,
                           cfg: Optional[TimeStepConfig] = None,
                           nu: float = 0.0) -> NDArrayFloat:
    """Compute local time step for each cell."""
    if cfg is None:
        cfg = TimeStepConfig()
    
    spec_rad: SpectralRadius = compute_spectral_radii(Q, Si_x, Si_y, Sj_x, Sj_y, beta)
    
    lambda_sum: NDArrayFloat = spec_rad.lambda_i + spec_rad.lambda_j
    lambda_sum = np.maximum(lambda_sum, 1e-12)
    dt_conv: NDArrayFloat = cfg.cfl * volume / lambda_sum
    
    dt: NDArrayFloat
    if nu > 1e-12:
        Si_avg: NDArrayFloat = 0.5 * (np.sqrt(Si_x[:-1, :]**2 + Si_y[:-1, :]**2) + 
                        np.sqrt(Si_x[1:, :]**2 + Si_y[1:, :]**2))
        Sj_avg: NDArrayFloat = 0.5 * (np.sqrt(Sj_x[:, :-1]**2 + Sj_y[:, :-1]**2) + 
                        np.sqrt(Sj_x[:, 1:]**2 + Sj_y[:, 1:]**2))
        
        dx: NDArrayFloat = volume / (Sj_avg + 1e-12)
        dy: NDArrayFloat = volume / (Si_avg + 1e-12)
        
        dx_min: NDArrayFloat = np.minimum(dx, dy)
        dt_visc: NDArrayFloat = 0.25 * dx_min**2 / nu * cfg.cfl
        
        dt = np.minimum(dt_conv, dt_visc)
    else:
        dt = dt_conv
    
    dt = np.clip(dt, cfg.min_dt, cfg.max_dt)
    
    if cfg.use_global_dt:
        dt_global: float = float(np.min(dt))
        dt = np.full_like(dt, dt_global)
    
    return dt


def compute_global_timestep(Q: NDArrayFloat,
                            Si_x: NDArrayFloat, Si_y: NDArrayFloat,
                            Sj_x: NDArrayFloat, Sj_y: NDArrayFloat,
                            volume: NDArrayFloat,
                            beta: float,
                            cfl: float = 0.8) -> float:
    """Compute global (minimum) time step."""
    cfg: TimeStepConfig = TimeStepConfig(cfl=cfl, use_global_dt=False)
    dt_local: NDArrayFloat = compute_local_timestep(Q, Si_x, Si_y, Sj_x, Sj_y, volume, beta, cfg)
    return float(np.min(dt_local))


class ExplicitEuler:
    """Explicit Euler time integration with local time stepping."""
    
    beta: float
    cfg: TimeStepConfig
    
    def __init__(self, beta: float, cfg: Optional[TimeStepConfig] = None) -> None:
        self.beta = beta
        self.cfg = cfg if cfg is not None else TimeStepConfig()
    
    def step(self, Q: NDArrayFloat, residual: NDArrayFloat,
             Si_x: NDArrayFloat, Si_y: NDArrayFloat,
             Sj_x: NDArrayFloat, Sj_y: NDArrayFloat,
             volume: NDArrayFloat) -> NDArrayFloat:
        """Perform one explicit Euler step."""
        dt: NDArrayFloat = compute_local_timestep(Q, Si_x, Si_y, Sj_x, Sj_y, 
                                    volume, self.beta, self.cfg)
        
        Q_new: NDArrayFloat = Q.copy()
        Q_new[NGHOST:-NGHOST, NGHOST:-NGHOST, :] += (dt / volume)[:, :, np.newaxis] * residual
        
        return Q_new


class RungeKutta5:
    """
    5-stage Runge-Kutta (Jameson scheme).
    
    Coefficients: α = [1/4, 1/6, 3/8, 1/2, 1]
    """
    
    ALPHA: list[float] = [0.25, 0.166666667, 0.375, 0.5, 1.0]
    
    beta: float
    cfg: TimeStepConfig
    irs_epsilon: float
    
    def __init__(self, beta: float, cfg: Optional[TimeStepConfig] = None,
                 irs_epsilon: float = 0.0) -> None:
        self.beta = beta
        self.cfg = cfg if cfg is not None else TimeStepConfig()
        self.irs_epsilon = irs_epsilon
    
    def step(self, Q: NDArrayFloat, 
             compute_residual: Callable[[NDArrayFloat], NDArrayFloat],
             apply_bc: Callable[[NDArrayFloat], NDArrayFloat],
             Si_x: NDArrayFloat, Si_y: NDArrayFloat,
             Sj_x: NDArrayFloat, Sj_y: NDArrayFloat,
             volume: NDArrayFloat) -> NDArrayFloat:
        """Perform one RK5 step."""
        dt: NDArrayFloat = compute_local_timestep(Q, Si_x, Si_y, Sj_x, Sj_y, 
                                    volume, self.beta, self.cfg)
        
        Q0: NDArrayFloat = Q.copy()
        Qk: NDArrayFloat = Q.copy()
        
        if self.irs_epsilon > 0.0:
            from ..numerics.smoothing import apply_residual_smoothing
        
        for alpha in self.ALPHA:
            Qk = apply_bc(Qk)
            R: NDArrayFloat = compute_residual(Qk)
            
            if self.irs_epsilon > 0.0:
                apply_residual_smoothing(R, self.irs_epsilon)
            
            Qk = Q0.copy()
            Qk[NGHOST:-NGHOST, NGHOST:-NGHOST, :] += alpha * (dt / volume)[:, :, np.newaxis] * R
        
        return Qk


RungeKutta4 = RungeKutta5
