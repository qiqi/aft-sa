"""
Local time stepping for 2D incompressible RANS solver.

Δt_{i,j} = CFL * Ω_{i,j} / (Λ^I_{i,j} + Λ^J_{i,j})

Reference: Jameson, Schmidt, Turkel (1981), AIAA 81-1259.

Both NumPy and JAX implementations provided.
"""

import numpy as np
import numpy.typing as npt
from typing import NamedTuple, Optional, Callable
from dataclasses import dataclass

from src.constants import NGHOST

# JAX imports
try:
    from src.physics.jax_config import jax, jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

NDArrayFloat = npt.NDArray[np.floating]


@dataclass
class TimeStepConfig:
    """Configuration for time stepping."""
    cfl: float = 0.8
    min_dt: float = 0.0
    max_dt: float = 1e10
    use_global_dt: bool = False
    k4: float = 0.016  # JST 4th-order dissipation coefficient
    martinelli_alpha: float = 0.667  # Exponent for aspect ratio scaling
    martinelli_max: float = 3.0  # Maximum Martinelli scaling factor
    c_safety: float = 8.0  # Safety factor for 4th-order stencil (accounts for stencil width and Martinelli)


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
    """Compute local time step for each cell, including dissipative stability limit.
    
    The effective spectral radius accounts for JST 4th-order dissipation with Martinelli scaling:
        λ_eff_i = λ_c_i * (1 + c_safety * k4 * f_i)
        λ_eff_j = λ_c_j * (1 + c_safety * k4 * f_j)
    where f_i, f_j are the Martinelli scaling factors.
    """
    if cfg is None:
        cfg = TimeStepConfig()
    
    # Get convective spectral radii
    spec_rad: SpectralRadius = compute_spectral_radii(Q, Si_x, Si_y, Sj_x, Sj_y, beta)
    lambda_c_i = spec_rad.lambda_i
    lambda_c_j = spec_rad.lambda_j
    
    # Compute Martinelli scaling factors (same logic as in fluxes.py)
    # f_i = 1 + (λ_j/λ_i)^α, f_j = 1 + (λ_i/λ_j)^α
    ratio_ji = lambda_c_j / (lambda_c_i + 1e-12)
    ratio_ij = lambda_c_i / (lambda_c_j + 1e-12)
    
    f_i: NDArrayFloat = 1.0 + np.power(ratio_ji, cfg.martinelli_alpha)
    f_j: NDArrayFloat = 1.0 + np.power(ratio_ij, cfg.martinelli_alpha)
    
    # Cap Martinelli factors to prevent excessive values
    f_i = np.minimum(f_i, cfg.martinelli_max)
    f_j = np.minimum(f_j, cfg.martinelli_max)
    
    # Compute effective spectral radii including dissipative stability
    # λ_eff = λ_c * (1 + c_safety * k4 * f)
    diss_factor_i = 1.0 + cfg.c_safety * cfg.k4 * f_i
    diss_factor_j = 1.0 + cfg.c_safety * cfg.k4 * f_j
    
    lambda_eff_i = lambda_c_i * diss_factor_i
    lambda_eff_j = lambda_c_j * diss_factor_j
    
    lambda_sum: NDArrayFloat = lambda_eff_i + lambda_eff_j
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
    smoothing_type: str
    smoothing_epsilon: float
    smoothing_passes: int
    
    def __init__(self, beta: float, cfg: Optional[TimeStepConfig] = None,
                 irs_epsilon: float = 0.0,
                 smoothing_type: str = "none",
                 smoothing_epsilon: float = 0.2,
                 smoothing_passes: int = 2) -> None:
        self.beta = beta
        self.cfg = cfg if cfg is not None else TimeStepConfig()
        self.irs_epsilon = irs_epsilon
        self.smoothing_type = smoothing_type
        self.smoothing_epsilon = smoothing_epsilon
        self.smoothing_passes = smoothing_passes
    
    def _apply_smoothing(self, R: NDArrayFloat) -> NDArrayFloat:
        """Apply residual smoothing based on configuration."""
        if self.smoothing_type == "explicit":
            from ..numerics.explicit_smoothing import apply_explicit_smoothing
            if self.smoothing_epsilon > 0.0 and self.smoothing_passes > 0:
                return apply_explicit_smoothing(R, self.smoothing_epsilon, self.smoothing_passes)
        elif self.smoothing_type == "implicit" or self.irs_epsilon > 0.0:
            from ..numerics.smoothing import apply_residual_smoothing
            eps = self.irs_epsilon if self.irs_epsilon > 0.0 else 0.5
            apply_residual_smoothing(R, eps)
        return R
    
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
        
        for alpha in self.ALPHA:
            Qk = apply_bc(Qk)
            R: NDArrayFloat = compute_residual(Qk)
            R = self._apply_smoothing(R)
            
            Qk = Q0.copy()
            Qk[NGHOST:-NGHOST, NGHOST:-NGHOST, :] += alpha * (dt / volume)[:, :, np.newaxis] * R
        
        return Qk


RungeKutta4 = RungeKutta5


# =============================================================================
# JAX Implementations
# =============================================================================

if JAX_AVAILABLE:
    
    @jax.jit
    def _compute_spectral_radii_jax_kernel(u, v, Si_x, Si_y, Sj_x, Sj_y, beta):
        """JIT-compiled kernel for spectral radii."""
        c_art = jnp.sqrt(u**2 + v**2 + beta)
        
        # I-direction
        Si_mag = jnp.sqrt(Si_x**2 + Si_y**2)
        Si_L = Si_mag[:-1, :]
        Si_R = Si_mag[1:, :]
        Si_avg = 0.5 * (Si_L + Si_R)
        
        Si_nx_avg = 0.5 * (Si_x[:-1, :] + Si_x[1:, :]) / (Si_avg + 1e-12)
        Si_ny_avg = 0.5 * (Si_y[:-1, :] + Si_y[1:, :]) / (Si_avg + 1e-12)
        
        U_I = jnp.abs(u * Si_nx_avg + v * Si_ny_avg) * Si_avg
        lambda_i = U_I + c_art * Si_avg
        
        # J-direction
        Sj_mag = jnp.sqrt(Sj_x**2 + Sj_y**2)
        Sj_B = Sj_mag[:, :-1]
        Sj_T = Sj_mag[:, 1:]
        Sj_avg = 0.5 * (Sj_B + Sj_T)
        
        Sj_nx_avg = 0.5 * (Sj_x[:, :-1] + Sj_x[:, 1:]) / (Sj_avg + 1e-12)
        Sj_ny_avg = 0.5 * (Sj_y[:, :-1] + Sj_y[:, 1:]) / (Sj_avg + 1e-12)
        
        U_J = jnp.abs(u * Sj_nx_avg + v * Sj_ny_avg) * Sj_avg
        lambda_j = U_J + c_art * Sj_avg
        
        return lambda_i, lambda_j
    
    def compute_spectral_radii_jax(Q, Si_x, Si_y, Sj_x, Sj_y, beta, nghost):
        """
        JAX: Compute spectral radii Λ = |U_n| + c * |S| in I and J directions.
        
        Parameters
        ----------
        Q : jnp.ndarray
            State array (NI+2*nghost, NJ+2*nghost, 4).
        Si_x, Si_y : jnp.ndarray
            I-face normal vectors (NI+1, NJ).
        Sj_x, Sj_y : jnp.ndarray
            J-face normal vectors (NI, NJ+1).
        beta : float
            Artificial compressibility parameter.
        nghost : int
            Number of ghost cells.
            
        Returns
        -------
        lambda_i, lambda_j : jnp.ndarray
            Spectral radii in I and J directions (NI, NJ).
        """
        NI = Si_x.shape[0] - 1
        NJ = Si_x.shape[1]
        
        # Extract interior velocities with concrete indices
        u = Q[nghost:nghost+NI, nghost:nghost+NJ, 1]
        v = Q[nghost:nghost+NI, nghost:nghost+NJ, 2]
        
        return _compute_spectral_radii_jax_kernel(u, v, Si_x, Si_y, Sj_x, Sj_y, beta)
    
    @jax.jit
    def _compute_local_timestep_jax_kernel(lambda_i, lambda_j, volume, 
                                            Si_x, Si_y, Sj_x, Sj_y,
                                            cfl, nu, min_dt, max_dt,
                                            k4, martinelli_alpha, martinelli_max, c_safety):
        """JIT-compiled kernel for local timestep with dissipative stability."""
        # Compute Martinelli scaling factors
        ratio_ji = lambda_j / (lambda_i + 1e-12)
        ratio_ij = lambda_i / (lambda_j + 1e-12)
        
        f_i = 1.0 + jnp.power(ratio_ji, martinelli_alpha)
        f_j = 1.0 + jnp.power(ratio_ij, martinelli_alpha)
        
        f_i = jnp.minimum(f_i, martinelli_max)
        f_j = jnp.minimum(f_j, martinelli_max)
        
        # Effective spectral radii with dissipative stability
        diss_factor_i = 1.0 + c_safety * k4 * f_i
        diss_factor_j = 1.0 + c_safety * k4 * f_j
        
        lambda_eff_i = lambda_i * diss_factor_i
        lambda_eff_j = lambda_j * diss_factor_j
        
        lambda_sum = jnp.maximum(lambda_eff_i + lambda_eff_j, 1e-12)
        dt_conv = cfl * volume / lambda_sum
        
        # Viscous time step limit
        def compute_with_viscous():
            Si_mag_L = jnp.sqrt(Si_x[:-1, :]**2 + Si_y[:-1, :]**2)
            Si_mag_R = jnp.sqrt(Si_x[1:, :]**2 + Si_y[1:, :]**2)
            Si_avg = 0.5 * (Si_mag_L + Si_mag_R)
            
            Sj_mag_B = jnp.sqrt(Sj_x[:, :-1]**2 + Sj_y[:, :-1]**2)
            Sj_mag_T = jnp.sqrt(Sj_x[:, 1:]**2 + Sj_y[:, 1:]**2)
            Sj_avg = 0.5 * (Sj_mag_B + Sj_mag_T)
            
            dx = volume / (Sj_avg + 1e-12)
            dy = volume / (Si_avg + 1e-12)
            
            dx_min = jnp.minimum(dx, dy)
            dt_visc = 0.25 * dx_min**2 / nu * cfl
            
            return jnp.minimum(dt_conv, dt_visc)
        
        dt = jax.lax.cond(
            nu > 1e-12,
            compute_with_viscous,
            lambda: dt_conv
        )
        
        dt = jnp.clip(dt, min_dt, max_dt)
        
        return dt
    
    def compute_local_timestep_jax(Q, Si_x, Si_y, Sj_x, Sj_y, volume, beta, cfl, 
                                    nghost, nu=0.0, min_dt=0.0, max_dt=1e10,
                                    k4=0.016, martinelli_alpha=0.667, 
                                    martinelli_max=3.0, c_safety=8.0):
        """
        JAX: Compute local time step for each cell with dissipative stability.
        
        Parameters
        ----------
        Q : jnp.ndarray
            State array (NI+2*nghost, NJ+2*nghost, 4).
        Si_x, Si_y : jnp.ndarray
            I-face normal vectors (NI+1, NJ).
        Sj_x, Sj_y : jnp.ndarray
            J-face normal vectors (NI, NJ+1).
        volume : jnp.ndarray
            Cell volumes (NI, NJ).
        beta : float
            Artificial compressibility parameter.
        cfl : float
            CFL number.
        nghost : int
            Number of ghost cells.
        nu : float
            Kinematic viscosity (for viscous time step limit).
        min_dt, max_dt : float
            Time step limits.
        k4 : float
            JST 4th-order dissipation coefficient.
        martinelli_alpha, martinelli_max : float
            Martinelli scaling parameters.
        c_safety : float
            Safety factor for 4th-order stencil width.
            
        Returns
        -------
        dt : jnp.ndarray
            Local time step (NI, NJ).
        """
        lambda_i, lambda_j = compute_spectral_radii_jax(
            Q, Si_x, Si_y, Sj_x, Sj_y, beta, nghost
        )
        
        return _compute_local_timestep_jax_kernel(
            lambda_i, lambda_j, volume, Si_x, Si_y, Sj_x, Sj_y,
            cfl, nu, min_dt, max_dt, k4, martinelli_alpha, martinelli_max, c_safety
        )
    
    def compute_local_timestep_jax_wrapper(Q, Si_x, Si_y, Sj_x, Sj_y, volume, 
                                            beta, cfg=None, nu=0.0):
        """Wrapper for JAX local timestep with config object."""
        if cfg is None:
            cfg = TimeStepConfig()
        
        nghost = NGHOST
        
        dt = compute_local_timestep_jax(
            Q, Si_x, Si_y, Sj_x, Sj_y, volume, beta, cfg.cfl,
            nghost, nu, cfg.min_dt, cfg.max_dt,
            cfg.k4, cfg.martinelli_alpha, cfg.martinelli_max, cfg.c_safety
        )
        
        if cfg.use_global_dt:
            dt_global = jnp.min(dt)
            dt = jnp.full_like(dt, dt_global)
        
        return dt
