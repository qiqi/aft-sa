"""
Local Time Stepping for 2D Incompressible RANS Solver.

This module provides local time stepping (LTS) to accelerate convergence
to steady state. Each cell uses its own timestep based on local stability
constraints.

Physics: Artificial Compressibility
    - Acoustic wave speed: c = sqrt(u² + v² + β)
    - This artificial speed determines the CFL constraint

Formula:
    Δt_{i,j} = CFL * Ω_{i,j} / (Λ^I_{i,j} + Λ^J_{i,j})
    
    where:
    - Ω_{i,j} is the cell volume
    - Λ^I, Λ^J are spectral radii in I and J directions
    - Spectral radius = |U_n| + c * |S| (convective + acoustic)

Reference:
    Jameson, A. (1991). "Time Dependent Calculations Using Multigrid, 
    with Applications to Unsteady Flows Past Airfoils and Wings."
    AIAA Paper 91-1596.
"""

import numpy as np
from typing import NamedTuple, Optional
from dataclasses import dataclass


@dataclass
class TimeStepConfig:
    """Configuration for time stepping."""
    
    cfl: float = 0.8           # CFL number (typically 0.5-1.0 for explicit)
    min_dt: float = 0.0        # Minimum allowed timestep (0 = no limit)
    max_dt: float = 1e10       # Maximum allowed timestep
    use_global_dt: bool = False  # If True, use minimum dt across all cells


class SpectralRadius(NamedTuple):
    """Spectral radii in each direction."""
    lambda_i: np.ndarray  # Shape: (NI, NJ) - spectral radius in I-direction
    lambda_j: np.ndarray  # Shape: (NI, NJ) - spectral radius in J-direction


def compute_spectral_radii(Q: np.ndarray, 
                           Si_x: np.ndarray, Si_y: np.ndarray,
                           Sj_x: np.ndarray, Sj_y: np.ndarray,
                           beta: float) -> SpectralRadius:
    """
    Compute spectral radii in I and J directions.
    
    The spectral radius in each direction is:
        Λ = |U_n| + c * |S|
        
    where:
        U_n = u*nx + v*ny (contravariant velocity, scaled by face area)
        c = sqrt(u² + v² + β) (artificial sound speed)
        |S| = sqrt(Sx² + Sy²) (face area)
    
    Parameters
    ----------
    Q : ndarray, shape (NI+2, NJ+2, 4)
        State vector with ghost cells: [p, u, v, ν̃].
    Si_x, Si_y : ndarray, shape (NI+1, NJ)
        I-face normals scaled by area.
    Sj_x, Sj_y : ndarray, shape (NI, NJ+1)
        J-face normals scaled by area.
    beta : float
        Artificial compressibility parameter.
        
    Returns
    -------
    SpectralRadius
        Spectral radii in I and J directions, shape (NI, NJ).
    """
    NI = Q.shape[0] - 2  # 1 I-ghost on each side
    NJ = Q.shape[1] - 3  # 2 J-ghosts at wall, 1 at farfield
    
    # Interior cell velocities (cell-centered, 2 J-ghosts at wall)
    u = Q[1:-1, 2:-1, 1]  # Shape: (NI, NJ)
    v = Q[1:-1, 2:-1, 2]  # Shape: (NI, NJ)
    
    # Artificial sound speed at each cell
    c_art = np.sqrt(u**2 + v**2 + beta)  # Shape: (NI, NJ)
    
    # --- I-direction spectral radius ---
    # Average face areas at left and right faces of each cell
    Si_mag = np.sqrt(Si_x**2 + Si_y**2)  # Shape: (NI+1, NJ)
    Si_L = Si_mag[:-1, :]  # Left face of cell (i, j), shape: (NI, NJ)
    Si_R = Si_mag[1:, :]   # Right face of cell (i, j), shape: (NI, NJ)
    Si_avg = 0.5 * (Si_L + Si_R)
    
    # Unit normal in I-direction (average of left and right faces)
    Si_nx_avg = 0.5 * (Si_x[:-1, :] + Si_x[1:, :]) / (Si_avg + 1e-12)
    Si_ny_avg = 0.5 * (Si_y[:-1, :] + Si_y[1:, :]) / (Si_avg + 1e-12)
    
    # Contravariant velocity in I-direction (scaled by area)
    U_I = np.abs(u * Si_nx_avg + v * Si_ny_avg) * Si_avg
    
    # Spectral radius: |U_n| + c * |S|
    lambda_i = U_I + c_art * Si_avg  # Shape: (NI, NJ)
    
    # --- J-direction spectral radius ---
    Sj_mag = np.sqrt(Sj_x**2 + Sj_y**2)  # Shape: (NI, NJ+1)
    Sj_B = Sj_mag[:, :-1]  # Bottom face of cell (i, j)
    Sj_T = Sj_mag[:, 1:]   # Top face of cell (i, j)
    Sj_avg = 0.5 * (Sj_B + Sj_T)
    
    # Unit normal in J-direction
    Sj_nx_avg = 0.5 * (Sj_x[:, :-1] + Sj_x[:, 1:]) / (Sj_avg + 1e-12)
    Sj_ny_avg = 0.5 * (Sj_y[:, :-1] + Sj_y[:, 1:]) / (Sj_avg + 1e-12)
    
    # Contravariant velocity in J-direction
    U_J = np.abs(u * Sj_nx_avg + v * Sj_ny_avg) * Sj_avg
    
    # Spectral radius
    lambda_j = U_J + c_art * Sj_avg  # Shape: (NI, NJ)
    
    return SpectralRadius(lambda_i=lambda_i, lambda_j=lambda_j)


def compute_local_timestep(Q: np.ndarray,
                           Si_x: np.ndarray, Si_y: np.ndarray,
                           Sj_x: np.ndarray, Sj_y: np.ndarray,
                           volume: np.ndarray,
                           beta: float,
                           cfg: Optional[TimeStepConfig] = None,
                           nu: float = 0.0) -> np.ndarray:
    """
    Compute local time step for each cell based on CFL and viscous stability.
    
    Formula (convective):
        Δt_conv = CFL * Ω / (Λ^I + Λ^J)
        
    Formula (viscous):
        Δt_visc = 0.5 * Ω^2 / (ν * (S_i² + S_j²))
        
    Final time step is the minimum of both constraints.
    
    Parameters
    ----------
    Q : ndarray, shape (NI+2, NJ+2, 4)
        State vector with ghost cells.
    Si_x, Si_y : ndarray, shape (NI+1, NJ)
        I-face normals (scaled by area).
    Sj_x, Sj_y : ndarray, shape (NI, NJ+1)
        J-face normals (scaled by area).
    volume : ndarray, shape (NI, NJ)
        Cell volumes.
    beta : float
        Artificial compressibility parameter.
    cfg : TimeStepConfig, optional
        Time stepping configuration.
    nu : float, optional
        Kinematic viscosity (1/Re). If 0, viscous constraint is ignored.
        
    Returns
    -------
    dt : ndarray, shape (NI, NJ)
        Local time step for each interior cell.
    """
    if cfg is None:
        cfg = TimeStepConfig()
    
    # Compute spectral radii
    spec_rad = compute_spectral_radii(Q, Si_x, Si_y, Sj_x, Sj_y, beta)
    
    # Convective timestep: dt = CFL * Volume / (λ_I + λ_J)
    lambda_sum = spec_rad.lambda_i + spec_rad.lambda_j
    lambda_sum = np.maximum(lambda_sum, 1e-12)
    dt_conv = cfg.cfl * volume / lambda_sum
    
    # Viscous stability constraint (if viscosity is significant)
    if nu > 1e-12:
        # Estimate cell size from volume and face areas
        # dx ~ volume / Sj (j-face area), dy ~ volume / Si (i-face area)
        Si_avg = 0.5 * (np.sqrt(Si_x[:-1, :]**2 + Si_y[:-1, :]**2) + 
                        np.sqrt(Si_x[1:, :]**2 + Si_y[1:, :]**2))
        Sj_avg = 0.5 * (np.sqrt(Sj_x[:, :-1]**2 + Sj_y[:, :-1]**2) + 
                        np.sqrt(Sj_x[:, 1:]**2 + Sj_y[:, 1:]**2))
        
        # Cell sizes: dx ~ Vol/Sy, dy ~ Vol/Sx
        dx = volume / (Sj_avg + 1e-12)
        dy = volume / (Si_avg + 1e-12)
        
        # Viscous stability: dt <= C * min(dx,dy)^2 / nu
        # Use C = 0.25 for safety with explicit schemes
        dx_min = np.minimum(dx, dy)
        dt_visc = 0.25 * dx_min**2 / nu * cfg.cfl
        
        # Take minimum of convective and viscous constraints
        dt = np.minimum(dt_conv, dt_visc)
    else:
        dt = dt_conv
    
    # Apply min/max limits
    dt = np.clip(dt, cfg.min_dt, cfg.max_dt)
    
    # Optionally use global (minimum) timestep
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
    """
    Compute global (minimum) time step for time-accurate integration.
    
    Parameters
    ----------
    Q : ndarray, shape (NI+2, NJ+2, 4)
        State vector with ghost cells.
    Si_x, Si_y : ndarray
        I-face normals.
    Sj_x, Sj_y : ndarray
        J-face normals.
    volume : ndarray
        Cell volumes.
    beta : float
        Artificial compressibility parameter.
    cfl : float
        CFL number.
        
    Returns
    -------
    dt : float
        Global time step (minimum across all cells).
    """
    cfg = TimeStepConfig(cfl=cfl, use_global_dt=False)
    dt_local = compute_local_timestep(Q, Si_x, Si_y, Sj_x, Sj_y, volume, beta, cfg)
    return float(np.min(dt_local))


class ExplicitEuler:
    """
    Explicit Euler time integration with local time stepping.
    
    Q^{n+1} = Q^n + Δt * R / Ω
    
    where R is the residual from flux computation.
    """
    
    def __init__(self, beta: float, cfg: Optional[TimeStepConfig] = None):
        """
        Initialize Euler integrator.
        
        Parameters
        ----------
        beta : float
            Artificial compressibility parameter.
        cfg : TimeStepConfig, optional
            Time stepping configuration.
        """
        self.beta = beta
        self.cfg = cfg if cfg is not None else TimeStepConfig()
    
    def step(self, Q: np.ndarray, residual: np.ndarray,
             Si_x: np.ndarray, Si_y: np.ndarray,
             Sj_x: np.ndarray, Sj_y: np.ndarray,
             volume: np.ndarray) -> np.ndarray:
        """
        Perform one explicit Euler step.
        
        Parameters
        ----------
        Q : ndarray, shape (NI+2, NJ+2, 4)
            Current state with ghost cells.
        residual : ndarray, shape (NI, NJ, 4)
            Flux residual for interior cells.
        Si_x, Si_y, Sj_x, Sj_y : ndarray
            Face normals.
        volume : ndarray, shape (NI, NJ)
            Cell volumes.
            
        Returns
        -------
        Q_new : ndarray, shape (NI+2, NJ+2, 4)
            Updated state (ghost cells unchanged).
        """
        # Compute local time step
        dt = compute_local_timestep(Q, Si_x, Si_y, Sj_x, Sj_y, 
                                    volume, self.beta, self.cfg)
        
        # Update interior cells: Q^{n+1} = Q^n + dt/Ω * R
        # Note: Residual is defined as net flux INTO cell (positive = accumulation)
        Q_new = Q.copy()
        Q_new[1:-1, 2:-1, :] += (dt / volume)[:, :, np.newaxis] * residual
        
        return Q_new


class RungeKutta5:
    """
    5-stage Runge-Kutta time integration for multigrid.
    
    This is Jameson's 5-stage scheme with extended stability region,
    optimized for central difference schemes and multigrid:
        Q^(0) = Q^n
        Q^(k) = Q^(0) + α_k * Δt * R(Q^(k-1))
        Q^(n+1) = Q^(5)
    
    Coefficients: α = [1/4, 1/6, 3/8, 1/2, 1]
    
    This extends the stability region along the negative real axis,
    allowing CFL ≈ 4 for central differences (vs ~2.8 for 4-stage).
    
    Reference: Jameson, Schmidt, Turkel (1981), AIAA Paper 81-1259
    
    Optionally supports Implicit Residual Smoothing (IRS) to allow
    even higher CFL numbers by filtering high-frequency errors.
    """
    
    # Jameson 5-stage coefficients for multigrid (extended stability region)
    # These maximize CFL for central differences: CFL_max ≈ 4.0
    ALPHA = [0.25, 0.166666667, 0.375, 0.5, 1.0]
    
    def __init__(self, beta: float, cfg: Optional[TimeStepConfig] = None,
                 irs_epsilon: float = 0.0):
        """
        Initialize RK5 integrator (Jameson multigrid scheme).
        
        Parameters
        ----------
        beta : float
            Artificial compressibility parameter.
        cfg : TimeStepConfig, optional
            Time stepping configuration.
        irs_epsilon : float, optional
            Implicit Residual Smoothing coefficient.
            0 = disabled (default), 0.5-2.0 = typical range.
        """
        self.beta = beta
        self.cfg = cfg if cfg is not None else TimeStepConfig()
        self.irs_epsilon = irs_epsilon
    
    def step(self, Q: np.ndarray, 
             compute_residual,
             apply_bc,
             Si_x: np.ndarray, Si_y: np.ndarray,
             Sj_x: np.ndarray, Sj_y: np.ndarray,
             volume: np.ndarray) -> np.ndarray:
        """
        Perform one RK5 step (Jameson's 5-stage multigrid scheme).
        
        Parameters
        ----------
        Q : ndarray, shape (NI+2, NJ+2, 4)
            Current state with ghost cells.
        compute_residual : callable
            Function: R = compute_residual(Q) returning (NI, NJ, 4) residual.
        apply_bc : callable
            Function: Q = apply_bc(Q) to update ghost cells.
        Si_x, Si_y, Sj_x, Sj_y : ndarray
            Face normals.
        volume : ndarray, shape (NI, NJ)
            Cell volumes.
            
        Returns
        -------
        Q_new : ndarray, shape (NI+2, NJ+2, 4)
            Updated state after full RK5 step.
        """
        # Compute local time step based on initial state
        dt = compute_local_timestep(Q, Si_x, Si_y, Sj_x, Sj_y, 
                                    volume, self.beta, self.cfg)
        
        Q0 = Q.copy()  # Store initial state
        Qk = Q.copy()
        
        # Import IRS if needed (lazy import to avoid circular dependency)
        if self.irs_epsilon > 0.0:
            from ..numerics.smoothing import apply_residual_smoothing
        
        for alpha in self.ALPHA:
            # Apply boundary conditions
            Qk = apply_bc(Qk)
            
            # Compute residual
            R = compute_residual(Qk)
            
            # Apply Implicit Residual Smoothing if enabled
            if self.irs_epsilon > 0.0:
                apply_residual_smoothing(R, self.irs_epsilon)
            
            # Update: Q^(k) = Q^(0) + α_k * dt/Ω * R
            Qk = Q0.copy()
            Qk[1:-1, 2:-1, :] += alpha * (dt / volume)[:, :, np.newaxis] * R
        
        return Qk


# Backward compatibility alias
RungeKutta4 = RungeKutta5

