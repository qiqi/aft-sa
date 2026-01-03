"""
Batch solver data structures and utilities.

Enables running multiple simulations (e.g., AoA sweep) in parallel on GPU.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple, Dict, Any
import numpy as np

from src.physics.jax_config import jax, jnp


@dataclass
class BatchFlowConditions:
    """
    Flow conditions for a batch of simulations.
    
    All arrays have shape (n_batch,) for per-case values.
    """
    n_batch: int
    alpha_deg: np.ndarray      # Angle of attack in degrees
    reynolds: np.ndarray       # Reynolds number
    mach: np.ndarray           # Mach number (0 = incompressible)
    
    # Derived quantities (computed from above)
    p_inf: np.ndarray = field(init=False)
    u_inf: np.ndarray = field(init=False)
    v_inf: np.ndarray = field(init=False)
    nu_t_inf: np.ndarray = field(init=False)
    nu_laminar: np.ndarray = field(init=False)
    
    def __post_init__(self):
        """Compute derived freestream quantities."""
        alpha_rad = np.radians(self.alpha_deg)
        
        self.p_inf = np.zeros(self.n_batch)
        self.u_inf = np.cos(alpha_rad)
        self.v_inf = np.sin(alpha_rad)
        
        # Laminar viscosity from Reynolds
        self.nu_laminar = np.where(self.reynolds > 0, 1.0 / self.reynolds, 0.0)
        
        # Turbulent viscosity at freestream (0.1% of laminar)
        self.nu_t_inf = 0.001 * self.nu_laminar
    
    @classmethod
    def from_single(cls, alpha_deg: float = 0.0, reynolds: float = 6e6,
                    mach: float = 0.0) -> 'BatchFlowConditions':
        """Create batch of 1 from single values."""
        return cls(
            n_batch=1,
            alpha_deg=np.array([alpha_deg]),
            reynolds=np.array([reynolds]),
            mach=np.array([mach]),
        )
    
    @classmethod
    def from_sweep(cls, alpha_spec: Union[float, Dict[str, Any]],
                   reynolds: float = 6e6, mach: float = 0.0) -> 'BatchFlowConditions':
        """
        Create batch from sweep specification.
        
        Args:
            alpha_spec: Either a single value, or a dict with:
                - {"sweep": [start, end, count]} for linear sweep
                - {"values": [v1, v2, ...]} for explicit list
            reynolds: Reynolds number (same for all cases)
            mach: Mach number (same for all cases)
        """
        alphas = expand_parameter(alpha_spec)
        n_batch = len(alphas)
        
        return cls(
            n_batch=n_batch,
            alpha_deg=np.array(alphas),
            reynolds=np.full(n_batch, reynolds),
            mach=np.full(n_batch, mach),
        )
    
    def to_jax(self) -> 'BatchFlowConditionsJax':
        """Convert to JAX arrays for GPU computation."""
        return BatchFlowConditionsJax(
            n_batch=self.n_batch,
            p_inf=jnp.array(self.p_inf),
            u_inf=jnp.array(self.u_inf),
            v_inf=jnp.array(self.v_inf),
            nu_t_inf=jnp.array(self.nu_t_inf),
            nu_laminar=jnp.array(self.nu_laminar),
            alpha_deg=jnp.array(self.alpha_deg),
            reynolds=jnp.array(self.reynolds),
        )
    
    def get_case(self, i: int) -> Dict[str, float]:
        """Get parameters for a single case."""
        return {
            'alpha_deg': float(self.alpha_deg[i]),
            'reynolds': float(self.reynolds[i]),
            'mach': float(self.mach[i]),
            'u_inf': float(self.u_inf[i]),
            'v_inf': float(self.v_inf[i]),
        }


@dataclass
class BatchFlowConditionsJax:
    """JAX version of BatchFlowConditions for GPU computation."""
    n_batch: int
    p_inf: jnp.ndarray
    u_inf: jnp.ndarray
    v_inf: jnp.ndarray
    nu_t_inf: jnp.ndarray
    nu_laminar: jnp.ndarray
    alpha_deg: jnp.ndarray
    reynolds: jnp.ndarray


def expand_parameter(spec: Union[float, int, Dict[str, Any]]) -> List[float]:
    """
    Expand a parameter specification to a list of values.
    
    Args:
        spec: Either:
            - A single numeric value
            - {"sweep": [start, end, count]} for linear sweep
            - {"values": [v1, v2, ...]} for explicit list
    
    Returns:
        List of parameter values
    
    Examples:
        >>> expand_parameter(4.0)
        [4.0]
        >>> expand_parameter({"sweep": [0, 10, 5]})
        [0.0, 2.5, 5.0, 7.5, 10.0]
        >>> expand_parameter({"values": [0, 2, 4, 8]})
        [0, 2, 4, 8]
    """
    if isinstance(spec, (int, float)):
        return [float(spec)]
    
    if isinstance(spec, dict):
        if 'sweep' in spec:
            start, end, count = spec['sweep']
            return list(np.linspace(start, end, count))
        elif 'values' in spec:
            return list(spec['values'])
        else:
            raise ValueError(f"Unknown parameter spec: {spec}")
    
    raise ValueError(f"Invalid parameter spec type: {type(spec)}")


@dataclass
class BatchState:
    """
    State for batch of simulations.
    
    All Q arrays have shape (n_batch, NI+2*nghost, NJ+2*nghost, 4).
    """
    n_batch: int
    NI: int
    NJ: int
    nghost: int
    
    # Solution state - NumPy (CPU) version
    Q: np.ndarray  # (n_batch, NI+2g, NJ+2g, 4)
    
    # JAX (GPU) version
    Q_jax: Optional[jnp.ndarray] = None
    
    # Per-case tracking
    residual_history: List[List[float]] = field(default_factory=list)
    converged: np.ndarray = field(init=False)  # (n_batch,) bool
    
    def __post_init__(self):
        self.converged = np.zeros(self.n_batch, dtype=bool)
        self.residual_history = [[] for _ in range(self.n_batch)]
    
    @classmethod
    def from_single_ic(cls, Q_single: np.ndarray, n_batch: int,
                       nghost: int = 2) -> 'BatchState':
        """
        Create batch state by replicating a single initial condition.
        
        Args:
            Q_single: Single IC with shape (NI+2g, NJ+2g, 4)
            n_batch: Number of cases in batch
            nghost: Number of ghost cells
        
        Returns:
            BatchState with Q replicated n_batch times
        """
        NI = Q_single.shape[0] - 2 * nghost
        NJ = Q_single.shape[1] - 2 * nghost
        
        # Replicate IC for all cases
        Q_batch = np.tile(Q_single[np.newaxis, ...], (n_batch, 1, 1, 1))
        
        return cls(
            n_batch=n_batch,
            NI=NI,
            NJ=NJ,
            nghost=nghost,
            Q=Q_batch,
        )
    
    def to_gpu(self) -> None:
        """Transfer Q to GPU."""
        self.Q_jax = jnp.array(self.Q)
    
    def to_cpu(self) -> None:
        """Transfer Q from GPU to CPU."""
        if self.Q_jax is not None:
            self.Q = np.array(self.Q_jax)
    
    def get_case_Q(self, i: int) -> np.ndarray:
        """Get Q for a single case (on CPU)."""
        if self.Q_jax is not None:
            return np.array(self.Q_jax[i])
        return self.Q[i]


def compute_batch_size(alpha_spec: Union[float, Dict],
                       reynolds_spec: Union[float, Dict] = None) -> int:
    """
    Compute total batch size from parameter specifications.
    
    For now, only alpha can vary (reynolds sweep comes in Phase 5).
    """
    alphas = expand_parameter(alpha_spec)
    return len(alphas)


@dataclass
class BatchForces:
    """Aerodynamic forces for a batch of cases."""
    CL: np.ndarray      # (n_batch,)
    CD: np.ndarray      # (n_batch,)
    CD_p: np.ndarray    # (n_batch,)
    CD_f: np.ndarray    # (n_batch,)
    alpha_deg: np.ndarray  # (n_batch,)
    
    def to_dataframe(self):
        """Convert to pandas DataFrame for easy analysis."""
        try:
            import pandas as pd
            return pd.DataFrame({
                'alpha_deg': self.alpha_deg,
                'CL': self.CL,
                'CD': self.CD,
                'CD_p': self.CD_p,
                'CD_f': self.CD_f,
                'L/D': self.CL / (self.CD + 1e-12),
            })
        except ImportError:
            # Return dict-like representation if pandas not available
            return {
                'alpha_deg': self.alpha_deg,
                'CL': self.CL,
                'CD': self.CD,
                'CD_p': self.CD_p,
                'CD_f': self.CD_f,
                'L/D': self.CL / (self.CD + 1e-12),
            }
    
    def save_csv(self, path: str):
        """Save to CSV file."""
        ld = self.CL / (self.CD + 1e-12)
        with open(path, 'w') as f:
            f.write('alpha_deg,CL,CD,CD_p,CD_f,L/D\n')
            for i in range(len(self.alpha_deg)):
                f.write(f'{self.alpha_deg[i]:.4f},{self.CL[i]:.6f},{self.CD[i]:.6f},'
                        f'{self.CD_p[i]:.6f},{self.CD_f[i]:.6f},{ld[i]:.4f}\n')
        return self.to_dataframe()


# =============================================================================
# Phase 3: Batch Kernels using jax.vmap
# =============================================================================

from src.numerics.fluxes import _compute_fluxes_jax_impl
from src.numerics.gradients import _compute_gradients_jax_impl
from src.numerics.viscous_fluxes import compute_viscous_fluxes_jax
from src.numerics.explicit_smoothing import smooth_explicit_jax
from src.solvers.time_stepping import _compute_spectral_radii_jax_kernel, _compute_local_timestep_jax_kernel
from src.constants import NGHOST


def _extract_flux_slices(Q, Si_x, Si_y, Sj_x, Sj_y, nghost):
    """Extract Q slices needed for flux computation.
    
    Works for both single (NI+2g, NJ+2g, 4) and batch (B, NI+2g, NJ+2g, 4) arrays.
    """
    NI_p1, NJ = Si_x.shape[-2], Si_x.shape[-1]
    NI = NI_p1 - 1
    
    if Q.ndim == 3:
        # Single case
        Q_L_i = Q[nghost-1:nghost+NI, nghost:nghost+NJ, :]
        Q_R_i = Q[nghost:nghost+NI+1, nghost:nghost+NJ, :]
        Q_Lm1_i = Q[nghost-2:nghost+NI-1, nghost:nghost+NJ, :]
        Q_Rp1_i = Q[nghost+1:nghost+NI+2, nghost:nghost+NJ, :]
        Q_L_j = Q[nghost:nghost+NI, nghost-1:nghost+NJ, :]
        Q_R_j = Q[nghost:nghost+NI, nghost:nghost+NJ+1, :]
        Q_Lm1_j = Q[nghost:nghost+NI, nghost-2:nghost+NJ-1, :]
        Q_Rp1_j = Q[nghost:nghost+NI, nghost+1:nghost+NJ+2, :]
        Q_int = Q[nghost:nghost+NI, nghost:nghost+NJ, :]
    else:
        # Batch case (B, NI+2g, NJ+2g, 4)
        Q_L_i = Q[:, nghost-1:nghost+NI, nghost:nghost+NJ, :]
        Q_R_i = Q[:, nghost:nghost+NI+1, nghost:nghost+NJ, :]
        Q_Lm1_i = Q[:, nghost-2:nghost+NI-1, nghost:nghost+NJ, :]
        Q_Rp1_i = Q[:, nghost+1:nghost+NI+2, nghost:nghost+NJ, :]
        Q_L_j = Q[:, nghost:nghost+NI, nghost-1:nghost+NJ, :]
        Q_R_j = Q[:, nghost:nghost+NI, nghost:nghost+NJ+1, :]
        Q_Lm1_j = Q[:, nghost:nghost+NI, nghost-2:nghost+NJ-1, :]
        Q_Rp1_j = Q[:, nghost:nghost+NI, nghost+1:nghost+NJ+2, :]
        Q_int = Q[:, nghost:nghost+NI, nghost:nghost+NJ, :]
    
    return Q_L_i, Q_R_i, Q_Lm1_i, Q_Rp1_i, Q_L_j, Q_R_j, Q_Lm1_j, Q_Rp1_j, Q_int


def compute_fluxes_single(Q, Si_x, Si_y, Sj_x, Sj_y, beta, k4, nghost, martinelli_alpha=0.667):
    """Single-case flux computation (for vmap)."""
    slices = _extract_flux_slices(Q, Si_x, Si_y, Sj_x, Sj_y, nghost)
    Q_L_i, Q_R_i, Q_Lm1_i, Q_Rp1_i, Q_L_j, Q_R_j, Q_Lm1_j, Q_Rp1_j, Q_int = slices
    
    return _compute_fluxes_jax_impl(
        Q_L_i, Q_R_i, Q_Lm1_i, Q_Rp1_i, Si_x, Si_y,
        Q_L_j, Q_R_j, Q_Lm1_j, Q_Rp1_j, Sj_x, Sj_y,
        Q_int, beta, k4, martinelli_alpha
    )


def compute_gradients_single(Q, Si_x, Si_y, Sj_x, Sj_y, volume, nghost):
    """Single-case gradient computation (for vmap)."""
    NI, NJ = volume.shape
    
    Q_L_i = Q[nghost-1:nghost+NI, nghost:nghost+NJ, :]
    Q_R_i = Q[nghost:nghost+NI+1, nghost:nghost+NJ, :]
    Q_L_j = Q[nghost:nghost+NI, nghost-1:nghost+NJ, :]
    Q_R_j = Q[nghost:nghost+NI, nghost:nghost+NJ+1, :]
    
    return _compute_gradients_jax_impl(Q_L_i, Q_R_i, Q_L_j, Q_R_j,
                                        Si_x, Si_y, Sj_x, Sj_y, volume)


def compute_timestep_single(Q, Si_x, Si_y, Sj_x, Sj_y, volume, beta, cfl, nghost, nu,
                            min_dt=0.0, max_dt=1e10, k4=0.016, 
                            martinelli_alpha=0.667, martinelli_max=3.0, c_safety=8.0):
    """Single-case timestep computation (for vmap)."""
    NI = Si_x.shape[0] - 1
    NJ = Si_x.shape[1]
    
    u = Q[nghost:nghost+NI, nghost:nghost+NJ, 1]
    v = Q[nghost:nghost+NI, nghost:nghost+NJ, 2]
    
    lambda_i, lambda_j = _compute_spectral_radii_jax_kernel(
        u, v, Si_x, Si_y, Sj_x, Sj_y, beta
    )
    
    return _compute_local_timestep_jax_kernel(
        lambda_i, lambda_j, volume, Si_x, Si_y, Sj_x, Sj_y,
        cfl, nu, min_dt, max_dt, k4, martinelli_alpha, martinelli_max, c_safety
    )


# Create vmapped versions
# Q is batched (axis 0), metrics are shared (None)
_compute_fluxes_batch_vmap = jax.vmap(
    compute_fluxes_single,
    in_axes=(0, None, None, None, None, None, None, None, None)
)

_compute_gradients_batch_vmap = jax.vmap(
    compute_gradients_single,
    in_axes=(0, None, None, None, None, None, None)
)

_compute_timestep_batch_vmap = jax.vmap(
    compute_timestep_single,
    in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
)

# Viscous fluxes and smoothing work on interior arrays, also need vmap
_compute_viscous_fluxes_batch_vmap = jax.vmap(
    compute_viscous_fluxes_jax,
    in_axes=(0, None, None, None, None, 0)  # grad batched, metrics shared, mu_eff batched
)

_smooth_explicit_batch_vmap = jax.vmap(
    lambda R, eps, n: smooth_explicit_jax(R, eps, n),
    in_axes=(0, None, None)
)


def compute_fluxes_batch(Q_batch, Si_x, Si_y, Sj_x, Sj_y, beta, k4, nghost, martinelli_alpha=0.667):
    """
    Compute flux residuals for a batch of cases.
    
    Parameters
    ----------
    Q_batch : jnp.ndarray
        Batched state array (n_batch, NI+2*nghost, NJ+2*nghost, 4).
    Si_x, Si_y : jnp.ndarray
        I-face normal vectors (NI+1, NJ) - shared across batch.
    Sj_x, Sj_y : jnp.ndarray
        J-face normal vectors (NI, NJ+1) - shared across batch.
    beta : float
        Artificial compressibility parameter.
    k4 : float
        4th-order dissipation coefficient.
    nghost : int
        Number of ghost cells.
    martinelli_alpha : float
        Exponent for aspect ratio scaling.
        
    Returns
    -------
    R_batch : jnp.ndarray
        Flux residuals (n_batch, NI, NJ, 4).
    """
    return _compute_fluxes_batch_vmap(
        Q_batch, Si_x, Si_y, Sj_x, Sj_y, beta, k4, nghost, martinelli_alpha
    )


def compute_gradients_batch(Q_batch, Si_x, Si_y, Sj_x, Sj_y, volume, nghost):
    """
    Compute gradients for a batch of cases.
    
    Parameters
    ----------
    Q_batch : jnp.ndarray
        Batched state array (n_batch, NI+2*nghost, NJ+2*nghost, 4).
    Si_x, Si_y, Sj_x, Sj_y, volume : jnp.ndarray
        Grid metrics - shared across batch.
    nghost : int
        Number of ghost cells.
        
    Returns
    -------
    grad_batch : jnp.ndarray
        Gradients (n_batch, NI, NJ, 4, 2).
    """
    return _compute_gradients_batch_vmap(
        Q_batch, Si_x, Si_y, Sj_x, Sj_y, volume, nghost
    )


def compute_timestep_batch(Q_batch, Si_x, Si_y, Sj_x, Sj_y, volume, beta, cfl, nghost, nu,
                           min_dt=0.0, max_dt=1e10, k4=0.016,
                           martinelli_alpha=0.667, martinelli_max=3.0, c_safety=8.0):
    """
    Compute local timesteps for a batch of cases.
    
    Parameters
    ----------
    Q_batch : jnp.ndarray
        Batched state array (n_batch, NI+2*nghost, NJ+2*nghost, 4).
    Si_x, Si_y, Sj_x, Sj_y, volume : jnp.ndarray
        Grid metrics - shared across batch.
    beta, cfl, nghost, nu, ... : 
        Solver parameters - same for all cases.
        
    Returns
    -------
    dt_batch : jnp.ndarray
        Local timesteps (n_batch, NI, NJ).
    """
    return _compute_timestep_batch_vmap(
        Q_batch, Si_x, Si_y, Sj_x, Sj_y, volume, beta, cfl, nghost, nu,
        min_dt, max_dt, k4, martinelli_alpha, martinelli_max, c_safety
    )


def compute_viscous_fluxes_batch(grad_batch, Si_x, Si_y, Sj_x, Sj_y, mu_eff_batch):
    """
    Compute viscous flux residuals for a batch of cases.
    
    Parameters
    ----------
    grad_batch : jnp.ndarray
        Batched gradients (n_batch, NI, NJ, 4, 2).
    Si_x, Si_y, Sj_x, Sj_y : jnp.ndarray
        Grid metrics - shared across batch.
    mu_eff_batch : jnp.ndarray
        Effective viscosity (n_batch, NI, NJ).
        
    Returns
    -------
    R_visc_batch : jnp.ndarray
        Viscous flux residuals (n_batch, NI, NJ, 4).
    """
    return _compute_viscous_fluxes_batch_vmap(
        grad_batch, Si_x, Si_y, Sj_x, Sj_y, mu_eff_batch
    )


def smooth_residual_batch(R_batch, epsilon, n_passes):
    """
    Apply explicit smoothing to a batch of residuals.
    
    Parameters
    ----------
    R_batch : jnp.ndarray
        Batched residuals (n_batch, NI, NJ, 4).
    epsilon : float
        Smoothing coefficient.
    n_passes : int
        Number of smoothing passes.
        
    Returns
    -------
    R_smooth_batch : jnp.ndarray
        Smoothed residuals (n_batch, NI, NJ, 4).
    """
    if epsilon <= 0.0 or n_passes <= 0:
        return R_batch
    return _smooth_explicit_batch_vmap(R_batch, epsilon, n_passes)


# =============================================================================
# Batch Boundary Conditions
# =============================================================================

def make_apply_bc_batch_jit(NI: int, NJ: int, n_wake_points: int,
                            nx: jnp.ndarray, ny: jnp.ndarray,
                            nghost: int = NGHOST):
    """
    Create a JIT-compiled batch BC function with per-case freestream.
    
    Unlike the single-case version, this accepts per-case freestream values
    as input arguments rather than capturing them in closure.
    
    Parameters
    ----------
    NI, NJ : int
        Number of interior cells.
    n_wake_points : int
        Number of wake points on each side.
    nx, ny : jnp.ndarray
        Farfield outward unit normals (NI,).
    nghost : int
        Number of ghost cells.
        
    Returns
    -------
    apply_bc_batch : callable
        JIT-compiled function: Q_batch_new = apply_bc_batch(Q_batch, u_inf, v_inf, p_inf, nu_t_inf)
        where u_inf, v_inf, p_inf, nu_t_inf are (n_batch,) arrays.
    """
    # Pre-compute all indices (constants in the JIT closure)
    n_wake = n_wake_points if n_wake_points > 0 else NI // 6
    i_wake_end_lower = n_wake
    i_wake_start_upper = NI - n_wake
    
    i_start = i_wake_end_lower + nghost  # airfoil start
    i_end = i_wake_start_upper + nghost  # airfoil end
    j_int_first = nghost
    j_int_last = NJ + nghost - 1
    i_upper_end = NI + nghost
    
    # Pre-compute I-outlet mask
    is_i_outlet = jnp.zeros(NI, dtype=bool)
    if n_wake_points > 0:
        is_i_outlet = is_i_outlet.at[:n_wake_points].set(True)
        is_i_outlet = is_i_outlet.at[-n_wake_points:].set(True)
    
    def apply_bc_single(Q, u_inf, v_inf, p_inf, nu_t_inf):
        """Apply BCs to a single case with given freestream."""
        
        # === Surface BC: Airfoil wall (no-slip) ===
        Q = Q.at[i_start:i_end, 1, 0].set(Q[i_start:i_end, j_int_first, 0])
        Q = Q.at[i_start:i_end, 1, 1].set(-Q[i_start:i_end, j_int_first, 1])
        Q = Q.at[i_start:i_end, 1, 2].set(-Q[i_start:i_end, j_int_first, 2])
        Q = Q.at[i_start:i_end, 1, 3].set(-Q[i_start:i_end, j_int_first, 3])
        
        Q = Q.at[i_start:i_end, 0, 0].set(Q[i_start:i_end, 1, 0])
        Q = Q.at[i_start:i_end, 0, 1].set(2*Q[i_start:i_end, 1, 1] - Q[i_start:i_end, j_int_first, 1])
        Q = Q.at[i_start:i_end, 0, 2].set(2*Q[i_start:i_end, 1, 2] - Q[i_start:i_end, j_int_first, 2])
        Q = Q.at[i_start:i_end, 0, 3].set(Q[i_start:i_end, 1, 3])
        
        # === Surface BC: Wake cut (periodic) ===
        lower_wake_int_j0 = Q[nghost:i_start, j_int_first, :]
        upper_wake_int_j0 = Q[i_end:i_upper_end, j_int_first, :]
        
        avg_j0 = 0.5 * (lower_wake_int_j0 + upper_wake_int_j0[::-1, :])
        Q = Q.at[nghost:i_start, j_int_first, :].set(avg_j0)
        Q = Q.at[i_end:i_upper_end, j_int_first, :].set(avg_j0[::-1, :])
        
        lower_wake_int_j0 = Q[nghost:i_start, j_int_first, :]
        lower_wake_int_j1 = Q[nghost:i_start, j_int_first + 1, :]
        upper_wake_int_j0 = Q[i_end:i_upper_end, j_int_first, :]
        upper_wake_int_j1 = Q[i_end:i_upper_end, j_int_first + 1, :]
        
        Q = Q.at[nghost:i_start, 1, :].set(upper_wake_int_j0[::-1, :])
        Q = Q.at[nghost:i_start, 0, :].set(upper_wake_int_j1[::-1, :])
        Q = Q.at[i_end:i_upper_end, 1, :].set(lower_wake_int_j0[::-1, :])
        Q = Q.at[i_end:i_upper_end, 0, :].set(lower_wake_int_j1[::-1, :])
        
        # === Farfield BC: J-direction (per-case freestream) ===
        p_int = Q[nghost:-nghost, j_int_last, 0]
        u_int = Q[nghost:-nghost, j_int_last, 1]
        v_int = Q[nghost:-nghost, j_int_last, 2]
        nu_t_int = Q[nghost:-nghost, j_int_last, 3]
        
        U_n = u_int * nx + v_int * ny
        is_outflow = jnp.logical_or(U_n >= 0, is_i_outlet)
        is_inflow = ~is_outflow
        
        u_b = jnp.where(is_inflow, u_inf, u_int)
        v_b = jnp.where(is_inflow, v_inf, v_int)
        nu_t_b = jnp.where(is_inflow, nu_t_inf, nu_t_int)
        p_b = jnp.where(is_inflow, p_int, p_inf)
        p_b = jnp.where(is_i_outlet, p_int, p_b)
        
        Q = Q.at[nghost:-nghost, -2, 0].set(2 * p_b - p_int)
        Q = Q.at[nghost:-nghost, -2, 1].set(2 * u_b - u_int)
        Q = Q.at[nghost:-nghost, -2, 2].set(2 * v_b - v_int)
        Q = Q.at[nghost:-nghost, -2, 3].set(2 * nu_t_b - nu_t_int)
        
        Q = Q.at[nghost:-nghost, -1, 0].set(2 * Q[nghost:-nghost, -2, 0] - p_b)
        Q = Q.at[nghost:-nghost, -1, 1].set(2 * Q[nghost:-nghost, -2, 1] - u_b)
        Q = Q.at[nghost:-nghost, -1, 2].set(2 * Q[nghost:-nghost, -2, 2] - v_b)
        Q = Q.at[nghost:-nghost, -1, 3].set(2 * Q[nghost:-nghost, -2, 3] - nu_t_b)
        
        # === Farfield BC: I-direction (downstream outlet) ===
        j_end = NJ + nghost
        Q = Q.at[1, :j_end, :].set(Q[nghost, :j_end, :])
        Q = Q.at[0, :j_end, :].set(2 * Q[nghost, :j_end, :] - Q[nghost + 1, :j_end, :])
        Q = Q.at[-2, :j_end, :].set(Q[-nghost - 1, :j_end, :])
        Q = Q.at[-1, :j_end, :].set(2 * Q[-nghost - 1, :j_end, :] - Q[-nghost - 2, :j_end, :])
        
        # Corner ghost cells
        corner_ll = Q[nghost, j_int_last, :]
        Q = Q.at[0, -2, :].set(corner_ll)
        Q = Q.at[0, -1, :].set(corner_ll)
        Q = Q.at[1, -2, :].set(corner_ll)
        Q = Q.at[1, -1, :].set(corner_ll)
        
        corner_lr = Q[-nghost - 1, j_int_last, :]
        Q = Q.at[-2, -2, :].set(corner_lr)
        Q = Q.at[-2, -1, :].set(corner_lr)
        Q = Q.at[-1, -2, :].set(corner_lr)
        Q = Q.at[-1, -1, :].set(corner_lr)
        
        return Q
    
    # Create vmapped version: Q batched, freestream values batched
    apply_bc_batch_vmap = jax.vmap(
        apply_bc_single,
        in_axes=(0, 0, 0, 0, 0)
    )
    
    @jax.jit
    def apply_bc_batch(Q_batch, u_inf, v_inf, p_inf, nu_t_inf):
        """Apply BCs to batch with per-case freestream (JIT-compiled)."""
        return apply_bc_batch_vmap(Q_batch, u_inf, v_inf, p_inf, nu_t_inf)
    
    return apply_bc_batch


# =============================================================================
# Unified Batch Step Function
# =============================================================================

def make_batch_step_jit(NI: int, NJ: int, n_wake_points: int,
                        nx: jnp.ndarray, ny: jnp.ndarray,
                        Si_x: jnp.ndarray, Si_y: jnp.ndarray,
                        Sj_x: jnp.ndarray, Sj_y: jnp.ndarray,
                        volume: jnp.ndarray,
                        beta: float, k4: float, nu: float,
                        smoothing_epsilon: float = 0.2,
                        smoothing_passes: int = 2,
                        nghost: int = NGHOST):
    """
    Create a JIT-compiled batch step function with all grid data baked in.
    
    Parameters
    ----------
    NI, NJ : int
        Number of interior cells.
    n_wake_points : int
        Number of wake points.
    nx, ny : jnp.ndarray
        Farfield normals (NI,).
    Si_x, Si_y, Sj_x, Sj_y, volume : jnp.ndarray
        Grid metrics.
    beta : float
        Artificial compressibility.
    k4 : float
        JST 4th-order dissipation.
    nu : float
        Kinematic viscosity (laminar).
    smoothing_epsilon, smoothing_passes : float, int
        Residual smoothing parameters.
    nghost : int
        Number of ghost cells.
        
    Returns
    -------
    batch_step : callable
        JIT-compiled function: 
        Q_new, R = batch_step(Q_batch, dt_batch, flow_conditions, alpha_rk)
        
        Q_batch: (n_batch, NI+2g, NJ+2g, 4)
        dt_batch: (n_batch, NI, NJ)
        flow_conditions: BatchFlowConditionsJax
        alpha_rk: RK stage coefficient
    """
    # Pre-compute volume inverse
    volume_inv = 1.0 / volume
    
    # Create batch BC function
    apply_bc_batch = make_apply_bc_batch_jit(NI, NJ, n_wake_points, nx, ny, nghost)
    
    def compute_mu_eff_batch(Q_batch):
        """Compute effective viscosity for batch."""
        Q_int = Q_batch[:, nghost:-nghost, nghost:-nghost, :]
        nu_tilde = jnp.maximum(Q_int[:, :, :, 3], 0.0)  # (B, NI, NJ)
        
        if nu > 0:
            chi = nu_tilde / nu
            cv1 = 7.1
            chi3 = chi ** 3
            f_v1 = chi3 / (chi3 + cv1 ** 3)
            mu_t = nu_tilde * f_v1
            mu_eff = nu + mu_t
        else:
            mu_eff = jnp.full_like(nu_tilde, nu)
        
        return mu_eff
    
    @jax.jit
    def batch_step(Q_batch, Q0_batch, dt_batch, u_inf, v_inf, p_inf, nu_t_inf, alpha_rk):
        """
        Perform one RK stage for entire batch.
        
        Parameters
        ----------
        Q_batch : jnp.ndarray
            Current state (n_batch, NI+2g, NJ+2g, 4).
        Q0_batch : jnp.ndarray
            Initial state for this RK step (same shape).
        dt_batch : jnp.ndarray
            Local timesteps (n_batch, NI, NJ).
        u_inf, v_inf, p_inf, nu_t_inf : jnp.ndarray
            Per-case freestream (n_batch,).
        alpha_rk : float
            RK stage coefficient.
            
        Returns
        -------
        Q_new : jnp.ndarray
            Updated state (n_batch, NI+2g, NJ+2g, 4).
        R : jnp.ndarray
            Residual (n_batch, NI, NJ, 4).
        """
        # Apply boundary conditions
        Q_batch = apply_bc_batch(Q_batch, u_inf, v_inf, p_inf, nu_t_inf)
        
        # Convective fluxes
        R = compute_fluxes_batch(
            Q_batch, Si_x, Si_y, Sj_x, Sj_y, beta, k4, nghost
        )
        
        # Gradients
        grad = compute_gradients_batch(
            Q_batch, Si_x, Si_y, Sj_x, Sj_y, volume, nghost
        )
        
        # Viscous fluxes
        mu_eff = compute_mu_eff_batch(Q_batch)
        R_visc = compute_viscous_fluxes_batch(
            grad, Si_x, Si_y, Sj_x, Sj_y, mu_eff
        )
        R = R + R_visc
        
        # Smoothing
        if smoothing_epsilon > 0 and smoothing_passes > 0:
            R = smooth_residual_batch(R, smoothing_epsilon, smoothing_passes)
        
        # RK update from Q0
        # Q_new_int = Q0_int + alpha * dt / vol * R
        Q_int_new = Q0_batch[:, nghost:-nghost, nghost:-nghost, :] + \
            alpha_rk * (dt_batch * volume_inv[None, :, :])[:, :, :, None] * R
        
        Q_new = Q0_batch.at[:, nghost:-nghost, nghost:-nghost, :].set(Q_int_new)
        
        return Q_new, R
    
    @jax.jit
    def compute_dt_batch(Q_batch, cfl):
        """Compute local timesteps for batch."""
        return compute_timestep_batch(
            Q_batch, Si_x, Si_y, Sj_x, Sj_y, volume, beta, cfl, nghost, nu
        )
    
    return batch_step, apply_bc_batch, compute_dt_batch


# =============================================================================
# Phase 4: Batch RANS Solver
# =============================================================================

from pathlib import Path
from src.grid.plot3d import read_plot3d
from src.grid.metrics import MetricComputer
from src.numerics.forces import compute_surface_forces_jax


class BatchRANSSolver:
    """
    Batch RANS Solver for running multiple AoA cases simultaneously.
    
    Uses JAX vmap to process N cases in parallel, maximizing GPU utilization.
    """
    
    # RK5 coefficients
    RK_ALPHAS = [0.25, 0.166666667, 0.375, 0.5, 1.0]
    
    def __init__(self, 
                 grid_file: str,
                 flow_conditions: BatchFlowConditions,
                 beta: float = 10.0,
                 k4: float = 0.04,
                 cfl_start: float = 0.1,
                 cfl_target: float = 5.0,
                 cfl_ramp_iters: int = 500,
                 max_iter: int = 5000,
                 tol: float = 1e-10,
                 print_freq: int = 50,
                 smoothing_epsilon: float = 0.2,
                 smoothing_passes: int = 2,
                 n_wake: int = 30,
                 output_dir: str = "output/batch"):
        """
        Initialize Batch RANS Solver.
        
        Parameters
        ----------
        grid_file : str
            Path to grid file (.p3d or .dat).
        flow_conditions : BatchFlowConditions
            Flow conditions for all cases.
        beta : float
            Artificial compressibility parameter.
        k4 : float
            JST 4th-order dissipation.
        cfl_start, cfl_target : float
            CFL ramping range.
        cfl_ramp_iters : int
            Number of iterations for CFL ramp.
        max_iter : int
            Maximum iterations.
        tol : float
            Convergence tolerance.
        print_freq : int
            Print frequency.
        smoothing_epsilon, smoothing_passes : float, int
            Residual smoothing parameters.
        n_wake : int
            Number of wake points.
        output_dir : str
            Output directory.
        """
        self.flow_conditions = flow_conditions
        self.n_batch = flow_conditions.n_batch
        self.beta = beta
        self.k4 = k4
        self.cfl_start = cfl_start
        self.cfl_target = cfl_target
        self.cfl_ramp_iters = cfl_ramp_iters
        self.max_iter = max_iter
        self.tol = tol
        self.print_freq = print_freq
        self.n_wake = n_wake
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Per-case tracking
        self.iteration = 0
        self.residual_history = [[] for _ in range(self.n_batch)]
        self.converged = np.zeros(self.n_batch, dtype=bool)
        
        # Load grid
        self._load_grid(grid_file)
        
        # Compute grid metrics
        self._compute_metrics()
        
        # Initialize state
        self._initialize_state()
        
        # Create JIT-compiled batch functions
        self._create_batch_functions(smoothing_epsilon, smoothing_passes)
        
        print(f"\n{'='*60}")
        print(f"Batch RANS Solver Initialized")
        print(f"{'='*60}")
        print(f"Grid size: {self.NI} x {self.NJ} cells")
        print(f"Batch size: {self.n_batch} cases")
        print(f"Alpha range: {flow_conditions.alpha_deg.min():.1f}° to {flow_conditions.alpha_deg.max():.1f}°")
        print(f"Reynolds: {flow_conditions.reynolds[0]:.2e}")
        print(f"Target CFL: {self.cfl_target}")
        print(f"Max iterations: {self.max_iter}")
        print(f"{'='*60}\n")
    
    def _load_grid(self, grid_file: str):
        """Load grid from Plot3D file or generate from airfoil .dat."""
        grid_path = Path(grid_file)
        if not grid_path.exists():
            raise FileNotFoundError(f"Grid file not found: {grid_path}")
        
        suffix = grid_path.suffix.lower()
        
        if suffix in ['.p3d', '.x', '.xyz']:
            print(f"Loading grid from: {grid_path}")
            self.X, self.Y = read_plot3d(str(grid_path))
        elif suffix == '.dat':
            print(f"Generating grid from airfoil: {grid_path}")
            from src.grid.mesher import Construct2DWrapper, GridOptions
            
            # Find Construct2D binary
            binary_paths = [
                Path("bin/construct2d"),
                Path("./construct2d"),
                Path("/usr/local/bin/construct2d"),
            ]
            binary_path = None
            for p in binary_paths:
                if p.exists():
                    binary_path = p
                    break
            
            if binary_path is None:
                raise FileNotFoundError(
                    "Construct2D binary not found. Please provide a .p3d grid file "
                    "or install Construct2D."
                )
            
            wrapper = Construct2DWrapper(str(binary_path))
            grid_opts = GridOptions(
                n_surface=129,
                n_normal=65,
                y_plus=1.0,
                reynolds=float(self.flow_conditions.reynolds[0])
            )
            self.X, self.Y = wrapper.generate(str(grid_path), grid_opts)
        else:
            raise ValueError(f"Unsupported grid file format: {suffix}")
        
        self.NI = self.X.shape[0] - 1
        self.NJ = self.X.shape[1] - 1
        
        print(f"Grid loaded: {self.NI} x {self.NJ} cells")
    
    def _compute_metrics(self):
        """Compute grid metrics."""
        print("Computing grid metrics...")
        computer = MetricComputer(self.X, self.Y, wall_j=0)
        self.metrics = computer.compute()
        
        # Transfer to JAX
        self.Si_x_jax = jnp.array(self.metrics.Si_x)
        self.Si_y_jax = jnp.array(self.metrics.Si_y)
        self.Sj_x_jax = jnp.array(self.metrics.Sj_x)
        self.Sj_y_jax = jnp.array(self.metrics.Sj_y)
        self.volume_jax = jnp.array(self.metrics.volume)
        
        # Farfield normals
        Sj_x_ff = self.metrics.Sj_x[:, -1]
        Sj_y_ff = self.metrics.Sj_y[:, -1]
        Sj_mag = np.sqrt(Sj_x_ff**2 + Sj_y_ff**2) + 1e-12
        self.nx_ff = jnp.array(Sj_x_ff / Sj_mag)
        self.ny_ff = jnp.array(Sj_y_ff / Sj_mag)
        
        gcl = computer.validate_gcl()
        print(f"  {gcl}")
    
    def _initialize_state(self):
        """Initialize state for all cases."""
        print("Initializing flow state for batch...")
        
        # Create single freestream state (use average alpha for initial)
        from src.solvers.boundary_conditions import initialize_state, FreestreamConditions
        avg_alpha = float(np.mean(self.flow_conditions.alpha_deg))
        avg_re = float(np.mean(self.flow_conditions.reynolds))
        
        freestream = FreestreamConditions.from_mach_alpha(0.0, avg_alpha, avg_re)
        Q_single = initialize_state(self.NI, self.NJ, freestream)
        
        # Replicate for batch
        self.batch_state = BatchState.from_single_ic(Q_single, self.n_batch, nghost=NGHOST)
        self.batch_state.to_gpu()
        
        # Convert flow conditions to JAX
        self.flow_jax = self.flow_conditions.to_jax()
        
        # Pre-compute laminar viscosity
        self.nu = float(1.0 / self.flow_conditions.reynolds[0]) if self.flow_conditions.reynolds[0] > 0 else 0.0
        
        print(f"  Initialized {self.n_batch} cases")
    
    def _create_batch_functions(self, smoothing_epsilon, smoothing_passes):
        """Create JIT-compiled batch functions."""
        print("Creating JIT-compiled batch functions...")
        
        self.batch_step, self.apply_bc_batch, self.compute_dt = make_batch_step_jit(
            self.NI, self.NJ, self.n_wake,
            self.nx_ff, self.ny_ff,
            self.Si_x_jax, self.Si_y_jax,
            self.Sj_x_jax, self.Sj_y_jax,
            self.volume_jax,
            self.beta, self.k4, self.nu,
            smoothing_epsilon, smoothing_passes,
            NGHOST
        )
        
        # Warmup JIT
        print("  Warming up JIT compilation...")
        Q_test = self.batch_state.Q_jax
        dt_test = self.compute_dt(Q_test, self.cfl_start)
        for alpha_rk in self.RK_ALPHAS[:2]:
            Q_test, _ = self.batch_step(
                Q_test, Q_test, dt_test,
                self.flow_jax.u_inf, self.flow_jax.v_inf,
                self.flow_jax.p_inf, self.flow_jax.nu_t_inf,
                alpha_rk
            )
        jax.block_until_ready(Q_test)
        print("  JIT compilation complete")
    
    def _get_cfl(self, iteration: int) -> float:
        """Get CFL with ramping."""
        if iteration >= self.cfl_ramp_iters:
            return self.cfl_target
        t = iteration / self.cfl_ramp_iters
        return self.cfl_start + t * (self.cfl_target - self.cfl_start)
    
    def step(self):
        """Perform one iteration (5-stage RK)."""
        cfl = self._get_cfl(self.iteration)
        
        # Compute timestep
        dt = self.compute_dt(self.batch_state.Q_jax, cfl)
        
        # RK5 stages
        Q0 = self.batch_state.Q_jax
        Q = Q0
        
        for alpha_rk in self.RK_ALPHAS:
            Q, R = self.batch_step(
                Q, Q0, dt,
                self.flow_jax.u_inf, self.flow_jax.v_inf,
                self.flow_jax.p_inf, self.flow_jax.nu_t_inf,
                alpha_rk
            )
        
        # Final BC
        Q = self.apply_bc_batch(
            Q, 
            self.flow_jax.u_inf, self.flow_jax.v_inf,
            self.flow_jax.p_inf, self.flow_jax.nu_t_inf
        )
        
        self.batch_state.Q_jax = Q
        self.R_jax = R
        self.iteration += 1
    
    def get_residual_rms(self) -> np.ndarray:
        """Compute RMS residual for each case."""
        # Compute fresh residual on current state
        Q = self.apply_bc_batch(
            self.batch_state.Q_jax,
            self.flow_jax.u_inf, self.flow_jax.v_inf,
            self.flow_jax.p_inf, self.flow_jax.nu_t_inf
        )
        R = compute_fluxes_batch(
            Q, self.Si_x_jax, self.Si_y_jax,
            self.Sj_x_jax, self.Sj_y_jax,
            self.beta, self.k4, NGHOST
        )
        
        # RMS over spatial dimensions
        R_rho = R[:, :, :, 0]  # (n_batch, NI, NJ)
        rms = jnp.sqrt(jnp.mean(R_rho**2, axis=(1, 2)))
        return np.array(rms)
    
    def compute_forces(self) -> BatchForces:
        """Compute aerodynamic forces for all cases."""
        Q_batch = self.batch_state.Q_jax
        n_batch = self.n_batch
        
        # Compute effective viscosity
        Q_int = Q_batch[:, NGHOST:-NGHOST, NGHOST:-NGHOST, :]
        nu_tilde = jnp.maximum(Q_int[:, :, :, 3], 0.0)
        
        if self.nu > 0:
            chi = nu_tilde / self.nu
            cv1 = 7.1
            f_v1 = chi**3 / (chi**3 + cv1**3)
            mu_eff_batch = self.nu + nu_tilde * f_v1
        else:
            mu_eff_batch = jnp.zeros_like(nu_tilde)
        
        # Compute forces for each case
        CL = np.zeros(n_batch)
        CD = np.zeros(n_batch)
        CD_p = np.zeros(n_batch)
        CD_f = np.zeros(n_batch)
        
        for i in range(n_batch):
            Q_i = Q_batch[i]
            mu_eff_i = mu_eff_batch[i]
            alpha_i = float(self.flow_conditions.alpha_deg[i])
            
            Fx_p, Fy_p, Fx_v, Fy_v = compute_surface_forces_jax(
                Q_i, self.Sj_x_jax, self.Sj_y_jax,
                self.volume_jax, mu_eff_i, self.n_wake, NGHOST
            )
            
            Fx = Fx_p + Fx_v
            Fy = Fy_p + Fy_v
            
            alpha_rad = np.radians(alpha_i)
            cos_a = np.cos(alpha_rad)
            sin_a = np.sin(alpha_rad)
            
            D = float(Fx * cos_a + Fy * sin_a)
            L = float(Fy * cos_a - Fx * sin_a)
            D_p = float(Fx_p * cos_a + Fy_p * sin_a)
            D_f = float(Fx_v * cos_a + Fy_v * sin_a)
            
            q_inf = 0.5  # rho=1, V=1
            CL[i] = L / q_inf
            CD[i] = D / q_inf
            CD_p[i] = D_p / q_inf
            CD_f[i] = D_f / q_inf
        
        return BatchForces(
            CL=CL, CD=CD, CD_p=CD_p, CD_f=CD_f,
            alpha_deg=self.flow_conditions.alpha_deg.copy()
        )
    
    def run(self) -> BatchForces:
        """Run batch simulation to convergence."""
        print(f"\n{'='*60}")
        print("Starting Batch Steady-State Iteration")
        print(f"{'='*60}")
        print(f"{'Iter':>6} {'Max Res':>12} {'Min Res':>12} {'CFL':>8} {'Conv':>6}")
        print(f"{'-'*52}")
        
        initial_res = None
        
        for n in range(self.max_iter):
            self.step()
            
            # Check residuals periodically
            if n % self.print_freq == 0 or n == 0:
                res_rms = self.get_residual_rms()
                
                if initial_res is None:
                    initial_res = res_rms.copy()
                
                # Track per-case history
                for i in range(self.n_batch):
                    self.residual_history[i].append(float(res_rms[i]))
                
                # Check convergence
                self.converged = res_rms < self.tol
                n_conv = int(np.sum(self.converged))
                
                cfl = self._get_cfl(self.iteration)
                print(f"{self.iteration:>6d} {res_rms.max():>12.4e} {res_rms.min():>12.4e} "
                      f"{cfl:>8.2f} {n_conv:>3}/{self.n_batch}")
                
                # Early termination if all converged
                if np.all(self.converged):
                    print(f"\n{'='*60}")
                    print(f"ALL {self.n_batch} CASES CONVERGED at iteration {self.iteration}")
                    print(f"{'='*60}")
                    break
                
                # Divergence check
                if np.any(res_rms > 1000 * initial_res):
                    print(f"\n{'='*60}")
                    print(f"DIVERGED at iteration {self.iteration}")
                    print(f"{'='*60}")
                    break
        else:
            print(f"\n{'='*60}")
            print(f"Maximum iterations ({self.max_iter}) reached")
            print(f"Converged: {np.sum(self.converged)}/{self.n_batch} cases")
            print(f"{'='*60}")
        
        # Compute final forces
        forces = self.compute_forces()
        
        # Save results
        csv_path = self.output_dir / "batch_results.csv"
        forces.save_csv(str(csv_path))
        print(f"\nResults saved to: {csv_path}")
        
        return forces
    
    def get_case_Q(self, i: int) -> np.ndarray:
        """Get Q array for a single case."""
        return np.array(self.batch_state.Q_jax[i])
    
    def save_residual_history(self, filename: str = None):
        """Save per-case residual history."""
        if filename is None:
            filename = self.output_dir / "residual_history.csv"
        
        # Find max history length
        max_len = max(len(h) for h in self.residual_history) if self.residual_history else 0
        if max_len == 0:
            print("No residual history to save")
            return
        
        with open(filename, 'w') as f:
            # Header
            headers = ['iteration'] + [f'alpha_{self.flow_conditions.alpha_deg[i]:.1f}' 
                                       for i in range(self.n_batch)]
            f.write(','.join(headers) + '\n')
            
            # Data rows
            for row_idx in range(max_len):
                row = [str(row_idx + 1)]
                for i in range(self.n_batch):
                    if row_idx < len(self.residual_history[i]):
                        row.append(f'{self.residual_history[i][row_idx]:.6e}')
                    else:
                        row.append('')
                f.write(','.join(row) + '\n')
        
        print(f"Residual history saved to: {filename}")
