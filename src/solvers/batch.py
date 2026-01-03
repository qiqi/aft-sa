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
