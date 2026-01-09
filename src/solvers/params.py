from typing import NamedTuple
from src.physics.jax_config import jax, jnp

class PhysicsParams(NamedTuple):
    """
    JAX-compatible tuple for passing physics parameters to JIT kernels.
    
    This avoids identifying these scalars as static constants, allowing
    runtime updates without recompilation.
    """
    # Freestream Conditions
    p_inf: float
    u_inf: float
    v_inf: float
    nu_t_inf: float
    
    # Grid/Scheme Parameters
    beta: float
    k4: float
    
    # Physical Constants
    mu_laminar: float  # 1/Re
    
    # AFT Model Coefficients
    aft_gamma_coeff: float
    aft_re_omega_scale: float
    aft_log_divisor: float
    aft_sigmoid_center: float
    aft_sigmoid_slope: float
    aft_rate_scale: float
    aft_blend_threshold: float
    aft_blend_width: float
