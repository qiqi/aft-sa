"""
Numerical update schemes for RANS solvers.

Implements positivity-preserving updates like the Patankar scheme
for the SA turbulence model working variable (nuHat).
"""

from src.physics.jax_config import jax, jnp

@jax.jit
def apply_patankar_update(
    Q: jnp.ndarray,
    dQ: jnp.ndarray,
    nu_index: int = 3
) -> jnp.ndarray:
    """Apply Patankar update to preserve positivity of a variable.
    
    The Patankar scheme ensures that the updated variable remains positive
    even when the update step dQ is negative and large, by using an implicit
    formulation:
        Q_new = Q_old / (1 - dQ/Q_old)  if dQ < 0
        Q_new = Q_old + dQ             if dQ >= 0
    
    Parameters
    ----------
    Q : jnp.ndarray
        Original state vector.
    dQ : jnp.ndarray
        Update increment (solution from linear system).
    nu_index : int, optional
        Index of the variable to apply Patankar to (default: 3 for nuHat).
        
    Returns
    -------
    jnp.ndarray
        Updated state vector.
    """
    # Standard explicit update for all components
    Q_new_explicit = Q + dQ
    
    # Extract working variable
    val_old = Q[..., nu_index]
    dVal = dQ[..., nu_index]
    val_explicit = Q_new_explicit[..., nu_index]
    
    # Patankar: when dVal < 0, use implicit form to ensure positivity
    # Safe when val_old > 0 since denominator = (val_old - dVal)/val_old > 0
    val_patankar = jnp.where(
        val_old > 0,
        val_old / (1.0 - dVal / val_old),
        val_explicit  # Fallback
    )
    
    # Use Patankar only for destruction dominated steps
    val_new = jnp.where(dVal < 0, val_patankar, val_explicit)
    
    # Apply physical floor
    val_new = jnp.maximum(val_new, 0.0)
    
    return Q_new_explicit.at[..., nu_index].set(val_new)
