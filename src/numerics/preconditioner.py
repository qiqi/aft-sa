"""
Block-Jacobi Preconditioner for Newton-Krylov solver.

The preconditioner approximates the inverse of the implicit Euler system matrix:
    P ≈ V/dt · I + J_diag

where J_diag contains only the 4×4 diagonal blocks ∂R[i,j]/∂Q[i,j].

This provides a good approximation for the coupled pressure-velocity-turbulence
system and can be inverted cheaply (batched 4×4 matrix inversion).
"""

from dataclasses import dataclass
from typing import Callable, Tuple

from src.physics.jax_config import jax, jnp
from src.constants import NGHOST

# Cache for JIT-compiled preconditioner apply functions
_precond_apply_cache = {}

# Cache for JIT-compiled block Jacobian computation
_block_jacobian_cache = {}


@dataclass
class BlockJacobiPreconditioner:
    """Block-Jacobi preconditioner with 4×4 blocks per cell.
    
    Attributes
    ----------
    P_inv : jnp.ndarray
        Inverted preconditioner blocks, shape (NI, NJ, 4, 4).
    NI, NJ : int
        Grid dimensions (interior cells).
    """
    P_inv: jnp.ndarray
    NI: int
    NJ: int
    
    @staticmethod
    def compute(
        residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
        Q: jnp.ndarray,
        dt: jnp.ndarray,
        volume: jnp.ndarray,
        nghost: int = NGHOST,
        eps: float = 1e-7,
    ) -> 'BlockJacobiPreconditioner':
        """Compute block-Jacobi preconditioner at current state.
        
        Forms the implicit Euler system matrix diagonal blocks:
            P[i,j] = V[i,j]/dt[i,j] · I + ∂R[i,j]/∂Q[i,j]
        
        Then inverts each 4×4 block.
        
        Parameters
        ----------
        residual_fn : callable
            R(Q) -> (NI, NJ, 4) residual function.
            Should include smoothing if used with Newton solver.
        Q : jnp.ndarray
            Current state (NI+2*nghost, NJ+2*nghost, 4).
        dt : jnp.ndarray
            Local timestep (NI, NJ) from CFL condition.
        volume : jnp.ndarray
            Cell volumes (NI, NJ).
        nghost : int
            Number of ghost cells.
        eps : float
            Finite difference step size for Jacobian approximation.
            
        Returns
        -------
        BlockJacobiPreconditioner
            Preconditioner with inverted blocks.
        """
        NI, NJ = volume.shape
        
        # Compute block Jacobians using JVP (exact automatic differentiation)
        # Use cached JIT function to avoid retracing
        cache_key = (id(residual_fn), NI, NJ, nghost)
        if cache_key not in _block_jacobian_cache:
            _block_jacobian_cache[cache_key] = _make_block_jacobian_jit(
                residual_fn, NI, NJ, nghost
            )
        compute_jacobians_jit = _block_jacobian_cache[cache_key]
        J_diag = compute_jacobians_jit(Q)
        
        # Form preconditioner: P = V/dt · I + J_diag
        # Diagonal scaling: V/dt for each cell, broadcast to 4×4 identity
        diag_scale = (volume / dt)[:, :, None, None]  # (NI, NJ, 1, 1)
        eye4 = jnp.eye(4)  # (4, 4)
        P = diag_scale * eye4 + J_diag  # (NI, NJ, 4, 4)
        
        # Invert each 4×4 block
        P_inv = _batch_invert_4x4(P)
        
        return BlockJacobiPreconditioner(P_inv=P_inv, NI=NI, NJ=NJ)
    
    def apply(self, v: jnp.ndarray) -> jnp.ndarray:
        """Apply P^{-1} to vector v.
        
        Parameters
        ----------
        v : jnp.ndarray
            Input vector, shape (NI, NJ, 4) or flattened (NI*NJ*4,).
            
        Returns
        -------
        jnp.ndarray
            P^{-1} @ v, same shape as input.
        """
        was_flat = v.ndim == 1
        if was_flat:
            v = v.reshape(self.NI, self.NJ, 4)
        
        # Batched matrix-vector: P_inv[i,j] @ v[i,j] for each cell
        result = jnp.einsum('ijkl,ijl->ijk', self.P_inv, v)
        
        if was_flat:
            result = result.flatten()
        
        return result
    
    def apply_jit(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Return a JIT-compiled apply function.
        
        Uses caching to avoid JAX retracing when called multiple times with
        the same grid dimensions.
        
        Useful for GMRES where apply is called many times.
        """
        P_inv = self.P_inv
        NI, NJ = self.NI, self.NJ
        
        # Get or create cached apply implementation
        cache_key = (NI, NJ)
        if cache_key not in _precond_apply_cache:
            @jax.jit
            def _apply_impl(P_inv_data, v):
                was_flat = v.ndim == 1
                if was_flat:
                    v = v.reshape(NI, NJ, 4)
                result = jnp.einsum('ijkl,ijl->ijk', P_inv_data, v)
                if was_flat:
                    result = result.flatten()
                return result
            _precond_apply_cache[cache_key] = _apply_impl
        
        _apply_impl = _precond_apply_cache[cache_key]
        
        # Create wrapper that captures current P_inv
        def _apply(v):
            return _apply_impl(P_inv, v)
        
        # Store cache key for GMRES caching
        _apply._cache_key = cache_key
        
        return _apply


def _make_block_jacobian_jit(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    NI: int,
    NJ: int,
    nghost: int,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Create a JIT-compiled block Jacobian computation function.
    
    Uses JVP (forward-mode AD) with graph coloring to compute diagonal
    blocks ∂R[i,j]/∂Q[i,j] efficiently.
    
    The function is JIT-compiled and cached to avoid retracing on
    subsequent calls with the same residual function and grid dimensions.
    
    Parameters
    ----------
    residual_fn : callable
        R(Q) -> (NI, NJ, 4) residual function.
    NI, NJ : int
        Interior grid dimensions.
    nghost : int
        Number of ghost cells.
        
    Returns
    -------
    compute_jacobians : callable
        JIT-compiled function Q -> J_diag (NI, NJ, 4, 4).
    """
    # Pre-compute color masks (static, computed once)
    I, J = jnp.meshgrid(jnp.arange(NI), jnp.arange(NJ), indexing='ij')
    color_masks = jnp.stack([
        ((I % 2 == ci) & (J % 2 == cj)).astype(jnp.float64)
        for ci in range(2) for cj in range(2)
    ])  # (4, NI, NJ)
    
    @jax.jit
    def compute_jacobians(Q: jnp.ndarray) -> jnp.ndarray:
        """Compute block Jacobians ∂R[i,j]/∂Q[i,j] using JVP."""
        
        # Create all 16 tangent vectors in parallel
        def make_tangent(k):
            color = k // 4
            var = k % 4
            mask = color_masks[color]
            tangent = jnp.zeros_like(Q)
            tangent = tangent.at[nghost:-nghost, nghost:-nghost, var].set(mask)
            return tangent
        
        tangents = jax.vmap(make_tangent)(jnp.arange(16))  # (16, NI+2*ng, NJ+2*ng, 4)
        
        # Compute all 16 JVPs in parallel
        def single_jvp(tangent):
            _, jvp_result = jax.jvp(residual_fn, (Q,), (tangent,))
            return jvp_result
        
        jvp_results = jax.vmap(single_jvp)(tangents)  # (16, NI, NJ, 4)
        
        # Assemble Jacobian blocks
        jvp_reshaped = jvp_results.reshape(4, 4, NI, NJ, 4)
        masks_expanded = color_masks[:, None, :, :, None]
        jvp_weighted = (jvp_reshaped * masks_expanded).sum(axis=0)
        J_diag = jvp_weighted.transpose(1, 2, 3, 0)
        
        return J_diag
    
    return compute_jacobians


def _compute_block_jacobians(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    Q: jnp.ndarray,
    nghost: int,
    eps: float = 1e-7,
) -> jnp.ndarray:
    """Compute diagonal block Jacobians ∂R[i,j]/∂Q[i,j] for all cells.
    
    Uses finite difference with graph coloring to compute exact diagonal blocks
    efficiently. The grid is partitioned into 4 colors (2×2 checkerboard pattern)
    such that no two cells of the same color are adjacent in the immediate stencil.
    
    For each color, we perturb all cells of that color simultaneously. Since they
    don't share immediate neighbors, the perturbation at cell (i,j) only affects
    R[i,j], giving us the correct diagonal block.
    
    Note: For stencils larger than 5-point (like JST with 4th-order dissipation),
    this gives an approximation. The approximation is still effective for
    preconditioning purposes.
    
    This implementation is fully vectorized using vmap to compute all 16
    perturbations (4 colors × 4 variables) in parallel on GPU.
    
    Parameters
    ----------
    residual_fn : callable
        R(Q) -> (NI, NJ, 4) residual function.
    Q : jnp.ndarray
        State array (NI+2*nghost, NJ+2*nghost, 4).
    nghost : int
        Number of ghost cells.
    eps : float
        Finite difference step size.
        
    Returns
    -------
    J_diag : jnp.ndarray
        Block Jacobians (NI, NJ, 4, 4) where J_diag[i,j] = ∂R[i,j]/∂Q[i,j].
    """
    NI = Q.shape[0] - 2 * nghost
    NJ = Q.shape[1] - 2 * nghost
    
    # Base residual
    R0 = residual_fn(Q)  # (NI, NJ, 4)
    
    # Create all 16 masks (4 colors × 4 variables)
    # masks[k] has shape (NI, NJ) for k = 4*color + var
    I, J = jnp.meshgrid(jnp.arange(NI), jnp.arange(NJ), indexing='ij')
    
    # Pre-compute color masks: (4, NI, NJ)
    color_masks = jnp.stack([
        ((I % 2 == ci) & (J % 2 == cj)).astype(jnp.float64)
        for ci in range(2) for cj in range(2)
    ])  # (4, NI, NJ)
    
    # Create all 16 perturbation vectors: (16, NI+2*nghost, NJ+2*nghost, 4)
    # For perturbation k = 4*color + var, we perturb variable 'var' at cells of 'color'
    def make_perturbation(k):
        color = k // 4
        var = k % 4
        mask = color_masks[color]  # (NI, NJ)
        pert = jnp.zeros_like(Q)
        pert = pert.at[nghost:-nghost, nghost:-nghost, var].set(eps * mask)
        return pert
    
    # Stack all perturbation vectors
    perturbations = jax.vmap(make_perturbation)(jnp.arange(16))  # (16, NI+2*ng, NJ+2*ng, 4)
    
    # Compute all 16 perturbed Q values
    Q_perturbed = Q[None, :, :, :] + perturbations  # (16, NI+2*ng, NJ+2*ng, 4)
    
    # Evaluate residual for all 16 perturbations in parallel
    R_perturbed = jax.vmap(residual_fn)(Q_perturbed)  # (16, NI, NJ, 4)
    
    # Compute finite differences
    dR = (R_perturbed - R0[None, :, :, :]) / eps  # (16, NI, NJ, 4)
    
    # Assemble Jacobian blocks - fully vectorized
    # J_diag[i,j,r,c] = dR/dQ[c] at cell (i,j), component r
    # For each perturbation k = 4*color + var, we have dR[k] which gives column 'var' of J
    # but only at cells of 'color'
    
    # Reshape dR to (4 colors, 4 vars, NI, NJ, 4)
    dR_reshaped = dR.reshape(4, 4, NI, NJ, 4)  # (color, var, NI, NJ, 4 residual components)
    
    # color_masks: (4, NI, NJ)
    # For each cell, exactly one color mask is 1, the rest are 0
    # We need to select the right dR for each cell based on its color
    
    # Expand masks for broadcasting: (4 colors, 1 var, NI, NJ, 1 residual)
    masks_expanded = color_masks[:, None, :, :, None]  # (4, 1, NI, NJ, 1)
    
    # Weight dR by masks and sum over colors
    # dR_reshaped * masks_expanded: (4, 4, NI, NJ, 4) - only one color contributes per cell
    # Sum over colors gives (4 vars, NI, NJ, 4 residual components)
    dR_weighted = (dR_reshaped * masks_expanded).sum(axis=0)  # (4, NI, NJ, 4)
    
    # Now dR_weighted[var, i, j, :] = dR/dQ[var] at cell (i,j)
    # We want J_diag[i, j, :, var] = dR/dQ[var]
    # So transpose: (4, NI, NJ, 4) -> (NI, NJ, 4, 4)
    J_diag = dR_weighted.transpose(1, 2, 3, 0)  # (NI, NJ, 4 residual, 4 vars)
    
    return J_diag


@jax.jit
def _batch_invert_4x4(P: jnp.ndarray) -> jnp.ndarray:
    """Invert batch of 4×4 matrices.
    
    Parameters
    ----------
    P : jnp.ndarray
        Batch of matrices, shape (..., 4, 4).
        
    Returns
    -------
    P_inv : jnp.ndarray
        Inverted matrices, shape (..., 4, 4).
        
    Notes
    -----
    Uses JAX's batched linear algebra. For singular matrices,
    returns pseudo-inverse or may produce NaN.
    """
    # Flatten batch dimensions, invert, reshape
    original_shape = P.shape
    batch_shape = original_shape[:-2]
    n_batch = 1
    for s in batch_shape:
        n_batch *= s
    
    P_flat = P.reshape(n_batch, 4, 4)
    
    # Batched inversion using vmap
    P_inv_flat = jax.vmap(jnp.linalg.inv)(P_flat)
    
    return P_inv_flat.reshape(original_shape)


def _compute_block_jacobians_jvp(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    Q: jnp.ndarray,
    nghost: int,
) -> jnp.ndarray:
    """Compute diagonal block Jacobians using JVP (exact, for reference).
    
    This is more expensive than finite difference but gives exact derivatives.
    Uses the same graph coloring approach but with JVP instead of FD.
    
    This implementation is fully vectorized using vmap to compute all 16
    JVPs (4 colors × 4 variables) in parallel on GPU.
    
    Parameters
    ----------
    residual_fn : callable
        R(Q) -> (NI, NJ, 4) residual function.
    Q : jnp.ndarray
        State array (NI+2*nghost, NJ+2*nghost, 4).
    nghost : int
        Number of ghost cells.
        
    Returns
    -------
    J_diag : jnp.ndarray
        Block Jacobians (NI, NJ, 4, 4).
    """
    NI = Q.shape[0] - 2 * nghost
    NJ = Q.shape[1] - 2 * nghost
    
    # Create color masks: (4, NI, NJ)
    I, J = jnp.meshgrid(jnp.arange(NI), jnp.arange(NJ), indexing='ij')
    color_masks = jnp.stack([
        ((I % 2 == ci) & (J % 2 == cj)).astype(jnp.float64)
        for ci in range(2) for cj in range(2)
    ])  # (4, NI, NJ)
    
    # Create all 16 tangent vectors: (16, NI+2*nghost, NJ+2*nghost, 4)
    def make_tangent(k):
        color = k // 4
        var = k % 4
        mask = color_masks[color]  # (NI, NJ)
        tangent = jnp.zeros_like(Q)
        tangent = tangent.at[nghost:-nghost, nghost:-nghost, var].set(mask)
        return tangent
    
    tangents = jax.vmap(make_tangent)(jnp.arange(16))  # (16, NI+2*ng, NJ+2*ng, 4)
    
    # Compute all 16 JVPs in parallel using vmap
    def single_jvp(tangent):
        _, jvp_result = jax.jvp(residual_fn, (Q,), (tangent,))
        return jvp_result
    
    jvp_results = jax.vmap(single_jvp)(tangents)  # (16, NI, NJ, 4)
    
    # Assemble Jacobian blocks - fully vectorized
    # Reshape to (4 colors, 4 vars, NI, NJ, 4)
    jvp_reshaped = jvp_results.reshape(4, 4, NI, NJ, 4)
    
    # Expand masks for broadcasting: (4 colors, 1 var, NI, NJ, 1 residual)
    masks_expanded = color_masks[:, None, :, :, None]
    
    # Weight by masks and sum over colors
    jvp_weighted = (jvp_reshaped * masks_expanded).sum(axis=0)  # (4, NI, NJ, 4)
    
    # Transpose: (4 vars, NI, NJ, 4 residual) -> (NI, NJ, 4 residual, 4 vars)
    J_diag = jvp_weighted.transpose(1, 2, 3, 0)
    
    return J_diag


# =============================================================================
# Utility functions for testing
# =============================================================================

def verify_preconditioner(
    P: BlockJacobiPreconditioner,
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    Q: jnp.ndarray,
    dt: jnp.ndarray,
    volume: jnp.ndarray,
    nghost: int = NGHOST,
) -> Tuple[float, float]:
    """Verify preconditioner quality by checking P^{-1} @ P ≈ I.
    
    Returns the maximum and mean absolute deviation from identity.
    
    Parameters
    ----------
    P : BlockJacobiPreconditioner
        Preconditioner to verify.
    residual_fn, Q, dt, volume, nghost
        Same as BlockJacobiPreconditioner.compute().
        
    Returns
    -------
    max_error : float
        Maximum |P^{-1} @ P - I| over all cells.
    mean_error : float
        Mean |P^{-1} @ P - I| over all cells.
    """
    # Recompute J_diag
    J_diag = _compute_block_jacobians(residual_fn, Q, nghost)
    
    # Form P again
    diag_scale = (volume / dt)[:, :, None, None]
    eye4 = jnp.eye(4)
    P_matrix = diag_scale * eye4 + J_diag
    
    # Compute P^{-1} @ P for each cell
    product = jnp.einsum('ijkl,ijlm->ijkm', P.P_inv, P_matrix)
    
    # Compare to identity
    eye4_broadcast = jnp.broadcast_to(eye4, product.shape)
    error = jnp.abs(product - eye4_broadcast)
    
    max_error = float(jnp.max(error))
    mean_error = float(jnp.mean(error))
    
    return max_error, mean_error

