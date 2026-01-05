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
        
        # Compute block Jacobians via finite difference
        J_diag = _compute_block_jacobians(residual_fn, Q, nghost, eps)
        
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
        
        Useful for GMRES where apply is called many times.
        """
        P_inv = self.P_inv
        NI, NJ = self.NI, self.NJ
        
        @jax.jit
        def _apply(v):
            was_flat = v.ndim == 1
            if was_flat:
                v = v.reshape(NI, NJ, 4)
            result = jnp.einsum('ijkl,ijl->ijk', P_inv, v)
            if was_flat:
                result = result.flatten()
            return result
        
        return _apply


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
    
    # Initialize Jacobian storage
    J_diag = jnp.zeros((NI, NJ, 4, 4))
    
    # 2×2 coloring pattern: (i%2, j%2) gives 4 colors
    # For each color, perturb all cells of that color for each variable
    for color_i in range(2):
        for color_j in range(2):
            # Create mask for this color
            I, J = jnp.meshgrid(jnp.arange(NI), jnp.arange(NJ), indexing='ij')
            mask = ((I % 2 == color_i) & (J % 2 == color_j))  # (NI, NJ)
            
            for var in range(4):
                # Perturb Q at all cells of this color for variable 'var'
                Q_pert = Q.at[
                    nghost:-nghost, nghost:-nghost, var
                ].add(eps * mask)
                
                R_pert = residual_fn(Q_pert)  # (NI, NJ, 4)
                
                # Finite difference: dR/dQ[var] at cells of this color
                dR = (R_pert - R0) / eps  # (NI, NJ, 4)
                
                # Store in Jacobian (only for cells of this color)
                # J_diag[i,j,:,var] = dR[i,j,:] where mask[i,j] is True
                J_diag = J_diag.at[:, :, :, var].add(
                    dR * mask[:, :, None]
                )
    
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
    
    J_diag = jnp.zeros((NI, NJ, 4, 4))
    
    # 2×2 coloring pattern
    for color_i in range(2):
        for color_j in range(2):
            I, J = jnp.meshgrid(jnp.arange(NI), jnp.arange(NJ), indexing='ij')
            mask = ((I % 2 == color_i) & (J % 2 == color_j)).astype(jnp.float64)
            
            for var in range(4):
                # Tangent vector: 1 at variable 'var' for cells of this color
                tangent = jnp.zeros_like(Q)
                tangent = tangent.at[nghost:-nghost, nghost:-nghost, var].set(mask)
                
                # JVP gives J @ tangent
                _, jvp_result = jax.jvp(residual_fn, (Q,), (tangent,))
                
                # Store in Jacobian
                J_diag = J_diag.at[:, :, :, var].add(
                    jvp_result * mask[:, :, None]
                )
    
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

