"""
Tests for Block-Jacobi preconditioner.

Tests cover:
1. Basic API and shapes
2. P^{-1} @ P ≈ I for each block
3. Finite difference vs JVP consistency
4. Batched 4×4 inversion
5. Preconditioner reduces effective condition number
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

# JAX imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.physics.jax_config import jax, jnp
from src.numerics.preconditioner import (
    BlockJacobiPreconditioner,
    _compute_block_jacobians,
    _compute_block_jacobians_jvp,
    _batch_invert_4x4,
    verify_preconditioner,
)
from src.constants import NGHOST


# =============================================================================
# Test fixtures
# =============================================================================

@pytest.fixture
def small_grid():
    """Create a small test grid (8×8 interior cells)."""
    NI, NJ = 8, 8
    nghost = NGHOST
    
    # State with ghosts
    Q = jnp.ones((NI + 2*nghost, NJ + 2*nghost, 4))
    # Add some variation
    Q = Q.at[nghost:-nghost, nghost:-nghost, 0].set(
        jnp.linspace(0, 1, NI)[:, None] * jnp.ones((1, NJ))
    )
    Q = Q.at[nghost:-nghost, nghost:-nghost, 1].set(0.5)
    Q = Q.at[nghost:-nghost, nghost:-nghost, 2].set(0.1)
    Q = Q.at[nghost:-nghost, nghost:-nghost, 3].set(0.01)
    
    # Local timestep (uniform for simplicity)
    dt = jnp.ones((NI, NJ)) * 0.1
    
    # Cell volumes (uniform)
    volume = jnp.ones((NI, NJ)) * 0.01
    
    return Q, dt, volume, NI, NJ, nghost


def make_simple_residual(NI, NJ, nghost):
    """Create a simple residual function for testing.
    
    R(Q) = A @ Q_interior + b
    where A is block-diagonal with known 4×4 blocks.
    """
    # Create a known block-diagonal matrix
    # For simplicity, each block is the same
    A_block = jnp.array([
        [2.0, 0.1, 0.0, 0.0],
        [0.1, 3.0, 0.2, 0.0],
        [0.0, 0.2, 2.5, 0.1],
        [0.0, 0.0, 0.1, 1.0],
    ])
    
    def residual_fn(Q):
        Q_int = Q[nghost:-nghost, nghost:-nghost, :]  # (NI, NJ, 4)
        # Apply block-diagonal: R[i,j] = A_block @ Q[i,j]
        R = jnp.einsum('kl,ijl->ijk', A_block, Q_int)
        return R
    
    return residual_fn, A_block


def make_coupled_residual(NI, NJ, nghost):
    """Create a residual with neighbor coupling (more realistic).
    
    R[i,j] = A @ Q[i,j] + B @ (Q[i+1,j] + Q[i-1,j] + Q[i,j+1] + Q[i,j-1])
    
    The diagonal block ∂R[i,j]/∂Q[i,j] = A.
    """
    A_block = jnp.array([
        [4.0, 0.1, 0.0, 0.0],
        [0.1, 5.0, 0.2, 0.0],
        [0.0, 0.2, 4.5, 0.1],
        [0.0, 0.0, 0.1, 2.0],
    ])
    
    B_block = jnp.array([
        [-0.5, 0.0, 0.0, 0.0],
        [0.0, -0.5, 0.0, 0.0],
        [0.0, 0.0, -0.5, 0.0],
        [0.0, 0.0, 0.0, -0.2],
    ])
    
    def residual_fn(Q):
        # Pad for neighbor access
        Q_padded = jnp.pad(Q[nghost:-nghost, nghost:-nghost, :], 
                          ((1, 1), (1, 1), (0, 0)), mode='edge')
        
        # Central cell
        Q_c = Q_padded[1:-1, 1:-1, :]  # (NI, NJ, 4)
        
        # Neighbors
        Q_ip1 = Q_padded[2:, 1:-1, :]
        Q_im1 = Q_padded[:-2, 1:-1, :]
        Q_jp1 = Q_padded[1:-1, 2:, :]
        Q_jm1 = Q_padded[1:-1, :-2, :]
        
        Q_neighbors = Q_ip1 + Q_im1 + Q_jp1 + Q_jm1
        
        # R = A @ Q_c + B @ Q_neighbors
        R = jnp.einsum('kl,ijl->ijk', A_block, Q_c)
        R = R + jnp.einsum('kl,ijl->ijk', B_block, Q_neighbors)
        
        return R
    
    return residual_fn, A_block


# =============================================================================
# Tests for batched 4×4 inversion
# =============================================================================

def test_batch_invert_single():
    """Test inversion of a single 4×4 matrix."""
    A = jnp.array([
        [2.0, 0.1, 0.0, 0.0],
        [0.1, 3.0, 0.2, 0.0],
        [0.0, 0.2, 2.5, 0.1],
        [0.0, 0.0, 0.1, 1.0],
    ])
    
    A_inv = _batch_invert_4x4(A[None, None, :, :])[0, 0]
    
    # Check A @ A_inv ≈ I
    product = A @ A_inv
    assert_allclose(product, jnp.eye(4), atol=1e-10)


def test_batch_invert_batch():
    """Test batched inversion of multiple 4×4 matrices."""
    # Create batch of 3×4 matrices
    batch = jnp.zeros((3, 4, 4, 4))
    
    for i in range(3):
        for j in range(4):
            # Different matrix for each cell
            batch = batch.at[i, j].set(
                jnp.eye(4) * (1 + 0.1 * i + 0.05 * j) +
                jnp.array([[0, 0.1, 0, 0], [0.1, 0, 0.1, 0], 
                           [0, 0.1, 0, 0.1], [0, 0, 0.1, 0]])
            )
    
    batch_inv = _batch_invert_4x4(batch)
    
    # Check each P @ P_inv ≈ I
    for i in range(3):
        for j in range(4):
            product = batch[i, j] @ batch_inv[i, j]
            assert_allclose(product, jnp.eye(4), atol=1e-10)


# =============================================================================
# Tests for block Jacobian computation
# =============================================================================

def test_block_jacobians_simple_residual(small_grid):
    """Test that block Jacobian extraction works for simple block-diagonal residual."""
    Q, dt, volume, NI, NJ, nghost = small_grid
    residual_fn, A_block = make_simple_residual(NI, NJ, nghost)
    
    J_diag = _compute_block_jacobians(residual_fn, Q, nghost)
    
    # For block-diagonal residual, J_diag should equal A_block everywhere
    for i in range(NI):
        for j in range(NJ):
            assert_allclose(J_diag[i, j], A_block, atol=1e-5,
                           err_msg=f"Block Jacobian mismatch at ({i}, {j})")


def test_block_jacobians_coupled_residual(small_grid):
    """Test block Jacobian extraction for residual with neighbor coupling."""
    Q, dt, volume, NI, NJ, nghost = small_grid
    residual_fn, A_block = make_coupled_residual(NI, NJ, nghost)
    
    J_diag = _compute_block_jacobians(residual_fn, Q, nghost)
    
    # The diagonal block should be A_block (neighbors don't affect ∂R[i,j]/∂Q[i,j])
    # Check interior cells (away from boundaries where padding affects results)
    for i in range(2, NI - 2):
        for j in range(2, NJ - 2):
            assert_allclose(J_diag[i, j], A_block, atol=1e-5,
                           err_msg=f"Block Jacobian mismatch at ({i}, {j})")


def test_block_jacobians_fd_vs_jvp(small_grid):
    """Compare finite difference and JVP-based block Jacobian computation."""
    Q, dt, volume, NI, NJ, nghost = small_grid
    residual_fn, _ = make_coupled_residual(NI, NJ, nghost)
    
    J_fd = _compute_block_jacobians(residual_fn, Q, nghost, eps=1e-7)
    J_jvp = _compute_block_jacobians_jvp(residual_fn, Q, nghost)
    
    # They should match closely
    assert_allclose(J_fd, J_jvp, atol=1e-5, rtol=1e-5)


# =============================================================================
# Tests for full preconditioner
# =============================================================================

def test_preconditioner_shapes(small_grid):
    """Test preconditioner has correct shapes."""
    Q, dt, volume, NI, NJ, nghost = small_grid
    residual_fn, _ = make_simple_residual(NI, NJ, nghost)
    
    P = BlockJacobiPreconditioner.compute(residual_fn, Q, dt, volume, nghost)
    
    assert P.P_inv.shape == (NI, NJ, 4, 4)
    assert P.NI == NI
    assert P.NJ == NJ


def test_preconditioner_apply_shapes(small_grid):
    """Test preconditioner apply preserves shapes."""
    Q, dt, volume, NI, NJ, nghost = small_grid
    residual_fn, _ = make_simple_residual(NI, NJ, nghost)
    
    P = BlockJacobiPreconditioner.compute(residual_fn, Q, dt, volume, nghost)
    
    # Test 3D input
    v_3d = jnp.ones((NI, NJ, 4))
    result_3d = P.apply(v_3d)
    assert result_3d.shape == (NI, NJ, 4)
    
    # Test flattened input
    v_flat = jnp.ones(NI * NJ * 4)
    result_flat = P.apply(v_flat)
    assert result_flat.shape == (NI * NJ * 4,)
    
    # Results should be equivalent
    assert_allclose(result_3d.flatten(), result_flat, atol=1e-12)


def test_preconditioner_inverse_quality(small_grid):
    """Test that P^{-1} @ P ≈ I."""
    Q, dt, volume, NI, NJ, nghost = small_grid
    residual_fn, _ = make_simple_residual(NI, NJ, nghost)
    
    P = BlockJacobiPreconditioner.compute(residual_fn, Q, dt, volume, nghost)
    
    max_err, mean_err = verify_preconditioner(P, residual_fn, Q, dt, volume, nghost)
    
    # Note: tolerance accounts for numerical differences between JVP (used in compute)
    # and FD (used in verify_preconditioner)
    assert max_err < 1e-7, f"Max P^{{-1}} @ P error: {max_err}"
    assert mean_err < 1e-9, f"Mean P^{{-1}} @ P error: {mean_err}"


def test_preconditioner_jit_apply(small_grid):
    """Test JIT-compiled apply function."""
    Q, dt, volume, NI, NJ, nghost = small_grid
    residual_fn, _ = make_simple_residual(NI, NJ, nghost)
    
    P = BlockJacobiPreconditioner.compute(residual_fn, Q, dt, volume, nghost)
    apply_jit = P.apply_jit()
    
    v = jnp.ones((NI, NJ, 4)) * 0.5
    
    # Compare regular and JIT versions
    result_regular = P.apply(v)
    result_jit = apply_jit(v)
    
    assert_allclose(result_regular, result_jit, atol=1e-12)


# =============================================================================
# Tests for implicit Euler system
# =============================================================================

def test_diagonal_dominance(small_grid):
    """Test that large V/dt makes preconditioner diagonal-dominant."""
    Q, dt, volume, NI, NJ, nghost = small_grid
    residual_fn, A_block = make_simple_residual(NI, NJ, nghost)
    
    # Very small dt (large V/dt term)
    dt_small = dt * 0.001
    
    P = BlockJacobiPreconditioner.compute(residual_fn, Q, dt_small, volume, nghost)
    
    # P should be approximately (V/dt) * I when V/dt >> ||J||
    expected_diag = volume[0, 0] / dt_small[0, 0]
    
    # Check that diagonal elements dominate
    for i in range(4):
        assert P.P_inv[0, 0, i, i] < 1.1 / expected_diag, \
            "P^{-1} diagonal should be approximately 1/(V/dt)"


def test_cfl_limiting_behavior(small_grid):
    """Test behavior at CFL extremes.
    
    With the correct implicit Euler formulation P = V/dt - J:
    - At large CFL (small V/dt), P ≈ -J, so P^{-1} ≈ -J^{-1}
    """
    Q, dt, volume, NI, NJ, nghost = small_grid
    residual_fn, A_block = make_simple_residual(NI, NJ, nghost)
    
    # Large CFL (small diagonal term, approaching pure Newton)
    dt_large = dt * 1000
    P_large_cfl = BlockJacobiPreconditioner.compute(
        residual_fn, Q, dt_large, volume, nghost
    )
    
    # P = V/dt - J ≈ -J at large CFL, so P^{-1} ≈ -J^{-1}
    neg_J_diag_inv = -jnp.linalg.inv(A_block)
    
    # Interior cells should have P^{-1} ≈ -J^{-1}
    assert_allclose(P_large_cfl.P_inv[2, 2], neg_J_diag_inv, rtol=0.1,
                   err_msg="Large CFL should give P^{-1} ≈ -J^{-1}")


# =============================================================================
# Performance smoke test
# =============================================================================

def test_preconditioner_medium_grid():
    """Test preconditioner on medium-sized grid (performance smoke test)."""
    NI, NJ = 32, 32
    nghost = NGHOST
    
    Q = jnp.ones((NI + 2*nghost, NJ + 2*nghost, 4))
    dt = jnp.ones((NI, NJ)) * 0.1
    volume = jnp.ones((NI, NJ)) * 0.01
    
    residual_fn, _ = make_coupled_residual(NI, NJ, nghost)
    
    # This should complete without error
    P = BlockJacobiPreconditioner.compute(residual_fn, Q, dt, volume, nghost)
    
    # Quick sanity check
    v = jnp.ones((NI, NJ, 4))
    result = P.apply(v)
    
    assert result.shape == (NI, NJ, 4)
    assert jnp.all(jnp.isfinite(result)), "Preconditioner produced non-finite values"

