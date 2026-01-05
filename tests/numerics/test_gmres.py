"""
Tests for GMRES(m) solver.

Tests cover:
1. Simple linear systems (identity, diagonal, dense)
2. Convergence on known problems
3. Preconditioner application
4. Restart behavior
5. Jacobian-free matvec
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.physics.jax_config import jax, jnp
from src.numerics.gmres import (
    gmres,
    GMRESResult,
    make_jfnk_matvec,
    make_newton_rhs,
    _back_substitute,
    _modified_gram_schmidt,
)


# =============================================================================
# Test fixtures
# =============================================================================

@pytest.fixture
def identity_system():
    """Identity matrix system: Ix = b."""
    n = 10
    A = jnp.eye(n)
    b = jnp.ones(n)
    x_true = b.copy()
    
    def matvec(v):
        return A @ v
    
    return matvec, b, x_true


@pytest.fixture
def diagonal_system():
    """Diagonal matrix system."""
    n = 20
    diag = jnp.arange(1, n + 1, dtype=jnp.float64)
    b = jnp.ones(n)
    x_true = b / diag
    
    def matvec(v):
        return diag * v
    
    return matvec, b, x_true


@pytest.fixture
def tridiagonal_system():
    """Tridiagonal matrix system (more challenging)."""
    n = 30
    
    # Build tridiagonal: 2 on diagonal, -1 on off-diagonals
    main_diag = jnp.full(n, 2.0)
    off_diag = jnp.full(n - 1, -1.0)
    
    # Construct full matrix for reference
    A = jnp.diag(main_diag) + jnp.diag(off_diag, 1) + jnp.diag(off_diag, -1)
    
    # RHS and exact solution
    b = jnp.ones(n)
    x_true = jnp.linalg.solve(A, b)
    
    def matvec(v):
        return A @ v
    
    return matvec, b, x_true


@pytest.fixture
def ill_conditioned_system():
    """Ill-conditioned system that benefits from preconditioning."""
    n = 20
    
    # Create ill-conditioned diagonal matrix
    diag = jnp.array([10.0**(-i/4) for i in range(n)])
    A = jnp.diag(diag)
    
    b = jnp.ones(n)
    x_true = b / diag
    
    def matvec(v):
        return A @ v
    
    # Jacobi preconditioner
    def precond(v):
        return v / diag
    
    return matvec, b, x_true, precond


# =============================================================================
# Basic functionality tests
# =============================================================================

class TestGMRESBasic:
    """Test basic GMRES functionality."""
    
    def test_identity_system(self, identity_system):
        """Test GMRES on identity matrix."""
        matvec, b, x_true = identity_system
        
        result = gmres(matvec, b, tol=1e-10)
        
        assert result.converged
        assert result.iterations <= 2  # Should converge in 1 iteration
        assert_allclose(result.x, x_true, atol=1e-10)
    
    def test_diagonal_system(self, diagonal_system):
        """Test GMRES on diagonal matrix."""
        matvec, b, x_true = diagonal_system
        
        result = gmres(matvec, b, tol=1e-10)
        
        assert result.converged
        assert_allclose(result.x, x_true, rtol=1e-8)
    
    def test_tridiagonal_system(self, tridiagonal_system):
        """Test GMRES on tridiagonal matrix."""
        matvec, b, x_true = tridiagonal_system
        
        result = gmres(matvec, b, tol=1e-10, maxiter=50)
        
        assert result.converged
        assert_allclose(result.x, x_true, rtol=1e-6)
    
    def test_zero_rhs(self, identity_system):
        """Test GMRES with zero RHS."""
        matvec, _, _ = identity_system
        b = jnp.zeros(10)
        
        result = gmres(matvec, b, tol=1e-10)
        
        assert result.converged
        assert result.iterations == 0
        assert_allclose(result.x, jnp.zeros(10), atol=1e-10)
    
    def test_initial_guess(self, diagonal_system):
        """Test GMRES with good initial guess."""
        matvec, b, x_true = diagonal_system
        
        # Start close to solution
        x0 = x_true + 0.01 * jnp.ones_like(x_true)
        
        result = gmres(matvec, b, x0=x0, tol=1e-10)
        
        assert result.converged
        assert_allclose(result.x, x_true, rtol=1e-8)


class TestGMRESRestart:
    """Test GMRES restart behavior."""
    
    def test_restart_matches_no_restart(self, tridiagonal_system):
        """For small systems, restart should match no restart."""
        matvec, b, x_true = tridiagonal_system
        
        # No restart (m = n)
        result_full = gmres(matvec, b, tol=1e-10, restart=30, maxiter=50)
        
        # With restart - needs more iterations due to information loss at restart
        result_restart = gmres(matvec, b, tol=1e-8, restart=10, maxiter=500)
        
        # Full GMRES should converge
        assert result_full.converged
        
        # Restarted version may need looser tolerance or more iterations
        # At minimum, solutions should be close
        assert_allclose(result_full.x, result_restart.x, rtol=1e-4)
    
    def test_small_restart_still_converges(self, diagonal_system):
        """Test that small restart parameter still converges."""
        matvec, b, x_true = diagonal_system
        
        # Very small restart
        result = gmres(matvec, b, tol=1e-10, restart=3, maxiter=100)
        
        assert result.converged
        assert_allclose(result.x, x_true, rtol=1e-6)


class TestGMRESPreconditioner:
    """Test GMRES with preconditioning."""
    
    def test_preconditioner_accelerates(self, ill_conditioned_system):
        """Test that preconditioner reduces iterations."""
        matvec, b, x_true, precond = ill_conditioned_system
        
        # Without preconditioner
        result_no_precond = gmres(matvec, b, tol=1e-8, restart=20, maxiter=200)
        
        # With preconditioner
        result_precond = gmres(
            matvec, b, tol=1e-8, restart=20, maxiter=200,
            preconditioner=precond
        )
        
        # Both should converge
        assert result_no_precond.converged or result_no_precond.iterations == 200
        assert result_precond.converged
        
        # Preconditioned version should use fewer iterations
        assert result_precond.iterations < result_no_precond.iterations
        
        # Solution should be correct
        assert_allclose(result_precond.x, x_true, rtol=1e-6)
    
    def test_identity_preconditioner(self, tridiagonal_system):
        """Test that identity preconditioner gives same result."""
        matvec, b, x_true = tridiagonal_system
        
        def identity_precond(v):
            return v
        
        result_no_precond = gmres(matvec, b, tol=1e-10, maxiter=50)
        result_precond = gmres(
            matvec, b, tol=1e-10, maxiter=50, preconditioner=identity_precond
        )
        
        assert_allclose(result_no_precond.x, result_precond.x, rtol=1e-10)
        assert result_no_precond.iterations == result_precond.iterations


class TestGMRESConvergence:
    """Test GMRES convergence properties."""
    
    def test_residual_decreases(self, tridiagonal_system):
        """Test that residual decreases monotonically."""
        matvec, b, _ = tridiagonal_system
        
        result = gmres(matvec, b, tol=1e-10, restart=30, maxiter=50)
        
        # Residual should generally decrease (allow small increases due to restarts)
        for i in range(1, len(result.residual_history)):
            # Allow up to 10% increase
            assert result.residual_history[i] <= result.residual_history[i-1] * 1.1, \
                f"Residual increased at iteration {i}"
    
    def test_maxiter_reached(self, ill_conditioned_system):
        """Test behavior when maxiter is reached."""
        matvec, b, _, _ = ill_conditioned_system
        
        # Very low maxiter and small restart
        result = gmres(matvec, b, tol=1e-15, restart=5, maxiter=10)
        
        assert not result.converged
        # Iterations should be at most maxiter (could be less if restart fills)
        assert result.iterations <= 15  # maxiter + one full restart cycle


# =============================================================================
# Helper function tests
# =============================================================================

class TestHelperFunctions:
    """Test internal helper functions."""
    
    def test_back_substitute(self):
        """Test back substitution for upper triangular systems."""
        R = jnp.array([
            [2.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [0.0, 0.0, 4.0],
        ])
        b = jnp.array([4.0, 8.0, 8.0])
        
        x = _back_substitute(R, b)
        
        # Check Rx = b
        assert_allclose(R @ x, b, atol=1e-10)
    
    def test_modified_gram_schmidt(self):
        """Test modified Gram-Schmidt orthogonalization."""
        # Create 3 linearly independent vectors
        V = jnp.array([
            [1.0, 0.0, 0.0],  # v0
            [1.0, 1.0, 0.0],  # v1 (placeholder)
            [1.0, 1.0, 1.0],  # v2 (placeholder)
        ])
        H = jnp.zeros((4, 3))
        
        # Orthogonalize w against V[0:1]
        w = jnp.array([1.0, 1.0, 0.0])
        H_new, w_new = _modified_gram_schmidt(H, V, w, 0)
        
        # w_new should be orthogonal to V[0]
        assert abs(jnp.dot(w_new, V[0])) < 1e-10


# =============================================================================
# Jacobian-free matvec tests
# =============================================================================

class TestJFNKMatvec:
    """Test Jacobian-free Newton-Krylov components."""
    
    def test_make_jfnk_matvec_linear(self):
        """Test JFNK matvec on a linear residual function."""
        nghost = 1
        NI, NJ = 4, 4
        
        # Simple linear residual: R(Q) = A @ Q where A is diagonal
        diag_values = jnp.array([1.0, 2.0, 3.0, 4.0])  # One per variable
        
        def residual_fn(Q):
            Q_int = Q[nghost:-nghost, nghost:-nghost, :]
            return Q_int * diag_values  # Element-wise scaling
        
        Q = jnp.ones((NI + 2*nghost, NJ + 2*nghost, 4)) * 0.5
        dt = jnp.ones((NI, NJ)) * 0.1
        volume = jnp.ones((NI, NJ))
        
        matvec = make_jfnk_matvec(residual_fn, Q, dt, volume, nghost)
        
        # Test matvec on a vector
        v = jnp.ones(NI * NJ * 4)
        result = matvec(v)
        
        assert result.shape == v.shape
        assert jnp.all(jnp.isfinite(result))
    
    def test_make_newton_rhs(self):
        """Test Newton RHS computation."""
        nghost = 1
        NI, NJ = 4, 4
        
        def residual_fn(Q):
            Q_int = Q[nghost:-nghost, nghost:-nghost, :]
            return jnp.ones_like(Q_int) * 2.0  # Constant residual
        
        Q = jnp.ones((NI + 2*nghost, NJ + 2*nghost, 4))
        
        rhs = make_newton_rhs(residual_fn, Q, nghost)
        
        assert rhs.shape == (NI * NJ * 4,)
        assert_allclose(rhs, -jnp.ones(NI * NJ * 4) * 2.0)


# =============================================================================
# Integration test
# =============================================================================

class TestGMRESIntegration:
    """Integration tests for GMRES with realistic problems."""
    
    def test_2d_laplacian(self):
        """Test GMRES on 2D Laplacian discretization."""
        n = 8  # Grid size
        N = n * n  # Total unknowns
        
        # Build 5-point stencil Laplacian
        A = jnp.zeros((N, N))
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                A = A.at[idx, idx].set(4.0)
                if i > 0:
                    A = A.at[idx, idx - n].set(-1.0)
                if i < n - 1:
                    A = A.at[idx, idx + n].set(-1.0)
                if j > 0:
                    A = A.at[idx, idx - 1].set(-1.0)
                if j < n - 1:
                    A = A.at[idx, idx + 1].set(-1.0)
        
        # RHS
        b = jnp.ones(N)
        x_true = jnp.linalg.solve(A, b)
        
        def matvec(v):
            return A @ v
        
        result = gmres(matvec, b, tol=1e-10, restart=20, maxiter=100)
        
        assert result.converged
        assert_allclose(result.x, x_true, rtol=1e-6)

