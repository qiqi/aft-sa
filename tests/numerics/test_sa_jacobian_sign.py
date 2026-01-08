"""
Test that SA destruction term Jacobian has the correct sign for implicit stability.

The SA destruction term D = cw1 * fw * (nuHat/d)^2 is a sink term:
    d(nuHat)/dt = Production - Destruction + Diffusion

For the residual R = d(nuHat)/dt:
    dR/d(nuHat) from destruction = -dD/d(nuHat) < 0

For implicit Euler stability, the system matrix (V/dt - J) should have positive diagonal:
    V/dt - J = V/dt - dR/d(nuHat) = V/dt + |dR/d(nuHat)| > 0

This test verifies that:
1. Decreasing CFL increases V/dt (more positive diagonal contribution)
2. The destruction Jacobian dR/d(nuHat) is NEGATIVE (helps stability when subtracted)
3. The combined matrix (V/dt - J) has all positive diagonal elements
"""

import pytest
import numpy as np
from pathlib import Path

from src.physics.jax_config import jax, jnp
from src.solvers.rans_solver import RANSSolver, SolverConfig
from src.numerics.preconditioner import _make_block_jacobian_jit
from src.constants import NGHOST


class TestSAJacobianSign:
    """Test SA Jacobian sign for implicit stability.
    
    Uses class-scoped fixtures to avoid redundant solver initialization
    and Jacobian computation across tests.
    """
    
    @pytest.fixture(scope="class")
    def solver_with_grid(self, naca0012_medium_grid):
        """Create a solver for testing (class-scoped for reuse)."""
        if naca0012_medium_grid is None:
            pytest.skip("Grid generation not available")
        
        grid_path = naca0012_medium_grid['path']
        config = SolverConfig(
            alpha=4.0,
            reynolds=1e6,
            solver_mode='newton',
            html_animation=False,
        )
        return RANSSolver(grid_path, config)
    
    @pytest.fixture(scope="class")
    def jacobian_data(self, solver_with_grid):
        """Compute Jacobian once and cache for all tests in this class."""
        solver = solver_with_grid
        nghost = NGHOST
        NI, NJ = solver.NI, solver.NJ
        
        compute_jacobians = _make_block_jacobian_jit(solver._jit_residual, NI, NJ, nghost)
        J_diag = compute_jacobians(solver.Q_jax)
        
        return {
            'J_diag': J_diag,
            'J_nuHat': J_diag[:, :, 3, 3],
            'NI': NI,
            'NJ': NJ,
            'nghost': nghost,
        }
    
    def test_lower_cfl_increases_v_over_dt(self, solver_with_grid):
        """Verify that lower CFL gives larger V/dt diagonal contribution."""
        solver = solver_with_grid
        
        v_over_dt_low = solver.volume_jax / solver._jit_compute_dt(solver.Q_jax, cfl=0.1)
        v_over_dt_high = solver.volume_jax / solver._jit_compute_dt(solver.Q_jax, cfl=1.0)
        
        # Lower CFL should give larger V/dt everywhere
        assert jnp.all(v_over_dt_low > v_over_dt_high), \
            "Lower CFL should increase V/dt (diagonal stabilization)"
        
        # Check the ratio
        ratio = v_over_dt_low / v_over_dt_high
        expected_ratio = 10.0  # CFL ratio is 10
        assert jnp.allclose(ratio, expected_ratio, rtol=1e-6), \
            f"V/dt should scale inversely with CFL: expected ratio {expected_ratio}"
    
    def test_destruction_jacobian_is_negative(self, solver_with_grid, jacobian_data):
        """Verify dR/d(nuHat) from destruction is negative."""
        J_nuHat = jacobian_data['J_nuHat']
        
        # Most cells should have negative Jacobian (destruction dominates)
        neg_fraction = jnp.sum(J_nuHat < 0) / J_nuHat.size
        
        assert neg_fraction > 0.5, \
            f"More than half of J[nuHat,nuHat] should be negative, got {neg_fraction:.2%}"
    
    def test_v_over_dt_minus_jacobian_is_positive(self, solver_with_grid, jacobian_data):
        """Verify (V/dt - J) has all positive diagonal for nuHat equation."""
        solver = solver_with_grid
        J_nuHat = jacobian_data['J_nuHat']
        
        # Compute V/dt at conservative CFL
        cfl = 0.1
        dt = solver._jit_compute_dt(solver.Q_jax, cfl)
        v_over_dt = solver.volume_jax / dt
        
        # The CORRECT implicit system matrix is (V/dt - J)
        # With J < 0 (from destruction), this gives V/dt + |J| > 0
        P_diag_nuHat = v_over_dt - J_nuHat
        
        min_diag = float(P_diag_nuHat.min())
        assert min_diag > 0, \
            f"All (V/dt - J) diagonals should be positive, min = {min_diag}"
    
    def test_jacobian_sign_consistent_with_fd(self, solver_with_grid, jacobian_data):
        """Verify Jacobian sign matches finite difference (single sample cell)."""
        solver = solver_with_grid
        J_diag = jacobian_data['J_diag']
        nghost = jacobian_data['nghost']
        NI, NJ = jacobian_data['NI'], jacobian_data['NJ']
        
        # Check at single cell (mid-domain) to reduce FD overhead
        i, j = NI // 2, NJ // 2
        J_jvp = float(J_diag[i, j, 3, 3])
        
        # Finite difference
        eps = 1e-7
        R_base = solver._jit_residual(solver.Q_jax)
        R_nuHat_base = float(R_base[i, j, 3])
        
        Q_pert = solver.Q_jax.at[i+nghost, j+nghost, 3].add(eps)
        R_pert = solver._jit_residual(Q_pert)
        R_nuHat_pert = float(R_pert[i, j, 3])
        
        J_fd = (R_nuHat_pert - R_nuHat_base) / eps
        
        # Signs should match
        assert np.sign(J_jvp) == np.sign(J_fd) or abs(J_fd) < 1e-6, \
            f"Jacobian sign mismatch at ({i},{j}): JVP={J_jvp:.3e}, FD={J_fd:.3e}"
