"""
Tests for Newton-GMRES solver mode.

Tests cover:
1. Newton step runs without error
2. Newton update with Patankar scheme
3. Exponential CFL ramping for Newton
4. Comparison with RK modes
"""

import pytest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.physics.jax_config import jax, jnp
from src.solvers.rans_solver import RANSSolver, SolverConfig


@pytest.fixture
def valid_grid_path(naca0012_medium_grid):
    """Return path to test grid using the session fixture."""
    if naca0012_medium_grid is None:
        pytest.skip("Construct2D not available for grid generation")
    return naca0012_medium_grid['path']


class TestNewtonCFLRamping:
    """Test CFL ramping behavior for Newton mode."""
    
    def test_exponential_cfl_ramping(self, valid_grid_path):
        """Test that Newton mode uses exponential CFL ramping.
        
        CFL grows by a factor of 10 every ramp_iter iterations.
        """
        config = SolverConfig(
            alpha=0.0,
            reynolds=1e6,
            cfl_start=0.1,
            cfl_target=1e12,  # Effectively infinity
            cfl_ramp_iters=100,  # Iterations per decade
            solver_mode='newton',
            html_animation=False,
        )
        
        solver = RANSSolver(valid_grid_path, config)
        
        # Check CFL values at different iterations
        cfl_0 = solver._get_cfl_newton(0)
        cfl_50 = solver._get_cfl_newton(50)
        cfl_100 = solver._get_cfl_newton(100)
        cfl_200 = solver._get_cfl_newton(200)
        
        # Should start at cfl_start
        assert abs(cfl_0 - 0.1) < 0.01
        
        # At 50 iters (half decade): CFL = 0.1 * 10^0.5 ≈ 0.316
        assert abs(cfl_50 - 0.1 * 10**0.5) < 0.05
        
        # At 100 iters (1 decade): CFL = 0.1 * 10^1 = 1.0
        assert abs(cfl_100 - 1.0) < 0.1
        
        # At 200 iters (2 decades): CFL = 0.1 * 10^2 = 10.0
        assert abs(cfl_200 - 10.0) < 1.0
    
    def test_cfl_capped_at_final(self, valid_grid_path):
        """Test that CFL is capped at the final value."""
        config = SolverConfig(
            alpha=0.0,
            reynolds=1e6,
            cfl_start=0.1,
            cfl_target=100.0,  # Set a reasonable cap
            cfl_ramp_iters=50,  # Fast ramping
            solver_mode='newton',
            html_animation=False,
        )
        
        solver = RANSSolver(valid_grid_path, config)
        
        # After many iterations, CFL should be capped at final
        cfl_1000 = solver._get_cfl_newton(1000)
        assert cfl_1000 == 100.0, f"CFL should be capped at 100, got {cfl_1000}"
    
    def test_rk_cfl_uses_linear_ramp(self, valid_grid_path):
        """Test that RK mode uses linear ramping."""
        config = SolverConfig(
            alpha=0.0,
            reynolds=1e6,
            cfl_start=0.5,
            cfl_target=5.0,
            cfl_ramp_iters=100,  # Linear ramp over 100 iterations
            solver_mode='rk5',
            html_animation=False,
        )
        
        solver = RANSSolver(valid_grid_path, config)
        
        # At iteration 50 (halfway): should be 0.5 + 0.5*(5.0-0.5) = 2.75
        cfl_50 = solver._get_cfl(50)
        expected = 0.5 + 0.5 * (5.0 - 0.5)
        assert abs(cfl_50 - expected) < 0.01, f"Expected {expected}, got {cfl_50}"
        
        # At iteration 100+: should be at target
        cfl_100 = solver._get_cfl(100)
        assert cfl_100 == 5.0, f"Expected 5.0, got {cfl_100}"
    
    def test_newton_cfl_exact_decade_values(self, valid_grid_path):
        """Test exact CFL values at decade boundaries.
        
        With cfl_start=0.1 and ramp_iters=100 (iterations per decade):
        - Iteration 0: CFL = 0.1
        - Iteration 100: CFL = 1.0 (first decade)
        - Iteration 200: CFL = 10.0 (second decade)
        - Iteration 300: CFL = 100.0 (third decade)
        """
        config = SolverConfig(
            alpha=0.0,
            reynolds=1e6,
            cfl_start=0.1,
            cfl_target=1e12,
            cfl_ramp_iters=100,  # 100 iterations per decade
            solver_mode='newton',
            html_animation=False,
        )
        
        solver = RANSSolver(valid_grid_path, config)
        
        # Test exact decade boundaries
        cfl_0 = solver._get_cfl_newton(0)
        cfl_100 = solver._get_cfl_newton(100)
        cfl_200 = solver._get_cfl_newton(200)
        cfl_300 = solver._get_cfl_newton(300)
        
        assert abs(cfl_0 - 0.1) < 1e-10, f"At iter 0: expected 0.1, got {cfl_0}"
        assert abs(cfl_100 - 1.0) < 1e-10, f"At iter 100: expected 1.0, got {cfl_100}"
        assert abs(cfl_200 - 10.0) < 1e-9, f"At iter 200: expected 10.0, got {cfl_200}"
        assert abs(cfl_300 - 100.0) < 1e-8, f"At iter 300: expected 100.0, got {cfl_300}"
        
        # Also verify intermediate values are smooth exponential
        # At iteration 10: CFL = 0.1 * 10^(10/100) ≈ 0.126
        cfl_10 = solver._get_cfl_newton(10)
        expected_10 = 0.1 * (10.0 ** 0.1)
        assert abs(cfl_10 - expected_10) < 1e-10, f"At iter 10: expected {expected_10}, got {cfl_10}"


class TestNewtonUpdate:
    """Test Newton update with Patankar scheme."""
    
    def test_patankar_preserves_positivity(self):
        """Test that Patankar scheme keeps nuHat >= 0."""
        # Create test data
        Q_int = jnp.array([
            [[1.0, 0.5, 0.1, 0.01]],  # Small positive nuHat
            [[1.0, 0.5, 0.1, 0.1]],   # Larger positive nuHat
        ])
        
        # Large negative update that would make nuHat negative without Patankar
        dQ = jnp.array([
            [[0.0, 0.0, 0.0, -0.02]],  # Would give -0.01 explicitly
            [[0.0, 0.0, 0.0, -0.2]],   # Would give -0.1 explicitly
        ])
        
        Q_new = RANSSolver._apply_newton_update_static(Q_int, dQ)
        
        # nuHat should be >= 0
        assert jnp.all(Q_new[:, :, 3] >= 0), f"nuHat negative: {Q_new[:, :, 3]}"
        
        # Other variables should be updated normally
        np.testing.assert_allclose(Q_new[:, :, 0], Q_int[:, :, 0] + dQ[:, :, 0])
        np.testing.assert_allclose(Q_new[:, :, 1], Q_int[:, :, 1] + dQ[:, :, 1])
        np.testing.assert_allclose(Q_new[:, :, 2], Q_int[:, :, 2] + dQ[:, :, 2])


class TestNewtonStep:
    """Test Newton-GMRES stepping."""
    
    def test_newton_step_runs(self, valid_grid_path):
        """Test that Newton step completes without error."""
        config = SolverConfig(
            alpha=0.0,  # Symmetric case is easier
            reynolds=1e6,
            cfl_start=0.1,
            cfl_target=1.0,
            cfl_ramp_iters=10,
            max_iter=5,
            solver_mode='newton',
            gmres_restart=10,
            gmres_tol=1e-2,  # Loose tolerance for quick test
            html_animation=False,
        )
        
        solver = RANSSolver(valid_grid_path, config)
        
        # Run a few Newton steps
        for _ in range(3):
            solver.step()
        
        # Check that state is mostly valid
        Q = solver.Q_jax
        finite_fraction = float(jnp.mean(jnp.isfinite(Q)))
        assert finite_fraction > 0.95, f"Only {finite_fraction:.1%} of Q is finite"
        
        # Check nuHat >= 0
        from src.constants import NGHOST
        nuHat = Q[NGHOST:-NGHOST, NGHOST:-NGHOST, 3]
        assert jnp.all(nuHat >= -1e-10), "nuHat should be non-negative"
    
    def test_newton_residual_finite(self, valid_grid_path):
        """Test that Newton produces finite residuals."""
        config = SolverConfig(
            alpha=2.0,
            reynolds=1e6,
            cfl_start=0.1,
            cfl_target=1.0,
            cfl_ramp_iters=20,
            max_iter=10,
            solver_mode='newton',
            gmres_restart=15,
            gmres_tol=1e-2,
            html_animation=False,
        )
        
        solver = RANSSolver(valid_grid_path, config)
        
        # Run iterations
        residuals = []
        for _ in range(5):
            solver.step()
            res = solver.get_residual_l1_scaled()
            residuals.append(max(res))
        
        # Check that residuals are finite
        valid_residuals = [r for r in residuals if np.isfinite(r)]
        assert len(valid_residuals) >= 3, f"Too few finite residuals: {residuals}"


class TestModeComparison:
    """Compare Newton with RK modes."""
    
    def test_all_three_modes_run(self, valid_grid_path):
        """Test that all three solver modes complete successfully."""
        base_config = dict(
            alpha=0.0,
            reynolds=1e6,
            cfl_start=0.1,
            cfl_target=1.0,
            max_iter=5,
            html_animation=False,
        )
        
        # Standard RK5
        config_rk5 = SolverConfig(**base_config, solver_mode='rk5')
        solver_rk5 = RANSSolver(valid_grid_path, config_rk5)
        for _ in range(3):
            solver_rk5.step()
        
        # Preconditioned RK5
        config_precond = SolverConfig(**base_config, solver_mode='rk5_precond')
        solver_precond = RANSSolver(valid_grid_path, config_precond)
        for _ in range(3):
            solver_precond.step()
        
        # Newton-GMRES
        config_newton = SolverConfig(
            **base_config, 
            solver_mode='newton',
            gmres_restart=10,
            gmres_tol=1e-2,
        )
        solver_newton = RANSSolver(valid_grid_path, config_newton)
        for _ in range(3):
            solver_newton.step()
        
        # All should produce mostly finite states
        for name, solver in [('rk5', solver_rk5), 
                              ('rk5_precond', solver_precond),
                              ('newton', solver_newton)]:
            finite = float(jnp.mean(jnp.isfinite(solver.Q_jax)))
            assert finite > 0.95, f"{name} mode: {finite:.1%} finite"

