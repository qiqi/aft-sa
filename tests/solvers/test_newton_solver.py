"""
Tests for Newton-GMRES solver mode.

Tests cover:
1. Newton step runs without error
2. Newton update with Patankar scheme
3. Exponential CFL ramping for Newton
4. Comparison with RK modes

Optimized for speed:
- CFL tests share a single solver (class-scoped)
- Mode comparison reuses grid/metrics
"""

import pytest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.physics.jax_config import jax, jnp
from src.solvers.rans_solver import RANSSolver, SolverConfig


class TestNewtonCFLRamping:
    """Test CFL ramping behavior for Newton mode.
    
    Uses class-scoped solver fixture since CFL methods only depend on config,
    not solver state. All 4 tests share the same solver.
    """
    
    @pytest.fixture(scope="class")
    def solver_newton(self, naca0012_medium_grid):
        """Create Newton solver once for all CFL tests."""
        if naca0012_medium_grid is None:
            pytest.skip("Construct2D not available for grid generation")
        
        config = SolverConfig(
            alpha=0.0,
            reynolds=1e6,
            cfl_start=0.1,
            cfl_target=1e12,  # Effectively infinity (for exponential test)
            cfl_ramp_iters=100,  # Iterations per decade
            solver_mode='newton',
            html_animation=False,
        )
        return RANSSolver(naca0012_medium_grid['path'], config)
    
    @pytest.fixture(scope="class")
    def solver_rk(self, naca0012_medium_grid):
        """Create RK solver once for linear CFL test."""
        if naca0012_medium_grid is None:
            pytest.skip("Construct2D not available for grid generation")
        
        config = SolverConfig(
            alpha=0.0,
            reynolds=1e6,
            cfl_start=0.5,
            cfl_target=5.0,
            cfl_ramp_iters=100,
            solver_mode='rk5',
            html_animation=False,
        )
        return RANSSolver(naca0012_medium_grid['path'], config)
    
    def test_exponential_cfl_ramping(self, solver_newton):
        """Test that Newton mode uses exponential CFL ramping."""
        solver = solver_newton
        
        cfl_0 = solver._get_cfl_newton(0)
        cfl_50 = solver._get_cfl_newton(50)
        cfl_100 = solver._get_cfl_newton(100)
        cfl_200 = solver._get_cfl_newton(200)
        
        assert abs(cfl_0 - 0.1) < 0.01
        assert abs(cfl_50 - 0.1 * 10**0.5) < 0.05
        assert abs(cfl_100 - 1.0) < 0.1
        assert abs(cfl_200 - 10.0) < 1.0
    
    def test_cfl_capped_at_final(self, solver_newton):
        """Test that CFL is capped at the final value."""
        # Modify config temporarily to test capping
        original_target = solver_newton.config.cfl_target
        original_ramp = solver_newton.config.cfl_ramp_iters
        
        solver_newton.config.cfl_target = 100.0
        solver_newton.config.cfl_ramp_iters = 50
        
        cfl_1000 = solver_newton._get_cfl_newton(1000)
        assert cfl_1000 == 100.0, f"CFL should be capped at 100, got {cfl_1000}"
        
        # Restore
        solver_newton.config.cfl_target = original_target
        solver_newton.config.cfl_ramp_iters = original_ramp
    
    def test_rk_cfl_uses_linear_ramp(self, solver_rk):
        """Test that RK mode uses linear ramping."""
        solver = solver_rk
        
        cfl_50 = solver._get_cfl(50)
        expected = 0.5 + 0.5 * (5.0 - 0.5)
        assert abs(cfl_50 - expected) < 0.01, f"Expected {expected}, got {cfl_50}"
        
        cfl_100 = solver._get_cfl(100)
        assert cfl_100 == 5.0, f"Expected 5.0, got {cfl_100}"
    
    def test_newton_cfl_exact_decade_values(self, solver_newton):
        """Test exact CFL values at decade boundaries."""
        solver = solver_newton
        
        # Ensure we're using the original config
        solver.config.cfl_start = 0.1
        solver.config.cfl_target = 1e12
        solver.config.cfl_ramp_iters = 100
        
        cfl_0 = solver._get_cfl_newton(0)
        cfl_100 = solver._get_cfl_newton(100)
        cfl_200 = solver._get_cfl_newton(200)
        cfl_300 = solver._get_cfl_newton(300)
        
        assert abs(cfl_0 - 0.1) < 1e-10
        assert abs(cfl_100 - 1.0) < 1e-10
        assert abs(cfl_200 - 10.0) < 1e-9
        assert abs(cfl_300 - 100.0) < 1e-8
        
        cfl_10 = solver._get_cfl_newton(10)
        expected_10 = 0.1 * (10.0 ** 0.1)
        assert abs(cfl_10 - expected_10) < 1e-10


class TestNewtonUpdate:
    """Test Newton update with Patankar scheme (no solver needed)."""
    
    def test_patankar_preserves_positivity(self):
        """Test that Patankar scheme keeps nuHat >= 0."""
        Q_int = jnp.array([
            [[1.0, 0.5, 0.1, 0.01]],
            [[1.0, 0.5, 0.1, 0.1]],
        ])
        
        dQ = jnp.array([
            [[0.0, 0.0, 0.0, -0.02]],
            [[0.0, 0.0, 0.0, -0.2]],
        ])
        
        Q_new = RANSSolver._apply_newton_update_static(Q_int, dQ)
        
        assert jnp.all(Q_new[:, :, 3] >= 0), f"nuHat negative: {Q_new[:, :, 3]}"
        
        np.testing.assert_allclose(Q_new[:, :, 0], Q_int[:, :, 0] + dQ[:, :, 0])
        np.testing.assert_allclose(Q_new[:, :, 1], Q_int[:, :, 1] + dQ[:, :, 1])
        np.testing.assert_allclose(Q_new[:, :, 2], Q_int[:, :, 2] + dQ[:, :, 2])


class TestNewtonStep:
    """Test Newton-GMRES stepping.
    
    Uses class-scoped solver to share initialization cost.
    """
    
    @pytest.fixture(scope="class")
    def newton_solver(self, naca0012_medium_grid):
        """Create Newton solver for stepping tests."""
        if naca0012_medium_grid is None:
            pytest.skip("Construct2D not available for grid generation")
        
        config = SolverConfig(
            alpha=0.0,
            reynolds=1e6,
            cfl_start=0.1,
            cfl_target=1.0,
            cfl_ramp_iters=10,
            max_iter=20,
            solver_mode='newton',
            gmres_restart=10,
            gmres_tol=1e-2,
            html_animation=False,
        )
        return RANSSolver(naca0012_medium_grid['path'], config)
    
    def test_newton_step_runs(self, newton_solver):
        """Test that Newton step completes without error."""
        solver = newton_solver
        
        # Run a few Newton steps
        for _ in range(3):
            solver.step()
        
        Q = solver.Q_jax
        finite_fraction = float(jnp.mean(jnp.isfinite(Q)))
        assert finite_fraction > 0.95, f"Only {finite_fraction:.1%} of Q is finite"
        
        from src.constants import NGHOST
        nuHat = Q[NGHOST:-NGHOST, NGHOST:-NGHOST, 3]
        assert jnp.all(nuHat >= -1e-10), "nuHat should be non-negative"
    
    def test_newton_residual_finite(self, newton_solver):
        """Test that Newton produces finite residuals."""
        solver = newton_solver
        
        # Continue from previous state
        residuals = []
        for _ in range(5):
            solver.step()
            res = solver.get_residual_l1_scaled()
            residuals.append(max(res))
        
        valid_residuals = [r for r in residuals if np.isfinite(r)]
        assert len(valid_residuals) >= 3, f"Too few finite residuals: {residuals}"


class TestModeComparison:
    """Compare Newton with RK modes.
    
    Uses class-scoped solvers to share grid/metrics initialization.
    """
    
    @pytest.fixture(scope="class")
    def grid_path(self, naca0012_medium_grid):
        """Get grid path (session-scoped dependency)."""
        if naca0012_medium_grid is None:
            pytest.skip("Construct2D not available for grid generation")
        return naca0012_medium_grid['path']
    
    @pytest.fixture(scope="class")  
    def base_config(self):
        """Base config shared by all modes."""
        return dict(
            alpha=0.0,
            reynolds=1e6,
            cfl_start=0.1,
            cfl_target=1.0,
            max_iter=5,
            html_animation=False,
        )
    
    @pytest.fixture(scope="class")
    def solver_rk5(self, grid_path, base_config):
        """RK5 solver (class-scoped)."""
        config = SolverConfig(**base_config, solver_mode='rk5')
        return RANSSolver(grid_path, config)
    
    @pytest.fixture(scope="class")
    def solver_precond(self, grid_path, base_config):
        """Preconditioned RK5 solver (class-scoped)."""
        config = SolverConfig(**base_config, solver_mode='rk5_precond')
        return RANSSolver(grid_path, config)
    
    @pytest.fixture(scope="class")
    def solver_newton(self, grid_path, base_config):
        """Newton solver (class-scoped)."""
        config = SolverConfig(**base_config, solver_mode='newton', 
                              gmres_restart=10, gmres_tol=1e-2)
        return RANSSolver(grid_path, config)
    
    def test_all_three_modes_run(self, solver_rk5, solver_precond, solver_newton):
        """Test that all three solver modes complete successfully."""
        # Run 3 steps for each mode
        for _ in range(3):
            solver_rk5.step()
        for _ in range(3):
            solver_precond.step()
        for _ in range(3):
            solver_newton.step()
        
        # All should produce mostly finite states
        for name, solver in [('rk5', solver_rk5), 
                              ('rk5_precond', solver_precond),
                              ('newton', solver_newton)]:
            finite = float(jnp.mean(jnp.isfinite(solver.Q_jax)))
            assert finite > 0.95, f"{name} mode: {finite:.1%} finite"
