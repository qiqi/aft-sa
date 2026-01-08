"""
Tests for preconditioned RK solver mode.

Tests cover:
1. Basic preconditioner computation in solver context
2. Preconditioned RK step produces valid output
3. Comparison between rk5 and rk5_precond modes

Optimized for speed:
- Class-scoped fixtures to share solver initialization
- Tests within a class share the same solver instance
"""

import pytest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.physics.jax_config import jax, jnp
from src.solvers.rans_solver import RANSSolver, SolverConfig


class TestPreconditionerIntegration:
    """Test preconditioner integration with RANS solver.
    
    Uses class-scoped solver to share initialization between tests.
    """
    
    @pytest.fixture(scope="class")
    def precond_solver(self, naca0012_medium_grid):
        """Create preconditioned solver once for this class."""
        if naca0012_medium_grid is None:
            pytest.skip("Construct2D not available for grid generation")
        
        config = SolverConfig(
            alpha=2.0,
            reynolds=1e6,
            cfl_start=0.5,
            cfl_target=2.0,
            max_iter=5,
            solver_mode='rk5_precond',
            html_animation=False,
        )
        solver = RANSSolver(naca0012_medium_grid['path'], config)
        # Pre-compute preconditioner for all tests
        solver.compute_preconditioner()
        return solver
    
    def test_compute_preconditioner(self, precond_solver):
        """Test that preconditioner computation succeeds."""
        solver = precond_solver
        
        assert hasattr(solver, '_preconditioner')
        assert solver._preconditioner is not None
        assert solver._preconditioner.P_inv.shape == (solver.NI, solver.NJ, 4, 4)
        
        finite_fraction = float(jnp.mean(jnp.isfinite(solver._preconditioner.P_inv)))
        assert finite_fraction > 0.9, f"Only {finite_fraction:.1%} of P_inv values are finite"
    
    def test_precond_apply_function(self, precond_solver):
        """Test that preconditioner apply function works."""
        solver = precond_solver
        
        v = jnp.ones((solver.NI, solver.NJ, 4))
        result = solver._precond_apply(v)
        
        assert result.shape == v.shape
        finite_fraction = float(jnp.mean(jnp.isfinite(result)))
        assert finite_fraction > 0.9


class TestPreconditionedRKStep:
    """Test preconditioned RK stepping.
    
    Uses class-scoped solver to share initialization.
    """
    
    @pytest.fixture(scope="class")
    def step_solver(self, naca0012_medium_grid):
        """Create solver for stepping tests."""
        if naca0012_medium_grid is None:
            pytest.skip("Construct2D not available for grid generation")
        
        config = SolverConfig(
            alpha=2.0,
            reynolds=1e6,
            cfl_start=0.1,
            cfl_target=0.5,
            cfl_ramp_iters=50,
            max_iter=30,
            solver_mode='rk5_precond',
            html_animation=False,
        )
        return RANSSolver(naca0012_medium_grid['path'], config)
    
    def test_step_produces_valid_state(self, step_solver):
        """Test that preconditioned RK step produces valid state."""
        solver = step_solver
        
        # Run a few steps
        for _ in range(3):
            solver.step()
        
        Q = solver.Q_jax
        finite_fraction = float(jnp.mean(jnp.isfinite(Q)))
        assert finite_fraction > 0.99, f"Only {finite_fraction:.1%} of Q values are finite"
        
        from src.constants import NGHOST
        nuHat = Q[NGHOST:-NGHOST, NGHOST:-NGHOST, 3]
        assert jnp.all(nuHat >= -1e-10), "nuHat should be non-negative"
    
    def test_residual_decreases(self, step_solver):
        """Test that residual decreases with preconditioned RK."""
        solver = step_solver
        
        # Continue from previous test's state
        residuals = []
        for _ in range(15):
            solver.step()
            res = solver.get_residual_l1_scaled()
            residuals.append(max(res))
        
        valid_residuals = [r for r in residuals if np.isfinite(r) and r < 1e10]
        
        assert len(valid_residuals) >= 10, \
            f"Preconditioned RK unstable: only {len(valid_residuals)} valid residuals"
        
        if len(valid_residuals) >= 5:
            mostly_decreasing = sum(1 for i in range(len(valid_residuals)-1) 
                                   if valid_residuals[i+1] <= valid_residuals[i] * 1.1)
            assert mostly_decreasing >= len(valid_residuals) // 2, \
                "Residual should be mostly decreasing or stable"


class TestModeComparison:
    """Compare rk5 and rk5_precond modes.
    
    Uses class-scoped fixtures to avoid redundant solver initialization.
    """
    
    @pytest.fixture(scope="class")
    def grid_path(self, naca0012_medium_grid):
        """Get grid path."""
        if naca0012_medium_grid is None:
            pytest.skip("Construct2D not available for grid generation")
        return naca0012_medium_grid['path']
    
    @pytest.fixture(scope="class")
    def solver_rk5(self, grid_path):
        """RK5 solver (class-scoped)."""
        config = SolverConfig(
            alpha=2.0,
            reynolds=1e6,
            cfl_start=0.5,
            cfl_target=2.0,
            max_iter=10,
            solver_mode='rk5',
            html_animation=False,
        )
        return RANSSolver(grid_path, config)
    
    @pytest.fixture(scope="class")
    def solver_precond(self, grid_path):
        """Preconditioned RK5 solver (class-scoped)."""
        config = SolverConfig(
            alpha=2.0,
            reynolds=1e6,
            cfl_start=0.5,
            cfl_target=2.0,
            max_iter=10,
            solver_mode='rk5_precond',
            html_animation=False,
        )
        return RANSSolver(grid_path, config)
    
    @pytest.fixture(scope="class")
    def solver_newton(self, grid_path):
        """Newton solver (class-scoped)."""
        config = SolverConfig(
            alpha=2.0,
            reynolds=1e6,
            cfl_start=0.1,
            cfl_target=1.0,
            solver_mode='newton',
            gmres_restart=10,
            gmres_tol=1e-2,
            html_animation=False,
        )
        return RANSSolver(grid_path, config)
    
    def test_both_modes_run(self, solver_rk5, solver_precond):
        """Test that both modes complete without error."""
        for _ in range(5):
            solver_rk5.step()
        for _ in range(5):
            solver_precond.step()
        
        rk5_finite = float(jnp.mean(jnp.isfinite(solver_rk5.Q_jax)))
        precond_finite = float(jnp.mean(jnp.isfinite(solver_precond.Q_jax)))
        
        assert rk5_finite > 0.99, f"rk5 mode: {rk5_finite:.1%} finite"
        assert precond_finite > 0.99, f"rk5_precond mode: {precond_finite:.1%} finite"
    
    def test_newton_mode_runs(self, solver_newton):
        """Test that newton mode runs without crashing."""
        solver_newton.step()
        
        finite_frac = float(jnp.mean(jnp.isfinite(solver_newton.Q_jax)))
        assert finite_frac > 0.95
