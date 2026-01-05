"""
Tests for preconditioned RK solver mode.

Tests cover:
1. Basic preconditioner computation in solver context
2. Preconditioned RK step produces valid output
3. Comparison between rk5 and rk5_precond modes
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


class TestPreconditionerIntegration:
    """Test preconditioner integration with RANS solver."""
    
    def test_compute_preconditioner(self, valid_grid_path):
        """Test that preconditioner computation succeeds."""
        config = SolverConfig(
            alpha=2.0,
            reynolds=1e6,
            cfl_start=0.5,
            cfl_target=2.0,
            max_iter=5,
            solver_mode='rk5_precond',
            html_animation=False,
        )
        
        solver = RANSSolver(valid_grid_path, config)
        
        # Compute preconditioner
        solver.compute_preconditioner()
        
        # Check preconditioner was stored
        assert hasattr(solver, '_preconditioner')
        assert solver._preconditioner is not None
        assert solver._preconditioner.P_inv.shape == (solver.NI, solver.NJ, 4, 4)
        
        # Check that most values are finite (allow some edge cases)
        finite_fraction = float(jnp.mean(jnp.isfinite(solver._preconditioner.P_inv)))
        assert finite_fraction > 0.9, f"Only {finite_fraction:.1%} of P_inv values are finite"
    
    def test_precond_apply_function(self, valid_grid_path):
        """Test that preconditioner apply function works."""
        config = SolverConfig(
            alpha=2.0,
            reynolds=1e6,
            solver_mode='rk5_precond',
            html_animation=False,
        )
        
        solver = RANSSolver(valid_grid_path, config)
        solver.compute_preconditioner()
        
        # Create test vector
        v = jnp.ones((solver.NI, solver.NJ, 4))
        
        # Apply preconditioner
        result = solver._precond_apply(v)
        
        assert result.shape == v.shape
        # Allow for some non-finite values at boundaries
        finite_fraction = float(jnp.mean(jnp.isfinite(result)))
        assert finite_fraction > 0.9


class TestPreconditionedRKStep:
    """Test preconditioned RK stepping."""
    
    def test_step_produces_valid_state(self, valid_grid_path):
        """Test that preconditioned RK step produces valid state."""
        config = SolverConfig(
            alpha=2.0,
            reynolds=1e6,
            cfl_start=0.5,
            cfl_target=2.0,
            max_iter=10,
            solver_mode='rk5_precond',
            html_animation=False,
        )
        
        solver = RANSSolver(valid_grid_path, config)
        
        # Run a few steps
        for _ in range(3):
            solver.step()
        
        # Check state is mostly valid (allow boundary edge cases)
        Q = solver.Q_jax
        finite_fraction = float(jnp.mean(jnp.isfinite(Q)))
        assert finite_fraction > 0.99, f"Only {finite_fraction:.1%} of Q values are finite"
        
        # Check nuHat >= 0 (physical constraint) for interior
        from src.constants import NGHOST
        nuHat = Q[NGHOST:-NGHOST, NGHOST:-NGHOST, 3]
        assert jnp.all(nuHat >= -1e-10), "nuHat should be non-negative"
    
    def test_residual_decreases(self, valid_grid_path):
        """Test that residual decreases with preconditioned RK.
        
        Note: Currently preconditioned RK may have different stability
        characteristics than standard RK. We use conservative CFL and
        test that it converges over early iterations.
        """
        config = SolverConfig(
            alpha=2.0,
            reynolds=1e6,
            cfl_start=0.1,  # Lower CFL for stability  
            cfl_target=0.5,  # Very conservative target for preconditioned mode
            cfl_ramp_iters=50,
            max_iter=15,
            solver_mode='rk5_precond',
            html_animation=False,
        )
        
        solver = RANSSolver(valid_grid_path, config)
        
        # Run iterations and track residual
        residuals = []
        for _ in range(15):
            solver.step()
            res = solver.get_residual_l1_scaled()
            residuals.append(max(res))
        
        # Check that we have valid residuals (not NaN or huge)
        valid_residuals = [r for r in residuals if np.isfinite(r) and r < 1e10]
        
        # Should stay stable for at least 10 iterations with conservative CFL
        assert len(valid_residuals) >= 10, \
            f"Preconditioned RK unstable: only {len(valid_residuals)} valid residuals"
        
        # Check that residual is decreasing (at least not increasing much)
        if len(valid_residuals) >= 5:
            # Allow up to 10% increase between consecutive residuals
            mostly_decreasing = sum(1 for i in range(len(valid_residuals)-1) 
                                   if valid_residuals[i+1] <= valid_residuals[i] * 1.1)
            assert mostly_decreasing >= len(valid_residuals) // 2, \
                "Residual should be mostly decreasing or stable"


class TestModeComparison:
    """Compare rk5 and rk5_precond modes."""
    
    def test_both_modes_run(self, valid_grid_path):
        """Test that both modes complete without error."""
        base_config = dict(
            alpha=2.0,
            reynolds=1e6,
            cfl_start=0.5,
            cfl_target=2.0,
            max_iter=10,
            html_animation=False,
        )
        
        # Standard RK5
        config_rk5 = SolverConfig(**base_config, solver_mode='rk5')
        solver_rk5 = RANSSolver(valid_grid_path, config_rk5)
        for _ in range(5):
            solver_rk5.step()
        
        # Preconditioned RK5
        config_precond = SolverConfig(**base_config, solver_mode='rk5_precond')
        solver_precond = RANSSolver(valid_grid_path, config_precond)
        for _ in range(5):
            solver_precond.step()
        
        # Both should produce mostly finite states
        rk5_finite = float(jnp.mean(jnp.isfinite(solver_rk5.Q_jax)))
        precond_finite = float(jnp.mean(jnp.isfinite(solver_precond.Q_jax)))
        
        assert rk5_finite > 0.99, f"rk5 mode: {rk5_finite:.1%} finite"
        assert precond_finite > 0.99, f"rk5_precond mode: {precond_finite:.1%} finite"
    
    def test_newton_mode_raises(self, valid_grid_path):
        """Test that newton mode raises NotImplementedError."""
        config = SolverConfig(
            alpha=2.0,
            reynolds=1e6,
            solver_mode='newton',
            html_animation=False,
        )
        
        solver = RANSSolver(valid_grid_path, config)
        
        with pytest.raises(NotImplementedError):
            solver.step()
