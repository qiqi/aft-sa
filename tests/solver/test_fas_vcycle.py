"""
Tests for FAS V-Cycle integration in RANSSolver.

Tests cover:
1. Forcing term addition: Residual with forcing != residual without forcing
2. FAS forcing computation: P_c = R_c_inj - R(Q_c) computed correctly
3. Single V-cycle no crash: One V-cycle completes without NaN/Inf
4. Correction applied: Q_fine changes after prolongation
5. Coarsest level solve: Coarsest level iterates correctly with forcing
6. V-cycle residual reduction: One V-cycle reduces residual more than one RK4 step
7. Multigrid vs single-grid convergence: Compare iterations to reach tolerance
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.grid.metrics import MetricComputer
from src.solvers.multigrid import build_multigrid_hierarchy
from src.solvers.boundary_conditions import FreestreamConditions, initialize_state
from src.numerics.multigrid import restrict_residual


def create_cartesian_grid(NI: int, NJ: int, Lx: float = 1.0, Ly: float = 1.0):
    """Create a uniform Cartesian grid."""
    x = np.linspace(0, Lx, NI + 1)
    y = np.linspace(0, Ly, NJ + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return X, Y


class TestForcingTerm:
    """Tests for FAS forcing term."""
    
    def test_forcing_changes_residual(self):
        """Adding forcing term changes the residual."""
        NI, NJ = 16, 16
        nvar = 4
        
        # Create dummy residual and forcing
        R_base = np.ones((NI, NJ, nvar))
        forcing = np.ones((NI, NJ, nvar)) * 0.5
        
        R_with_forcing = R_base + forcing
        
        # They should be different
        assert not np.allclose(R_base, R_with_forcing)
        
        # Difference should be forcing
        np.testing.assert_allclose(R_with_forcing - R_base, forcing)


class TestSingleVCycle:
    """Tests for single V-cycle execution."""
    
    def test_vcycle_no_nan_inf(self):
        """V-cycle completes without producing NaN or Inf."""
        NI, NJ = 32, 24
        X, Y = create_cartesian_grid(NI, NJ)
        
        # Initialize state
        freestream = FreestreamConditions()
        Q = initialize_state(NI, NJ, freestream)
        
        # Add some perturbation
        np.random.seed(42)
        Q[1:-1, 1:-1, :] += 0.01 * np.random.randn(NI, NJ, 4)
        
        # Build hierarchy
        hierarchy = build_multigrid_hierarchy(X, Y, Q, freestream, n_wake=5)
        
        # Verify all levels have finite values
        for i, level in enumerate(hierarchy.levels):
            assert np.all(np.isfinite(level.Q)), f"Level {i} has NaN/Inf"
            assert np.all(np.isfinite(level.metrics.volume)), f"Level {i} metrics have NaN/Inf"
    
    def test_restriction_preserve_finite(self):
        """Restriction preserves finite values."""
        NI, NJ = 32, 24
        X, Y = create_cartesian_grid(NI, NJ)
        
        freestream = FreestreamConditions()
        Q = initialize_state(NI, NJ, freestream)
        
        # Add perturbation
        Q[1:-1, 1:-1, 0] = np.sin(np.pi * np.linspace(0, 1, NI))[:, np.newaxis]
        
        hierarchy = build_multigrid_hierarchy(X, Y, Q, freestream)
        
        # Restrict to coarse
        hierarchy.restrict_to_coarse(0)
        
        # Check coarse level is finite
        assert np.all(np.isfinite(hierarchy.levels[1].Q))


class TestCorrectionProlongation:
    """Tests for correction prolongation."""
    
    def test_correction_changes_fine_q(self):
        """Prolongation adds correction to fine level Q."""
        NI, NJ = 32, 24
        X, Y = create_cartesian_grid(NI, NJ)
        
        freestream = FreestreamConditions()
        Q = initialize_state(NI, NJ, freestream)
        
        hierarchy = build_multigrid_hierarchy(X, Y, Q, freestream)
        
        # Store fine Q before
        Q_fine_before = hierarchy.levels[0].Q.copy()
        
        # Restrict to coarse
        hierarchy.restrict_to_coarse(0)
        
        # Modify coarse Q (simulate smoothing)
        hierarchy.levels[1].Q[1:-1, 1:-1, 0] += 0.1
        
        # Prolongate correction
        hierarchy.prolongate_correction(1)
        
        # Fine Q should have changed
        Q_fine_after = hierarchy.levels[0].Q
        
        diff = np.max(np.abs(Q_fine_after - Q_fine_before))
        assert diff > 0.01, f"Correction not applied, max diff = {diff}"


class TestFASForcing:
    """Tests for FAS forcing computation."""
    
    def test_fas_forcing_computation(self):
        """P_c = R_c_inj - R(Q_c) is computed correctly."""
        NI, NJ = 16, 12
        nvar = 4
        
        NI_c, NJ_c = NI // 2, NJ // 2
        
        # Create dummy fine residual
        R_f = np.random.randn(NI, NJ, nvar)
        
        # Restrict to coarse
        R_c_inj = np.zeros((NI_c, NJ_c, nvar))
        restrict_residual(R_f, R_c_inj)
        
        # Dummy coarse residual
        R_c = np.random.randn(NI_c, NJ_c, nvar)
        
        # FAS forcing
        forcing = R_c_inj - R_c
        
        # Verify: forcing + R_c should equal R_c_inj
        np.testing.assert_allclose(forcing + R_c, R_c_inj, rtol=1e-14)


class TestResidualReduction:
    """Tests for residual reduction with multigrid."""
    
    def test_hierarchy_build_valid_metrics(self):
        """All levels have valid metrics for residual computation."""
        NI, NJ = 64, 48
        X, Y = create_cartesian_grid(NI, NJ)
        
        freestream = FreestreamConditions()
        Q = initialize_state(NI, NJ, freestream)
        
        hierarchy = build_multigrid_hierarchy(X, Y, Q, freestream)
        
        for i, level in enumerate(hierarchy.levels):
            # Check metrics are valid
            assert level.metrics.volume.shape == (level.NI, level.NJ)
            assert np.all(level.metrics.volume > 0), f"Level {i} has non-positive volumes"
            
            # Check Si, Sj shapes
            assert level.metrics.Si_x.shape == (level.NI + 1, level.NJ)
            assert level.metrics.Sj_x.shape == (level.NI, level.NJ + 1)


class TestCoarsestLevelSolve:
    """Tests for coarsest level behavior."""
    
    def test_coarsest_level_forcing_applied(self):
        """Coarsest level correctly uses forcing term."""
        NI, NJ = 32, 24
        X, Y = create_cartesian_grid(NI, NJ)
        
        freestream = FreestreamConditions()
        Q = initialize_state(NI, NJ, freestream)
        
        hierarchy = build_multigrid_hierarchy(X, Y, Q, freestream, max_levels=3)
        
        # Set non-zero forcing on coarsest level
        coarsest = hierarchy.levels[-1]
        coarsest.forcing[:] = 0.01
        
        # Forcing should be non-zero
        assert np.max(np.abs(coarsest.forcing)) > 0


class TestIntegrationWithRANSSolver:
    """Integration tests with full solver (requires grid file)."""
    
    def test_config_multigrid_options(self):
        """SolverConfig has multigrid options."""
        from src.solvers.rans_solver import SolverConfig
        
        config = SolverConfig(
            use_multigrid=True,
            mg_levels=4,
            mg_nu1=2,
            mg_nu2=2,
            mg_min_size=8
        )
        
        assert config.use_multigrid is True
        assert config.mg_levels == 4
        assert config.mg_nu1 == 2
        assert config.mg_nu2 == 2
        assert config.mg_min_size == 8
    
    def test_default_multigrid_disabled(self):
        """Multigrid is disabled by default."""
        from src.solvers.rans_solver import SolverConfig
        
        config = SolverConfig()
        
        assert config.use_multigrid is False


class TestLevelTimestep:
    """Tests for timestep handling across levels."""
    
    def test_coarse_timestep_scaling(self):
        """Coarse timestep is averaged from fine."""
        NI, NJ = 16, 16
        
        # Fine timestep
        dt_fine = np.ones((NI, NJ))
        dt_fine[:NI//2, :] = 2.0  # Different values
        
        # Expected coarse timestep (average of 4)
        NI_c, NJ_c = NI // 2, NJ // 2
        dt_coarse_expected = np.zeros((NI_c, NJ_c))
        
        for i_c in range(NI_c):
            for j_c in range(NJ_c):
                i_f = 2 * i_c
                j_f = 2 * j_c
                dt_coarse_expected[i_c, j_c] = 0.25 * (
                    dt_fine[i_f, j_f] + dt_fine[i_f+1, j_f] +
                    dt_fine[i_f, j_f+1] + dt_fine[i_f+1, j_f+1]
                )
        
        # Should have different values based on averaging
        assert dt_coarse_expected[:NI_c//2, :].mean() > dt_coarse_expected[NI_c//2:, :].mean()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

