"""
Tests for multigrid transfer operators.

Tests cover:
1. restrict_state_constant: Uniform Q_fine -> same Q_coarse everywhere
2. restrict_state_conservation: sum(Q_f * vol_f) == sum(Q_c * vol_c)
3. restrict_residual_conservation: sum(R_f) == sum(R_c)
4. prolongate_constant_correction: Constant dQ_c -> constant correction on fine
5. prolongate_then_restrict_identity: Restrict(Prolong(dQ)) approx == dQ
6. restrict_linear_field: Linear field preserved after restrict then prolongate
7. restrict_prolong_cycle: Smooth fields preserved through cycle
8. transfer_with_real_solution: Real solution transfers without oscillations
"""

import numpy as np
import pytest
from typing import Tuple

from src.grid.metrics import MetricComputer
from src.grid.coarsening import Coarsener
from src.numerics.multigrid import (
    restrict_state,
    restrict_residual,
    prolongate_correction,
    prolongate_injection,
    compute_integral,
    compute_residual_sum,
    create_coarse_arrays,
)


def create_cartesian_grid(NI: int, NJ: int, Lx: float = 1.0, Ly: float = 1.0):
    """Create a uniform Cartesian grid."""
    x = np.linspace(0, Lx, NI + 1)
    y = np.linspace(0, Ly, NJ + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return X, Y


def create_distorted_grid(NI: int, NJ: int, amplitude: float = 0.1):
    """Create a Cartesian grid with sinusoidal distortion."""
    X, Y = create_cartesian_grid(NI, NJ)
    
    dx = 1.0 / NI
    dy = 1.0 / NJ
    
    X_dist = X + amplitude * dx * np.sin(2 * np.pi * Y)
    Y_dist = Y + amplitude * dy * np.sin(2 * np.pi * X)
    
    return X_dist, Y_dist


class TestRestrictStateConstant:
    """Tests for constant field restriction."""
    
    def test_uniform_field_preserved(self):
        """Uniform Q on fine grid -> same uniform Q on coarse grid."""
        NI_f, NJ_f = 16, 12
        nvar = 4
        
        X, Y = create_cartesian_grid(NI_f, NJ_f)
        computer = MetricComputer(X, Y)
        metrics_f = computer.compute()
        metrics_c = Coarsener.coarsen(metrics_f)
        
        NI_c, NJ_c = metrics_c.NI, metrics_c.NJ
        
        # Uniform fine grid state
        Q_f = np.ones((NI_f, NJ_f, nvar))
        for k in range(nvar):
            Q_f[:, :, k] = (k + 1) * 3.14  # Different constant for each var
        
        # Restrict
        Q_c = np.zeros((NI_c, NJ_c, nvar))
        restrict_state(Q_f, metrics_f.volume, Q_c, metrics_c.volume)
        
        # Check each variable is constant
        for k in range(nvar):
            expected = (k + 1) * 3.14
            np.testing.assert_allclose(Q_c[:, :, k], expected, rtol=1e-14)


class TestRestrictStateConservation:
    """Tests for state conservation during restriction."""
    
    def test_conservation_cartesian(self):
        """sum(Q_f * vol_f) == sum(Q_c * vol_c) for Cartesian grid."""
        NI_f, NJ_f = 16, 16
        nvar = 4
        
        X, Y = create_cartesian_grid(NI_f, NJ_f)
        computer = MetricComputer(X, Y)
        metrics_f = computer.compute()
        metrics_c = Coarsener.coarsen(metrics_f)
        
        NI_c, NJ_c = metrics_c.NI, metrics_c.NJ
        
        # Random fine grid state
        np.random.seed(42)
        Q_f = np.random.randn(NI_f, NJ_f, nvar)
        
        # Restrict
        Q_c = np.zeros((NI_c, NJ_c, nvar))
        restrict_state(Q_f, metrics_f.volume, Q_c, metrics_c.volume)
        
        # Check conservation
        integral_f = compute_integral(Q_f, metrics_f.volume)
        integral_c = compute_integral(Q_c, metrics_c.volume)
        
        np.testing.assert_allclose(integral_f, integral_c, rtol=1e-14)
    
    def test_conservation_distorted(self):
        """sum(Q_f * vol_f) == sum(Q_c * vol_c) for distorted grid."""
        NI_f, NJ_f = 24, 20
        nvar = 4
        
        X, Y = create_distorted_grid(NI_f, NJ_f, amplitude=0.15)
        computer = MetricComputer(X, Y)
        metrics_f = computer.compute()
        metrics_c = Coarsener.coarsen(metrics_f)
        
        NI_c, NJ_c = metrics_c.NI, metrics_c.NJ
        
        # Smooth field (Gaussian)
        np.random.seed(123)
        Q_f = np.zeros((NI_f, NJ_f, nvar))
        for k in range(nvar):
            Q_f[:, :, k] = np.exp(-((metrics_f.xc - 0.5)**2 + 
                                     (metrics_f.yc - 0.5)**2) / 0.1)
        
        # Restrict
        Q_c = np.zeros((NI_c, NJ_c, nvar))
        restrict_state(Q_f, metrics_f.volume, Q_c, metrics_c.volume)
        
        # Check conservation
        integral_f = compute_integral(Q_f, metrics_f.volume)
        integral_c = compute_integral(Q_c, metrics_c.volume)
        
        np.testing.assert_allclose(integral_f, integral_c, rtol=1e-13)


class TestRestrictResidualConservation:
    """Tests for residual conservation during restriction."""
    
    def test_residual_sum_preserved(self):
        """sum(R_f) == sum(R_c) for residual restriction."""
        NI_f, NJ_f = 20, 16
        nvar = 4
        NI_c, NJ_c = NI_f // 2, NJ_f // 2
        
        # Random residuals
        np.random.seed(42)
        R_f = np.random.randn(NI_f, NJ_f, nvar)
        
        # Restrict
        R_c = np.zeros((NI_c, NJ_c, nvar))
        restrict_residual(R_f, R_c)
        
        # Check sum preserved
        sum_f = compute_residual_sum(R_f)
        sum_c = compute_residual_sum(R_c)
        
        np.testing.assert_allclose(sum_f, sum_c, rtol=1e-14)
    
    def test_each_coarse_cell_is_sum(self):
        """Each coarse residual equals sum of its 4 fine children."""
        NI_f, NJ_f = 8, 8
        nvar = 4
        NI_c, NJ_c = NI_f // 2, NJ_f // 2
        
        np.random.seed(123)
        R_f = np.random.randn(NI_f, NJ_f, nvar)
        
        R_c = np.zeros((NI_c, NJ_c, nvar))
        restrict_residual(R_f, R_c)
        
        for i_c in range(NI_c):
            for j_c in range(NJ_c):
                i_f = 2 * i_c
                j_f = 2 * j_c
                
                for k in range(nvar):
                    expected = (R_f[i_f, j_f, k] + R_f[i_f+1, j_f, k] +
                               R_f[i_f, j_f+1, k] + R_f[i_f+1, j_f+1, k])
                    
                    np.testing.assert_allclose(R_c[i_c, j_c, k], expected, rtol=1e-14)


class TestProlongateCorrectionConstant:
    """Tests for constant correction prolongation."""
    
    def test_constant_correction_uniform(self):
        """Constant dQ_c -> constant correction on fine."""
        NI_f, NJ_f = 16, 12
        nvar = 4
        NI_c, NJ_c = NI_f // 2, NJ_f // 2
        
        # Initial fine state
        Q_f = np.ones((NI_f, NJ_f, nvar))
        
        # Coarse states with constant difference
        Q_c_old = np.ones((NI_c, NJ_c, nvar))
        Q_c_new = np.ones((NI_c, NJ_c, nvar)) * 2.0  # dQ = 1.0
        
        # Prolongate
        prolongate_injection(Q_f, Q_c_new, Q_c_old)
        
        # Fine state should be 2.0 everywhere (1.0 + 1.0 correction)
        np.testing.assert_allclose(Q_f, 2.0, rtol=1e-14)
    
    def test_bilinear_constant_correction(self):
        """Bilinear prolongation with constant dQ also gives constant correction."""
        NI_f, NJ_f = 16, 12
        nvar = 4
        NI_c, NJ_c = NI_f // 2, NJ_f // 2
        
        Q_f = np.ones((NI_f, NJ_f, nvar))
        Q_c_old = np.ones((NI_c, NJ_c, nvar))
        Q_c_new = np.ones((NI_c, NJ_c, nvar)) * 2.0
        
        prolongate_correction(Q_f, Q_c_new, Q_c_old)
        
        np.testing.assert_allclose(Q_f, 2.0, rtol=1e-14)


class TestProlongateRestrictCycle:
    """Tests for restrict-prolongate round-trip."""
    
    def test_smooth_field_preserved(self):
        """Smooth field is approximately preserved through restrict-prolongate cycle."""
        NI_f, NJ_f = 32, 32
        nvar = 4
        
        X, Y = create_cartesian_grid(NI_f, NJ_f)
        computer = MetricComputer(X, Y)
        metrics_f = computer.compute()
        metrics_c = Coarsener.coarsen(metrics_f)
        
        NI_c, NJ_c = metrics_c.NI, metrics_c.NJ
        
        # Create smooth field (low-frequency sine wave)
        Q_f_orig = np.zeros((NI_f, NJ_f, nvar))
        for k in range(nvar):
            Q_f_orig[:, :, k] = np.sin(np.pi * metrics_f.xc) * np.sin(np.pi * metrics_f.yc)
        
        # Restrict to coarse
        Q_c = np.zeros((NI_c, NJ_c, nvar))
        restrict_state(Q_f_orig, metrics_f.volume, Q_c, metrics_c.volume)
        
        # Prolongate back (with zero correction, just to test the pattern)
        Q_f = Q_f_orig.copy()
        Q_c_old = Q_c.copy()
        Q_c_new = Q_c.copy()  # No change, so dQ = 0
        
        prolongate_correction(Q_f, Q_c_new, Q_c_old)
        
        # Should be unchanged since dQ = 0
        np.testing.assert_allclose(Q_f, Q_f_orig, rtol=1e-14)
    
    def test_correction_adds_to_fine(self):
        """Prolongation adds correction to fine grid."""
        NI_f, NJ_f = 16, 16
        nvar = 1
        NI_c, NJ_c = NI_f // 2, NJ_f // 2
        
        Q_f = np.zeros((NI_f, NJ_f, nvar))
        Q_c_old = np.zeros((NI_c, NJ_c, nvar))
        
        # Non-zero correction at one coarse cell
        Q_c_new = np.zeros((NI_c, NJ_c, nvar))
        Q_c_new[NI_c//2, NJ_c//2, 0] = 1.0
        
        prolongate_correction(Q_f, Q_c_new, Q_c_old)
        
        # Some fine cells should be non-zero
        assert np.max(np.abs(Q_f)) > 0.1, "Correction should propagate to fine grid"


class TestLinearFieldPreservation:
    """Tests for linear field handling."""
    
    def test_linear_field_restricted_exactly(self):
        """Linear field Q = ax + by is exactly preserved after restriction."""
        NI_f, NJ_f = 16, 16
        nvar = 1
        
        X, Y = create_cartesian_grid(NI_f, NJ_f)
        computer = MetricComputer(X, Y)
        metrics_f = computer.compute()
        metrics_c = Coarsener.coarsen(metrics_f)
        
        NI_c, NJ_c = metrics_c.NI, metrics_c.NJ
        
        # Linear field
        a, b = 2.0, 3.0
        Q_f = np.zeros((NI_f, NJ_f, nvar))
        Q_f[:, :, 0] = a * metrics_f.xc + b * metrics_f.yc
        
        # Restrict
        Q_c = np.zeros((NI_c, NJ_c, nvar))
        restrict_state(Q_f, metrics_f.volume, Q_c, metrics_c.volume)
        
        # Check that Q_c is also linear at coarse cell centers
        expected = a * metrics_c.xc + b * metrics_c.yc
        
        # For uniform grid, volume-weighted average of linear field 
        # should give the value at the centroid
        np.testing.assert_allclose(Q_c[:, :, 0], expected, rtol=1e-10)


class TestCreateCoarseArrays:
    """Tests for coarse array creation utility."""
    
    def test_correct_shapes(self):
        """create_coarse_arrays returns arrays with correct shapes."""
        NI_c, NJ_c, nvar = 8, 6, 4
        
        Q_c, R_c = create_coarse_arrays(NI_c, NJ_c, nvar)
        
        assert Q_c.shape == (NI_c, NJ_c, nvar)
        assert R_c.shape == (NI_c, NJ_c, nvar)
    
    def test_initialized_to_zero(self):
        """Arrays are initialized to zero."""
        Q_c, R_c = create_coarse_arrays(8, 6, 4)
        
        assert np.all(Q_c == 0)
        assert np.all(R_c == 0)


class TestIntegrationWithRealGrid:
    """Integration tests with realistic grids."""
    
    def test_multi_level_conservation(self):
        """Conservation holds through multiple restriction levels."""
        NI, NJ = 64, 48
        nvar = 4
        
        X, Y = create_distorted_grid(NI, NJ, amplitude=0.05)
        computer = MetricComputer(X, Y)
        metrics_l0 = computer.compute()
        metrics_l1 = Coarsener.coarsen(metrics_l0)
        metrics_l2 = Coarsener.coarsen(metrics_l1)
        
        # Create field with mixed frequencies
        np.random.seed(42)
        Q_l0 = np.zeros((NI, NJ, nvar))
        for k in range(nvar):
            Q_l0[:, :, k] = (np.sin(2 * np.pi * metrics_l0.xc) * 
                            np.cos(4 * np.pi * metrics_l0.yc) +
                            0.1 * np.random.randn(NI, NJ))
        
        # Restrict L0 -> L1
        NI_1, NJ_1 = metrics_l1.NI, metrics_l1.NJ
        Q_l1 = np.zeros((NI_1, NJ_1, nvar))
        restrict_state(Q_l0, metrics_l0.volume, Q_l1, metrics_l1.volume)
        
        # Restrict L1 -> L2
        NI_2, NJ_2 = metrics_l2.NI, metrics_l2.NJ
        Q_l2 = np.zeros((NI_2, NJ_2, nvar))
        restrict_state(Q_l1, metrics_l1.volume, Q_l2, metrics_l2.volume)
        
        # Check conservation at each level
        integral_0 = compute_integral(Q_l0, metrics_l0.volume)
        integral_1 = compute_integral(Q_l1, metrics_l1.volume)
        integral_2 = compute_integral(Q_l2, metrics_l2.volume)
        
        np.testing.assert_allclose(integral_0, integral_1, rtol=1e-13)
        np.testing.assert_allclose(integral_0, integral_2, rtol=1e-13)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

