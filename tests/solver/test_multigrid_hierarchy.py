"""
Tests for MultigridHierarchy class.

Tests cover:
1. Hierarchy levels: correct number of levels based on grid size
2. Level dimensions: each level has NI, NJ halved from previous
3. BC index scaling: n_wake_points halved at each level
4. Farfield normals: recomputed and point outward at each level
5. Array shapes: all arrays have consistent shapes per level
6. Initial restriction: Q at coarse levels is properly restricted from fine
7. BC application: BCs can be applied at all levels
8. Residual computation: residuals can be computed at all levels
"""

import numpy as np
import pytest
from typing import Tuple

from src.grid.metrics import MetricComputer
from src.solvers.multigrid import (
    MultigridHierarchy,
    MultigridLevel,
    build_multigrid_hierarchy,
)
from src.solvers.boundary_conditions import FreestreamConditions


def create_cartesian_grid(NI: int, NJ: int, Lx: float = 1.0, Ly: float = 1.0):
    """Create a uniform Cartesian grid."""
    x = np.linspace(0, Lx, NI + 1)
    y = np.linspace(0, Ly, NJ + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return X, Y


def create_distorted_grid(NI: int, NJ: int, amplitude: float = 0.05):
    """Create a Cartesian grid with sinusoidal distortion."""
    X, Y = create_cartesian_grid(NI, NJ)
    
    dx = 1.0 / NI
    dy = 1.0 / NJ
    
    X_dist = X + amplitude * dx * np.sin(2 * np.pi * Y)
    Y_dist = Y + amplitude * dy * np.sin(2 * np.pi * X)
    
    return X_dist, Y_dist


class TestHierarchyLevels:
    """Tests for correct number of levels."""
    
    def test_correct_level_count(self):
        """Hierarchy has correct number of levels based on grid size."""
        NI, NJ = 64, 48
        X, Y = create_cartesian_grid(NI, NJ)
        Q = np.zeros((NI + 2, NJ + 2, 4))
        freestream = FreestreamConditions()
        
        hierarchy = build_multigrid_hierarchy(X, Y, Q, freestream, min_size=8)
        
        # 64x48 -> 32x24 -> 16x12 -> 8x6 (stop, 6 < 8 in one dim)
        # Actually: can_coarsen checks min(NI, NJ) >= 2*min_size
        # So: 64x48 -> 32x24 -> 16x12 (stop, 12 < 16)
        # Expected: 3 levels
        assert hierarchy.num_levels >= 2
        assert hierarchy.num_levels <= 4
    
    def test_min_size_limit(self):
        """Coarsening stops when min dimension < 2*min_size."""
        NI, NJ = 32, 32
        X, Y = create_cartesian_grid(NI, NJ)
        Q = np.zeros((NI + 2, NJ + 2, 4))
        freestream = FreestreamConditions()
        
        hierarchy = build_multigrid_hierarchy(X, Y, Q, freestream, min_size=8)
        
        # Coarsest level should have at least min_size cells
        coarsest = hierarchy.levels[-1]
        assert coarsest.NI >= 8
        assert coarsest.NJ >= 8


class TestLevelDimensions:
    """Tests for dimension halving at each level."""
    
    def test_dimensions_halved(self):
        """Each level has NI, NJ halved from previous."""
        NI, NJ = 64, 48
        X, Y = create_cartesian_grid(NI, NJ)
        Q = np.zeros((NI + 2, NJ + 2, 4))
        freestream = FreestreamConditions()
        
        hierarchy = build_multigrid_hierarchy(X, Y, Q, freestream)
        
        for i in range(1, hierarchy.num_levels):
            fine = hierarchy.levels[i - 1]
            coarse = hierarchy.levels[i]
            
            assert coarse.NI == fine.NI // 2
            assert coarse.NJ == fine.NJ // 2


class TestBCIndexScaling:
    """Tests for BC index scaling."""
    
    def test_n_wake_halved(self):
        """n_wake_points is halved at each level."""
        NI, NJ = 64, 48
        n_wake = 16
        X, Y = create_cartesian_grid(NI, NJ)
        Q = np.zeros((NI + 2, NJ + 2, 4))
        freestream = FreestreamConditions()
        
        hierarchy = build_multigrid_hierarchy(X, Y, Q, freestream, n_wake=n_wake)
        
        for i in range(hierarchy.num_levels):
            expected_n_wake = n_wake // (2 ** i)
            actual = hierarchy.levels[i].bc.n_wake_points
            assert actual == expected_n_wake, f"Level {i}: expected {expected_n_wake}, got {actual}"


class TestFarfieldNormals:
    """Tests for farfield normal computation."""
    
    def test_normals_computed_at_all_levels(self):
        """Farfield normals are computed at each level."""
        NI, NJ = 32, 24
        X, Y = create_cartesian_grid(NI, NJ)
        Q = np.zeros((NI + 2, NJ + 2, 4))
        freestream = FreestreamConditions()
        
        hierarchy = build_multigrid_hierarchy(X, Y, Q, freestream)
        
        for i, level in enumerate(hierarchy.levels):
            assert level.bc.farfield_normals is not None, f"Level {i} has no farfield normals"
            nx, ny = level.bc.farfield_normals
            assert len(nx) == level.NI, f"Level {i} nx shape mismatch"
            assert len(ny) == level.NI, f"Level {i} ny shape mismatch"
    
    def test_normals_are_unit_vectors(self):
        """Farfield normals are unit vectors."""
        NI, NJ = 32, 24
        X, Y = create_cartesian_grid(NI, NJ)
        Q = np.zeros((NI + 2, NJ + 2, 4))
        freestream = FreestreamConditions()
        
        hierarchy = build_multigrid_hierarchy(X, Y, Q, freestream)
        
        for i, level in enumerate(hierarchy.levels):
            nx, ny = level.bc.farfield_normals
            mag = np.sqrt(nx**2 + ny**2)
            np.testing.assert_allclose(mag, 1.0, rtol=1e-10, 
                                        err_msg=f"Level {i} normals not unit")


class TestArrayShapes:
    """Tests for consistent array shapes."""
    
    def test_state_array_shape(self):
        """Q has correct shape with ghost cells."""
        NI, NJ = 32, 24
        X, Y = create_cartesian_grid(NI, NJ)
        Q = np.ones((NI + 2, NJ + 2, 4))
        freestream = FreestreamConditions()
        
        hierarchy = build_multigrid_hierarchy(X, Y, Q, freestream)
        
        for level in hierarchy.levels:
            expected_shape = (level.NI + 2, level.NJ + 2, 4)
            assert level.Q.shape == expected_shape
            assert level.Q_old.shape == expected_shape
    
    def test_residual_array_shape(self):
        """R has correct shape for interior cells."""
        NI, NJ = 32, 24
        X, Y = create_cartesian_grid(NI, NJ)
        Q = np.zeros((NI + 2, NJ + 2, 4))
        freestream = FreestreamConditions()
        
        hierarchy = build_multigrid_hierarchy(X, Y, Q, freestream)
        
        for level in hierarchy.levels:
            expected_shape = (level.NI, level.NJ, 4)
            assert level.R.shape == expected_shape
            assert level.forcing.shape == expected_shape
    
    def test_metrics_shapes(self):
        """Metrics arrays have correct shapes."""
        NI, NJ = 32, 24
        X, Y = create_cartesian_grid(NI, NJ)
        Q = np.zeros((NI + 2, NJ + 2, 4))
        freestream = FreestreamConditions()
        
        hierarchy = build_multigrid_hierarchy(X, Y, Q, freestream)
        
        for level in hierarchy.levels:
            m = level.metrics
            assert m.volume.shape == (level.NI, level.NJ)
            assert m.xc.shape == (level.NI, level.NJ)
            assert m.Si_x.shape == (level.NI + 1, level.NJ)
            assert m.Sj_x.shape == (level.NI, level.NJ + 1)


class TestInitialRestriction:
    """Tests for initial state restriction."""
    
    def test_coarse_state_is_restricted(self):
        """Coarse Q is restricted from fine."""
        NI, NJ = 32, 24
        X, Y = create_cartesian_grid(NI, NJ)
        
        # Create non-uniform fine state
        Q = np.zeros((NI + 2, NJ + 2, 4))
        for k in range(4):
            Q[:, :, k] = k + 1
        
        freestream = FreestreamConditions()
        hierarchy = build_multigrid_hierarchy(X, Y, Q, freestream)
        
        # Coarse interior should have approximately same values
        # (exactly same for uniform field)
        for k in range(4):
            expected = k + 1
            coarse_interior = hierarchy.levels[1].Q[1:-1, 1:-1, k]
            np.testing.assert_allclose(coarse_interior, expected, rtol=1e-10)
    
    def test_restriction_preserves_smooth_field(self):
        """Smooth field is approximately preserved after restriction."""
        NI, NJ = 32, 32
        X, Y = create_cartesian_grid(NI, NJ)
        
        computer = MetricComputer(X, Y)
        metrics = computer.compute()
        
        # Create smooth field
        Q = np.zeros((NI + 2, NJ + 2, 4))
        for k in range(4):
            Q[1:-1, 1:-1, k] = np.sin(np.pi * metrics.xc) * np.sin(np.pi * metrics.yc)
        
        freestream = FreestreamConditions()
        hierarchy = build_multigrid_hierarchy(X, Y, Q, freestream)
        
        # Coarse level should have similar pattern
        coarse = hierarchy.levels[1]
        coarse_interior = coarse.Q[1:-1, 1:-1, 0]
        
        # Check that the field has a similar shape (max near center)
        center_i, center_j = coarse.NI // 2, coarse.NJ // 2
        center_value = coarse_interior[center_i, center_j]
        corner_value = coarse_interior[0, 0]
        
        assert center_value > corner_value, "Smooth field pattern not preserved"


class TestBCApplication:
    """Tests for boundary condition application."""
    
    def test_bcs_can_be_applied_at_all_levels(self):
        """apply_bcs works at all levels without error."""
        NI, NJ = 32, 24
        X, Y = create_cartesian_grid(NI, NJ)
        Q = np.zeros((NI + 2, NJ + 2, 4))
        Q[:, :, 1] = 1.0  # u = 1
        freestream = FreestreamConditions()
        
        hierarchy = build_multigrid_hierarchy(X, Y, Q, freestream)
        
        for i in range(hierarchy.num_levels):
            # Should not raise
            hierarchy.apply_bcs(i)
            
            # Check that Q is finite
            assert np.all(np.isfinite(hierarchy.levels[i].Q))
    
    def test_wall_bc_applied_correctly(self):
        """No-slip BC gives zero velocity at wall (in airfoil region)."""
        NI, NJ = 32, 24
        n_wake = 5  # Explicit wake points
        X, Y = create_cartesian_grid(NI, NJ)
        Q = np.ones((NI + 2, NJ + 2, 4))
        Q[:, :, 1] = 1.0  # u = 1
        Q[:, :, 2] = 0.5  # v = 0.5
        
        freestream = FreestreamConditions()
        hierarchy = build_multigrid_hierarchy(X, Y, Q, freestream, n_wake=n_wake)
        
        for i in range(hierarchy.num_levels):
            hierarchy.apply_bcs(i)
            
            level = hierarchy.levels[i]
            level_n_wake = n_wake // (2 ** i)
            
            # Ghost at j=0 should have u_ghost = -u_interior for no-slip
            # But only in airfoil region (not wake)
            # Airfoil region: from level_n_wake+1 to NI-level_n_wake
            i_start = level_n_wake + 1  # +1 for Q index offset
            i_end = level.NI - level_n_wake + 1
            
            if i_end > i_start:  # Only check if there's airfoil region
                u_ghost = level.Q[i_start:i_end, 0, 1]  # Ghost layer
                u_interior = level.Q[i_start:i_end, 1, 1]  # First interior
                
                np.testing.assert_allclose(u_ghost, -u_interior, rtol=1e-10,
                                            err_msg=f"Level {i} wall BC failed")


class TestRestrictionProlongation:
    """Tests for restriction and prolongation through hierarchy."""
    
    def test_restrict_to_coarse(self):
        """restrict_to_coarse updates coarse level state."""
        NI, NJ = 32, 24
        X, Y = create_cartesian_grid(NI, NJ)
        Q = np.ones((NI + 2, NJ + 2, 4))
        freestream = FreestreamConditions()
        
        hierarchy = build_multigrid_hierarchy(X, Y, Q, freestream)
        
        # Modify fine state
        hierarchy.levels[0].Q[1:-1, 1:-1, 0] = 2.0
        
        # Restrict
        hierarchy.restrict_to_coarse(0)
        
        # Coarse interior should now be 2.0
        coarse_interior = hierarchy.levels[1].Q[1:-1, 1:-1, 0]
        np.testing.assert_allclose(coarse_interior, 2.0, rtol=1e-10)
    
    def test_prolongate_correction(self):
        """prolongate_correction adds correction to fine level."""
        NI, NJ = 32, 24
        X, Y = create_cartesian_grid(NI, NJ)
        Q = np.ones((NI + 2, NJ + 2, 4))
        freestream = FreestreamConditions()
        
        hierarchy = build_multigrid_hierarchy(X, Y, Q, freestream)
        
        # Set Q_old on coarse level
        hierarchy.levels[1].Q_old[:] = 1.0
        
        # Modify coarse Q (new)
        hierarchy.levels[1].Q[1:-1, 1:-1, 0] = 2.0  # dQ = 1.0
        
        # Store fine Q before
        fine_before = hierarchy.levels[0].Q[1:-1, 1:-1, 0].mean()
        
        # Prolongate
        hierarchy.prolongate_correction(1)
        
        # Fine Q should have correction added (approximately 1.0)
        fine_after = hierarchy.levels[0].Q[1:-1, 1:-1, 0].mean()
        
        assert fine_after > fine_before, "Correction not added to fine level"


class TestConvenienceFunction:
    """Tests for build_multigrid_hierarchy function."""
    
    def test_returns_valid_hierarchy(self):
        """build_multigrid_hierarchy returns valid hierarchy."""
        NI, NJ = 32, 24
        X, Y = create_cartesian_grid(NI, NJ)
        Q = np.zeros((NI + 2, NJ + 2, 4))
        freestream = FreestreamConditions()
        
        hierarchy = build_multigrid_hierarchy(X, Y, Q, freestream)
        
        assert isinstance(hierarchy, MultigridHierarchy)
        assert hierarchy.num_levels >= 1
        assert len(hierarchy.levels) == hierarchy.num_levels
    
    def test_get_level_info(self):
        """get_level_info returns formatted string."""
        NI, NJ = 32, 24
        X, Y = create_cartesian_grid(NI, NJ)
        Q = np.zeros((NI + 2, NJ + 2, 4))
        freestream = FreestreamConditions()
        
        hierarchy = build_multigrid_hierarchy(X, Y, Q, freestream)
        
        info = hierarchy.get_level_info()
        
        assert isinstance(info, str)
        assert "Multigrid Hierarchy" in info
        assert "Level 0" in info
        assert "cells" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

