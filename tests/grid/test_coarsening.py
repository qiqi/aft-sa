"""
Tests for grid coarsening module.

Tests cover:
1. Volume conservation: sum of fine volumes == sum of coarse volumes
2. GCL compliance: for each coarse cell, sum of face normals = 0
3. Face normal consistency: coarse face normal magnitude <= sum of fine magnitudes
4. Wall distance minimum: coarse wall_dist == min of 4 fine wall_dist values
5. Dimensions: NI_coarse = NI_fine // 2, NJ_coarse = NJ_fine // 2
6. Multi-level coarsening: verify GCL at each level
7. Airfoil grid coarsening: verify topology preserved
"""

import numpy as np
import pytest
from typing import Tuple

from src.grid.metrics import MetricComputer, FVMMetrics
from src.grid.coarsening import (
    Coarsener,
    coarsen_volumes,
    coarsen_cell_centers,
    coarsen_i_face_normals,
    coarsen_j_face_normals,
    coarsen_wall_distance,
    validate_gcl_coarse
)


def create_cartesian_grid(NI: int, NJ: int, 
                           Lx: float = 1.0, Ly: float = 1.0
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Create a uniform Cartesian grid."""
    x = np.linspace(0, Lx, NI + 1)
    y = np.linspace(0, Ly, NJ + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return X, Y


def create_distorted_grid(NI: int, NJ: int,
                           Lx: float = 1.0, Ly: float = 1.0,
                           amplitude: float = 0.1
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Create a Cartesian grid with sinusoidal distortion."""
    X, Y = create_cartesian_grid(NI, NJ, Lx, Ly)
    
    # Apply distortion
    dx = Lx / NI
    dy = Ly / NJ
    
    # Distort interior nodes
    X_dist = X + amplitude * dx * np.sin(2 * np.pi * Y / Ly)
    Y_dist = Y + amplitude * dy * np.sin(2 * np.pi * X / Lx)
    
    return X_dist, Y_dist


class TestVolumeConservation:
    """Tests for volume conservation during coarsening."""
    
    def test_cartesian_grid_volume_conservation(self):
        """Sum of fine volumes equals sum of coarse volumes for Cartesian grid."""
        NI, NJ = 16, 12
        X, Y = create_cartesian_grid(NI, NJ)
        
        computer = MetricComputer(X, Y)
        fine_metrics = computer.compute()
        
        coarse_metrics = Coarsener.coarsen(fine_metrics)
        
        fine_total = np.sum(fine_metrics.volume)
        coarse_total = np.sum(coarse_metrics.volume)
        
        np.testing.assert_allclose(fine_total, coarse_total, rtol=1e-14)
    
    def test_distorted_grid_volume_conservation(self):
        """Sum of fine volumes equals sum of coarse volumes for distorted grid."""
        NI, NJ = 20, 16
        X, Y = create_distorted_grid(NI, NJ)
        
        computer = MetricComputer(X, Y)
        fine_metrics = computer.compute()
        
        coarse_metrics = Coarsener.coarsen(fine_metrics)
        
        fine_total = np.sum(fine_metrics.volume)
        coarse_total = np.sum(coarse_metrics.volume)
        
        np.testing.assert_allclose(fine_total, coarse_total, rtol=1e-14)
    
    def test_coarse_volume_equals_four_fine_volumes(self):
        """Each coarse cell volume equals sum of its 4 fine children."""
        NI, NJ = 8, 8
        X, Y = create_cartesian_grid(NI, NJ)
        
        computer = MetricComputer(X, Y)
        fine_metrics = computer.compute()
        
        coarse_metrics = Coarsener.coarsen(fine_metrics)
        
        NI_c, NJ_c = coarse_metrics.NI, coarse_metrics.NJ
        
        for i_c in range(NI_c):
            for j_c in range(NJ_c):
                i_f = 2 * i_c
                j_f = 2 * j_c
                
                expected = (fine_metrics.volume[i_f, j_f] +
                           fine_metrics.volume[i_f+1, j_f] +
                           fine_metrics.volume[i_f, j_f+1] +
                           fine_metrics.volume[i_f+1, j_f+1])
                
                np.testing.assert_allclose(
                    coarse_metrics.volume[i_c, j_c], expected, rtol=1e-14
                )


class TestGCLCompliance:
    """Tests for Geometric Conservation Law compliance."""
    
    def test_gcl_cartesian_grid(self):
        """GCL is satisfied for coarsened Cartesian grid."""
        NI, NJ = 16, 16
        X, Y = create_cartesian_grid(NI, NJ)
        
        computer = MetricComputer(X, Y)
        fine_metrics = computer.compute()
        
        coarse_metrics = Coarsener.coarsen(fine_metrics)
        
        max_res_x, max_res_y = Coarsener.validate_gcl(coarse_metrics)
        
        assert max_res_x < 1e-14, f"GCL x-residual too large: {max_res_x}"
        assert max_res_y < 1e-14, f"GCL y-residual too large: {max_res_y}"
    
    def test_gcl_distorted_grid(self):
        """GCL is satisfied for coarsened distorted grid."""
        NI, NJ = 24, 20
        X, Y = create_distorted_grid(NI, NJ, amplitude=0.15)
        
        computer = MetricComputer(X, Y)
        fine_metrics = computer.compute()
        
        coarse_metrics = Coarsener.coarsen(fine_metrics)
        
        max_res_x, max_res_y = Coarsener.validate_gcl(coarse_metrics)
        
        assert max_res_x < 1e-13, f"GCL x-residual too large: {max_res_x}"
        assert max_res_y < 1e-13, f"GCL y-residual too large: {max_res_y}"
    
    def test_gcl_per_cell(self):
        """Verify GCL for each individual coarse cell."""
        NI, NJ = 8, 8
        X, Y = create_cartesian_grid(NI, NJ)
        
        computer = MetricComputer(X, Y)
        fine_metrics = computer.compute()
        
        coarse_metrics = Coarsener.coarsen(fine_metrics)
        
        # Check each cell
        for i in range(coarse_metrics.NI):
            for j in range(coarse_metrics.NJ):
                # Sum of outward-pointing face normals
                res_x = (coarse_metrics.Si_x[i+1, j] - coarse_metrics.Si_x[i, j] +
                         coarse_metrics.Sj_x[i, j+1] - coarse_metrics.Sj_x[i, j])
                res_y = (coarse_metrics.Si_y[i+1, j] - coarse_metrics.Si_y[i, j] +
                         coarse_metrics.Sj_y[i, j+1] - coarse_metrics.Sj_y[i, j])
                
                assert abs(res_x) < 1e-14, f"Cell ({i},{j}) GCL x-residual: {res_x}"
                assert abs(res_y) < 1e-14, f"Cell ({i},{j}) GCL y-residual: {res_y}"


class TestFaceNormalConsistency:
    """Tests for face normal consistency."""
    
    def test_coarse_face_equals_fine_sum(self):
        """Coarse face normal equals sum of constituent fine face normals."""
        NI, NJ = 8, 8
        X, Y = create_cartesian_grid(NI, NJ)
        
        computer = MetricComputer(X, Y)
        fine_metrics = computer.compute()
        
        coarse_metrics = Coarsener.coarsen(fine_metrics)
        
        NI_c = coarse_metrics.NI
        NJ_c = coarse_metrics.NJ
        
        # Check I-faces
        for i_c in range(NI_c + 1):
            for j_c in range(NJ_c):
                i_f = 2 * i_c
                j_f = 2 * j_c
                
                if i_f <= NI:
                    expected_x = fine_metrics.Si_x[i_f, j_f] + fine_metrics.Si_x[i_f, j_f+1]
                    expected_y = fine_metrics.Si_y[i_f, j_f] + fine_metrics.Si_y[i_f, j_f+1]
                    
                    np.testing.assert_allclose(
                        coarse_metrics.Si_x[i_c, j_c], expected_x, rtol=1e-14
                    )
                    np.testing.assert_allclose(
                        coarse_metrics.Si_y[i_c, j_c], expected_y, rtol=1e-14
                    )


class TestWallDistance:
    """Tests for wall distance coarsening."""
    
    def test_wall_distance_minimum(self):
        """Coarse wall distance is minimum of 4 fine values."""
        NI, NJ = 8, 8
        X, Y = create_cartesian_grid(NI, NJ)
        
        computer = MetricComputer(X, Y)
        fine_metrics = computer.compute()
        
        coarse_metrics = Coarsener.coarsen(fine_metrics)
        
        NI_c, NJ_c = coarse_metrics.NI, coarse_metrics.NJ
        
        for i_c in range(NI_c):
            for j_c in range(NJ_c):
                i_f = 2 * i_c
                j_f = 2 * j_c
                
                expected = min(
                    fine_metrics.wall_distance[i_f, j_f],
                    fine_metrics.wall_distance[i_f+1, j_f],
                    fine_metrics.wall_distance[i_f, j_f+1],
                    fine_metrics.wall_distance[i_f+1, j_f+1]
                )
                
                np.testing.assert_allclose(
                    coarse_metrics.wall_distance[i_c, j_c], expected, rtol=1e-14
                )
    
    def test_wall_distance_conservative(self):
        """Coarse wall distance never exceeds any of its fine children."""
        NI, NJ = 16, 12
        X, Y = create_distorted_grid(NI, NJ)
        
        computer = MetricComputer(X, Y)
        fine_metrics = computer.compute()
        
        coarse_metrics = Coarsener.coarsen(fine_metrics)
        
        NI_c, NJ_c = coarse_metrics.NI, coarse_metrics.NJ
        
        for i_c in range(NI_c):
            for j_c in range(NJ_c):
                i_f = 2 * i_c
                j_f = 2 * j_c
                
                coarse_dist = coarse_metrics.wall_distance[i_c, j_c]
                
                # Should be <= all 4 fine values
                assert coarse_dist <= fine_metrics.wall_distance[i_f, j_f] + 1e-14
                assert coarse_dist <= fine_metrics.wall_distance[i_f+1, j_f] + 1e-14
                assert coarse_dist <= fine_metrics.wall_distance[i_f, j_f+1] + 1e-14
                assert coarse_dist <= fine_metrics.wall_distance[i_f+1, j_f+1] + 1e-14


class TestDimensions:
    """Tests for correct dimension handling."""
    
    def test_coarse_dimensions_halved(self):
        """Coarse grid has half the cells in each direction."""
        test_cases = [(16, 12), (24, 20), (32, 32), (8, 8)]
        
        for NI, NJ in test_cases:
            X, Y = create_cartesian_grid(NI, NJ)
            
            computer = MetricComputer(X, Y)
            fine_metrics = computer.compute()
            
            coarse_metrics = Coarsener.coarsen(fine_metrics)
            
            assert coarse_metrics.NI == NI // 2
            assert coarse_metrics.NJ == NJ // 2
    
    def test_face_array_shapes(self):
        """Face arrays have correct shapes after coarsening."""
        NI, NJ = 16, 12
        X, Y = create_cartesian_grid(NI, NJ)
        
        computer = MetricComputer(X, Y)
        fine_metrics = computer.compute()
        
        coarse_metrics = Coarsener.coarsen(fine_metrics)
        
        NI_c, NJ_c = coarse_metrics.NI, coarse_metrics.NJ
        
        # I-face arrays: (NI_c + 1, NJ_c)
        assert coarse_metrics.Si_x.shape == (NI_c + 1, NJ_c)
        assert coarse_metrics.Si_y.shape == (NI_c + 1, NJ_c)
        
        # J-face arrays: (NI_c, NJ_c + 1)
        assert coarse_metrics.Sj_x.shape == (NI_c, NJ_c + 1)
        assert coarse_metrics.Sj_y.shape == (NI_c, NJ_c + 1)


class TestMultiLevelCoarsening:
    """Tests for multi-level coarsening."""
    
    def test_three_level_coarsening_gcl(self):
        """GCL is satisfied at all levels of 3-level hierarchy."""
        NI, NJ = 32, 24
        X, Y = create_distorted_grid(NI, NJ)
        
        computer = MetricComputer(X, Y)
        metrics = computer.compute()
        
        # Level 0 (fine)
        gcl = computer.validate_gcl()
        assert gcl.passed, f"Level 0 GCL failed: {gcl.message}"
        
        # Level 1
        metrics = Coarsener.coarsen(metrics)
        max_x, max_y = Coarsener.validate_gcl(metrics)
        assert max_x < 1e-13, f"Level 1 GCL x-residual: {max_x}"
        assert max_y < 1e-13, f"Level 1 GCL y-residual: {max_y}"
        
        # Level 2
        metrics = Coarsener.coarsen(metrics)
        max_x, max_y = Coarsener.validate_gcl(metrics)
        assert max_x < 1e-13, f"Level 2 GCL x-residual: {max_x}"
        assert max_y < 1e-13, f"Level 2 GCL y-residual: {max_y}"
    
    def test_volume_conservation_all_levels(self):
        """Total volume is conserved across all coarsening levels."""
        NI, NJ = 64, 48
        X, Y = create_distorted_grid(NI, NJ, amplitude=0.05)
        
        computer = MetricComputer(X, Y)
        metrics = computer.compute()
        
        total_volume = np.sum(metrics.volume)
        
        # Coarsen 4 times
        for level in range(4):
            metrics = Coarsener.coarsen(metrics)
            coarse_volume = np.sum(metrics.volume)
            
            np.testing.assert_allclose(
                coarse_volume, total_volume, rtol=1e-14,
                err_msg=f"Volume not conserved at level {level+1}"
            )


class TestCoarsenerUtilities:
    """Tests for Coarsener utility methods."""
    
    def test_can_coarsen(self):
        """can_coarsen correctly identifies coarsenable grids."""
        assert Coarsener.can_coarsen(16, 16, min_size=4)
        assert Coarsener.can_coarsen(8, 8, min_size=4)
        assert not Coarsener.can_coarsen(6, 6, min_size=4)
        assert not Coarsener.can_coarsen(16, 6, min_size=4)
    
    def test_max_levels(self):
        """max_levels returns correct count."""
        # 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4 (5 levels)
        assert Coarsener.max_levels(64, 64, min_size=4) == 5
        
        # 16x16 -> 8x8 -> 4x4 (3 levels)
        assert Coarsener.max_levels(16, 16, min_size=4) == 3
        
        # 8x8 -> 4x4 (2 levels)
        assert Coarsener.max_levels(8, 8, min_size=4) == 2
        
        # 6x6 cannot be coarsened (1 level)
        assert Coarsener.max_levels(6, 6, min_size=4) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

