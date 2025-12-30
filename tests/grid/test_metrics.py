#!/usr/bin/env python3
"""
Tests for the MetricComputer class (src/grid/metrics.py).

Validates:
1. Cell volume computation (cross-product formula)
2. Face normal directions and magnitudes
3. Geometric Conservation Law (GCL)
4. Wall distance calculation
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.grid.metrics import MetricComputer, FVMMetrics, GCLValidation, compute_metrics


class TestMetrics:
    """Test suite for MetricComputer."""
    
    def __init__(self):
        self.NI = 20
        self.NJ = 10
        self.results = []
    
    def setup_uniform_grid(self):
        """Create a simple uniform Cartesian grid."""
        x = np.linspace(0, 2, self.NI + 1)
        y = np.linspace(0, 1, self.NJ + 1)
        X, Y = np.meshgrid(x, y, indexing='ij')
        return X, Y
    
    def setup_stretched_grid(self):
        """Create a stretched grid (dense near y=0)."""
        x = np.linspace(0, 2, self.NI + 1)
        # Geometric stretching in y
        y_normalized = np.linspace(0, 1, self.NJ + 1)
        y = y_normalized ** 1.5  # Stretching toward y=0
        X, Y = np.meshgrid(x, y, indexing='ij')
        return X, Y
    
    def setup_curved_grid(self):
        """Create a curved grid (simulating O-grid around cylinder)."""
        NI, NJ = self.NI, self.NJ
        
        # Angular coordinate
        theta = np.linspace(0, 2*np.pi, NI + 1)
        # Radial coordinate
        r = np.linspace(1.0, 3.0, NJ + 1)
        
        THETA, R = np.meshgrid(theta, r, indexing='ij')
        X = R * np.cos(THETA)
        Y = R * np.sin(THETA)
        
        return X, Y
    
    def run_test(self, name, test_func):
        """Run a test and record result."""
        try:
            passed, message = test_func()
            self.results.append((name, passed, message))
        except Exception as e:
            self.results.append((name, False, f"Exception: {e}"))
    
    # ===== Cell Volume Tests =====
    
    def test_uniform_grid_volume(self):
        """Uniform grid should have uniform cell volumes."""
        X, Y = self.setup_uniform_grid()
        computer = MetricComputer(X, Y)
        metrics = computer.compute()
        
        # Expected volume: dx * dy = (2/NI) * (1/NJ)
        expected = (2.0 / self.NI) * (1.0 / self.NJ)
        
        if not np.allclose(metrics.volume, expected, rtol=1e-10):
            return False, f"Volume {metrics.volume[0,0]:.6f} != expected {expected:.6f}"
        
        return True, f"Uniform volume = {expected:.6f}"
    
    def test_stretched_grid_volume_varies(self):
        """Stretched grid should have varying cell volumes."""
        X, Y = self.setup_stretched_grid()
        computer = MetricComputer(X, Y)
        metrics = computer.compute()
        
        # Volumes near y=0 should be smaller
        vol_near_wall = metrics.volume[:, 0].mean()
        vol_farfield = metrics.volume[:, -1].mean()
        
        if vol_near_wall >= vol_farfield:
            return False, f"Wall volume {vol_near_wall:.6f} >= farfield {vol_farfield:.6f}"
        
        ratio = vol_farfield / vol_near_wall
        return True, f"Volume ratio (ff/wall) = {ratio:.2f}x"
    
    def test_curved_grid_volume_positive(self):
        """All cell volumes should be positive."""
        X, Y = self.setup_curved_grid()
        computer = MetricComputer(X, Y)
        metrics = computer.compute()
        
        if np.any(metrics.volume <= 0):
            min_vol = np.min(metrics.volume)
            return False, f"Negative volume: {min_vol:.6e}"
        
        return True, f"All volumes positive (min: {metrics.volume.min():.6e})"
    
    # ===== Face Normal Tests =====
    
    def test_uniform_grid_face_normals(self):
        """Uniform grid should have axis-aligned face normals."""
        X, Y = self.setup_uniform_grid()
        computer = MetricComputer(X, Y)
        metrics = computer.compute()
        
        dx = 2.0 / self.NI
        dy = 1.0 / self.NJ
        
        # I-faces should have normals (dy, 0) pointing in +x
        if not np.allclose(metrics.Si_x, dy, rtol=1e-10):
            return False, f"Si_x = {metrics.Si_x[0,0]:.6f} != dy = {dy:.6f}"
        if not np.allclose(metrics.Si_y, 0, atol=1e-10):
            return False, f"Si_y = {metrics.Si_y[0,0]:.6f} != 0"
        
        # J-faces should have normals (0, dx) pointing in +y
        if not np.allclose(metrics.Sj_x, 0, atol=1e-10):
            return False, f"Sj_x = {metrics.Sj_x[0,0]:.6f} != 0"
        if not np.allclose(metrics.Sj_y, dx, rtol=1e-10):
            return False, f"Sj_y = {metrics.Sj_y[0,0]:.6f} != dx = {dx:.6f}"
        
        return True, f"Si=(0,{dy:.4f}), Sj=({dx:.4f},0)"
    
    def test_face_normal_magnitudes(self):
        """Face normal magnitudes should equal face lengths."""
        X, Y = self.setup_curved_grid()
        computer = MetricComputer(X, Y)
        metrics = computer.compute()
        
        # Compute expected face lengths manually
        # I-face at (i,j) connects nodes (i,j) and (i,j+1)
        expected_Si_mag = np.sqrt(
            (X[:, 1:] - X[:, :-1])**2 + 
            (Y[:, 1:] - Y[:, :-1])**2
        )
        
        if not np.allclose(metrics.Si_mag, expected_Si_mag, rtol=1e-10):
            return False, "Si_mag doesn't match face lengths"
        
        # J-face at (i,j) connects nodes (i,j) and (i+1,j)
        expected_Sj_mag = np.sqrt(
            (X[1:, :] - X[:-1, :])**2 + 
            (Y[1:, :] - Y[:-1, :])**2
        )
        
        if not np.allclose(metrics.Sj_mag, expected_Sj_mag, rtol=1e-10):
            return False, "Sj_mag doesn't match face lengths"
        
        return True, "Face magnitudes correct"
    
    # ===== GCL Tests =====
    
    def test_gcl_uniform_grid(self):
        """GCL should be satisfied exactly on uniform grid."""
        X, Y = self.setup_uniform_grid()
        computer = MetricComputer(X, Y)
        computer.compute()
        
        gcl = computer.validate_gcl(tol=1e-12)
        
        if not gcl.passed:
            return False, gcl.message
        
        return True, f"GCL: max residual {max(gcl.max_x_residual, gcl.max_y_residual):.2e}"
    
    def test_gcl_stretched_grid(self):
        """GCL should be satisfied on stretched grid."""
        X, Y = self.setup_stretched_grid()
        computer = MetricComputer(X, Y)
        computer.compute()
        
        gcl = computer.validate_gcl(tol=1e-12)
        
        if not gcl.passed:
            return False, gcl.message
        
        return True, f"GCL: max residual {max(gcl.max_x_residual, gcl.max_y_residual):.2e}"
    
    def test_gcl_curved_grid(self):
        """GCL should be satisfied on curved grid."""
        X, Y = self.setup_curved_grid()
        computer = MetricComputer(X, Y)
        computer.compute()
        
        gcl = computer.validate_gcl(tol=1e-10)
        
        if not gcl.passed:
            return False, gcl.message
        
        return True, f"GCL: max residual {max(gcl.max_x_residual, gcl.max_y_residual):.2e}"
    
    # ===== Wall Distance Tests =====
    
    def test_wall_distance_at_wall(self):
        """Wall distance should be small (half cell height) at j=0."""
        X, Y = self.setup_uniform_grid()
        computer = MetricComputer(X, Y, wall_j=0)
        metrics = computer.compute()
        
        # First cell distance should be ~ 0.5 * dy
        expected = 0.5 * (1.0 / self.NJ)
        
        if not np.allclose(metrics.wall_distance[:, 0], expected, rtol=0.1):
            return False, f"Wall dist {metrics.wall_distance[0,0]:.6f} != {expected:.6f}"
        
        return True, f"Wall distance at j=0: {metrics.wall_distance[0,0]:.4f}"
    
    def test_wall_distance_increases(self):
        """Wall distance should increase with j."""
        X, Y = self.setup_uniform_grid()
        computer = MetricComputer(X, Y, wall_j=0)
        metrics = computer.compute()
        
        # Check monotonicity along a column
        d = metrics.wall_distance[self.NI//2, :]
        
        if not np.all(np.diff(d) > 0):
            return False, "Wall distance not monotonically increasing"
        
        return True, f"Wall distance range: [{d.min():.4f}, {d.max():.4f}]"
    
    # ===== Convenience Function Test =====
    
    def test_convenience_function(self):
        """Test the compute_metrics convenience function."""
        X, Y = self.setup_uniform_grid()
        
        metrics = compute_metrics(X, Y, wall_j=0)
        
        if not isinstance(metrics, FVMMetrics):
            return False, f"Wrong return type: {type(metrics)}"
        
        if metrics.volume.shape != (self.NI, self.NJ):
            return False, f"Wrong volume shape: {metrics.volume.shape}"
        
        return True, "Convenience function works"
    
    def run_all(self):
        """Run all tests."""
        # Cell volume tests
        self.run_test("uniform_grid_volume", self.test_uniform_grid_volume)
        self.run_test("stretched_grid_volume", self.test_stretched_grid_volume_varies)
        self.run_test("curved_grid_volume_positive", self.test_curved_grid_volume_positive)
        
        # Face normal tests
        self.run_test("uniform_face_normals", self.test_uniform_grid_face_normals)
        self.run_test("face_normal_magnitudes", self.test_face_normal_magnitudes)
        
        # GCL tests
        self.run_test("gcl_uniform", self.test_gcl_uniform_grid)
        self.run_test("gcl_stretched", self.test_gcl_stretched_grid)
        self.run_test("gcl_curved", self.test_gcl_curved_grid)
        
        # Wall distance tests
        self.run_test("wall_distance_at_wall", self.test_wall_distance_at_wall)
        self.run_test("wall_distance_increases", self.test_wall_distance_increases)
        
        # Convenience function
        self.run_test("convenience_function", self.test_convenience_function)
    
    def print_results(self):
        """Print test results."""
        passed = sum(1 for _, p, _ in self.results if p)
        failed = len(self.results) - passed
        
        print(f"\nResults: {passed} passed, {failed} failed\n")
        
        for name, success, message in self.results:
            status = "✓" if success else "✗"
            print(f"  {status} {name}: {message}")
        
        if failed > 0:
            print(f"\nFAILED: {failed} test(s)")
            return False
        else:
            print("\nAll tests passed!")
            return True


def main():
    print("=" * 60)
    print("Grid Metrics Module Tests")
    print("=" * 60)
    
    tester = TestMetrics()
    tester.run_all()
    success = tester.print_results()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

