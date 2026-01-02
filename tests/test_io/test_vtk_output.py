#!/usr/bin/env python3
"""
Tests for VTK output writer (src/io/output.py).

Validates:
1. VTK file generation
2. Correct file format
3. Scalar and vector field output
4. Time series output
"""

import os
import sys
import numpy as np
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.io.output import write_vtk, write_vtk_series, VTKWriter


class VTKOutputChecker:
    """Test suite for VTK output."""
    
    def __init__(self):
        self.NI = 10
        self.NJ = 5
        self.results = []
        self.temp_dir = None
    
    def setup(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown(self):
        """Clean up temporary directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def setup_grid(self):
        """Create a simple test grid."""
        x = np.linspace(0, 2, self.NI + 1)
        y = np.linspace(0, 1, self.NJ + 1)
        X, Y = np.meshgrid(x, y, indexing='ij')
        return X, Y
    
    def setup_solution(self, with_ghosts=False):
        """Create a test solution."""
        if with_ghosts:
            Q = np.zeros((self.NI + 2, self.NJ + 3, 4))
            Q[1:-1, 2:-1, 0] = 1.0  # p
            Q[1:-1, 2:-1, 1] = 0.5  # u
            Q[1:-1, 2:-1, 2] = 0.1  # v
            Q[1:-1, 2:-1, 3] = 1e-5  # nu_t
        else:
            Q = np.zeros((self.NI, self.NJ, 4))
            Q[:, :, 0] = 1.0
            Q[:, :, 1] = 0.5
            Q[:, :, 2] = 0.1
            Q[:, :, 3] = 1e-5
        return Q
    
    def run_test(self, name, test_func):
        """Run a test and record result."""
        try:
            passed, message = test_func()
            self.results.append((name, passed, message))
        except Exception as e:
            self.results.append((name, False, f"Exception: {e}"))
    
    # ===== Basic Output Tests =====
    
    def test_write_vtk_creates_file(self):
        """write_vtk should create a VTK file."""
        X, Y = self.setup_grid()
        Q = self.setup_solution()
        
        filename = os.path.join(self.temp_dir, "test.vtk")
        result = write_vtk(filename, X, Y, Q)
        
        if not os.path.exists(result):
            return False, "File not created"
        
        return True, f"Created {os.path.basename(result)}"
    
    def test_write_vtk_adds_extension(self):
        """write_vtk should add .vtk extension if missing."""
        X, Y = self.setup_grid()
        Q = self.setup_solution()
        
        filename = os.path.join(self.temp_dir, "test_no_ext")
        result = write_vtk(filename, X, Y, Q)
        
        if not result.endswith('.vtk'):
            return False, f"Extension not added: {result}"
        
        return True, "Extension added"
    
    def test_write_vtk_handles_ghost_cells(self):
        """write_vtk should handle Q with ghost cells."""
        X, Y = self.setup_grid()
        Q = self.setup_solution(with_ghosts=True)
        
        filename = os.path.join(self.temp_dir, "test_ghost.vtk")
        result = write_vtk(filename, X, Y, Q)
        
        if not os.path.exists(result):
            return False, "File not created with ghost cells"
        
        return True, "Ghost cells handled"
    
    # ===== File Format Tests =====
    
    def test_vtk_header(self):
        """VTK file should have correct header."""
        X, Y = self.setup_grid()
        Q = self.setup_solution()
        
        filename = os.path.join(self.temp_dir, "test_header.vtk")
        write_vtk(filename, X, Y, Q)
        
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        if not lines[0].startswith("# vtk DataFile"):
            return False, f"Wrong header: {lines[0]}"
        if "ASCII" not in lines[2]:
            return False, "Not ASCII format"
        if "STRUCTURED_GRID" not in lines[3]:
            return False, "Not structured grid"
        
        return True, "Header correct"
    
    def test_vtk_dimensions(self):
        """VTK file should have correct dimensions."""
        X, Y = self.setup_grid()
        Q = self.setup_solution()
        
        filename = os.path.join(self.temp_dir, "test_dims.vtk")
        write_vtk(filename, X, Y, Q)
        
        with open(filename, 'r') as f:
            content = f.read()
        
        expected_dims = f"DIMENSIONS {self.NI + 1} {self.NJ + 1} 1"
        if expected_dims not in content:
            return False, f"Wrong dimensions, expected: {expected_dims}"
        
        return True, f"Dimensions: {self.NI + 1} x {self.NJ + 1}"
    
    def test_vtk_point_count(self):
        """VTK file should have correct number of points."""
        X, Y = self.setup_grid()
        Q = self.setup_solution()
        
        filename = os.path.join(self.temp_dir, "test_points.vtk")
        write_vtk(filename, X, Y, Q)
        
        with open(filename, 'r') as f:
            content = f.read()
        
        expected_points = (self.NI + 1) * (self.NJ + 1)
        if f"POINTS {expected_points}" not in content:
            return False, f"Wrong point count"
        
        return True, f"{expected_points} points"
    
    def test_vtk_cell_data_count(self):
        """VTK file should have correct number of cells."""
        X, Y = self.setup_grid()
        Q = self.setup_solution()
        
        filename = os.path.join(self.temp_dir, "test_cells.vtk")
        write_vtk(filename, X, Y, Q)
        
        with open(filename, 'r') as f:
            content = f.read()
        
        expected_cells = self.NI * self.NJ
        if f"CELL_DATA {expected_cells}" not in content:
            return False, f"Wrong cell count"
        
        return True, f"{expected_cells} cells"
    
    # ===== Field Tests =====
    
    def test_vtk_contains_pressure(self):
        """VTK file should contain pressure field."""
        X, Y = self.setup_grid()
        Q = self.setup_solution()
        
        filename = os.path.join(self.temp_dir, "test_pressure.vtk")
        write_vtk(filename, X, Y, Q)
        
        with open(filename, 'r') as f:
            content = f.read()
        
        if "SCALARS Pressure" not in content:
            return False, "Pressure field missing"
        
        return True, "Pressure field present"
    
    def test_vtk_contains_velocity(self):
        """VTK file should contain velocity vector field."""
        X, Y = self.setup_grid()
        Q = self.setup_solution()
        
        filename = os.path.join(self.temp_dir, "test_velocity.vtk")
        write_vtk(filename, X, Y, Q)
        
        with open(filename, 'r') as f:
            content = f.read()
        
        if "VECTORS Velocity" not in content:
            return False, "Velocity field missing"
        
        return True, "Velocity field present"
    
    def test_vtk_contains_turbulent_viscosity(self):
        """VTK file should contain turbulent viscosity field."""
        X, Y = self.setup_grid()
        Q = self.setup_solution()
        
        filename = os.path.join(self.temp_dir, "test_nu_t.vtk")
        write_vtk(filename, X, Y, Q)
        
        with open(filename, 'r') as f:
            content = f.read()
        
        if "SCALARS TurbulentViscosity" not in content:
            return False, "TurbulentViscosity field missing"
        
        return True, "TurbulentViscosity field present"
    
    def test_vtk_contains_mach(self):
        """VTK file should contain Mach number field."""
        X, Y = self.setup_grid()
        Q = self.setup_solution()
        
        filename = os.path.join(self.temp_dir, "test_mach.vtk")
        write_vtk(filename, X, Y, Q)
        
        with open(filename, 'r') as f:
            content = f.read()
        
        if "SCALARS MachNumber" not in content:
            return False, "MachNumber field missing"
        
        return True, "MachNumber field present"
    
    # ===== Additional Fields Tests =====
    
    def test_additional_scalars(self):
        """write_vtk should handle additional scalar fields."""
        X, Y = self.setup_grid()
        Q = self.setup_solution()
        
        extra = {
            "Temperature": np.ones((self.NI, self.NJ)) * 300.0,
            "Density": np.ones((self.NI, self.NJ)) * 1.225
        }
        
        filename = os.path.join(self.temp_dir, "test_extra.vtk")
        write_vtk(filename, X, Y, Q, additional_scalars=extra)
        
        with open(filename, 'r') as f:
            content = f.read()
        
        if "SCALARS Temperature" not in content:
            return False, "Temperature field missing"
        if "SCALARS Density" not in content:
            return False, "Density field missing"
        
        return True, "Additional scalars written"
    
    # ===== Time Series Tests =====
    
    def test_vtk_series(self):
        """write_vtk_series should create multiple files and series file."""
        X, Y = self.setup_grid()
        
        solutions = {
            0: self.setup_solution(),
            10: self.setup_solution(),
            20: self.setup_solution()
        }
        
        base = os.path.join(self.temp_dir, "series")
        result = write_vtk_series(base, X, Y, solutions)
        
        # Check series file exists
        if not os.path.exists(result):
            return False, "Series file not created"
        
        # Check individual files exist
        for n in solutions.keys():
            vtk_file = f"{base}_{n:06d}.vtk"
            if not os.path.exists(vtk_file):
                return False, f"Missing: {os.path.basename(vtk_file)}"
        
        return True, f"Created {len(solutions)} files + series"
    
    def test_vtk_writer_class(self):
        """VTKWriter class should work for iterative writing."""
        X, Y = self.setup_grid()
        
        base = os.path.join(self.temp_dir, "writer_test")
        writer = VTKWriter(base, X, Y, beta=10.0)
        
        # Write a few iterations
        for n in [0, 5, 10]:
            Q = self.setup_solution()
            Q[:, :, 0] = n  # Set pressure to iteration number
            writer.write(Q, iteration=n)
        
        # Finalize
        series_file = writer.finalize()
        
        if not os.path.exists(series_file):
            return False, "Series file not created by VTKWriter"
        
        return True, "VTKWriter works"
    
    def run_all(self):
        """Run all tests."""
        self.setup()
        
        try:
            # Basic output tests
            self.run_test("creates_file", self.test_write_vtk_creates_file)
            self.run_test("adds_extension", self.test_write_vtk_adds_extension)
            self.run_test("handles_ghost_cells", self.test_write_vtk_handles_ghost_cells)
            
            # File format tests
            self.run_test("vtk_header", self.test_vtk_header)
            self.run_test("vtk_dimensions", self.test_vtk_dimensions)
            self.run_test("vtk_point_count", self.test_vtk_point_count)
            self.run_test("vtk_cell_data_count", self.test_vtk_cell_data_count)
            
            # Field tests
            self.run_test("contains_pressure", self.test_vtk_contains_pressure)
            self.run_test("contains_velocity", self.test_vtk_contains_velocity)
            self.run_test("contains_turbulent_viscosity", self.test_vtk_contains_turbulent_viscosity)
            self.run_test("contains_mach", self.test_vtk_contains_mach)
            
            # Additional fields
            self.run_test("additional_scalars", self.test_additional_scalars)
            
            # Time series
            self.run_test("vtk_series", self.test_vtk_series)
            self.run_test("vtk_writer_class", self.test_vtk_writer_class)
        
        finally:
            self.teardown()
    
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
    print("VTK Output Module Tests")
    print("=" * 60)
    
    tester = VTKOutputChecker()
    tester.run_all()
    success = tester.print_results()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

