"""
Pytest tests for Construct2DWrapper grid generation.

Tests verify:
1. Grid generation works in isolated directory
2. No files are created outside working_dir
3. Output files exist in correct location
"""

import pytest
from pathlib import Path

# Skip all tests if construct2d binary not available
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONSTRUCT2D_BIN = PROJECT_ROOT / "bin" / "construct2d"
SKIP_NO_BINARY = pytest.mark.skipif(
    not CONSTRUCT2D_BIN.exists(),
    reason="construct2d binary not found"
)

from src.grid.mesher import Construct2DWrapper, GridOptions, Construct2DError


# Sample NACA 0012 coordinates (simplified)
NACA0012_COORDS = """NACA 0012
1.000000  0.001260
0.950000  0.011490
0.900000  0.020560
0.800000  0.035270
0.700000  0.046030
0.600000  0.053140
0.500000  0.056150
0.400000  0.054710
0.300000  0.048500
0.200000  0.037100
0.100000  0.021610
0.050000  0.013260
0.025000  0.008650
0.012500  0.005820
0.000000  0.000000
0.012500 -0.005820
0.025000 -0.008650
0.050000 -0.013260
0.100000 -0.021610
0.200000 -0.037100
0.300000 -0.048500
0.400000 -0.054710
0.500000 -0.056150
0.600000 -0.053140
0.700000 -0.046030
0.800000 -0.035270
0.900000 -0.020560
0.950000 -0.011490
1.000000 -0.001260
"""


@SKIP_NO_BINARY
class TestConstruct2DWrapper:
    """Tests for Construct2DWrapper with proper isolation."""
    
    def test_binary_path_is_absolute(self):
        """Wrapper should store absolute path to binary."""
        wrapper = Construct2DWrapper(str(CONSTRUCT2D_BIN))
        assert wrapper.binary_path.is_absolute()
        assert wrapper.binary_path == CONSTRUCT2D_BIN.resolve()
    
    def test_grid_generation_in_tmp_path(self, tmp_path):
        """
        Grid generation should work entirely within tmp_path.
        
        This tests:
        1. Input file is copied to working_dir
        2. Output .p3d file is created in working_dir
        3. No files are created in project root
        """
        # Create airfoil file in tmp_path
        airfoil_file = tmp_path / "test_airfoil.dat"
        airfoil_file.write_text(NACA0012_COORDS)
        
        # Get initial state of project root
        root_files_before = set(PROJECT_ROOT.glob("*.p3d"))
        root_files_before.update(PROJECT_ROOT.glob("*.nmf"))
        root_files_before.update(PROJECT_ROOT.glob("grid_options.in"))
        
        # Generate grid
        wrapper = Construct2DWrapper(str(CONSTRUCT2D_BIN))
        options = GridOptions(
            n_surface=40,  # Small for fast test
            n_normal=10,
            topology='CGRD',
        )
        
        X, Y = wrapper.generate(
            str(airfoil_file),
            options=options,
            working_dir=str(tmp_path),
            verbose=False
        )
        
        # Verify grid was generated
        assert X is not None
        assert Y is not None
        assert X.shape == Y.shape
        assert X.ndim == 2
        
        # Verify output files exist in tmp_path
        expected_p3d = tmp_path / "test_airfoil.p3d"
        assert expected_p3d.exists(), f"Expected {expected_p3d} to exist"
        
        # Verify no new files in project root
        root_files_after = set(PROJECT_ROOT.glob("*.p3d"))
        root_files_after.update(PROJECT_ROOT.glob("*.nmf"))
        root_files_after.update(PROJECT_ROOT.glob("grid_options.in"))
        
        new_files = root_files_after - root_files_before
        assert len(new_files) == 0, f"Files created in project root: {new_files}"
    
    def test_grid_dimensions(self, tmp_path):
        """Generated grid should have expected dimensions."""
        airfoil_file = tmp_path / "naca0012.dat"
        airfoil_file.write_text(NACA0012_COORDS)
        
        wrapper = Construct2DWrapper(str(CONSTRUCT2D_BIN))
        options = GridOptions(
            n_surface=60,
            n_normal=20,
            topology='CGRD',
        )
        
        X, Y = wrapper.generate(
            str(airfoil_file),
            options=options,
            working_dir=str(tmp_path),
            verbose=False
        )
        
        # For C-grid: ni = n_surface + n_wake, nj = n_normal
        # (actual ni depends on construct2d internals)
        assert X.shape[1] == options.n_normal
        assert Y.shape[1] == options.n_normal
    
    def test_auto_cleanup_temp_dir(self, tmp_path):
        """When working_dir is None, temp files should be cleaned up."""
        # Use airfoil from data/ if available
        data_airfoil = PROJECT_ROOT / "data" / "naca0012.dat"
        if not data_airfoil.exists():
            airfoil_file = tmp_path / "naca0012.dat"
            airfoil_file.write_text(NACA0012_COORDS)
            data_airfoil = airfoil_file
        
        wrapper = Construct2DWrapper(str(CONSTRUCT2D_BIN))
        options = GridOptions(n_surface=40, n_normal=10)
        
        # Generate with working_dir=None (uses temp dir, auto cleanup)
        X, Y = wrapper.generate(
            str(data_airfoil),
            options=options,
            working_dir=None,  # Auto temp dir
            keep_files=False,
            verbose=False
        )
        
        assert X is not None
        # Temp dir should be cleaned up (we can't easily verify this
        # without modifying internals, but at least verify no error)
    
    def test_missing_airfoil_raises_error(self, tmp_path):
        """Should raise Construct2DError for missing airfoil file."""
        wrapper = Construct2DWrapper(str(CONSTRUCT2D_BIN))
        
        with pytest.raises(Construct2DError, match="not found"):
            wrapper.generate(
                str(tmp_path / "nonexistent.dat"),
                working_dir=str(tmp_path)
            )


class TestConstruct2DWrapperNoBinary:
    """Tests that don't require the binary."""
    
    def test_missing_binary_raises_error(self, tmp_path):
        """Should raise Construct2DError for missing binary."""
        fake_binary = tmp_path / "fake_construct2d"
        
        with pytest.raises(Construct2DError, match="not found"):
            Construct2DWrapper(str(fake_binary))
    
    def test_non_executable_raises_error(self, tmp_path):
        """Should raise Construct2DError for non-executable file."""
        fake_binary = tmp_path / "fake_construct2d"
        fake_binary.write_text("#!/bin/bash\necho test")
        # Don't set executable permission
        
        with pytest.raises(Construct2DError, match="not executable"):
            Construct2DWrapper(str(fake_binary))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

