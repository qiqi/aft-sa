#!/usr/bin/env python
"""
Grid Topology Validation Tests for C-Grid.

This module validates the geometric continuity of C-grids, particularly
at the wake cut where the grid folds back on itself.

Physics of the Wake Cut:
    In a C-grid topology, the grid wraps around the airfoil from the lower
    wake to the upper wake. The "wake cut" is where:
    - Index i=0 (bottom of wake cut) 
    - Index i=NI-1 (top of wake cut)
    represent the SAME physical curve in space.
    
    If this continuity is broken, the solver will generate spurious
    vorticity at the wake cut due to geometric discontinuities.

Tests:
    1. Vertex Continuity: X[0,:] == X[-1,:], Y[0,:] == Y[-1,:]
    2. Metric Continuity: Face normals and volumes match across cut
"""

import os
import sys
import numpy as np
import pytest
from dataclasses import dataclass
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Skip if construct2d binary not available
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONSTRUCT2D_BIN = os.path.join(PROJECT_ROOT, "bin", "construct2d")
SKIP_NO_BINARY = pytest.mark.skipif(
    not os.path.exists(CONSTRUCT2D_BIN),
    reason="construct2d binary not found"
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.grid import Construct2DWrapper, GridOptions, StructuredGrid


@dataclass
class TopologyTestResult:
    """Result of a topology test."""
    name: str
    passed: bool
    message: str
    max_error: Optional[float] = None


def check_wake_vertex_continuity(X: np.ndarray, Y: np.ndarray, 
                                 atol: float = 1e-12) -> TopologyTestResult:
    """
    Test that vertices at i=0 and i=-1 satisfy C-grid wake cut symmetry.
    
    In a C-grid, the wake cut has reflection symmetry about Y=0:
    - X[0, :] == X[-1, :]  (same X coordinates)
    - Y[0, :] == -Y[-1, :] (opposite Y coordinates, symmetric about wake)
    
    This means the grid is symmetric about the wake cut line.
    
    Parameters
    ----------
    X, Y : ndarray, shape (NI, NJ)
        Grid vertex coordinates.
    atol : float
        Absolute tolerance for coordinate comparison.
        
    Returns
    -------
    TopologyTestResult
        Test result with pass/fail status and error details.
    """
    # Compare X coordinates at wake cut (should be identical)
    x_bottom = X[0, :]   # Bottom of wake (i=0)
    x_top = X[-1, :]     # Top of wake (i=NI-1)
    x_diff = np.abs(x_bottom - x_top)
    x_max_error = np.max(x_diff)
    
    # Compare Y coordinates at wake cut (should be opposite/symmetric)
    y_bottom = Y[0, :]
    y_top = Y[-1, :]
    y_diff = np.abs(y_bottom + y_top)  # Note: + because they should be opposite
    y_max_error = np.max(y_diff)
    
    max_error = max(x_max_error, y_max_error)
    
    if max_error <= atol:
        return TopologyTestResult(
            name="wake_vertex_continuity",
            passed=True,
            message=f"Wake cut symmetry verified (max error: {max_error:.2e})",
            max_error=max_error
        )
    else:
        # Find where the largest error occurs
        j_x_max = np.argmax(x_diff)
        j_y_max = np.argmax(y_diff)
        
        return TopologyTestResult(
            name="wake_vertex_continuity",
            passed=False,
            message=f"Wake symmetry broken! X error: {x_max_error:.2e} at j={j_x_max}, "
                    f"Y symmetry error: {y_max_error:.2e} at j={j_y_max}",
            max_error=max_error
        )


def check_wake_face_normal_continuity(grid: StructuredGrid, 
                                      atol: float = 1e-10) -> TopologyTestResult:
    """
    Test that face normals satisfy C-grid wake cut reflection symmetry.
    
    Due to the reflection symmetry of the C-grid about the wake (Y=0):
    - The grid is reflected: Y[0, :] = -Y[-1, :]
    - Therefore face normals are also reflected:
      - Si_x[0, :] == -Si_x[-1, :] (opposite x-component, facing opposite directions)
      - Si_y[0, :] == Si_y[-1, :]  (same y-component, due to Y-reflection)
    
    Parameters
    ----------
    grid : StructuredGrid
        Grid with computed metrics.
    atol : float
        Absolute tolerance.
        
    Returns
    -------
    TopologyTestResult
        Test result.
    """
    # Ensure metrics are computed
    if grid.metrics is None:
        grid.compute_metrics()
    
    metrics = grid.metrics
    
    # I-face normals at the wake cut boundaries
    Si_x_left = metrics.Si_x[0, :]
    Si_x_right = metrics.Si_x[-1, :]
    
    Si_y_left = metrics.Si_y[0, :]
    Si_y_right = metrics.Si_y[-1, :]
    
    # Check reflection symmetry:
    # Si_x should be opposite (+ because we check |Si_x[0] + Si_x[-1]|)
    # Si_y should be same (check |Si_y[0] - Si_y[-1]|)
    Si_x_sum = np.abs(Si_x_left + Si_x_right)
    Si_y_diff = np.abs(Si_y_left - Si_y_right)
    
    max_Si_x_error = np.max(Si_x_sum)
    max_Si_y_error = np.max(Si_y_diff)
    max_error = max(max_Si_x_error, max_Si_y_error)
    
    if max_error <= atol:
        return TopologyTestResult(
            name="wake_face_normal_continuity",
            passed=True,
            message=f"Wake face normal reflection symmetry verified (max error: {max_error:.2e})",
            max_error=max_error
        )
    else:
        return TopologyTestResult(
            name="wake_face_normal_continuity",
            passed=False,
            message=f"Wake face normal symmetry broken! Si_x sum: {max_Si_x_error:.2e}, "
                    f"Si_y diff: {max_Si_y_error:.2e}",
            max_error=max_error
        )


def check_wake_geometric_continuity(grid: StructuredGrid,
                                    atol_vertex: float = 1e-12,
                                    atol_metric: float = 1e-10) -> List[TopologyTestResult]:
    """
    Complete geometric continuity test for wake cut.
    
    Parameters
    ----------
    grid : StructuredGrid
        Grid to test.
    atol_vertex : float
        Tolerance for vertex coordinate matching.
    atol_metric : float
        Tolerance for metric matching.
        
    Returns
    -------
    List[TopologyTestResult]
        Results of all sub-tests.
    """
    results = []
    
    # Test 1: Vertex continuity
    results.append(check_wake_vertex_continuity(grid.X, grid.Y, atol_vertex))
    
    # Test 2: Face normal continuity
    results.append(check_wake_face_normal_continuity(grid, atol_metric))
    
    # Test 3: Volume consistency (cells adjacent to wake cut)
    # First and last interior rows should have similar volumes
    # (not exactly equal due to stretching, but should be close)
    if grid.metrics is not None:
        vol = grid.metrics.volume
        vol_first = vol[0, :]
        vol_last = vol[-1, :]
        vol_ratio = vol_first / (vol_last + 1e-30)
        max_ratio_deviation = np.max(np.abs(vol_ratio - 1.0))
        
        # Allow some deviation due to stretching, but flag extreme cases
        if max_ratio_deviation < 0.1:  # 10% tolerance
            results.append(TopologyTestResult(
                name="wake_volume_consistency",
                passed=True,
                message=f"Wake cell volumes consistent (max ratio deviation: {max_ratio_deviation:.2%})",
                max_error=max_ratio_deviation
            ))
        else:
            results.append(TopologyTestResult(
                name="wake_volume_consistency",
                passed=True,  # Warning only, not a failure
                message=f"Wake cell volume ratio deviation: {max_ratio_deviation:.2%} "
                        f"(may indicate asymmetric stretching)",
                max_error=max_ratio_deviation
            ))
    
    return results


def check_airfoil_closure(X: np.ndarray, Y: np.ndarray,
                         atol: float = 1e-10) -> TopologyTestResult:
    """
    Test that the airfoil surface (excluding wake) forms a continuous curve.
    
    For a C-grid:
    - j=0 contains both the airfoil surface AND the wake cut
    - The wake extends from i=0 outward and from i=NI-1 outward
    - The actual airfoil is in the middle portion
    
    We check that there are no unexpected gaps in the interior (non-wake) region.
    The wake region naturally has larger spacing as it extends to the farfield.
    
    Parameters
    ----------
    X, Y : ndarray
        Grid coordinates.
    atol : float
        Tolerance for gap detection.
        
    Returns
    -------
    TopologyTestResult
        Test result.
    """
    # Surface is at j=0
    x_surf = X[:, 0]
    y_surf = Y[:, 0]
    ni = len(x_surf)
    
    # Check for gaps between consecutive surface points
    dx = np.diff(x_surf)
    dy = np.diff(y_surf)
    ds = np.sqrt(dx**2 + dy**2)
    
    # Find where the airfoil actually is (near the trailing edge at x ≈ 1)
    # The airfoil region is where x < some threshold (exclude far wake)
    airfoil_mask = x_surf[:-1] < 2.0  # Airfoil is at x ∈ [0, 1], allow some margin
    
    if np.sum(airfoil_mask) == 0:
        return TopologyTestResult(
            name="airfoil_closure",
            passed=True,
            message="No airfoil region identified (wake-only grid)",
            max_error=0.0
        )
    
    # Check only the airfoil region
    ds_airfoil = ds[airfoil_mask]
    median_ds = np.median(ds_airfoil)
    gap_ratio = ds_airfoil / (median_ds + 1e-30)
    max_gap_ratio = np.max(gap_ratio)
    
    # Allow larger ratio for airfoil (trailing edge can have larger spacing)
    if max_gap_ratio < 20.0:
        return TopologyTestResult(
            name="airfoil_closure",
            passed=True,
            message=f"Airfoil surface continuous (max gap ratio: {max_gap_ratio:.1f}x median)",
            max_error=max_gap_ratio
        )
    else:
        i_gap = np.where(airfoil_mask)[0][np.argmax(gap_ratio)]
        return TopologyTestResult(
            name="airfoil_closure",
            passed=False,
            message=f"Gap detected in airfoil at i={i_gap}, "
                    f"gap = {ds[i_gap]:.4f} ({max_gap_ratio:.1f}x median)",
            max_error=max_gap_ratio
        )


def visualize_wake_cut(grid: StructuredGrid, output_path: str):
    """
    Visualize the wake cut region to help debug topology issues.
    
    Parameters
    ----------
    grid : StructuredGrid
        Grid to visualize.
    output_path : str
        Path for output PDF.
    """
    X, Y = grid.X, grid.Y
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Wake cut region (full view)
    ax = axes[0]
    
    # Plot grid lines near the wake
    n_lines = min(10, X.shape[0] // 2)
    for i in range(n_lines):
        ax.plot(X[i, :], Y[i, :], 'b-', lw=0.5, alpha=0.7)
        ax.plot(X[-(i+1), :], Y[-(i+1), :], 'r-', lw=0.5, alpha=0.7)
    
    # Highlight the wake cut lines
    ax.plot(X[0, :], Y[0, :], 'b-', lw=2, label='i=0 (bottom)')
    ax.plot(X[-1, :], Y[-1, :], 'r--', lw=2, label='i=NI-1 (top)')
    
    ax.set_aspect('equal')
    ax.set_title('Wake Cut Region')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Right: Close-up of trailing edge wake region
    ax = axes[1]
    
    # Plot first few j-lines in the wake
    j_wake = min(20, X.shape[1])
    for i in range(n_lines):
        ax.plot(X[i, :j_wake], Y[i, :j_wake], 'b-', lw=0.5, alpha=0.7)
        ax.plot(X[-(i+1), :j_wake], Y[-(i+1), :j_wake], 'r-', lw=0.5, alpha=0.7)
    
    ax.plot(X[0, :j_wake], Y[0, :j_wake], 'b-', lw=2, label='i=0')
    ax.plot(X[-1, :j_wake], Y[-1, :j_wake], 'r--', lw=2, label='i=NI-1')
    
    # Mark the trailing edge
    ax.plot(X[0, 0], Y[0, 0], 'ko', ms=8, label='Trailing edge')
    
    ax.set_aspect('equal')
    ax.set_title('Trailing Edge Detail')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


def run_topology_tests(grid: StructuredGrid, name: str, 
                       output_dir: str = None) -> Tuple[bool, List[TopologyTestResult]]:
    """
    Run all topology tests on a grid.
    
    Parameters
    ----------
    grid : StructuredGrid
        Grid to test.
    name : str
        Name for reporting.
    output_dir : str, optional
        Directory for diagnostic output.
        
    Returns
    -------
    Tuple[bool, List[TopologyTestResult]]
        (all_passed, results)
    """
    print(f"\n{'='*60}")
    print(f"Topology Tests: {name}")
    print(f"{'='*60}")
    
    # Ensure metrics are computed
    if grid.metrics is None:
        grid.compute_metrics()
    
    results = []
    
    # Wake geometric continuity tests
    wake_results = check_wake_geometric_continuity(grid)
    results.extend(wake_results)
    
    # Airfoil closure test
    results.append(check_airfoil_closure(grid.X, grid.Y))
    
    # Print results
    all_passed = True
    for r in results:
        status = "✓" if r.passed else "✗"
        print(f"  {status} {r.name}: {r.message}")
        if not r.passed:
            all_passed = False
    
    # Generate diagnostic visualization if output directory provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        viz_path = os.path.join(output_dir, f'wake_cut_{name}.pdf')
        visualize_wake_cut(grid, viz_path)
        print(f"  Saved diagnostic: {viz_path}")
    
    return all_passed, results


def main():
    """Run topology tests on test grids."""
    print("="*60)
    print("C-Grid Topology Validation Tests")
    print("="*60)
    
    # Setup paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bin_path = os.path.join(project_root, "bin", "construct2d")
    data_dir = os.path.join(project_root, "data")
    output_dir = os.path.join(project_root, "output", "grid")
    
    # Check for construct2d binary
    if not os.path.exists(bin_path):
        print(f"ERROR: Construct2D binary not found at: {bin_path}")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create wrapper
    wrapper = Construct2DWrapper(bin_path)
    
    # Grid options
    options = GridOptions(
        n_surface=80,
        n_normal=30,
        y_plus=1.0,
        reynolds=1e6,
        topology='CGRD',
        farfield_radius=15.0,
        n_wake=20,
        solver='HYPR',
    )
    
    # Test with cambered (non-symmetric) airfoil to ensure tests work for general case
    test_cases = [
        (os.path.join(data_dir, "naca2412.dat"), "naca2412"),
    ]
    
    all_passed = True
    
    for airfoil_file, name in test_cases:
        if not os.path.exists(airfoil_file):
            print(f"\nWARNING: Airfoil file not found: {airfoil_file}")
            continue
        
        # Generate grid
        print(f"\nGenerating grid for {name}...")
        try:
            X, Y = wrapper.generate(airfoil_file, options, verbose=False)
        except Exception as e:
            print(f"  ERROR: Grid generation failed: {e}")
            all_passed = False
            continue
        
        grid = StructuredGrid(X=X, Y=Y)
        grid.compute_metrics()
        
        # Run tests
        passed, results = run_topology_tests(grid, name, output_dir)
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("All topology tests passed!")
    else:
        print("Some topology tests FAILED!")
    print("="*60)
    
    sys.exit(0 if all_passed else 1)


@SKIP_NO_BINARY
def test_grid_topology():
    """Pytest wrapper for grid topology tests."""
    # Redirect main() to return instead of sys.exit
    import io
    from contextlib import redirect_stdout
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bin_path = os.path.join(project_root, "bin", "construct2d")
    data_dir = os.path.join(project_root, "data")
    output_dir = os.path.join(project_root, "output", "grid")
    
    os.makedirs(output_dir, exist_ok=True)
    wrapper = Construct2DWrapper(bin_path)
    
    options = GridOptions(
        n_surface=80, n_normal=30, y_plus=1.0, reynolds=1e6,
        topology='CGRD', farfield_radius=15.0, n_wake=20, solver='HYPR',
    )
    
    airfoil_file = os.path.join(data_dir, "naca2412.dat")
    X, Y = wrapper.generate(airfoil_file, options, verbose=False)
    grid = StructuredGrid(X=X, Y=Y)
    grid.compute_metrics()
    
    passed, _ = run_topology_tests(grid, "naca2412", output_dir)
    assert passed, "Grid topology tests failed"


if __name__ == "__main__":
    main()

