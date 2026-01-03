#!/usr/bin/env python
"""
Test script for Construct2D wrapper and grid generation.

Generates C-grids for common airfoils used in RANS transition model testing:
- NACA 0012: Symmetric, widely used benchmark
- NACA 2412: Cambered, general aviation
- NLF(1)-0416: Natural Laminar Flow airfoil for transition studies
"""

import os
import sys
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.grid import Construct2DWrapper, GridOptions, StructuredGrid, check_grid_quality

# Skip if construct2d binary not available
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONSTRUCT2D_BIN = os.path.join(PROJECT_ROOT, "bin", "construct2d")
SKIP_NO_BINARY = pytest.mark.skipif(
    not os.path.exists(CONSTRUCT2D_BIN),
    reason="construct2d binary not found"
)


def compute_cell_quality_metrics(X: np.ndarray, Y: np.ndarray) -> dict:
    """
    Compute cell-level quality metrics for visualization.
    
    Returns metrics defined at cell centers (ni-1, nj-1).
    """
    ni, nj = X.shape
    
    # Cell vertices
    # Cell (i,j) has corners: (i,j), (i+1,j), (i+1,j+1), (i,j+1)
    
    # Edge vectors
    # Bottom edge: (i+1,j) - (i,j)
    dx_bot = X[1:, :-1] - X[:-1, :-1]
    dy_bot = Y[1:, :-1] - Y[:-1, :-1]
    
    # Top edge: (i+1,j+1) - (i,j+1)
    dx_top = X[1:, 1:] - X[:-1, 1:]
    dy_top = Y[1:, 1:] - Y[:-1, 1:]
    
    # Left edge: (i,j+1) - (i,j)
    dx_left = X[:-1, 1:] - X[:-1, :-1]
    dy_left = Y[:-1, 1:] - Y[:-1, :-1]
    
    # Right edge: (i+1,j+1) - (i+1,j)
    dx_right = X[1:, 1:] - X[1:, :-1]
    dy_right = Y[1:, 1:] - Y[1:, :-1]
    
    # Edge lengths
    len_bot = np.sqrt(dx_bot**2 + dy_bot**2)
    len_top = np.sqrt(dx_top**2 + dy_top**2)
    len_left = np.sqrt(dx_left**2 + dy_left**2)
    len_right = np.sqrt(dx_right**2 + dy_right**2)
    
    # Cell area using cross product of diagonals
    # Diagonal 1: (i+1,j+1) - (i,j)
    dx_d1 = X[1:, 1:] - X[:-1, :-1]
    dy_d1 = Y[1:, 1:] - Y[:-1, :-1]
    
    # Diagonal 2: (i,j+1) - (i+1,j)
    dx_d2 = X[:-1, 1:] - X[1:, :-1]
    dy_d2 = Y[:-1, 1:] - Y[1:, :-1]
    
    # Area = 0.5 * |d1 x d2|
    area = 0.5 * np.abs(dx_d1 * dy_d2 - dy_d1 * dx_d2)
    
    # Aspect ratio: max edge / min edge
    max_edge = np.maximum(np.maximum(len_bot, len_top), np.maximum(len_left, len_right))
    min_edge = np.minimum(np.minimum(len_bot, len_top), np.minimum(len_left, len_right))
    aspect_ratio = max_edge / (min_edge + 1e-12)
    
    # Orthogonality at each corner (angle between edges)
    # At corner (i,j): angle between bottom and left edges
    def angle_between(dx1, dy1, dx2, dy2):
        """Compute angle in degrees between two vectors."""
        dot = dx1 * dx2 + dy1 * dy2
        cross = dx1 * dy2 - dy1 * dx2
        angle = np.abs(np.arctan2(np.abs(cross), dot))
        return angle * 180 / np.pi
    
    # Angles at each corner
    angle_bl = angle_between(dx_bot, dy_bot, dx_left, dy_left)     # bottom-left
    angle_br = angle_between(-dx_bot, -dy_bot, dx_right, dy_right)  # bottom-right
    angle_tl = angle_between(dx_top, dy_top, -dx_left, -dy_left)    # top-left
    angle_tr = angle_between(-dx_top, -dy_top, -dx_right, -dy_right)  # top-right
    
    # Skewness: deviation from 90 degrees (ideal = 0)
    skew_bl = np.abs(angle_bl - 90)
    skew_br = np.abs(angle_br - 90)
    skew_tl = np.abs(angle_tl - 90)
    skew_tr = np.abs(angle_tr - 90)
    
    # Maximum skewness in cell
    max_skew = np.maximum(np.maximum(skew_bl, skew_br), np.maximum(skew_tl, skew_tr))
    
    # Minimum angle (quality measure - should be > 0)
    min_angle = np.minimum(np.minimum(angle_bl, angle_br), np.minimum(angle_tl, angle_tr))
    
    return {
        'area': area,
        'aspect_ratio': aspect_ratio,
        'max_skew': max_skew,
        'min_angle': min_angle,
        'log_aspect': np.log10(aspect_ratio + 1),
    }


def plot_grid_quality(grid: StructuredGrid, name: str, output_dir: str = "."):
    """Plot grid colored by mesh quality metrics."""
    X, Y = grid.X, grid.Y
    
    # Compute quality metrics
    metrics = compute_cell_quality_metrics(X, Y)
    
    # Simplified: just show skew angle (most important quality metric)
    # Full domain + near-airfoil view
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    metric = metrics['max_skew']
    cmap, vmin, vmax = 'RdYlGn_r', 0, 45
    
    # Full domain view
    ax = axes[0]
    pc = ax.pcolormesh(X.T, Y.T, metric.T, cmap=cmap, vmin=vmin, vmax=vmax, 
                       shading='flat', rasterized=True)
    ax.plot(X[:, 0], Y[:, 0], 'k-', lw=1.5)
    ax.set_aspect('equal')
    ax.set_title(f'{name} - Max Skew Angle (°)')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    plt.colorbar(pc, ax=ax, shrink=0.8)
    
    # Near-airfoil view
    ax = axes[1]
    pc = ax.pcolormesh(X.T, Y.T, metric.T, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading='flat', rasterized=True)
    ax.plot(X[:, 0], Y[:, 0], 'k-', lw=2)
    ax.set_aspect('equal')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.2, 0.2)
    ax.set_title(f'{name} - Max Skew Angle (Near Airfoil)')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    plt.colorbar(pc, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'quality_{name}.pdf')
    plt.savefig(output_path, dpi=100)
    print(f"  Saved quality plot: {output_path}")
    plt.close()


def plot_grid_overview(grid: StructuredGrid, name: str, output_dir: str = "."):
    """Plot basic grid visualization."""
    X, Y = grid.X, grid.Y
    
    # Use stride for faster plotting (every 2nd line is sufficient for visualization)
    stride_i = max(1, X.shape[0] // 60)  # Aim for ~60 lines in i-direction
    stride_j = max(1, X.shape[1] // 30)  # Aim for ~30 lines in j-direction
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Full domain view
    ax = axes[0]
    ax.plot(X[::stride_i, :].T, Y[::stride_i, :].T, 'b-', lw=0.3, alpha=0.6)
    ax.plot(X[:, ::stride_j], Y[:, ::stride_j], 'b-', lw=0.3, alpha=0.6)
    ax.plot(X[:, 0], Y[:, 0], 'r-', lw=1.5, label='Airfoil')
    ax.set_aspect('equal')
    ax.set_title(f'{name} - Full Domain')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Near-airfoil view - full resolution near wall
    ax = axes[1]
    ax.plot(X.T, Y.T, 'b-', lw=0.3, alpha=0.6)
    ax.plot(X, Y, 'b-', lw=0.3, alpha=0.6)
    ax.plot(X[:, 0], Y[:, 0], 'r-', lw=2, label='Airfoil')
    ax.set_aspect('equal')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.2, 0.2)
    ax.set_title(f'{name} - Near Airfoil')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'grid_{name}.pdf')
    plt.savefig(output_path, dpi=100)
    print(f"  Saved grid plot: {output_path}")
    plt.close()


def run_airfoil_test(wrapper: Construct2DWrapper, airfoil_file: str, name: str, 
                     options: GridOptions, output_dir: str):
    """Generate and test grid for a single airfoil."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    # Generate grid
    try:
        X, Y = wrapper.generate(airfoil_file, options, verbose=True)
    except Exception as e:
        print(f"  ERROR: Grid generation failed: {e}")
        return False
    
    # Create StructuredGrid for analysis
    grid = StructuredGrid(X=X, Y=Y)
    
    # Compute metrics
    fvm_metrics = grid.compute_metrics(wall_j=0)
    
    # Check quality
    quality = check_grid_quality(X, Y)
    
    # Compute cell-level metrics for summary
    cell_metrics = compute_cell_quality_metrics(X, Y)
    
    print(f"\n  Grid Statistics:")
    print(f"    Dimensions:        {grid.ni} x {grid.nj}")
    print(f"    Total cells:       {grid.n_cells:,}")
    print(f"    First cell height: {fvm_metrics.wall_distance[:, 1].min():.2e}")
    print(f"    Min Jacobian:      {quality['min_jacobian']:.4e}")
    print(f"    Max skew angle:    {cell_metrics['max_skew'].max():.2f} deg")
    print(f"    Min corner angle:  {cell_metrics['min_angle'].min():.2f} deg")
    print(f"    Max aspect ratio:  {cell_metrics['aspect_ratio'].max():.1f}")
    print(f"    Mean aspect ratio: {cell_metrics['aspect_ratio'].mean():.1f}")
    
    # Check for negative Jacobians (inverted cells)
    if quality['min_jacobian'] < 0:
        print(f"  WARNING: Grid has inverted cells (negative Jacobian)!")
    
    # Check for poor quality cells
    n_high_skew = np.sum(cell_metrics['max_skew'] > 45)
    if n_high_skew > 0:
        print(f"  WARNING: {n_high_skew} cells with skew angle > 45°")
    
    # Save grid
    grid_file = os.path.join(output_dir, f'{name}.p3d')
    grid.write_plot3d(grid_file)
    print(f"  Saved grid: {grid_file}")
    
    # Plot grid overview
    plot_grid_overview(grid, name, output_dir)
    
    # Plot quality metrics
    plot_grid_quality(grid, name, output_dir)
    
    return True


def main():
    """Main test function."""
    print("="*60)
    print("Construct2D Wrapper Test Suite")
    print("="*60)
    
    # Setup paths (go up two levels: grid/ -> scripts/ -> project root)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bin_path = os.path.join(project_root, "bin", "construct2d")
    data_dir = os.path.join(project_root, "data")
    output_dir = os.path.join(project_root, "output", "grid")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check binary exists
    if not os.path.exists(bin_path):
        print(f"ERROR: Construct2D binary not found at: {bin_path}")
        print("Please ensure the binary is compiled and symlinked to bin/")
        sys.exit(1)
    
    # Initialize wrapper
    wrapper = Construct2DWrapper(bin_path)
    print(f"Binary: {bin_path}")
    print(f"Output: {output_dir}")
    
    # Define test cases
    # Grid options - smaller grid for fast testing (full production grids would use 300x100)
    options = GridOptions(
        n_surface=80,           # Points on airfoil surface (production: 300)
        n_normal=30,            # Points in wall-normal direction (production: 100)
        y_plus=1.0,             # Target y+ (relaxed for speed)
        reynolds=1e6,           # Reynolds number for y+ calculation
        topology='CGRD',        # C-grid for wake resolution
        farfield_radius=15.0,
        n_wake=20,              # Points along wake (production: 60)
        solver='HYPR',          # Hyperbolic grid generator
    )
    
    test_cases = [
        (os.path.join(data_dir, "naca0012.dat"), "naca0012"),
        (os.path.join(data_dir, "naca2412.dat"), "naca2412"),
        (os.path.join(data_dir, "nlf0416.dat"), "nlf0416"),
    ]
    
    # Run tests
    results = {}
    for airfoil_file, name in test_cases:
        if not os.path.exists(airfoil_file):
            print(f"\nWARNING: Airfoil file not found: {airfoil_file}")
            results[name] = False
            continue
        
        results[name] = run_airfoil_test(wrapper, airfoil_file, name, options, output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  {name}: {status}")
    
    n_passed = sum(results.values())
    n_total = len(results)
    print(f"\nTotal: {n_passed}/{n_total} passed")
    
    print(f"\nOutput files in: {output_dir}")
    print("  - grid_*.pdf: Grid structure plots")
    print("  - quality_*.pdf: Quality metric visualizations")
    print("  - *.p3d: Plot3D grid files")
    
    if n_passed == n_total:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed.")
        return 1


@pytest.mark.slow
@SKIP_NO_BINARY
def test_construct2d_grids():
    """
    Pytest wrapper for construct2d grid generation tests.
    
    Marked as slow because it generates 3 airfoil grids (~5s total).
    Run with: pytest -m "not slow" to skip.
    """
    result = main()
    assert result == 0, "Some grid generation tests failed"


if __name__ == "__main__":
    sys.exit(main())
