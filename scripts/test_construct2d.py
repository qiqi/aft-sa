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
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.grid import Construct2DWrapper, GridOptions, StructuredGrid, check_grid_quality


def plot_grid(grid: StructuredGrid, name: str, output_dir: str = "."):
    """Plot and save grid visualization."""
    X, Y = grid.X, grid.Y
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Full domain view
    ax = axes[0]
    stride = max(1, X.shape[0] // 100)
    ax.plot(X[::stride, :].T, Y[::stride, :].T, 'b-', lw=0.3, alpha=0.6)
    ax.plot(X[:, ::stride], Y[:, ::stride], 'b-', lw=0.3, alpha=0.6)
    ax.plot(X[:, 0], Y[:, 0], 'r-', lw=1.5, label='Airfoil')
    ax.set_aspect('equal')
    ax.set_title(f'{name} - Full Domain')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Near-airfoil view
    ax = axes[1]
    # Find indices near airfoil (first ~20 j-levels)
    j_max = min(20, X.shape[1])
    stride_i = max(1, X.shape[0] // 150)
    for j in range(j_max):
        ax.plot(X[::stride_i, j], Y[::stride_i, j], 'b-', lw=0.4, alpha=0.7)
    for i in range(0, X.shape[0], stride_i):
        ax.plot(X[i, :j_max], Y[i, :j_max], 'b-', lw=0.4, alpha=0.7)
    ax.plot(X[:, 0], Y[:, 0], 'r-', lw=2, label='Airfoil')
    ax.set_aspect('equal')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.15, 0.15)
    ax.set_title(f'{name} - Near Wall (Boundary Layer Region)')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'grid_{name}.pdf')
    plt.savefig(output_path, dpi=150)
    print(f"  Saved: {output_path}")
    plt.close()


def test_airfoil(wrapper: Construct2DWrapper, airfoil_file: str, name: str, 
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
    metrics = grid.compute_metrics(wall_j=0)
    
    # Check quality
    quality = check_grid_quality(X, Y)
    
    print(f"\n  Grid Statistics:")
    print(f"    Dimensions:       {grid.ni} x {grid.nj}")
    print(f"    Total cells:      {grid.n_cells:,}")
    print(f"    First cell height: {metrics.wall_distance[:, 1].min():.2e}")
    print(f"    Min Jacobian:     {quality['min_jacobian']:.4e}")
    print(f"    Max skew angle:   {quality['max_skew_angle']:.2f} deg")
    print(f"    Max aspect ratio: {quality['max_aspect_ratio']:.1f}")
    
    # Check for negative Jacobians (inverted cells)
    if quality['min_jacobian'] < 0:
        print(f"  WARNING: Grid has inverted cells (negative Jacobian)!")
    
    # Save grid
    grid_file = os.path.join(output_dir, f'{name}.p3d')
    grid.write_plot3d(grid_file)
    print(f"  Saved grid: {grid_file}")
    
    # Plot
    plot_grid(grid, name, output_dir)
    
    return True


def main():
    """Main test function."""
    print("="*60)
    print("Construct2D Wrapper Test Suite")
    print("="*60)
    
    # Setup paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bin_path = os.path.join(project_root, "bin", "construct2d")
    data_dir = os.path.join(project_root, "data")
    output_dir = os.path.join(project_root, "cases", "grids")
    
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
    # Grid options for transition-capable grids (fine near wall)
    options = GridOptions(
        n_surface=300,          # Points on airfoil surface
        n_normal=100,           # Points in wall-normal direction
        y_plus=0.5,             # Target y+ (< 1 for resolving viscous sublayer)
        reynolds=1e6,           # Reynolds number for y+ calculation
        topology='CGRD',        # C-grid for wake resolution
        farfield_radius=25.0,   # Farfield distance in chord lengths
        n_wake=60,              # Points along wake
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
        
        results[name] = test_airfoil(wrapper, airfoil_file, name, options, output_dir)
    
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
    
    if n_passed == n_total:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
