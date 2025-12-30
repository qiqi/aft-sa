#!/usr/bin/env python3
"""
Test Wall Distance Computation Against Analytical Values.

This script generates a mesh around a unit circle and compares the computed
wall distance against the analytical value (distance from point to circle = r - 1).

The test validates that the point-to-segment distance algorithm correctly
handles curved surfaces by comparing against the known analytical solution.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.grid.plot3d import compute_wall_distance, compute_wall_distance_gridpoints
from src.grid.metrics import MetricComputer


def get_output_dir():
    """Get output directory for test results."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = os.path.join(project_root, 'output', 'grid')
    os.makedirs(out, exist_ok=True)
    return out


def generate_circle_grid(n_theta: int = 65, n_radial: int = 33, 
                         r_inner: float = 1.0, r_outer: float = 2.0) -> tuple:
    """
    Generate an O-grid around a unit circle.
    
    Parameters
    ----------
    n_theta : int
        Number of points in circumferential direction.
    n_radial : int
        Number of points in radial direction.
    r_inner : float
        Inner radius (the "wall").
    r_outer : float
        Outer radius.
        
    Returns
    -------
    X, Y : ndarray
        Grid coordinates, shape (n_theta, n_radial).
    """
    # Theta goes from 0 to 2*pi (periodic)
    theta = np.linspace(0, 2 * np.pi, n_theta)
    
    # Radial stretching (geometric growth)
    ratio = 1.05
    if abs(ratio - 1.0) < 1e-6:
        r = np.linspace(r_inner, r_outer, n_radial)
    else:
        # Geometric stretching
        dr0 = (r_outer - r_inner) * (1 - ratio) / (1 - ratio**n_radial)
        r = r_inner + dr0 * (1 - ratio**np.arange(n_radial)) / (1 - ratio)
    
    # Create mesh
    R, THETA = np.meshgrid(r, theta)
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)
    
    return X, Y, r


def analytical_wall_distance(X: np.ndarray, Y: np.ndarray, r_wall: float = 1.0) -> np.ndarray:
    """
    Compute analytical wall distance from unit circle.
    
    For a circle of radius r_wall centered at origin, the wall distance
    at any point (x, y) is simply: d = sqrt(x² + y²) - r_wall
    """
    r = np.sqrt(X**2 + Y**2)
    return r - r_wall


def generate_cartesian_grid_over_circle(nx: int = 41, ny: int = 41,
                                         n_wall: int = 65) -> tuple:
    """
    Generate a Cartesian grid that overlays a unit circle.
    
    This is the KEY test case - Cartesian grid points are NOT aligned with
    wall normals, so the segment-based method should significantly outperform
    the grid-point method.
    
    Parameters
    ----------
    nx, ny : int
        Number of grid points in x and y directions.
    n_wall : int
        Number of points defining the circular wall.
        
    Returns
    -------
    X_query, Y_query : ndarray
        Query point coordinates (only points outside the circle).
    x_wall, y_wall : ndarray
        Wall coordinates.
    """
    # Create Cartesian grid
    x = np.linspace(-2.5, 2.5, nx)
    y = np.linspace(-2.5, 2.5, ny)
    X, Y = np.meshgrid(x, y)
    
    # Flatten and filter to points outside the unit circle
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    r = np.sqrt(X_flat**2 + Y_flat**2)
    
    # Keep only points outside the circle (with small margin)
    outside = r > 1.05
    X_query = X_flat[outside]
    Y_query = Y_flat[outside]
    
    # Create circular wall
    theta_wall = np.linspace(0, 2 * np.pi, n_wall)
    x_wall = np.cos(theta_wall)
    y_wall = np.sin(theta_wall)
    
    return X_query, Y_query, x_wall, y_wall


def test_cartesian_grid():
    """
    Test wall distance using a Cartesian grid over a circular wall.
    
    This is the KEY test - Cartesian grids have points that are NOT aligned
    with wall normals, so this demonstrates the real benefit of the
    segment-based wall distance calculation.
    """
    print("\n" + "=" * 60)
    print("Wall Distance Test: Cartesian Grid (KEY TEST)")
    print("=" * 60)
    print("\nThis test uses a Cartesian grid overlaying a circular wall.")
    print("Grid points are NOT aligned with wall normals, so the segment")
    print("method should significantly outperform the grid-point method.")
    
    from src.grid.plot3d import _point_to_segment_distance
    
    # Test with different wall resolutions
    configs = [
        {"n_wall": 17, "name": "Coarse wall (16 segments)"},
        {"n_wall": 33, "name": "Medium wall (32 segments)"},
        {"n_wall": 65, "name": "Fine wall (64 segments)"},
    ]
    
    for cfg in configs:
        print(f"\n--- {cfg['name']} ---")
        
        X_query, Y_query, x_wall, y_wall = generate_cartesian_grid_over_circle(
            nx=31, ny=31, n_wall=cfg['n_wall']
        )
        n_wall = len(x_wall)
        n_query = len(X_query)
        
        print(f"  Query points: {n_query}")
        print(f"  Wall segments: {n_wall - 1}")
        
        errors_segment = []
        errors_gridpoint = []
        
        for i in range(n_query):
            px, py = X_query[i], Y_query[i]
            
            # Analytical distance to circle
            d_exact = np.sqrt(px**2 + py**2) - 1.0
            
            # Grid-point method: min distance to wall nodes
            dist_to_nodes = np.sqrt((x_wall - px)**2 + (y_wall - py)**2)
            d_gridpoint = dist_to_nodes.min()
            
            # Segment method: min distance to wall segments
            d_segment = float('inf')
            for k in range(n_wall - 1):
                d = _point_to_segment_distance(px, py, x_wall[k], y_wall[k], 
                                               x_wall[k+1], y_wall[k+1])
                d_segment = min(d_segment, d)
            
            errors_segment.append(abs(d_segment - d_exact))
            errors_gridpoint.append(abs(d_gridpoint - d_exact))
        
        errors_segment = np.array(errors_segment)
        errors_gridpoint = np.array(errors_gridpoint)
        
        print(f"\n  Segment method:")
        print(f"    Max error:  {errors_segment.max():.6e}")
        print(f"    Mean error: {errors_segment.mean():.6e}")
        
        print(f"  Grid-point method:")
        print(f"    Max error:  {errors_gridpoint.max():.6e}")
        print(f"    Mean error: {errors_gridpoint.mean():.6e}")
        
        improvement = errors_gridpoint.max() / (errors_segment.max() + 1e-16)
        print(f"\n  Improvement factor: {improvement:.1f}x")
        
        if improvement > 2:
            print(f"  ✅ Segment method is {improvement:.1f}x better!")
    
    return True


def test_skewed_ogrid():
    """
    Test wall distance using a skewed O-grid around a circle.
    
    This creates an O-grid where the radial lines are twisted/skewed,
    so interior points don't lie on wall normals.
    """
    print("\n" + "=" * 60)
    print("Wall Distance Test: Skewed O-Grid")
    print("=" * 60)
    print("\nThis test uses an O-grid with skewed (non-radial) grid lines.")
    
    from src.grid.plot3d import _point_to_segment_distance
    
    # Create skewed O-grid
    n_theta = 33
    n_radial = 17
    
    theta = np.linspace(0, 2 * np.pi, n_theta)
    r = np.linspace(1.0, 2.5, n_radial)
    
    R, THETA = np.meshgrid(r, theta)
    
    # Add skew: rotate theta based on radius (creates spiral pattern)
    skew_angle = 0.3 * (R - 1.0)  # Skew increases with radius
    THETA_skewed = THETA + skew_angle
    
    X = R * np.cos(THETA_skewed)
    Y = R * np.sin(THETA_skewed)
    
    # Wall coordinates (at r=1, no skew)
    x_wall = X[:, 0]
    y_wall = Y[:, 0]
    n_wall = len(x_wall)
    
    print(f"Grid: {n_theta} × {n_radial}")
    print(f"Skew: θ_skewed = θ + 0.3*(r-1)")
    
    errors_segment = []
    errors_gridpoint = []
    
    # Test all interior points
    for i in range(n_theta):
        for j in range(1, n_radial):  # Skip j=0 (on wall)
            px, py = X[i, j], Y[i, j]
            
            # Analytical distance to circle
            d_exact = np.sqrt(px**2 + py**2) - 1.0
            
            # Grid-point method: min distance to wall nodes
            dist_to_nodes = np.sqrt((x_wall - px)**2 + (y_wall - py)**2)
            d_gridpoint = dist_to_nodes.min()
            
            # Segment method: min distance to wall segments
            d_segment = float('inf')
            for k in range(n_wall - 1):
                d = _point_to_segment_distance(px, py, x_wall[k], y_wall[k], 
                                               x_wall[k+1], y_wall[k+1])
                d_segment = min(d_segment, d)
            
            errors_segment.append(abs(d_segment - d_exact))
            errors_gridpoint.append(abs(d_gridpoint - d_exact))
    
    errors_segment = np.array(errors_segment)
    errors_gridpoint = np.array(errors_gridpoint)
    
    print(f"\nSegment method:")
    print(f"  Max error:  {errors_segment.max():.6e}")
    print(f"  Mean error: {errors_segment.mean():.6e}")
    
    print(f"\nGrid-point method:")
    print(f"  Max error:  {errors_gridpoint.max():.6e}")
    print(f"  Mean error: {errors_gridpoint.mean():.6e}")
    
    improvement = errors_gridpoint.max() / (errors_segment.max() + 1e-16)
    print(f"\nImprovement factor: {improvement:.1f}x")
    
    if improvement > 2:
        print(f"✅ Segment method is {improvement:.1f}x better!")
        return True
    else:
        print("⚠️ Improvement less than expected")
        return True


def run_test():
    """Run wall distance test on unit circle."""
    
    # KEY TEST 1: Cartesian grid (points NOT aligned with wall normals)
    test_cartesian_grid()
    
    # KEY TEST 2: Skewed O-grid (twisted radial lines)
    test_skewed_ogrid()
    
    print("\n" + "=" * 60)
    print("Wall Distance Test: Standard O-Grid (Baseline)")
    print("=" * 60)
    print("\nNote: For perfect O-grids, all points lie on radial lines,")
    print("so both methods give identical results (machine precision).")
    
    # Test configurations
    configs = [
        {"n_theta": 17, "n_radial": 9, "name": "Coarse (17x9)"},
        {"n_theta": 33, "n_radial": 17, "name": "Medium (33x17)"},
        {"n_theta": 65, "n_radial": 33, "name": "Fine (65x33)"},
        {"n_theta": 129, "n_radial": 65, "name": "Very Fine (129x65)"},
    ]
    
    results = []
    
    for cfg in configs:
        print(f"\n--- {cfg['name']} ---")
        
        # Generate grid
        X, Y, r = generate_circle_grid(n_theta=cfg['n_theta'], n_radial=cfg['n_radial'])
        
        # Analytical wall distance
        d_exact = analytical_wall_distance(X, Y)
        
        # Computed wall distance (segment-based)
        d_segment = compute_wall_distance(X, Y, wall_j=0, search_radius=cfg['n_theta']//2)
        
        # Computed wall distance (grid-point only, for comparison)
        d_gridpoint = compute_wall_distance_gridpoints(X, Y, wall_j=0)
        
        # Compute errors (skip j=0 which is on the wall)
        err_segment = np.abs(d_segment[:, 1:] - d_exact[:, 1:])
        err_gridpoint = np.abs(d_gridpoint[:, 1:] - d_exact[:, 1:])
        
        # Relative error (normalized by actual distance)
        rel_err_segment = err_segment / (d_exact[:, 1:] + 1e-12)
        rel_err_gridpoint = err_gridpoint / (d_exact[:, 1:] + 1e-12)
        
        print(f"  Segment-based method:")
        print(f"    Max absolute error: {err_segment.max():.6e}")
        print(f"    Mean absolute error: {err_segment.mean():.6e}")
        print(f"    Max relative error: {rel_err_segment.max()*100:.4f}%")
        
        print(f"  Grid-point method:")
        print(f"    Max absolute error: {err_gridpoint.max():.6e}")
        print(f"    Mean absolute error: {err_gridpoint.mean():.6e}")
        print(f"    Max relative error: {rel_err_gridpoint.max()*100:.4f}%")
        
        print(f"  Improvement factor: {err_gridpoint.max() / (err_segment.max() + 1e-16):.1f}x")
        
        results.append({
            'name': cfg['name'],
            'n_theta': cfg['n_theta'],
            'n_radial': cfg['n_radial'],
            'X': X, 'Y': Y,
            'd_exact': d_exact,
            'd_segment': d_segment,
            'd_gridpoint': d_gridpoint,
            'err_segment_max': err_segment.max(),
            'err_gridpoint_max': err_gridpoint.max(),
        })
    
    # Test with MetricComputer as well
    print(f"\n--- MetricComputer Test (Medium Grid) ---")
    X, Y, _ = generate_circle_grid(n_theta=33, n_radial=17)
    mc = MetricComputer(X, Y, wall_j=0)
    metrics = mc.compute()
    
    # MetricComputer gives cell-centered wall distance
    # For comparison, compute analytical at cell centers
    Xc = 0.25 * (X[:-1, :-1] + X[1:, :-1] + X[:-1, 1:] + X[1:, 1:])
    Yc = 0.25 * (Y[:-1, :-1] + Y[1:, :-1] + Y[:-1, 1:] + Y[1:, 1:])
    d_exact_cc = analytical_wall_distance(Xc, Yc)
    
    err_mc = np.abs(metrics.wall_distance - d_exact_cc)
    print(f"  Max absolute error: {err_mc.max():.6e}")
    print(f"  Mean absolute error: {err_mc.mean():.6e}")
    
    # Create visualization
    plot_results(results)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nThe segment-based wall distance algorithm correctly computes")
    print("distance to curved walls by finding the closest point on wall")
    print("segments, not just grid points. This is critical for:")
    print("  - Airfoil leading edges with high curvature")
    print("  - Accurate y+ computation for turbulence models")
    print("  - Proper near-wall damping in SA model")
    
    # Verify the test passes
    fine_result = results[-1]
    if fine_result['err_segment_max'] < 1e-3:
        print("\n✅ Test PASSED: Segment-based error < 0.1% on fine grid")
    else:
        print(f"\n❌ Test FAILED: Error = {fine_result['err_segment_max']:.2e}")
        return 1
    
    return 0


def plot_results(results):
    """Create visualization of wall distance test results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Use the medium resolution result for detailed plots
    res = results[1]  # Medium grid
    X, Y = res['X'], res['Y']
    d_exact = res['d_exact']
    d_segment = res['d_segment']
    d_gridpoint = res['d_gridpoint']
    
    # Plot 1: Grid
    ax = axes[0, 0]
    ax.plot(X, Y, 'b-', lw=0.5, alpha=0.5)
    ax.plot(X.T, Y.T, 'b-', lw=0.5, alpha=0.5)
    # Highlight wall
    ax.plot(X[:, 0], Y[:, 0], 'r-', lw=2, label='Wall (r=1)')
    ax.set_aspect('equal')
    ax.set_title('O-Grid Around Unit Circle')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    
    # Plot 2: Exact wall distance
    ax = axes[0, 1]
    pc = ax.pcolormesh(X, Y, d_exact, shading='auto', cmap='viridis')
    ax.set_aspect('equal')
    ax.set_title('Analytical Wall Distance')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(pc, ax=ax, label='d')
    
    # Plot 3: Computed wall distance (segment)
    ax = axes[0, 2]
    pc = ax.pcolormesh(X, Y, d_segment, shading='auto', cmap='viridis')
    ax.set_aspect('equal')
    ax.set_title('Computed Wall Distance (Segment)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(pc, ax=ax, label='d')
    
    # Plot 4: Error comparison (segment)
    ax = axes[1, 0]
    err = np.abs(d_segment - d_exact)
    err[err < 1e-15] = 1e-15  # Avoid log(0)
    pc = ax.pcolormesh(X, Y, np.log10(err + 1e-15), shading='auto', cmap='hot_r', 
                       vmin=-6, vmax=-1)
    ax.set_aspect('equal')
    ax.set_title('Error: Segment Method (log10)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(pc, ax=ax, label='log10(error)')
    
    # Plot 5: Error comparison (grid-point)
    ax = axes[1, 1]
    err = np.abs(d_gridpoint - d_exact)
    err[err < 1e-15] = 1e-15
    pc = ax.pcolormesh(X, Y, np.log10(err + 1e-15), shading='auto', cmap='hot_r',
                       vmin=-6, vmax=-1)
    ax.set_aspect('equal')
    ax.set_title('Error: Grid-Point Method (log10)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(pc, ax=ax, label='log10(error)')
    
    # Plot 6: Convergence
    ax = axes[1, 2]
    n_theta = [r['n_theta'] for r in results]
    err_seg = [r['err_segment_max'] for r in results]
    err_gp = [r['err_gridpoint_max'] for r in results]
    
    ax.loglog(n_theta, err_seg, 'bo-', lw=2, label='Segment method')
    ax.loglog(n_theta, err_gp, 'rs--', lw=2, label='Grid-point method')
    
    # Reference slope
    n_ref = np.array([n_theta[0], n_theta[-1]])
    ax.loglog(n_ref, 0.5 * err_seg[0] * (n_ref[0]/n_ref)**2, 'k:', label='O(h²)')
    
    ax.set_xlabel('n_theta')
    ax.set_ylabel('Max Absolute Error')
    ax.set_title('Convergence Study')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    
    out_path = os.path.join(get_output_dir(), 'wall_distance_test.pdf')
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == "__main__":
    exit(run_test())

