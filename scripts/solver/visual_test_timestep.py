#!/usr/bin/env python3
"""
Visual Test: Time Step Heatmap (LTS Verification)

This test verifies that Local Time Stepping (LTS) produces physically
sensible timestep distributions on a stretched grid.

Expected behavior:
- Small dt at "leading edge" region (dense i-spacing)
- Small dt near wall (dense j-spacing)  
- Large dt in farfield (coarse spacing)

If the plot looks uniform, the metric calculation is broken.

Output: output/solver/test_lts_map.pdf
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.solvers.time_stepping import compute_local_timestep


def test_lts_heatmap():
    """Generate timestep heatmap for stretched grid."""
    
    NI, NJ = 100, 50
    
    # 1. Create a "Fake" C-Grid (Stretched)
    # i runs around the airfoil, j runs normal to wall
    i = np.linspace(0, 1, NI)
    j = np.linspace(0, 1, NJ)
    I, J = np.meshgrid(i, j, indexing='ij')  # Shape: (NI, NJ)
    
    # Stretching function: Dense near i=0.5 (Leading Edge) and j=0 (Wall)
    # This simulates the tiny cells in a real boundary layer mesh
    dx = 0.01 + 0.5 * (np.abs(I - 0.5))  # Smallest at center (i=0.5)
    dy = 0.001 + 0.5 * J**2              # Smallest at wall (j=0)
    
    volume = dx * dy  # Shape: (NI, NJ)
    
    # Mock Metrics (assuming orthogonal for visualization)
    # Si: face normals in i-direction, shape (NI+1, NJ)
    # Sj: face normals in j-direction, shape (NI, NJ+1)
    
    # For i-faces: area ~ dy, normal in x-direction
    dy_faces = 0.5 * (np.vstack([dy[0:1, :], dy]) + np.vstack([dy, dy[-1:, :]]))
    dy_faces = dy_faces[:-1, :] + dy_faces[1:, :]  # Average to get (NI+1, NJ)
    # Simpler approach: just use dy at cell centers extended to faces
    Si_x = np.zeros((NI + 1, NJ))
    for ii in range(NI + 1):
        if ii == 0:
            Si_x[ii, :] = dy[0, :]
        elif ii == NI:
            Si_x[ii, :] = dy[-1, :]
        else:
            Si_x[ii, :] = 0.5 * (dy[ii-1, :] + dy[ii, :])
    Si_y = np.zeros((NI + 1, NJ))
    
    # For j-faces: area ~ dx, normal in y-direction
    Sj_x = np.zeros((NI, NJ + 1))
    Sj_y = np.zeros((NI, NJ + 1))
    for jj in range(NJ + 1):
        if jj == 0:
            Sj_y[:, jj] = dx[:, 0]
        elif jj == NJ:
            Sj_y[:, jj] = dx[:, -1]
        else:
            Sj_y[:, jj] = 0.5 * (dx[:, jj-1] + dx[:, jj])
    
    # Uniform Flow Q (with ghost cells)
    Q = np.zeros((NI + 2, NJ + 2, 4))
    Q[:, :, 1] = 1.0  # u = 1
    
    beta = 1.0
    
    # Compute DT
    dt = compute_local_timestep(Q, Si_x, Si_y, Sj_x, Sj_y, volume, beta)
    
    # 2. Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot DT Heatmap
    ax1 = axes[0]
    cf = ax1.contourf(I, J, np.log10(dt), levels=20, cmap='jet')
    plt.colorbar(cf, ax=ax1, label='log₁₀(Δt)')
    ax1.set_title("Time Step Distribution (Log Scale)\n"
                  "Should be blue at i=0.5 (LE) and j=0 (wall)")
    ax1.set_xlabel("Streamwise (i) - LE at center")
    ax1.set_ylabel("Wall-normal (j) - Wall at bottom")
    
    # Mark the expected small-dt regions
    ax1.axvline(0.5, color='white', linestyle='--', alpha=0.5, label='Leading Edge')
    ax1.axhline(0.0, color='white', linestyle='-', alpha=0.5, label='Wall')
    
    # Physics Check: DT Profiles
    ax2 = axes[1]
    ax2.plot(i, dt[:, 0], 'b-', linewidth=2, label='Wall (j=0)')
    ax2.plot(i, dt[:, NJ//2], 'g--', linewidth=2, label='Mid-field (j=NJ/2)')
    ax2.plot(i, dt[:, -1], 'r-.', linewidth=2, label='Farfield (j=max)')
    ax2.set_xlabel("Streamwise index i")
    ax2.set_ylabel("Δt")
    ax2.set_title("Time Step Profiles\n"
                  "Valley at i=0.5 (LE), smaller at wall")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    # Save output
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "output", "solver")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test_lts_map.pdf")
    plt.savefig(output_path, dpi=150)
    print(f"Generated: {output_path}")
    
    # Basic sanity check
    dt_le = dt[NI//2, 0]  # Leading edge, wall
    dt_ff = dt[0, -1]     # Farfield corner
    ratio = dt_ff / dt_le
    print(f"\nPhysics check:")
    print(f"  dt at LE/wall:     {dt_le:.2e}")
    print(f"  dt at farfield:    {dt_ff:.2e}")
    print(f"  Ratio (ff/le):     {ratio:.1f}x")
    
    if ratio < 10:
        print("  ⚠ WARNING: Ratio seems low. Check metric computation.")
    else:
        print("  ✓ Ratio looks reasonable (>10x)")
    
    plt.close()


if __name__ == "__main__":
    test_lts_heatmap()

