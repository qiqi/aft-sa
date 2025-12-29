#!/usr/bin/env python3
"""
Visual Test: Ghost Cell Boundary Conditions

This test visualizes how boundary conditions extend or reflect the interior
solution into the ghost cells.

Tests:
1. Wall BC (No-slip): Ghost velocity should mirror interior to make u=0 at face
2. Wake Cut BC (Periodic): Ghost should copy from opposite end

Expected behavior:
- Wall profile: u crosses zero exactly at the wall face (j=0.5)
- Wake cut: Ghost at i=0 matches interior at i=NI

Output: output/solver/test_bc_visual.pdf
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.solvers.boundary_conditions import BoundaryConditions, FreestreamConditions, initialize_state


def test_bc_visual():
    """Generate visual verification of boundary conditions."""
    
    NI, NJ = 20, 10
    fs = FreestreamConditions(p_inf=0, u_inf=10, v_inf=0, nu_t_inf=0)
    Q = initialize_state(NI, NJ, fs)
    
    # 1. Setup a Gradient for Velocity
    # Interior varies from u=10 (j=1) to u=20 (j=NJ)
    for j in range(1, NJ + 1):
        Q[:, j, 1] = 10.0 + j
    
    # Store original for reference
    Q_orig = Q.copy()
    
    # 2. Apply BCs
    bc = BoundaryConditions(freestream=fs)
    Q_bc = bc.apply(Q)
    
    # 3. Extract vertical profile at i=10
    u_profile = Q_bc[10, :, 1]
    j_indices = np.arange(NJ + 2) - 0.5  # Ghost at -0.5, first interior at 0.5, etc.
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ===== Plot 1: Wall BC Check =====
    ax1 = axes[0]
    
    # Plot U Velocity Profile
    ax1.plot(u_profile, j_indices, 'bo-', markersize=8, linewidth=2, label='U Velocity')
    ax1.axhline(0.0, color='r', linestyle='--', linewidth=2, label='Wall Face (j=0)')
    ax1.axhline(-0.5, color='gray', linestyle=':', alpha=0.5, label='Ghost Cell Center')
    ax1.axvline(0, color='k', linestyle='--', alpha=0.3)
    
    # Annotation
    u_ghost = u_profile[0]   # Index 0 is ghost
    u_first = u_profile[1]   # Index 1 is first interior
    u_face = 0.5 * (u_ghost + u_first)  # Value at wall face
    
    # Mark key points
    ax1.plot(u_ghost, -0.5, 'rs', markersize=12, label=f'Ghost: {u_ghost:.1f}')
    ax1.plot(u_first, 0.5, 'gs', markersize=12, label=f'Interior: {u_first:.1f}')
    ax1.plot(u_face, 0.0, 'mo', markersize=12, label=f'Face: {u_face:.1f}')
    
    ax1.set_title(f"Wall BC Check (No-Slip)\n"
                  f"Ghost={u_ghost:.1f}, Interior={u_first:.1f}, Face={u_face:.1f}\n"
                  f"Face should be ≈0")
    ax1.set_xlabel("U Velocity")
    ax1.set_ylabel("Grid Index J (0=Wall Face)")
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-15, 25)
    
    # Physics check annotation
    if abs(u_face) < 0.1:
        status = "✓ No-slip: u=0 at wall"
        color = 'green'
    else:
        status = f"✗ Slip condition! u={u_face:.1f} at wall"
        color = 'red'
    ax1.text(0.95, 0.05, status, transform=ax1.transAxes, fontsize=12,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    # ===== Plot 2: Wake Cut Periodic BC =====
    ax2 = axes[1]
    
    # Re-initialize and set a distinct spike at the wake cut interior
    Q2 = initialize_state(NI, NJ, fs)
    Q2[1:-1, 1:-1, 0] = 1.0  # Baseline pressure
    Q2[-2, :, 0] = 50.0  # Spike at right interior boundary (source)
    Q2[1, :, 0] = 10.0   # Different value at left interior (to check direction)
    
    Q2_bc = bc.apply(Q2)
    
    # Plot pressure along i at fixed j
    p_profile = Q2_bc[:, 5, 0]
    i_indices = np.arange(NI + 2) - 0.5  # Ghost at -0.5, first interior at 0.5
    
    ax2.plot(i_indices, p_profile, 'bo-', markersize=8, linewidth=2)
    
    # Highlight the ghost cells
    ax2.plot(-0.5, p_profile[0], 'rs', markersize=15, label=f'Left Ghost: {p_profile[0]:.0f}')
    ax2.plot(NI + 0.5, p_profile[-1], 'gs', markersize=15, label=f'Right Ghost: {p_profile[-1]:.0f}')
    
    # Mark the source cells
    ax2.plot(NI - 0.5, p_profile[-2], 'r^', markersize=12, label=f'Right Interior: {p_profile[-2]:.0f}')
    ax2.plot(0.5, p_profile[1], 'g^', markersize=12, label=f'Left Interior: {p_profile[1]:.0f}')
    
    # Draw arrows showing the copy direction
    ax2.annotate('', xy=(-0.5, 48), xytext=(NI - 0.5, 48),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax2.text(NI/2 - 1, 52, 'Copy: Right Int → Left Ghost', fontsize=10, color='red')
    
    ax2.annotate('', xy=(NI + 0.5, 12), xytext=(0.5, 12),
                 arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax2.text(NI/2 - 1, 15, 'Copy: Left Int → Right Ghost', fontsize=10, color='green')
    
    ax2.set_title("Wake Cut Periodic BC\n"
                  "Ghost cells copy from opposite interior")
    ax2.set_xlabel("Streamwise Index I")
    ax2.set_ylabel("Pressure")
    ax2.legend(loc='center right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Physics check
    left_ok = np.isclose(p_profile[0], p_profile[-2])  # Left ghost = Right interior
    right_ok = np.isclose(p_profile[-1], p_profile[1])  # Right ghost = Left interior
    
    if left_ok and right_ok:
        status = "✓ Periodic BC correct"
        color = 'green'
    else:
        status = "✗ Periodic BC broken!"
        color = 'red'
    ax2.text(0.95, 0.05, status, transform=ax2.transAxes, fontsize=12,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    plt.tight_layout()
    
    # Save output
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "output", "solver")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test_bc_visual.pdf")
    plt.savefig(output_path, dpi=150)
    print(f"Generated: {output_path}")
    
    # Summary
    print("\nBoundary Condition Verification:")
    print(f"  Wall BC:     Face u = {u_face:.2f} (should be ≈0)")
    print(f"  Wake Left:   Ghost = {p_profile[0]:.0f}, Source = {p_profile[-2]:.0f}")
    print(f"  Wake Right:  Ghost = {p_profile[-1]:.0f}, Source = {p_profile[1]:.0f}")
    
    plt.close()


if __name__ == "__main__":
    test_bc_visual()

