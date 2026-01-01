#!/usr/bin/env python
"""
Visual verification of multigrid transfer operators.

Generates a PDF showing:
1. RESTRICTION TEST: Fine vs coarse Q fields
2. RESTRICTION RESIDUAL TEST: Fine vs coarse residuals with sum comparison
3. PROLONGATION TEST: Coarse correction propagated to fine
4. ROUND-TRIP TEST: Original -> Restrict -> Prolongate -> Result

Output: output/tests/transfers_visual.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.grid.metrics import MetricComputer
from src.grid.coarsening import Coarsener
from src.numerics.multigrid import (
    restrict_state,
    restrict_residual,
    prolongate_correction,
    compute_integral,
    compute_residual_sum,
)


def create_polar_grid(NI: int, NJ: int, r_inner: float = 0.2, r_outer: float = 1.0):
    """
    Create a polar grid (like C-grid topology around airfoil).
    
    This is more representative of actual CFD grids than a Cartesian grid.
    """
    # Radial direction (j: from inner to outer)
    r = np.linspace(r_inner, r_outer, NJ + 1)
    
    # Angular direction (i: wraps around)
    theta = np.linspace(0, 2 * np.pi, NI + 1)
    
    R, THETA = np.meshgrid(r, theta, indexing='ij')
    
    # Swap to get (i, j) ordering where i is angular
    R = R.T
    THETA = THETA.T
    
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)
    
    return X, Y


def create_distorted_grid(NI: int, NJ: int, amplitude: float = 0.1):
    """Create a Cartesian grid with sinusoidal distortion (fallback)."""
    x = np.linspace(0, 1, NI + 1)
    y = np.linspace(0, 1, NJ + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    dx = 1.0 / NI
    dy = 1.0 / NJ
    
    X_dist = X + amplitude * dx * np.sin(2 * np.pi * Y)
    Y_dist = Y + amplitude * dy * np.sin(2 * np.pi * X)
    
    return X_dist, Y_dist


def main():
    """Generate visual verification PDF."""
    print("Generating multigrid transfer visualization...")
    
    # Create non-Cartesian aligned grid (polar grid like C-grid around airfoil)
    NI_f, NJ_f = 32, 24
    nvar = 4
    
    X, Y = create_polar_grid(NI_f, NJ_f, r_inner=0.2, r_outer=1.0)
    computer = MetricComputer(X, Y)
    metrics_f = computer.compute()
    metrics_c = Coarsener.coarsen(metrics_f)
    
    NI_c, NJ_c = metrics_c.NI, metrics_c.NJ
    
    output_path = Path(__file__).parent.parent.parent / 'output' / 'tests' / 'transfers_visual.pdf'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with PdfPages(output_path) as pdf:
        # =================================================================
        # Page 1: State Restriction Test
        # =================================================================
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        fig.suptitle('State Restriction: Volume-Weighted Average', fontsize=14)
        
        # Create smooth field (Gaussian)
        Q_f = np.zeros((NI_f, NJ_f, nvar))
        for k in range(nvar):
            Q_f[:, :, k] = np.exp(-((metrics_f.xc - 0.5)**2 + 
                                     (metrics_f.yc - 0.3 * k)**2) / 0.08)
        
        # Restrict
        Q_c = np.zeros((NI_c, NJ_c, nvar))
        restrict_state(Q_f, metrics_f.volume, Q_c, metrics_c.volume)
        
        # Compute conservation
        integral_f = compute_integral(Q_f, metrics_f.volume)
        integral_c = compute_integral(Q_c, metrics_c.volume)
        
        # Plot first variable
        k = 0
        vmin, vmax = Q_f[:, :, k].min(), Q_f[:, :, k].max()
        
        im1 = axes[0, 0].pcolormesh(metrics_f.xc, metrics_f.yc, Q_f[:, :, k],
                                      shading='auto', cmap='viridis', 
                                      vmin=vmin, vmax=vmax)
        axes[0, 0].set_title(f'Fine Grid: {NI_f}×{NJ_f}')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        axes[0, 0].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].pcolormesh(metrics_c.xc, metrics_c.yc, Q_c[:, :, k],
                                      shading='auto', cmap='viridis',
                                      vmin=vmin, vmax=vmax)
        axes[0, 1].set_title(f'Coarse Grid: {NI_c}×{NJ_c}')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')
        axes[0, 1].set_aspect('equal')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Conservation bar chart
        x_pos = np.arange(nvar)
        width = 0.35
        
        axes[0, 2].bar(x_pos - width/2, integral_f, width, label='Fine', alpha=0.8)
        axes[0, 2].bar(x_pos + width/2, integral_c, width, label='Coarse', alpha=0.8)
        axes[0, 2].set_xlabel('Variable')
        axes[0, 2].set_ylabel('Volume Integral')
        axes[0, 2].set_title('Conservation Check')
        axes[0, 2].legend()
        axes[0, 2].set_xticks(x_pos)
        axes[0, 2].set_xticklabels([f'Q{i}' for i in range(nvar)])
        
        # Difference plot
        # Interpolate coarse to fine for comparison (simple)
        Q_c_interp = np.zeros((NI_f, NJ_f))
        for i_f in range(NI_f):
            for j_f in range(NJ_f):
                i_c = i_f // 2
                j_c = j_f // 2
                Q_c_interp[i_f, j_f] = Q_c[i_c, j_c, k]
        
        diff = Q_f[:, :, k] - Q_c_interp
        
        im3 = axes[1, 0].pcolormesh(metrics_f.xc, metrics_f.yc, diff,
                                      shading='auto', cmap='RdBu_r')
        axes[1, 0].set_title('Fine - Coarse(interpolated)')
        axes[1, 0].set_aspect('equal')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Conservation error
        rel_error = np.abs(integral_f - integral_c) / (np.abs(integral_f) + 1e-10)
        
        axes[1, 1].bar(x_pos, rel_error * 100)
        axes[1, 1].set_xlabel('Variable')
        axes[1, 1].set_ylabel('Relative Error (%)')
        axes[1, 1].set_title('Conservation Error')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels([f'Q{i}' for i in range(nvar)])
        axes[1, 1].set_ylim(0, max(1e-10, rel_error.max() * 100 * 1.5))
        
        # Text summary
        axes[1, 2].axis('off')
        summary = f"""State Restriction Summary
        
Fine Grid: {NI_f} × {NJ_f} = {NI_f * NJ_f} cells
Coarse Grid: {NI_c} × {NJ_c} = {NI_c * NJ_c} cells
Coarsening Ratio: 4:1

Conservation (sum Q*V):
  Max relative error: {rel_error.max():.2e}
  
Status: {'PASS ✓' if rel_error.max() < 1e-10 else 'FAIL ✗'}
"""
        axes[1, 2].text(0.1, 0.5, summary, transform=axes[1, 2].transAxes,
                        fontsize=11, verticalalignment='center', family='monospace')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
        # =================================================================
        # Page 2: Residual Restriction Test
        # =================================================================
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Residual Restriction: Simple Summation', fontsize=14)
        
        # Create checkerboard residual pattern
        R_f = np.zeros((NI_f, NJ_f, nvar))
        for i in range(NI_f):
            for j in range(NJ_f):
                sign = (-1) ** (i + j)
                for k in range(nvar):
                    R_f[i, j, k] = sign * (k + 1) * 0.1
        
        # Also add smooth component
        R_f[:, :, 0] += np.sin(2 * np.pi * metrics_f.xc) * np.cos(2 * np.pi * metrics_f.yc)
        
        # Restrict
        R_c = np.zeros((NI_c, NJ_c, nvar))
        restrict_residual(R_f, R_c)
        
        sum_f = compute_residual_sum(R_f)
        sum_c = compute_residual_sum(R_c)
        
        # Plot
        im1 = axes[0, 0].pcolormesh(metrics_f.xc, metrics_f.yc, R_f[:, :, 0],
                                      shading='auto', cmap='RdBu_r')
        axes[0, 0].set_title('Fine Residual (k=0)')
        axes[0, 0].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].pcolormesh(metrics_c.xc, metrics_c.yc, R_c[:, :, 0],
                                      shading='auto', cmap='RdBu_r')
        axes[0, 1].set_title('Coarse Residual (k=0)')
        axes[0, 1].set_aspect('equal')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Sum comparison
        width = 0.35
        axes[1, 0].bar(x_pos - width/2, sum_f, width, label='Fine Σ R_f', alpha=0.8)
        axes[1, 0].bar(x_pos + width/2, sum_c, width, label='Coarse Σ R_c', alpha=0.8)
        axes[1, 0].set_xlabel('Variable')
        axes[1, 0].set_ylabel('Sum of Residuals')
        axes[1, 0].set_title('Residual Sum Conservation')
        axes[1, 0].legend()
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels([f'R{i}' for i in range(nvar)])
        
        # Error
        res_error = np.abs(sum_f - sum_c) / (np.abs(sum_f) + 1e-10)
        
        axes[1, 1].axis('off')
        summary = f"""Residual Restriction Summary
        
Sum Conservation:
  Fine:   {sum_f}
  Coarse: {sum_c}
  
Max relative error: {res_error.max():.2e}

Note: Checkerboard component cancels in coarse
(expected since sum of alternating ±1 = 0)

Status: {'PASS ✓' if res_error.max() < 1e-10 else 'FAIL ✗'}
"""
        axes[1, 1].text(0.1, 0.5, summary, transform=axes[1, 1].transAxes,
                        fontsize=11, verticalalignment='center', family='monospace')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
        # =================================================================
        # Page 3: Prolongation Test
        # =================================================================
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        fig.suptitle('Correction Prolongation: Bilinear Interpolation', fontsize=14)
        
        # Initial fine state
        Q_f_init = np.zeros((NI_f, NJ_f, nvar))
        
        # Coarse correction: bump at center
        Q_c_old = np.zeros((NI_c, NJ_c, nvar))
        Q_c_new = np.zeros((NI_c, NJ_c, nvar))
        
        # Create smooth bump correction
        for k in range(nvar):
            Q_c_new[:, :, k] = np.exp(-((metrics_c.xc - 0.5)**2 + 
                                         (metrics_c.yc - 0.5)**2) / 0.1) * (k + 1)
        
        dQ_c = Q_c_new - Q_c_old
        
        # Apply prolongation
        Q_f_corrected = Q_f_init.copy()
        prolongate_correction(Q_f_corrected, Q_c_new, Q_c_old)
        
        # Plot coarse correction
        k = 0
        vmax_c = np.abs(dQ_c[:, :, k]).max()
        
        im1 = axes[0, 0].pcolormesh(metrics_c.xc, metrics_c.yc, dQ_c[:, :, k],
                                      shading='auto', cmap='viridis')
        axes[0, 0].set_title('Coarse Correction ΔQ_c')
        axes[0, 0].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Plot fine grid after prolongation
        im2 = axes[0, 1].pcolormesh(metrics_f.xc, metrics_f.yc, Q_f_corrected[:, :, k],
                                      shading='auto', cmap='viridis')
        axes[0, 1].set_title('Fine Grid After Prolongation')
        axes[0, 1].set_aspect('equal')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Cross-section
        j_mid_f = NJ_f // 2
        j_mid_c = NJ_c // 2
        
        x_f = metrics_f.xc[:, j_mid_f]
        x_c = metrics_c.xc[:, j_mid_c]
        
        axes[0, 2].plot(x_c, dQ_c[:, j_mid_c, k], 'o-', label='Coarse ΔQ', markersize=8)
        axes[0, 2].plot(x_f, Q_f_corrected[:, j_mid_f, k], '.-', label='Fine (prolongated)')
        axes[0, 2].set_xlabel('x')
        axes[0, 2].set_ylabel('Q')
        axes[0, 2].set_title(f'Cross-section at j={j_mid_f}')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Multiple variable comparison
        for k in range(min(3, nvar)):
            axes[1, k].plot(x_c, dQ_c[:, j_mid_c, k], 'o-', 
                           label=f'Coarse ΔQ[{k}]', markersize=6)
            axes[1, k].plot(x_f, Q_f_corrected[:, j_mid_f, k], '.-', 
                           label=f'Fine Q[{k}]', alpha=0.7)
            axes[1, k].set_xlabel('x')
            axes[1, k].set_ylabel('Value')
            axes[1, k].set_title(f'Variable {k}')
            axes[1, k].legend()
            axes[1, k].grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
        # =================================================================
        # Page 4: Round-Trip Test
        # =================================================================
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Round-Trip Test: Restrict → Modify → Prolongate', fontsize=14)
        
        # Original smooth field
        Q_f_orig = np.zeros((NI_f, NJ_f, nvar))
        for k in range(nvar):
            Q_f_orig[:, :, k] = np.sin(np.pi * metrics_f.xc) * np.sin(np.pi * metrics_f.yc)
        
        # Restrict to coarse
        Q_c = np.zeros((NI_c, NJ_c, nvar))
        restrict_state(Q_f_orig, metrics_f.volume, Q_c, metrics_c.volume)
        
        # "Solve" on coarse: add small correction
        Q_c_solved = Q_c + 0.1 * Q_c
        
        # Prolongate correction back
        Q_f_final = Q_f_orig.copy()
        prolongate_correction(Q_f_final, Q_c_solved, Q_c)
        
        # Expected change: ~0.1 * Q_f_orig (roughly)
        change = Q_f_final - Q_f_orig
        
        k = 0
        
        im1 = axes[0, 0].pcolormesh(metrics_f.xc, metrics_f.yc, Q_f_orig[:, :, k],
                                      shading='auto', cmap='viridis')
        axes[0, 0].set_title('Original Fine Q')
        axes[0, 0].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].pcolormesh(metrics_f.xc, metrics_f.yc, Q_f_final[:, :, k],
                                      shading='auto', cmap='viridis')
        axes[0, 1].set_title('Final Fine Q (after round-trip)')
        axes[0, 1].set_aspect('equal')
        plt.colorbar(im2, ax=axes[0, 1])
        
        im3 = axes[1, 0].pcolormesh(metrics_f.xc, metrics_f.yc, change[:, :, k],
                                      shading='auto', cmap='RdBu_r')
        axes[1, 0].set_title('Change (Final - Original)')
        axes[1, 0].set_aspect('equal')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Cross-section comparison
        j_mid = NJ_f // 2
        axes[1, 1].plot(metrics_f.xc[:, j_mid], Q_f_orig[:, j_mid, k], 
                        label='Original', linewidth=2)
        axes[1, 1].plot(metrics_f.xc[:, j_mid], Q_f_final[:, j_mid, k], 
                        '--', label='After round-trip', linewidth=2)
        axes[1, 1].plot(metrics_f.xc[:, j_mid], change[:, j_mid, k], 
                        ':', label='Change', linewidth=2)
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('Q')
        axes[1, 1].set_title(f'Cross-section at j={j_mid}')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    print(f"  Saved to: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()

