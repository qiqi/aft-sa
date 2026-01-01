#!/usr/bin/env python
"""
Visual verification of grid coarsening.

Generates a PDF showing:
1. Fine grid with cell boundaries and face normals as arrows
2. Coarse grid overlaid, showing how 2x2 blocks map to coarse cells
3. Zoom on a few cells showing face normal summation
4. Table of volume totals at each level

Output: output/tests/coarsening_visual.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.backends.backend_pdf import PdfPages
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.grid.metrics import MetricComputer
from src.grid.coarsening import Coarsener


def create_distorted_grid(NI: int, NJ: int, amplitude: float = 0.1):
    """Create a Cartesian grid with sinusoidal distortion."""
    x = np.linspace(0, 1, NI + 1)
    y = np.linspace(0, 1, NJ + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    dx = 1.0 / NI
    dy = 1.0 / NJ
    
    X_dist = X + amplitude * dx * np.sin(2 * np.pi * Y)
    Y_dist = Y + amplitude * dy * np.sin(2 * np.pi * X)
    
    return X_dist, Y_dist


def plot_grid_with_normals(ax, X, Y, metrics, title, show_normals=True, 
                            normal_scale=0.3, alpha=0.7):
    """Plot grid cells and face normals."""
    NI, NJ = metrics.NI, metrics.NJ
    
    # Plot grid lines
    for i in range(NI + 1):
        ax.plot(X[i, :], Y[i, :], 'k-', linewidth=0.5, alpha=0.5)
    for j in range(NJ + 1):
        ax.plot(X[:, j], Y[:, j], 'k-', linewidth=0.5, alpha=0.5)
    
    # Plot cell centers
    ax.scatter(metrics.xc.flatten(), metrics.yc.flatten(), 
               c='blue', s=10, alpha=0.5, zorder=5)
    
    if show_normals:
        # Plot I-face normals (skip boundary for clarity)
        for i in range(1, NI):
            for j in range(NJ):
                # Face midpoint
                xm = 0.5 * (X[i, j] + X[i, j+1])
                ym = 0.5 * (Y[i, j] + Y[i, j+1])
                
                # Normalized normal
                sx = metrics.Si_x[i, j]
                sy = metrics.Si_y[i, j]
                mag = np.sqrt(sx**2 + sy**2) + 1e-10
                
                ax.arrow(xm, ym, normal_scale * sx / mag * 0.5 / NI, 
                         normal_scale * sy / mag * 0.5 / NJ,
                         head_width=0.01, head_length=0.005, 
                         fc='red', ec='red', alpha=alpha)
        
        # Plot J-face normals
        for i in range(NI):
            for j in range(1, NJ):
                # Face midpoint
                xm = 0.5 * (X[i, j] + X[i+1, j])
                ym = 0.5 * (Y[i, j] + Y[i+1, j])
                
                # Normalized normal
                sx = metrics.Sj_x[i, j]
                sy = metrics.Sj_y[i, j]
                mag = np.sqrt(sx**2 + sy**2) + 1e-10
                
                ax.arrow(xm, ym, normal_scale * sx / mag * 0.5 / NI, 
                         normal_scale * sy / mag * 0.5 / NJ,
                         head_width=0.01, head_length=0.005, 
                         fc='green', ec='green', alpha=alpha)
    
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def plot_coarse_overlay(ax, X_f, Y_f, X_c, Y_c, metrics_f, metrics_c):
    """Plot fine grid with coarse grid overlay."""
    NI_f, NJ_f = metrics_f.NI, metrics_f.NJ
    NI_c, NJ_c = metrics_c.NI, metrics_c.NJ
    
    # Plot fine grid in light gray
    for i in range(NI_f + 1):
        ax.plot(X_f[i, :], Y_f[i, :], 'gray', linewidth=0.3, alpha=0.5)
    for j in range(NJ_f + 1):
        ax.plot(X_f[:, j], Y_f[:, j], 'gray', linewidth=0.3, alpha=0.5)
    
    # Highlight 2x2 blocks with alternating colors
    colors = ['lightblue', 'lightyellow', 'lightgreen', 'lightsalmon']
    
    for i_c in range(NI_c):
        for j_c in range(NJ_c):
            i_f = 2 * i_c
            j_f = 2 * j_c
            
            # Create polygon for 2x2 block
            corners = [
                (X_f[i_f, j_f], Y_f[i_f, j_f]),
                (X_f[i_f+2, j_f], Y_f[i_f+2, j_f]),
                (X_f[i_f+2, j_f+2], Y_f[i_f+2, j_f+2]),
                (X_f[i_f, j_f+2], Y_f[i_f, j_f+2]),
            ]
            
            color_idx = (i_c + j_c) % len(colors)
            poly = Polygon(corners, facecolor=colors[color_idx], 
                          edgecolor='black', linewidth=1.5, alpha=0.6)
            ax.add_patch(poly)
    
    # Plot coarse cell centers
    ax.scatter(metrics_c.xc.flatten(), metrics_c.yc.flatten(), 
               c='red', s=50, marker='x', linewidths=2, zorder=10)
    
    ax.set_aspect('equal')
    ax.set_title(f'Fine Grid ({NI_f}×{NJ_f}) with Coarse Overlay ({NI_c}×{NJ_c})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def plot_normal_summation_detail(ax, X, Y, metrics_f, metrics_c, i_c=1, j_c=1):
    """Show detailed view of how face normals are summed."""
    i_f = 2 * i_c
    j_f = 2 * j_c
    
    # Zoom to the 2x2 block plus a margin
    margin = 0.1
    x_min = X[i_f, j_f] - margin
    x_max = X[i_f+2, j_f+2] + margin
    y_min = Y[i_f, j_f] - margin
    y_max = Y[i_f+2, j_f+2] + margin
    
    # Plot fine grid cells
    for i in range(i_f, i_f + 3):
        ax.plot(X[i, j_f:j_f+3], Y[i, j_f:j_f+3], 'k-', linewidth=1)
    for j in range(j_f, j_f + 3):
        ax.plot(X[i_f:i_f+3, j], Y[i_f:i_f+3, j], 'k-', linewidth=1)
    
    # Highlight coarse cell boundary
    corners = [
        (X[i_f, j_f], Y[i_f, j_f]),
        (X[i_f+2, j_f], Y[i_f+2, j_f]),
        (X[i_f+2, j_f+2], Y[i_f+2, j_f+2]),
        (X[i_f, j_f+2], Y[i_f, j_f+2]),
    ]
    poly = Polygon(corners, facecolor='lightyellow', 
                  edgecolor='blue', linewidth=3, alpha=0.5)
    ax.add_patch(poly)
    
    # Plot fine I-face normals on left boundary (i_f)
    for dj in range(2):
        j = j_f + dj
        xm = 0.5 * (X[i_f, j] + X[i_f, j+1])
        ym = 0.5 * (Y[i_f, j] + Y[i_f, j+1])
        
        sx = metrics_f.Si_x[i_f, j]
        sy = metrics_f.Si_y[i_f, j]
        
        ax.arrow(xm, ym, sx * 0.3, sy * 0.3,
                head_width=0.02, head_length=0.01,
                fc='red', ec='red', linewidth=2)
        ax.annotate(f'S_i[{i_f},{j}]', (xm - 0.05, ym), fontsize=8)
    
    # Plot coarse I-face normal (sum of two fine)
    xm_c = 0.5 * (X[i_f, j_f] + X[i_f, j_f+2])
    ym_c = 0.5 * (Y[i_f, j_f] + Y[i_f, j_f+2])
    
    sx_c = metrics_c.Si_x[i_c, j_c]
    sy_c = metrics_c.Si_y[i_c, j_c]
    
    ax.arrow(xm_c, ym_c, sx_c * 0.3, sy_c * 0.3,
            head_width=0.025, head_length=0.012,
            fc='darkred', ec='darkred', linewidth=3)
    ax.annotate(f'S_i_c[{i_c},{j_c}] = sum', (xm_c - 0.1, ym_c + 0.05), 
                fontsize=9, color='darkred', fontweight='bold')
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_title(f'Face Normal Summation Detail (Coarse cell [{i_c},{j_c}])')


def create_volume_table(fig, metrics_list, level_names):
    """Create a table showing volume totals at each level."""
    ax = fig.add_subplot(2, 2, 4)
    ax.axis('off')
    
    # Table data
    data = []
    for name, metrics in zip(level_names, metrics_list):
        total_vol = np.sum(metrics.volume)
        data.append([name, f'{metrics.NI}', f'{metrics.NJ}', 
                    f'{metrics.NI * metrics.NJ}', f'{total_vol:.10f}'])
    
    columns = ['Level', 'NI', 'NJ', 'Cells', 'Total Volume']
    
    table = ax.table(cellText=data, colLabels=columns, 
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    ax.set_title('Volume Conservation Across Levels', pad=20)


def main():
    """Generate visual verification PDF."""
    print("Generating grid coarsening visualization...")
    
    # Create grid
    NI, NJ = 16, 12
    X, Y = create_distorted_grid(NI, NJ, amplitude=0.1)
    
    # Compute metrics at multiple levels
    computer = MetricComputer(X, Y)
    metrics_l0 = computer.compute()
    metrics_l1 = Coarsener.coarsen(metrics_l0)
    metrics_l2 = Coarsener.coarsen(metrics_l1)
    
    # Verify GCL at all levels
    gcl_l0 = computer.validate_gcl()
    gcl_l1_x, gcl_l1_y = Coarsener.validate_gcl(metrics_l1)
    gcl_l2_x, gcl_l2_y = Coarsener.validate_gcl(metrics_l2)
    
    print(f"  Level 0 GCL: {gcl_l0}")
    print(f"  Level 1 GCL: max_x={gcl_l1_x:.2e}, max_y={gcl_l1_y:.2e}")
    print(f"  Level 2 GCL: max_x={gcl_l2_x:.2e}, max_y={gcl_l2_y:.2e}")
    
    # Create PDF
    output_path = Path(__file__).parent.parent.parent / 'output' / 'tests' / 'coarsening_visual.pdf'
    # When run from tests/grid/, go up two levels to project root
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with PdfPages(output_path) as pdf:
        # Page 1: Grid hierarchy overview
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle('Grid Coarsening Visualization', fontsize=14)
        
        # Fine grid with normals
        ax1 = fig.add_subplot(2, 2, 1)
        plot_grid_with_normals(ax1, X, Y, metrics_l0, 
                               f'Level 0 (Fine): {NI}×{NJ}', 
                               show_normals=True, normal_scale=0.4)
        
        # Coarse overlay
        ax2 = fig.add_subplot(2, 2, 2)
        # Generate coarse node coordinates (every 2nd node)
        X_c = X[::2, ::2]
        Y_c = Y[::2, ::2]
        plot_coarse_overlay(ax2, X, Y, X_c, Y_c, metrics_l0, metrics_l1)
        
        # Normal summation detail
        ax3 = fig.add_subplot(2, 2, 3)
        plot_normal_summation_detail(ax3, X, Y, metrics_l0, metrics_l1, i_c=2, j_c=2)
        
        # Volume table
        create_volume_table(fig, 
                           [metrics_l0, metrics_l1, metrics_l2],
                           ['Level 0 (Fine)', 'Level 1', 'Level 2 (Coarse)'])
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
        # Page 2: Multi-level grid comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Grid Hierarchy: Fine to Coarse', fontsize=14)
        
        # Level 0
        plot_grid_with_normals(axes[0], X, Y, metrics_l0, 
                               f'Level 0: {metrics_l0.NI}×{metrics_l0.NJ}',
                               show_normals=False)
        
        # Level 1
        X_l1 = X[::2, ::2]
        Y_l1 = Y[::2, ::2]
        plot_grid_with_normals(axes[1], X_l1, Y_l1, metrics_l1,
                               f'Level 1: {metrics_l1.NI}×{metrics_l1.NJ}',
                               show_normals=False)
        
        # Level 2
        X_l2 = X[::4, ::4]
        Y_l2 = Y[::4, ::4]
        plot_grid_with_normals(axes[2], X_l2, Y_l2, metrics_l2,
                               f'Level 2: {metrics_l2.NI}×{metrics_l2.NJ}',
                               show_normals=False)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
        # Page 3: Wall distance comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Wall Distance Field Across Levels', fontsize=14)
        
        for ax, metrics, title in zip(axes, 
                                       [metrics_l0, metrics_l1, metrics_l2],
                                       ['Level 0', 'Level 1', 'Level 2']):
            im = ax.imshow(metrics.wall_distance.T, origin='lower', 
                          aspect='auto', cmap='viridis')
            plt.colorbar(im, ax=ax, label='Wall Distance')
            ax.set_title(f'{title}: {metrics.NI}×{metrics.NJ}')
            ax.set_xlabel('i')
            ax.set_ylabel('j')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    print(f"  Saved to: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()

