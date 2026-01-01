#!/usr/bin/env python
"""
Visual verification of multigrid hierarchy.

Generates a PDF showing:
1. GRID HIERARCHY: Grids at levels 0, 1, 2, 3 with dimensions
2. Q FIELD ACROSS LEVELS: Pressure contours at each level
3. BOUNDARY CONDITIONS: Wall and wake regions highlighted
4. WALL DISTANCE FIELD: Contours at each level

Output: output/tests/hierarchy_visual.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.grid.metrics import MetricComputer
from src.solvers.multigrid import build_multigrid_hierarchy
from src.solvers.boundary_conditions import FreestreamConditions


def create_c_grid_like(NI: int, NJ: int, radius: float = 1.0):
    """
    Create a simplified C-grid-like mesh for testing.
    
    This is a sector of a polar grid that mimics the topology
    of a C-grid around an airfoil.
    """
    # Radial direction (j)
    r = np.linspace(0.1, radius, NJ + 1)
    
    # Angular direction (i) - wrap around
    theta = np.linspace(-np.pi, np.pi, NI + 1)
    
    R, THETA = np.meshgrid(r, theta, indexing='ij')
    
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)
    
    # Swap to (i, j) ordering
    X = X.T
    Y = Y.T
    
    return X, Y


def create_distorted_grid(NI: int, NJ: int, amplitude: float = 0.05):
    """Create a Cartesian grid with sinusoidal distortion."""
    x = np.linspace(0, 2, NI + 1)
    y = np.linspace(0, 1, NJ + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    dx = 2.0 / NI
    dy = 1.0 / NJ
    
    X_dist = X + amplitude * dx * np.sin(2 * np.pi * Y)
    Y_dist = Y + amplitude * dy * np.sin(np.pi * X)
    
    return X_dist, Y_dist


def main():
    """Generate visual verification PDF."""
    print("Generating multigrid hierarchy visualization...")
    
    # Create non-Cartesian aligned grid (polar/C-grid-like)
    NI, NJ = 64, 48
    n_wake = 10
    
    X, Y = create_c_grid_like(NI, NJ, radius=1.0)
    
    # Create initial state with smooth variation
    Q = np.zeros((NI + 2, NJ + 2, 4))
    
    # Compute metrics for creating smooth field
    computer = MetricComputer(X, Y)
    metrics = computer.compute()
    
    # Create smooth Gaussian-like field
    for k in range(4):
        xc_pad = np.pad(metrics.xc, ((1, 1), (1, 1)), mode='edge')
        yc_pad = np.pad(metrics.yc, ((1, 1), (1, 1)), mode='edge')
        Q[:, :, k] = np.exp(-((xc_pad - 1)**2 + (yc_pad - 0.5)**2) / 0.2) * (k + 1)
    
    freestream = FreestreamConditions()
    
    # Build hierarchy
    hierarchy = build_multigrid_hierarchy(X, Y, Q, freestream, n_wake=n_wake)
    
    print(hierarchy.get_level_info())
    
    output_path = Path(__file__).parent.parent.parent / 'output' / 'tests' / 'hierarchy_visual.pdf'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with PdfPages(output_path) as pdf:
        # =================================================================
        # Page 1: Grid Hierarchy
        # =================================================================
        n_levels = min(4, hierarchy.num_levels)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Multigrid Grid Hierarchy', fontsize=14)
        
        for idx, ax in enumerate(axes.flat):
            if idx >= n_levels:
                ax.axis('off')
                continue
            
            level = hierarchy.levels[idx]
            
            # Get grid coordinates for this level (every 2^idx nodes)
            step = 2 ** idx
            X_lvl = X[::step, ::step]
            Y_lvl = Y[::step, ::step]
            
            # Plot grid lines
            for i in range(X_lvl.shape[0]):
                ax.plot(X_lvl[i, :], Y_lvl[i, :], 'k-', linewidth=0.3, alpha=0.7)
            for j in range(X_lvl.shape[1]):
                ax.plot(X_lvl[:, j], Y_lvl[:, j], 'k-', linewidth=0.3, alpha=0.7)
            
            # Plot cell centers
            ax.scatter(level.metrics.xc.flatten(), level.metrics.yc.flatten(),
                      c='blue', s=2, alpha=0.5)
            
            ax.set_title(f'Level {idx}: {level.NI} × {level.NJ} = {level.NI * level.NJ} cells')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
        # =================================================================
        # Page 2: Q Field Across Levels
        # =================================================================
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('State Variable Q[0] (Pressure) Across Levels', fontsize=14)
        
        # Find global colormap range
        vmin = min(lvl.Q[1:-1, 1:-1, 0].min() for lvl in hierarchy.levels[:n_levels])
        vmax = max(lvl.Q[1:-1, 1:-1, 0].max() for lvl in hierarchy.levels[:n_levels])
        
        for idx, ax in enumerate(axes.flat):
            if idx >= n_levels:
                ax.axis('off')
                continue
            
            level = hierarchy.levels[idx]
            Q_interior = level.Q[1:-1, 1:-1, 0]
            
            im = ax.imshow(Q_interior.T, origin='lower', aspect='auto',
                          cmap='viridis', vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax)
            
            ax.set_title(f'Level {idx}: {level.NI} × {level.NJ}')
            ax.set_xlabel('i')
            ax.set_ylabel('j')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
        # =================================================================
        # Page 3: Boundary Regions
        # =================================================================
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Boundary Condition Regions (j=0 face)', fontsize=14)
        
        for idx, ax in enumerate(axes.flat):
            if idx >= n_levels:
                ax.axis('off')
                continue
            
            level = hierarchy.levels[idx]
            level_n_wake = n_wake // (2 ** idx)
            
            # Create mask: 0 = wake, 1 = airfoil
            mask = np.zeros(level.NI)
            mask[level_n_wake:level.NI - level_n_wake] = 1
            
            # Plot as bar
            colors = ['lightblue' if m == 0 else 'coral' for m in mask]
            ax.bar(range(level.NI), np.ones(level.NI), color=colors, width=1.0)
            
            ax.axvline(x=level_n_wake - 0.5, color='black', linestyle='--', linewidth=2)
            ax.axvline(x=level.NI - level_n_wake - 0.5, color='black', linestyle='--', linewidth=2)
            
            ax.set_xlim(-0.5, level.NI - 0.5)
            ax.set_ylim(0, 1.5)
            ax.set_xlabel('i index')
            ax.set_title(f'Level {idx}: wake={level_n_wake} cells each side')
            
            # Legend
            ax.text(level_n_wake / 2, 1.2, 'Wake', ha='center', fontsize=10, color='blue')
            ax.text(level.NI / 2, 1.2, 'Airfoil', ha='center', fontsize=10, color='red')
            ax.text(level.NI - level_n_wake / 2, 1.2, 'Wake', ha='center', fontsize=10, color='blue')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
        # =================================================================
        # Page 4: Wall Distance Field
        # =================================================================
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Wall Distance Field Across Levels', fontsize=14)
        
        for idx, ax in enumerate(axes.flat):
            if idx >= n_levels:
                ax.axis('off')
                continue
            
            level = hierarchy.levels[idx]
            wall_dist = level.metrics.wall_distance
            
            im = ax.imshow(wall_dist.T, origin='lower', aspect='auto', cmap='plasma')
            plt.colorbar(im, ax=ax, label='Wall Distance')
            
            ax.set_title(f'Level {idx}: {level.NI} × {level.NJ}')
            ax.set_xlabel('i')
            ax.set_ylabel('j')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
        # =================================================================
        # Page 5: Level Summary Table
        # =================================================================
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Build table data
        data = []
        for i, level in enumerate(hierarchy.levels):
            cells = level.NI * level.NJ
            vol_total = np.sum(level.metrics.volume)
            n_wake_lvl = level.bc.n_wake_points
            
            data.append([
                f'Level {i}',
                f'{level.NI}',
                f'{level.NJ}',
                f'{cells:,}',
                f'{vol_total:.6f}',
                f'{n_wake_lvl}',
            ])
        
        columns = ['Level', 'NI', 'NJ', 'Cells', 'Total Volume', 'Wake Pts']
        
        table = ax.table(cellText=data, colLabels=columns,
                        loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.5, 2.0)
        
        ax.set_title('Multigrid Hierarchy Summary', fontsize=14, pad=20)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    print(f"  Saved to: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()

