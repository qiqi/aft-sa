#!/usr/bin/env python
"""
Visual verification of FAS V-Cycle.

Generates a PDF showing:
1. FAS FORCING FIELD: R_inj, R(Q_c), and P_c = R_inj - R(Q_c)
2. CORRECTION PROPAGATION: Q before/after prolongation
3. LEVEL-BY-LEVEL RESIDUAL: Residual magnitude at each level
4. HIERARCHY OVERVIEW: Grid and Q at each level

Output: output/tests/vcycle_visual.pdf
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
from src.solvers.boundary_conditions import FreestreamConditions, initialize_state
from src.numerics.multigrid import restrict_residual


def create_polar_grid(NI: int, NJ: int, r_inner: float = 0.2, r_outer: float = 1.0):
    """
    Create a polar grid (like C-grid topology around airfoil).
    """
    r = np.linspace(r_inner, r_outer, NJ + 1)
    theta = np.linspace(0, 2 * np.pi, NI + 1)
    
    R, THETA = np.meshgrid(r, theta, indexing='ij')
    R = R.T
    THETA = THETA.T
    
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)
    
    return X, Y


def main():
    """Generate visual verification PDF."""
    print("Generating FAS V-Cycle visualization...")
    
    # Create non-Cartesian aligned grid
    NI, NJ = 64, 48
    n_wake = 8
    
    X, Y = create_polar_grid(NI, NJ, r_inner=0.15, r_outer=1.0)
    
    # Compute metrics
    computer = MetricComputer(X, Y)
    metrics = computer.compute()
    
    # Initialize state with perturbation
    freestream = FreestreamConditions()
    Q = initialize_state(NI, NJ, freestream)
    
    # Add smooth perturbation (Gaussian bump)
    for k in range(4):
        Q[1:-1, 2:-1, k] += 0.1 * np.exp(-((metrics.xc)**2 + (metrics.yc)**2) / 0.1) * (k + 1)
    
    # Build hierarchy
    hierarchy = build_multigrid_hierarchy(X, Y, Q, freestream, n_wake=n_wake)
    
    print(hierarchy.get_level_info())
    
    output_path = Path(__file__).parent.parent.parent / 'output' / 'tests' / 'vcycle_visual.pdf'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with PdfPages(output_path) as pdf:
        # =================================================================
        # Page 1: Grid Hierarchy with State
        # =================================================================
        n_levels = min(4, hierarchy.num_levels)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('FAS V-Cycle: Grid Hierarchy with State Q[0]', fontsize=14)
        
        for idx, ax in enumerate(axes.flat):
            if idx >= n_levels:
                ax.axis('off')
                continue
            
            level = hierarchy.levels[idx]
            Q_interior = level.Q[1:-1, 2:-1, 0]
            
            # Plot state field
            step = 2 ** idx
            X_lvl = X[::step, ::step]
            Y_lvl = Y[::step, ::step]
            
            # Use scatter for non-structured visualization
            sc = ax.scatter(level.metrics.xc.flatten(), level.metrics.yc.flatten(),
                           c=Q_interior.flatten(), cmap='viridis', s=20, alpha=0.8)
            plt.colorbar(sc, ax=ax, label='Q[0]')
            
            # Plot grid outline
            ax.plot(X_lvl[0, :], Y_lvl[0, :], 'k-', linewidth=0.5, alpha=0.3)
            ax.plot(X_lvl[-1, :], Y_lvl[-1, :], 'k-', linewidth=0.5, alpha=0.3)
            ax.plot(X_lvl[:, 0], Y_lvl[:, 0], 'k-', linewidth=0.5, alpha=0.3)
            ax.plot(X_lvl[:, -1], Y_lvl[:, -1], 'k-', linewidth=0.5, alpha=0.3)
            
            ax.set_title(f'Level {idx}: {level.NI} × {level.NJ}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
        # =================================================================
        # Page 2: FAS Forcing Demonstration
        # =================================================================
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('FAS Forcing Term: P_c = R_inj - R(Q_c)', fontsize=14)
        
        # Create dummy residuals to demonstrate
        level0 = hierarchy.levels[0]
        level1 = hierarchy.levels[1]
        
        # Simulated fine residual
        R_f = np.zeros((level0.NI, level0.NJ, 4))
        R_f[:, :, 0] = np.sin(2 * np.pi * level0.metrics.xc) * np.cos(2 * np.pi * level0.metrics.yc)
        
        # Restrict to coarse
        R_inj = np.zeros((level1.NI, level1.NJ, 4))
        restrict_residual(R_f, R_inj)
        
        # Simulated coarse residual R(Q_c) - different pattern
        R_Qc = np.zeros((level1.NI, level1.NJ, 4))
        R_Qc[:, :, 0] = np.sin(np.pi * level1.metrics.xc) * np.cos(np.pi * level1.metrics.yc)
        
        # FAS forcing
        P_c = R_inj - R_Qc
        
        # Plot fine residual
        sc1 = axes[0, 0].scatter(level0.metrics.xc.flatten(), level0.metrics.yc.flatten(),
                                  c=R_f[:, :, 0].flatten(), cmap='RdBu_r', s=5, alpha=0.8)
        plt.colorbar(sc1, ax=axes[0, 0])
        axes[0, 0].set_title('Fine Residual R_f')
        axes[0, 0].set_aspect('equal')
        
        # Plot restricted residual R_inj
        sc2 = axes[0, 1].scatter(level1.metrics.xc.flatten(), level1.metrics.yc.flatten(),
                                  c=R_inj[:, :, 0].flatten(), cmap='RdBu_r', s=20, alpha=0.8)
        plt.colorbar(sc2, ax=axes[0, 1])
        axes[0, 1].set_title('Restricted R_inj = ΣR_f')
        axes[0, 1].set_aspect('equal')
        
        # Plot coarse residual R(Q_c)
        sc3 = axes[0, 2].scatter(level1.metrics.xc.flatten(), level1.metrics.yc.flatten(),
                                  c=R_Qc[:, :, 0].flatten(), cmap='RdBu_r', s=20, alpha=0.8)
        plt.colorbar(sc3, ax=axes[0, 2])
        axes[0, 2].set_title('Coarse Residual R(Q_c)')
        axes[0, 2].set_aspect('equal')
        
        # Plot FAS forcing P_c
        sc4 = axes[1, 0].scatter(level1.metrics.xc.flatten(), level1.metrics.yc.flatten(),
                                  c=P_c[:, :, 0].flatten(), cmap='RdBu_r', s=20, alpha=0.8)
        plt.colorbar(sc4, ax=axes[1, 0])
        axes[1, 0].set_title('FAS Forcing P_c = R_inj - R(Q_c)')
        axes[1, 0].set_aspect('equal')
        
        # Sum comparison
        axes[1, 1].bar(['R_f', 'R_inj', 'R(Q_c)', 'P_c'], 
                       [R_f[:, :, 0].sum(), R_inj[:, :, 0].sum(), 
                        R_Qc[:, :, 0].sum(), P_c[:, :, 0].sum()])
        axes[1, 1].set_ylabel('Sum')
        axes[1, 1].set_title('Residual Sums')
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Text explanation
        axes[1, 2].axis('off')
        explanation = """FAS Forcing Explanation:

The forcing term P_c ensures that the 
coarse grid problem is consistent with 
the fine grid:

P_c = R_f^{inj} - R(Q_c)

Where:
- R_f^{inj}: Residual restricted from fine
- R(Q_c): Residual computed on coarse Q

The coarse grid then solves:
R(Q_c) + P_c = 0

This corrects for the fine-grid 
information lost during restriction.
"""
        axes[1, 2].text(0.1, 0.5, explanation, transform=axes[1, 2].transAxes,
                        fontsize=11, verticalalignment='center', family='monospace')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
        # =================================================================
        # Page 3: Correction Prolongation
        # =================================================================
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Correction Prolongation: Fine Q Before/After', fontsize=14)
        
        # Store Q before correction
        Q_before = hierarchy.levels[0].Q.copy()
        
        # Simulate restriction and modification
        hierarchy.restrict_to_coarse(0)
        
        # Modify coarse Q (simulate smoothing)
        hierarchy.levels[1].Q[1:-1, 2:-1, 0] += 0.05
        
        # Prolongate correction
        hierarchy.prolongate_correction(1)
        
        Q_after = hierarchy.levels[0].Q.copy()
        
        # Plot before
        sc1 = axes[0, 0].scatter(level0.metrics.xc.flatten(), level0.metrics.yc.flatten(),
                                  c=Q_before[1:-1, 2:-1, 0].flatten(), cmap='viridis', s=5)
        plt.colorbar(sc1, ax=axes[0, 0])
        axes[0, 0].set_title('Q[0] Before Correction')
        axes[0, 0].set_aspect('equal')
        
        # Plot after
        sc2 = axes[0, 1].scatter(level0.metrics.xc.flatten(), level0.metrics.yc.flatten(),
                                  c=Q_after[1:-1, 2:-1, 0].flatten(), cmap='viridis', s=5)
        plt.colorbar(sc2, ax=axes[0, 1])
        axes[0, 1].set_title('Q[0] After Correction')
        axes[0, 1].set_aspect('equal')
        
        # Plot difference
        diff = Q_after[1:-1, 2:-1, 0] - Q_before[1:-1, 2:-1, 0]
        sc3 = axes[1, 0].scatter(level0.metrics.xc.flatten(), level0.metrics.yc.flatten(),
                                  c=diff.flatten(), cmap='RdBu_r', s=5)
        plt.colorbar(sc3, ax=axes[1, 0])
        axes[1, 0].set_title('Correction ΔQ = After - Before')
        axes[1, 0].set_aspect('equal')
        
        # Statistics
        axes[1, 1].axis('off')
        stats = f"""Correction Statistics:
        
Max |ΔQ|: {np.max(np.abs(diff)):.4f}
Mean |ΔQ|: {np.mean(np.abs(diff)):.4f}
Correction Range: [{diff.min():.4f}, {diff.max():.4f}]

The correction is bilinearly interpolated
from the coarse grid to the fine grid.

This propagates the low-frequency error
corrections from coarse to fine.
"""
        axes[1, 1].text(0.1, 0.5, stats, transform=axes[1, 1].transAxes,
                        fontsize=11, verticalalignment='center', family='monospace')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
        # =================================================================
        # Page 4: Level Summary
        # =================================================================
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Build table data
        data = []
        for i, level in enumerate(hierarchy.levels):
            cells = level.NI * level.NJ
            vol_total = np.sum(level.metrics.volume)
            q_mean = level.Q[1:-1, 2:-1, :].mean()
            
            data.append([
                f'Level {i}',
                f'{level.NI}',
                f'{level.NJ}',
                f'{cells:,}',
                f'{vol_total:.4f}',
                f'{q_mean:.4f}',
            ])
        
        columns = ['Level', 'NI', 'NJ', 'Cells', 'Volume', 'Mean Q']
        
        table = ax.table(cellText=data, colLabels=columns,
                        loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.5, 2.0)
        
        ax.set_title('V-Cycle Level Summary\n(Non-Cartesian Polar Grid)', fontsize=14, pad=20)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    print(f"  Saved to: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()

