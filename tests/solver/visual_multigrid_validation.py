#!/usr/bin/env python
"""
Final Multigrid Validation Visual Test.

Generates a comprehensive PDF showing:
1. GRID HIERARCHY: All levels of the multigrid
2. GCL VERIFICATION: Geometric conservation at all levels
3. TRANSFER CONSERVATION: State and residual conservation
4. COMPONENT SUMMARY: All multigrid components verified

Output: output/tests/multigrid_validation.pdf
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
from src.solvers.multigrid import build_multigrid_hierarchy
from src.solvers.boundary_conditions import FreestreamConditions, initialize_state
from src.numerics.multigrid import (
    restrict_state, 
    restrict_residual, 
    compute_integral,
    compute_residual_sum
)


def create_polar_grid(NI: int, NJ: int, r_inner: float = 0.2, r_outer: float = 1.0):
    """Create a polar grid (like C-grid topology)."""
    r = np.linspace(r_inner, r_outer, NJ + 1)
    theta = np.linspace(0, 2 * np.pi, NI + 1)
    
    R, THETA = np.meshgrid(r, theta, indexing='ij')
    R = R.T
    THETA = THETA.T
    
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)
    
    return X, Y


def main():
    """Generate comprehensive multigrid validation PDF."""
    print("=" * 60)
    print("MULTIGRID VALIDATION SUITE")
    print("=" * 60)
    
    # Create non-Cartesian grid
    NI, NJ = 64, 48
    n_wake = 10
    
    print(f"\nCreating polar grid: {NI} x {NJ}")
    X, Y = create_polar_grid(NI, NJ, r_inner=0.15, r_outer=1.0)
    
    # Compute fine metrics
    computer = MetricComputer(X, Y)
    metrics_fine = computer.compute()
    
    # Validate GCL on fine grid
    gcl_fine = computer.validate_gcl()
    print(f"Fine grid GCL: {gcl_fine}")
    
    # Initialize state
    freestream = FreestreamConditions()
    Q = initialize_state(NI, NJ, freestream)
    
    # Add perturbation for testing
    Q[1:-1, 1:-1, 0] += 0.1 * np.exp(-((metrics_fine.xc)**2 + (metrics_fine.yc)**2) / 0.1)
    
    # Build hierarchy
    print("\nBuilding multigrid hierarchy...")
    hierarchy = build_multigrid_hierarchy(X, Y, Q, freestream, n_wake=n_wake)
    print(hierarchy.get_level_info())
    
    # Validate all levels
    print("\nValidating all levels:")
    results = []
    
    for i, level in enumerate(hierarchy.levels):
        gcl_x, gcl_y = Coarsener.validate_gcl(level.metrics)
        vol_total = np.sum(level.metrics.volume)
        q_integral = compute_integral(level.Q[1:-1, 1:-1, :], level.metrics.volume)
        
        result = {
            'level': i,
            'NI': level.NI,
            'NJ': level.NJ,
            'cells': level.NI * level.NJ,
            'gcl_x': gcl_x,
            'gcl_y': gcl_y,
            'gcl_pass': gcl_x < 1e-10 and gcl_y < 1e-10,
            'volume': vol_total,
            'q_integral': q_integral[0],
        }
        results.append(result)
        
        status = "PASS" if result['gcl_pass'] else "FAIL"
        print(f"  Level {i}: {level.NI}x{level.NJ}, GCL: {status} "
              f"(max: {max(gcl_x, gcl_y):.2e})")
    
    # Test transfer conservation
    print("\nTesting transfer conservation...")
    
    if hierarchy.num_levels >= 2:
        l0 = hierarchy.levels[0]
        l1 = hierarchy.levels[1]
        
        # Create test Q
        Q_f = np.random.rand(l0.NI, l0.NJ, 4)
        Q_c = np.zeros((l1.NI, l1.NJ, 4))
        
        restrict_state(Q_f, l0.metrics.volume, Q_c, l1.metrics.volume)
        
        integral_f = compute_integral(Q_f, l0.metrics.volume)
        integral_c = compute_integral(Q_c, l1.metrics.volume)
        
        conservation_error = np.max(np.abs(integral_f - integral_c) / (np.abs(integral_f) + 1e-30))
        print(f"  State restriction conservation error: {conservation_error:.2e}")
        
        # Residual conservation
        R_f = np.random.rand(l0.NI, l0.NJ, 4)
        R_c = np.zeros((l1.NI, l1.NJ, 4))
        
        restrict_residual(R_f, R_c)
        
        sum_f = compute_residual_sum(R_f)
        sum_c = compute_residual_sum(R_c)
        
        residual_error = np.max(np.abs(sum_f - sum_c) / (np.abs(sum_f) + 1e-30))
        print(f"  Residual restriction conservation error: {residual_error:.2e}")
    
    # Create output PDF
    output_path = Path(__file__).parent.parent.parent / 'output' / 'tests' / 'multigrid_validation.pdf'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with PdfPages(output_path) as pdf:
        # =================================================================
        # Page 1: Grid Hierarchy
        # =================================================================
        n_levels = min(4, hierarchy.num_levels)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Multigrid Validation: Grid Hierarchy (Polar Grid)', fontsize=14)
        
        for idx, ax in enumerate(axes.flat):
            if idx >= n_levels:
                ax.axis('off')
                continue
            
            level = hierarchy.levels[idx]
            
            # Plot grid using cell centers
            ax.scatter(level.metrics.xc.flatten(), level.metrics.yc.flatten(),
                      c='blue', s=max(1, 20 - 4*idx), alpha=0.6)
            
            # Plot boundaries
            step = 2 ** idx
            X_lvl = X[::step, ::step]
            Y_lvl = Y[::step, ::step]
            
            ax.plot(X_lvl[0, :], Y_lvl[0, :], 'k-', linewidth=1)
            ax.plot(X_lvl[-1, :], Y_lvl[-1, :], 'k-', linewidth=1)
            ax.plot(X_lvl[:, 0], Y_lvl[:, 0], 'k-', linewidth=1)
            ax.plot(X_lvl[:, -1], Y_lvl[:, -1], 'k-', linewidth=1)
            
            ax.set_title(f'Level {idx}: {level.NI} × {level.NJ} = {level.NI * level.NJ} cells')
            ax.set_aspect('equal')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
        # =================================================================
        # Page 2: GCL Verification
        # =================================================================
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Geometric Conservation Law (GCL) Verification', fontsize=14)
        
        for idx, ax in enumerate(axes.flat):
            if idx >= n_levels:
                ax.axis('off')
                continue
            
            level = hierarchy.levels[idx]
            result = results[idx]
            
            # Compute GCL residual for each cell
            m = level.metrics
            gcl_res = np.sqrt(
                (m.Si_x[1:, :] - m.Si_x[:-1, :] + m.Sj_x[:, 1:] - m.Sj_x[:, :-1])**2 +
                (m.Si_y[1:, :] - m.Si_y[:-1, :] + m.Sj_y[:, 1:] - m.Sj_y[:, :-1])**2
            )
            
            sc = ax.scatter(level.metrics.xc.flatten(), level.metrics.yc.flatten(),
                           c=np.log10(gcl_res.flatten() + 1e-20), cmap='RdYlGn_r',
                           s=max(1, 20 - 4*idx), alpha=0.8, vmin=-18, vmax=-10)
            plt.colorbar(sc, ax=ax, label='log10(|GCL residual|)')
            
            status = "PASS ✓" if result['gcl_pass'] else "FAIL ✗"
            ax.set_title(f'Level {idx}: {status} (max: {max(result["gcl_x"], result["gcl_y"]):.2e})')
            ax.set_aspect('equal')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
        # =================================================================
        # Page 3: Volume Conservation
        # =================================================================
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Volume Conservation Across Levels', fontsize=14)
        
        levels = list(range(hierarchy.num_levels))
        volumes = [r['volume'] for r in results]
        
        axes[0].bar(levels, volumes, color='steelblue', alpha=0.8)
        axes[0].set_xlabel('Level')
        axes[0].set_ylabel('Total Volume')
        axes[0].set_title('Total Volume at Each Level')
        axes[0].set_xticks(levels)
        
        # Volume error relative to fine
        vol_errors = [abs(v - volumes[0]) / volumes[0] * 100 for v in volumes]
        axes[1].bar(levels, vol_errors, color='coral', alpha=0.8)
        axes[1].set_xlabel('Level')
        axes[1].set_ylabel('Relative Error (%)')
        axes[1].set_title('Volume Error Relative to Finest Grid')
        axes[1].set_xticks(levels)
        
        if max(vol_errors) < 1e-10:
            axes[1].set_ylim(0, 1e-10)
            axes[1].annotate('All errors < 1e-10%', xy=(0.5, 0.5), 
                            xycoords='axes fraction', ha='center', fontsize=12)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
        # =================================================================
        # Page 4: Summary Table
        # =================================================================
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('off')
        
        # Build comprehensive table
        headers = ['Level', 'NI', 'NJ', 'Cells', 'Volume', 'GCL Max', 'GCL Status']
        data = []
        
        for r in results:
            gcl_max = max(r['gcl_x'], r['gcl_y'])
            status = 'PASS' if r['gcl_pass'] else 'FAIL'
            data.append([
                f"Level {r['level']}",
                str(r['NI']),
                str(r['NJ']),
                f"{r['cells']:,}",
                f"{r['volume']:.6f}",
                f"{gcl_max:.2e}",
                status
            ])
        
        # Add totals row
        total_cells = sum(r['cells'] for r in results)
        all_pass = all(r['gcl_pass'] for r in results)
        data.append([
            'TOTAL',
            '-',
            '-',
            f"{total_cells:,}",
            f"{results[0]['volume']:.6f}",
            '-',
            'ALL PASS' if all_pass else 'SOME FAIL'
        ])
        
        table = ax.table(cellText=data, colLabels=headers,
                        loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.3, 2.0)
        
        # Color the status column
        for i, r in enumerate(results):
            cell = table[(i + 1, 6)]
            if r['gcl_pass']:
                cell.set_facecolor('lightgreen')
            else:
                cell.set_facecolor('lightcoral')
        
        # Total row
        cell = table[(len(results) + 1, 6)]
        if all_pass:
            cell.set_facecolor('green')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor('red')
            cell.set_text_props(color='white', fontweight='bold')
        
        ax.set_title('Multigrid Validation Summary\nNon-Cartesian Polar Grid Test', 
                    fontsize=14, pad=30)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    print(f"\nOutput saved to: {output_path}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    all_gcl_pass = all(r['gcl_pass'] for r in results)
    print(f"GCL at all levels:        {'PASS ✓' if all_gcl_pass else 'FAIL ✗'}")
    print(f"Volume conservation:      {'PASS ✓' if vol_errors[-1] < 1e-10 else 'FAIL ✗'}")
    print(f"State conservation:       {'PASS ✓' if conservation_error < 1e-10 else 'FAIL ✗'}")
    print(f"Residual conservation:    {'PASS ✓' if residual_error < 1e-10 else 'FAIL ✗'}")
    
    overall = all_gcl_pass and vol_errors[-1] < 1e-10 and conservation_error < 1e-10 and residual_error < 1e-10
    print(f"\nOVERALL:                  {'PASS ✓' if overall else 'FAIL ✗'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

