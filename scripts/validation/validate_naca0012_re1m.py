#!/usr/bin/env python
"""
Validation: NACA 0012 at Re = 1,000,000

Compare RANS solver (with SA turbulence) against mfoil (XFOIL-like).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.solvers.rans_solver import RANSSolver, SolverConfig
from src.validation.mfoil_runner import run_turbulent
from src.numerics.forces import compute_aerodynamic_forces


def run_rans_case(alpha: float, reynolds: float = 1e6, 
                  max_iter: int = 3000, verbose: bool = True) -> dict:
    """Run RANS solver for a single case."""
    config = SolverConfig(
        alpha=alpha,
        reynolds=reynolds,
        mach=0.15,
        beta=10.0,
        cfl_start=0.5,
        cfl_target=5.0,
        cfl_ramp_iters=500,
        max_iter=max_iter,
        tol=1e-8,
        jst_k4=0.04,
        smoothing_epsilon=0.2,
        smoothing_passes=2,
        sponge_thickness=15,
        html_animation=False,
        print_freq=100 if verbose else max_iter + 1,
        diagnostic_freq=100,
    )
    
    solver = RANSSolver('data/naca0012.dat', config)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running RANS: alpha = {alpha}°, Re = {reynolds:.1e}")
        print(f"{'='*60}")
    
    # Run solver
    for i in range(max_iter):
        solver.step()
        
        # Check convergence
        if solver.converged:
            if verbose:
                print(f"Converged at iteration {solver.iteration}")
            break
    
    # Compute forces
    forces = solver.compute_forces()
    
    # Extract surface data
    surface = solver.get_surface_data()
    
    result = {
        'CL': forces.CL,
        'CD': forces.CD,
        'CD_p': forces.CD_p,
        'CD_f': forces.CD_f,
        'converged': solver.converged,
        'iterations': solver.iteration,
        'final_residual': solver.residual_history[-1] if solver.residual_history else np.nan,
    }
    
    if surface is not None:
        result['x_surf'] = surface.get('x', None)
        result['Cp'] = surface.get('cp', None)  # lowercase
        result['Cf'] = surface.get('cf', None)  # lowercase
    
    return result


def run_mfoil_case(alpha: float, reynolds: float = 1e6) -> dict:
    """Run mfoil for comparison."""
    result = run_turbulent(reynolds=reynolds, alpha=alpha, naca='0012', ncrit=9.0)
    
    return {
        'CL': result['cl'],
        'CD': result['cd'],
        'CD_p': result['cdp'],
        'CD_f': result['cdf'],
        'converged': result['converged'],
        'x_upper': result.get('x_upper'),
        'x_lower': result.get('x_lower'),
        'Cp_upper': result.get('cp_upper'),
        'Cp_lower': result.get('cp_lower'),
        'Cf_upper': result.get('cf_upper'),
        'Cf_lower': result.get('cf_lower'),
    }


def main():
    """Run validation comparison."""
    
    # Test angles
    alphas = [0.0, 2.0, 4.0, 6.0, 8.0]
    reynolds = 1e6
    
    print("="*70)
    print(f"NACA 0012 Validation at Re = {reynolds:.1e}")
    print("="*70)
    
    # Storage for results
    rans_results = []
    mfoil_results = []
    
    # Run mfoil first (faster)
    print("\n--- Running mfoil (XFOIL-like) ---")
    for alpha in alphas:
        print(f"  mfoil: alpha = {alpha}°...", end=" ", flush=True)
        result = run_mfoil_case(alpha, reynolds)
        mfoil_results.append(result)
        print(f"CL={result['CL']:.4f}, CD={result['CD']:.5f}")
    
    # Run RANS solver
    print("\n--- Running RANS solver (with SA turbulence) ---")
    for alpha in alphas:
        result = run_rans_case(alpha, reynolds, max_iter=2000, verbose=True)
        rans_results.append(result)
    
    # Print comparison table
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    print(f"{'Alpha':>6} | {'CL (RANS)':>10} {'CL (mfoil)':>10} {'Δ':>7} | "
          f"{'CD (RANS)':>10} {'CD (mfoil)':>10} {'Δ':>7}")
    print("-"*70)
    
    for i, alpha in enumerate(alphas):
        rans = rans_results[i]
        mfoil = mfoil_results[i]
        
        cl_diff = rans['CL'] - mfoil['CL']
        cd_diff = (rans['CD'] - mfoil['CD']) * 10000  # counts
        
        print(f"{alpha:>6.1f}° | {rans['CL']:>10.4f} {mfoil['CL']:>10.4f} {cl_diff:>+7.4f} | "
              f"{rans['CD']:>10.5f} {mfoil['CD']:>10.5f} {cd_diff:>+7.1f}ct")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # CL vs alpha
    ax = axes[0, 0]
    cl_rans = [r['CL'] for r in rans_results]
    cl_mfoil = [r['CL'] for r in mfoil_results]
    ax.plot(alphas, cl_rans, 'bo-', label='RANS (SA)', markersize=8)
    ax.plot(alphas, cl_mfoil, 'rs--', label='mfoil', markersize=8)
    ax.set_xlabel('Angle of Attack (°)')
    ax.set_ylabel('CL')
    ax.set_title('Lift Coefficient')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # CD vs alpha
    ax = axes[0, 1]
    cd_rans = [r['CD'] for r in rans_results]
    cd_mfoil = [r['CD'] for r in mfoil_results]
    ax.plot(alphas, cd_rans, 'bo-', label='RANS (SA)', markersize=8)
    ax.plot(alphas, cd_mfoil, 'rs--', label='mfoil', markersize=8)
    ax.set_xlabel('Angle of Attack (°)')
    ax.set_ylabel('CD')
    ax.set_title('Drag Coefficient')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Drag polar
    ax = axes[1, 0]
    ax.plot(cd_rans, cl_rans, 'bo-', label='RANS (SA)', markersize=8)
    ax.plot(cd_mfoil, cl_mfoil, 'rs--', label='mfoil', markersize=8)
    ax.set_xlabel('CD')
    ax.set_ylabel('CL')
    ax.set_title('Drag Polar')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # L/D vs alpha
    ax = axes[1, 1]
    ld_rans = [cl_rans[i] / (cd_rans[i] + 1e-10) for i in range(len(alphas))]
    ld_mfoil = [cl_mfoil[i] / (cd_mfoil[i] + 1e-10) for i in range(len(alphas))]
    ax.plot(alphas, ld_rans, 'bo-', label='RANS (SA)', markersize=8)
    ax.plot(alphas, ld_mfoil, 'rs--', label='mfoil', markersize=8)
    ax.set_xlabel('Angle of Attack (°)')
    ax.set_ylabel('L/D')
    ax.set_title('Lift-to-Drag Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('output/validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'naca0012_re1m_comparison.png', dpi=150)
    print(f"\nPlot saved to: {output_dir / 'naca0012_re1m_comparison.png'}")
    
    # Surface distribution comparison for alpha = 4°
    idx = alphas.index(4.0)
    rans = rans_results[idx]
    mfoil = mfoil_results[idx]
    
    if rans.get('Cp') is not None and mfoil.get('Cp_upper') is not None:
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
        
        # Cp distribution
        ax = axes2[0]
        if rans.get('x_surf') is not None:
            ax.plot(rans['x_surf'], -rans['Cp'], 'b-', label='RANS (SA)', linewidth=1.5)
        ax.plot(mfoil['x_upper'], -mfoil['Cp_upper'], 'r--', label='mfoil upper', linewidth=1.5)
        ax.plot(mfoil['x_lower'], -mfoil['Cp_lower'], 'r:', label='mfoil lower', linewidth=1.5)
        ax.set_xlabel('x/c')
        ax.set_ylabel('-Cp')
        ax.set_title(f'Pressure Distribution (α = 4°)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Cf distribution
        ax = axes2[1]
        if rans.get('Cf') is not None:
            ax.plot(rans['x_surf'], rans['Cf'], 'b-', label='RANS (SA)', linewidth=1.5)
        ax.plot(mfoil['x_upper'], mfoil['Cf_upper'], 'r--', label='mfoil upper', linewidth=1.5)
        ax.plot(mfoil['x_lower'], mfoil['Cf_lower'], 'r:', label='mfoil lower', linewidth=1.5)
        ax.set_xlabel('x/c')
        ax.set_ylabel('Cf')
        ax.set_title(f'Skin Friction Distribution (α = 4°)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig2.savefig(output_dir / 'naca0012_re1m_surface_alpha4.png', dpi=150)
        print(f"Surface plot saved to: {output_dir / 'naca0012_re1m_surface_alpha4.png'}")
    
    plt.show()
    
    return rans_results, mfoil_results


if __name__ == '__main__':
    main()
