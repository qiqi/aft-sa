#!/usr/bin/env python3
"""
Plot Falkner-Skan Boundary Layer Profiles.

This script solves and visualizes the Falkner-Skan similarity solutions
for different pressure gradients (favorable, zero, adverse).

Assertions:
- No-slip at wall: u(0) = 0
- Freestream convergence: u(∞) → 1
- Blasius wall shear: f''(0) ≈ 0.332 (well-known result)
- Positive shear for attached flows
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def get_output_dir():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = os.path.join(project_root, 'output', 'models')
    os.makedirs(out, exist_ok=True)
    return out

from src.physics.boundary_layer import solve_falkner_skan


def run():
    cases = [
        {'beta': 0.0, 'color': 'r', 'name': r'Blasius ($\beta=0$)', 'shear_at_wall': 0.332},
        {'beta': 1.0, 'color': 'g', 'name': r'Stagnation ($\beta=1$)', 'shear_at_wall': 1.233},
        {'beta': -0.19, 'color': 'b', 'name': r'Separation ($\beta=-0.19$)', 'shear_at_wall': 0.0}
    ]

    fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    all_passed = True

    for ax, case in zip(axs, cases):
        beta = case['beta']
        col = case['color']
        name = case['name']
        expected_shear = case['shear_at_wall']

        eta, u, dudy, v = solve_falkner_skan(beta)

        # ===== PHYSICAL ASSERTIONS =====
        
        # 1. No-slip at wall: u(0) = 0
        if abs(u[0]) > 1e-10:
            print(f"❌ {name}: No-slip violated, u(0) = {u[0]:.6f}")
            all_passed = False
        
        # 2. Freestream convergence: u(∞) → 1
        if abs(u[-1] - 1.0) > 0.01:
            print(f"❌ {name}: Freestream not reached, u(∞) = {u[-1]:.4f}")
            all_passed = False
        
        # 3. Wall shear matches known values (within 5%)
        wall_shear = dudy[0]
        if expected_shear > 0:
            rel_error = abs(wall_shear - expected_shear) / expected_shear
            if rel_error > 0.05:
                print(f"❌ {name}: Wall shear mismatch, f''(0) = {wall_shear:.3f}, expected {expected_shear:.3f}")
                all_passed = False
        
        # 4. For Blasius and favorable, shear should be positive in boundary layer
        # (allow small numerical noise ~1e-6 at domain edge)
        if beta >= 0:
            if (dudy < -1e-5).any():
                print(f"❌ {name}: Significant negative shear in attached flow: {dudy.min():.2e}")
                all_passed = False

        # Plot u, shear, and v
        ax.plot(u, eta, color=col, linewidth=2.5, label=r'$u/U_e$')
        ax.plot(dudy, eta, color='k', linestyle=':', linewidth=2, label=r"$f''$ (Shear)")
        ax.plot(v, eta, color='k', linestyle='--', linewidth=1.5, label=r'$v_{sc}$')

        # Formatting
        ax.set_title(name, fontsize=14)
        ax.set_xlabel("Dimensionless Values")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 6)

        # Set x-limits locally because 'v' varies wildly between cases
        if beta == -0.19:
            ax.set_xlim(-0.2, 2.5)
        elif beta == 1.0:
            ax.set_xlim(-1.2, 1.2)
        else:
            ax.set_xlim(-0.2, 1.2)

        ax.axvline(0, color='k', linewidth=0.5)

    axs[0].set_ylabel(r"$\eta=y\sqrt{\frac{U_{e}}{\nu x}}$", fontsize=14)
    axs[0].legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(get_output_dir(), 'falkner_skan_profiles.pdf')
    plt.savefig(out_path)
    print(f'Saved: {out_path}')
    
    # Summary
    if all_passed:
        print("✅ All Falkner-Skan physical constraints satisfied")
        return 0
    else:
        print("❌ Some physical constraints violated")
        return 1


if __name__ == "__main__":
    exit(run())
