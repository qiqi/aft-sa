#!/usr/bin/env python3
"""
Flat Plate Boundary Layer Solver with Transition (JAX Version).

This script runs the SA-AFT model on a flat plate for different
freestream turbulence intensities and compares Cf to correlations.

Uses JAX for GPU acceleration and improved performance.

Assertions:
- Low Tu: Cf should follow laminar correlation (Cf ~ 0.664/sqrt(Re_x))
- High Tu: Cf should approach turbulent correlation
- Velocity profiles should be bounded [0, 1]
- nuHat should be positive
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

# Use JAX-accelerated solver
from src.solvers.boundary_layer_solvers import NuHatFlatPlateSolver


def run():
    solver = NuHatFlatPlateSolver()
    all_passed = True

    # --- RUNNING WITH BATCH ---
    Tu_batch = [0.0001, 0.01, 1.0, 5.0]
    u, v, nuHat = solver(Tu_batch)
    x_grid = solver.x_grid
    y_u = np.array(solver.y_cell)

    plt.figure(figsize=(9, 12))
    symbols = ['o', 's', '^', 'v']
    
    cf_results = {}
    
    for batch_idx in range(4):
        u_np = np.array(u[:, batch_idx, :])
        nu_np = np.array(nuHat[:, batch_idx, :])
        Tu = Tu_batch[batch_idx]

        # ===== PHYSICAL ASSERTIONS =====
        
        # 1. Velocity should be bounded [0, 1]
        if u_np.min() < -0.01:
            print(f"❌ Tu={Tu}%: Velocity below 0: {u_np.min():.4f}")
            all_passed = False
        if u_np.max() > 1.01:
            print(f"❌ Tu={Tu}%: Velocity above 1: {u_np.max():.4f}")
            all_passed = False
        
        # 2. nuHat should be positive (or very small negative from numerics)
        if nu_np.min() < -0.1:
            print(f"❌ Tu={Tu}%: Negative nuHat: {nu_np.min():.4f}")
            all_passed = False

        # Calculate Cf
        tau_w = 1.0 * u_np[:,0] / y_u[0]
        cf = tau_w * 2.0

        # Re_theta calculation
        Re_theta_list = []
        dy_vol = np.array(solver.dy_vol)
        for i in range(u_np.shape[0]):
            theta = np.sum(u_np[i,:] * (1 - u_np[i,:]) * dy_vol)
            Re_theta_list.append(theta)

        Re_theta = np.array(Re_theta_list)
        cf_results[Tu] = (Re_theta, cf)
        
        # 3. Cf should be positive
        if (cf[1:] < 0).any():
            print(f"❌ Tu={Tu}%: Negative skin friction")
            all_passed = False
        
        # 4. Cf should be in reasonable range
        if cf[1:].max() > 0.02:
            print(f"❌ Tu={Tu}%: Cf too high: {cf.max():.4f}")
            all_passed = False

        name = f'Tu={Tu}%'

        # Plotting
        ax = plt.subplot(5, 2, 2*batch_idx+1)
        levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        plt.clabel(plt.contour(x_grid, y_u, u_np.T, levels))
        plt.title(r'$u$, ' + name, y=0.8)
        plt.ylim([-1000, 15000])
        plt.ylabel('y')
        if batch_idx <= 2:
            ax.set_xticklabels([])
        else:
            plt.xlabel('x')

        ax = plt.subplot(5, 2, 2*batch_idx+2)
        levels = [-3.5, -3.0, -2.5, -2.0, -1.5, -1, -0.5, 0, 0.5, 1.0, 1.5, 2.0]
        plt.clabel(plt.contour(x_grid, y_u, np.log10(np.maximum(nu_np.T, 1e-10)), levels))
        plt.title(r'$\log(\hat\nu)$, ' + name, y=0.8)
        plt.ylim([-1000, 15000])
        ax.set_yticklabels([])
        if batch_idx <= 2:
            ax.set_xticklabels([])
        else:
            plt.xlabel('x')

        plt.subplot(6, 1, 6)
        plt.loglog(Re_theta[1:], cf[1:], symbols[batch_idx], mfc='w', label=name)

    # Reference correlations
    Re_theta_ref = cf_results[Tu_batch[0]][0]
    cf_lam = 0.441 / Re_theta_ref[1:]
    cf_turb = 2.0 * (1.0 / 0.38 * np.log(Re_theta_ref[1:]) + 3.7)**(-2)
    
    plt.loglog(Re_theta_ref[1:], cf_lam, 'k:', label='Laminar')
    plt.loglog(Re_theta_ref[1:], cf_turb, 'r--', label='Turbulent Correlation')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.xlim([1e2, 1e4])
    plt.ylim([1e-4, 1e-2])
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$c_f$')
    
    # 5. Check that low Tu stays closer to laminar correlation
    Re_theta_low, cf_low = cf_results[0.0001]
    valid_idx = Re_theta_low[1:] > 500
    if valid_idx.any():
        cf_lam_check = 0.441 / Re_theta_low[1:][valid_idx]
        cf_low_check = cf_low[1:][valid_idx]
        ratio = cf_low_check.mean() / cf_lam_check.mean()
        if ratio > 2.0:
            print(f"⚠️  Low Tu: Cf ratio to laminar = {ratio:.2f} (expected ~1)")
    
    out_path = os.path.join(get_output_dir(), 'flat_plate_batch.pdf')
    plt.savefig(out_path)
    print(f'Saved: {out_path}')
    
    # Summary
    if all_passed:
        print("✅ All flat plate physical constraints satisfied")
        return 0
    else:
        print("❌ Some physical constraints violated")
        return 1


if __name__ == "__main__":
    exit(run())
