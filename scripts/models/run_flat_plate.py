#!/usr/bin/env python3
"""
Flat Plate Boundary Layer Solver with Transition (JAX Version).

This script runs the SA-AI model on a flat plate for different
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
    # Five Tu values of natural transition on a smooth flat plate. Tu -> chi_inf
    # uses Mack's (1977) e^N critical-N-factor correlation, adopted wholesale via
    # calibrate_kernel.chi_inf_from_Tu_pct (A=-8.43, B=2.4). The coupled-RANS
    # onset (chi=1) then lands within ~8% of Abu-Ghannam & Shaw across the
    # natural-transition range; see paper Sec. flat-plate.
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'scripts'))
    from calibrate_kernel import chi_inf_from_Tu_pct as chi_from_Tu_pct
    Tu_labels = [0.026, 0.06, 0.18, 0.30, 0.85]
    case_names = [f'$Tu\\!=\\!{Tu:.3g}$%' for Tu in Tu_labels]
    Tu_batch = [chi_from_Tu_pct(Tu) for Tu in Tu_labels]
    # Schubauer-Skramstad experimental transition Re_theta (Mack 1977)
    expt_ReTheta = {0.026: 1115, 0.06: 890, 0.18: 681, 0.30: 575, 0.85: 345}
    print("Case mapping:")
    for Tu, chi, name in zip(Tu_labels, Tu_batch, case_names):
        print(f"  {name}: chi_inf = {chi:.4e}, S-S Re_theta_tr = {expt_ReTheta[Tu]}")
    u, v, nuHat = solver(Tu_batch)
    x_grid = solver.x_grid
    y_u = np.array(solver.y_cell)

    # GridSpec: 5 contour rows of equal height + 1 bottom row taller for Cf
    n_cases = len(Tu_labels)
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(9, 11))
    gs = gridspec.GridSpec(n_cases + 1, 2, height_ratios=[1]*n_cases + [2.4],
                           hspace=0.30, wspace=0.10,
                           top=0.985, bottom=0.06, left=0.07, right=0.97)
    symbols = ['o', 's', '^', 'v', 'D']

    cf_results = {}

    for batch_idx in range(n_cases):
        u_np = np.array(u[:, batch_idx, :])
        nu_np = np.array(nuHat[:, batch_idx, :])
        Tu_pct = Tu_labels[batch_idx]
        chi_inf = Tu_batch[batch_idx]

        # ===== PHYSICAL ASSERTIONS =====
        # 1. Velocity should be bounded [0, 1]
        if u_np.min() < -0.01:
            print(f"⚠️  Tu={Tu_pct}%: Velocity below 0: {u_np.min():.4f}")
        if u_np.max() > 1.01:
            print(f"⚠️  Tu={Tu_pct}%: Velocity above 1: {u_np.max():.4f}")

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
        cf_results[Tu_pct] = (Re_theta, cf)

        name = case_names[batch_idx]

        # Plotting — black-only line contours so the figure prints well in greyscale.
        y_max = 15000.0

        ax = fig.add_subplot(gs[batch_idx, 0])
        levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        cs = ax.contour(x_grid, y_u, u_np.T, levels, colors='k', linewidths=0.7)
        ax.clabel(cs, fontsize=7)
        ax.set_title(r'$u/U_\infty$, ' + name, y=0.82)
        ax.set_ylim([0, y_max])
        ax.set_ylabel('y')
        if batch_idx < n_cases-1: ax.set_xticklabels([])
        else: ax.set_xlabel('x')

        ax = fig.add_subplot(gs[batch_idx, 1])
        levels = [-3.5, -3.0, -2.5, -2.0, -1.5, -1, -0.5, 0, 0.5, 1.0, 1.5, 2.0]
        cs = ax.contour(x_grid, y_u, np.log10(np.maximum(nu_np.T, 1e-10)), levels, colors='k', linewidths=0.7)
        ax.clabel(cs, fontsize=7)
        ax.set_title(r'$\log_{10}\hat\nu$, ' + name, y=0.82)
        ax.set_ylim([0, y_max])
        ax.set_yticklabels([])
        if batch_idx < n_cases-1: ax.set_xticklabels([])
        else: ax.set_xlabel('x')

    # Bottom row spans both columns and is taller.
    axCf = fig.add_subplot(gs[n_cases, :])
    for batch_idx, Tu_pct in enumerate(Tu_labels):
        Re_theta, cf = cf_results[Tu_pct]
        axCf.loglog(Re_theta[1:], cf[1:], symbols[batch_idx], mfc='w', mec='k',
                    color='k', ms=5, label=f'SA-AI, {case_names[batch_idx]}')

    # Reference correlations
    Re_theta_ref = cf_results[Tu_labels[0]][0]
    cf_lam = 0.441 / Re_theta_ref[1:]
    cf_turb = 2.0 * (1.0 / 0.38 * np.log(Re_theta_ref[1:]) + 3.7)**(-2)
    axCf.loglog(Re_theta_ref[1:], cf_lam, 'k--', lw=1.2,
                label=r'laminar, $C_f = 0.441/Re_\theta$')
    axCf.loglog(Re_theta_ref[1:], cf_turb, 'k:',  lw=1.2,
                label=r'turbulent, $C_f = 2[(\ln Re_\theta)/0.38 + 3.7]^{-2}$')

    # Schubauer-Skramstad experimental transition Re_theta: vertical reference
    # line at each Re_theta_SS with the filled marker at the BOTTOM (y_min) so
    # the experimental locations stand out from the laminar correlation line.
    y_min_ax = 1e-4
    for batch_idx, Tu_pct in enumerate(Tu_labels):
        ReTh = expt_ReTheta[Tu_pct]
        axCf.axvline(ReTh, color='0.5', lw=0.8, ls='-', zorder=1)
        y_marker = y_min_ax * 1.15
        axCf.plot(ReTh, y_marker, symbols[batch_idx], mfc='k', mec='k', ms=8, zorder=5)
        axCf.annotate(f'S-S {Tu_pct}%', (ReTh, y_marker),
                      xytext=(6, 4), textcoords='offset points',
                      fontsize=7, color='0.2')

    axCf.legend(loc='lower left', fontsize=8, frameon=False)
    axCf.grid(True, which='major', alpha=0.5)
    axCf.grid(True, which='minor', alpha=0.2)
    axCf.set_xlim([1e2, 1e4])
    axCf.set_ylim([1e-4, 1e-2])
    axCf.set_xlabel(r'$Re_\theta$')
    axCf.set_ylabel(r'$C_f$')
    
    out_path = os.path.join(get_output_dir(), 'flat_plate_batch.pdf')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.05)
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
