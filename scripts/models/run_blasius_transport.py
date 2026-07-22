#!/usr/bin/env python3
"""
Blasius Boundary Layer Transport Solver (JAX Version).

This script solves the nuHat transport equation on a Blasius boundary layer
and visualizes the growth of disturbances.

Uses JAX for GPU acceleration and improved performance.
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
from src.solvers.boundary_layer_solvers import NuHatBlasiusSolver


def run():
    solver = NuHatBlasiusSolver()
    nuHat = solver(1.0)
    
    # Convert to numpy for plotting
    nuHat_np = np.array(nuHat)
    
    x = np.arange(solver.nx + 1) * solver.dx
    y = (np.arange(solver.ny) + 0.5) * solver.dy
    
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 4.1))

    # (a) contour map of N = ln(nuHat) in the (Re_x, Re_y) plane
    axA.plot(x, 0.665 * np.sqrt(x), '--k')
    axA.plot(x, 1.72 * np.sqrt(x), '-k')
    axA.plot(x, 5 * np.sqrt(x), ':k')
    axA.set_ylim([0, 9000])
    axA.legend([r'$\theta$', r'$\delta^*$', r'$\delta_{99}$'], loc='upper left')
    levels = np.linspace(0, 14, 15)
    levels[0] = 0.1
    cs = axA.contour(x, y, np.log(nuHat_np.T), levels, colors='k', linewidths=0.7)
    axA.clabel(cs, fontsize=8)
    axA.set_xlabel(r'$Re_x$')
    axA.set_ylabel(r'$Re_y$')
    axA.text(0.02, 0.02, '(a)', transform=axA.transAxes, fontsize=13, fontweight='bold')

    # (b) the same solution as an envelope: max_y nuHat vs Re_theta (log left,
    # N linear right), against the Drela-Giles Blasius envelope (zero to
    # Re_theta_crit, then slope dn/dRe_theta at H = 2.59)
    Re_th = 0.664 * np.sqrt(x)
    max_nu = np.maximum(nuHat_np.max(axis=1), 1.0)
    H = 2.591
    slope = 0.01 * np.sqrt((2.4*H - 3.7 + 2.5*np.tanh(1.5*H - 4.65))**2 + 0.25)
    log10_recrit = ((1.415/(H - 1) - 0.489)*np.tanh(20.0/(H - 1) - 12.9)
                    + 3.295/(H - 1) + 0.44)
    recrit = 10.0**log10_recrit
    N_dr = np.clip(slope*(Re_th - recrit), 0.0, None)
    NMAX = 15.0
    axB.semilogy(Re_th, max_nu, 'k-', lw=1.9, label=r'transport \(model\)')
    axB.set_xlabel(r'$Re_\theta$')
    axB.set_ylabel(r'$\max_y \hat\nu$')
    axB.set_xlim(0, Re_th[-1])
    axB.set_ylim(1.0, np.exp(NMAX))
    axB.text(0.02, 0.02, '(b)', transform=axB.transAxes, fontsize=13, fontweight='bold')
    axC = axB.twinx()
    axC.plot(Re_th, N_dr, 'k--', lw=1.7)
    axC.set_ylim(0.0, NMAX)
    axC.set_ylabel(r'$N$')
    import matplotlib.lines as mlines
    axB.legend(handles=[
        mlines.Line2D([], [], color='k', ls='-', lw=1.9,
                      label=r'model: $N=\ln\hat\nu_{\max}$'),
        mlines.Line2D([], [], color='k', ls='--', lw=1.7,
                      label=(r'Drela--Giles: $0$ to $Re_{\theta,\mathrm{crit}}$'
                             '\n' r'$=%.0f$, then slope $%.4f$' % (recrit, slope))),
    ], loc='upper left', fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(get_output_dir(), 'blasius_nuHat_solution.pdf')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.05)
    print(f'Saved: {out_path}')


if __name__ == "__main__":
    run()
