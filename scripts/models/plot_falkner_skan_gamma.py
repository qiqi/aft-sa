#!/usr/bin/env python3
"""
Plot Falkner-Skan Phase Space Diagram (JAX Version).

This script visualizes the transition phase space showing Γ vs Re_Ω
for various pressure gradient parameters.

Uses JAX versions of physics functions for consistency with GPU solvers.
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

# Use JAX versions
from src.physics.jax_config import jnp
from src.physics.laminar import Re_Omega


def run():
    # --- 2. Phase Space Calculation ---
    betas = [1.0, 0.5, 0.1, 0.0, -0.05, -0.1, -0.198]
    Re_theta_tr = [10_000, 3_000, 1_500, 1_200, 450, 300, 100]
    rate_target = [
     0.0006056181711137146,
     0.0006918126210183442,
     0.0013695543979811036,
     0.0023502079523144664,
     0.0036664580824200812,
     0.005267077397042096,
     0.025112605506072142]

    labels = ['Stagnation', 'Strong Favorable', 'Favorable', 'Blasius', 'Adverse', 'Strong Adverse', 'Separation']
    colors = plt.cm.viridis(np.linspace(0, 1, len(betas)))

    _, ax = plt.subplots(2, 1, figsize=(7, 8))

    for beta, Re_theta, label, color in zip(betas, Re_theta_tr, labels, colors):
        eta, u, shear, v_sc = solve_falkner_skan(beta)

        # Convert to JAX arrays
        shear_jax = jnp.array(shear)
        eta_jax = jnp.array(eta)

        # --- Compute Invariants ---
        gamma = 2 * (eta * shear)**2 / (u**2 + (eta * shear)**2)

        # --- Compute Shape Factor H (for reference) ---
        theta = np.trapezoid(u * (1 - u), eta)
        delta_star = np.trapezoid(1 - u, eta)
        H = delta_star / theta

        # Use JAX version of Re_Omega
        re_omega = np.array(Re_Omega(shear_jax, eta_jax))

        # --- Plotting ---
        ax[0].plot(gamma[1:], re_omega[1:] * Re_theta / theta,
                 color=color, linewidth=2.5,
                 label=f'{label} (H={H:.1f})')

        amax = np.max(np.log10(re_omega[1:] * Re_theta / theta / 1000) / 50 + gamma[1:])
        ax[1].plot(amax, rate_target[Re_theta_tr.index(Re_theta)], 'o', color=color)

    # --- 3. Decorate the Map ---
    ax[0].set_title(f"Transition Phase Space: $\\Gamma$ vs $Re_\\Omega$ (Calculated at $Re_\\theta = 500$)")
    ax[0].set_xlabel(r"Detachment Parameter $\Gamma = \frac{d \cdot \Omega}{U_{rel}}$")
    ax[0].set_ylabel(r"Vorticity Reynolds $Re_\Omega = \frac{d^2 \Omega}{\nu}$")
    ax[0].set_xlim(0.8, 1.1)
    ax[0].set_ylim(10, 10000)
    ax[0].grid(True, which='both', alpha=0.3)
    ax[0].legend(loc='upper left')

    amax = np.linspace(0.9, 1.5, 61)
    ax[1].plot(amax, 0.2 / (1 + np.exp(-3.8 * (amax - 2.01))) / (1 + np.exp(-40 * (amax - 0.97))))

    re_omega = np.logspace(1, 4, 51)
    Gamma = np.linspace(0, 2, 51)
    re_omega, Gamma = np.meshgrid(re_omega, Gamma, indexing='ij')
    a = np.log10(re_omega / 1000) / 50 + Gamma
    ax[0].clabel(ax[0].contour(Gamma, re_omega, a, [0.9, 0.95, 1, 1.05, 1.1], linestyles=':'))

    ax[0].set_yscale('log')
    ax[0].set_ylim([10, 10000])
    ax[1].set_yscale('log')
    
    out_path = os.path.join(get_output_dir(), 'falkner_skan_gamma.pdf')
    plt.savefig(out_path)
    print(f'Saved: {out_path}')


if __name__ == "__main__":
    run()
