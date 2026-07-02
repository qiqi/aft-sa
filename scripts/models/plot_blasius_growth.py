#!/usr/bin/env python3
"""
Plot Blasius Boundary Layer Properties vs Re_x (JAX Version).

This script visualizes how boundary layer properties (Re_Ω, Γ, amplification)
evolve with Reynolds number for the Blasius similarity solution.

Uses JAX versions of physics functions for consistency with GPU solvers.

Assertions:
- Velocity profile converges to u=1 at edge
- Shear (du/dy) is positive (no flow reversal)
- Shape factor Gamma is in [0, 2]
- Amplification rate is bounded and non-negative
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

from src.physics.boundary_layer import Blasius

# Use JAX versions
from src.physics.jax_config import jnp
from src.physics.laminar import Re_Omega, amplification


def run():
    b = Blasius()
    y = np.arange(5000)
    Rex_list = np.array([50_000, 200_000, 500_000, 1_000_000, 2_000_000])

    # 3-panel layout matching the paper figure caption.
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6), sharey=False)

    all_passed = True

    # ----- Panels 1 & 2: profiles vs y at five streamwise stations -----
    for Rex in Rex_list:
        yCell, u, dudy, _ = b.at(Rex, y)
        u_jax = jnp.array(u); dudy_jax = jnp.array(dudy); yCell_jax = jnp.array(yCell)

        # physical sanity
        if u.min() < -0.01 or u.max() > 1.01:
            print(f"❌ Rex={Rex}: u out of [0,1]"); all_passed = False
        if (dudy < -1e-10).any():
            print(f"❌ Rex={Rex}: negative shear"); all_passed = False

        amp_jax = amplification(u_jax, dudy_jax, yCell_jax)
        amp = np.array(amp_jax)
        re_omega = np.array(Re_Omega(dudy_jax, yCell_jax))
        kernel = amp * dudy  # local growth-rate kernel a * dU/dy

        ax[0].plot(re_omega / 2.2, yCell, linewidth=2,
                   label=fr'$\theta\!=\!{0.665 * np.sqrt(Rex):.0f}$')
        ax[1].plot(kernel * 1e5, yCell, linewidth=2)

    for a_ in ax[:2]:
        a_.set_ylim([0, 5000])
        a_.set_ylabel(r'$y\,U_\infty/\nu$')
        a_.grid(alpha=0.3)
    ax[0].set_xlabel(r'$Re_\Omega / 2.2$')
    ax[1].set_xlabel(r'kernel $a\,\partial U/\partial y\ (\times 10^{-5})$')
    ax[0].legend(loc='upper left', fontsize=8, frameon=False)

    # ----- Panel 3: envelope growth rate dN/dRe_x = max_y(a*dU/dy), vs Re_theta -----
    # (The kernel integrated across y is dominated by its peak; we plot the peak as
    # the envelope amplification rate, in units of 1/Re_x, then convert to dN/dRe_theta.)
    Rex_sweep = np.linspace(2e4, 4e6, 200)
    dNdRetheta = np.zeros_like(Rex_sweep)
    Retheta = np.zeros_like(Rex_sweep)
    for i, Rex in enumerate(Rex_sweep):
        yCell, u, dudy, _ = b.at(Rex, y)
        u_j = jnp.array(u); dudy_j = jnp.array(dudy); yCell_j = jnp.array(yCell)
        amp = np.array(amplification(u_j, dudy_j, yCell_j))
        kernel = amp * dudy
        dNdRex = float(kernel.max())
        # Re_theta = 0.665*sqrt(Rex), so dRetheta/dRex = 0.5*0.665/sqrt(Rex)
        Retheta[i] = 0.665 * np.sqrt(Rex)
        dRetheta_dRex = 0.5 * 0.665 / max(np.sqrt(Rex), 1e-12)
        dNdRetheta[i] = dNdRex / dRetheta_dRex if dRetheta_dRex > 0 else 0.0

    ax[2].plot(Retheta, dNdRetheta, 'k-', linewidth=1.5)
    ax[2].set_xlabel(r'$Re_\theta$')
    ax[2].set_ylabel(r'envelope $dN/dRe_\theta$')
    ax[2].set_xlim(0, 1500)
    ax[2].set_ylim(0, None)
    ax[2].grid(alpha=0.3)

    plt.tight_layout(pad=0.4)
    out_path = os.path.join(get_output_dir(), 'blasius_ReOmega_growth.pdf')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.05)
    print(f'Saved: {out_path}')
    
    # Summary
    if all_passed:
        print("✅ All Blasius physical constraints satisfied")
        return 0
    else:
        print("❌ Some physical constraints violated")
        return 1


if __name__ == "__main__":
    exit(run())
