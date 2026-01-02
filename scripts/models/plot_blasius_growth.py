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

    fig, ax = plt.subplots(1, 4, figsize=(10, 4), sharey='row')
    axp = [a.twiny() for a in ax]
    
    all_passed = True

    for Rex in Rex_list:
        yCell, u, dudy, _ = b.at(Rex, y)
        
        # Convert to JAX arrays
        u_jax = jnp.array(u)
        dudy_jax = jnp.array(dudy)
        yCell_jax = jnp.array(yCell)
        
        # ===== PHYSICAL ASSERTIONS =====
        
        # 1. Velocity should be bounded [0, 1]
        if u.min() < -0.01:
            print(f"❌ Rex={Rex}: Velocity below 0: {u.min():.4f}")
            all_passed = False
        if u.max() > 1.01:
            print(f"❌ Rex={Rex}: Velocity above 1: {u.max():.4f}")
            all_passed = False
        
        # 2. Shear should be non-negative (no flow reversal in Blasius)
        if (dudy < -1e-10).any():
            print(f"❌ Rex={Rex}: Negative shear detected")
            all_passed = False
        
        # 3. Shape factor Gamma should be in [0, 2]
        valid = u > 0.01
        Gamma = 2 * (dudy[valid] * yCell[valid])**2 / (u[valid]**2 + (dudy[valid] * yCell[valid])**2)
        if Gamma.min() < -0.01 or Gamma.max() > 2.01:
            print(f"❌ Rex={Rex}: Gamma out of bounds [{Gamma.min():.3f}, {Gamma.max():.3f}]")
            all_passed = False
        
        # 4. Amplification should be non-negative and bounded (using JAX version)
        amp_jax = amplification(u_jax, dudy_jax, yCell_jax)
        amp = np.array(amp_jax)
        if (amp < -1e-10).any():
            print(f"❌ Rex={Rex}: Negative amplification")
            all_passed = False
        if amp.max() > 1.0:
            print(f"❌ Rex={Rex}: Amplification too high: {amp.max():.4f}")
            all_passed = False
        
        # 5. Re_Omega should be non-negative (using JAX version)
        re_omega_jax = Re_Omega(dudy_jax, yCell_jax)
        re_omega = np.array(re_omega_jax)
        if (re_omega < -1e-10).any():
            print(f"❌ Rex={Rex}: Negative Re_Omega")
            all_passed = False

        # Plotting
        for a in axp:
            a.plot(u, yCell, '--', linewidth=2, alpha=0.5)
        ax[0].plot(re_omega / 2.2, yCell, linewidth=3)
        ax[1].plot(2 * (dudy * yCell)**2 / (u**2 + (dudy * yCell)**2 + 1e-20), yCell, linewidth=3)
        ax[2].plot(amp * 1e3, yCell, linewidth=3)
        ax[3].plot(dudy * 1000, yCell, linewidth=3)

    for a in ax:
        a.set_ylim([0, 5000])
    for a in axp:
        a.set_xticks([])
        
    ax[0].legend([f'$\\theta = {0.665 * np.sqrt(Rex):.1f}$' for Rex in Rex_list], loc='upper left')
    ax[0].set_ylabel(r"$y$")
    ax[0].set_xlabel(r"$Re_{\Omega} / 2.2$")
    ax[1].set_xlabel(r"$\Gamma$")
    ax[2].set_xlabel(r"amplification $(\times 10^{-3})$")
    ax[3].set_xlabel(r"$dU/dy (\times 10^{-3})$")
    
    out_path = os.path.join(get_output_dir(), 'blasius_ReOmega_growth.pdf')
    plt.savefig(out_path)
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
