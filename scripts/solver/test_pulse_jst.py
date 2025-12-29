#!/usr/bin/env python
"""
Visual test: Artificial Compressibility Pressure Pulse.

For a real compressible code, you use a Shock Tube. For Artificial
Compressibility, the equivalent is the Pressure Pulse.

Since AC creates artificial acoustic waves, if you place a high-pressure
"dot" in the middle of a quiescent domain, it should expand as a perfect circle.

Diagnostics:
- Diamond shape: Flux splitting too aligned with grid (upwinding error)
- Square shape: Metric terms are wrong
- Noise trailing: 4th order dissipation (k4) too low
- Smears instantly: 2nd order dissipation (k2) too high
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path (go up two levels: solver/ -> scripts/ -> project root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.numerics.fluxes import compute_fluxes, FluxConfig, GridMetrics

def run_pulse_test():
    # 1. Setup Grid
    NI, NJ = 80, 80
    dx, dy = 1.0, 1.0
    
    # 2. Initial State (Quiescent with Pressure Pulse)
    Q = np.zeros((NI + 2, NJ + 2, 4))
    Q[:, :, 0] = 1.0  # Background pressure
    Q[:, :, 1] = 0.0  # u = 0 (quiescent)
    Q[:, :, 2] = 0.0  # v = 0 (quiescent)
    Q[:, :, 3] = 1e-6 # Small nu_tilde
    
    # Gaussian Pulse in center
    cx, cy = (NI + 2) // 2, (NJ + 2) // 2
    sigma = 3.0  # Pulse width
    for i in range(NI + 2):
        for j in range(NJ + 2):
            dist_sq = (i - cx)**2 + (j - cy)**2
            Q[i, j, 0] += 1.0 * np.exp(-dist_sq / (2 * sigma**2))

    # 3. Physics Params
    beta = 10.0  # Artificial compressibility factor
    
    # CFL-based time step: dt < CFL * dx / c_art, where c_art = sqrt(beta)
    c_art = np.sqrt(beta)
    cfl = 0.5
    dt = cfl * dx / c_art
    
    cfg = FluxConfig(k2=0.5, k4=0.02)  # Standard JST coeffs
    
    # Grid Metrics (Cartesian)
    vol = np.ones((NI, NJ)) * dx * dy
    metrics = GridMetrics(
        Si_x=np.ones((NI+1, NJ)) * dy, Si_y=np.zeros((NI+1, NJ)),
        Sj_x=np.zeros((NI, NJ+1)),     Sj_y=np.ones((NI, NJ+1)) * dx,
        volume=vol
    )

    # 4. Run explicit time stepping
    print("Running Pulse Propagation...")
    print(f"  Grid: {NI}x{NJ}, beta={beta}, CFL={cfl}, dt={dt:.4f}")
    
    steps = 100
    plot_interval = 25
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    plot_idx = 0
    
    for n in range(steps + 1):
        if n % plot_interval == 0 and plot_idx < 4:
            ax = axes[plot_idx]
            p = Q[1:-1, 1:-1, 0].T
            im = ax.imshow(p, origin='lower', cmap='RdBu_r', 
                          vmin=0.8, vmax=2.2, extent=[0, NI, 0, NJ])
            ax.set_title(f"Step {n}, t = {n*dt:.2f}")
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
            
            # Draw reference circle
            theta = np.linspace(0, 2*np.pi, 100)
            radius = c_art * n * dt  # Expected wave position
            ax.plot(NI/2 + radius * np.cos(theta), 
                   NJ/2 + radius * np.sin(theta), 
                   'k--', lw=1, label=f'r = c·t = {radius:.1f}')
            if radius > 0:
                ax.legend(loc='upper right', fontsize=8)
            
            plot_idx += 1
        
        if n < steps:
            # Compute Residuals
            R = compute_fluxes(Q, metrics, beta, cfg)
            
            # Explicit Euler update: Q_new = Q_old + dt/Vol * R
            # (R is defined as net flux INTO cell, so positive R = accumulation)
            Q[1:-1, 1:-1, :] += (dt / vol[:, :, np.newaxis]) * R
            
            # Neumann BCs (copy to ghost cells)
            Q[0, :, :] = Q[1, :, :]
            Q[-1, :, :] = Q[-2, :, :]
            Q[:, 0, :] = Q[:, 1, :]
            Q[:, -1, :] = Q[:, -2, :]

    plt.suptitle("AC Pulse Test: Should be Circular\n"
                 "(Diamond = upwinding error, Square = metric error, "
                 "Noise = low k4, Smeared = high k2)", fontsize=10)
    plt.tight_layout()
    plt.savefig("test_pulse.png", dpi=150)
    print("Generated 'test_pulse.png'. Open it to verify circular symmetry.")
    
    # Check final state for NaN/Inf
    if np.any(~np.isfinite(Q)):
        print("WARNING: Solution contains NaN or Inf!")
        return False
    
    print(f"  Final pressure range: [{Q[:,:,0].min():.3f}, {Q[:,:,0].max():.3f}]")
    
    # Quantitative symmetry check
    p = Q[1:-1, 1:-1, 0]
    center = NI // 2
    r_check = 20
    
    # Check 4-fold symmetry (should be perfect)
    p_left = p[center - r_check, center]
    p_right = p[center + r_check, center]
    p_top = p[center, center + r_check]
    p_bot = p[center, center - r_check]
    
    sym_error = max(abs(p_left - p_right), abs(p_top - p_bot), 
                   abs(p_left - p_top))
    print(f"  Symmetry error (L/R/T/B): {sym_error:.2e}")
    
    # Check circularity (axis vs diagonal at same radius)
    # For radius r: axis point is at (r, 0), diagonal point is at (r/√2, r/√2)
    r_diag = int(r_check / np.sqrt(2))  # ~14 for r_check=20
    p_axis = p[center + r_diag, center]  # Use same r as diagonal for fair comparison
    p_diag = p[center + r_diag, center + r_diag]  # actual r = r_diag * √2 ≈ r_check
    # Actually let's compare at same actual radius
    # Axis at r=14: (center+14, center), actual r=14
    # Diagonal at r≈14: (center+10, center+10), actual r=14.14
    r_axis = 14
    r_d = 10  # 10*sqrt(2) ≈ 14.14
    p_axis = p[center + r_axis, center]
    p_diag = p[center + r_d, center + r_d]
    circ_diff = abs(p_axis - p_diag)
    print(f"  Circularity (axis vs diag): {circ_diff:.4f}")
    
    if circ_diff > 0.3:
        print("  NOTE: Diamond/square shape detected - check metrics or flux splitting")
    elif circ_diff > 0.1:
        print("  NOTE: Slight grid anisotropy (normal for JST scalar dissipation)")
    else:
        print("  Pulse is circular (good!)")
    
    return True

if __name__ == "__main__":
    run_pulse_test()