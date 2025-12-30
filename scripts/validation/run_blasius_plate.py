#!/usr/bin/env python3
"""
Blasius Flat Plate Validation Case.

Validates the viscous solver against the Blasius solution.
Cf * sqrt(Re_x) = 0.664
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.numerics.gradients import compute_gradients, GradientMetrics
from src.numerics.fluxes import compute_fluxes, FluxConfig, GridMetrics as FluxGridMetrics
from src.numerics.viscous_fluxes import add_viscous_fluxes


def create_grid(NI, NJ, L, H):
    """Create simple uniform grid."""
    x = np.linspace(0, L, NI + 1)
    y = np.linspace(0, H, NJ + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return X, Y


def compute_metrics(X, Y):
    """Compute FVM metrics."""
    NI, NJ = X.shape[0] - 1, X.shape[1] - 1
    dx = X[1, 0] - X[0, 0]
    dy = Y[0, 1] - Y[0, 0]
    
    Si_x = np.full((NI + 1, NJ), dy)
    Si_y = np.zeros((NI + 1, NJ))
    Sj_x = np.zeros((NI, NJ + 1))
    Sj_y = np.full((NI, NJ + 1), dx)
    volume = np.full((NI, NJ), dx * dy)
    
    return FluxGridMetrics(Si_x, Si_y, Sj_x, Sj_y, volume), \
           GradientMetrics(Si_x, Si_y, Sj_x, Sj_y, volume), dx, dy


def apply_bc(Q, u_inf=1.0):
    """Apply boundary conditions."""
    Q = Q.copy()
    # Inlet (i=0): Dirichlet velocity
    Q[0, :, 1] = 2 * u_inf - Q[1, :, 1]
    Q[0, :, 2] = -Q[1, :, 2]
    Q[0, :, 0] = Q[1, :, 0]
    
    # Outlet (i=-1): Zero gradient (convective outflow)
    Q[-1, :, :] = Q[-2, :, :]
    
    # Wall (j=0): No-slip
    Q[:, 0, 0] = Q[:, 1, 0]
    Q[:, 0, 1] = -Q[:, 1, 1]
    Q[:, 0, 2] = -Q[:, 1, 2]
    
    # Top (j=-1): Zero gradient (allow v > 0 outflow for displacement)
    Q[:, -1, 0] = Q[:, -2, 0]  # Neumann pressure
    Q[:, -1, 1] = Q[:, -2, 1]  # Neumann u (not forced to u_inf)
    Q[:, -1, 2] = Q[:, -2, 2]  # Neumann v (allow upward outflow!)
    
    return Q


def plot_solution(X, Y, Q, output_dir, iteration, Re):
    """Plot velocity contours and BL profiles."""
    NI, NJ = X.shape[0] - 1, X.shape[1] - 1
    
    # Cell centers
    Xc = 0.5 * (X[:-1, :-1] + X[1:, 1:])
    Yc = 0.5 * (Y[:-1, :-1] + Y[1:, 1:])
    
    # Interior values (remove ghost cells)
    u = Q[1:-1, 1:-1, 1]
    v = Q[1:-1, 1:-1, 2]
    
    fig = plt.figure(figsize=(16, 12))
    
    # U velocity contour
    ax1 = fig.add_subplot(2, 2, 1)
    c1 = ax1.contourf(Xc, Yc, u, levels=50, cmap='RdBu_r')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'u velocity (iter {iteration})')
    ax1.set_aspect('equal')
    plt.colorbar(c1, ax=ax1)
    
    # V velocity contour
    ax2 = fig.add_subplot(2, 2, 2)
    c2 = ax2.contourf(Xc, Yc, v, levels=50, cmap='RdBu_r')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(f'v velocity (iter {iteration})')
    ax2.set_aspect('equal')
    plt.colorbar(c2, ax=ax2)
    
    # Boundary layer profiles at 4 locations
    ax3 = fig.add_subplot(2, 2, 3)
    x_locs = [0.2, 0.4, 0.6, 0.8]
    colors = ['b', 'g', 'r', 'm']
    
    for x_loc, color in zip(x_locs, colors):
        i = int(x_loc * NI)
        y_profile = Yc[i, :]
        u_profile = u[i, :]
        
        # Blasius: delta = 5x / sqrt(Re_x)
        Re_x = Re * x_loc
        delta = 5.0 * x_loc / np.sqrt(Re_x) if Re_x > 0 else 0.01
        
        ax3.plot(u_profile, y_profile / delta, f'{color}-', linewidth=2, 
                 label=f'x={x_loc:.1f}')
    
    ax3.set_xlabel('u/U∞')
    ax3.set_ylabel('y/δ')
    ax3.set_title('Boundary Layer Profiles')
    ax3.legend()
    ax3.set_xlim([0, 1.2])
    ax3.set_ylim([0, 3])
    ax3.grid(True, alpha=0.3)
    
    # Cf * sqrt(Re_x) vs x
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Compute wall shear
    u_wall = Q[1:-1, 1, 1]  # First interior cell
    y_first = 0.5 * (Y[0, 0] + Y[0, 1])
    nu = 1.0 / Re
    tau_w = nu * u_wall / y_first
    Cf = 2.0 * tau_w
    
    x_wall = Xc[:, 0]
    Re_x = Re * x_wall
    cf_rex = Cf * np.sqrt(Re_x + 1e-12)
    
    ax4.plot(x_wall, cf_rex, 'b-', linewidth=2, label='Computed')
    ax4.axhline(y=0.664, color='r', linestyle='--', linewidth=2, label='Blasius (0.664)')
    ax4.set_xlabel('x')
    ax4.set_ylabel('Cf * sqrt(Re_x)')
    ax4.set_title('Skin Friction')
    ax4.legend()
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 2])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'solution_{iteration:05d}.pdf', dpi=150)
    plt.close()
    print(f"  Saved: solution_{iteration:05d}.pdf")


def run():
    """Run flat plate simulation."""
    print("=" * 60)
    print("BLASIUS FLAT PLATE VALIDATION")
    print("=" * 60)
    
    output_dir = project_root / 'output' / 'validation' / 'blasius'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Grid parameters
    NI = 100      # x cells
    NJ = 50       # y cells  
    L = 1.0       # plate length
    H = 0.1       # domain height (> δ ≈ 0.05)
    
    # Flow parameters
    Re = 10000
    nu = 1.0 / Re
    beta = 5.0    # artificial compressibility
    
    # Numerics
    cfl = 0.5     # Increase CFL for faster convergence
    jst_k2 = 0.0
    jst_k4 = 0.002
    max_iter = 50000
    dump_freq = 10000
    
    print(f"\nGrid: {NI} x {NJ}, L={L}, H={H}")
    print(f"Re = {Re}, nu = {nu:.2e}")
    print(f"beta = {beta}, CFL = {cfl}")
    print(f"JST: k2={jst_k2}, k4={jst_k4}")
    
    # Create grid and metrics
    X, Y = create_grid(NI, NJ, L, H)
    flux_met, grad_met, dx, dy = compute_metrics(X, Y)
    
    print(f"dx = {dx:.4f}, dy = {dy:.4f}")
    print(f"Aspect ratio: dx/dy = {dx/dy:.1f}")
    
    # Time step
    dt_conv = cfl * min(dx, dy) / (1.0 + np.sqrt(beta))
    dt_visc = cfl * min(dx, dy)**2 / (4 * nu)
    dt = min(dt_conv, dt_visc)
    print(f"dt = {dt:.2e} (conv={dt_conv:.2e}, visc={dt_visc:.2e})")
    
    # Initialize
    Q = np.zeros((NI + 2, NJ + 2, 4))
    Q[:, :, 1] = 1.0  # u = 1
    Q = apply_bc(Q)
    
    # Dump initial
    plot_solution(X, Y, Q, output_dir, 0, Re)
    
    # Run
    print(f"\nRunning {max_iter} iterations...")
    print(f"{'Iter':>8} {'Residual':>12} {'Max u':>8} {'Max v':>8}")
    print("-" * 40)
    
    flux_cfg = FluxConfig(k2=jst_k2, k4=jst_k4)
    
    for n in range(max_iter):
        Q0 = Q.copy()
        Qk = Q.copy()
        
        for alpha in [0.25, 0.333, 0.5, 1.0]:
            Qk = apply_bc(Qk)
            
            # Convective flux
            R = compute_fluxes(Qk, flux_met, beta, flux_cfg)
            
            # Viscous flux
            grad = compute_gradients(Qk, grad_met)
            R = add_viscous_fluxes(R, Qk, grad, grad_met, mu_laminar=nu)
            
            Qk = Q0.copy()
            Qk[1:-1, 1:-1, :] += alpha * dt / flux_met.volume[:, :, np.newaxis] * R
        
        Q = apply_bc(Qk)
        
        # Residual
        res = np.sqrt(np.mean(R[:, :, 0]**2))
        
        if (n + 1) % 1000 == 0 or n == 0:
            u_max = Q[1:-1, 1:-1, 1].max()
            v_max = np.abs(Q[1:-1, 1:-1, 2]).max()
            print(f"{n+1:>8d} {res:>12.4e} {u_max:>8.4f} {v_max:>8.4f}")
        
        if (n + 1) % dump_freq == 0:
            plot_solution(X, Y, Q, output_dir, n + 1, Re)
        
        if np.isnan(res):
            print("DIVERGED!")
            break
    
    # Final dump
    plot_solution(X, Y, Q, output_dir, max_iter, Re)
    
    # Final validation
    u_wall = Q[1:-1, 1, 1]
    y_first = 0.5 * dy
    tau_w = nu * u_wall / y_first
    Cf = 2.0 * tau_w
    x_wall = 0.5 * (X[:-1, 0] + X[1:, 0])
    Re_x = Re * x_wall
    cf_rex = Cf * np.sqrt(Re_x + 1e-12)
    
    mask = x_wall > 0.1
    mean_cf_rex = cf_rex[mask].mean()
    
    print(f"\n{'='*60}")
    print(f"RESULT: Cf*sqrt(Re_x) = {mean_cf_rex:.4f} (Blasius = 0.664)")
    print(f"Error: {abs(mean_cf_rex - 0.664)/0.664 * 100:.1f}%")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    exit(run())
