#!/usr/bin/env python3
"""
Test wall flux computation with the fix.

This script verifies that the viscous flux at the wall is computed correctly
using the BC-applied ghost cells.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import jax.numpy as jnp

from src.config.loader import load_yaml
from src.grid.loader import load_or_generate_grid
from src.grid.metrics import MetricComputer
from src.numerics.viscous_fluxes import compute_viscous_fluxes_tight_with_ghosts_jax
from src.constants import NGHOST


def main():
    # Load config
    config = load_yaml('config/examples/naca0012_re1m.yaml')
    
    # Load grid (small for testing)
    grid_params = config.grid
    X, Y = load_or_generate_grid(
        grid_params.airfoil,
        n_surface=65,  # Coarse
        n_normal=17,   # Coarse
        n_wake=16,
        y_plus=1.0,
        reynolds=1e6,
        farfield_radius=10.0,
        max_first_cell=0.005,
        verbose=False
    )
    
    NI = X.shape[0] - 1
    NJ = X.shape[1] - 1
    
    print(f"Grid size: {NI} x {NJ}")
    
    # Compute metrics
    mc = MetricComputer(X, Y)
    metrics = mc.compute()
    face_geom = mc.compute_face_geometry()
    ls_weights = mc.compute_ls_weights(face_geom)
    
    # Create a simple test Q: linear velocity profile near wall
    # Q shape with ghosts: (NI + 2*NGHOST, NJ + 2*NGHOST, 4)
    Q_full = np.zeros((NI + 2*NGHOST, NJ + 2*NGHOST, 4))
    
    nu = 1e-6  # Laminar viscosity at Re=1M
    
    # Set up a simple velocity profile: u = u_tau^2 / nu * y
    # For Cf = 0.004, u_tau = sqrt(0.002) ≈ 0.0447
    u_tau = 0.045
    
    # Wall distance for interior cells
    wall_dist = metrics.wall_distance  # (NI, NJ)
    
    # Set interior velocity: u = u_tau^2 / nu * wall_dist
    for i in range(NI):
        for j in range(NJ):
            y = wall_dist[i, j]
            u = u_tau**2 / nu * y  # Linear in viscous sublayer
            Q_full[i + NGHOST, j + NGHOST, 1] = u  # u-velocity
            Q_full[i + NGHOST, j + NGHOST, 2] = 0  # v-velocity
            Q_full[i + NGHOST, j + NGHOST, 3] = 1e-7  # nuHat (small)
    
    # Apply wall BC: ghost velocity = -interior velocity
    # Ghost layer 1 (j=1 in Q_full)
    i_start = NGHOST  # First interior i
    i_end = NI + NGHOST  # Last interior i + 1
    Q_full[i_start:i_end, 1, 1] = -Q_full[i_start:i_end, NGHOST, 1]  # u = -u_int
    Q_full[i_start:i_end, 1, 2] = -Q_full[i_start:i_end, NGHOST, 2]  # v = -v_int
    Q_full[i_start:i_end, 1, 3] = -Q_full[i_start:i_end, NGHOST, 3]  # nuHat = -nuHat_int
    
    # Ghost layer 0 (j=0 in Q_full) - extrapolate
    Q_full[i_start:i_end, 0, :] = 2*Q_full[i_start:i_end, 1, :] - Q_full[i_start:i_end, NGHOST, :]
    
    print(f"\nSetup:")
    print(f"  u_tau = {u_tau}")
    print(f"  nu = {nu}")
    print(f"  Expected tau_wall = rho * u_tau^2 = {u_tau**2:.6f}")
    print(f"  Expected Cf = 2 * u_tau^2 = {2*u_tau**2:.5f}")
    
    # Check ghost cell values
    i_mid = NI // 2 + NGHOST
    print(f"\nAt mid-airfoil (i={i_mid}):")
    print(f"  Interior j=0: u = {Q_full[i_mid, NGHOST, 1]:.6f}")
    print(f"  Ghost j=1:    u = {Q_full[i_mid, 1, 1]:.6f} (should be -{Q_full[i_mid, NGHOST, 1]:.6f})")
    print(f"  wall_dist[j=0] = {wall_dist[i_mid-NGHOST, 0]:.2e}")
    
    # Extract Q_with_ghosts for the new function
    Q_with_ghosts = jnp.asarray(Q_full[NGHOST-1:-(NGHOST-1), NGHOST-1:-(NGHOST-1), :])
    print(f"\nQ_with_ghosts shape: {Q_with_ghosts.shape}")
    print(f"Expected: ({NI+2}, {NJ+2}, 4)")
    
    # Check Q_with_ghosts values
    print(f"\nQ_with_ghosts at mid-airfoil:")
    print(f"  j=0 (ghost):    u = {Q_with_ghosts[i_mid-NGHOST+1, 0, 1]:.6f}")
    print(f"  j=1 (interior): u = {Q_with_ghosts[i_mid-NGHOST+1, 1, 1]:.6f}")
    
    # Prepare geometry arrays
    Si_x = jnp.asarray(metrics.Si_x)
    Si_y = jnp.asarray(metrics.Si_y)
    Sj_x = jnp.asarray(metrics.Sj_x)
    Sj_y = jnp.asarray(metrics.Sj_y)
    
    d_coord_i = jnp.asarray(face_geom.d_coord_i)
    e_coord_i_x = jnp.asarray(face_geom.e_coord_i_x)
    e_coord_i_y = jnp.asarray(face_geom.e_coord_i_y)
    e_ortho_i_x = jnp.asarray(face_geom.e_ortho_i_x)
    e_ortho_i_y = jnp.asarray(face_geom.e_ortho_i_y)
    
    d_coord_j = jnp.asarray(face_geom.d_coord_j)
    e_coord_j_x = jnp.asarray(face_geom.e_coord_j_x)
    e_coord_j_y = jnp.asarray(face_geom.e_coord_j_y)
    e_ortho_j_x = jnp.asarray(face_geom.e_ortho_j_x)
    e_ortho_j_y = jnp.asarray(face_geom.e_ortho_j_y)
    
    ls_weights_i = jnp.asarray(ls_weights.weights_i)
    ls_weights_j = jnp.asarray(ls_weights.weights_j)
    
    # Interior mu_eff and nuHat
    mu_eff = jnp.full((NI, NJ), nu)  # Laminar only for test
    nuHat = jnp.asarray(Q_full[NGHOST:-NGHOST, NGHOST:-NGHOST, 3])
    
    # Compute viscous flux residual
    R_visc = compute_viscous_fluxes_tight_with_ghosts_jax(
        Q_with_ghosts, Si_x, Si_y, Sj_x, Sj_y,
        d_coord_i, e_coord_i_x, e_coord_i_y, e_ortho_i_x, e_ortho_i_y,
        d_coord_j, e_coord_j_x, e_coord_j_y, e_ortho_j_x, e_ortho_j_y,
        ls_weights_i, ls_weights_j,
        mu_eff, nu, nuHat
    )
    
    R_visc = np.asarray(R_visc)
    
    # Check residual at first interior cell
    i_check = i_mid - NGHOST
    print(f"\nViscous residual at mid-airfoil, j=0 (first interior cell):")
    print(f"  R_u = {R_visc[i_check, 0, 1]:.6e}")
    print(f"  R_v = {R_visc[i_check, 0, 2]:.6e}")
    
    # For a linear velocity profile, the viscous flux at interior faces should be
    # approximately constant, so the net flux into the first cell should be
    # approximately the wall flux (since flux[j+1] - flux[j=0]).
    
    # The wall shear stress is tau = mu * du/dy = mu * u/y (at first cell)
    # tau = nu * u_int / wall_dist[0] = nu * (u_tau^2/nu * wall_dist[0]) / wall_dist[0] = u_tau^2
    tau_expected = u_tau**2
    print(f"\nExpected wall shear tau = {tau_expected:.6f}")
    
    # The viscous flux residual for the first cell should be related to tau * S
    # where S is the face area
    Sj_mag = np.sqrt(metrics.Sj_x[i_check, 0]**2 + metrics.Sj_y[i_check, 0]**2)
    print(f"Wall face area |Sj| = {Sj_mag:.6f}")
    print(f"Expected flux ~ tau * S = {tau_expected * Sj_mag:.6e}")
    
    # The residual sign should be NEGATIVE (removes momentum from cell = drag)
    if R_visc[i_check, 0, 1] < 0:
        print("\n✓ Residual is NEGATIVE (correct for wall drag)")
    else:
        print("\n✗ Residual is POSITIVE (wrong! should be negative for drag)")


if __name__ == '__main__':
    main()
