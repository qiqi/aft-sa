#!/usr/bin/env python3
"""
SA Physics Diagnostic Script

This script analyzes boundary layer profiles and checks physical invariants
of the Spalart-Allmaras turbulence model.

Usage:
    python scripts/diagnose_sa_physics.py [--data output/bl_full_diagnostic.npz]

The script will:
1. Check log-layer equilibrium (P ≈ D for 30 < y+ < 100)
2. Verify chi follows the expected scaling (chi ≈ κ·y+)
3. Check near-wall behavior (fv1 → 0, correct damping)
4. Provide actionable debugging guidance
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.physics.spalart_allmaras import (
    diagnose_sa_physics,
    check_log_layer_equilibrium,
    verify_wall_scaling,
    KAPPA, CB1, CW1, CV1
)


def compute_y_plus(wall_dist, vol, Sj_x, Sj_y, Q, nu):
    """Compute y+ from boundary layer data."""
    # Wall-normal direction
    S0 = np.sqrt(Sj_x[0]**2 + Sj_y[0]**2)
    tx = -Sj_y[0] / S0
    ty = Sj_x[0] / S0
    
    # Tangential velocity
    u_tan = Q[:, 1] * tx + Q[:, 2] * ty
    if np.mean(u_tan) < 0:
        u_tan = -u_tan
    
    # Wall shear and friction velocity
    dy_wall = vol[0] / S0 / 2
    tau_wall = nu * u_tan[0] / dy_wall
    u_tau = np.sqrt(np.abs(tau_wall))
    
    # y+
    y_plus = wall_dist * u_tau / nu
    
    return y_plus, u_tau


def analyze_single_station(data, i_key, nu, verbose=True):
    """Analyze a single streamwise station."""
    i_idx = int(data[i_key])
    
    # Extract data at this station
    Q = data['Q'][i_idx, :]
    wall_dist = data['wall_dist'][i_idx, :]
    vol = data['vol'][i_idx, :]
    Sj_x = data['Sj_x'][i_idx, :]
    Sj_y = data['Sj_y'][i_idx, :]
    
    # Compute y+
    y_plus, u_tau = compute_y_plus(wall_dist, vol, Sj_x, Sj_y, Q, nu)
    
    # Extract SA variables
    nuHat = Q[:, 3]
    
    # Compute vorticity (simplified - just du/dy)
    u = Q[:, 1]
    v = Q[:, 2]
    S0 = np.sqrt(Sj_x[0]**2 + Sj_y[0]**2)
    tx, ty = -Sj_y[0]/S0, Sj_x[0]/S0
    u_tan = u * tx + v * ty
    if np.mean(u_tan) < 0:
        u_tan = -u_tan
    omega = np.abs(np.gradient(u_tan, wall_dist))
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"STATION: {i_key} (i = {i_idx})")
        print(f"{'='*70}")
        print(f"u_tau = {u_tau:.4f}, Cf = {2*u_tau**2:.5f}")
    
    # Run comprehensive diagnostic
    result = diagnose_sa_physics(
        y_plus=y_plus,
        nuHat=nuHat,
        omega=omega,
        d=wall_dist,
        nu_laminar=nu,
        verbose=verbose
    )
    
    return result


def main():
    parser = argparse.ArgumentParser(description='SA Physics Diagnostic')
    parser.add_argument('--data', type=str, default='output/bl_full_diagnostic.npz',
                        help='Path to diagnostic data file')
    parser.add_argument('--station', type=str, default='all',
                        help='Station to analyze (i_05, i_10, i_25, i_50, or all)')
    args = parser.parse_args()
    
    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        print("Run the solver first to generate diagnostic data.")
        sys.exit(1)
    
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    
    nu = float(data['nu'])
    print(f"Reynolds number: {1/nu:.0f}")
    
    # Find available stations
    stations = [k for k in data.files if k.startswith('i_')]
    print(f"Available stations: {stations}")
    
    # Print model constants
    print(f"\n{'='*70}")
    print("SA MODEL CONSTANTS")
    print(f"{'='*70}")
    print(f"  κ (von Karman):   {KAPPA}")
    print(f"  cb1:              {CB1}")
    print(f"  cw1:              {CW1:.4f} (derived: cb1/κ² + (1+cb2)/σ)")
    print(f"  cv1:              {CV1}")
    print(f"  Expected χ slope: {KAPPA} (dχ/dy⁺ in log layer)")
    print(f"{'='*70}")
    
    # Analyze stations
    results = {}
    if args.station == 'all':
        for station in stations:
            results[station] = analyze_single_station(data, station, nu)
    else:
        if args.station not in stations:
            print(f"Error: Station '{args.station}' not found. Available: {stations}")
            sys.exit(1)
        results[args.station] = analyze_single_station(data, args.station, nu)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    all_ok = True
    for station, result in results.items():
        log_ok = result['log_layer']['equilibrium_satisfied']
        wall_ok = result['wall_scaling']['wall_scaling_ok']
        
        if log_ok is None:
            log_status = "N/A"
        elif log_ok:
            log_status = "✓"
        else:
            log_status = "✗"
            all_ok = False
        
        if wall_ok is None:
            wall_status = "N/A"
        elif wall_ok:
            wall_status = "✓"
        else:
            wall_status = "✗"
            all_ok = False
        
        chi_ratio = result['log_layer'].get('chi_slope_ratio', 0)
        print(f"  {station}: Log layer [{log_status}], Wall [{wall_status}], "
              f"χ slope = {chi_ratio*100:.0f}% of expected")
    
    if all_ok:
        print("\nAll physics checks PASSED.")
    else:
        print("\nSome physics checks FAILED. See detailed output above.")
        print("\nKey diagnostic questions:")
        print("  1. Is χ = κ·y⁺ in the log layer? If not, ν̃ is not correct.")
        print("  2. Is P ≈ D in the log layer? If not, equilibrium is broken.")
        print("  3. Is r ≈ 1? If r >> 1, destruction dominates; if r << 1, production dominates.")


if __name__ == '__main__':
    main()
