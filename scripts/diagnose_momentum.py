#!/usr/bin/env python3
"""
Momentum Diagnostic Script - STEP 1 in SA Debugging Hierarchy

This script checks whether the velocity profile u⁺ is logarithmic.
If u⁺ is NOT logarithmic, the error is in the FLOW SOLVER, not the SA model.

The Law of the Wall:
    Viscous sublayer (y⁺ < 5):   u⁺ = y⁺
    Log layer (y⁺ > 30):         u⁺ = (1/κ) ln(y⁺) + B
    
    where κ = 0.41, B ≈ 5.0

Usage:
    python scripts/diagnose_momentum.py [--data output/bl_full_diagnostic.npz]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Physical constants
KAPPA = 0.41
B_CONSTANT = 5.0  # Additive constant in log law


def spalding_profile(y_plus):
    """
    Spalding's unified law of the wall.
    
    Gives u⁺ as implicit function of y⁺:
        y⁺ = u⁺ + exp(-κB)[exp(κu⁺) - 1 - κu⁺ - (κu⁺)²/2 - (κu⁺)³/6]
    
    We solve this numerically for u⁺.
    """
    from scipy.optimize import brentq
    
    def residual(u_plus, y_p):
        kappa_u = KAPPA * u_plus
        exp_term = np.exp(-KAPPA * B_CONSTANT) * (
            np.exp(kappa_u) - 1 - kappa_u - kappa_u**2/2 - kappa_u**3/6
        )
        return u_plus + exp_term - y_p
    
    u_plus = np.zeros_like(y_plus)
    for i, yp in enumerate(y_plus):
        if yp < 0.1:
            u_plus[i] = yp  # Linear in viscous sublayer
        else:
            try:
                u_plus[i] = brentq(residual, 0, yp + 50, args=(yp,))
            except:
                u_plus[i] = (1/KAPPA) * np.log(yp) + B_CONSTANT
    
    return u_plus


def log_law(y_plus):
    """Simple log law: u⁺ = (1/κ) ln(y⁺) + B"""
    return (1/KAPPA) * np.log(y_plus) + B_CONSTANT


def compute_velocity_profile(data, i_key, nu):
    """Extract velocity profile at a station."""
    i_idx = int(data[i_key])
    
    # Extract data
    Q = data['Q'][i_idx, :]
    wall_dist = data['wall_dist'][i_idx, :]
    vol = data['vol'][i_idx, :]
    Sj_x = data['Sj_x'][i_idx, :]
    Sj_y = data['Sj_y'][i_idx, :]
    
    # Wall-normal direction (tangent to wall)
    S0 = np.sqrt(Sj_x[0]**2 + Sj_y[0]**2)
    tx = -Sj_y[0] / S0
    ty = Sj_x[0] / S0
    
    # Velocities
    u = Q[:, 1]
    v = Q[:, 2]
    
    # Tangential velocity (parallel to wall)
    u_tan = u * tx + v * ty
    if np.mean(u_tan) < 0:
        u_tan = -u_tan
    
    # Wall-normal velocity
    u_norm = u * (-ty) + v * tx
    
    # Wall shear stress and friction velocity
    dy_wall = vol[0] / S0 / 2  # Distance from wall to first cell center
    tau_wall = nu * u_tan[0] / dy_wall
    u_tau = np.sqrt(np.abs(tau_wall))
    
    # Wall units
    y_plus = wall_dist * u_tau / nu
    u_plus = u_tan / u_tau
    
    # Also get nuHat for chi
    nuHat = Q[:, 3]
    chi = nuHat / nu
    
    return {
        'y_plus': y_plus,
        'u_plus': u_plus,
        'u_tan': u_tan,
        'u_norm': u_norm,
        'u_tau': u_tau,
        'wall_dist': wall_dist,
        'chi': chi,
        'nuHat': nuHat,
        'Cf': 2 * u_tau**2,
    }


def check_log_law_velocity(y_plus, u_plus, verbose=True):
    """
    Check if velocity profile follows the log law.
    
    Returns diagnostics about velocity profile quality.
    """
    # Focus on log layer (30 < y+ < 100)
    log_mask = (y_plus > 30) & (y_plus < 100)
    
    if not np.any(log_mask):
        return {
            'log_law_ok': None,
            'message': 'No points in log layer (30 < y+ < 100)'
        }
    
    y_plus_log = y_plus[log_mask]
    u_plus_log = u_plus[log_mask]
    
    # Expected u+ from log law
    u_plus_expected = log_law(y_plus_log)
    
    # Compute error
    error = u_plus_log - u_plus_expected
    mean_error = np.mean(error)
    rms_error = np.sqrt(np.mean(error**2))
    relative_error = rms_error / np.mean(u_plus_expected)
    
    # Check if slope matches (should be 1/κ ≈ 2.44)
    if len(y_plus_log) > 2:
        # Fit: u+ = a * ln(y+) + b
        log_yp = np.log(y_plus_log)
        slope, intercept = np.polyfit(log_yp, u_plus_log, 1)
        expected_slope = 1/KAPPA
        slope_ratio = slope / expected_slope
    else:
        slope = None
        slope_ratio = None
    
    # Criteria for "good" log law
    log_law_ok = (relative_error < 0.2) and (slope_ratio is not None) and (0.7 < slope_ratio < 1.3)
    
    message = []
    if slope is not None:
        message.append(f"Log-layer slope du⁺/d(ln y⁺) = {slope:.3f} (expected: {1/KAPPA:.3f})")
        message.append(f"Slope ratio: {slope_ratio:.2%} of expected")
        if slope_ratio < 0.7:
            message.append("  → SLOPE TOO LOW: Velocity not growing fast enough with y")
        elif slope_ratio > 1.3:
            message.append("  → SLOPE TOO HIGH: Velocity growing too fast with y")
    
    message.append(f"Mean u⁺ error in log layer: {mean_error:+.2f}")
    message.append(f"RMS u⁺ error: {rms_error:.2f} ({relative_error:.1%} relative)")
    
    if not log_law_ok:
        message.append("")
        message.append("*** VELOCITY PROFILE DOES NOT FOLLOW LOG LAW ***")
        message.append("The error is in the FLOW SOLVER, not the SA model!")
        message.append("Check: viscous fluxes, pressure solver, boundary conditions")
    else:
        message.append("")
        message.append("Velocity profile follows log law: OK")
    
    return {
        'log_law_ok': log_law_ok,
        'slope': slope,
        'expected_slope': 1/KAPPA,
        'slope_ratio': slope_ratio,
        'mean_error': mean_error,
        'rms_error': rms_error,
        'relative_error': relative_error,
        'message': '\n'.join(message)
    }


def check_viscous_sublayer(y_plus, u_plus, verbose=True):
    """Check if u⁺ = y⁺ in viscous sublayer."""
    # Focus on viscous sublayer (y+ < 5)
    visc_mask = y_plus < 5
    
    if not np.any(visc_mask):
        return {
            'sublayer_ok': None,
            'message': 'No points in viscous sublayer (y+ < 5)'
        }
    
    y_plus_visc = y_plus[visc_mask]
    u_plus_visc = u_plus[visc_mask]
    
    # In viscous sublayer: u+ = y+
    error = u_plus_visc - y_plus_visc
    mean_error = np.mean(error)
    rms_error = np.sqrt(np.mean(error**2))
    
    # Check slope (should be 1.0)
    if len(y_plus_visc) > 2:
        slope, intercept = np.polyfit(y_plus_visc, u_plus_visc, 1)
    else:
        slope = u_plus_visc[-1] / y_plus_visc[-1] if len(y_plus_visc) > 0 else None
        intercept = 0
    
    sublayer_ok = (slope is not None) and (0.7 < slope < 1.3)
    
    message = []
    if slope is not None:
        message.append(f"Viscous sublayer slope du⁺/dy⁺ = {slope:.3f} (expected: 1.0)")
        if slope < 0.7:
            message.append("  → SLOPE TOO LOW: Wall shear stress may be wrong")
        elif slope > 1.3:
            message.append("  → SLOPE TOO HIGH: Check viscous flux calculation")
    
    message.append(f"Mean error in sublayer: {mean_error:+.3f}")
    
    return {
        'sublayer_ok': sublayer_ok,
        'slope': slope,
        'mean_error': mean_error,
        'message': '\n'.join(message)
    }


def analyze_station(data, i_key, nu, verbose=True):
    """Full momentum analysis at a station."""
    profile = compute_velocity_profile(data, i_key, nu)
    
    y_plus = profile['y_plus']
    u_plus = profile['u_plus']
    u_tau = profile['u_tau']
    Cf = profile['Cf']
    
    log_check = check_log_law_velocity(y_plus, u_plus)
    visc_check = check_viscous_sublayer(y_plus, u_plus)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"MOMENTUM DIAGNOSTIC - STATION {i_key}")
        print(f"{'='*70}")
        print(f"Friction velocity u_τ = {u_tau:.4f}")
        print(f"Skin friction Cf = {Cf:.5f}")
        print(f"Expected Cf for turbulent BL at this Re: ~0.003-0.006")
        
        if Cf > 0.01:
            print(f"  ⚠️  Cf is HIGH - suggests incorrect wall shear stress")
        
        print(f"\n--- VISCOUS SUBLAYER (y⁺ < 5) ---")
        print(visc_check['message'])
        
        print(f"\n--- LOG LAYER (30 < y⁺ < 100) ---")
        print(log_check['message'])
        
        # Summary
        print(f"\n--- STEP 1 VERDICT ---")
        if log_check['log_law_ok']:
            print("✓ Velocity profile IS logarithmic. Proceed to Step 2 (geometry).")
        else:
            print("✗ Velocity profile is NOT logarithmic!")
            print("  → The error is in the FLOW SOLVER.")
            print("  → DO NOT debug SA model until this is fixed.")
            print("")
            print("  Possible flow solver issues:")
            print("    1. Viscous flux computation incorrect")
            print("    2. Pressure solver not converged")
            print("    3. Boundary conditions wrong (wall, farfield)")
            print("    4. Artificial compressibility parameter β")
            print("    5. Numerical dissipation too high/low")
    
    return {
        'profile': profile,
        'log_check': log_check,
        'visc_check': visc_check,
    }


def plot_velocity_profiles(results, stations, output_path=None):
    """Plot velocity profiles against law of the wall."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Create reference profiles
    y_plus_ref = np.logspace(-1, 3, 500)
    u_plus_spalding = spalding_profile(y_plus_ref)
    u_plus_linear = y_plus_ref  # u+ = y+ for viscous sublayer
    u_plus_log = log_law(y_plus_ref)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(stations)))
    
    # Left plot: Full profile (log-log)
    ax = axes[0]
    ax.semilogx(y_plus_ref, u_plus_spalding, 'k-', lw=2, label='Spalding (exact)')
    ax.semilogx(y_plus_ref[y_plus_ref < 10], y_plus_ref[y_plus_ref < 10], 
                'k--', lw=1, label='u⁺ = y⁺')
    ax.semilogx(y_plus_ref[y_plus_ref > 10], u_plus_log[y_plus_ref > 10], 
                'k:', lw=1, label='Log law')
    
    for i, station in enumerate(stations):
        profile = results[station]['profile']
        ax.semilogx(profile['y_plus'], profile['u_plus'], 'o-', 
                    color=colors[i], markersize=3, label=f'{station}')
    
    ax.set_xlabel('y⁺')
    ax.set_ylabel('u⁺')
    ax.set_title('Velocity Profile (Law of the Wall)')
    ax.legend(loc='upper left')
    ax.set_xlim(0.1, 1000)
    ax.set_ylim(0, 30)
    ax.grid(True, alpha=0.3)
    
    # Right plot: Log layer detail (linear-log)
    ax = axes[1]
    ax.semilogx(y_plus_ref, u_plus_log, 'k-', lw=2, label='Log law: u⁺=(1/κ)ln(y⁺)+B')
    
    for i, station in enumerate(stations):
        profile = results[station]['profile']
        mask = profile['y_plus'] > 10
        ax.semilogx(profile['y_plus'][mask], profile['u_plus'][mask], 'o-', 
                    color=colors[i], markersize=4, label=f'{station}')
    
    ax.set_xlabel('y⁺')
    ax.set_ylabel('u⁺')
    ax.set_title('Log Layer Detail')
    ax.legend(loc='upper left')
    ax.set_xlim(10, 1000)
    ax.set_ylim(5, 30)
    ax.grid(True, alpha=0.3)
    ax.axvline(30, color='gray', linestyle='--', alpha=0.5, label='y⁺=30')
    ax.axvline(100, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Momentum Profile Diagnostic')
    parser.add_argument('--data', type=str, default='output/bl_full_diagnostic.npz',
                        help='Path to diagnostic data file')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--station', type=str, default='all',
                        help='Station to analyze (i_05, i_10, i_25, i_50, or all)')
    args = parser.parse_args()
    
    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)
    
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    
    nu = float(data['nu'])
    Re = 1/nu
    print(f"Reynolds number: {Re:.0f}")
    
    # Find available stations
    stations = sorted([k for k in data.files if k.startswith('i_')])
    print(f"Available stations: {stations}")
    
    print("\n" + "="*70)
    print("STEP 1: CHECK MOMENTUM - Is u⁺ logarithmic?")
    print("="*70)
    print("If u⁺ is NOT logarithmic, the error is in the FLOW SOLVER.")
    print("The SA model is INNOCENT until momentum is correct.")
    
    # Analyze stations
    results = {}
    if args.station == 'all':
        for station in stations:
            results[station] = analyze_station(data, station, nu)
    else:
        if args.station not in stations:
            print(f"Error: Station '{args.station}' not found.")
            sys.exit(1)
        results[args.station] = analyze_station(data, args.station, nu)
        stations = [args.station]
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: MOMENTUM CHECK")
    print("="*70)
    
    all_ok = True
    for station in stations:
        log_ok = results[station]['log_check']['log_law_ok']
        slope_ratio = results[station]['log_check'].get('slope_ratio', 0)
        
        status = "✓" if log_ok else "✗"
        if log_ok is None:
            status = "?"
            
        print(f"  {station}: [{status}] Log layer slope = {slope_ratio*100:.0f}% of expected")
        
        if not log_ok:
            all_ok = False
    
    print("")
    if all_ok:
        print("RESULT: Momentum is OK. Proceed to Step 2 (check geometry).")
    else:
        print("RESULT: Momentum is WRONG. Fix flow solver before debugging SA model!")
    
    # Plot if requested
    if args.plot:
        plot_velocity_profiles(results, stations, 'output/momentum_profile.png')


if __name__ == '__main__':
    main()
