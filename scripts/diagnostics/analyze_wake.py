#!/usr/bin/env python3
"""
Wake Analysis Script.

Analyzes the turbulent wake structure from simulation output files.
computes gradients, effective viscosity flux, and identifies wake edges.
"""

import numpy as np
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger
from src.utils.logging import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze wake profiles from CFD output.")
    parser.add_argument("output_dir", nargs="?", default="output/naca0012_newton",
                        help="Directory containing output .npy files (default: output/naca0012_newton)")
    parser.add_argument("--tol", type=float, default=0.05,
                        help="Tolerance for finding points near x-location")
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging()
    
    output_dir = args.output_dir
    logger.info(f"Analyzing data in: {output_dir}")
    
    # Load data
    try:
        Q = np.load(os.path.join(output_dir, "final_q.npy"))
        X = np.load(os.path.join(output_dir, "final_x.npy"))
        Y = np.load(os.path.join(output_dir, "final_y.npy"))
    except FileNotFoundError:
        logger.error("Data files (final_q.npy, etc.) not found. Run simulation first.")
        return 1

    # Compute cell centers
    # X, Y are nodes (NI+1, NJ+1)
    # Q is cells (NI, NJ)
    xc = 0.25 * (X[:-1, :-1] + X[1:, :-1] + X[:-1, 1:] + X[1:, 1:])
    yc = 0.25 * (Y[:-1, :-1] + Y[1:, :-1] + Y[:-1, 1:] + Y[1:, 1:])
    
    # Q has shape (292, 130, 4) -> 2 ghost layers
    # Grid size is 288x126.
    # Strip ghosts
    nghost = 2
    Q = Q[nghost:-nghost, nghost:-nghost, :]
    
    # Use cell centers for masking
    if Q.shape[0] != xc.shape[0] or Q.shape[1] != xc.shape[1]:
        logger.error(f"Shape mismatch: Q {Q.shape}, Xc {xc.shape}")
        return 1
        
    X = xc
    Y = yc
    
    # Constants
    nu_laminar = 1.0 / 1.0e6  # Re=1e6 (Approximation, should load from config but this is sufficient for quick analysis)
    sigma = 2.0/3.0
    
    # Extract profiles at specific x locations
    x_locs = [1.4, 1.5, 1.6]
    tol = args.tol
    
    logger.info(f"{'X_target':<10} {'Y':<10} {'nuHat':<12} {'dNu/dy':<12} {'Diff_Term':<12} {'Region':<10}")
    logger.info("-" * 70)
    
    for x_target in x_locs:
        # Find points near x_target
        # Using a mask. In C-grid, wake is localized.
        mask = (np.abs(X - x_target) < tol) & (np.abs(Y) < 0.2) # Focus on near wake
        
        # Extract data
        y_pts = Y[mask]
        nu_pts = Q[mask, 3] # nuHat
        
        if len(y_pts) == 0:
            logger.warning(f"No points found at x={x_target}")
            continue

        # Sort by Y
        sort_idx = np.argsort(y_pts)
        y_pts = y_pts[sort_idx]
        nu_pts = nu_pts[sort_idx]
        
        # Compute finite difference derivative dy
        # Simple central diff
        dy = np.gradient(y_pts)
        dnu = np.gradient(nu_pts)
        dnudy = dnu / (dy + 1e-12)
        
        # Estimate diffusion flux term ~ (nu + nuHat)/sigma * dnu/dy
        nu_eff = nu_laminar + np.maximum(0, nu_pts)
        flux = (nu_eff / sigma) * dnudy
        
        logger.info(f"\nProfile at x={x_target}")
        
        max_nu = np.max(nu_pts)
        
        for k in range(0, len(y_pts), 2): # Skip every other point
            y = y_pts[k]
            nu = nu_pts[k]
            grad = dnudy[k]
            diff_flux = flux[k]
            chi = nu / nu_laminar
            
            # Label region
            if nu > 0.1 * max_nu:
                region = "Core"
            elif nu > 1e-6:
                region = "Edge"
            else:
                region = "Free"
            
            # Print mainly the Edges
            if region == "Edge" or (region == "Core" and k % 5 == 0) or (region=="Free" and -0.05 < y < 0.05):
                 logger.info(f"{x_target:<10.2f} {y:<10.5f} {nu:<12.2e} {chi:<10.2f} {grad:<12.2e} {diff_flux:<12.2e} {region}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
