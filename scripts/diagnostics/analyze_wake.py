
import numpy as np
import os
import sys

def main():
    output_dir = "output/naca0012_newton"
    
    # Load data
    try:
        Q = np.load(os.path.join(output_dir, "final_q.npy"))
        X = np.load(os.path.join(output_dir, "final_x.npy"))
        Y = np.load(os.path.join(output_dir, "final_y.npy"))
    except FileNotFoundError:
        print("Data files not found. Run simulation first.")
        return

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
        print(f"Shape mismatch: Q {Q.shape}, Xc {xc.shape}")
        return
        
    X = xc
    Y = yc
    
    # Constants
    nu_laminar = 1.0 / 1.0e6  # Re=1e6
    sigma = 2.0/3.0
    
    # Extract profiles at specific x locations
    x_locs = [1.4, 1.5, 1.6]
    tol = 0.05  # Increased tolerance to catch points on coarse grids



    
    print(f"{'X_target':<10} {'Y':<10} {'nuHat':<12} {'dNu/dy':<12} {'Diff_Term':<12} {'Region':<10}")
    print("-" * 70)
    
    for x_target in x_locs:
        # Find points near x_target
        # Using a mask. In C-grid, wake is localized.
        mask = (np.abs(X - x_target) < tol) & (np.abs(Y) < 0.2) # Focus on near wake
        
        # Extract data
        y_pts = Y[mask]
        nu_pts = Q[mask, 3] # nuHat
        
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
        # This is strictly the flux, not the divergence (flux gradient).
        # But if flux is zero, divergence is determined by neighbors.
        # User asked about "Viscous terms".
        
        nu_eff = nu_laminar + np.maximum(0, nu_pts)
        flux = (nu_eff / sigma) * dnudy
        
        # Identify sharp drop
        # Look for regions where nuHat drops to near-zero quickly
        
        # Print a subset of points (decimated) to avoid spam
        # Focus on the "Edge" of the wake
        # Wake center is likely where nuHat is max.
        
        print(f"\nProfile at x={x_target}")
        
        max_nu = np.max(nu_pts)
        # Find indices where nuHat transitions
        
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
                 print(f"{x_target:<10.2f} {y:<10.5f} {nu:<12.2e} {chi:<10.2f} {grad:<12.2e} {diff_flux:<12.2e} {region}")


if __name__ == "__main__":
    main()
