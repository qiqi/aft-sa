
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
    # X, Y are nodes (NI+1, NJ+1). Q is cells (NI, NJ) (possibly with ghosts)
    xc = 0.25 * (X[:-1, :-1] + X[1:, :-1] + X[:-1, 1:] + X[1:, 1:])
    yc = 0.25 * (Y[:-1, :-1] + Y[1:, :-1] + Y[:-1, 1:] + Y[1:, 1:])
    
    # Strip ghosts from Q if needed
    # Check shape match
    ni_grid, nj_grid = xc.shape
    if Q.shape[0] != ni_grid:
        # Assuming Q has NGHOST=2 padding
        nghost = (Q.shape[0] - ni_grid) // 2
        print(f"Stripping {nghost} ghost layers from Q")
        Q = Q[nghost:-nghost, nghost:-nghost, :]

    # Constants
    nu_laminar = 1.0 / 1.0e6
    sigma = 2.0/3.0
    
    # Extract profiles
    x_locs = [1.4, 1.5, 1.6]
    tol = 0.05
    
    profile_data = {}
    
    for x_target in x_locs:
        # Mask for points near x_target
        mask = (np.abs(xc - x_target) < tol) & (np.abs(yc) < 0.2)
        
        y_pts = yc[mask]
        nu_pts = Q[mask, 3]
        
        # Sort
        sort_idx = np.argsort(y_pts)
        y_pts = y_pts[sort_idx]
        nu_pts = nu_pts[sort_idx]
        
        # Gradients
        dy = np.gradient(y_pts)
        dnu = np.gradient(nu_pts)
        dnudy = dnu / (dy + 1e-12)
        
        # Save to dict
        key_base = f"x_{x_target:.1f}"
        profile_data[f"{key_base}_y"] = y_pts
        profile_data[f"{key_base}_nuHat"] = nu_pts
        profile_data[f"{key_base}_dnudy"] = dnudy
        
        print(f"Extracted profile at x={x_target}: {len(y_pts)} points")

    # Save to NPZ
    save_path = os.path.join(output_dir, "wake_profiles.npz")
    np.savez(save_path, **profile_data)
    print(f"Saved wake profiles to {save_path}")

if __name__ == "__main__":
    main()
