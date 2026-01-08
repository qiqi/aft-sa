import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.grid.mesher import Construct2DWrapper, GridOptions

def verify_wake_fanning():
    wrapper = Construct2DWrapper('/home/qiqi/aft-sa/bin/construct2d')
    airfoil_file = '/home/qiqi/aft-sa/data/naca0012.dat'
    
    # Baseline: No fanning (wkfn=0.0)
    print("Generating baseline grid (wkfn=0.0)...")
    options_base = GridOptions(
        n_surface=100,
        n_normal=30,
        n_wake=20,
        wake_fan_factor=0.0,
        topology='CGRD',
        solver='HYPR'
    )
    X_base, Y_base = wrapper.generate(airfoil_file, options=options_base, verbose=True)
    
    # Fanned: wkfn=1.0 (Blend to uniform)
    print("Generating fanned grid (wkfn=1.0)...")
    options_fan = GridOptions(
        n_surface=100,
        n_normal=30,
        n_wake=20,
        wake_fan_factor=0.01,
        topology='CGRD',
        solver='HYPR'
    )
    X_fan, Y_fan = wrapper.generate(airfoil_file, options=options_fan, verbose=True)
    
    # Verify that fanned grid has LARGER spacing near wake cut at farfield
    # Wake cut is at j=0 (technically ghost cells, but first physical points are near j=0)
    # The wake cut corresponds to y=0.
    # We want to check the normal spacing (dy) at the wake cut.
    # The grid Y coordinates at i=0 (farfield inflow) or i=imax-1 (farfield outflow)
    # Let's check i=0 (one end of C-grid) which is in the farfield wake.
    # Corresponding index in array is 0.
    
    # Calculate dy at i=0 for baseline and fanned
    dy_base = np.abs(Y_base[0, 1] - Y_base[0, 0])
    dy_fan = np.abs(Y_fan[0, 1] - Y_fan[0, 0])
    
    print(f"Baseline dy at farfield wake: {dy_base:.6e}")
    print(f"Fanned dy at farfield wake:   {dy_fan:.6e}")
    
    if dy_fan > dy_base:
        print("PASS: Fanned grid has larger spacing near wake cut.")
    else:
        print("FAIL: Fanned grid spacing is not larger.")

    # Check uniformity?
    # Compare dy near wall (j=0) vs dy near outer boundary (j=-1)
    dy_fan_outer = np.abs(Y_fan[0, -1] - Y_fan[0, -2])
    ratio_fan = dy_fan_outer / dy_fan
    ratio_base = np.abs(Y_base[0, -1] - Y_base[0, -2]) / dy_base
    
    print(f"Baseline outer/inner ratio: {ratio_base:.2f}")
    print(f"Fanned outer/inner ratio:   {ratio_fan:.2f}")
    
    if ratio_fan < ratio_base:
         print("PASS: Fanned grid is more uniform (lower stretching ratio).")
    else:
         print("FAIL: Fanned grid is not more uniform.")
    
    print("\nVerification Complete")

if __name__ == "__main__":
    verify_wake_fanning()
