
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.grid.mesher import Construct2DWrapper, GridOptions

def dump_mesh(wkfn=0.0):
    print(f"Generating grid with wake_fan_factor={wkfn}...")
    
    # Defaults from load_or_generate_grid for naca0012.dat
    # Assuming standard behavior unless user overrides
    options = GridOptions(
        n_surface=100,
        n_normal=30,
        n_wake=20, # Default used in other scripts seems to be 20 or 30
        wake_fan_factor=wkfn,
        y_plus=1.0,
        reynolds=1e6,
        topology='CGRD'
    )
    
    wrapper = Construct2DWrapper('./bin/construct2d')
    # Using the standard data file
    airfoil_file = 'data/naca0012.dat'
    
    if not os.path.exists(airfoil_file):
        # Maybe absolute path needed
        airfoil_file = '/home/qiqi/aft-sa/data/naca0012.dat'
        
    try:
        X, Y = wrapper.generate(airfoil_file, options=options, verbose=True)
        
        # Save to numpy array [NI, NJ, 2]
        # X is [NI+1, NJ+1] (nodes)
        grid_data = np.stack([X, Y], axis=-1)
        np.save('mesh_dump.npy', grid_data)
        print(f"Mesh dumped to mesh_dump.npy. Shape: {grid_data.shape}")
        
        # Check wake spacing at farfield (i=0, i=IMAX)
        # i=0 corresponds to the wake boundary
        # Check normal spacing dy at i=0
        dy = np.sqrt((X[0,1]-X[0,0])**2 + (Y[0,1]-Y[0,0])**2)
        print(f"DEBUG: dy at farfield wake (j=0->1): {dy:.6e}")
        
    except Exception as e:
        print(f"Error generating grid: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wkfn", type=float, default=0.0, help="Wake fan factor")
    args = parser.parse_args()
    
    dump_mesh(args.wkfn)
