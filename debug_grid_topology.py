
import sys
import numpy as np
import logging
from pathlib import Path

# Setup simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock RANS Solver minimal init to load grid
from src.grid.mesher import Construct2DWrapper, GridOptions

def load_and_analyze_grid():
    construct2d_path = Path("bin/construct2d")
    if not construct2d_path.exists():
        construct2d_path = Path("/usr/local/bin/construct2d")
    
    wrapper = Construct2DWrapper(str(construct2d_path))
    grid_opts = GridOptions(
        n_surface=193, # from log
        n_normal=127,  # from log (computed)
        y_plus=1.0,
        reynolds=1e6
    )
    
    # We need to mimic the logic in rans_solver.py that determines n_wake?
    # rans_solver.py:
    #   GridOptions(n_surface=250, n_normal=100, ...) -- WAIT, Log said 193/127?
    #   Config has defaults?
    #   User config line 10: n_wake: 48.
    
    # The Log said:
    #   Surface points: 193
    #   Wake points: 48
    #   Normal points: 127
    
    # Construct2D wrapper .generate() returns X, Y.
    X, Y = wrapper.generate("data/naca0012.dat", grid_opts)
    
    NI = X.shape[0] - 1
    NJ = X.shape[1] - 1
    
    print(f"Grid Shape: {X.shape}")
    print(f"NI={NI}, NJ={NJ}")
    
    n_wake = 48
    print(f"Assumed n_wake={n_wake}")
    
    # Analyze Lower Wake (i=0..n_wake)
    # Note: Indices in Python slices 0:n_wake+1
    x_lower = X[0 : n_wake+1, 0]
    y_lower = Y[0 : n_wake+1, 0]
    
    # Analyze Upper Wake (i=NI-n_wake..NI+1)
    # Reverse it to match Lower (Assuming Lower runs Downstream->TE)
    x_upper = X[NI-n_wake : NI+1, 0]
    y_upper = Y[NI-n_wake : NI+1, 0]
    x_upper_rev = x_upper[::-1]
    y_upper_rev = y_upper[::-1]
    
    print("\nLOWER Wake (0..5):")
    for i in range(5):
        print(f"  {i}: ({x_lower[i]:.4f}, {y_lower[i]:.4f})")
    print("...")
    print(f"  {n_wake}: ({x_lower[n_wake]:.4f}, {y_lower[n_wake]:.4f}) (TE?)")
    
    print("\nUPPER Wake (Reversed) (0..5):")
    for i in range(5):
        print(f"  {i}: ({x_upper_rev[i]:.4f}, {y_upper_rev[i]:.4f})")
    print("...")
    print(f"  {n_wake}: ({x_upper_rev[n_wake]:.4f}, {y_upper_rev[n_wake]:.4f}) (TE?)")
    
    # Compute Delta
    if len(x_lower) != len(x_upper_rev):
        print(f"FATAL: Length mismatch. Lower={len(x_lower)}, Upper={len(x_upper_rev)}")
    else:
        delta_x = np.abs(x_lower - x_upper_rev)
        delta_y = np.abs(y_lower - y_upper_rev)
        print(f"\nMax Delta X: {np.max(delta_x):.6f}")
        print(f"Max Delta Y: {np.max(delta_y):.6f}")
        
        if np.max(delta_x) > 1e-4:
            print("MISMATCH DETECTED!")
            # Determine correct shift?
            # Try shifting upper by 1?
            
if __name__ == "__main__":
    load_and_analyze_grid()
