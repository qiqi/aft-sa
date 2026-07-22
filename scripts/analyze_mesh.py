import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    debug_dir = Path("debug_mesh_output")
    if not debug_dir.exists():
        print(f"Directory {debug_dir} not found. Run debug_mesh_gen.py first.")
        return

    # 1. Compare cdfj[:,-1]
    # ... (existing code for comparison) ...
    
    # 2. Check Explicit March (Offset) for Noise
    try:
        xoff = np.loadtxt(debug_dir / "debug_offset_x.txt")
        yoff = np.loadtxt(debug_dir / "debug_offset_y.txt")
        
        plt.figure(figsize=(10, 10))
        plt.plot(xoff[:,1], yoff[:,1], '.-', linewidth=0.5, markersize=2)
        plt.title("Explicit March Profile (j=max-1)")
        plt.axis('equal')
        plt.grid(True)
        plt.savefig(debug_dir / "offset_profile.png")
        print("Saved offset_profile.png")
        
        # Check smoothness (diff)
        dx = np.diff(xoff[:,1])
        plt.figure()
        plt.plot(dx)
        plt.title("First Difference of xoff (dX)")
        plt.savefig(debug_dir / "offset_dx.png")
        print("Saved offset_dx.png")
        
    except Exception as e:
        print(f"Error analyzing offset: {e}")

if __name__ == "__main__":
    main()
