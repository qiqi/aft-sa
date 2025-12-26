import os
import numpy as np
import matplotlib.pyplot as plt
from src.grid.construct2d_wrapper import Construct2DWrapper

def generate_naca4(code, filename, n_points=160):
    """Generates standard NACA 4-digit coordinates."""
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0
    
    # Cosine spacing
    beta = np.linspace(0, np.pi, n_points)
    x = 0.5 * (1 - np.cos(beta))
    
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    
    for i in range(len(x)):
        if p == 0:
            yc[i] = 0.0
            dyc_dx[i] = 0.0
        elif x[i] <= p:
            yc[i] = m / p**2 * (2 * p * x[i] - x[i]**2)
            dyc_dx[i] = 2 * m / p**2 * (p - x[i])
        else:
            yc[i] = m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x[i] - x[i]**2)
            dyc_dx[i] = 2 * m / (1 - p)**2 * (p - x[i])
            
    theta = np.arctan(dyc_dx)
    
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    
    # Write Selig format (Upper TE -> LE -> Lower TE)
    # Construct2D needs "top-to-bottom" order typically
    # WARNING: Construct2D's Fortran reader might choke on text headers on some systems?
    # We write a dummy numerical header just in case, or keep it standard.
    # User suggested removing the name line. But the code skips the first line.
    # So we replace the name with a safe dummy line.
    with open(filename, 'w') as f:
        f.write(f"0.0 0.0\n") # Dummy header that looks like numbers
        for i in range(len(xu)-1, -1, -1):
            f.write(f"{xu[i]:.6f} {yu[i]:.6f}\n")
        for i in range(1, len(xl)):
            f.write(f"{xl[i]:.6f} {yl[i]:.6f}\n")

def plot_grid(X, Y, name):
    plt.figure(figsize=(12, 10))
    stride = 2
    plt.plot(X[::stride, ::stride], Y[::stride, ::stride], 'k-', lw=0.3, alpha=0.5)
    plt.plot(X[::stride, ::stride].T, Y[::stride, ::stride].T, 'k-', lw=0.3, alpha=0.5)
    plt.plot(X[:, 0], Y[:, 0], 'r-', lw=1.5, label='Surface')
    plt.axis('equal')
    plt.title(f"Construct2D Grid: {name}")
    plt.savefig(f"grid_c2d_{name}.pdf")
    print(f"Saved grid_c2d_{name}.pdf")
    plt.close()

def run_test():
    if not os.path.exists('data'):
        os.makedirs('data')
        
    wrapper = Construct2DWrapper("external/construct2d/construct2d")
    
    foils = ['0012', '2412']
    
    for code in foils:
        name = f"naca{code}"
        dat_file = f"data/{name}.dat"
        grid_file = f"data/{name}.p3d"
        
        generate_naca4(code, dat_file)
        
        # Run Construct2D
        # Note: Construct2D needs .dat in current dir usually? handled in wrapper
        wrapper.run(dat_file, grid_file, n_chord=200, n_normal=80, y_plus=1.0)
        
        # Read and Plot
        X, Y = wrapper.read_plot3d(grid_file)
        plot_grid(X, Y, name)

if __name__ == "__main__":
    run_test()
