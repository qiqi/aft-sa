import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def get_output_dir():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = os.path.join(project_root, 'output', 'models')
    os.makedirs(out, exist_ok=True)
    return out

from src.solvers.boundary_layer_solvers import NuHatFlatPlateSolver

def run():
    solver = NuHatFlatPlateSolver()

    # --- RUNNING WITH BATCH ---
    # Example: Run 3 simulations with different Tu
    Tu_batch = [0.0001, 0.01, 1.0, 5.0]
    u, v, nuHat = solver.forward(Tu_batch)
    x_grid = solver.x_grid
    y_u = solver.y_cell.detach().cpu().numpy()

    plt.figure(figsize=(9, 12))
    symbols = ['o', 's', '^', 'v']
    
    for batch_idx in range(4):
        u_np = u[:, batch_idx, :].detach().cpu().numpy()
        nu_np = nuHat[:, batch_idx, :].detach().cpu().numpy()

        # Calculate Cf
        tau_w = 1.0 * u_np[:,0] / y_u[0]
        cf = tau_w * 2.0

        # Re_theta calculation
        Re_theta_list = []
        for i in range(u_np.shape[0]):
            theta = np.sum(u_np[i,:] * (1 - u_np[i,:]) * solver.dy_vol.numpy())
            Re_theta_list.append(theta)

        Re_theta = np.array(Re_theta_list)
        name = f'Tu={Tu_batch[batch_idx]}%'

        # Plotting
        ax = plt.subplot(5, 2, 2*batch_idx+1)
        levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        plt.clabel(plt.contour(x_grid, y_u, u_np.T, levels))
        plt.title(r'$u$, ' + name, y=0.8)
        plt.ylim([-1000, 15000])
        plt.ylabel('y')
        if batch_idx <= 2:
            ax.set_xticklabels([])
        else:
            plt.xlabel('x')

        ax = plt.subplot(5, 2, 2*batch_idx+2)
        levels = [-3.5, -3.0, -2.5, -2.0, -1.5, -1, -0.5, 0, 0.5, 1.0, 1.5, 2.0]
        plt.clabel(plt.contour(x_grid, y_u, np.log10(nu_np.T), levels))
        plt.title(r'$\log(\hat\nu)$, ' + name, y=0.8)
        plt.ylim([-1000, 15000])
        ax.set_yticklabels([])
        if batch_idx <= 2:
            ax.set_xticklabels([])
        else:
            plt.xlabel('x')

        plt.subplot(6, 1, 6)
        plt.loglog(Re_theta[1:], cf[1:], symbols[batch_idx], mfc='w', label=name)

    plt.loglog(Re_theta[1:], 0.441 / Re_theta[1:], 'k:', label='Laminar')
    cf_turb = 2.0 * (1.0 / 0.38 * np.log(Re_theta[1:]) + 3.7)**(-2)
    plt.loglog(Re_theta[1:], cf_turb, 'r--', label='Turbulent Correlation')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.xlim([1e2, 1e4])
    plt.ylim([1e-4, 1e-2])
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$c_f$')
    
    out_path = os.path.join(get_output_dir(), 'flat_plate_batch.pdf'); plt.savefig(out_path); print(f'Saved: {out_path}')

if __name__ == "__main__":
    run()
