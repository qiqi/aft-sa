import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def get_output_dir():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = os.path.join(project_root, 'output', 'models')
    os.makedirs(out, exist_ok=True)
    return out

from src.physics.correlations import dN_dRe_theta, Re_theta0, compute_nondimensional_spatial_rate

def run():
    # Define H range from 2 to 6
    H_vals = np.linspace(2, 6, 51)  # Start slightly above 2 to avoid singularity at H=1
    # dN_vals = dN_dRe_theta(H_vals) # Unused in this plot, but available
    amp_rate_vals = compute_nondimensional_spatial_rate(H_vals)
    Re_theta0_vals = Re_theta0(H_vals)

    # Plotting with two y-axes
    fig, ax1 = plt.subplots(figsize=(5, 4))

    # Left axis: growth rate
    ax1.semilogy(H_vals, amp_rate_vals, linewidth=2, color='b', label=r'$d\tilde{N}/dRe_\theta$')
    ax1.set_xlabel(r"Shape Factor $H_{12}$")
    ax1.set_ylabel(r"$\theta \frac{d\tilde{N}}{dx}$ (log scale)", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    ax1.set_xlim(2, 5)
    ax1.set_ylim(0.001, 0.1)

    # Vertical reference lines
    ax1.axvline(x=2.59, color='k', linestyle='--', alpha=0.3, label='Blasius (Flat Plate)')
    ax1.axvline(x=4.0, color='r', linestyle='--', alpha=0.3, label='Separation approx.')

    # Right axis: critical Re_theta0
    ax2 = ax1.twinx()
    ax2.semilogy(H_vals, Re_theta0_vals, linewidth=2, color='g', linestyle='-', label=r'$Re_{\theta0}$')
    ax2.set_ylabel(r"$Re_{\theta0}$", color='g')
    ax2.set_ylim([10, 1E4])
    ax2.tick_params(axis='y', labelcolor='g')

    # Build a combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title(r"Growth Rate and Critical $Re_\theta$ vs Shape Factor $H$")
    plt.tight_layout()
    out_path = os.path.join(get_output_dir(), 'drela_correlation.pdf'); plt.savefig(out_path); print(f'Saved: {out_path}')

if __name__ == "__main__":
    run()
