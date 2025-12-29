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

from src.physics.boundary_layer import solve_falkner_skan

def run():
    cases = [
        {'beta': 0.0, 'color': 'r', 'name': r'Blasius ($\beta=0$)'},
        {'beta': 1.0, 'color': 'g', 'name': r'Stagnation ($\beta=1$)'},
        {'beta': -0.19, 'color': 'b', 'name': r'Separation ($\beta=-0.19$)'}
    ]

    fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    for ax, case in zip(axs, cases):
        beta = case['beta']
        col = case['color']
        name = case['name']

        eta, u, dudy, v = solve_falkner_skan(beta)

        # Plot u, shear, and v
        ax.plot(u, eta, color=col, linewidth=2.5, label=r'$u/U_e$')
        ax.plot(dudy, eta, color='k', linestyle=':', linewidth=2, label=r"$f''$ (Shear)")
        ax.plot(v, eta, color='k', linestyle='--', linewidth=1.5, label=r'$v_{sc}$')

        # Formatting
        ax.set_title(name, fontsize=14)
        ax.set_xlabel("Dimensionless Values")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 6)

        # Set x-limits locally because 'v' varies wildly between cases
        if beta == -0.19:
            ax.set_xlim(-0.2, 2.5) # Separation needs room for large positive v
        elif beta == 1.0:
            ax.set_xlim(-1.2, 1.2) # Stagnation needs room for negative v
        else:
            ax.set_xlim(-0.2, 1.2) # Blasius standard range

        ax.axvline(0, color='k', linewidth=0.5)

    # Add y-label only to the first plot
    axs[0].set_ylabel(r"$\eta=y\sqrt{\frac{U_{e}}{\nu x}}$", fontsize=14)

    # Add a single legend to the first plot (representative of all)
    axs[0].legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(get_output_dir(), 'falkner_skan_profiles.pdf'); plt.savefig(out_path); print(f'Saved: {out_path}')

if __name__ == "__main__":
    run()
