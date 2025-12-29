import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.solvers.boundary_layer_solvers import NuHatBlasiusSolver

def run():
    solver = NuHatBlasiusSolver()
    nuHat = solver(1.0)
    x = np.arange(solver.nx + 1) * solver.dx
    y = (np.arange(solver.ny) + 0.5) * solver.dy
    
    plt.figure(figsize=(6,4))
    #plt.contour(x, y, np.log(nuHat.detach().cpu().numpy().T), [0], colors='r');
    plt.plot(x, 0.665 * np.sqrt(x), '--k')
    plt.plot(x, 1.72 * np.sqrt(x), '-k')
    plt.plot(x, 5 * np.sqrt(x), ':k')
    plt.ylim([0, 9000])
    plt.legend([r'$\theta$', r'$\delta^*$', r'$\delta_{99}$'])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    
    levels = np.linspace(0, 11, 12)
    levels[0] = 0.1
    plt.clabel(plt.contour(x, y, np.log(nuHat.detach().cpu().numpy().T), levels))
    plt.xlabel(r'$Re_x$')
    plt.ylabel(r'$Re_y$')
    
    plt.savefig('blasius_nuHat_solution.pdf')
    print("Plot saved to blasius_nuHat_solution.pdf")

if __name__ == "__main__":
    run()
