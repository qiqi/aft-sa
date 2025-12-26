import numpy as np
import matplotlib.pyplot as plt
from src.physics.boundary_layer import Blasius
from src.physics.laminar import Re_Omega, amplification

def run():
    b = Blasius()
    y = np.arange(5000)
    Rex_list = np.array([50_000, 200_000, 500_000, 1_000_000, 2_000_000])

    fig, ax = plt.subplots(1, 4, figsize=(10,4), sharey='row')
    axp = [a.twiny() for a in ax]

    for Rex in Rex_list:
        yCell, u, dudy, _ = b.at(Rex, y)
        for a in axp:
            a.plot(u, yCell, '--', linewidth=2, alpha=0.5)
        ax[0].plot(Re_Omega(dudy, yCell) / 2.2, yCell, linewidth=3)
        ax[1].plot(2 * (dudy * yCell)**2 / (u**2 + (dudy * yCell)**2), yCell, linewidth=3)
        ax[2].plot(amplification(u, dudy, yCell) * 1e3, yCell, linewidth=3)
        ax[3].plot(dudy * 1000, yCell, linewidth=3)

    for a in ax:
        a.set_ylim([0, 5000])
    for a in axp:
        a.set_xticks([])
        
    ax[0].legend([f'$\\theta = {0.665 * np.sqrt(Rex):.1f}$' for Rex in Rex_list], loc='upper left')
    ax[0].set_ylabel(r"$y$")
    ax[0].set_xlabel(r"$Re_{\Omega} / 2.2$")
    ax[1].set_xlabel(r"$\Gamma$")
    ax[2].set_xlabel(r"amplfication $(\times 10^{-3})$")
    ax[3].set_xlabel(r"$dU/dy (\times 10^{-3})$")
    
    plt.savefig('blasius_ReOmega_growth.pdf')
    print("Plot saved to blasius_ReOmega_growth.pdf")

if __name__ == "__main__":
    run()
