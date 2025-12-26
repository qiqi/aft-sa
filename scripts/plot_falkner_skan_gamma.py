import numpy as np
import matplotlib.pyplot as plt
from src.physics.laminar import Re_Omega
from src.physics.boundary_layer import solve_falkner_skan

def run():
    # --- 2. Phase Space Calculation ---
    betas = [1.0, 0.5, 0.1, 0.0, -0.05, -0.1, -0.198]
    Re_theta_tr = [10_000, 3_000, 1_500, 1_200, 450, 300, 100]
    rate_target = [
     0.0006056181711137146,
     0.0006918126210183442,
     0.0013695543979811036,
     0.0023502079523144664,
     0.0036664580824200812,
     0.005267077397042096,
     0.025112605506072142]

    labels = ['Stagnation', 'Strong Favorable', 'Favorable', 'Blasius', 'Adverse', 'Strong Adverse', 'Separation']
    colors = plt.cm.viridis(np.linspace(0, 1, len(betas)))

    # Reference Reynolds Number for scaling Re_Omega (Gamma is invariant)

    # plt.figure(figsize=(7, 8))
    _, ax = plt.subplots(2,1,figsize=(7, 8))

    for beta, Re_theta, label, color in zip(betas, Re_theta_tr, labels, colors):
        eta, u, shear, v_sc = solve_falkner_skan(beta)

        # --- Compute Invariants ---
        # 1. Gamma = 2 (y * Omega)**2 / (U**2 + (y * Omega)**2)
        # Avoid divide by zero at wall (limit is 1.0 for attached, 2.0 for sep)
        gamma = 2 * (eta * shear)**2 / (u**2 + (eta * shear)**2)

        # --- Compute Shape Factor H (for reference) ---
        # Theta = Integral u(1-u) d_eta
        # Delta_star = Integral (1-u) d_eta
        theta = np.trapezoid(u * (1 - u), eta)
        delta_star = np.trapezoid(1 - u, eta)
        H = delta_star / theta

        # --- Plotting ---
        ax[0].plot(gamma[1:], Re_Omega(shear, eta)[1:] * Re_theta / theta,
                 color=color, linewidth=2.5,
                 label=f'{label} (H={H:.1f})')

        # --- Plotting ---
        amax = np.max(np.log10(Re_Omega(shear, eta)[1:] * Re_theta / theta / 1000) / 50 + gamma[1:])
        ax[1].plot(amax, rate_target[Re_theta_tr.index(Re_theta)], 'o', color=color)

    # --- 3. Decorate the Map ---
    ax[0].set_title(f"Transition Phase Space: $\\Gamma$ vs $Re_\\Omega$ (Calculated at $Re_\\theta = 500$)")
    ax[0].set_xlabel(r"Detachment Parameter $\Gamma = \frac{d \cdot \Omega}{U_{rel}}$")
    ax[0].set_ylabel(r"Vorticity Reynolds $Re_\Omega = \frac{d^2 \Omega}{\nu}$")
    ax[0].set_xlim(0.8, 1.1)
    ax[0].set_ylim(10, 10000)
    ax[0].grid(True, which='both', alpha=0.3)
    ax[0].legend(loc='upper left')

    amax = np.linspace(0.9, 1.5, 61)
    ax[1].plot(amax, 0.2 / (1 + np.exp(-3.8 * (amax - 2.01))) / (1 + np.exp(-40 * (amax - 0.97))))

    re_omega = np.logspace(1, 4, 51)
    Gamma = np.linspace(0, 2, 51)
    re_omega, Gamma = np.meshgrid(re_omega, Gamma, indexing='ij')
    # amp_rate = compute_nondimensional_amplification_rate(re_omega, Gamma)
    a = np.log10(re_omega / 1000) / 50 + Gamma
    ax[0].clabel(ax[0].contour(Gamma, re_omega, a, [0.9, 0.95, 1, 1.05, 1.1], linestyles=':'))
    #amp_rate = np.exp(5) * 0.005
    #plt.clabel(plt.contour(Gamma, re_omega, amp_rate, [0.0001, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1], linestyles=':'))

    ax[0].set_yscale('log')
    ax[0].set_ylim([10, 10000])
    ax[1].set_yscale('log')
    
    plt.savefig('falkner_skan_gamma.pdf')
    print("Plot saved to falkner_skan_gamma.pdf")

if __name__ == "__main__":
    run()
