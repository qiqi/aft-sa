"""fig:onsetgraze -> paper/figs/onset_graze.pdf.

The LST neutral-point graze that fixes the SHAPE of the onset threshold
Re_Omega_crit(S_hat*g). Each Falkner-Skan profile, evaluated at its
Drela-Giles critical Reynolds number Re_theta_c(H), traces a curve in the
(S_hat*g, Re_Omega) plane (log-log); the soft-min threshold
    Re_Omega_crit = softmin_n(CEIL, A + B/(S_hat*g)^2),  n = 2,
with shape constants (CEIL, A, B) = (2600, 175, 2), grazes the attached
family's outer envelope (markers: each profile's closest-approach point).
The separated reversed-flow profile rides above the threshold at its
critical Reynolds number -- it ignites immediately, as it should.
The single multiplicative scale k that converts the shape into the model
threshold is anchored at the marched Blasius N=1 crossing (Sec. II.D).

Run from paper/: python3 repro/analytic/fig02_onset_graze.py
"""
import _saai  # noqa: F401
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lib.boundary_layer import FalknerSkanWedge
from lib.correlations import Re_theta0

CEIL, A_, B_, N_ = 2600.0, 175.0, 2.0, 2.0

PROFILES = [
    (0.15,  None,  'C0', r'favorable $\beta=+0.15$'),
    (0.0,   None,  'C2', r'Blasius $\beta=0$'),
    (-0.10, None,  'C1', r'adverse $\beta=-0.10$'),
    (-0.1988, None, 'C3', r'incipient separation'),
    (-0.12, -0.10, 'C4', r'separated (reversed flow)'),
]


def curve(beta, guess):
    fs = FalknerSkanWedge(beta, guess=guess) if guess is not None else FalknerSkanWedge(beta)
    eta, u, up = fs.eta, fs.u, fs.dudeta
    I_th = float(np.trapezoid(u*(1 - u), eta))
    H = float(np.trapezoid(1 - u, eta))/I_th
    Rtc = float(np.asarray(Re_theta0(H)))
    sqRex = Rtc/I_th
    upp = np.gradient(up, eta)
    X = u; Y = eta*up; Z = 0.5*eta**2*upp
    R = np.sqrt(X*X + Y*Y + Z*Z) + 1e-30
    Shat = Y/np.sqrt(X*X + Y*Y + 1e-30)
    g = (Y - X - Z)/R
    P = Shat*g
    ReOm = eta**2*np.abs(up)*sqRex
    m = (P > 1e-4) & (ReOm > 1.0)
    return P[m], ReOm[m], H, Rtc


def softmin(P):
    pw = A_ + B_/np.maximum(P, 1e-9)**2
    return (CEIL**(-N_) + pw**(-N_))**(-1.0/N_)


def main():
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    Pg = np.geomspace(3e-3, 1.2, 400)
    ax.loglog(Pg, softmin(Pg), 'k--', lw=2.0, zorder=5,
              label=r'$\mathrm{softmin}_2(2600,\ 175+2\,(\hat S g)^{-2})$')
    for beta, guess, col, lab in PROFILES:
        P, ReOm, H, Rtc = curve(beta, guess)
        ax.loglog(P, ReOm, '-', color=col, lw=1.7,
                  label=lab + fr'  ($H\!=\!{H:.2f}$, $Re_{{\theta c}}\!=\!{Rtc:.0f}$)')
        ratio = ReOm/softmin(P)
        i = int(np.argmax(ratio))
        ax.plot(P[i], ReOm[i], 'o', color=col, ms=6.5, mec='k', mew=0.6, zorder=6)
        print(f'beta={beta:+.3f} H={H:5.2f} Re_theta_c={Rtc:6.0f} '
              f'graze ratio={ratio[i]:.3f} at Shat*g={P[i]:.3f}', flush=True)
    ax.set_xlabel(r'$\hat S g$')
    ax.set_ylabel(r'$Re_\Omega = d^2\omega/\nu$ at $Re_\theta = Re_{\theta c}(H)$')
    ax.set_xlim(3e-3, 1.3); ax.set_ylim(30, 6e3)
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=7.2, loc='lower left')
    plt.tight_layout()
    plt.savefig('figs/onset_graze.pdf')
    plt.savefig('repro/analytic/figs_explore/onset_graze.png', dpi=140)
    print('wrote figs/onset_graze.pdf')


if __name__ == '__main__':
    main()
