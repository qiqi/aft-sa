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

# Attached family (10 members, +0.35..-0.1988) through incipient separation, plus the
# separated reversed-flow (Stewartson lower-branch) profile at beta=-0.19
# (H~4.9) drawn DOTTED. Deeper members (H>~5) are excluded: Drela's OS data
# stop at H=5 and no similarity family is validated beyond.
# No legend: the betas are listed in the caption; color runs with beta.
PROFILES = [
    (0.15,  None,  None, 'favorable'),
    (0.10,  None,  None, 'favorable'),
    (0.05,  None,  None, 'favorable'),
    (0.0,   None,  None, 'Blasius'),
    (-0.05, None,  None, 'adverse'),
    (-0.10, None,  None, 'adverse'),
    (-0.15, None,  None, 'adverse'),
    (-0.19, None,  None, 'adverse'),
    (-0.1988, None, None, 'incipient separation'),
    (-0.19, -0.03, None, 'separated (reversed)'),
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
    ax.loglog(Pg, softmin(Pg), 'k--', lw=2.0, zorder=5)
    # the model threshold: the same shape scaled by the calibrated k
    # (anchored by the march, Sec. II.D), shifted down by 29%
    from fig04_shapefactor import K_ANCHOR
    ax.loglog(Pg, K_ANCHOR*softmin(Pg), 'k-.', lw=1.4, zorder=5)
    cmap = plt.cm.coolwarm
    natt = sum(1 for p in PROFILES if p[1] is None)
    j = 0
    for beta, guess, _, lab in PROFILES:
        P, ReOm, H, Rtc = curve(beta, guess)
        if guess is None:
            col = cmap(j/(natt - 1)); j += 1
            ls = '-'
        else:
            col = 'purple'; ls = ':'
        ax.loglog(P, ReOm, ls, color=col, lw=1.7)
        ratio = ReOm/softmin(P)
        i = int(np.argmax(ratio))
        ax.plot(P[i], ReOm[i], 'o', color=col, ms=6.0, mec='k', mew=0.6, zorder=6)
        print(f'beta={beta:+.3f} H={H:5.2f} Re_theta_c={Rtc:6.0f} '
              f'graze ratio={ratio[i]:.3f} at Shat*g={P[i]:.3f}', flush=True)
    ax.set_xlabel(r'$\hat S g$')
    ax.set_ylabel(r'$Re_\Omega = d^2\omega/\nu$ at $Re_\theta = Re_{\theta 0}(H)$')
    ax.set_xlim(3e-3, 1.3); ax.set_ylim(30, 6e3)
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('figs/onset_graze.pdf')
    plt.savefig('repro/analytic/figs_explore/onset_graze.png', dpi=140)
    print('wrote figs/onset_graze.pdf')


if __name__ == '__main__':
    main()
