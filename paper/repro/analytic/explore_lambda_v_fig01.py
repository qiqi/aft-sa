"""explore-lambda-v: fig01-equivalent indicator plane (exploratory, figs_explore/).

Left: the committed (Gamma_g, Gamma) plane -- the Falkner-Skan family
overlaps (the sign-blind squared curvature discards the beta grading).
Right: the proposed (Lambda_v, Gamma) plane -- nested petals; Q_v =
1 - sqrt(Gamma(2-Gamma))/(1 + cV|Lambda_v|) iso-lines for the chosen cV = 1.

SIGN CONVENTION: Lambda_v = (d2u/dn2 . (n x omega)) d^3 / ((w d)^2 + u^2)
= -u'u'' d^3/(...) in 2D, i.e. proportional to -d(u'^2)/dn: POSITIVE where
the shear decays away from the wall (all of Blasius, all favorable), and
NEGATIVE only in the sub-inflection shear-growth layer of adverse profiles
(wall slope +beta/(2a)) -- the negative lobe IS the inflectional signature.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/analytic")
import _saai  # noqa: F401
from lib.boundary_layer import FalknerSkanWedge

CV = 1.0
FIGD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs_explore')
UPPER = [(-0.1988, None), (-0.15, None), (-0.10, None), (0.0, None),
         (0.30, None), (1.0, None)]
LOWER = [(-0.19, -0.03), (-0.15, -0.08), (-0.12, -0.10)]


def indicators(fs, beta):
    eta, u, up = fs.eta, fs.u, fs.dudeta
    m = beta/(2 - beta)
    f = np.concatenate([[0.0], np.cumsum(0.5*(u[1:] + u[:-1])*np.diff(eta))])
    upp = -0.5*(m + 1)*f*up - m*(1 - u**2)     # exact f''' (no np.gradient)
    den = (up*eta)**2 + u**2 + 1e-30
    Gam = 2*(up*eta)**2/den
    Gg = (upp**2)*eta**4/den
    Lv = -up*upp*eta**3/den          # flipped sign: Blasius all positive
    return eta, Gam, Gg, Lv


def main():
    os.makedirs(FIGD, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.6))
    allc = UPPER + LOWER
    cmap = plt.cm.viridis
    for k, (beta, guess) in enumerate(allc):
        fs = FalknerSkanWedge(beta, guess=guess)
        eta, Gam, Gg, Lv = indicators(fs, beta)
        mm = eta < np.interp(0.999, np.maximum.accumulate(fs.u), eta)
        if guess:
            # reversed-flow profiles: start at the omega=0 point (velocity
            # minimum). Inside it the profile is just an attached boundary
            # layer running backwards -- its trace duplicates the attached
            # family and carries no shear-layer instability information.
            mm &= eta >= eta[int(np.argmin(fs.u))]
        lab = f'$\\beta$={beta}' + (' (rev)' if guess else '')
        c = cmap(k/(len(allc) - 1))
        ls = '--' if guess else '-'
        axs[0].plot(Gg[mm], Gam[mm], ls, color=c, lw=1.4, label=lab)
        axs[1].plot(Lv[mm], Gam[mm], ls, color=c, lw=1.4, label=lab)

    # Q_v iso-lines on the (Lambda_v, Gamma) plane for the chosen cV
    lv = np.linspace(-1.2, 1.2, 481)
    ga = np.linspace(0.0, 2.0, 401)
    LV, GA = np.meshgrid(lv, ga)
    QV = 1.0 - np.sqrt(np.clip(GA*(2 - GA), 0, None))/(1 + CV*np.abs(LV))
    cs = axs[1].contour(LV, GA, QV, levels=[0.1, 0.3, 0.5, 0.7, 0.9],
                        colors='0.55', linewidths=0.7, linestyles=':')
    axs[1].clabel(cs, fmt='$Q_v$=%.1f', fontsize=6.5)

    axs[0].set_xlabel(r'$\Gamma_g$ (committed indicator)')
    axs[0].set_ylabel(r'$\Gamma$')
    axs[0].set_xlim(-0.05, 1.5)
    axs[1].set_xlabel(r'$\Lambda_v$ (proposed, $c_V$=%g)' % CV)
    axs[1].set_ylabel(r'$\Gamma$')
    axs[1].set_xlim(-1.2, 1.2)
    for ax in axs:
        ax.set_ylim(0.0, 2.02)
        ax.grid(alpha=0.3)
    axs[1].legend(fontsize=7, loc='upper left')
    plt.tight_layout()
    out = os.path.join(FIGD, 'lambda_v_indicator_plane.png')
    plt.savefig(out, dpi=140)
    print('wrote', out)


if __name__ == '__main__':
    main()
