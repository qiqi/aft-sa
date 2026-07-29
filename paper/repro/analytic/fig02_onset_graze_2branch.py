"""PART IX (2026-07-29): two-branch onset-graze companion figure.

Visual justification of the vg two-branch onset gate
    Re_Omega^c = softmin( A + B/P_I^2 ,  A + B_c/P_curv^2 ),
each softmin branch grazing its OWN family in its OWN coordinate:

  LEFT  (existing coordinate P_I = Omega_hat*<I_hat>+): the inflectional branch
        A + B/P_I^2 grazes the adverse/inflectional family. The strong-FPG /
        low-H members collapse to P_I ~ 0 and VANISH off the left edge (drawn as
        left-edge arrows at their Re_Omega*) -- the onset-resolution limit.
  RIGHT (curvature coordinate P_curv = Omega_hat*<Zhat>+, Zhat = -Z/R the
        normalized curvature indicator, +ve for favorable curvature): the SECOND
        branch A + B_c/P_curv^2 (B_c = the vg value) grazes exactly those low-H
        members (H = 2.216, 2.30, 2.40, 2.50, + Blasius 2.59) that vanish on the
        left. This is the visual proof the two-branch gate resolves onset across
        the whole family where the single-branch (P_I-only) gate could not.

Reuses fig02_onset_graze / _enriched machinery (ODE-exact curvature). Canon
figs/onset_graze.pdf and tex UNTOUCHED. Candidate: figs_explore/
onset_graze_2branch.png (+ graze table onset_graze_2branch.json).

Run from paper/: python3 repro/analytic/fig02_onset_graze_2branch.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _saai  # noqa: F401
from fig02_onset_graze import A_, B_, CEIL, softmin   # k=1 shape: A=175, B=2, CEIL=2600
from fig02_onset_graze_enriched import curve
from fig04_shapefactor import K_ANCHOR        # 0.712
from lib.boundary_layer import FalknerSkanWedge
from lib.correlations import Re_theta0
from scipy.optimize import brentq

OUT_DIR = os.path.join('repro', 'analytic', 'figs_explore')
BC_KCARRY = 129.74                            # the vg curvature-branch B_c (k-carrying)
BC_K1 = BC_KCARRY/K_ANCHOR                    # k=1 shape value (= B/c_o^2 scale)


def br_I(P_I):                                # inflectional branch (k=1 shape)
    return A_ + B_/np.maximum(P_I, 1e-12)**2


def br_curv(P_curv):                          # curvature branch (k=1 shape)
    return A_ + BC_K1/np.maximum(P_curv, 1e-12)**2


def H_of(beta):
    fs = FalknerSkanWedge(beta)
    I = np.trapezoid(fs.u*(1-fs.u), fs.eta)
    return float(np.trapezoid(1-fs.u, fs.eta)/I)


def beta_for_H(Ht):
    if Ht <= 2.2164:
        return 1.0
    return float(brentq(lambda x: H_of(x)-Ht, 1e-4, 1.0))


# LEFT-panel family: the canon adverse/inflectional members (unchanged roles)
LEFT = [(0.15, None), (0.10, None), (0.05, None), (0.0, None),
        (-0.05, None), (-0.10, None), (-0.15, None), (-0.19, None),
        (-0.1988, None), (-0.19, -0.03)]
# the strong-FPG members that VANISH on the left (P_I ~ 0)
VANISH = [1.0, 0.5, 0.35, 0.2]
# RIGHT-panel low-H family (coordinator's list), by target H
RIGHT_H = [2.216, 2.30, 2.40, 2.50, 2.59]
RIGHT_COL = {2.216: 'C3', 2.30: 'C1', 2.40: 'C0', 2.50: 'C2', 2.59: '0.35'}


def main():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 4.9))

    # ---------- LEFT panel: inflectional coordinate P_I (CANON, unchanged) ----
    # canon threshold = softmin(CEIL, A+B/P_I^2): the FLAT CEIL segment is the
    # non-resolving stand-in the vg curvature branch replaces.
    Pg = np.geomspace(3e-3, 1.3, 400)
    axL.loglog(Pg, softmin(Pg), 'k--', lw=2.0, zorder=5,
               label=r'canon $\mathrm{softmin}(C, A+B/P_I^2)$ (k=1)')
    axL.loglog(Pg, K_ANCHOR*softmin(Pg), 'k-.', lw=1.3, zorder=5,
               label=r'model $k\cdot\mathrm{softmin}$')
    axL.axhline(CEIL, color='firebrick', ls=':', lw=1.1, zorder=4)
    cmap = plt.cm.coolwarm
    natt = sum(1 for _, g in LEFT if g is None)
    j = 0
    left_rows = []
    for beta, guess in LEFT:
        c = curve(beta, guess)
        P, ReOm = c['P'], c['ReOm']
        m = (P > 1e-4) & (ReOm > 1.0)
        P, ReOm = P[m], ReOm[m]
        if guess is None:
            col = cmap(j/(natt-1)); j += 1; ls = '-'
        else:
            col = 'purple'; ls = ':'
        axL.loglog(P, ReOm, ls, color=col, lw=1.6)
        ratio = ReOm/softmin(P); i = int(np.argmax(ratio))
        axL.plot(P[i], ReOm[i], 'o', color=col, ms=5.5, mec='k', mew=0.6, zorder=6)
        left_rows.append(dict(beta=beta, H=c['H'], graze_infl=float(ratio[i])))
    axL.set_xlabel(r'inflectional coordinate $P_I=\langle\hat\Omega\hat I\rangle_+$')
    axL.set_ylabel(r'$Re_\Omega=d^2\omega/\nu$ at $Re_\theta=Re_{\theta0}(H)$')
    axL.set_xlim(3e-3, 1.3); axL.set_ylim(30, 1.25e4)
    axL.grid(alpha=0.3, which='both'); axL.legend(fontsize=7.2, loc='lower left')

    # ---------- RIGHT panel: curvature coordinate P_curv ----------
    Pcg = np.geomspace(0.08, 0.5, 400)
    axR.loglog(Pcg, br_curv(Pcg), 'k--', lw=2.0, zorder=5,
               label=rf'curvature branch $A+B_c/P_{{curv}}^2$ (k=1, $B_c={BC_KCARRY:.0f}$)')
    axR.loglog(Pcg, K_ANCHOR*br_curv(Pcg), 'k-.', lw=1.3, zorder=5,
               label=r'model $k\,(A+B_c/P_{curv}^2)$')
    right_rows = []
    for Ht in RIGHT_H:
        beta = beta_for_H(Ht)
        c = curve(beta, None)
        Pz, ReOm = c['Pz'], c['ReOm']
        m = (Pz > 1e-3) & (ReOm > 1.0)
        Pz, ReOm = Pz[m], ReOm[m]
        col = RIGHT_COL[Ht]
        axR.loglog(Pz, ReOm, '-', color=col, lw=1.7,
                   label=rf'$H={c["H"]:.3f}$ ($Re_{{\theta0}}={c["Rtc"]:.0f}$)')
        ratio = ReOm/br_curv(Pz); i = int(np.argmax(ratio))
        axR.plot(Pz[i], ReOm[i], 'o', color=col, ms=6.5, mec='k', mew=0.7, zorder=6)
        right_rows.append(dict(H=c['H'], beta=beta, Re_theta0=c['Rtc'],
                               graze_curv=float(ratio[i]),
                               P_curv_at_graze=float(Pz[i]),
                               ReOm_star=float(ReOm[i])))
        print(f"  H={c['H']:.3f} beta={beta:+.4f} Re_th0={c['Rtc']:6.0f} "
              f"curv-graze={ratio[i]:.3f} at P_curv={Pz[i]:.4f}", flush=True)
    axR.set_xlabel(r'curvature coordinate $P_{curv}=\langle\hat\Omega\hat Z\rangle_+$'
                   r',  $\hat Z=-Z/R$')
    axR.set_ylabel(r'$Re_\Omega$ at $Re_\theta=Re_{\theta0}(H)$')
    axR.set_xlim(0.08, 0.5); axR.set_ylim(3e2, 2e4)
    axR.grid(alpha=0.3, which='both'); axR.legend(fontsize=7.2, loc='upper right')
    plt.tight_layout()
    fp = os.path.join(OUT_DIR, 'onset_graze_2branch.png')
    plt.savefig(fp, dpi=150, facecolor='white')
    print(f'wrote {fp}', flush=True)

    with open(os.path.join(OUT_DIR, 'onset_graze_2branch.json'), 'w') as f:
        json.dump(dict(BC_kcarry=BC_KCARRY, BC_k1=BC_K1,
                       left_inflectional=left_rows,
                       right_curvature=right_rows), f, indent=1)
    print('wrote onset_graze_2branch.json', flush=True)


if __name__ == '__main__':
    main()
