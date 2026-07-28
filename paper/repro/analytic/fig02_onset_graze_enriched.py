"""Part IV (2026-07-28, user directive): enrich fig:onsetgraze with
strongly-favorable members, all the way to the stagnation-point profile.

CANDIDATE figure -- does NOT overwrite paper/figs/onset_graze.pdf. Outputs
(all under repro/analytic/figs_explore/):
    onset_graze_enriched.pdf/.png   adoption candidate (canon styling)
    onset_graze_flatness.png        DIAGNOSTIC companion (not for the paper)
    onset_graze_enriched.json       tables

METHOD (requirement 1): the canonical figure's "LST neutral point" is each
Falkner-Skan profile evaluated at its DRELA-GILES Eq. 30 critical Reynolds
number Re_theta0(H) (repo fit, lib/correlations.py) -- a correlation
distilled from Drela's Orr-Sommerfeld database, NOT a per-profile OS solve.
The same method is used for the new members beta = {0.35, 0.5, 0.7, 1.0}:
Eq. 30 is smooth down to the stagnation H = 2.216 (Re_theta0 = 6668); the
published OS critical for the Hiemenz profile (Wazzan, Okamura & Smith
1968: Re_delta*_crit ~ 12490 -> Re_theta0 ~ 5640) sits ~15% below it, so
the extension is defensible; the figure inherits the correlation's low-H
uncertainty.

KEY STRUCTURAL FINDING (changes how the members can be drawn): with the
curvature indicator computed EXACTLY from the repo's FS ODE
(f''' = -[f f'' + beta(1-f'^2)]/(2-beta), Blasius-consistent eta; the
np.gradient of the canon script agrees to all digits on this smooth
profile), the amplifying coordinate P = Omega_hat*I_hat satisfies
    max_y P = 1.7e-3 at beta = 0.35,   P <= 0 EVERYWHERE for beta >= 0.5
(the 2026-07-28-1041 audit's ~5e-4 readings at beta >= 0.5 were estimator
noise on RANS fields; the rig profile is clean). So beta = 0.35 joins the
canon (P, Re_Omega) plane normally, while the beta >= 0.5 members have NO
locus in the plane at all -- they are drawn as left-edge markers at their
profile-max Re_Omega and quantified against the ceiling in the table.
That IS the flatness mechanism: beyond beta ~ 0.4 the gate coordinate
itself dies, so any threshold Re_Omega_c(P), capped or not, presents its
P->0 limit (the ceiling) to these layers while Drela's Re_theta0 keeps
rising.

Run from paper/:  python3 repro/analytic/fig02_onset_graze_enriched.py
"""
import json
import os

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _saai  # noqa: F401
from fig02_onset_graze import CEIL, A_, B_, N_, softmin
from fig04_shapefactor import K_ANCHOR
from lib.boundary_layer import FalknerSkanWedge
from lib.correlations import Re_theta0

OUT_DIR = os.path.join('repro', 'analytic', 'figs_explore')
C8000_K1 = 8000.0/K_ANCHOR                 # Part-III retuned ceiling, k=1 scale

# canon members (UNCHANGED, same order/roles) + the Part-IV enrichment
NEW = [1.0, 0.7, 0.5, 0.35]
PROFILES = ([(b, None, 'favorable (new)') for b in NEW]
            + [(0.15, None, 'favorable'),
               (0.10, None, 'favorable'),
               (0.05, None, 'favorable'),
               (0.0, None, 'Blasius'),
               (-0.05, None, 'adverse'),
               (-0.10, None, 'adverse'),
               (-0.15, None, 'adverse'),
               (-0.19, None, 'adverse'),
               (-0.1988, None, 'incipient separation'),
               (-0.19, -0.03, 'separated (reversed)')])

# canon graze ratios (fig02_onset_graze printout), to verify the legacy
# members are untouched by the exact-curvature refactor
CANON_RATIO = {0.15: 1.004, 0.10: 1.085, 0.05: 1.137, 0.0: 1.035,
               -0.05: 0.959, -0.10: 0.996, -0.15: 1.037, -0.19: 0.984,
               -0.1988: 0.932}


def softmin_general(P, ceil):
    pw = A_ + B_/np.maximum(P, 1e-9)**2
    return (ceil**(-N_) + pw**(-N_))**(-1.0/N_)


def curve(beta, guess):
    """(P, Re_Omega) locus of the profile at its Drela-Giles critical
    Re_theta0(H), with ODE-exact curvature; also the Part-III viscous
    coordinate P_o = Omega_hat*<-Z>+/R for the diagnostic."""
    fs = FalknerSkanWedge(beta, guess=guess) if guess is not None \
        else FalknerSkanWedge(beta)
    eta, u, up = fs.eta, fs.u, fs.dudeta
    f = np.concatenate([[0.0], np.cumsum(0.5*(u[1:] + u[:-1])*np.diff(eta))])
    upp = -(f*up + beta*(1.0 - u*u))/(2.0 - beta)     # repo FS ODE, exact
    I_th = float(np.trapezoid(u*(1 - u), eta))
    H = float(np.trapezoid(1 - u, eta))/I_th
    Rtc = float(np.asarray(Re_theta0(H)))
    X = u; Y = eta*up; Z = 0.5*eta**2*upp
    R = np.sqrt(X*X + Y*Y + Z*Z) + 1e-30
    Shat = Y/np.sqrt(X*X + Y*Y + 1e-30)
    g = (Y - X - Z)/R
    P = Shat*g
    Pz = Shat*np.clip(-Z/R, 0.0, None)
    ReOm = eta**2*np.abs(up)*(Rtc/I_th)     # eta^2*|u'|*sqrt(Re_x) at Rtc
    ok = (eta < 8.0) & (ReOm > 1.0)         # drop far-field integration residue
    return dict(P=P[ok], Pz=Pz[ok], ReOm=ReOm[ok], H=H, Rtc=Rtc)


def main():
    rows = []
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    Pg = np.geomspace(1e-4, 1.2, 500)
    ax.loglog(Pg, softmin(Pg), 'k--', lw=2.0, zorder=5)
    ax.loglog(Pg, K_ANCHOR*softmin(Pg), 'k-.', lw=1.4, zorder=5)
    cmap = plt.cm.coolwarm
    natt = sum(1 for p in PROFILES if p[1] is None)
    j = 0
    curves = {}
    for beta, guess, lab in PROFILES:
        c = curve(beta, guess)
        curves[(beta, guess is not None)] = c
        if guess is None:
            col = cmap(j/(natt - 1)); j += 1
            ls = '-'
        else:
            col = 'purple'; ls = ':'
        m = c['P'] > 1e-4                      # canon plotting mask
        row = dict(beta=beta, lower=guess is not None, H=c['H'],
                   Rtc=c['Rtc'], ReOm_max=float(c['ReOm'].max()),
                   maxP=float(c['P'].max()))
        if m.sum() >= 2:                       # in-plane member (canon path)
            P, ReOm = c['P'][m], c['ReOm'][m]
            ax.loglog(P, ReOm, ls, color=col, lw=1.7)
            ratio = ReOm/softmin(P)
            i = int(np.argmax(ratio))
            ax.plot(P[i], ReOm[i], 'o', color=col, ms=6.0, mec='k', mew=0.6,
                    zorder=6)
            row.update(P_star=float(P[i]), ReOm_star=float(ReOm[i]),
                       graze_canon=float(ratio[i]),
                       graze_C8000=float(ReOm[i]
                                         / softmin_general(P[i], C8000_K1)))
        else:                                  # beta >= 0.5: P <= 0 interior
            ax.annotate('', xy=(1.15e-4, c['ReOm'].max()),
                        xytext=(2.6e-4, c['ReOm'].max()),
                        arrowprops=dict(arrowstyle='->', color=col, lw=1.6))
            ax.text(2.9e-4, c['ReOm'].max(), rf'$\beta\!=\!+{beta:g}$',
                    fontsize=7, color=col, va='center')
            # its P -> 0: every threshold presents the CEILING to it
            row.update(P_star=None, ReOm_star=float(c['ReOm'].max()),
                       graze_canon=float(c['ReOm'].max()/CEIL),
                       graze_C8000=float(c['ReOm'].max()/C8000_K1))
        rows.append(row)
        ps = 'P<=0 interior' if row['P_star'] is None \
            else f"P*={row['P_star']:.2e}"
        tag = '  NEW' if beta in NEW else ''
        print(f"beta={beta:+.3f} H={c['H']:5.3f} Re_theta0={c['Rtc']:6.0f} "
              f"graze={row['graze_canon']:.3f} (C8000: "
              f"{row['graze_C8000']:.3f}) ReOm*={row['ReOm_star']:.0f} "
              f"{ps}{tag}", flush=True)
        if beta in CANON_RATIO and guess is None:
            d = row['graze_canon']/CANON_RATIO[beta] - 1.0
            assert abs(d) < 0.02, (beta, row['graze_canon'])
    print('legacy members reproduce the canon graze ratios to <2%. OK',
          flush=True)

    ax.set_xlabel(r'$\hat\Omega \hat I$')
    ax.set_ylabel(r'$Re_\Omega = d^2\omega/\nu$ at $Re_\theta = Re_{\theta 0}(H)$')
    ax.set_xlim(1e-4, 1.3); ax.set_ylim(30, 1e5)
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'onset_graze_enriched.pdf'))
    plt.savefig(os.path.join(OUT_DIR, 'onset_graze_enriched.png'), dpi=150)
    print('wrote onset_graze_enriched.{pdf,png}', flush=True)

    # ------------- DIAGNOSTIC companion (figs_explore only) --------------
    # Left: the members' needed threshold Re_Omega*(H) vs any constant
    # ceiling -- the flatness in one panel. Right: the beta >= 0.35 loci in
    # the Part-III viscous coordinate Omega_hat*<-Z>+/R, where they exist,
    # with the neutral points and a tracking power-law fit.
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    Hs = [r['H'] for r in rows if not r['lower']]
    Om = [r['ReOm_star'] for r in rows if not r['lower']]
    axl.semilogy(Hs, Om, '-o', color='C0', ms=5,
                 label=r'member neutral $Re_\Omega^*$ at $Re_{\theta 0}(H)$')
    axl.axhline(CEIL, color='k', ls='--', lw=1.2,
                label=r'canon ceiling $C=2600$ ($k\!=\!1$)')
    axl.axhline(C8000_K1, color='C3', ls='-', lw=1.2,
                label=r'retuned $C=8000$ ($k\!=\!1$: 11236)')
    axl.set_xlabel(r'$H$'); axl.set_ylabel(r'$Re_\Omega$')
    axl.set_xlim(2.15, 4.1); axl.grid(alpha=0.3, which='both')
    axl.legend(fontsize=7.5, loc='upper right')
    axl.set_title('any constant ceiling goes flat; the neutral locus '
                  'keeps rising', fontsize=9)

    # right panel: what coordinate CARRIES the rise? The neutral Re_Omega*
    # tracks the DG critical station itself (slope ~1), while the two local
    # candidates fail: P dies (<= 0 at beta >= 0.5) and the Part-III viscous
    # coordinate P_o barely varies across the strong-FPG family.
    att = [r for r in rows if not r['lower']]
    Rt0 = np.array([r['Rtc'] for r in att])
    Oms = np.array([r['ReOm_star'] for r in att])
    axr.loglog(Rt0, Oms, 'o', color='C0', ms=6, mec='k', mew=0.5,
               label='attached members')
    for r in att:
        if r['beta'] in NEW:
            axr.annotate(rf'$+{r["beta"]:g}$', (r['Rtc'], r['ReOm_star']),
                         textcoords='offset points', xytext=(6, -3),
                         fontsize=7)
    rr = np.geomspace(25, 9000, 50)
    axr.loglog(rr, 1.45*rr, 'k--', lw=1.2,
               label=r'$Re_\Omega^*\!=\!1.45\,Re_{\theta 0}$ (guide)')
    axr.axhline(CEIL, color='0.4', ls=':', lw=1.2,
                label=r'constant ceiling $C$ (any value): flat')
    axr.set_xlabel(r'Drela--Giles $Re_{\theta 0}(H)$')
    axr.set_ylabel(r'neutral $Re_\Omega^*$')
    axr.grid(alpha=0.3, which='both'); axr.legend(fontsize=7.5, loc='upper left')
    axr.set_title(r'the rise is carried by $Re_{\theta 0}$ itself; local'
                  r' coordinates: $P$ dies, $P_o$ stays $0.12$--$0.16$',
                  fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'onset_graze_flatness.png'), dpi=150)

    # viscous-coordinate power fit through the new members' neutral points
    # (max-ReOm point of each locus) -- recorded with its ill-conditioning
    pts = []
    for beta in NEW:
        c = curves[(beta, False)]
        m = (c['Pz'] > 1e-4)
        i = int(np.argmax(c['ReOm'][m]))
        pts.append((beta, float(c['Pz'][m][i]), float(c['ReOm'][m][i])))
    lp = np.log([p for _, p, _ in pts]); lo = np.log([o for _, _, o in pts])
    m_fit, c_fit = np.polyfit(lp, lo, 1)
    D_fit = float(np.exp(c_fit))
    print('wrote onset_graze_flatness.png', flush=True)
    print(f'viscous-coordinate neutral points (beta, P_o*, ReOm*): {pts}',
          flush=True)
    print(f'power fit ReOm* ~ {D_fit:.0f} * P_o^{m_fit:+.2f} -- '
          f'ILL-CONDITIONED (P_o spans only '
          f'{min(p for _, p, _ in pts):.3f}-{max(p for _, p, _ in pts):.3f})',
          flush=True)

    with open(os.path.join(OUT_DIR, 'onset_graze_enriched.json'), 'w') as f:
        json.dump(dict(rows=rows, viscous_pts=pts,
                       fit_power=float(m_fit), fit_coef=D_fit,
                       fit_caveat='ill-conditioned: P_o spans 0.12-0.16',
                       method='Drela-Giles Eq.30 Re_theta0(H) '
                              '(lib/correlations.py); FS-ODE-exact curvature;'
                              ' beta>=0.5 members have P<=0 interior'),
                  f, indent=1)
    print('wrote onset_graze_enriched.json', flush=True)


if __name__ == '__main__':
    main()
