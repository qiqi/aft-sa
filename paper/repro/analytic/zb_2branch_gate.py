"""TASK 3 / PART VIII (2026-07-29): the two-branch onset gate.

USER PROPOSAL: the rate is two-branch but the canon gate is single-branch
(softmin(C, A+B/P_I^2), P_I = Om*I only). At low H, P_I ~ 0 for all profiles
(Task-2 sphere degeneracy) so the softmin saturates to the constant C -- a
non-H-resolving stand-in. FIX: give the gate a SECOND, curvature branch using
the same viscous coordinate the rate uses:
    Re_Omega^c = softmin_2( A + B/P_I^2 ,  A_c + B_c/P_curv^2 ),   NO constant C
P_curv = Om*<-Z>+/R. Form 'vg' in fpg_recalibration_study.

This driver: (1) calibrates B_c (A_c = A shared, net-zero) by grazing the
stagnation member at 1; (2) tabulates the vg onset threshold across the FPG
family vs the constant C (does it resolve H=2.2 from 2.3?); (3) marches the
favorable ladder for canon / vbs / vg and reports both-panel deviations +
low-H onsets; (4) regression: separated/Stewartson branch (curvature branch
must NOT fire), wall-boundedness, Spalart zero-suite; (5) candidate figure.

OFFLINE; no tex/solver edits. Writes figs_explore artifacts only.
"""
import os
import json
import numpy as np

import _saai  # noqa: F401
import fig04_shapefactor as f4
import fpg_recalibration_study as S
from lib.boundary_layer import FalknerSkanWedge
from lib.correlations import dN_dRe_theta, Re_theta0

OUT_DIR = os.path.join('repro', 'analytic', 'figs_explore')
EPS_R = 0.1455                 # a_visc = 0.0276 (shared by vbs & vg rate)
CO_VBS = 0.1046               # Part-V vbs gate blend weight
K = f4.K_ANCHOR               # 0.712
REOM_A, REOM_B = f4.REOM_A, f4.REOM_B   # k-carrying 124.6, 1.424
REOM_N = f4.REOM_N


def graze_vg(beta, BC, guess=None):
    """graze ratio (max ReOm/threshold at k=1 shape) for the vg gate."""
    S.FORM[0] = 'vg'; S.EPS[0] = EPS_R; S.AC[0] = None; S.BC[0] = BC
    fs = FalknerSkanWedge(beta, guess=guess) if guess is not None \
        else FalknerSkanWedge(beta)
    eta, u, upr = fs.eta, fs.u, fs.dudeta
    I_th = float(np.trapezoid(u*(1-u), eta)); H = float(np.trapezoid(1-u, eta))/I_th
    Rtc = float(np.asarray(Re_theta0(H)))
    upp = np.gradient(upr, eta)
    X = u; Y = eta*upr; Z = 0.5*eta**2*upp
    R = np.sqrt(X*X+Y*Y+Z*Z)+1e-30
    Shat = Y/np.sqrt(X*X+Y*Y+1e-30); g = (Y-X-Z)/R
    P, thr_k = S._P_and_thresh(Shat, g, Z/R)
    thr = thr_k/K                          # k=1 shape
    ReOm = eta**2*np.abs(upr)*(Rtc/I_th)
    m = (ReOm > 1.0)
    return float((ReOm[m]/thr[m]).max()), H, Rtc


def calibrate_BC():
    """B_c so the stagnation member (beta=1) grazes at 1 (vbs-style anchor)."""
    def f(bc):
        return graze_vg(1.0, bc)[0] - 1.0
    lo, hi = 20.0, 600.0
    flo, fhi = f(lo), f(hi)
    for _ in range(40):
        mid = 0.5*(lo+hi); fm = f(mid)
        if flo*fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
        if abs(fm) < 1e-3:
            break
    return 0.5*(lo+hi)


def threshold_table(BC):
    """vg onset threshold Re_Omega^c(k=1) vs the constant C=2600, per member;
    shows the two-branch gate resolves distinct onsets at low H."""
    betas = [(1.0, None), (0.5, None), (0.35, None), (0.2, None), (0.1, None),
             (0.0, None), (-0.10, None), (-0.1988, None), (-0.19, -0.03)]
    rows = []
    for beta, guess in betas:
        gr, H, Rtc = graze_vg(beta, BC, guess=guess)
        # the marched-relevant threshold: reconstruct br_I, br_c at the profile
        fs = FalknerSkanWedge(beta, guess=guess) if guess is not None \
            else FalknerSkanWedge(beta)
        eta, u, upr = fs.eta, fs.u, fs.dudeta
        upp = np.gradient(upr, eta)
        Xx = u; Yy = eta*upr; Zz = 0.5*eta**2*upp
        Rr = np.sqrt(Xx*Xx+Yy*Yy+Zz*Zz)+1e-30
        Sh = Yy/np.sqrt(Xx*Xx+Yy*Yy+1e-30); gg = (Yy-Xx-Zz)/Rr
        P_I = Sh*np.clip(gg, 0, None); P_curv = Sh*np.clip(-Zz/Rr, 0, None)
        brI = REOM_A + REOM_B/np.maximum(P_I, 1e-9)**2
        brc = REOM_A + BC/np.maximum(P_curv, 1e-9)**2
        # value at the max-ReOm (neutral) point
        ReOm = eta**2*np.abs(upr)*(float(np.asarray(Re_theta0(H)))/np.trapezoid(u*(1-u), eta))
        j = int(np.argmax(ReOm))
        rows.append(dict(beta=beta, H=H, Re_theta0=Rtc, graze_vg=gr,
                         br_I_at_np=float(brI[j]), br_curv_at_np=float(brc[j]),
                         constant_C=2600.0, wins='curv' if brc[j] < brI[j] else 'infl'))
        print(f"  beta={beta:+.3f} H={H:5.3f} Re_th0={Rtc:6.0f} graze_vg={gr:.3f} "
              f"| br_I={brI[j]:8.0f} br_curv={brc[j]:7.0f} const_C=2600 "
              f"-> {rows[-1]['wins']} wins", flush=True)
    return rows


def march_form(form, betas, **kw):
    """set the form + its knobs, then ladder()."""
    S.FORM[0] = form; S.EPS[0] = EPS_R
    if form == 'vbs':
        S.CO[0] = CO_VBS
    if form == 'vg':
        S.AC[0] = None; S.BC[0] = kw['BC']
    rows = [S.row(b) for b in betas]
    return rows


def scores(rows):
    return S.scores(rows)


def main():
    print('== calibrate B_c (A_c = A shared; beta=1 graze = 1) ==', flush=True)
    BC = calibrate_BC()
    BC_eq_vbs = REOM_B/CO_VBS**2          # vbs strong-FPG limit equivalent
    print(f"  B_c = {BC:.2f} (k-carrying); vbs strong-FPG equiv B/c_o^2 = "
          f"{BC_eq_vbs:.2f}", flush=True)

    print('\n== onset-threshold table: two-branch gate vs constant C ==',
          flush=True)
    tbl = threshold_table(BC)

    FAV = [1.0, 0.55, 0.35, 0.2, 0.1, 0.05, 0.0]
    ADV = [-0.03, -0.06, -0.09, -0.12, -0.15, -0.18, -0.1988]
    LOW = [(-0.19, -0.03), (-0.17, -0.06), (-0.15, -0.08), (-0.12, -0.10)]

    # BASELINES from the zb JSON (same 1600x1200 grid, same two-branch RATE):
    # canon = variant B with the CONSTANT-C gate (C=1851, the zb-figure BLUE
    # curve); C8000 = the green curve. The controlled comparison is: same rate,
    # three GATES (constant-C softmin / vbs P-blend / vg two-branch softmin).
    zbj = json.load(open(os.path.join(OUT_DIR, 'fpg_recalibration_zb.json')))
    canon = [r for r in zbj['final'] if r['beta'] >= 0]
    green = [r for r in zbj['final_C8000'] if r['beta'] >= 0]
    print('== baselines loaded: canon-C1851 (blue), C8000 (green) from zb JSON ==',
          flush=True)
    print('== march favorable: vbs (c_o=0.1046) ==', flush=True)
    vbs = march_form('vbs', FAV)
    print('== march favorable: vg (two-branch gate) ==', flush=True)
    vg = march_form('vg', FAV, BC=BC)

    da_c, db_c = scores(canon); da_v, db_v = scores(vbs); da_g, db_g = scores(vg)
    da_gr, db_gr = scores(green)
    print(f"\n  DEV canon-C1851: (a){np.exp(da_c):.2f}x (b){np.exp(db_c):.2f}x",
          flush=True)
    print(f"  DEV C8000: (a){np.exp(da_gr):.2f}x (b){np.exp(db_gr):.2f}x",
          flush=True)
    print(f"  DEV vbs  : (a){np.exp(da_v):.2f}x (b){np.exp(db_v):.2f}x", flush=True)
    print(f"  DEV vg   : (a){np.exp(da_g):.2f}x (b){np.exp(db_g):.2f}x", flush=True)

    # low-H onset resolution: H=2.216 vs H=2.285 (beta 1 vs 0.55)
    def onset(rows):
        return {round(r['H'], 3): (r['Rt1'], r['Rt1']/r['Rt1_DG'],
                                   r['s_late']/r['s_DG']) for r in rows}
    print('\n== low-H onset (Rt1, xDG-N1, late xDG) ==', flush=True)
    for lab, rows in [('canon', canon), ('vbs', vbs), ('vg', vg)]:
        o = onset(rows)
        s = '  '.join(f"H{h}:Rt1={v[0]:.0f}({v[1]:.2f}x,late{v[2]:.2f})"
                      for h, v in sorted(o.items())[:3])
        print(f"  {lab:6s}: {s}", flush=True)

    print('\n== vg regression: adverse + separated/lower branch ==', flush=True)
    S.FORM[0] = 'vg'; S.EPS[0] = EPS_R; S.AC[0] = None; S.BC[0] = BC
    vg_adv = [S.row(b) for b in ADV]
    vg_low = [S.row(b, guess=g, ufrac=0.03) for b, g in LOW]
    print('== vg wall-boundedness ==', flush=True)
    S.FORM[0] = 'vg'; S.EPS[0] = EPS_R; S.BC[0] = BC
    bnd = S.boundedness(EPS_R)
    print('== vg zero-suite (Spalart parasitic) ==', flush=True)
    zs = S.zero_suite(EPS_R)
    print('== vg graze on separated/lower branch (curv branch must not fire) ==',
          flush=True)
    sep_graze = graze_vg(-0.19, BC, guess=-0.03)
    print(f"  separated (beta=-0.19 lower) graze_vg = {sep_graze[0]:.3f} "
          f"(H={sep_graze[1]:.2f})", flush=True)

    out = dict(BC=BC, BC_eq_vbs=BC_eq_vbs, threshold_table=tbl,
               dev=dict(canon_C1851=[float(np.exp(da_c)), float(np.exp(db_c))],
                        C8000=[float(np.exp(da_gr)), float(np.exp(db_gr))],
                        vbs=[float(np.exp(da_v)), float(np.exp(db_v))],
                        vg=[float(np.exp(da_g)), float(np.exp(db_g))]),
               canon=canon, green=green, vbs=vbs, vg=vg,
               vg_adverse=vg_adv, vg_lower=vg_low,
               boundedness=bnd, zero_suite=zs,
               separated_graze=dict(graze=sep_graze[0], H=sep_graze[1]))
    with open(os.path.join(OUT_DIR, 'zb_2branch_gate.json'), 'w') as f:
        json.dump(out, f, indent=1)
    figure(canon, green, vbs, vg, BC)
    print('DONE', flush=True)


def figure(canon, green, vbs, vg, BC):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def cols(rows):
        H = np.array([r['H'] for r in rows]); sl = np.array([r['s_late'] for r in rows])
        R = np.array([r['Rt1'] for r in rows]); o = np.argsort(H)
        return H[o], sl[o], R[o]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.2, 4.3))
    fig.patch.set_facecolor('white')
    Hg = np.geomspace(2.2, 2.75, 200)
    dg = np.asarray(dN_dRe_theta(Hg)); Rtc = np.asarray(Re_theta0(Hg))
    N1 = Rtc + 1.0/dg
    for rows, col, lab, mk in [(canon, '0.6', 'canon-C1851 (const C)', 'o'),
                               (green, 'C2', 'C8000 (const C)', 'D'),
                               (vbs, 'C1', 'vbs (P-blend)', 'v'),
                               (vg, 'C0', 'vg (two-branch gate)', '^')]:
        H, sl, R = cols(rows)
        axa.semilogy(H, sl, '-'+mk, color=col, lw=1.4, ms=5, label=lab)
        axb.semilogy(H, R, '-'+mk, color=col, lw=1.4, ms=5, label=lab)
    axa.semilogy(Hg, dg, 'k--', lw=1.8, label='Drela-Giles')
    axb.semilogy(Hg, Rtc, '--', color='0.5', lw=1.2, label='Drela crit $Re_{\\theta0}$')
    axb.semilogy(Hg, N1, 'k--', lw=1.8, label='DG $N=1$ station')
    for ax, yl in [(axa, r'$dN/dRe_\theta$ (late secant)'), (axb, r'onset $Re_\theta$')]:
        ax.set_xlabel('H'); ax.set_ylabel(yl); ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=7.5)
    axa.set_title('(a) rate'); axb.set_title('(b) onset')
    fig.suptitle(rf'Part VIII two-branch onset gate (vg): '
                 rf'$Re_\Omega^c=\mathrm{{softmin}}(A+B/P_I^2, A+B_c/P_{{curv}}^2)$, '
                 rf'$B_c={BC:.0f}$', fontsize=10)
    plt.tight_layout()
    fp = os.path.join(OUT_DIR, 'model_calibrate_candidate_2branchgate.png')
    plt.savefig(fp, dpi=150, facecolor='white')
    print(f'wrote {fp}', flush=True)


if __name__ == '__main__':
    main()
