"""STEP 5 -- how much of the trailing-edge pressure recovery is REAL?

The coupled inviscid solution shows a very rapid recovery in the last few
percent of the fore element. Three candidate causes, separated here:

  1. GENUINE inviscid physics. A closed trailing edge with a finite included
     angle carries a stagnation point, so Cp -> 1 there in potential flow.
     This is real, and no amount of panel refinement removes it.
  2. PANEL RESOLUTION. Cosine spacing puts vanishingly small panels at the TE;
     the last collocation point creeps toward the singular point as N grows.
     If the recovery keeps steepening with N without converging, that part is
     numerical.
  3. VISCOUS ERASURE. The boundary layer's displacement thickness rounds the
     trailing edge and unloads it. A pressure feature confined to a streamwise
     extent SMALLER than the local displacement thickness cannot survive.

Test 3 is the decisive one: measure the streamwise extent of the recovery and
compare it with delta* at the Reynolds numbers we would actually run.

Run:  python3 step5_te_recovery.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import panel2e as M
import step4_coupled as S
from step1_flap_and_camber import flap_nodes, FORE_CHORD, ALPHA
from step2_questions import refit_inc

T_FORE = 0.08
RE_LIST = (3e5, 5e5, 1e6, 2e6)


def fore_upper(P):
    _, up = M.surfaces(P, 0)
    return P.xc[up], P.Cp[up]


def recovery_metrics(x, cp, chord):
    """Locate the aft suction minimum and measure the recovery to the TE."""
    n = len(x)
    tail = slice(int(0.55*n), n)
    i = tail.start + int(np.argmin(cp[tail]))       # most negative Cp aft
    dCp = cp[-1] - cp[i]
    dx = x[-1] - x[i]
    return dict(x_min=float(x[i]), cp_min=float(cp[i]), cp_te=float(cp[-1]),
                dCp=float(dCp), dx=float(dx), dx_c=float(dx/chord),
                slope=float(dCp/max(dx/chord, 1e-9)))


if __name__ == '__main__':
    n2 = flap_nodes()
    Pf, _ = M.solve_elements([n2], ALPHA)
    xs, zs = M.streamline_camber(Pf, 0.0, 0.0, FORE_CHORD, n=400)
    inc = refit_inc(S.M_CAM, S.P_CAM, xs, zs)

    print('=== 1-2. panel convergence of the TE recovery ===')
    print('%6s %10s %10s %10s %10s %10s' %
          ('N', 'x_min/c', 'Cp_min', 'Cp_TE', 'dx/c', 'dCp/d(x/c)'))
    store = {}
    for N in (100, 200, 400, 800, 1600):
        els = [M.place(M.airfoil_nodes(N, S.M_CAM, S.P_CAM, T_FORE,
                                       modified=True, xm=S.XM, ik=S.IK,
                                       km=S.KM, le_blend=0.15),
                       FORE_CHORD, inc, 0.0, 0.0),
               M.place(M.airfoil_nodes(N, 0.09, 0.40, 0.16), 0.30, -8.0,
                       0.70, 0.04)]
        P, _ = M.solve_elements(els, ALPHA)
        x, cp = fore_upper(P)
        r = recovery_metrics(x, cp, FORE_CHORD)
        store[N] = (x, cp, r)
        print('%6d %10.4f %10.4f %10.4f %10.5f %10.2f'
              % (N, r['x_min']/FORE_CHORD, r['cp_min'], r['cp_te'],
                 r['dx_c'], r['slope']))
    print('  -> Cp_TE converging means the stagnation recovery is REAL;')
    print('     dx/c shrinking with N means its measured STEEPNESS is not.')

    print('\n=== 3. can the boundary layer survive it? ===')
    r = store[400][2]
    print('recovery: Cp %.3f -> %.3f over dx/c = %.4f (%.2f%% of fore chord)'
          % (r['cp_min'], r['cp_te'], r['dx_c'], 100*r['dx_c']))
    print('%10s %12s %12s %10s' % ('Re_c', 'delta*/c @TE', 'dx/delta*',
                                   'verdict'))
    xte = FORE_CHORD
    for Re in RE_LIST:
        Rex = Re*xte
        dstar = 1.72*xte/np.sqrt(Rex)          # laminar Blasius delta*
        ratio = r['dx']/dstar
        verdict = ('smeared out' if ratio < 3 else
                   'partly felt' if ratio < 10 else 'real, felt')
        print('%10.1e %12.5f %12.2f %10s' % (Re, dstar, ratio, verdict))
    print('  (delta* from the laminar Blasius estimate 1.72 x/sqrt(Re_x);')
    print('   a Cp feature narrower than ~3 delta* cannot be sustained.)')

    # ---- 4. does a thinner/cusped trailing edge soften it? ----------------
    print('\n=== 4. TE geometry lever: aft-loading parameter K_m ===')
    print('%8s %10s %10s %10s' % ('K_m', 'Cp_TE', 'dCp', 'dCp/d(x/c)'))
    for km in (0.5, 0.7, 1.0, 1.4):
        els = [M.place(M.airfoil_nodes(400, S.M_CAM, S.P_CAM, T_FORE,
                                       modified=True, xm=S.XM, ik=S.IK,
                                       km=km, le_blend=0.15),
                       FORE_CHORD, inc, 0.0, 0.0),
               M.place(M.airfoil_nodes(400, 0.09, 0.40, 0.16), 0.30, -8.0,
                       0.70, 0.04)]
        P, _ = M.solve_elements(els, ALPHA)
        rr = recovery_metrics(*fore_upper(P), FORE_CHORD)
        print('%8.1f %10.4f %10.4f %10.2f'
              % (km, rr['cp_te'], rr['dCp'], rr['slope']))

    # ------------------------------------------------------------- figure --
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for N in (100, 400, 1600):
        x, cp, _ = store[N]
        ax[0].plot(x/FORE_CHORD, cp, '-', lw=1.4, label='N=%d' % N)
    ax[0].invert_yaxis(); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    ax[0].set_xlabel('$x/c_{fore}$'); ax[0].set_ylabel('$C_p$')
    ax[0].set_title('fore upper surface, panel convergence', fontsize=10)

    x, cp, _ = store[1600]
    ax[1].plot(x/FORE_CHORD, cp, '-', color='#1f4e9c', lw=1.8, label='inviscid')
    for Re, c in zip(RE_LIST, ('#fca5a5', '#f87171', '#ef4444', '#b91c1c')):
        ds = 1.72*FORE_CHORD/np.sqrt(Re*FORE_CHORD)
        ax[1].axvspan(1.0 - 3*ds/FORE_CHORD, 1.0, color=c, alpha=0.13)
        ax[1].text(1.0 - 3*ds/FORE_CHORD, -0.05, ' Re=%.0e' % Re,
                   rotation=90, fontsize=6.5, va='bottom', ha='right')
    ax[1].set_xlim(0.85, 1.005)
    ax[1].invert_yaxis(); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    ax[1].set_xlabel('$x/c_{fore}$')
    ax[1].set_title(r'TE zoom; shaded = $3\delta^*$ the BL smears',
                    fontsize=10)
    fig.tight_layout()
    fig.savefig('step5_te_recovery.pdf')
    fig.savefig('step5_te_recovery.png', dpi=125)
    print('\nwrote step5_te_recovery.pdf / .png')
