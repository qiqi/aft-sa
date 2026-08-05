"""STEP 4 -- COUPLED inviscid solve of the settled two-element geometry.

No thin-airfoil superposition, no fitting: build both elements and solve them
together, sweeping the fore-element thickness.

Geometry settled in steps 1-3:
    flap       NACA 9416, chord 0.30, LE (0.70, 0.04), inc -8 deg
    fore mean  NACA 4-digit, m = -3.5% at p = 70%, mounted at the incidence
               that best fits the flap-alone streamline through its LE
    fore thick NACA 4-digit MODIFIED, x_m = 0.50, K_m = 0.7  (the modified-
               4-digit emulation of a 65-series laminar thickness form)

The thin-airfoil estimates disagreed with each other because they targeted
different windows: cancelling the FULL trailing-edge acceleration wants
t/c ~ 0.15, cancelling it over 5-95% of chord wants t/c ~ 0.08. The coupled
solve settles it.

Run:  python3 step4_coupled.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import panel2e as M
from step1_flap_and_camber import (ALPHA, FLAP, FLAP_CHORD, FLAP_LE, FLAP_INC,
                                   FORE_CHORD, flap_nodes)
from step2_questions import refit_inc

M_CAM, P_CAM = -0.035, 0.70
XM, IK, KM = 0.50, 6.0, 0.7
NPAN = 200


def build(t, inc, alpha=ALPHA):
    n2 = M.place(M.airfoil_nodes(NPAN, FLAP['m'], FLAP['p'], FLAP['t']),
                 FLAP_CHORD, FLAP_INC, *FLAP_LE)
    n1 = M.place(M.airfoil_nodes(NPAN, M_CAM, P_CAM, t, modified=True,
                                 xm=XM, ik=IK, km=KM, le_blend=0.15),
                 FORE_CHORD, inc, 0.0, 0.0)
    return [n1, n2]


def metrics(P, k=0, lo_f=0.02, hi_f=0.98):
    """Per-surface Cp range and worst LOCAL adverse gradient."""
    out = {}
    lo, up = M.surfaces(P, k)
    for tag, idx in (('up', up), ('lo', lo)):
        s = P.xc[idx]
        s = (s - s.min())/(s.max() - s.min() + 1e-30)
        m = (s >= lo_f) & (s <= hi_f)
        cp = P.Cp[idx][m]
        g = np.gradient(cp, s[m])
        out[tag] = dict(rng=float(cp.max() - cp.min()),
                        adv_max=float(np.clip(g, 0, None).max()),
                        adv_mean=float(np.clip(g, 0, None).mean()),
                        cp0=float(cp[0]), cp1=float(cp[-1]))
    return out


if __name__ == '__main__':
    n2 = flap_nodes()
    Pf, _ = M.solve_elements([n2], ALPHA)
    xs, zs = M.streamline_camber(Pf, 0.0, 0.0, FORE_CHORD, n=400)
    inc = refit_inc(M_CAM, P_CAM, xs, zs)
    print('fore mean line: NACA m=%+.3f p=%.2f mounted at %+.2f deg' %
          (M_CAM, P_CAM, inc))
    print('fore thickness: 4-digit modified, x_m=%.2f I=%.0f K_m=%.1f\n'
          % (XM, IK, KM))

    print('%6s | %28s | %28s | %8s' %
          ('t/c', 'UPPER  rng  advmax  advmean', 'LOWER  rng  advmax  advmean',
           'Cl'))
    print('-'*84)
    best, rows = None, []
    for t in (0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12):
        els = build(t, inc)
        P, res = M.solve_elements(els, ALPHA)
        mt = metrics(P)
        score = max(mt['up']['adv_max'], mt['lo']['adv_max'])
        rows.append((t, mt, res, score))
        print('%6.3f | %8.3f %8.3f %8.3f     | %8.3f %8.3f %8.3f     | %8.3f'
              % (t, mt['up']['rng'], mt['up']['adv_max'], mt['up']['adv_mean'],
                 mt['lo']['rng'], mt['lo']['adv_max'], mt['lo']['adv_mean'],
                 res['Cl']))
        if best is None or score < best[3]:
            best = (t, mt, res, score)

    t_best = best[0]
    print('\nlowest worst-surface adverse gradient at t/c = %.3f' % t_best)

    els = build(t_best, inc)
    P, res = M.solve_elements(els, ALPHA)
    cfg = M.Config(alpha=ALPHA, n_panels=NPAN, fore_camber='4digit',
                   fore_m=M_CAM, fore_p=P_CAM, fore_t=t_best, fore_xm=XM,
                   fore_ik=IK, fore_km=KM, fore_chord=FORE_CHORD,
                   fore_inc=inc, aft_m=FLAP['m'], aft_p=FLAP['p'],
                   aft_t=FLAP['t'], aft_chord=FLAP_CHORD, aft_inc=FLAP_INC,
                   aft_x_le=FLAP_LE[0], aft_z_le=FLAP_LE[1])
    info = dict(res)
    mt = metrics(P)
    info.update(rms_up=mt['up']['rng'], rms_lo=mt['lo']['rng'],
                adv_up=mt['up']['adv_mean'], adv_lo=mt['lo']['adv_mean'],
                dCp=0.0, spike=0.0, peak=float((-P.Cp).max()),
                share=res['Cl1']/res['Cl'], gap=0.0, overlap=0.0)
    M.plot_xfoil(cfg, P, info, 'step4_coupled_cp.pdf',
                 title='coupled inviscid, t/c=%.3f, $\\alpha$=%.0f$^\\circ$'
                       % (t_best, ALPHA))
    import json
    json.dump(M.asdict(cfg), open('twoelement_final.json', 'w'), indent=2)
    print('wrote twoelement_final.json')

    # thickness sweep figure
    fig, ax = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True)
    for t, mt_, res_, _ in rows:
        if t in (0.06, 0.08, 0.10, 0.12):
            Pi, _ = M.solve_elements(build(t, inc), ALPHA)
            lo, up = M.surfaces(Pi, 0)
            ax[0].plot(Pi.xc[up], Pi.Cp[up], '-', lw=1.4, label='t/c=%.2f' % t)
            ax[1].plot(Pi.xc[lo], Pi.Cp[lo], '-', lw=1.4, label='t/c=%.2f' % t)
    for a, nm in ((ax[0], 'fore UPPER'), (ax[1], 'fore LOWER')):
        a.invert_yaxis(); a.grid(alpha=.3); a.legend(fontsize=8)
        a.set_ylabel('$C_p$'); a.set_title(nm + ' surface', fontsize=10)
    ax[1].set_xlabel('x')
    fig.tight_layout(); fig.savefig('step4_sweep.pdf')
    fig.savefig('step4_sweep.png', dpi=125)
    print('wrote step4_coupled_cp.pdf, step4_sweep.pdf/.png')
