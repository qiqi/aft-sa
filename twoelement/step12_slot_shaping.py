"""STEP 12 -- micro-adjust the fore TE to shape the slot.

The fore element's LOWER surface develops a suction spike right at the slot
entrance (Cp ~ -0.75 near x = 0.65-0.70). User's reading (2026-08-04): this is
a slot-shaping problem, and the response is NONLINEAR and ASYMMETRIC -- moving
the lower surface DOWN (toward the flap) narrows the channel and produces far
more suction than moving the upper surface up by the same amount.

Within the NACA family the thickness is SYMMETRIC about the mean line, so the
lower surface can only be moved by trading:
    thickness near the TE   -- x_m forward and lower K_m give a gentler aft
                               closure, pulling the lower surface UP (away)
    mean line height        -- m and p reshape the reflex; incidence lifts the
                               whole TE
So the micro-adjustment is a coupled (m, p, inc, t, x_m, K_m) search around the
streamline-fitted starting point, with the spike explicitly in the objective.

Run:  python3 step12_slot_shaping.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import panel2e as M
from step9_extent import flap_at, streamline_and_cp, fit4

ALPHA, CHORD, NPAN = -1.0, 0.700, 300
FLAP = dict(m=0.09, p=0.40, t=0.16, chord=0.30, inc=-8.0, le=(0.70, 0.04))

# start = the streamline fit + the shape-matched thickness
P0 = dict(m=-0.0409, p=0.743, inc=7.85, t=0.070, xm=0.50, km=0.70)
LO = dict(m=-0.065, p=0.60, inc=6.6, t=0.045, xm=0.33, km=0.40)
HI = dict(m=-0.020, p=0.88, inc=9.2, t=0.105, xm=0.60, km=1.40)
KEYS = ('m', 'p', 'inc', 't', 'xm', 'km')


def build(p):
    n1 = M.place(M.airfoil_nodes(NPAN, p['m'], p['p'], p['t'], modified=True,
                                 xm=p['xm'], ik=6.0, km=p['km'], le_blend=0.15),
                 CHORD, p['inc'], 0.0, 0.0)
    n2 = M.place(M.airfoil_nodes(NPAN, FLAP['m'], FLAP['p'], FLAP['t']),
                 FLAP['chord'], FLAP['inc'], *FLAP['le'])
    return [n1, n2]


def analyse(p):
    els = build(p)
    P, res = M.solve_elements(els, ALPHA)
    lo_i, up_i = M.surfaces(P, 0)
    out = {}
    for tag, idx in (('up', up_i), ('lo', lo_i)):
        s = P.xc[idx]; s = (s - s.min())/(s.max() - s.min() + 1e-30)
        m = (s >= 0.02) & (s <= 0.99)
        cp, ss = P.Cp[idx][m], s[m]
        g = np.gradient(cp, ss)
        mid = (ss > 0.15) & (ss < 0.75)
        aft = ss >= 0.82
        out[tag] = dict(rng=float(cp.max() - cp.min()),
                        adv=float(np.clip(g, 0, None).mean()),
                        spike=float(max(0.0, (-cp[aft]).max()
                                        - (-cp[mid]).mean())),
                        cp=cp, s=ss)
    gap = float(np.min(np.hypot(els[0][:, 0][:, None] - els[1][:, 0][None, :],
                                els[0][:, 1][:, None] - els[1][:, 1][None, :])))
    out['gap'], out['Cl'], out['P'], out['els'] = gap, res['Cl'], P, els
    return out


def cost(p, w_spike=2.5, w_adv=2.0):
    try:
        a = analyse(p)
    except Exception:
        return 1e6
    J = (a['up']['rng'] + a['lo']['rng']
         + w_spike*(a['up']['spike'] + 1.6*a['lo']['spike'])   # lower matters more
         + w_adv*(a['up']['adv'] + a['lo']['adv']))
    J += 8.0*max(0.0, 0.020 - a['gap'])        # keep a physical slot
    return J


def vec(p):
    return np.array([p[k] for k in KEYS])


def unvec(x):
    return {k: float(np.clip(v, LO[k], HI[k])) for k, v in zip(KEYS, x)}


if __name__ == '__main__':
    a0 = analyse(P0)
    print('START  m=%+.4f p=%.3f inc=%+.2f t=%.3f xm=%.2f km=%.2f'
          % tuple(P0[k] for k in KEYS))
    print('  upper rng %.3f adv %.3f spike %.3f | lower rng %.3f adv %.3f '
          'spike %.3f | gap %.4f  J=%.4f'
          % (a0['up']['rng'], a0['up']['adv'], a0['up']['spike'],
             a0['lo']['rng'], a0['lo']['adv'], a0['lo']['spike'],
             a0['gap'], cost(P0)))

    rng = np.random.default_rng(3)
    lo_v = np.array([LO[k] for k in KEYS]); hi_v = np.array([HI[k] for k in KEYS])
    best = (cost(P0), vec(P0))
    for i in range(7):
        x0 = vec(P0) if i == 0 else lo_v + (hi_v - lo_v)*rng.random(len(KEYS))
        r = minimize(lambda x: cost(unvec(x)), x0, method='Nelder-Mead',
                     options=dict(maxfev=1200, xatol=1e-4, fatol=1e-6))
        if r.fun < best[0]:
            best = (r.fun, r.x)
        print('  start %d: J=%.4f%s' % (i, r.fun, '  <-- best' if r.fun == best[0] else ''))

    pb = unvec(best[1]); ab = analyse(pb)
    print('\nTUNED  m=%+.4f p=%.3f inc=%+.2f t=%.3f xm=%.2f km=%.2f'
          % tuple(pb[k] for k in KEYS))
    print('  upper rng %.3f adv %.3f spike %.3f | lower rng %.3f adv %.3f '
          'spike %.3f | gap %.4f  J=%.4f'
          % (ab['up']['rng'], ab['up']['adv'], ab['up']['spike'],
             ab['lo']['rng'], ab['lo']['adv'], ab['lo']['spike'],
             ab['gap'], best[0]))
    print('  Cl %.4f -> %.4f' % (a0['Cl'], ab['Cl']))

    import json
    json.dump(dict(alpha=ALPHA, chord=CHORD, fore=pb, flap=FLAP),
              open('twoelement_tuned_slot.json', 'w'), indent=2)

    fig, ax = plt.subplots(2, 1, figsize=(9.0, 7.4))
    for a, lab, ls in ((a0, 'start', '--'), (ab, 'slot-shaped', '-')):
        for tag, c in (('up', '#1f4e9c'), ('lo', '#b03060')):
            ax[0].plot(a[tag]['s'], a[tag]['cp'], ls, color=c, lw=1.6,
                       alpha=1.0 if ls == '-' else .55,
                       label='%s %s' % (lab, tag))
    ax[0].invert_yaxis(); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7.5, ncol=2)
    ax[0].set_xlabel('$s$ along the fore element'); ax[0].set_ylabel('$C_p$')
    ax[0].set_title('fore-element $C_p$: the lower-surface slot spike',
                    fontsize=10)
    for a, lab, ls in ((a0, 'start', '--'), (ab, 'slot-shaped', '-')):
        for nd, c in zip(a['els'], ('#1f4e9c', '#1a8a5a')):
            ax[1].plot(nd[:, 0], nd[:, 1], ls, color=c, lw=1.5,
                       alpha=1.0 if ls == '-' else .5)
    ax[1].set_aspect('equal'); ax[1].grid(alpha=.3)
    ax[1].set_xlim(0.45, 1.02); ax[1].set_xlabel('x')
    ax[1].set_title('slot region (dashed = start, solid = shaped)', fontsize=10)
    fig.tight_layout()
    fig.savefig('step12_slot_shaping.pdf')
    fig.savefig('step12_slot_shaping.png', dpi=125)
    print('wrote step12_slot_shaping.pdf / .png')
