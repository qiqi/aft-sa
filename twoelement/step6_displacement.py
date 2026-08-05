"""STEP 6 -- how much H is needed to erase the trailing-edge recovery?

User's test (2026-08-04): if H is at the Blasius level where the recovery
starts, how far must H rise for the DISPLACEMENT body to hold Cp at the aft
suction peak all the way to the trailing edge? Hold theta fixed -- only H (and
hence delta* = H theta) grows. If the answer is near the laminar separation
value H ~ 4, the recovery is not survivable and the geometry needs changing.
If H barely moves, the recovery is cosmetic.

Method: offset both fore surfaces outward by delta*(s) along the local normal,
with H ramping from Blasius (2.59) at the start of the recovery to H_te at the
trailing edge, then re-solve the COUPLED two-element problem. Note that the
displacement body has a blunt trailing edge of thickness 2 delta*, which by
itself removes the potential-flow stagnation singularity -- part of what real
viscosity does.

Run:  python3 step6_displacement.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import panel2e as M
import step4_coupled as S
from step1_flap_and_camber import flap_nodes, FORE_CHORD, ALPHA
from step2_questions import refit_inc

T_FORE, NPAN = 0.08, 400
H_BLASIUS, H_SEP = 2.59, 3.98          # Blasius; laminar separation limit
RE = 1.0e6                              # on the OVERALL chord (=1)
X_START = 0.95                          # recovery starts here (x/c_fore)


def displaced_fore(inc, H_te, Re=RE, ramp_from=X_START):
    """Fore element offset by delta* = H(s) theta(s), Blasius theta."""
    nd = M.place(M.airfoil_nodes(NPAN, S.M_CAM, S.P_CAM, T_FORE, modified=True,
                                 xm=S.XM, ik=S.IK, km=S.KM, le_blend=0.15),
                 FORE_CHORD, inc, 0.0, 0.0)
    n = len(nd)
    half = n//2
    out = nd.copy()
    for seg in (slice(0, half + 1), slice(half, n)):       # lower, upper
        pts = nd[seg]
        # arc length measured FROM the leading edge along each surface
        d = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        s = np.concatenate([[0.0], np.cumsum(d)])
        if seg.start == 0:                       # lower runs TE -> LE
            s = s[-1] - s
        s = np.maximum(s, 1e-6)
        theta = 0.664*np.sqrt(s/Re)              # Blasius momentum thickness
        xf = pts[:, 0]/FORE_CHORD
        f = np.clip((xf - ramp_from)/(1.0 - ramp_from), 0.0, 1.0)
        H = H_BLASIUS + (H_te - H_BLASIUS)*f
        dstar = H*theta
        # outward normal from the local tangent
        t = np.gradient(pts, axis=0)
        t /= (np.linalg.norm(t, axis=1)[:, None] + 1e-30)
        nrm = np.column_stack([-t[:, 1], t[:, 0]])
        if seg.start == 0:
            nrm = -nrm                            # lower surface points down
        out[seg] = pts + nrm*dstar[:, None]
    out[-1] = out[0] if False else out[-1]
    return out


def solve_with(H_te, inc):
    els = [displaced_fore(inc, H_te),
           M.place(M.airfoil_nodes(NPAN, 0.09, 0.40, 0.16), 0.30, -8.0,
                   0.70, 0.04)]
    P, res = M.solve_elements(els, ALPHA)
    _, up = M.surfaces(P, 0)
    x, cp = P.xc[up], P.Cp[up]
    xf = x/FORE_CHORD
    i0 = int(np.argmin(np.abs(xf - X_START)))
    m = xf >= X_START
    return dict(cp_start=float(cp[i0]), cp_te=float(cp[-1]),
                cp_max=float(cp[m].max()),
                recov=float(cp[m].max() - cp[i0]), x=xf, cp=cp, Cl=res['Cl'])


if __name__ == '__main__':
    n2 = flap_nodes()
    Pf, _ = M.solve_elements([n2], ALPHA)
    xs, zs = M.streamline_camber(Pf, 0.0, 0.0, FORE_CHORD, n=400)
    inc = refit_inc(S.M_CAM, S.P_CAM, xs, zs)

    s_te = FORE_CHORD
    th = 0.664*np.sqrt(s_te/RE)
    print('at the start of the recovery (x/c_fore = %.2f), Re_c = %.0e:' %
          (X_START, RE))
    print('  theta   = %.5f c   (Blasius)' % th)
    print('  delta*  = %.5f c   at H = %.2f' % (H_BLASIUS*th, H_BLASIUS))
    print('  fore half-thickness closes at slope ~%.3f per chord\n'
          % (T_FORE*2.3))

    print('%8s %10s %10s %10s %10s' %
          ('H_te', 'delta*/c', 'Cp_start', 'Cp_peak', 'recovery'))
    base = None
    rows = []
    for H in (2.59, 3.0, 3.5, 3.98, 5.0, 7.0, 10.0, 14.0, 20.0):
        r = solve_with(H, inc)
        rows.append((H, r))
        if base is None:
            base = r['recov']
        print('%8.2f %10.5f %10.4f %10.4f %10.4f   (%.0f%% of Blasius)'
              % (H, H*th, r['cp_start'], r['cp_max'], r['recov'],
                 100*r['recov']/base))

    # how far to halve it / kill it
    Hs = np.array([r[0] for r in rows])
    rc = np.array([r[1]['recov'] for r in rows])
    for frac, lbl in ((0.5, 'HALVE'), (0.1, 'remove 90% of')):
        tgt = frac*base
        if rc.min() <= tgt:
            Hneed = float(np.interp(-tgt, -rc[::-1], Hs[::-1]))
            print('\n  to %s the recovery: H_te = %.2f  (%s separation H=%.2f)'
                  % (lbl, Hneed, 'PAST' if Hneed > H_SEP else 'below', H_SEP))
        else:
            print('\n  to %s the recovery: NOT reachable within H_te <= %.0f'
                  % (lbl, Hs.max()))

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for H, r in rows:
        if H in (2.59, 3.98, 7.0, 14.0):
            ax[0].plot(r['x'], r['cp'], '-', lw=1.5, label='$H_{te}$=%.2f' % H)
    ax[0].set_xlim(0.85, 1.01); ax[0].invert_yaxis(); ax[0].grid(alpha=.3)
    ax[0].legend(fontsize=8); ax[0].set_xlabel('$x/c_{fore}$')
    ax[0].set_ylabel('$C_p$')
    ax[0].set_title('fore upper surface with a displacement body', fontsize=10)

    ax[1].plot(Hs, rc, 'o-', color='#1f4e9c', lw=1.6)
    ax[1].axvline(H_SEP, color='#b91c1c', ls='--', lw=1.4,
                  label='laminar separation $H$=3.98')
    ax[1].axhline(0, color='0.6', lw=.8)
    ax[1].set_xlabel('$H$ at the trailing edge')
    ax[1].set_ylabel('residual recovery $\\Delta C_p$')
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    ax[1].set_title('how much $H$ buys', fontsize=10)
    fig.tight_layout()
    fig.savefig('step6_displacement.pdf')
    fig.savefig('step6_displacement.png', dpi=125)
    print('\nwrote step6_displacement.pdf / .png')
