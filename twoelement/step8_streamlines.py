"""STEP 8 -- streamlines of the coupled inviscid solution, and a direct check
of the fore-element camber sign.

Two panels:
  (a) streamlines through the two-element configuration, seeded across the
      inflow, with the stagnation streamline of each element highlighted.
  (b) the fore element's MEAN LINE against the flap-alone streamline it was
      fitted to, plus the chord line, so the sign of the camber is visible
      rather than inferred from the NACA digits.

The fitted mean line is NACA m = -3.5% at p = 70% mounted at +8.6 deg. The
camber is NEGATIVE (reflex) while the mean line as a whole RISES, because the
incidence dominates. Panel (b) separates the two.

Run:  python3 step8_streamlines.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import panel2e as M
import step4_coupled as S
from step1_flap_and_camber import flap_nodes, FORE_CHORD, ALPHA
from step2_questions import refit_inc, line_from

T_FORE = 0.08


def trace(P, x0, z0, x_end, n=900, back=False):
    xs = np.linspace(x0, x_end, n)
    h = xs[1] - xs[0]
    zs = np.empty(n); zs[0] = z0; z = z0
    for i in range(n - 1):
        def sl(xx, zz):
            Vx, Vz = M.field_velocity(xx, zz, P)
            return float(Vz[0]/(Vx[0] + 1e-30))
        k1 = sl(xs[i], z); k2 = sl(xs[i] + .5*h, z + .5*h*k1)
        k3 = sl(xs[i] + .5*h, z + .5*h*k2); k4 = sl(xs[i] + h, z + h*k3)
        z += (h/6.)*(k1 + 2*k2 + 2*k3 + k4)
        zs[i + 1] = z
        if abs(z) > 1.5:
            return xs[:i+2], zs[:i+2]
    return xs, zs


if __name__ == '__main__':
    n2f = flap_nodes()
    Pf, _ = M.solve_elements([n2f], ALPHA)
    xsl, zsl = M.streamline_camber(Pf, 0.0, 0.0, FORE_CHORD, n=400)
    inc = refit_inc(S.M_CAM, S.P_CAM, xsl, zsl)
    els = S.build(T_FORE, inc)
    P, res = M.solve_elements(els, ALPHA)
    print('coupled: Cl=%.4f (fore %.4f, flap %.4f)' %
          (res['Cl'], res['Cl1'], res['Cl2']))
    print('fore mean line: NACA m=%+.3f  p=%.2f  mounted %+.2f deg'
          % (S.M_CAM, S.P_CAM, inc))

    # camber decomposition
    xq = np.linspace(1e-6, FORE_CHORD, 300)
    zfit = line_from(S.M_CAM, S.P_CAM, inc, xq)
    t = np.radians(inc)
    chord = xq*np.tan(t)                        # the chord line alone
    yc_only, _ = M.naca_camber(xq/FORE_CHORD, S.M_CAM, S.P_CAM)
    print('camber alone: max |yc| = %.5f c, sign %s (NEGATIVE = reflex)'
          % (np.abs(yc_only).max()*FORE_CHORD,
             '+' if yc_only[np.argmax(np.abs(yc_only))] > 0 else '-'))
    print('mean line rises %.4f -> %.4f ; streamline rises %.4f -> %.4f'
          % (zfit[0], zfit[-1], zsl[0], zsl[-1]))
    print('rms(mean line - streamline) = %.5f c'
          % np.sqrt(np.mean((np.interp(xq, xsl, zsl) - zfit)**2)))

    fig, ax = plt.subplots(2, 1, figsize=(9.6, 8.4),
                           gridspec_kw=dict(height_ratios=[1.35, 1.0]))
    # ---- (a) streamlines --------------------------------------------------
    for nd, c in zip(els, ('#1f4e9c', '#1a8a5a')):
        ax[0].fill(nd[:, 0], nd[:, 1], color=c, alpha=.20, zorder=3)
        ax[0].plot(nd[:, 0], nd[:, 1], '-', color=c, lw=1.6, zorder=4)
    for z0 in np.linspace(-0.28, 0.34, 33):
        x, z = trace(P, -0.45, z0, 1.55)
        ax[0].plot(x, z, '-', color='0.55', lw=0.65, zorder=1)
    # stagnation streamlines: seed just off each nose
    for nd, c in zip(els, ('#b03060', '#b45309')):
        le = nd[np.argmin(nd[:, 0])]
        for dz in (-2e-4, 2e-4):
            x, z = trace(P, le[0] - 0.30, le[1] + dz, le[0] - 1e-3, n=500)
            ax[0].plot(x, z, '-', color=c, lw=1.7, zorder=5)
    ax[0].set_xlim(-0.45, 1.55); ax[0].set_ylim(-0.32, 0.40)
    ax[0].set_aspect('equal'); ax[0].grid(alpha=.25)
    ax[0].set_title('(a) coupled inviscid streamlines, $\\alpha$=%.0f$^\\circ$; '
                    'red/amber = stagnation streamlines' % ALPHA, fontsize=10)

    # ---- (b) camber decomposition ----------------------------------------
    ax[1].plot(xsl, zsl, '-', color='#b03060', lw=2.4,
               label='flap-alone streamline through the fore LE')
    ax[1].plot(xq, zfit, '--', color='#1f4e9c', lw=1.8,
               label='fitted mean line (NACA m=%+.3f, p=%.2f @ %+.1f$^\\circ$)'
                     % (S.M_CAM, S.P_CAM, inc))
    ax[1].plot(xq, chord, ':', color='0.45', lw=1.5,
               label='chord line alone (+%.1f$^\\circ$ incidence)' % inc)
    ax[1].plot(xq, yc_only*FORE_CHORD, '-', color='#d97706', lw=1.6,
               label='camber alone (NEGATIVE / reflex)')
    ax[1].axhline(0, color='0.7', lw=.8)
    ax[1].set_aspect('equal'); ax[1].grid(alpha=.3)
    ax[1].legend(fontsize=8, loc='upper left')
    ax[1].set_xlabel('x'); ax[1].set_ylabel('z')
    ax[1].set_title('(b) mean line = chord line (+8.6$^\\circ$) + NEGATIVE camber',
                    fontsize=10)
    fig.tight_layout()
    fig.savefig('step8_streamlines.pdf')
    fig.savefig('step8_streamlines.png', dpi=125)
    print('wrote step8_streamlines.pdf / .png')
