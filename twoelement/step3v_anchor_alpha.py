"""STEP 3v -- choose the ANCHORING incidence for the mean-line trace so the slot
gap is large enough to fine-tune in.

At the alpha = -1 anchor the fore trailing edge ends up 0.0132 from the flap's
displacement body, which is too tight to shape without spending Kulfan terms on
it. The anchor incidence is the knob: it need NOT equal the operating incidence,
it only sets the fore element's SHAPE.

Two effects push the same way as the anchor incidence rises:
  - more flap circulation, so the streamline through (0,0) sits higher;
  - stronger flap nose suction, so the Cp = -0.75 station moves UPSTREAM, which
    pulls the fore trailing edge forward as well as up.

Everything else is held fixed: the same Cp threshold sets x_end, the same CST
orders, and the same thickness coefficients from step2v, so the gap is the only
thing being compared. The flap displacement body is the OPERATING one
(alpha = -1), because that is the geometry the tuning will actually see.

Run:  python3 step3v_anchor_alpha.py [out.pdf]
"""
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import panel2e as M
import step1v_camber as S
import step2v_build as B
from step1v_kulfan import fit_cst
from flap_displacement import flap_body

CP_TARGET = -0.75
OPER_ALPHA = -1.0
ANCHORS = (-1.0, 0.0, 1.0, 2.0, 3.0)


def case_dir(a):
    return 'case_F1' if a == -1.0 else 'case_A%+05.1f' % a


def analyse(anchor, flap_nd, thk, verbose=True):
    """Trace at this anchor, set x_end by the Cp threshold, build the fore
    element with the FIXED thickness coefficients, and measure the slot."""
    rf = S.RansField(case_dir(anchor))
    xs, zs = S.trace(rf, 0.0, 0.0, 0.80, n=1600)
    u, v = rf(xs, zs)
    cp = 1.0 - (np.hypot(u, v)/S.MACH)**2
    ok = np.isfinite(cp)
    kmin = int(np.nanargmin(np.where(ok, cp, np.inf)))
    k = int(np.argmin(np.abs(cp[:kmin+1] - CP_TARGET)))
    x_end = float(xs[k])

    x2, z2 = S.trace(rf, 0.0, 0.0, x_end, n=900)
    _, _, cam, rms = fit_cst(x2, z2, x_end, n=B.N_CAM)

    nd, xf, zc, th = B.fore_nodes(thk, cam, x_end)
    te = nd[0]                           # first node is the lower TE
    gap = float(np.min(np.hypot(flap_nd[:, 0] - te[0], flap_nd[:, 1] - te[1])))
    # shortest distance from the whole fore contour to the flap body, which is
    # what actually constrains the mesh -- not just the trailing-edge point
    d = np.min(np.hypot(nd[:, None, 0] - flap_nd[None, :, 0],
                        nd[:, None, 1] - flap_nd[None, :, 1]))
    overlap = te[0] - 0.70               # +ve means the fore TE is aft of the flap LE

    P, res = M.solve_elements([nd, flap_nd], OPER_ALPHA)
    lo, up = M.surfaces(P, 0)
    xn_lo = P.xc[lo]/x_end
    aft = xn_lo >= 0.85
    cp_aft = float(P.Cp[lo][aft].min()) if aft.any() else np.nan
    cost, det = M.fore_flatness(P)
    return dict(anchor=anchor, x_end=x_end, z_te=float(te[1]), gap_te=gap,
                gap_min=float(d), overlap=overlap, cam_rms=rms,
                cp_lower_aft=cp_aft, CL=res['Cl'], flat=cost,
                rms_up=det['rms_up'], rms_lo=det['rms_lo'],
                nd=nd, xs=x2, zs=z2, cam=cam, cp=cp, xtrace=xs, ztrace=zs)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'step3v_anchor_alpha.pdf'
    d0 = np.load('step2v_build.npz')
    thk = d0['thk']
    flap_nd, finfo = flap_body(with_wake=False)
    print('operating flap displacement body: mfoil cl %.4f\n' % finfo['cl_mfoil'])

    rows = []
    for a in ANCHORS:
        try:
            rows.append(analyse(a, flap_nd, thk))
        except Exception as e:                                  # noqa: BLE001
            print('anchor %+.1f: unavailable (%s)' % (a, str(e)[:70]))

    print('%-8s %8s %8s %9s %9s %9s %9s %8s'
          % ('anchor', 'x_end', 'z_TE', 'gap(TE)', 'gap(min)', 'overlap',
             'Cp_lo_aft', 'CL'))
    for r in rows:
        print('%-8.1f %8.4f %8.4f %9.4f %9.4f %+9.4f %9.3f %8.4f'
              % (r['anchor'], r['x_end'], r['z_te'], r['gap_te'], r['gap_min'],
                 r['overlap'], r['cp_lower_aft'], r['CL']))

    base = rows[0]['gap_min']
    print('\ngap relative to the alpha=-1 anchor (target ~2x)')
    for r in rows:
        print('  anchor %+.1f  gap %.4f  = %.2fx   fore flatness %.4f '
              '(rms up %.4f, lo %.4f)'
              % (r['anchor'], r['gap_min'], r['gap_min']/base, r['flat'],
                 r['rms_up'], r['rms_lo']))

    np.savez('step3v_anchor_alpha.npz',
             anchors=np.array([r['anchor'] for r in rows]),
             gap=np.array([r['gap_min'] for r in rows]),
             x_end=np.array([r['x_end'] for r in rows]),
             cp_lower_aft=np.array([r['cp_lower_aft'] for r in rows]))

    # ------------------------------------------------------------ figure ---
    fig, ax = plt.subplots(3, 1, figsize=(9.2, 10.2),
                           gridspec_kw=dict(height_ratios=[1.3, 1.0, 1.0]))
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(rows)))
    ax[0].plot(np.append(flap_nd[:, 0], flap_nd[0, 0]),
               np.append(flap_nd[:, 1], flap_nd[0, 1]), '-',
               color='#1a8a5a', lw=1.4)
    ax[0].fill(flap_nd[:, 0], flap_nd[:, 1], color='#1a8a5a', alpha=.12)
    for r, c in zip(rows, cols):
        ax[0].plot(np.append(r['nd'][:, 0], r['nd'][0, 0]),
                   np.append(r['nd'][:, 1], r['nd'][0, 1]), '-', color=c,
                   lw=1.3, label=r'anchor $%+.0f^\circ$, gap %.4f'
                   % (r['anchor'], r['gap_min']))
    ax[0].set_aspect('equal'); ax[0].set_xlim(-0.04, 1.06)
    ax[0].legend(fontsize=7.5, loc='upper left')
    ax[0].set_title('Fore element vs anchoring incidence (thickness held fixed)',
                    fontsize=10)

    for r, c in zip(rows, cols):
        ax[1].plot(r['xtrace'], r['cp'], '-', color=c, lw=1.3,
                   label=r'anchor $%+.0f^\circ$' % r['anchor'])
        ax[1].plot(r['x_end'], CP_TARGET, 'o', color=c, ms=5)
    ax[1].axhline(CP_TARGET, color='#d97706', ls='--', lw=1.0)
    ax[1].set_xlim(0.0, 0.80); ax[1].set_ylim(0.4, -2.2)
    ax[1].set_ylabel('$C_p$ along the streamline')
    ax[1].legend(fontsize=7.5); ax[1].grid(alpha=.25, lw=.6)

    g = np.array([r['gap_min'] for r in rows])
    an = np.array([r['anchor'] for r in rows])
    ax[2].plot(an, g, 'o-', color='#1f4e9c', lw=1.6)
    ax[2].axhline(2*base, color='#c44e52', ls='--', lw=1.2,
                  label='2x the $\\alpha=-1$ gap')
    ax[2].set_xlabel('anchoring incidence, deg')
    ax[2].set_ylabel('shortest fore-to-flap distance')
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.25, lw=.6)
    fig.tight_layout()
    fig.savefig(out)
    print('\nwrote', out)
