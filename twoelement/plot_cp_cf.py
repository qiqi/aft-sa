"""Surface Cp and skin friction for the two-element ladder, with transition,
laminar separation and reattachment located from the Cf signature.

Ordering. The surface points are matched onto the mesher's own contour
(contours.txt) by nearest node, which gives an exact arc coordinate and a clean
upper/lower split. The previous angular sort about the centroid is not a valid
arc ordering for a thin, highly cambered element at incidence -- upper and lower
points interleave near the trailing edge.

Sign of Cf. The solver writes |Cf| only. The sign comes from the mid-plane
volume velocity sampled one near-wall offset out along the wall normal, dotted
with the downstream tangent, so reverse flow inside a laminar separation bubble
is recovered.

Markers, using s = arc from the leading edge (a fraction of the element chord):
  SEP   signed Cf goes + -> -
  REATT signed Cf goes - -> +
  TR    steepest rise of log|Cf| downstream of the last Cf minimum, i.e. the
        turbulent reattachment ramp; for an attached case it is the classic
        laminar-to-turbulent jump.

Run:  python3 plot_cp_cf.py out.pdf case_L0_idxfix case_L1_idxfix
"""
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.spatial import cKDTree

import plot_solution_mesh as PM
from measure_l1_spacing import read_contours, surface_frame

ELEMS = ('fore', 'flap')


def midplane_surface(path):
    """Wall points in the span mid-plane, with Cp, |Cf|, yPlus."""
    g, pts, arr = PM.read_vtu(path)
    nun = [len(np.unique(np.round(pts[:, i], 9))) for i in range(3)]
    span = int(np.argmin(nun))
    keep = np.isclose(pts[:, span], np.unique(pts[:, span])[0], atol=1e-9)
    P = np.delete(pts[keep], span, axis=1)
    cf = arr['Cf'][keep]
    if cf.ndim > 1:
        cf = np.linalg.norm(cf, axis=1)
    return P, arr['Cp'][keep], cf, arr['yPlus'][keep]


class Interp:
    """Mid-plane velocity interpolant for the in-plane components."""

    def __init__(self, case):
        g, pts, arr = PM.read_vtu('%s/volume_proc0.vtu' % case)
        idx, tris, P2 = PM.midplane_tris(g, pts)
        vel = arr['velocity'][idx]
        T = mtri.Triangulation(P2[:, 0], P2[:, 1], tris)
        self.fu = mtri.LinearTriInterpolator(T, vel[:, 0])
        self.fv = mtri.LinearTriInterpolator(T, vel[:, 2])

    def __call__(self, x, z):
        u, v = self.fu(x, z), self.fv(x, z)
        return (np.ma.filled(u, np.nan).astype(float),
                np.ma.filled(v, np.nan).astype(float))


def order_on_contour(P, cont, ile, upper_is_second):
    """Map surface points onto contour nodes -> arc index, side mask."""
    j = cKDTree(cont).query(P, k=1)[1]
    second = j >= ile
    upper = second if upper_is_second else ~second
    return j, upper


def signed_cf(P, cf, cont, j, ile, interp, offset):
    """Attach a sign to |Cf| from the near-wall tangential velocity.

    The tangent must point DOWNSTREAM, i.e. away from the leading edge on each
    side. The contour is ordered lower TE -> LE -> upper TE, so its own
    direction is downstream only on the upper surface; on the lower surface it
    runs upstream and has to be flipped, or every attached station comes out
    negative.
    """
    n = len(cont)
    tang = cont[np.minimum(j + 1, n - 1)] - cont[np.maximum(j - 1, 0)]
    tang /= np.maximum(np.linalg.norm(tang, axis=1), 1e-30)[:, None]
    tang *= np.where(j >= ile, 1.0, -1.0)[:, None]
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
    # outward normal: the one that points away from the element centroid
    c = cont.mean(axis=0)
    flip = ((P - c)*nrm).sum(1) < 0
    nrm[flip] *= -1.0
    q = P + offset*nrm
    u, v = interp(q[:, 0], q[:, 1])
    dot = u*tang[:, 0] + v*tang[:, 1]
    # a vanishing tangential velocity carries no sign information
    bad = ~np.isfinite(dot) | (np.abs(dot) < 1e-6)
    sgn = np.where(bad, 1.0, np.sign(dot))
    sgn[sgn == 0] = 1.0
    return cf*sgn, int(bad.sum())


RE_UNIT = 1.0e6      # Reynolds number per mesh length unit (total chord ~ 1)


def cf_refs(s_phys):
    """Local flat-plate references at the same running Reynolds number:
    Blasius laminar and a Prandtl-Schlichting-style turbulent correlation."""
    re_s = np.maximum(RE_UNIT*s_phys, 1.0)
    return 0.664/np.sqrt(re_s), 0.0576/re_s**0.2


def crossings(s, f, lo=0.03, hi=0.99):
    """Sign changes of f(s) inside [lo, hi], as (s_cross, was_positive)."""
    out = []
    for i in range(len(f) - 1):
        if not (lo <= s[i] <= hi):
            continue
        if f[i] == 0 or not np.isfinite(f[i]) or not np.isfinite(f[i+1]):
            continue
        if f[i]*f[i+1] < 0:
            t = f[i]/(f[i] - f[i+1])
            out.append((s[i] + t*(s[i+1] - s[i]), f[i] > 0))
    return out


def transition_point(s, cf, s_phys, lo=0.10, hi=0.95):
    """First station where |Cf| has climbed to the geometric mean of the
    laminar and turbulent references and stays above it -- a threshold that is
    Reynolds-number aware rather than a fixed Cf level. The search starts at
    s/c = 0.10: nearer the leading edge the flow is still accelerating out of
    the stagnation point and a flat-plate reference means nothing there."""
    lam, turb = cf_refs(s_phys)
    thr = np.sqrt(lam*turb)
    m = (s >= lo) & (s <= hi) & np.isfinite(cf)
    if m.sum() < 12:
        return None, thr
    ss, above = s[m], np.abs(cf[m]) > thr[m]
    # require the excursion to persist over the remaining 10 stations
    for i in range(len(ss) - 10):
        if above[i] and above[i:i+10].mean() > 0.8:
            return float(ss[i]), thr
    return None, thr


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'cp_cf_ladder.pdf'
    cases = sys.argv[2:] or ['case_L0_idxfix', 'case_L1_idxfix']
    cols = {'case_L0_idxfix': '#c44e52', 'case_L1_idxfix': '#1f4e9c',
            'case_L2_idxfix': '#2ca02c'}
    lbl = {c: c.replace('case_', '').replace('_idxfix', '') for c in cases}

    # geometry / arc frames from the finest case's contour
    hdr, pts, curves = read_contours('%s/contours_L1.txt' % cases[-1])
    walls = [n for w, n in curves if w]
    frames = {}
    for k, nm in enumerate(ELEMS):
        p, seg, s, ile, u2 = surface_frame(pts, walls[k])
        frames[nm] = (p, seg, s, ile, u2)

    chord = {'fore': 0.7039, 'flap': 0.3}
    xle = {'fore': p[0], 'flap': p[0]}
    data = {}
    report = []
    for case in cases:
        interp = Interp(case)
        # a near-wall sampling offset that is a few first-cell heights
        off = {'case_L0_idxfix': 6e-4, 'case_L1_idxfix': 3e-4,
               'case_L2_idxfix': 2e-4}.get(case, 3e-4)
        for nm in ELEMS:
            cont, seg, sarc, ile, u2 = frames[nm]
            P, cp, cf, yp = midplane_surface(
                '%s/surface_fluid_%s_proc0.vtu' % (case, nm))
            j, upper = order_on_contour(P, cont, ile, u2)
            cfs, nbad = signed_cf(P, cf, cont, j, ile, interp, off)
            # arc from the LE, positive downstream on each side, in element chords
            sfromle = (sarc[j] - sarc[ile])/chord[nm]
            for side, mask in (('upper', upper), ('lower', ~upper)):
                o = np.argsort(np.abs(sfromle[mask]))
                key = (case, nm, side)
                ss = np.abs(sfromle[mask])[o]
                sphys = ss*chord[nm]
                data[key] = dict(s=ss, sphys=sphys, x=P[mask][:, 0][o],
                                 z=P[mask][:, 1][o], cp=cp[mask][o],
                                 cf=cfs[mask][o], yp=yp[mask][o])
                cr = crossings(ss, data[key]['cf'])
                tr, thr = transition_point(ss, data[key]['cf'], sphys)
                data[key]['thr'] = thr
                report.append((lbl[case], nm, side,
                               [(round(a, 4), b) for a, b in cr],
                               None if tr is None else round(tr, 4),
                               float(np.nanmax(data[key]['yp'])), nbad))

    # ---------------------------------------------------------------- print
    print('%-4s %-5s %-6s %-40s %8s %7s %5s'
          % ('lvl', 'elem', 'side', 'Cf sign changes (s/c from LE)',
             'TR s/c', 'y+ max', 'nosgn'))
    for lv, nm, side, cr, tr, yp, nbad in report:
        sep = ', '.join('%s@%.3f' % ('SEP' if g else 'REATT', a) for a, g in cr)
        print('%-4s %-5s %-6s %-40s %8s %7.2f %5d'
              % (lv, nm, side, sep[:40] or '-- attached --',
                 '%.3f' % tr if tr else 'laminar', yp, nbad))

    # ---------------------------------------------------------------- plot
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 11.0),
                             gridspec_kw={'height_ratios': [1.0, 1.5, 1.5]})
    for k, nm in enumerate(ELEMS):
        cont = frames[nm][0]
        ax = axes[0, k]
        ax.plot(np.append(cont[:, 0], cont[0, 0]),
                np.append(cont[:, 1], cont[0, 1]), '-k', lw=1.0)
        for kk, nn in enumerate(walls):
            if kk != k:
                q = pts[nn]
                ax.plot(q[:, 0], q[:, 1], '-', color='0.75', lw=0.8)
        ax.set_aspect('equal'); ax.set_yticks([])
        ax.set_title('%s element' % nm)
        ax.set_xlim(-0.06, 1.06)

        for row, fld, ylab in ((1, 'cp', '$C_p$'), (2, 'cf', '$C_f$ (signed)')):
            ax = axes[row, k]
            for case in cases:
                for side, ls in (('upper', '-'), ('lower', '--')):
                    d = data[(case, nm, side)]
                    ax.plot(d['s'], d[fld], ls, color=cols.get(case, 'k'),
                            lw=1.3 if side == 'upper' else 1.0,
                            alpha=1.0 if side == 'upper' else 0.7,
                            label='%s %s' % (lbl[case], side))
            ax.set_xlabel('$s/c$ from LE (element chord)')
            ax.set_ylabel(ylab)
            ax.grid(alpha=0.25, lw=0.6)
            ax.set_xlim(0, 1.02)
            if fld == 'cp':
                ax.invert_yaxis()
                ax.axhline(0, color='0.6', lw=0.7, zorder=0)
                ax.legend(fontsize=7, ncol=2, loc='lower right')
            else:
                ax.axhline(0, color='0.4', lw=0.9, zorder=0)
                ax.set_yscale('symlog', linthresh=1e-4)
                d = data[(cases[-1], nm, 'upper')]
                lam, turb = cf_refs(d['sphys'])
                ax.plot(d['s'], lam, ':', color='#666', lw=1.1,
                        label=r'Blasius $0.664/\sqrt{Re_s}$')
                ax.plot(d['s'], turb, ':', color='#000', lw=1.1,
                        label=r'turbulent $0.0576/Re_s^{0.2}$')
                ax.plot(d['s'], d['thr'], '-', color='#e08214', lw=0.8,
                        alpha=0.6, label='transition threshold')
                for lv, en, side, cr, tr, yp, nbad in report:
                    if en != nm:
                        continue
                    for a, going in cr:
                        ax.axvline(a, color='#888', ls=':', lw=0.8)
                    if tr:
                        ax.axvline(tr, color='#e08214', ls='-.', lw=1.1)
                ax.legend(fontsize=6.5, ncol=2, loc='lower right')
                ax.text(0.01, 0.97, 'grey dotted: $C_f$ sign change (SEP/REATT)'
                        '   orange dash-dot: transition',
                        transform=ax.transAxes, fontsize=7.5, va='top',
                        color='#555')

    fig.suptitle('Two-element SA-AI, Re=1e6, M=0.1, '
                 r'$\alpha=-1^\circ$: surface $C_p$ and $C_f$', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(out)
    print('wrote', out)


if __name__ == '__main__':
    main()
