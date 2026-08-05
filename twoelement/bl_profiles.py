"""Boundary-layer integral parameters along both elements, extracted from the
mid-plane volume solution, to locate transition independently of Cf.

Why not just look at Cf. The surface output carries |Cf| only, and a flat-plate
Cf reference is meaningless over the flap's strongly accelerated forward half --
a thin laminar layer under a favourable gradient can sit above the Blasius line
without being turbulent. The shape factor does not have that ambiguity:
H = 2.59 is Blasius, H ~ 1.4-1.6 is a turbulent layer, and H climbing past ~3.5
means the layer is separating.

Edge detection. On the flap the fore element's wake rides ABOVE the flap's own
boundary layer, so the wall-normal tangential-velocity profile is not monotone:
it rises through the layer, peaks, dips through the wake, and recovers. The edge
is taken at the FIRST interior maximum of u_t, which is the boundary-layer edge
proper and excludes the wake deficit. Stations where no such maximum is found
inside the search height are reported as unresolved rather than guessed.

Run:  python3 bl_profiles.py out.pdf case_L0_idxfix case_L1_idxfix
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
CHORD = {'fore': 0.7039, 'flap': 0.3}
RE_UNIT = 1.0e6
MACH = 0.10                 # velocities are non-dimensionalised by c_inf
H_BLASIUS = 2.59
H_TURB = 1.45


def profile_grid(ymax, n=260, y0=2.0e-6):
    """Geometric wall-normal stations."""
    r = (ymax/y0)**(1.0/(n - 1))
    return y0*r**np.arange(n)


def integrals(y, ut):
    """delta*, theta, H, u_e, delta from a tangential-velocity profile, with the
    edge at the first interior maximum of u_t."""
    good = np.isfinite(ut)
    if good.sum() < 30:
        return None
    y, ut = y[good], ut[good]
    # first interior maximum: last index of a monotone-ish rise
    k = int(np.argmax(ut))
    for i in range(2, len(ut) - 2):
        if ut[i] >= ut[i-1] and ut[i] > ut[i+1] and ut[i] > 0.3*ut.max():
            k = i
            break
    if k < 8:
        return None
    ue = ut[k]
    if ue <= 1e-6:
        return None
    yy, f = y[:k+1], np.clip(ut[:k+1]/ue, -2.0, 1.0)
    yy = np.concatenate([[0.0], yy])
    f = np.concatenate([[0.0], f])
    dstar = np.trapezoid(1.0 - f, yy)
    theta = np.trapezoid(f*(1.0 - f), yy)
    if theta <= 0:
        return None
    return dict(dstar=dstar, theta=theta, H=dstar/theta, ue=ue, delta=y[k],
                re_theta=RE_UNIT*(ue/MACH)*theta)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'bl_profiles.pdf'
    cases = sys.argv[2:] or ['case_L0_idxfix', 'case_L1_idxfix']
    cols = {'case_L0_idxfix': '#c44e52', 'case_L1_idxfix': '#1f4e9c',
            'case_L2_idxfix': '#2ca02c'}
    lbl = {c: c.replace('case_', '').replace('_idxfix', '') for c in cases}

    hdr, pts, curves = read_contours('%s/contours_L1.txt' % cases[-1])
    walls = [n for w, n in curves if w]
    frames = {}
    for k, nm in enumerate(ELEMS):
        p, seg, sarc, ile, u2 = surface_frame(pts, walls[k])
        frames[nm] = (p, seg, sarc, ile, u2)

    res = {}
    for case in cases:
        g, vpts, arr = PM.read_vtu('%s/volume_proc0.vtu' % case)
        idx, tris, P2 = PM.midplane_tris(g, vpts)
        vel = arr['velocity'][idx]
        T = mtri.Triangulation(P2[:, 0], P2[:, 1], tris)
        fu = mtri.LinearTriInterpolator(T, vel[:, 0])
        fv = mtri.LinearTriInterpolator(T, vel[:, 2])
        print('%s: mid-plane %d nodes' % (lbl[case], len(P2)))

        for nm in ELEMS:
            cont, seg, sarc, ile, u2 = frames[nm]
            n = len(cont)
            # tangent pointing downstream on each side, outward normal
            tang = cont[np.minimum(np.arange(n) + 1, n - 1)] \
                - cont[np.maximum(np.arange(n) - 1, 0)]
            tang /= np.maximum(np.linalg.norm(tang, axis=1), 1e-30)[:, None]
            tang *= np.where(np.arange(n) >= ile, 1.0, -1.0)[:, None]
            nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
            c0 = cont.mean(axis=0)
            nrm[((cont - c0)*nrm).sum(1) < 0] *= -1.0

            s_c = (sarc - sarc[ile])/CHORD[nm]
            y = profile_grid(0.045)
            for side, sel in (('upper', np.arange(ile, n - 1)),
                              ('lower', np.arange(1, ile))):
                # ~110 stations spread over the side, skipping the blunt face
                cand = sel[np.abs(s_c[sel]) < 1.0]
                take = cand[np.linspace(0, len(cand) - 1, 110).astype(int)]
                rows = []
                for i in take:
                    q = cont[i][None, :] + y[:, None]*nrm[i][None, :]
                    u = np.ma.filled(fu(q[:, 0], q[:, 1]), np.nan)
                    v = np.ma.filled(fv(q[:, 0], q[:, 1]), np.nan)
                    ut = np.asarray(u, float)*tang[i, 0] \
                        + np.asarray(v, float)*tang[i, 1]
                    r = integrals(y, ut)
                    if r:
                        r['s'] = abs(s_c[i])
                        rows.append(r)
                if not rows:
                    continue
                rows.sort(key=lambda d: d['s'])
                res[(case, nm, side)] = {
                    k: np.array([r[k] for r in rows])
                    for k in ('s', 'H', 're_theta', 'theta', 'dstar', 'delta')}
                nres = len(take) - len(rows)
                if nres:
                    print('   %s %s: %d of %d stations unresolved'
                          % (nm, side, nres, len(take)))

    # ------------------------------------------------------------- transition
    print()
    print('%-4s %-5s %-6s %7s %7s %9s %9s %9s %7s'
          % ('lvl', 'elem', 'side', 'H@0.2', 'H max', 's(Hmax)',
             'TR s/c', 'Re_th@TR', 'dropH'))
    for (case, nm, side), d in sorted(res.items()):
        H, s = d['H'], d['s']
        # beyond ~0.95 the blunt trailing edge and the merging wake make the
        # edge detection unreliable, so H there is not evidence of anything
        m = (s > 0.05) & (s < 0.95)
        if m.sum() < 12:
            continue
        Hm, sm, rth = H[m], s[m], d['re_theta'][m]
        # A fixed H threshold cannot separate a favourable-gradient LAMINAR
        # layer (H ~ 2.0-2.1 here on the flap lower surface) from a turbulent
        # one under adverse gradient (H ~ 1.9 after the flap's bubble
        # reattaches). What is unambiguous is the COLLAPSE of H: transition
        # drops it fast and by a lot. Locate the steepest fall and require the
        # peak-to-trough drop around it to be substantial.
        Hs = np.convolve(Hm, np.ones(3)/3.0, mode='same')
        Hs[0], Hs[-1] = Hm[0], Hm[-1]
        dH = np.gradient(Hs, sm)
        i = int(np.argmin(dH))
        pk = float(np.max(Hs[:i+1])) if i > 0 else Hs[0]
        tr_ = float(np.min(Hs[i:]))
        drop = pk - tr_
        tr = sm[i] if drop > 0.8 else None
        kmax = int(np.argmax(Hm))
        i02 = int(np.argmin(np.abs(sm - 0.2)))
        print('%-4s %-5s %-6s %7.2f %7.2f %9.3f %9s %9s %7s'
              % (lbl[case], nm, side, Hm[i02], Hm[kmax], sm[kmax],
                 '%.3f' % tr if tr else 'laminar',
                 '%.0f' % rth[i] if tr else '--',
                 '%.2f' % drop if tr else '%.2f' % drop))

    # ------------------------------------------------------------------- plot
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))
    for k, nm in enumerate(ELEMS):
        for row, fld, ylab, ylim in (
                (0, 'H', 'shape factor $H=\\delta^*/\\theta$', (1.0, 6.0)),
                (1, 're_theta', r'$Re_\theta$', None)):
            ax = axes[row, k]
            for case in cases:
                for side, ls in (('upper', '-'), ('lower', '--')):
                    d = res.get((case, nm, side))
                    if d is None:
                        continue
                    ax.plot(d['s'], d[fld], ls, color=cols.get(case, 'k'),
                            lw=1.4 if side == 'upper' else 1.0,
                            alpha=1.0 if side == 'upper' else 0.7,
                            label='%s %s' % (lbl[case], side))
            if fld == 'H':
                ax.axhline(H_BLASIUS, color='#666', ls=':', lw=1.0)
                ax.axhline(H_TURB, color='#000', ls=':', lw=1.0)
                ax.text(0.015, H_BLASIUS, ' Blasius 2.59', fontsize=7,
                        va='bottom', color='#666')
                ax.text(0.015, H_TURB, ' turbulent ~1.45', fontsize=7,
                        va='bottom')
                ax.axhline(0.5*(H_BLASIUS + H_TURB), color='#e08214',
                           lw=0.8, alpha=0.7)
                ax.set_ylim(*ylim)
            else:
                ax.set_yscale('log')
            ax.set_xlim(0, 1.0)
            ax.set_xlabel('$s/c$ from LE (element chord)')
            ax.set_ylabel(ylab)
            ax.grid(alpha=0.25, lw=0.6)
            ax.set_title('%s element' % nm)
            ax.legend(fontsize=7, ncol=2, loc='upper left')
    fig.suptitle('Two-element SA-AI, Re=1e6, M=0.1, '
                 r'$\alpha=-1^\circ$: boundary-layer state', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out)
    print('\nwrote', out)


if __name__ == '__main__':
    main()
