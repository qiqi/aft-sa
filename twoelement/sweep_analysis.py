"""Alpha sweep: when does the MAIN element's upper surface transition, and what
does the flap's laminar separation bubble do when it happens?

This is the question the whole case exists to answer. While the fore element is
laminar its wake is a laminar wake, and the flap's bubble is left alone. Once the
fore upper surface transitions, a turbulent wake convects over the flap's suction
side, and the expectation is that it quenches the bubble. The sweep tests that.

For every (level, alpha) it reports, from the mid-plane volume field:

  fore upper / lower   transition location from the collapse of H, and whether
                       the surface separates at all
  flap upper           bubble extent from the signed Cf, the H peak inside it,
                       and the transition location
  forces               CL, CD

Transition is located by the COLLAPSE of H rather than a threshold on H or Cf: a
favourable-gradient laminar layer sits near H = 2.0 and a turbulent layer under
an adverse gradient also sits near 1.9, so no fixed level separates them, but the
drop through transition is large and fast.

Run:  python3 sweep_analysis.py [out.pdf]
"""
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.spatial import cKDTree

import panel2e as M
import plot_solution_mesh as PM
import bl_profiles as BL
import plot_cp_cf as C
from measure_l1_spacing import read_contours, surface_frame

CHORD = {'fore': 0.7039, 'flap': 0.3}
H_BLAS, H_TURB = 2.59, 1.45


def case_dirs():
    out = []
    for d in sorted(glob.glob('case_L*_v2_yp0.5*')):
        if not os.path.isdir(d):
            continue
        m = re.match(r'case_(L\d)_v2_yp0\.5(?:_a([+-][\d.]+))?$', d)
        if not m:
            continue
        out.append((m.group(1), float(m.group(2) or -1.0), d))
    return sorted(out, key=lambda t: (t[0], t[1]))


def analyse(case, alpha):
    hdr, pts, curves = read_contours('%s/contours_L1.txt' % case)
    walls = [n for w, n in curves if w]
    g, vpts, arr = PM.read_vtu('%s/volume_proc0.vtu' % case)
    idx, tris, P2 = PM.midplane_tris(g, vpts)
    vel = arr['velocity'][idx]
    T = mtri.Triangulation(P2[:, 0], P2[:, 1], tris)
    fu = mtri.LinearTriInterpolator(T, vel[:, 0])
    fv = mtri.LinearTriInterpolator(T, vel[:, 2])

    def interp(x, z):
        u = np.ma.filled(fu(np.atleast_1d(x), np.atleast_1d(z)), np.nan)
        v = np.ma.filled(fv(np.atleast_1d(x), np.atleast_1d(z)), np.nan)
        return np.asarray(u, float), np.asarray(v, float)

    res = {}
    for k, nm in enumerate(('fore', 'flap')):
        cont, seg, sarc, ile, u2 = surface_frame(pts, walls[k])
        n = len(cont)
        tang = cont[np.minimum(np.arange(n)+1, n-1)] \
            - cont[np.maximum(np.arange(n)-1, 0)]
        tang /= np.maximum(np.linalg.norm(tang, axis=1), 1e-30)[:, None]
        tang *= np.where(np.arange(n) >= ile, 1.0, -1.0)[:, None]
        nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
        c0 = cont.mean(axis=0)
        nrm[((cont - c0)*nrm).sum(1) < 0] *= -1.0
        s_c = (sarc - sarc[ile])/CHORD[nm]
        y = BL.profile_grid(0.045)

        # signed Cf from the surface output, for separation
        Pp, cp, cf, yp = C.midplane_surface(
            '%s/surface_fluid_%s_proc0.vtu' % (case, nm))
        j, upper = C.order_on_contour(Pp, cont, ile, u2)
        # Sign |Cf| using the SAME interpolant already built for the profiles,
        # rather than constructing plot_cp_cf's Interp again -- that would reread
        # and retriangulate the volume file once per element per case.
        tg = tang[j]
        nn = nrm[j]
        q = Pp + 2.0e-4*nn
        uu, vv = interp(q[:, 0], q[:, 1])
        dot = uu*tg[:, 0] + vv*tg[:, 1]
        sgn = np.where(~np.isfinite(dot) | (np.abs(dot) < 1e-6), 1.0,
                       np.sign(dot))
        cfs = cf*sgn
        s_surf = np.abs((sarc[j] - sarc[ile])/CHORD[nm])

        for side, sel in (('upper', np.arange(ile, n-1)),
                          ('lower', np.arange(1, ile))):
            cand = sel[np.abs(s_c[sel]) < 1.0]
            take = cand[np.linspace(0, len(cand)-1, 110).astype(int)]
            rows = []
            for i in take:
                qq = cont[i][None, :] + y[:, None]*nrm[i][None, :]
                u, v = interp(qq[:, 0], qq[:, 1])
                ut = u*tang[i, 0] + v*tang[i, 1]
                r = BL.integrals(y, ut)
                if r:
                    r['s'] = abs(s_c[i])
                    rows.append(r)
            if len(rows) < 15:
                continue
            rows.sort(key=lambda d: d['s'])
            H = np.array([r['H'] for r in rows])
            S = np.array([r['s'] for r in rows])
            RT = np.array([r['re_theta'] for r in rows])
            m = (S > 0.05) & (S < 0.95)
            Hm, Sm, RTm = H[m], S[m], RT[m]
            Hs = np.convolve(Hm, np.ones(3)/3.0, mode='same')
            Hs[0], Hs[-1] = Hm[0], Hm[-1]
            dH = np.gradient(Hs, Sm)
            i0 = int(np.argmin(dH))
            # The peak-to-trough drop must be computed even when the steepest
            # fall is at the first station: guarding with `if i0 > 0 else 0.0`
            # silently reported dropH = 0 (hence "laminar") for most of the
            # sweep, which is a detector artefact, not a physical result.
            drop = float(np.max(Hs[:i0+1]) - np.min(Hs[i0:]))
            # H beyond ~10 means the edge detection has lost the layer (massive
            # separation), not a real shape factor; flag rather than believe it.
            broken = bool(np.nanmax(Hm) > 10.0)
            tr = float(Sm[i0]) if (drop > 0.8 and not broken) else None

            # separation from the signed Cf on this side
            up_mask = (j >= ile) if u2 else (j < ile)
            sm = up_mask if side == 'upper' else ~up_mask
            o = np.argsort(s_surf[sm])
            ss, cc = s_surf[sm][o], cfs[sm][o]
            cr = C.crossings(ss, cc, lo=0.03, hi=0.98)
            res[(nm, side)] = dict(tr=tr, drop=drop, broken=broken,
                                   Hmax=float(Hm.max()),
                                   sHmax=float(Sm[int(np.argmax(Hm))]),
                                   re_tr=float(RTm[i0]) if tr else np.nan,
                                   sep=[(round(a, 3), b) for a, b in cr],
                                   ypmax=float(np.nanmax(yp)))
    return res


def forces(level, alpha):
    sfx = '_yp0.5' + ('' if alpha == -1.0 else '_a%+05.1f' % alpha)
    p = 'ladder_results_v2%s.json' % sfx
    if not os.path.exists(p):
        return np.nan, np.nan
    d = json.load(open(p))
    if level not in d:
        return np.nan, np.nan
    f = d[level].get('forces', '')
    def gg(pat):
        r = re.search(pat, f)
        return float(r.group(1)) if r else np.nan
    return gg(r"'CL': ([-0-9.e+]+)"), gg(r"'CD': ([-0-9.e+]+)")


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'sweep_analysis.pdf'
    rows = []
    for level, alpha, d in case_dirs():
        try:
            r = analyse(d, alpha)
        except Exception as e:                                  # noqa: BLE001
            print('%s alpha %+.1f: FAILED %s' % (level, alpha, str(e)[:80]))
            continue
        CL, CD = forces(level, alpha)
        rows.append((level, alpha, r, CL, CD))
        print('%s a=%+.1f done' % (level, alpha))

    print('\n%-4s %-6s | %-22s | %-30s | %8s %8s'
          % ('lvl', 'alpha', 'FORE upper', 'FLAP upper bubble', 'CL', 'CD'))
    print('%-4s %-6s | %-22s | %-30s |'
          % ('', '', 'TR s/c   dropH  sep', 'SEP    REATT   len   TR    Hpk'))
    for level, alpha, r, CL, CD in rows:
        fu = r.get(('fore', 'upper'), {})
        fl = r.get(('flap', 'upper'), {})
        sep = [a for a, pos in fl.get('sep', []) if pos]
        rea = [a for a, pos in fl.get('sep', []) if not pos]
        s0 = sep[0] if sep else np.nan
        r0 = rea[0] if rea else np.nan
        fsep = 'yes' if fu.get('sep') else 'no'
        flag = '*' if (fu.get('broken') or fl.get('broken')) else ' '
        print('%-4s %+6.1f%s| %6s %6.2f  %-5s | %6.3f %6.3f %6.3f %6s %5.2f | '
              '%8.4f %8.5f'
              % (level, alpha, flag,
                 '%.3f' % fu['tr'] if fu.get('tr') else 'lam',
                 fu.get('drop', np.nan), fsep,
                 s0, r0, (r0 - s0) if np.isfinite(r0*s0) else np.nan,
                 '%.3f' % fl['tr'] if fl.get('tr') else 'lam',
                 fl.get('Hmax', np.nan), CL, CD))

    # -------------------------------------------------------------- figure --
    fig, ax = plt.subplots(3, 1, figsize=(8.8, 10.4), sharex=True)
    for level, c in (('L0', '#c44e52'), ('L1', '#1f4e9c'), ('L2', '#2ca02c')):
        sub = [(a, r, CL, CD) for lv, a, r, CL, CD in rows if lv == level]
        if not sub:
            continue
        al = np.array([s[0] for s in sub])
        tr = np.array([s[1].get(('fore', 'upper'), {}).get('tr') or np.nan
                       for s in sub], dtype=float)
        ax[0].plot(al, tr, 'o-', color=c, lw=1.5, label=level)
        sp = np.array([[a for a, p in s[1].get(('flap', 'upper'),
                                               {}).get('sep', []) if p][:1]
                       or [np.nan] for s in sub], dtype=float).ravel()
        re_ = np.array([[a for a, p in s[1].get(('flap', 'upper'),
                                                {}).get('sep', []) if not p][:1]
                        or [np.nan] for s in sub], dtype=float).ravel()
        ax[1].plot(al, sp, 'o-', color=c, lw=1.5, label='%s separation' % level)
        ax[1].plot(al, re_, 's--', color=c, lw=1.2,
                   label='%s reattachment' % level)
        hp = np.array([s[1].get(('flap', 'upper'), {}).get('Hmax', np.nan)
                       for s in sub], dtype=float)
        ax[2].plot(al, hp, 'o-', color=c, lw=1.5, label=level)
    ax[0].set_ylabel('fore UPPER transition, $s/c$')
    ax[0].set_title('does the main element transition, and where?', fontsize=10)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.25, lw=.6)
    ax[1].set_ylabel('flap upper bubble, $s/c$')
    ax[1].set_title('flap laminar separation bubble', fontsize=10)
    ax[1].legend(fontsize=7.5, ncol=2); ax[1].grid(alpha=.25, lw=.6)
    ax[2].set_ylabel(r'$H$ peak in the bubble')
    ax[2].set_xlabel(r'$\alpha$, deg')
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.25, lw=.6)
    fig.tight_layout()
    fig.savefig(out)
    print('\nwrote', out)
