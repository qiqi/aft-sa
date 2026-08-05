"""Unrolled chi map over the FLAP's UPPER surface, with the wall-normal range
extended far enough to contain the fore element's wake.

Same frame as the paper's regen_bl_contour.py -- arc length across, wall
distance up, pcolormesh with a log norm and marker contours -- but restricted to
the one surface that matters here and taken out to d = 0.06 total chord instead
of 0.04, because the fore wake rides roughly 0.01-0.02 above the flap and the
whole question is whether its chi reaches down into the flap's own layer.

Layout: one ROW per incidence, tangential velocity on the LEFT and chi on the
RIGHT. Line contours rather than filled -- a filled map reads as "turbulent
here, laminar there" and hides where the c_v1 crossing actually sits, whereas
lines put the level values on the picture and let the near-wall band be
distinguished from the outer flow.

Velocity is the panel in which the fore wake is unambiguous; chi alone cannot
separate "wake arriving" from "layer amplifying".

Two loci are overlaid on both panels so the geometry of the interaction is
readable rather than inferred:
  BL edge      first interior maximum of the tangential velocity (the flap's own
               boundary-layer thickness)
  wake centre  maximum of omega ABOVE 1.2 * that edge, i.e. the vorticity peak
               that is not the wall layer

Run:  python3 plot_flap_chi_map.py out.pdf case_dir [case_dir ...]
"""
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator

import plot_solution_mesh as PM
import plot_cp_cf as C
import plot_paper_style as PS
from measure_l1_spacing import read_contours, surface_frame

import os
# Probe out to 0.06 so the outer deficit locus is still found above the
# display window, but show only the inner 0.04 where the contours matter.
L_PROBE = float(os.environ.get('L_PROBE', '0.06'))
Y_SHOW = float(os.environ.get('Y_SHOW', '0.04'))
N_PROBE = 240
# Which wall and which side to unroll. Wall 0 = fore, 1 = flap (the order the
# mesher sees them). Defaults to the flap upper surface.
ELEM = os.environ.get('ELEM', 'flap')
SIDE = os.environ.get('SIDE', 'upper')
WALL_IDX = {'fore': 0, 'flap': 1}[ELEM]
CHORD_FLAP = {'fore': 0.7039, 'flap': 0.30}[ELEM]
C_V1 = 7.1
CHI_INF = C_V1*np.exp(-9.0)


def flap_upper_map(case):
    hdr, pts, curves = read_contours('%s/contours_L1.txt' % case)
    walls = [n for w, n in curves if w]
    cont, seg, sarc, ile, u2 = surface_frame(pts, walls[WALL_IDX])
    n = len(cont)
    t = cont[np.minimum(np.arange(n)+1, n-1)] - cont[np.maximum(np.arange(n)-1, 0)]
    t /= np.maximum(np.linalg.norm(t, axis=1), 1e-30)[:, None]
    t *= np.where(np.arange(n) >= ile, 1.0, -1.0)[:, None]
    nrm = np.column_stack([-t[:, 1], t[:, 0]])
    c0 = cont.mean(axis=0)
    nrm[((cont - c0)*nrm).sum(1) < 0] *= -1.0

    second = np.arange(n) >= ile
    upper = second if u2 else ~second
    ii = np.where(upper if SIDE == 'upper' else ~upper)[0]
    ii = ii[(ii > 0) & (ii < n-1)]
    s_c = np.abs((sarc[ii] - sarc[ile])/CHORD_FLAP)
    o = np.argsort(s_c)
    ii, s_c = ii[o], s_c[o]
    keep = s_c <= 1.0
    ii, s_c = ii[keep], s_c[keep]

    fld = PS.Field(case)
    nu = PS.nu_of(case)
    dist = np.linspace(1e-6, L_PROBE, N_PROBE)
    X = cont[ii][None, :, 0] + dist[:, None]*nrm[ii][None, :, 0]
    Z = cont[ii][None, :, 1] + dist[:, None]*nrm[ii][None, :, 1]
    sh = X.shape
    chi = fld('nuHat', X.ravel(), Z.ravel()).reshape(sh)/nu
    om = fld('omega', X.ravel(), Z.ravel()).reshape(sh)
    u = fld('u', X.ravel(), Z.ravel()).reshape(sh)
    w = fld('w', X.ravel(), Z.ravel()).reshape(sh)
    tt = t[ii]
    ut = u*tt[None, :, 0] + w*tt[None, :, 1]

    # BL edge: first interior maximum of the tangential velocity
    M = sh[1]
    delta = np.full(M, np.nan)
    for jj in range(M):
        a = ut[:, jj]
        g = np.isfinite(a)
        if g.sum() < 20:
            continue
        k = int(np.nanargmax(a))
        for i in range(2, len(a)-2):
            if (np.isfinite(a[i-1:i+2]).all() and a[i] >= a[i-1] and a[i] > a[i+1]
                    and a[i] > 0.3*np.nanmax(a)):
                k = i
                break
        delta[jj] = dist[k]

    # Wake centre from the VELOCITY DEFICIT, not the vorticity peak. Locating it
    # as the omega maximum above the boundary-layer edge does not work: the wall
    # layer's omega is ~1e3 while the wake's is ~1-10, so any threshold relative
    # to the column maximum rejects the wake entirely and the locus came out all
    # NaN. A wake is a momentum deficit, so take the local MINIMUM of the
    # tangential velocity above the edge.
    wake = np.full(M, np.nan)
    for jj in range(M):
        lim = 1.2*delta[jj]
        if not np.isfinite(lim):
            continue
        m = (dist > lim) & np.isfinite(ut[:, jj])
        if m.sum() < 8:
            continue
        idx = np.where(m)[0]
        a = ut[idx, jj]
        for i in range(1, len(a)-1):
            if a[i] < a[i-1] and a[i] <= a[i+1]:
                # require a real deficit relative to the local outer flow
                outer = np.nanmax(a[i:])
                if outer > 0 and (outer - a[i])/outer > 0.02:
                    wake[jj] = dist[idx[i]]
                break
    return s_c, dist, chi, om, delta, wake, ut


def bubble(case):
    """Separation / reattachment on the flap upper surface, for annotation."""
    try:
        hdr, pts, curves = read_contours('%s/contours_L1.txt' % case)
        walls = [n for w, n in curves if w]
        cont, seg, sarc, ile, u2 = surface_frame(pts, walls[WALL_IDX])
        P, cp, cf, yp = C.midplane_surface(
            '%s/surface_fluid_%s_proc0.vtu' % (case, ELEM))
        j, upper = C.order_on_contour(P, cont, ile, u2)
        itp = C.Interp(case)
        cfs, _ = C.signed_cf(P, cf, cont, j, ile, itp, 3e-4)
        s = np.abs((sarc[j] - sarc[ile])/CHORD_FLAP)
        msk = upper if SIDE == 'upper' else ~upper
        o = np.argsort(s[msk])
        return C.crossings(s[msk][o], cfs[msk][o], lo=0.05, hi=0.97)
    except Exception:
        return []


def main():
    out = sys.argv[1]
    cases = sorted(sys.argv[2:], key=lambda c: PS.parse(c.rstrip('/'))[1])
    n = len(cases)
    CHI_LEVELS = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
    U_LEVELS = np.arange(0.2, 1.61, 0.2)
    # Rows are deliberately tall: the unrolled frame has no natural aspect,
    # and at 1.85 in a row the near-surface contours were compressed into a
    # sliver at the bottom of each panel.
    fig, axs = plt.subplots(n, 2, figsize=(11.0, 2.45*n + 0.8), squeeze=False,
                            sharex=True, sharey=True,
                            gridspec_kw=dict(hspace=0.0, wspace=0.06))
    for r, case in enumerate(cases):
        al = PS.parse(case.rstrip('/'))[1]
        s_c, dist, chi, om, delta, wake, ut = flap_upper_map(case)
        S, D = np.meshgrid(s_c, dist)
        cr = bubble(case)

        axu, axc = axs[r, 0], axs[r, 1]
        cu = axu.contour(S, D, ut/0.10, levels=U_LEVELS, cmap='viridis',
                         norm=matplotlib.colors.Normalize(0.0, 1.6),
                         linewidths=0.8)
        cc = axc.contour(S, D, np.clip(chi, 1e-6, None), levels=CHI_LEVELS,
                         cmap='magma', norm=LogNorm(1e-4, 1e2), linewidths=0.8)
        axc.contour(S, D, np.clip(chi, 1e-6, None), levels=[C_V1], colors='k',
                    linewidths=1.8)
        if r == 0:
            axu.clabel(cu, fmt='%.1f', fontsize=5.5, inline=True)
            axc.clabel(cc, fmt='%g', fontsize=5.5, inline=True)
        for a in (axu, axc):
            a.plot(s_c, delta, '-', color='0.25', lw=1.0)
            a.plot(s_c, wake, '--', color='0.25', lw=1.1)
            for a_, pos in cr:
                a.axvline(a_, color='c', ls=':', lw=1.0)
            a.set_ylim(0, Y_SHOW)
            a.set_xlim(s_c.min(), 1.0)
            a.grid(alpha=0.15, lw=0.4)
            # With hspace=0 the shared y ticks collide at the row boundaries;
            # prune the end ticks so the endpoints do not print twice.
            a.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
            for sp in ('top', 'right'):
                a.spines[sp].set_visible(False)
            if r != n - 1:
                a.spines['bottom'].set_visible(False)
                a.tick_params(axis='x', length=0)
        # Incidence label in-plot rather than as a y-label in the margin.
        for a in (axu, axc):
            a.text(0.012, 0.92, r'$\alpha=%+.0f^\circ$' % al,
                   transform=a.transAxes, fontsize=9, va='top', ha='left',
                   bbox=dict(fc='white', ec='none', alpha=0.7, pad=1.2))
        axc.tick_params(axis='y', labelleft=False)
    axs[0, 0].set_title(r'$u_t/V_\infty$', fontsize=10)
    axs[0, 1].set_title(r'$\chi$   (heavy $\chi=c_{v1}=7.1$)', fontsize=10)
    for c in (0, 1):
        axs[-1, c].set_xlabel('$s/c$ along the %s %s surface' % (ELEM, SIDE))
    fig.suptitle('%s %s surface: grey solid boundary-layer edge, grey dashed '
                 'outer deficit locus, cyan separation and reattachment'
                 % (ELEM.upper(), SIDE), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out)
    print('wrote %s  (%d rows)' % (out, n))


if __name__ == '__main__':
    main()
