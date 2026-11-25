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
distinguished from the outer flow. The lines are black with the value written
on them, as in the paper's other near-wall figures; chi < 1 is dashed and
chi = c_v1 is heavy. The script also writes <out>.json with the stations and
the chi levels that the paper's text quotes.

Velocity is the panel in which the fore wake is unambiguous; chi alone cannot
separate "wake arriving" from "layer amplifying".

The boundary-layer edge and outer-deficit loci were dropped: against line
contours they clutter the panel without adding anything the velocity contours do
not already show. They are still computed, since the wake locus is what
established that the deficit band sits clear of the layer until the aft 15%.

The flap's leading edge sits under the fore element's trailing edge, so a probe
shot normal from the first ~9% of flap arc length runs into the fore body. Those
points are not fluid; they are masked out of the fields and the region is shaded
grey with a drawn edge, the same way the global chi panels shade the two bodies,
so the slot reads as geometry rather than as a blank patch in the flow.

Run:  python3 plot_flap_chi_map.py out.pdf case_dir [case_dir ...]
"""
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm
from matplotlib.path import Path
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
    other = Path(pts[walls[1 - WALL_IDX]][:, :2])   # the element we are NOT unrolling
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
    # Probe points that landed inside the OTHER element are solid, not fluid:
    # whatever the interpolator returns there is meaningless, so drop it before
    # anything reduces or contours these arrays.
    solid = other.contains_points(
        np.column_stack([X.ravel(), Z.ravel()])).reshape(sh)
    chi = fld('nuHat', X.ravel(), Z.ravel()).reshape(sh)/nu
    om = fld('omega', X.ravel(), Z.ravel()).reshape(sh)
    u = fld('u', X.ravel(), Z.ravel()).reshape(sh)
    w = fld('w', X.ravel(), Z.ravel()).reshape(sh)
    tt = t[ii]
    ut = u*tt[None, :, 0] + w*tt[None, :, 1]
    for a in (chi, om, ut):
        a[solid] = np.nan

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
    return s_c, dist, chi, om, delta, wake, ut, solid


def wall_cf(case):
    """Arc length and signed skin friction along the unrolled surface."""
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
    return s[msk][o], cfs[msk][o]


def bubble(case):
    """Separation / reattachment on the flap upper surface, for annotation."""
    try:
        s, cfs = wall_cf(case)
        return C.crossings(s, cfs, lo=0.05, hi=0.97)
    except Exception:
        return []


# Contour levels and labels in the style of the paper's near-wall figures
# (repro/cfd/regen_chi_sheets.py): black lines, the value written on the line,
# chi < 1 dashed, chi = c_v1 heavy.
U_LEVELS = [-0.05, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
U_LABEL = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
CHI_MINOR = [-2.5, -1.5, -0.5]
CHI_LEVELS = [-3, -2, -1, 0, np.log10(C_V1), np.log10(30.0), 2.0, 3.0]
CHI_LW = [0.7, 0.7, 0.7, 0.7, 1.5, 0.7, 0.7, 0.7]
CHI_FMT = {-3: '-3', -2: '-2', -1: '-1', 0: r'$\chi{=}1$',
           np.log10(C_V1): r'$c_{v1}$', np.log10(30.0): '30',
           2.0: r'$10^2$', 3.0: r'$10^3$'}
# The band of wall distances, and the stations ahead of the bubble, over which
# the chi inside the flap's boundary layer is reported in the JSON.
LAYER_D = (0.0005, 0.002)
LAYER_S = (0.15, 0.40)


def label_points(field, s, d, levels, stations, ymin=0.004, ymax=0.031,
                 gap=0.0035):
    """Where each level crosses the wall-normal line at each station: label
    positions that keep clear of the wall, of the incidence label and of one
    another. (clabel's own placement crowds the wall, and its manual mode
    fails when a level has no line in the panel.)"""
    pts = []
    for sv in stations:
        j = int(np.argmin(abs(s - sv)))
        f0 = field[:, j]
        taken = []
        for lev in levels:
            f = f0 - lev
            ok = np.isfinite(f[:-1]) & np.isfinite(f[1:])
            for i in np.where(ok & (f[:-1]*f[1:] < 0))[0]:
                y = d[i] + f[i]/(f[i] - f[i+1])*(d[i+1] - d[i])
                if ymin < y < ymax and all(abs(y - q) > gap for q in taken):
                    taken.append(y)
                    pts.append((s[j], y, lev))
    return pts


def put_labels(ax, pts, fmt):
    for x, y, lev in pts:
        ax.text(x, y, fmt(lev), fontsize=6.5, ha='center', va='center',
                zorder=8, bbox=dict(fc='white', ec='none', pad=0.6))


def one_case(case):
    al = PS.parse(case.rstrip('/'))[1]
    s_c, dist, chi, om, delta, wake, ut, solid = flap_upper_map(case)
    s, cfs = wall_cf(case)
    cr = C.crossings(s, cfs, lo=0.05, hi=0.97)
    m = (s > 0.05) & (s < 0.97)
    k = int(np.argmin(np.where(m, cfs, np.inf)))
    band = chi[(dist >= LAYER_D[0]) & (dist <= LAYER_D[1])][
        :, (s_c >= LAYER_S[0]) & (s_c <= LAYER_S[1])]
    summary = dict(alpha=al,
                   stations=[round(float(x), 4) for x, _ in cr],
                   cf_min=float('%.3g' % cfs[k]), s_cf_min=round(float(s[k]), 4),
                   chi_layer=[float('%.3g' % np.nanmin(band)),
                              float('%.3g' % np.nanmax(band))])
    return al, s_c, dist, chi, ut, solid, [x for x, _ in cr], summary


def main():
    import json
    from multiprocessing import Pool
    out = sys.argv[1]
    cases = sorted(sys.argv[2:], key=lambda c: PS.parse(c.rstrip('/'))[1])
    n = len(cases)
    with Pool(n) as pool:
        results = pool.map(one_case, cases)
    # The canvas is the page's text width, so the type prints at its declared
    # size; rows are as tall as a page of five allows.
    plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6})
    fig, axs = plt.subplots(n, 2, figsize=(6.5, 1.45*n + 0.45), squeeze=False,
                            sharex=True, sharey=True,
                            gridspec_kw=dict(hspace=0.0, wspace=0.04))
    for r, (al, s_c, dist, chi, ut, solid, cr, _) in enumerate(results):
        S, D = np.meshgrid(s_c, dist)
        ut = ut/0.10
        logchi = np.log10(np.clip(chi, 1e-6, None))
        axu, axc = axs[r, 0], axs[r, 1]
        # Fore-element interior first, so the contours draw over its edge and
        # the shading reads as the body it is (grey/edge as in chi_global.py).
        for a in (axu, axc):
            a.contourf(S, D, solid.astype(float), levels=[0.5, 1.5],
                       colors=['0.85'], zorder=2)
            a.contour(S, D, solid.astype(float), levels=[0.5],
                      colors='0.35', linewidths=0.8, zorder=3)
        axu.contour(S, D, ut, levels=U_LEVELS, colors='k', linewidths=0.6,
                    zorder=4)
        put_labels(axu, label_points(ut, s_c, dist, U_LABEL, (0.30, 0.72)),
                   lambda v: '%g' % v)
        axc.contour(S, D, logchi, levels=CHI_MINOR, colors='k',
                    linewidths=0.4, zorder=4)
        axc.contour(S, D, logchi, levels=CHI_LEVELS, colors='k',
                    linewidths=CHI_LW, zorder=4)
        put_labels(axc, label_points(logchi, s_c, dist, CHI_LEVELS,
                                     (0.33, 0.78)), lambda v: CHI_FMT[v])
        for a in (axu, axc):
            for x in cr:
                a.axvline(x, color='#1f77b4', ls=':', lw=1.0, zorder=6)
            a.set_ylim(0, Y_SHOW)
            a.set_xlim(0, 1.0)
            # With hspace=0 the shared y ticks collide at the row boundaries;
            # prune the end ticks so the endpoints do not print twice.
            a.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
            a.tick_params(length=2.5, width=0.5)
            a.text(0.985, 0.93, r'$\alpha=%+.0f^\circ$' % al,
                   transform=a.transAxes, fontsize=8.5, va='top', ha='right',
                   bbox=dict(fc='white', ec='none', alpha=0.8, pad=1.0),
                   zorder=7)
        if r == n//2:
            axu.set_ylabel('wall distance / total chord')
        axu.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axs[0, 0].set_title(r'$u_t/V_\infty$', fontsize=9, pad=3)
    axs[0, 1].set_title(r'$\log_{10}\chi$', fontsize=9, pad=3)
    for c in (0, 1):
        axs[-1, c].set_xlabel('arc length / %s chord' % ELEM)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.972, bottom=0.056)
    # the two columns share the tick at the seam; print it once
    fig.canvas.draw()
    axs[-1, 0].get_xticklabels()[-1].set_visible(False)
    fig.savefig(out)
    print('wrote %s  (%d rows)' % (out, n))
    js = os.path.splitext(out)[0] + '.json'
    doc = {'_doc': 'Upper surface of the flap, per incidence: stations (arc length / '
                   'flap chord) where the signed C_f changes sign, the smallest signed '
                   'C_f and its station, and the range of chi for wall distances '
                   '%g to %g (total chord) at stations %g to %g. Written by '
                   'twoelement/plot_flap_chi_map.py.' % (LAYER_D + LAYER_S),
           'cases': [r[-1] for r in results]}
    with open(js, 'w') as f:
        json.dump(doc, f, indent=1)
        f.write('\n')
    print('wrote', js)
    for r in results:
        print(r[-1])


if __name__ == '__main__':
    main()
