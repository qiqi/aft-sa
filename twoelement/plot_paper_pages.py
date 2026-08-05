"""Five-row surface diagnostic, one COLUMN per angle of attack, two angles per
full page -- the layout the paper's single-element figures use
(paper/repro/cfd/regen_epp_resweep_suite.make_fig, 5 rows x 2 columns).

Rows, top to bottom:
    1  max Re_Omega        d^2 |omega| / nu along a wall-normal probe
    2  max Omega_hat I_hat the sphere-kernel rate coordinate
    3  max chi             nuHat / muRef, with c_v1 and the e^9 seed marked
    4  -Cp
    5  Cf (signed)

Conventions as in the paper: upper surface blue (C0), lower red (C3),
refinement level by LINE WIDTH (L0 0.8, L1 1.6, L2 2.4) with all levels overlaid
on the same axes, and here the ELEMENT by line style -- fore solid, flap dashed
-- with both elements against global x.

Per-case probe results are cached to .paper_cache/<case>.npz. The expensive part
is reading a 31 MB volume and building six interpolators over ~150k nodes (~15 s
a case); the cache makes re-plotting free, which matters because these figures
get regenerated every time another refinement level lands.

Run:  python3 plot_paper_pages.py out.pdf case_dir [case_dir ...]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

import plot_paper_style as PS
from measure_l1_spacing import read_contours, surface_frame

CACHE = '.paper_cache'
ELEMS = PS.ELEMS


MAX_PTS = 500


def _thin(a, n=MAX_PTS):
    """Decimate along the last axis to at most n columns. Rasterizing these
    panels at 300 dpi made the file LARGER (2.26 -> 3.19 MB); thinning the
    curves keeps them vector and cuts the path count, and the surface arrays
    carry ~1200 points a side which the panel cannot resolve anyway."""
    if a.shape[-1] <= n:
        return a
    idx = np.linspace(0, a.shape[-1] - 1, n).astype(int)
    return a[..., idx]


def extract(case):
    """Probe maxima and surface Cp/Cf for both elements, cached."""
    os.makedirs(CACHE, exist_ok=True)
    key = os.path.join(CACHE, case.rstrip('/').replace('/', '_') + '.npz')
    if os.path.exists(key):
        z = np.load(key, allow_pickle=True)
        return {k: z[k] for k in z.files}
    hdr, pts, curves = read_contours('%s/contours_L1.txt' % case)
    walls = [n for w, n in curves if w]
    fld = PS.Field(case)
    nu = PS.nu_of(case)
    out = {}
    for k, nm in enumerate(ELEMS):
        cont, seg, sarc, ile, u2 = surface_frame(pts, walls[k])
        pm = PS.probe_maxima(fld, cont, ile, u2, nu)
        for side in ('upper', 'lower'):
            if side in pm:
                x, reo, oi, chi = pm[side]
                o = np.argsort(x)
                out['%s_%s_probe' % (nm, side)] = np.vstack(
                    [x[o], reo[o], oi[o], chi[o]])
        sc = PS.surface_cp_cf(case, nm, cont, ile, u2, fld=fld)
        for side in ('upper', 'lower'):
            x, cp, cf = sc[side]
            out['%s_%s_surf' % (nm, side)] = np.vstack([x, cp, cf])
    np.savez_compressed(key, **out)
    return out


def main():
    out = sys.argv[1]
    cases = sys.argv[2:]
    groups = {}
    for c in cases:
        lvl, al = PS.parse(c.rstrip('/'))
        groups.setdefault(al, []).append((lvl, c))
    alphas = sorted(groups)
    print('alphas: %s' % ', '.join('%+.0f' % a for a in alphas))

    with PdfPages(out) as pdf:
        for i in range(0, len(alphas), 2):
            chunk = alphas[i:i+2]
            fig, axs = plt.subplots(5, len(chunk), figsize=(5.76*len(chunk), 13),
                                    sharex=True, squeeze=False)
            for col, al in enumerate(chunk):
                ax_reo, ax_P, ax_chi, ax_cp, ax_cf = (axs[r, col]
                                                      for r in range(5))
                for lvl, case in sorted(groups[al]):
                    lw = PS.LEVEL_LW.get(lvl, 1.6)
                    try:
                        d = extract(case)
                    except Exception as e:                      # noqa: BLE001
                        print('%s: %s' % (case, str(e)[:70]))
                        continue
                    for nm in ELEMS:
                        ls = PS.ELEM_LS[nm]
                        for side, cflr in (('upper', PS.UP_COLOR),
                                           ('lower', PS.LO_COLOR)):
                            k = '%s_%s_probe' % (nm, side)
                            if k in d:
                                x, reo, oi, chi = _thin(d[k])
                                ax_reo.semilogy(x, reo, ls=ls, lw=lw, color=cflr)
                                ax_P.semilogy(x, np.clip(oi, 1e-4, None),
                                              ls=ls, lw=lw, color=cflr)
                                ax_chi.semilogy(x, np.clip(chi, PS.CHI_LO*1e-2,
                                                           None),
                                                ls=ls, lw=lw, color=cflr)
                            k = '%s_%s_surf' % (nm, side)
                            if k in d:
                                x, cp, cf = _thin(d[k])
                                ax_cp.plot(x, -cp, ls=ls, lw=lw, color=cflr)
                                ax_cf.plot(x, cf, ls=ls, lw=lw, color=cflr)
                ax_reo.axhline(PS.REOMC_FLOOR, color='gray', ls='--', lw=0.6,
                               alpha=0.6)
                ax_reo.set_ylim(1e2, 1e4)
                ax_P.set_ylim(1e-3, 1.0)
                ax_chi.axhline(PS.C_V1, color='gray', ls=':', lw=0.6, alpha=0.6)
                ax_chi.axhline(PS.CHI_INF, color='gray', ls='-.', lw=0.6,
                               alpha=0.6)
                ax_chi.set_ylim(PS.CHI_LO, PS.CHI_HI)
                ax_cf.axhline(0.0, color='gray', lw=0.6, alpha=0.6)
                ax_cf.set_ylim(-0.004, 0.012)
                ax_cf.set_xlabel('$x$')
                for a in (ax_reo, ax_P, ax_chi, ax_cp, ax_cf):
                    a.grid(alpha=0.3, which='both')
                # The paper drops in-figure column titles and lets the caption
                # carry the assignment. That works for one figure; across a
                # multi-page sweep the reader cannot track it, so the angle is
                # labelled compactly here.
                ax_reo.text(0.02, 0.94, r'$\alpha=%+.0f^\circ$' % al,
                            transform=ax_reo.transAxes, fontsize=11,
                            va='top', ha='left')
                if col == 0:
                    ax_reo.set_ylabel(r'$\max Re_\Omega$ (log)')
                    ax_P.set_ylabel(r'$\max \hat\Omega \hat I$ (log)')
                    ax_chi.set_ylabel(r'$\max \chi$ (log)')
                    ax_cp.set_ylabel(r'$-C_p$')
                    ax_cf.set_ylabel(r'$C_f$')
            lv = sorted({l for a in chunk for l, _ in groups[a]})
            handles = [Line2D([], [], color=PS.UP_COLOR, lw=1.6, label='upper'),
                       Line2D([], [], color=PS.LO_COLOR, lw=1.6, label='lower'),
                       Line2D([], [], color='0.3', lw=1.6, ls='-', label='fore'),
                       Line2D([], [], color='0.3', lw=1.6, ls='--', label='flap')]
            handles += [Line2D([], [], color='0.3', lw=PS.LEVEL_LW[l], label=l)
                        for l in lv]
            axs[0, 0].legend(handles=handles, fontsize=7, ncol=4,
                             loc='upper right')
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print('wrote %s  (%d pages)' % (out, (len(alphas) + 1)//2))


if __name__ == '__main__':
    main()
