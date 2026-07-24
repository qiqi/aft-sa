"""2D-style transition diagnostic at one spanwise station of the 3D wing:
max-chi(x/c) from the RANS solution (log axis) with the mfoil e^N envelope
N(x) overlaid on a linked linear axis (N = ln(chi/chi_inf), one unit = one
e-fold), plus a -Cp panel for the pressure-gradient context. Diagnoses the
model's implied amplification against the e^N reference.

Usage: python3 section_chi_diag.py [eta=0.305] [alpha=5] [levels=L1
       (comma list, e.g. L1,L2 — cavity included only where the case exists)]
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy

sys.path.insert(0, '/home/qiqi/flexcompute/sa-ai/src/validation')
from contextlib import redirect_stdout, redirect_stderr
import mfoil as MF
from loguru import logger
logger.disable('mfoil')

from wing_geometry import chord, HALF_SPAN, XQC, C_ROOT
import sectional_compare as SC

HERE = os.path.dirname(os.path.abspath(__file__))
ETA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.305
ALPHA = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
CHI_INF = 8.76e-6
NCRIT = np.log(1.0 / CHI_INF)          # ledger N at chi=1 (11.65)
NCRIT_MACK = 13.6                       # Mack N_crit of this seed
RE_ROOT = 5.0e5
LEVELS = (sys.argv[3] if len(sys.argv) > 3 else 'L1').split(',')
CASES = {}
for lv in LEVELS:
    for fam, surf in (('ogrid', 'surface_fluid_wing.pvtu'),
                      ('cavity', 'surface_farfield_body.pvtu')):
        c = f'case_{fam}_{lv}_saai_a{int(ALPHA)}'
        if os.path.exists(f'{os.path.dirname(os.path.abspath(__file__))}/{c}/chi_surface.npz'):
            CASES[f'{fam} {lv}'] = (c, surf)
SEC_DAT = None  # resolved below from the AVL work dir


def section_slice(case, surf, y0, band):
    """(xc, chi) upper surface from chi_surface.npz + (xc, cp) from surface pvtu."""
    d = np.load(f'{HERE}/{case}/chi_surface.npz')
    w, chi = d['wall'], d['chi']
    if band < 0.05:                     # O-grid: snap to the nearest wall plane
        ys = np.unique(np.round(w[:, 1], 6))
        y0 = float(ys[np.argmin(np.abs(ys - y0))])
        band = 1e-3
    m = np.abs(w[:, 1] - y0) < band
    c_loc = chord(np.clip(np.abs(w[m, 1]) / HALF_SPAN, 0, 1))
    xc = (w[m, 0] - (XQC - 0.25 * c_loc)) / c_loc
    zc = np.interp(np.clip(xc, 0, 1), SC._CAM_X, SC._CAM_Z) * c_loc
    up = w[m, 2] >= zc
    o = np.argsort(xc[up])
    sl = dict(xc=xc[up][o], chi=chi[m][up][o])

    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(f'{HERE}/{case}/{surf}')
    r.Update()
    g = r.GetOutput()
    p = vtk_to_numpy(g.GetPoints().GetData())
    cp = vtk_to_numpy(g.GetPointData().GetArray('Cp'))
    m2 = np.abs(p[:, 1] - y0) < band
    c2 = chord(np.clip(np.abs(p[m2, 1]) / HALF_SPAN, 0, 1))
    xc2 = (p[m2, 0] - (XQC - 0.25 * c2)) / c2
    up2 = p[m2, 2] >= np.interp(np.clip(xc2, 0, 1), SC._CAM_X, SC._CAM_Z) * c2
    o2 = np.argsort(xc2[up2])
    sl['xc_cp'], sl['cp'] = xc2[up2][o2], cp[m2][up2][o2]
    return sl


def section_coords(eta):
    """Clean unit-chord Selig loop (TE->upper->LE->lower->TE) from the family."""
    from wing_geometry import SectionFamily
    f = SectionFamily(128)
    zu, zl = f._blend(eta)
    xs = f.xs
    x = np.concatenate([xs[::-1], xs[1:]])
    z = np.concatenate([zu[::-1], zl[1:]])
    # mfoil derives the wake direction from the TE gap vector; an exactly
    # sharp TE makes it zero. Open a standard tiny blunt TE.
    z[0] += 2.5e-4
    z[-1] -= 2.5e-4
    return np.vstack([x, z])


def _solve_alpha(coords, alpha, Re, ncrit):
    for attempt in range(2):
        m = MF.mfoil(coords=coords)
        m.setoper(alpha=alpha, Re=Re, Ma=0.1)
        m.param.ncrit = ncrit
        try:
            with open(os.devnull, 'w') as nf:
                with redirect_stdout(nf), redirect_stderr(nf):
                    m.solve()
            return m
        except AssertionError:
            coords = coords[:, ::-1]
    raise RuntimeError('mfoil failed both windings')


def run_mfoil(coords, cl_target, Re, ncrit):
    # the cl-spec path in this mfoil copy is broken (ueinVref typo);
    # secant-iterate alpha to the target cl instead
    a = 4.0
    m = _solve_alpha(coords, a, Re, ncrit)
    for _ in range(4):
        if abs(m.post.cl - cl_target) < 0.005:
            break
        a += (cl_target - m.post.cl) / 0.11
        m = _solve_alpha(coords, a, Re, ncrit)
    Is = np.unique(np.concatenate([np.asarray(m.vsol.Is[0]),
                                   np.asarray(m.vsol.Is[1])]))
    x, z = m.foil.x[0][Is], m.foil.x[1][Is]
    n = np.where(m.vsol.turb[Is], np.nan, m.glob.U[2, Is])
    cp = m.post.cp[Is]
    zc = 0.5 * (np.interp(x, *_side(coords, 'up')) +
                np.interp(x, *_side(coords, 'lo')))
    up = z > zc
    o = np.argsort(x[up])
    return dict(x=x[up][o], n=n[up][o], cp=cp[up][o],
                xtr=float(m.vsol.Xt[1, 1]), cl=float(m.post.cl),
                alpha=float(m.oper.alpha), conv=bool(m.glob.conv))


def _side(coords, which):
    x, z = coords
    le = int(np.argmin(x))
    if which == 'up':
        return x[:le + 1][::-1], z[:le + 1][::-1]
    return x[le:], z[le:]


if __name__ == '__main__':
    y0 = ETA * HALF_SPAN
    Re_loc = RE_ROOT * chord(ETA) / C_ROOT

    # local cl from the RANS native strips (apples-to-apples target for mfoil)
    SC.ALPHA = np.deg2rad(ALPHA)
    e_n, cl_n, _ = SC.native_strips(next(v[0] for k, v in CASES.items()
                                         if 'ogrid' in k))
    cl_loc = float(np.interp(ETA, e_n, cl_n))

    coords = section_coords(ETA)
    print(f'eta {ETA} (y={y0:.3f}), alpha {ALPHA}, Re {Re_loc:.3g}, '
          f'target cl {cl_loc:.3f}', flush=True)

    mf = run_mfoil(coords, cl_loc, Re_loc, NCRIT_MACK)
    print(f"mfoil: conv={mf['conv']} cl={mf['cl']:.3f} alpha={mf['alpha']:.2f} "
          f"xtr_up={mf['xtr']:.3f}", flush=True)

    slices = {k: section_slice(c, s, y0, 0.02 if 'ogrid' in k else 0.11)
              for k, (c, s) in CASES.items()}

    def style(k):
        return dict(ls='-' if 'ogrid' in k else '--',
                    lw=2.0 if 'L2' in k else 1.3,
                    color='C0' if 'L2' in k else 'C2')

    fig, (axc, ax) = plt.subplots(2, 1, figsize=(9, 8.5), sharex=True,
                                  gridspec_kw={'height_ratios': [1, 1.6]})
    for k in CASES:
        axc.plot(slices[k]['xc_cp'], -slices[k]['cp'], **style(k),
                 label=f'RANS {k}')
    axc.plot(mf['x'], -mf['cp'], 'k:', lw=1.6, label='mfoil')
    axc.set_ylabel('$-C_p$ (upper)')
    axc.grid(alpha=0.3)
    axc.legend(fontsize=8)

    for k in CASES:
        ax.semilogy(slices[k]['xc'], np.clip(slices[k]['chi'], 1e-8, None),
                    **style(k), label=f'SA-AI {k} $\\chi$')
    ax.axhline(1.0, color='C3', lw=0.8, ls=':')
    ax.set_ylim(1e-6, 1e2)
    ax.set_ylabel(r'$\max_n \chi=\tilde\nu/\nu$')
    ax.set_xlabel('$x/c$')
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    lo, hi = np.log(1e-6 / CHI_INF), np.log(1e2 / CHI_INF)
    ax2.set_ylim(lo, hi)
    ax2.plot(mf['x'], mf['n'], 'k:', lw=1.8, label='mfoil $N$')
    ax2.axhline(NCRIT_MACK, color='k', lw=0.8, ls='--')
    ax2.text(0.02, NCRIT_MACK + 0.2, r'$N_\mathrm{crit}=13.6$', fontsize=8)
    ax2.axvline(mf['xtr'], color='k', lw=1.0, ls='-.')
    ax2.set_ylabel(r'$N=\ln(\chi/\chi_\infty)$')
    ax.legend(fontsize=8, loc='upper left')

    # slope diagnosis over the common laminar-growth window
    for k in CASES:
        s = slices[k]
        m = (s['chi'] > 1e-4) & (s['chi'] < 1e-1) & (s['xc'] > 0.05)
        if m.sum() > 3:
            p = np.polyfit(s['xc'][m], np.log(s['chi'][m]), 1)
            print(f'SA-AI {k}: dN/d(x/c) = {p[0]:.1f} over chi 1e-4..1e-1',
                  flush=True)
    mn = np.isfinite(mf['n'])
    mm = mn & (mf['x'] > 0.05) & (mf['n'] > 2) & (mf['n'] < 12)
    if mm.sum() > 3:
        p = np.polyfit(mf['x'][mm], mf['n'][mm], 1)
        print(f'mfoil:  dN/d(x/c) = {p[0]:.1f} over N 2..12', flush=True)

    fig.suptitle(f'Daedalus section $\\eta={ETA}$, $\\alpha={ALPHA:.0f}^\\circ$, '
                 f'$Re={Re_loc:.2g}$, $c_l={cl_loc:.2f}$: SA-AI amplification vs '
                 f'mfoil $e^N$', fontsize=11)
    out = f'{HERE}/section_chi_eta{int(ETA*100):03d}_a{int(ALPHA)}.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    fig.savefig(out[:-4] + '.pdf', bbox_inches='tight')
    print('wrote', out, '+pdf', flush=True)
