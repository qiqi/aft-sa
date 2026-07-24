"""Appendix contour sheets -> paper/figs/chi_sheet_<af>_a<alpha>_<side>.pdf.

One sheet per (airfoil case, surface): 6 rows = the six grids paired by level
(cavity L0, O-grid L0, cavity L1, O-grid L1, cavity L2, O-grid L2), 2 columns
in the flat-plate figure's style -- left: line contours of velocity magnitude
|u|/U_inf; right: line contours of log10(chi), dashed = laminar levels
(chi < 1), solid = chi = 1 and c_v1. The x axis is the x/c of the wall anchor;
the y axis is wall-normal distance from that anchor (each wall-normal probe
scan is one vertical line of the sheet); the zoom holds the laminar band in
frame and lets the turbulent part overshoot.

chi = nuHat * Re/M: the center-span slice's nuHat is a*L-normalized
(freestream check: chi_inf*M/Re exactly). Model constant c_v1 imported.
"""
import os
import sys
import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from lib.calibrate_kernel import C_V1  # model constant: import, never restate

B = os.environ.get("SAAI_CFD_ROOT", "/home/qiqi/flexcompute/sa-ai/flow360_tie")
FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'figs')
MACH = 0.1

AF_SETUP = {
    'nlf0416':   dict(casetag='nlf0416_Re4M',    Re=4e6, L_up=0.0015, L_lo=0.0025),
    'eppler387': dict(casetag='eppler387_Re200k', Re=2e5, L_up=0.0067, L_lo=0.0112),
}
ROWS = [('cavL0prop', 'cavity L0'), ('strL0prop', 'O-grid L0'),
        ('cavL1prop', 'cavity L1'), ('strL1prop', 'O-grid L1'),
        ('cavL2prop', 'cavity L2'), ('strL2prop', 'O-grid L2')]
ALPHAS = {'nlf0416': (0, 4, 9, 15), 'eppler387': (0, 2, 5, 7)}

CHI_MAJOR = [-3, -2, -1, 0, np.log10(C_V1)]
CHI_MINOR = [-2.5, -1.5, -0.5]
CHI_FMT = {-3: '-3', -2: '-2', -1: '-1', 0: r'$\chi{=}1$', np.log10(C_V1): r'$c_{v1}$'}
U_LEV = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.1, 1.3]
U_LABEL = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.1, 1.3]


def _mod(af):
    if af == 'nlf0416':
        import regen_nlf_v2 as m
    else:
        import regen_eppler_v2 as m
    return m


def scan(m, af, case_d, side, L_probe, n_probe=140):
    """(x_anchors, dists, chi[n,M], umag[n,M]) probed along outward normals."""
    Xm, Zm, up_idx, lo_idx = m.walk_contour_xz(case_d)
    idx = up_idx if side == 'upper' else lo_idx
    xs = Xm[idx]; zs = Zm[idx]
    tx_raw = np.gradient(xs); tz_raw = np.gradient(zs)
    s = np.sqrt(tx_raw**2 + tz_raw**2) + 1e-30
    tx, tz = tx_raw/s, tz_raw/s
    nx, nz = tz, -tx
    if side == 'upper' and np.mean(nz) < 0:
        nx, nz = -nx, -nz
    elif side == 'lower' and np.mean(nz) > 0:
        nx, nz = -nx, -nz
    M = len(xs)
    dists = np.linspace(1e-6, L_probe, n_probe)
    y0 = m.slice_y_plane(case_d)
    pts = np.empty((M*n_probe, 3))
    for j, d in enumerate(dists):
        pts[j*M:(j+1)*M, 0] = xs + d*nx
        pts[j*M:(j+1)*M, 1] = y0
        pts[j*M:(j+1)*M, 2] = zs + d*nz
    vp = vtk.vtkPoints(); vp.SetData(numpy_to_vtk(pts, deep=True))
    poly = vtk.vtkPolyData(); poly.SetPoints(vp)
    g, _, _ = m.load_slice(case_d)
    pr = vtk.vtkProbeFilter(); pr.SetInputData(poly); pr.SetSourceData(g); pr.Update()
    pdd = pr.GetOutput().GetPointData()
    Re = AF_SETUP[af]['Re']
    nu = vtk_to_numpy(pdd.GetArray('nuHat')) * (Re/MACH)
    vel = vtk_to_numpy(pdd.GetArray('velocity'))
    umag = np.linalg.norm(vel, axis=1) / MACH
    valid = vtk_to_numpy(pr.GetValidPoints())
    mask = np.zeros(M*n_probe, bool); mask[valid] = True
    chi = np.where(mask, nu, np.nan).reshape(n_probe, M)
    um = np.where(mask, umag, np.nan).reshape(n_probe, M)
    o = np.argsort(xs)
    return xs[o], dists, chi[:, o], um[:, o]


def sheet(af, alpha, side):
    m = _mod(af)
    cfg = AF_SETUP[af]
    L = cfg['L_up'] if side == 'upper' else cfg['L_lo']
    Re = cfg['Re']                       # d*U_inf/nu = (d/c)*Re_c
    fig, axes = plt.subplots(6, 2, figsize=(11.5, 12.5), sharex=True, sharey=True)
    for r, (gname, glabel) in enumerate(ROWS):
        case = f"{B}/{gname}_{cfg['casetag']}_a{alpha}"
        if not os.path.exists(f"{case}/slice_centerSpan.pvtu"):
            # the current campaign tree tracks completion in the campaign
            # JSONs; the slice is written at the final step only
            raise FileNotFoundError(case)
        x, d, chi, um = scan(m, af, case, side, L)
        axU, axC = axes[r]
        cs = axU.contour(x, d*Re, um, levels=U_LEV, colors='k', linewidths=0.6)
        axU.clabel(cs, levels=U_LABEL, fmt='%g', fontsize=6.5, inline_spacing=2)
        logchi = np.log10(np.clip(chi, 1e-8, None))
        axC.contour(x, d*Re, logchi, levels=CHI_MINOR, colors='k', linewidths=0.4)
        cs = axC.contour(x, d*Re, logchi, levels=CHI_MAJOR, colors='k', linewidths=0.8)
        axC.clabel(cs, fmt=CHI_FMT, fontsize=6.5, inline_spacing=2)
        axU.set_xlim(0, 1); axU.set_ylim(0, L*Re)
        axU.set_ylabel(f'{glabel}\n' + r'$d\,U_\infty/\nu$', fontsize=9)
        print(f"  {gname}: scanned", flush=True)
    axes[0, 0].set_title(r'$|\mathbf{u}|/U_\infty$', fontsize=10)
    axes[0, 1].set_title(r'$\log_{10}\chi$', fontsize=10)
    for c in range(2):
        axes[-1, c].set_xlabel('wall-anchor x/c')
    out = os.path.join(FIGS, f'chi_sheet_{af}_a{alpha}_{side}.pdf')
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


SWEEP_RE = [('Re60k', 6e4, r'$6\times10^4$'), ('Re100k', 1e5, r'$10^5$'),
            ('Re300k', 3e5, r'$3\times10^5$'), ('Re460k', 4.6e5, r'$4.6\times10^5$')]


def _sweep_case(fam, lvl, retag):
    """fam in {cav, str}, lvl in {L0, L1, L2}; L1 keeps the legacy names."""
    if lvl == 'L1':
        return f"{B}/sweep_{'str_' if fam == 'str' else ''}{retag}_a5"
    return f"{B}/sweep_{fam}{lvl}_{retag}_a5"


def sweep_re_sheet(retag, Re, relabel, side):
    """One appendix sheet per sweep Reynolds number, in the alpha-sheet
    style: 6 rows = the six grids paired by level (Re = 2e5 is covered by
    the main-matrix sheets). At Re = 1e5 two extra rows carry the L2
    solutions warm-started from the converged 2e5 state (the reattaching
    branch of the bistable band, Sec. eppbistab): both branches plotted.
    Wall-normal range scales as 1/sqrt(Re) from the Re = 2e5 values."""
    import regen_eppler_v2 as m
    L_ref = AF_SETUP['eppler387']['L_up' if side == 'upper' else 'L_lo']
    L = L_ref*np.sqrt(2e5/Re)
    rows = [('cav', 'L0', 'cavity L0'), ('str', 'L0', 'O-grid L0'),
            ('cav', 'L1', 'cavity L1'), ('str', 'L1', 'O-grid L1'),
            ('cav', 'L2', 'cavity L2'), ('str', 'L2', 'O-grid L2')]
    forks = []
    if retag == 'Re100k':
        forks = [('cav', 'fork', 'cavity L2, warm start'),
                 ('str', 'fork', 'O-grid L2, warm start')]
    nrows = len(rows) + len(forks)
    fig, axes = plt.subplots(nrows, 2, figsize=(11.5, 2.1*nrows + 0.6),
                             sharex=True, sharey=True)
    for r, (fam, lvl, glabel) in enumerate(rows + forks):
        case = (_sweep_case(fam, lvl, retag) if lvl != 'fork'
                else f"{B}/ext_fork_{fam}L2_{retag}_a5")
        if not os.path.exists(f"{case}/slice_centerSpan.pvtu"):
            raise FileNotFoundError(case)
        x, d, chi, um = scan(m, 'eppler387', case, side, L)
        chi = chi*(Re/2e5)   # scan() scales nuHat by the benchmark Re
        axU, axC = axes[r]
        cs = axU.contour(x, d*Re, um, levels=U_LEV, colors='k', linewidths=0.6)
        axU.clabel(cs, levels=U_LABEL, fmt='%g', fontsize=6.5, inline_spacing=2)
        logchi = np.log10(np.clip(chi, 1e-8, None))
        axC.contour(x, d*Re, logchi, levels=CHI_MINOR, colors='k', linewidths=0.4)
        cs = axC.contour(x, d*Re, logchi, levels=CHI_MAJOR, colors='k', linewidths=0.8)
        axC.clabel(cs, fmt=CHI_FMT, fontsize=6.5, inline_spacing=2)
        axU.set_xlim(0, 1); axU.set_ylim(0, L*Re)
        axU.set_ylabel(f'{glabel}\n' + r'$d\,U_\infty/\nu$', fontsize=9)
        print(f"  {retag} {glabel}: scanned", flush=True)
    axes[0, 0].set_title(r'$|\mathbf{u}|/U_\infty$', fontsize=10)
    axes[0, 1].set_title(r'$\log_{10}\chi$', fontsize=10)
    for c in range(2):
        axes[-1, c].set_xlabel('wall-anchor x/c')
    out = os.path.join(FIGS, f'chi_sheet_eppler387_{retag}_{side}.pdf')
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main(only=None):
    for af in ('nlf0416', 'eppler387'):
        for alpha in ALPHAS[af]:
            for side in ('upper', 'lower'):
                if only and (af, alpha, side) != only:
                    continue
                try:
                    print(f"{af} a{alpha} {side}:", flush=True)
                    sheet(af, alpha, side)
                except FileNotFoundError as e:
                    print(f"  SKIP (case incomplete): {e}", flush=True)
    if only:
        return
    for retag, Re, relabel in SWEEP_RE:
        for side in ('upper', 'lower'):
            try:
                print(f"eppler387 {retag} {side}:", flush=True)
                sweep_re_sheet(retag, Re, relabel, side)
            except FileNotFoundError as e:
                print(f"  SKIP (case incomplete): {e}", flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 3:
        main(only=(sys.argv[1], int(sys.argv[2]), sys.argv[3]))
    else:
        main()
