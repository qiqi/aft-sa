"""Appendix contour sheets -> paper/figs/chi_sheet_<af>_a<alpha>_<side>.pdf.

One sheet per (airfoil case, surface): 6 rows = the six grids paired by level
(cavity L0, O-grid L0, cavity L1, O-grid L1, cavity L2, O-grid L2), 2 columns
in the flat-plate figure's style -- left: line contours of streamwise
velocity u_x/U_inf (negative levels mark reverse flow in the LSB);
right: line contours of log10(chi), laminar levels chi < 1 plus solid
chi = 1, c_v1, 30, and 100 (the last two track the handover's completion
and the turbulent interior). The x axis is the x/c of the wall anchor;
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

B = os.environ.get("SAAI_CFD_ROOT", "/home/qiqi/flexcompute/sa-ai/flow360_fv1")
FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'figs')
MACH = 0.1

AF_SETUP = {
    'nlf0416':   dict(casetag='nlf0416_Re4M',    Re=4e6, L_up=0.0015, L_lo=0.0025),
    'eppler387': dict(casetag='eppler387_Re200k', Re=2e5, L_up=0.0335, L_lo=0.0112),
    # eppler upper L_up raised 5x (0.0067 -> 0.0335) to show the transition
    # handover above the bubble shear layer (annotated request 2026-07-25)
}
ROWS = [('cavL0prop', 'unstructured L0'), ('strL0prop', 'O-grid L0'),
        ('cavL1prop', 'unstructured L1'), ('strL1prop', 'O-grid L1'),
        ('cavL2prop', 'unstructured L2'), ('strL2prop', 'O-grid L2')]
# Every incidence the campaign ran, on all six grids. The L0/L1 levels of the
# extension incidences were filled in on 2026-08-08, so each sheet is now the
# full six rows; sheet() still drops rows whose slice_centerSpan.pvtu is
# absent, so a partially-run alpha degrades rather than failing.
ALPHAS = {'nlf0416': (-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 9, 15),
          'eppler387': (-4, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8.5, 9)}
# The tree names the Eppler alpha=4 case 'a4x', not 'a4'.
TAG_OVERRIDE = {('eppler387', 4.0): 'a4x'}


def _alpha_tag(alpha, af=None):
    """Flow360 case-name alpha token: -2 -> am2, 8.5 -> a8p5, else a{int}."""
    over = TAG_OVERRIDE.get((af, float(alpha)))
    if over:
        return over
    if abs(alpha - 8.5) < 1e-9:
        return 'a8p5'
    if alpha < 0:
        return f'am{abs(int(round(alpha)))}'
    return f'a{int(round(alpha))}'

# levels beyond c_v1 show the HANDOVER dynamics: sigma_P is 97% complete
# and f_v1 90% by chi ~ 15; chi = 30 marks the effectively completed
# handover (both > 0.98) and chi ~ 100 the fully turbulent interior
CHI_MAJOR = [-3, -2, -1, 0, np.log10(C_V1), np.log10(30.0), 2.0]
CHI_MINOR = [-2.5, -1.5, -0.5]
CHI_FMT = {-3: '-3', -2: '-2', -1: '-1', 0: r'$\chi{=}1$', np.log10(C_V1): r'$c_{v1}$',
           np.log10(30.0): '30', 2.0: r'$10^2$'}
# Streamwise velocity: negative bands resolve the reverse-flow region of an LSB.
UX_LEV = [-0.05, -0.02, -0.01, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
          0.8, 0.9, 0.99, 1.1, 1.3]
UX_LABEL = [-0.05, -0.02, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.1, 1.3]
# Back-compat aliases (older call sites / explore scripts).
U_LEV, U_LABEL = UX_LEV, UX_LABEL


def _mod(af):
    if af == 'nlf0416':
        import regen_nlf_v2 as m
    else:
        import regen_eppler_v2 as m
    return m


def scan(m, af, case_d, side, L_probe, n_probe=320):
    """(x_anchors, dists, chi[n,M], ux[n,M]) probed along outward normals.

    ux is the lab-frame streamwise velocity / U_inf (= vel_x / Mach in the
    Flow360 nondimensionalisation), so reverse flow is negative.
    """
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
    ux = vel[:, 0] / MACH
    valid = vtk_to_numpy(pr.GetValidPoints())
    mask = np.zeros(M*n_probe, bool); mask[valid] = True
    chi = np.where(mask, nu, np.nan).reshape(n_probe, M)
    uxf = np.where(mask, ux, np.nan).reshape(n_probe, M)
    o = np.argsort(xs)
    return xs[o], dists, chi[:, o], uxf[:, o]


def sheet(af, alpha, side):
    m = _mod(af)
    cfg = AF_SETUP[af]
    L = cfg['L_up'] if side == 'upper' else cfg['L_lo']
    Re = cfg['Re']                       # d*U_inf/nu = (d/c)*Re_c
    atag = _alpha_tag(alpha, af)
    rows = []
    for gname, glabel in ROWS:
        case = f"{B}/{gname}_{cfg['casetag']}_{atag}"
        if os.path.exists(f"{case}/slice_centerSpan.pvtu"):
            rows.append((gname, glabel, case))
    if not rows:
        raise FileNotFoundError(
            f"{B}/{{cav,str}}*_{cfg['casetag']}_{atag}")
    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 2, figsize=(11.5, 2.1 * nrows + 0.6),
                             sharex=True, sharey=True)
    if nrows == 1:
        axes = np.asarray([axes])
    for r, (gname, glabel, case) in enumerate(rows):
        x, d, chi, ux = scan(m, af, case, side, L)
        axU, axC = axes[r]
        cs = axU.contour(x, d*Re, ux, levels=UX_LEV, colors='k', linewidths=0.6)
        axU.clabel(cs, levels=UX_LABEL, fmt='%g', fontsize=6.5, inline_spacing=2)
        logchi = np.log10(np.clip(chi, 1e-8, None))
        axC.contour(x, d*Re, logchi, levels=CHI_MINOR, colors='k', linewidths=0.4)
        cs = axC.contour(x, d*Re, logchi, levels=CHI_MAJOR, colors='k', linewidths=0.8)
        axC.clabel(cs, fmt=CHI_FMT, fontsize=6.5, inline_spacing=2)
        axU.set_xlim(0, 1); axU.set_ylim(0, L*Re)
        axU.set_ylabel(f'{glabel}\n' + r'$d\,U_\infty/\nu$', fontsize=9)
        print(f"  {gname}: scanned", flush=True)
    axes[0, 0].set_title(r'$u_x/U_\infty$', fontsize=10)
    axes[0, 1].set_title(r'$\log_{10}\chi$', fontsize=10)
    for c in range(2):
        axes[-1, c].set_xlabel('wall-anchor x/c')
    out = os.path.join(FIGS, f'chi_sheet_{af}_{atag}_{side}.pdf')
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def sheet_combined(af, alpha, y_half=None):
    """One sheet per incidence with BOTH surfaces in each pane.

    This is the only figure of the paper carrying the upper and lower surface
    of the same case, and as two separate pages the reader had to hold one in
    memory to compare it with the other. Here the pane is mirrored about the
    wall: the upper surface occupies y > 0 and the lower surface the same
    wall-normal probe reflected onto y < 0, so y = 0 IS the wall for both and
    the two layers are read against each other directly.

    Both sides are probed to the same depth (the upper surface's frame) so the
    mirror is symmetric; the axis is +-y_half in d*U_inf/nu.
    """
    m = _mod(af)
    cfg = AF_SETUP[af]
    Re = cfg['Re']
    L = cfg['L_up']                      # one probe depth for both sides
    if y_half is None:
        y_half = L*Re
    atag = _alpha_tag(alpha, af)
    rows = []
    for gname, glabel in ROWS:
        case = f"{B}/{gname}_{cfg['casetag']}_{atag}"
        if os.path.exists(f"{case}/slice_centerSpan.pvtu"):
            rows.append((gname, glabel, case))
    if not rows:
        raise FileNotFoundError(f"{B}/{{cav,str}}*_{cfg['casetag']}_{atag}")
    nrows = len(rows)
    # Aspect set so the PAGE WIDTH binds, not the page height: a LaTeX text
    # block is 6.5 x 9 in, so W/H must stay above ~0.77 or the height cap
    # shrinks the figure and leaves the margins unused. Six rows at 2.55 in
    # gives W/H = 0.79.
    fig, axes = plt.subplots(nrows, 2, figsize=(12.5, 2.55*nrows + 0.6),
                             sharex=True, sharey=True)
    if nrows == 1:
        axes = np.asarray([axes])
    for r, (gname, glabel, case) in enumerate(rows):
        axU, axC = axes[r]
        for side, sgn in (('upper', 1.0), ('lower', -1.0)):
            x, d, chi, ux = scan(m, af, case, side, L)
            y = sgn*d*Re
            cs = axU.contour(x, y, ux, levels=UX_LEV, colors='k',
                             linewidths=0.6)
            axU.clabel(cs, levels=UX_LABEL, fmt='%g', fontsize=6.5,
                       inline_spacing=2)
            logchi = np.log10(np.clip(chi, 1e-8, None))
            axC.contour(x, y, logchi, levels=CHI_MINOR, colors='k',
                        linewidths=0.4)
            cs = axC.contour(x, y, logchi, levels=CHI_MAJOR, colors='k',
                             linewidths=0.8)
            axC.clabel(cs, fmt=CHI_FMT, fontsize=6.5, inline_spacing=2)
        for a_ in (axU, axC):
            a_.axhline(0.0, color='k', lw=1.1)      # the shared wall
            a_.set_xlim(0, 1)
            a_.set_ylim(-y_half, y_half)
        axU.set_ylabel(glabel + '\n' + r'$d\,U_\infty/\nu$', fontsize=9)
        if r == 0:
            for a_ in (axU, axC):
                a_.text(0.012, 0.965, 'upper', transform=a_.transAxes,
                        fontsize=8, va='top')
                a_.text(0.012, 0.035, 'lower', transform=a_.transAxes,
                        fontsize=8, va='bottom')
        print(f"  {gname}: both surfaces scanned", flush=True)
    axes[0, 0].set_title(r'$u_x/U_\infty$', fontsize=10)
    axes[0, 1].set_title(r'$\log_{10}\chi$', fontsize=10)
    for c in range(2):
        axes[-1, c].set_xlabel('wall-anchor x/c')
    out = os.path.join(FIGS, f'chi_sheet_{af}_{atag}_both.pdf')
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


SWEEP_RE = [('Re60k', 6e4, r'$6\times10^4$'), ('Re100k', 1e5, r'$10^5$'),
            ('Re300k', 3e5, r'$3\times10^5$'), ('Re460k', 4.6e5, r'$4.6\times10^5$')]


def _sweep_case(fam, lvl, retag):
    """fam in {cav, str}, lvl in {L0, L1, L2}. The fv1 tree uses uniform
    names for all levels; the legacy L1 aliases remain as a fallback for
    the old tree."""
    uni = f"{B}/sweep_{fam}{lvl}_{retag}_a5"
    if lvl != 'L1' or os.path.isdir(uni):
        return uni
    return f"{B}/sweep_{'str_' if fam == 'str' else ''}{retag}_a5"


def sweep_re_sheet(retag, Re, relabel, side):
    """One appendix sheet per sweep Reynolds number, in the alpha-sheet
    style: 6 rows = the six grids paired by level (Re = 2e5 is covered by
    the main-matrix sheets). Every sheet of the sweep carries the SAME six
    rows -- the Re = 1e5 sheet used to append two more, the L2 solutions
    warm-started from the converged 2e5 state, but they coincided with the
    cold-start L2 rows above them (there is no second branch; Sec.
    eppbistab makes that case on the lift and drag numbers) and the extra
    height made the page an odd one out.
    Wall-normal range scales as 1/sqrt(Re) from the Re = 2e5 values."""
    import regen_eppler_v2 as m
    L_ref = AF_SETUP['eppler387']['L_up' if side == 'upper' else 'L_lo']
    L = L_ref*np.sqrt(2e5/Re)
    rows = [('cav', 'L0', 'unstructured L0'), ('str', 'L0', 'O-grid L0'),
            ('cav', 'L1', 'unstructured L1'), ('str', 'L1', 'O-grid L1'),
            ('cav', 'L2', 'unstructured L2'), ('str', 'L2', 'O-grid L2')]
    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 2, figsize=(11.5, 2.1*nrows + 0.6),
                             sharex=True, sharey=True)
    for r, (fam, lvl, glabel) in enumerate(rows):
        case = _sweep_case(fam, lvl, retag)
        if not os.path.exists(f"{case}/slice_centerSpan.pvtu"):
            raise FileNotFoundError(case)
        x, d, chi, ux = scan(m, 'eppler387', case, side, L)
        chi = chi*(Re/2e5)   # scan() scales nuHat by the benchmark Re
        axU, axC = axes[r]
        cs = axU.contour(x, d*Re, ux, levels=UX_LEV, colors='k', linewidths=0.6)
        axU.clabel(cs, levels=UX_LABEL, fmt='%g', fontsize=6.5, inline_spacing=2)
        logchi = np.log10(np.clip(chi, 1e-8, None))
        axC.contour(x, d*Re, logchi, levels=CHI_MINOR, colors='k', linewidths=0.4)
        cs = axC.contour(x, d*Re, logchi, levels=CHI_MAJOR, colors='k', linewidths=0.8)
        axC.clabel(cs, fmt=CHI_FMT, fontsize=6.5, inline_spacing=2)
        axU.set_xlim(0, 1); axU.set_ylim(0, L*Re)
        axU.set_ylabel(f'{glabel}\n' + r'$d\,U_\infty/\nu$', fontsize=9)
        print(f"  {retag} {glabel}: scanned", flush=True)
    axes[0, 0].set_title(r'$u_x/U_\infty$', fontsize=10)
    axes[0, 1].set_title(r'$\log_{10}\chi$', fontsize=10)
    for c in range(2):
        axes[-1, c].set_xlabel('wall-anchor x/c')
    out = os.path.join(FIGS, f'chi_sheet_eppler387_{retag}_{side}.pdf')
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main(only=None, afs=None, sides=None, do_sweep=True):
    afs = afs or ('nlf0416', 'eppler387')
    sides = sides or ('upper', 'lower')
    for af in afs:
        for alpha in ALPHAS[af]:
            for side in sides:
                if only and not (only[0] == af and only[2] == side
                                 and abs(only[1] - alpha) < 1e-9):
                    continue
                try:
                    print(f"{af} {_alpha_tag(alpha, af)} {side}:", flush=True)
                    sheet(af, alpha, side)
                except FileNotFoundError as e:
                    print(f"  SKIP (case incomplete): {e}", flush=True)
    if only:
        return
    if not do_sweep:
        return
    for retag, Re, relabel in SWEEP_RE:
        for side in sides:
            try:
                print(f"eppler387 {retag} {side}:", flush=True)
                sweep_re_sheet(retag, Re, relabel, side)
            except FileNotFoundError as e:
                print(f"  SKIP (case incomplete): {e}", flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 3:
        main(only=(sys.argv[1], float(sys.argv[2]), sys.argv[3]))
    else:
        main()
