"""LSB-zoom sheets for the Eppler 60k/100k upper surface (author request):
the appendix wall-anchored view, but (a) frame extended to cover the WHOLE
bubble, (b) SIGNED tangential velocity u_t/U_inf with contour levels
refined near 0 (negative levels dashed -- the recirculation itself), and
(c) chi contours extended above 1 (2, 4, c_v1, 15, 30, 60, 120) to watch
the handover evolve inside the bubble.

Also prints the reverse-layer momentum budget at selected stations:
integrating steady x-momentum across the near-stagnant backflow layer,
    tau_top - tau_wall ~ h * dp/dx   =>   Cf_top ~ Cf_wall + h * dCp/dx
-- if Cf_wall (strongly negative) and h*dCp/dx (recovery gradient) nearly
cancel, the bubble is in pressure-viscous balance and closure rests on the
shear-layer stress tau_top, which is nu_t-limited (chi ~ c_v1, f_v1 ~ 1/2).

-> figs_explore/lsb_zoom_Re{60,100}k.png + printed budget table.
"""
import os, sys
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _H)
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import regen_eppler_v2 as m

B = os.environ.get("SAAI_CFD_ROOT", "/home/qiqi/flexcompute/sa-ai/flow360_fv1")
PREV = os.path.abspath(os.path.join(_H, '..', 'analytic', 'figs_explore'))
MACH = 0.1
C_V1 = 7.1

UT_NEG = [-0.10, -0.05, -0.02, -0.01]
UT_POS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1]
CHI_DASH = [-3, -2, -1]
CHI_SOLID = [0, np.log10(2), np.log10(4), np.log10(C_V1), np.log10(15),
             np.log10(30), np.log10(60), np.log10(120)]
CHI_FMT = {0: '1', np.log10(2): '2', np.log10(4): '4',
           np.log10(C_V1): '$c_{v1}$', np.log10(15): '15',
           np.log10(30): '30', np.log10(60): '60', np.log10(120): '120'}


def scan_signed(case_d, Re, L_probe, n_probe=160):
    """(x, dists, chi, u_t/U_inf) along upper-surface outward normals,
    velocity projected on the LOCAL WALL TANGENT (signed: <0 = reversed)."""
    Xm, Zm, up_idx, _ = m.walk_contour_xz(case_d)
    xs, zs = Xm[up_idx], Zm[up_idx]
    o = np.argsort(xs); xs, zs = xs[o], zs[o]
    tx = np.gradient(xs); tz = np.gradient(zs)
    s = np.hypot(tx, tz) + 1e-30
    tx, tz = tx/s, tz/s                       # downstream tangent
    nx, nz = -tz, tx
    if np.mean(nz) < 0:
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
    nuh = vtk_to_numpy(pdd.GetArray('nuHat'))*(Re/MACH)
    vel = vtk_to_numpy(pdd.GetArray('velocity'))
    valid = vtk_to_numpy(pr.GetValidPoints())
    mask = np.zeros(M*n_probe, bool); mask[valid] = True
    TX = np.tile(tx, n_probe); TZ = np.tile(tz, n_probe)
    ut = (vel[:, 0]*TX + vel[:, 2]*TZ)/MACH
    chi = np.where(mask, nuh, np.nan).reshape(n_probe, M)
    ut = np.where(mask, ut, np.nan).reshape(n_probe, M)
    return xs, dists, chi, ut


def budget(case_d, Re, stations, xs, dists, ut):
    (xu, cfu, cpu), _ = m.airfoil_walk_contour(case_d)
    o = np.argsort(xu); xu, cfu, cpu = xu[o], cfu[o], cpu[o]
    cps = np.convolve(cpu, np.ones(21)/21, mode='same')     # smoothed Cp
    print(f"  {'x':>5} {'h_rev':>7} {'Cf_wall':>9} {'dCp/dx':>8} "
          f"{'h*dCp/dx':>9} {'Cf_top':>8}")
    for x0 in stations:
        i = np.argmin(np.abs(xs - x0))
        col = ut[:, i]
        neg = np.where(col < 0)[0]
        h = dists[neg[-1]] if len(neg) else 0.0
        cfw = float(np.interp(x0, xu, cfu))
        j = np.argmin(np.abs(xu - x0))
        w = 15
        dcpdx = float(np.polyfit(xu[j-w:j+w], cps[j-w:j+w], 1)[0])
        print(f"  {x0:5.2f} {h:7.4f} {cfw:9.5f} {dcpdx:8.3f} "
              f"{h*dcpdx:9.5f} {cfw + h*dcpdx:8.5f}")


def make(Rk, L_probe):
    cases = [(f'{B}/sweep_strL2_Re{Rk}k_a5', 'O-grid L2'),
             (f'{B}/sweep_cavL2_Re{Rk}k_a5', 'cavity L2')]
    Re = Rk*1000.0
    fig, axs = plt.subplots(len(cases), 2, figsize=(13, 4.1*len(cases)),
                            sharex=True)
    for r, (d, lab) in enumerate(cases):
        xs, dists, chi, ut = scan_signed(d, Re, L_probe)
        axU, axC = axs[r]
        x2, d2 = np.meshgrid(xs, dists)
        cs = axU.contour(x2, d2, ut, levels=UT_NEG, colors='r',
                         linewidths=0.8, linestyles='dashed')
        axU.clabel(cs, fmt='%g', fontsize=6, inline_spacing=1)
        cs = axU.contour(x2, d2, ut, levels=UT_POS, colors='k', linewidths=0.6)
        axU.clabel(cs, levels=[0.0, 0.05, 0.2, 0.6, 1.0], fmt='%g',
                   fontsize=6.5, inline_spacing=2)
        with np.errstate(invalid='ignore', divide='ignore'):
            lc = np.log10(np.maximum(chi, 1e-12))
        axC.contour(x2, d2, lc, levels=CHI_DASH, colors='k',
                    linewidths=0.5, linestyles='dashed')
        cs = axC.contour(x2, d2, lc, levels=CHI_SOLID, colors='k', linewidths=0.8)
        axC.clabel(cs, fmt=lambda v: CHI_FMT.get(min(CHI_FMT, key=lambda k: abs(k-v)), ''),
                   fontsize=6.5, inline_spacing=2)
        for ax in (axU, axC):
            ax.set_xlim(0.25, 1.0); ax.set_ylim(0, L_probe)
            ax.set_ylabel(f'{lab}\n$d/c$')
        axU.set_title('$u_t/U_\\infty$ (red dashed: reversed)' if r == 0 else '')
        axC.set_title(r'$\log_{10}\chi$' if r == 0 else '')
        print(f"-- Re={Rk}k {lab}: reverse-layer momentum budget")
        budget(d, Re, (0.55, 0.7, 0.85, 0.95), xs, dists, ut)
    for ax in axs[-1]:
        ax.set_xlabel('wall-anchor $x/c$')
    plt.tight_layout()
    out = f'{PREV}/lsb_zoom_Re{Rk}k.png'
    plt.savefig(out, dpi=140)
    print('wrote', out)


os.makedirs(PREV, exist_ok=True)
make(100, 0.09)
make(60, 0.13)
