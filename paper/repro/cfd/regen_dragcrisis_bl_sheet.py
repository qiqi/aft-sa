"""Cylinder boundary-layer sheet through the drag crisis
-> paper/figs/dragcrisis_bl_sheet.pdf

The field figure (dragcrisis_fields) shows the whole near-wake at every
Reynolds number, which is the right view for the global picture and the wrong
one for the crisis itself: everything that decides the crisis happens in a
boundary layer a few thousand viscous units thick, over a 60-degree arc, and at
Re_D = 10^6 that band is 0.3% of the diameter -- invisible at wake scale.

So this is the airfoil sheets' view, wrapped onto the cylinder: abscissa the
angle from the forward stagnation point (60-120 deg, which brackets laminar
separation and the transition front through the whole crisis), ordinate the
wall distance in viscous units d*U_inf/nu, one row per Reynolds number over the
six points that span the crisis (1.78e5 -> 3e6, subcritical through
supercritical). Columns are the tangential velocity -- negative inside the
reversed-flow region, exactly as u_x is on the airfoil sheets -- and log10(chi).

Because the ordinate is viscous units, the SAME y window is the same physical
band in wall variables at every Reynolds number, so the rows are directly
comparable even though the layer thins by an order of magnitude down the page.

Run:  python3 regen_dragcrisis_bl_sheet.py [--root ...]
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, '..', '..', 'figs'))
PREV = os.path.join(HERE, 'figs_explore')

# Type sizes in points. The sheet is 12.5 in wide and is placed at 0.52 of that width,
# so the contour labels are 5.7 pt and the tick labels 6.2 pt on the page.
FS_CLABEL, FS_LABEL, FS_TITLE, FS_TICK = 11, 11, 13, 12
plt.rcParams.update({'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK,
                     'axes.labelsize': FS_TITLE})

MACH = 0.1
SYS = ("/local_data/qiqi/sa-ai/dragcrisis_matrix/"
       "systematic_Tu0.2_summary.jsonl")
# The ladder points that span the crisis, subcritical to supercritical. The
# top of the window is 3e6 rather than 1.78e6 so the series carries one row
# past the crisis proper, where the front has settled ahead of separation.
RE_LO, RE_HI = 1.7e5, 3.1e6
CX, CR = 0.5, 0.5              # cylinder centre and radius, diameters
TH_LO, TH_HI = 60.0, 120.0     # degrees from the forward stagnation point
Y_TOP = 6000.0                 # wall distance in viscous units, d*U_inf/nu
NTH, ND = 361, 320
# same level conventions as the airfoil sheets
UT_LEV = [-0.4, -0.2, -0.1, -0.05, 0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
UT_LABEL = [-0.2, -0.05, 0.0, 0.2, 0.6, 1.0, 1.4]
CHI_MINOR = [-2.5, -1.5, -0.5]
CHI_MAJOR = [-3, -2, -1, 0, np.log10(7.1), np.log10(30.0), 2.0, 4.0]
CHI_FMT = {-3: '-3', -2: '-2', -1: '-1', 0: r'$\chi{=}1$',
           np.log10(7.1): r'$c_{v1}$', np.log10(30.0): '30',
           2.0: r'$10^2$', 4.0: r'$10^4$'}


def relabel(re):
    e = int(np.floor(np.log10(re) + 1e-9))
    m = re / 10.0 ** e
    if abs(m - 1.0) < 1e-6:
        return r"$10^{%d}$" % e
    return r"$%g\times10^{%d}$" % (round(m, 2), e)


def load_cases():
    up = [json.loads(l) for l in open(SYS) if l.strip()]
    up = [r for r in up if r["dir"] == "up" and RE_LO <= r["re"] <= RE_HI]
    up.sort(key=lambda r: r["re"])
    return [(r["case"], r["re"], r["Cd"]) for r in up]


def probe(case_dir, re_d):
    """(theta_deg, y_plus_like, ut/U_inf, log10 chi) on a wall-normal fan."""
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(os.path.join(case_dir, "slice_centerSpan.pvtu"))
    r.Update()
    g = r.GetOutput()
    y0 = vtk_to_numpy(g.GetPoints().GetData())[0, 1]
    th = np.radians(np.linspace(TH_LO, TH_HI, NTH))
    # d/D from the viscous-unit ceiling: d*U/nu = (d/D)*Re_D
    yv = np.linspace(0.0, Y_TOP, ND)
    dd = yv / re_d
    TH, DD = np.meshgrid(th, dd)
    # outward radial unit vector at angle TH from the forward stagnation point
    nx, nz = -np.cos(TH), np.sin(TH)
    X = CX + (CR + DD) * nx
    Z = 0.0 + (CR + DD) * nz
    pts = np.column_stack([X.ravel(), np.full(X.size, y0), Z.ravel()])
    vp = vtk.vtkPoints(); vp.SetData(numpy_to_vtk(pts, deep=True))
    poly = vtk.vtkPolyData(); poly.SetPoints(vp)
    pr = vtk.vtkProbeFilter(); pr.SetInputData(poly); pr.SetSourceData(g)
    pr.Update()
    pd = pr.GetOutput().GetPointData()
    ok = np.zeros(X.size, bool)
    ok[vtk_to_numpy(pr.GetValidPoints())] = True
    v = vtk_to_numpy(pd.GetArray("velocity")) / MACH
    # tangential component, positive with increasing theta
    tx, tz = np.sin(TH).ravel(), np.cos(TH).ravel()
    ut = v[:, 0] * tx + v[:, 2] * tz
    mu = json.load(open(os.path.join(case_dir, "Flow360.json"))
                   )["freestream"]["muRef"]
    chi = vtk_to_numpy(pd.GetArray("nuHat")) / mu
    ut = np.where(ok, ut, np.nan).reshape(ND, NTH)
    lchi = np.where(ok, np.log10(np.clip(chi, 1e-8, None)), np.nan
                    ).reshape(ND, NTH)
    return np.degrees(th), yv, ut, lchi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/local_data/qiqi/sa-ai/dragcrisis_matrix")
    args = ap.parse_args()
    cases = load_cases()
    print(f"  {len(cases)} cases across the crisis", flush=True)
    n = len(cases)
    # width-binding aspect, as for the airfoil sheets
    fig, axes = plt.subplots(n, 2, figsize=(12.5, 2.35 * n + 0.6),
                             sharex=True, sharey=True)
    if n == 1:
        axes = np.asarray([axes])

    def chifmt(L):
        return f"${10**L:.0f}$" if L >= 0 else f"$10^{{{int(L)}}}$"

    for row, (case, re_d, cd) in enumerate(cases):
        thd, yv, ut, lchi = probe(os.path.join(args.root, case), re_d)
        axU, axC = axes[row]
        cs = axU.contour(thd, yv, ut, levels=UT_LEV, colors="k", linewidths=0.6)
        axU.clabel(cs, levels=UT_LABEL, fmt="%g", fontsize=FS_CLABEL,
                   inline_spacing=2)
        axU.contour(thd, yv, ut, levels=[0.0], colors="k", linewidths=1.6)
        axC.contour(thd, yv, lchi, levels=CHI_MINOR, colors="k",
                    linewidths=0.4)
        cs = axC.contour(thd, yv, lchi, levels=CHI_MAJOR, colors="k",
                         linewidths=0.8)
        axC.clabel(cs, fmt=CHI_FMT, fontsize=FS_CLABEL, inline_spacing=2)
        for a in (axU, axC):
            a.set_xlim(TH_LO, TH_HI)
            a.set_ylim(0, Y_TOP)
            a.axhline(0.0, color="k", lw=1.1)
        axU.set_ylabel(rf"$Re_D=${relabel(re_d)}, $C_d={cd:.3f}$"
                       "\n" r"$d\,U_\infty/\nu$", fontsize=FS_LABEL)
        print(f"  {case}: probed", flush=True)
    axes[0, 0].set_title(r"$u_t/U_\infty$ (heavy: $u_t=0$)", fontsize=FS_TITLE)
    axes[0, 1].set_title(r"$\log_{10}\chi$", fontsize=FS_TITLE)
    for c in range(2):
        axes[-1, c].set_xlabel("angle from forward stagnation [deg]")
    plt.tight_layout()
    os.makedirs(PREV, exist_ok=True)
    out = os.path.join(OUT, "dragcrisis_bl_sheet.pdf")
    fig.savefig(out)
    fig.savefig(os.path.join(PREV, "dragcrisis_bl_sheet.png"), dpi=130)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
