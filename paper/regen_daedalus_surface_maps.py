"""figs/daedalus_surface_a{4,5,6}.pdf -- one page-filling figure per incidence.

3x3 panels, upper surface only, chord (physical x) horizontal and span (y)
vertical: rows = structured O-grid L2, unstructured L2, e^9 strip theory
(FlexFoil at AVL local cl / local Re / N_crit=13.6, from
flow360_ai/flexfoil_daedalus_strips.pkl); columns = |C_f| (left),
-Cp (left), streamwise C_fx (middle; the bubble is bounded by the bold
C_fx=0 contour), and the near-wall amplification max chi (right, decade
contours, bold at the transition front). The e^9 row's Cp comes from
Karman-Tsien-corrected u_e and its amplification is chi_inf e^N."""
import os
import pickle
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import vtk
from vtk.util.numpy_support import vtk_to_numpy

D = '/home/qiqi/flexcompute/sa-ai/scripts/daedalus'
sys.path.insert(0, D)
import sectional_compare as SC                      # noqa: E402
from wing_geometry import chord, HALF_SPAN, XQC     # noqa: E402
from chi_surface_map import tri_faces               # noqa: E402

CHI_INF = 8.76e-6
NCRIT = 13.6
STRIPS = pickle.load(open(
    '/home/qiqi/flexcompute/sa-ai/flow360_ai/flexfoil_daedalus_strips.pkl', 'rb'))
CASES = {4.0: ('case_ogrid_L2_saai_a4', 'case_cavity_L2_saai_a4'),
         5.0: ('case_ogrid_L2_saai_a5', 'case_cavity_L2_saai_a5'),
         6.0: ('case_ogrid_L2_saai_a6', 'case_cavity_L2_saai_a6')}
SURF = {'str': 'surface_fluid_wing.pvtu', 'cav': 'surface_farfield_body.pvtu'}
ROW_LAB = {'str': 'structured O-grid L2', 'cav': 'unstructured L2',
           'e9': r'$e^N$ strips (FlexFoil, AVL $c_l$)'}

LEV_CP = [-1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5]
LEV_CFX = [-0.002, -0.001, -0.0005, 0.0, 0.001, 0.002, 0.004, 0.006, 0.010]
LEV_CHI = list(range(-5, 3))            # log10 chi decades


def rans_row(case, surf):
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(f'{D}/{case}/{surf}')
    r.Update()
    g = r.GetOutput()
    p = vtk_to_numpy(g.GetPoints().GetData())
    cfv = vtk_to_numpy(g.GetPointData().GetArray('CfVec'))
    cp = vtk_to_numpy(g.GetPointData().GetArray('Cp'))
    tris = tri_faces(g)
    c_loc = chord(np.clip(np.abs(p[:, 1]) / HALF_SPAN, 0, 1))
    xc = np.clip((p[:, 0] - (XQC - 0.25 * c_loc)) / c_loc, 0, 1)
    up = p[:, 2] >= np.interp(xc, SC._CAM_X, SC._CAM_Z) * c_loc
    keep = up[tris].all(axis=1)
    tri = mtri.Triangulation(p[:, 0], p[:, 1], tris[keep])
    d = np.load(f'{D}/{case}/chi_surface.npz')
    wall, chi = d['wall'], d['chi']
    # chi npz shares the same wall pvtu node order
    logchi = np.log10(np.clip(chi, 1e-8, None))
    return tri, -cp, cfv[:, 0], logchi


def e9_row(a):
    s = STRIPS[a]
    etas = np.asarray(s['eta'])
    xcg = np.linspace(0.002, 0.998, 240)
    CF = np.full((len(etas), len(xcg)), np.nan)
    CP = np.full((len(etas), len(xcg)), np.nan)
    N = np.full((len(etas), len(xcg)), np.nan)
    MACH = 0.1
    beta_kt = np.sqrt(1 - MACH**2)
    for i, st in enumerate(s['stations']):
        if st is None:
            continue
        CF[i] = np.interp(xcg, np.asarray(st['xc'], float),
                          np.asarray(st['cf'], float))
        ue = np.interp(xcg, np.asarray(st['xc'], float),
                       np.asarray(st['ue'], float))
        cpi = 1.0 - ue**2
        CP[i] = cpi / (beta_kt + MACH**2 / 2 * cpi / (1 + beta_kt))
        n = np.asarray(st['n'], float)
        m = np.isfinite(n)
        xcf = np.asarray(st['xc'], float)
        N[i] = np.interp(xcg, xcf[m], n[m]) if m.sum() > 2 else np.nan
        N[i][xcg > (xcf[m].max() if m.any() else 0)] = np.nan
    c = chord(etas)[:, None]
    X = (XQC - 0.25 * c) + xcg[None, :] * c
    Y = (etas * HALF_SPAN)[:, None] * np.ones_like(xcg)[None, :]
    logchi_eq = np.log10(CHI_INF) + N / np.log(10.0)
    return X, Y, -CP, CF, logchi_eq, N


def make_fig(a, out):
    fig, axs = plt.subplots(3, 3, figsize=(7.6, 9.4), sharex=True, sharey=True,
                            layout='constrained')

    def chi_panel_tri(ax, tri, logchi):
        cs = ax.tricontour(tri, logchi, levels=LEV_CHI, colors='k',
                           linewidths=0.5)
        ax.clabel(cs, LEV_CHI[::2], fmt=lambda v: f'$10^{{{int(v)}}}$',
                  fontsize=5.5, inline_spacing=1)
        ax.tricontour(tri, logchi, levels=[0.0], colors='k', linewidths=1.5)

    def cf_panel_tri(ax, tri, v, levels, bold_zero=False):
        cs = ax.tricontour(tri, v, levels=levels, colors='k', linewidths=0.5)
        fmt = '%.2f' if max(abs(l) for l in levels) > 0.1 else '%.3f'
        ax.clabel(cs, levels[::2], fmt=fmt, fontsize=5.5, inline_spacing=1)
        if bold_zero:
            ax.tricontour(tri, v, levels=[0.0], colors='k', linewidths=1.5)

    for row, fam in enumerate(('str', 'cav')):
        tri, cfm, cfx, logchi = rans_row(CASES[a][row], SURF[fam])
        cf_panel_tri(axs[row, 0], tri, cfm, [-l for l in LEV_CP[::-1]])
        cf_panel_tri(axs[row, 1], tri, cfx, LEV_CFX, bold_zero=True)
        chi_panel_tri(axs[row, 2], tri, logchi)
        axs[row, 0].set_ylabel(f'{ROW_LAB[fam]}\n$y$ [m]', fontsize=9)

    X, Y, cfm, cfx, logchi_eq, N = e9_row(a)
    ok = np.isfinite(cfm)
    levcp = [-l for l in LEV_CP[::-1]]
    cs = axs[2, 0].contour(X, Y, np.where(ok, cfm, np.nan), levels=levcp,
                           colors='k', linewidths=0.5)
    axs[2, 0].clabel(cs, levcp[::2], fmt='%.2f', fontsize=5.5)
    cs = axs[2, 1].contour(X, Y, np.where(ok, cfx, np.nan), levels=LEV_CFX,
                           colors='k', linewidths=0.5)
    axs[2, 1].clabel(cs, LEV_CFX[::2], fmt='%.3f', fontsize=5.5)
    axs[2, 1].contour(X, Y, np.where(ok, cfx, np.nan), levels=[0.0],
                      colors='k', linewidths=1.5)
    with np.errstate(invalid='ignore'):
        cs = axs[2, 2].contour(X, Y, logchi_eq, levels=LEV_CHI, colors='k',
                               linewidths=0.5)
        axs[2, 2].clabel(cs, LEV_CHI[::2], fmt=lambda v: f'$10^{{{int(v)}}}$',
                         fontsize=5.5)
    # bold transition line: N is NaN past transition, so the N_crit contour
    # sits on the NaN boundary and marching squares drops it -- draw the
    # per-station transition polyline explicitly instead
    s = STRIPS[a]
    xt, yt = [], []
    for i, st in enumerate(s['stations']):
        if st is None:
            continue
        n = np.asarray(st['n'], float)
        xcf = np.asarray(st['xc'], float)
        turb = ~np.isfinite(n)
        if turb.any() and np.isfinite(n).any():
            j = int(np.argmax(turb))
            if 0 < j < len(xcf):
                eta = float(s['eta'][i])
                cl_ = float(chord(eta))
                xt.append((XQC - 0.25 * cl_) + xcf[j] * cl_)
                yt.append(eta * HALF_SPAN)
    axs[2, 2].plot(xt, yt, 'k-', lw=1.5)
    axs[2, 0].set_ylabel(f'{ROW_LAB["e9"]}\n$y$ [m]', fontsize=9)

    # wing planform outline on every panel
    ee = np.linspace(0, 1, 200)
    xle = XQC - 0.25 * chord(ee)
    xte = XQC + 0.75 * chord(ee)
    for ax in axs.flat:
        ax.plot(xle, ee * HALF_SPAN, 'k-', lw=0.8)
        ax.plot(xte, ee * HALF_SPAN, 'k-', lw=0.8)
        ax.plot([xle[0], xte[0]], [0, 0], 'k-', lw=0.8)
        ax.plot([xle[-1], xte[-1]], [HALF_SPAN, HALF_SPAN], 'k-', lw=0.8)
        ax.set_xlim(-0.02, 0.95)
        ax.set_ylim(0, HALF_SPAN * 1.005)
    for ax in axs[2]:
        ax.set_xlabel('$x$ [m]')
    titles = [r'$-C_p$', r'$C_{f,x}$ (bold: $C_{f,x}=0$)',
              r'$\max_n\chi$ (bold: transition)']
    for c_, t in enumerate(titles):
        axs[0, c_].set_title(t, fontsize=10)
    fig.suptitle(f'Daedalus wing, upper surface, $\\alpha={a:.0f}^\\circ$',
                 fontsize=11)
    fig.savefig(out)
    print('wrote', out)


if __name__ == '__main__':
    for a in (4.0, 5.0, 6.0):
        make_fig(a, f'figs/daedalus_surface_a{int(a)}.pdf')
