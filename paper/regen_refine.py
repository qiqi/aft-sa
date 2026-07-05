"""Fig 7: grid-refinement at alpha=4, 5 levels per family. CD from run_fix_results.json;
xtr re-extracted robustly (excluding the leading-edge Cf spike, x<0.04)."""
import json, csv, os, numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 11, 'font.family': 'serif', 'axes.linewidth': 0.8})
B = "/home/qiqi/flexcompute/aft-sa/flow360"; PD = "/home/qiqi/flexcompute/aft-sa/paper"
r = json.load(open(f"{B}/run_fix_results.json"))
CAV = [('coarse', 35912), ('med', 69772), ('fine', 136288), ('xfine', 271564), ('xxfine', 537348)]
STR = [('coarse', 8791), ('med', 24651), ('fine', 63441), ('sxfine', 124657), ('sxxfine', 254881)]
def tband(d):  # transition BAND (x10, x50, x90) of the 10-90% Cf rise, contour-walk upper surface
    rr = vtk.vtkXMLPUnstructuredGridReader(); rr.SetFileName(f"{B}/{d}/surface_fluid_naca0012.pvtu"); rr.Update()
    g = rr.GetOutput(); p = vtk_to_numpy(g.GetPoints().GetData()); pd = g.GetPointData()
    nm = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    a = vtk_to_numpy(pd.GetArray(next(n for n in nm if n.lower().startswith('cf')))); v = np.linalg.norm(a, axis=1) if a.ndim > 1 else a
    x, y, z = p[:, 0], p[:, 1], p[:, 2]; s = np.abs(y - y.min()) < 1e-6; X, Z, CF = x[s], z[s], v[s]; n = len(X); pts = np.column_stack([X, Z])
    st = int(np.argmin(X)); o = [st]; u = np.zeros(n, bool); u[st] = True
    for _ in range(n - 1):
        c = o[-1]; dd = np.sum((pts - pts[c])**2, 1); dd[u] = 1e9; nx = int(np.argmin(dd)); o.append(nx); u[nx] = True
    o = np.array(o); xo, zo, cfo = X[o], Z[o], CF[o]; te = int(np.argmax(xo)); b1, b2 = slice(0, te+1), slice(te, n)
    up = b1 if zo[b1].mean() >= zo[b2].mean() else b2; xu, cu = xo[up], cfo[up]; oo = np.argsort(xu); xu, cu = xu[oo], cu[oo]
    k = np.ones(7)/7; cm = np.convolve(cu, k, 'same'); m = (xu > 0.03) & (xu < 0.9); xx, cc = xu[m], cm[m]
    dcf = np.diff(cc); dm = .5*(xx[1:] + xx[:-1]); w = (dm > 0.04) & (dm < 0.6)
    if not w.any() or dcf[w].max() <= 0: return (np.nan, np.nan, np.nan)
    x50 = dm[np.where(w)[0][int(np.argmax(dcf[w]))]]; cflam = cc[xx <= x50].min(); ipk = int(np.argmax(cc[xx >= x50])); cfturb = cc[xx >= x50][ipk]
    if cfturb <= cflam: return (np.nan, x50, np.nan)
    lo = cflam + 0.1*(cfturb - cflam); hi = cflam + 0.9*(cfturb - cflam)
    xdip = xx[xx <= x50][int(np.argmin(cc[xx <= x50]))]; xpk = xx[xx >= x50][ipk]
    rise = (xx >= xdip) & (xx <= xpk); xr, cr = xx[rise], cc[rise]
    cx = lambda l: (float(xr[np.where(cr >= l)[0][0]]) if (cr >= l).any() else np.nan)
    return (cx(lo), x50, cx(hi))
def cdof(d):
    rows = list(csv.reader(open(f"{B}/{d}/total_forces_v2.csv"))); hh = [x.strip() for x in rows[0]]
    last = [c for c in rows[-1] if c.strip() != '']; return float(last[hh.index('CD')])
def series(levels, dirpfx):  # CD + transition band, read from the (gated-max re-run) dirs
    h, CD, x10, x50, x90 = [], [], [], [], []
    for lv, nc in levels:
        d = f"{dirpfx}_{lv}"
        if os.path.exists(f"{B}/{d}/total_forces_v2.csv"):
            b = tband(d); h.append(nc**-0.5); CD.append(cdof(d)); x10.append(b[0]); x50.append(b[1]); x90.append(b[2])
    return tuple(np.array(z) for z in (h, CD, x10, x50, x90))
hc, CDc, c10, c50, c90 = series(CAV, 'proper_refineA4_cavity'); hs, CDs, s10, s50, s90 = series(STR, 'refineA4_struct')
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
axs[0].plot(hc, CDc, 'ko-', ms=5, lw=1.3, label='cavity (unstructured)')
axs[0].plot(hs, CDs, '^--', color='C3', ms=6, lw=1.3, label='O-grid (structured)')
axs[0].set_xlabel(r'$h \sim N_{\rm cell}^{-1/2}$'); axs[0].set_ylabel('$C_D$'); axs[0].set_xlim(0, None); axs[0].grid(alpha=0.3)
axs[0].legend(fontsize=7.5, frameon=False); axs[0].set_title(r'drag vs.\ mesh size, $\alpha=4^\circ$', fontsize=9.5)
axs[1].errorbar(hc, c50, yerr=[c50-c10, c90-c50], fmt='ko-', ms=5, lw=1.3, capsize=3, label='cavity')
axs[1].errorbar(hs, s50, yerr=[s50-s10, s90-s50], fmt='^--', color='C3', ms=6, lw=1.3, capsize=3, label='O-grid')
axs[1].set_xlabel(r'$h \sim N_{\rm cell}^{-1/2}$'); axs[1].set_ylabel(r'$x_{tr}/c$ (band: 10--90\% $C_f$ rise)'); axs[1].set_xlim(0, None); axs[1].set_ylim(0, 0.4); axs[1].grid(alpha=0.3)
axs[1].legend(fontsize=7.5, frameon=False); axs[1].set_title('transition band vs.\\ mesh size', fontsize=9.5)
plt.tight_layout(pad=0.4); plt.savefig(f"{PD}/figs/grid_convergence.pdf"); plt.savefig('/tmp/refine4.png', dpi=120)
print("cavity CD:", [round(c, 5) for c in CDc], "x50:", [round(x, 3) for x in c50], "width:", [round(b-a,3) for a,b in zip(c10,c90)])
print("struct CD:", [round(c, 5) for c in CDs], "x50:", [round(x, 3) for x in s50], "width:", [round(b-a,3) for a,b in zip(s10,s90)])
