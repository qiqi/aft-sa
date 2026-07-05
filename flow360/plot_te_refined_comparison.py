"""Compare cavity L0/L1/L2 vs TE-refined cavity L0_TE/L1_TE/L2_TE.
Output: (a) Cp/Cf upper+lower, (b) force convergence overlay, (c) TE mesh viz.
"""
import numpy as np, vtk, csv, os, json
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

F = "/home/qiqi/flexcompute/aft-sa/flow360"

def contour_walk_both(d, af='naca0012'):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{d}/surface_fluid_{af}.pvtu"); r.Update()
    g = r.GetOutput(); p = vtk_to_numpy(g.GetPoints().GetData()); pd = g.GetPointData()
    arrs = {pd.GetArrayName(i): vtk_to_numpy(pd.GetArray(i)) for i in range(pd.GetNumberOfArrays())}
    cf = arrs['Cf']; cf = np.linalg.norm(cf, axis=1) if cf.ndim > 1 else cf
    cp = arrs['Cp']; cp = cp if cp.ndim == 1 else cp[:,0]
    x, y, z = p[:,0], p[:,1], p[:,2]
    s = np.abs(y - y.min()) < 1e-6
    X, Z, CF, CP = x[s], z[s], cf[s], cp[s]
    n = len(X); pts = np.column_stack([X, Z])
    st = int(np.argmin(X)); o = [st]; u = np.zeros(n, bool); u[st] = True
    for _ in range(n-1):
        c = o[-1]; dd = np.sum((pts - pts[c])**2, 1); dd[u] = 1e9
        nx = int(np.argmin(dd)); o.append(nx); u[nx] = True
    o = np.array(o); xo, zo, cfo, cpo = X[o], Z[o], CF[o], CP[o]
    te = int(np.argmax(xo)); b1, b2 = slice(0, te+1), slice(te, n)
    up, lo = (b1, b2) if zo[b1].mean() >= zo[b2].mean() else (b2, b1)
    def srt(sl):
        xs = xo[sl]; oo = np.argsort(xs)
        return xs[oo], cfo[sl][oo], cpo[sl][oo]
    return srt(up), srt(lo)

def forces(d):
    tr = list(csv.reader(open(f"{d}/total_forces_v2.csv")))
    th = [x.strip() for x in tr[0]]
    last = [x.strip() for x in tr[-1] if x.strip() != '']
    g = lambda k: float(last[th.index(k)])
    return g('CL'), g('CD'), g('CMy')

def ncell(d):
    return int(json.load(open(f"{d}/mesh.cgns.json"))['nNodes'])

cases = [
    ('proper_cav_L0',    'cav L0', 'k',  '-'),
    ('proper_cav_L1',    'cav L1', 'C0', '-'),
    ('proper_cav_L2',    'cav L2', 'C2', '-'),
    ('proper_cav_L3',    'cav L3', 'C1', '-'),
    ('proper_cav_L0_TE', 'cav L0 (TE-refined)', 'k',  '--'),
    ('proper_cav_L1_TE', 'cav L1 (TE-refined)', 'C0', '--'),
    ('proper_cav_L2_TE', 'cav L2 (TE-refined)', 'C2', '--'),
]

# (a) force convergence
fig1, axs1 = plt.subplots(1, 3, figsize=(13, 4))
rows = []
for d, lbl, c, ls in cases:
    full = f"{F}/{d}"
    if not os.path.exists(f"{full}/total_forces_v2.csv"):
        print(f"SKIP {d}"); continue
    CL, CD, CMy = forces(full); nc = ncell(full)
    h = 1/np.sqrt(nc)
    rows.append((d, lbl, c, ls, h, nc, CL, CD, CMy))

# split original vs TE-refined
orig = [r for r in rows if 'TE' not in r[0]]
te_r = [r for r in rows if 'TE' in r[0]]
orig.sort(key=lambda r: r[4]); te_r.sort(key=lambda r: r[4])
for ax, idx, ylab, title in [(axs1[0], 6, '$C_L$', '(a) lift'),
                              (axs1[1], 7, '$C_D$', '(b) drag'),
                              (axs1[2], 8, '$C_{M,y}$', '(c) pitching moment')]:
    x_o = [r[4] for r in orig]; v_o = [r[idx] for r in orig]
    x_t = [r[4] for r in te_r]; v_t = [r[idx] for r in te_r]
    ax.plot(x_o, v_o, 'ko-', ms=6, lw=1.3, label='original')
    ax.plot(x_t, v_t, 's--', color='C3', ms=6, lw=1.3, label='TE-refined ($h_{TE}=5h_0$)')
    ax.set_xlabel(r'$h\sim 1/\sqrt{N_{\rm nodes}}$'); ax.set_ylabel(ylab); ax.set_title(title, fontsize=10)
    ax.set_xlim(0, None); ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=9)
plt.tight_layout(pad=0.5)
plt.savefig('/tmp/te_refined_convergence.png', dpi=140)
plt.savefig('/home/qiqi/flexcompute/aft-sa/paper/figs/te_refined_convergence.pdf')
print('wrote /tmp/te_refined_convergence.png + paper figs/te_refined_convergence.pdf')

print('\n=== L0/L1/L2 original vs TE-refined ===')
print(f'{"case":>22s}  {"CL":>8s}  {"CD":>9s}  {"CMy":>8s}  {"N":>9s}')
for d, lbl, c, ls, h, nc, CL, CD, CMy in rows:
    print(f'  {d:20s}  {CL:.4f}  {CD:.5f}  {CMy:+.4f}  {nc:9d}')

# (b) Cp/Cf upper+lower comparison at L1 (the problem case)
fig2, axs2 = plt.subplots(2, 2, figsize=(13, 8))
for d, lbl, c, ls in [('proper_cav_L1', 'cav L1 (orig)', 'C0', '-'),
                       ('proper_cav_L1_TE', 'cav L1 (TE-refined)', 'C3', '-'),
                       ('proper_cav_L2', 'cav L2 (orig, ref)', 'k', '--')]:
    full = f"{F}/{d}"
    if not os.path.exists(f"{full}/surface_fluid_naca0012.pvtu"): continue
    (xu, cfu, cpu), (xl, cfl, cpl) = contour_walk_both(full)
    axs2[0,0].plot(xu, -cpu, color=c, ls=ls, lw=1.4, label=lbl)
    axs2[0,1].plot(xl, -cpl, color=c, ls=ls, lw=1.4, label=lbl)
    axs2[1,0].plot(xu, cfu, color=c, ls=ls, lw=1.4, label=lbl)
    axs2[1,1].plot(xl, cfl, color=c, ls=ls, lw=1.4, label=lbl)
for ax, title in [(axs2[0,0], '(a) $-C_p$ upper'),
                  (axs2[0,1], '(b) $-C_p$ lower'),
                  (axs2[1,0], '(c) $C_f$ upper'),
                  (axs2[1,1], '(d) $C_f$ lower')]:
    ax.set_xlabel('$x/c$'); ax.set_xlim(0, 1); ax.grid(alpha=0.3); ax.set_title(title, fontsize=10)
axs2[0,0].set_ylabel('$-C_p$'); axs2[0,1].set_ylabel('$-C_p$')
axs2[1,0].set_ylabel('$C_f$'); axs2[1,1].set_ylabel('$C_f$')
axs2[1,0].set_ylim(0, 0.012); axs2[1,1].set_ylim(0, 0.012)
axs2[0,0].legend(fontsize=9, frameon=False)
plt.tight_layout(pad=0.5)
plt.savefig('/tmp/te_refined_cp_cf_L1.png', dpi=140)
plt.savefig('/home/qiqi/flexcompute/aft-sa/paper/figs/te_refined_cp_cf_L1.pdf')
print('wrote /tmp/te_refined_cp_cf_L1.png + paper figs/te_refined_cp_cf_L1.pdf')

# (c) TE mesh viz: original L1 vs TE-refined L1
def load_slice_cells(d):
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(f"{d}/slice_centerSpan.pvtu"); r.Update()
    g = r.GetOutput(); pts = vtk_to_numpy(g.GetPoints().GetData())
    pts2 = pts[:, [0, 2]]
    edges = set()
    for i in range(g.GetNumberOfCells()):
        cell = g.GetCell(i); npts = cell.GetNumberOfPoints()
        ids = [cell.GetPointId(j) for j in range(npts)]
        for j in range(npts):
            a, b = ids[j], ids[(j+1) % npts]
            edges.add((min(a,b), max(a,b)))
    edge_arr = np.array(list(edges), dtype=int)
    return pts2, pts2[edge_arr]

zooms = [('te_wide', (0.85, 1.05), (-0.05, 0.05)),
         ('te_close', (0.98, 1.02), (-0.01, 0.01)),
         ('te_micro', (0.995, 1.005), (-0.002, 0.002))]
te_cases = [('proper_cav_L1', 'cav L1 (orig)'),
            ('proper_cav_L1_TE', 'cav L1 (TE-refined)')]

fig3, axs3 = plt.subplots(len(zooms), len(te_cases), figsize=(10, 11))
for col, (d, lbl) in enumerate(te_cases):
    full = f"{F}/{d}"
    if not os.path.exists(f"{full}/slice_centerSpan.pvtu"):
        print(f"SKIP {d}"); continue
    pts2, segs = load_slice_cells(full)
    for row, (zname, xlim, zlim) in enumerate(zooms):
        ax = axs3[row, col]
        in_box = ((segs[:,:,0] >= xlim[0]-0.05) & (segs[:,:,0] <= xlim[1]+0.05) &
                  (segs[:,:,1] >= zlim[0]-0.02) & (segs[:,:,1] <= zlim[1]+0.02))
        keep = in_box.any(axis=1)
        lc = LineCollection(segs[keep], colors='k', linewidths=0.3, rasterized=True)
        ax.add_collection(lc)
        ax.set_xlim(*xlim); ax.set_ylim(*zlim); ax.set_aspect('equal')
        ax.grid(alpha=0.2, lw=0.3); ax.tick_params(labelsize=7)
        if row == 0: ax.set_title(lbl, fontsize=10)
        if col == 0: ax.set_ylabel(f"{zname}\nz/c", fontsize=9)
        if row == len(zooms)-1: ax.set_xlabel("x/c", fontsize=9)
plt.tight_layout(pad=0.5)
plt.savefig('/tmp/te_refined_mesh.png', dpi=140)
plt.savefig('/home/qiqi/flexcompute/aft-sa/paper/figs/te_refined_mesh.pdf', dpi=200)
print('wrote /tmp/te_refined_mesh.png + paper figs/te_refined_mesh.pdf')
