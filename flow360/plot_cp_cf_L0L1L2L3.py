"""Cp+Cf upper & lower surface, plus CL/CD/CM grid convergence. L0-L3 cavity, L0-L2 O-grid."""
import numpy as np, vtk, csv, os
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt

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
    if zo[b1].mean() >= zo[b2].mean():
        up, lo = b1, b2
    else:
        up, lo = b2, b1
    def srt(sl):
        xs = xo[sl]; oo = np.argsort(xs)
        return xs[oo], cfo[sl][oo], cpo[sl][oo]
    return srt(up), srt(lo)

def transition_band(x, cf):
    k = np.ones(7)/7; cm = np.convolve(cf, k, 'same')
    m = (x > 0.03) & (x < 0.9); xx, cc = x[m], cm[m]
    if len(xx) < 5: return (np.nan,)*3
    dcf = np.diff(cc); dm = 0.5*(xx[1:] + xx[:-1])
    w = (dm > 0.04) & (dm < 0.7)
    if not w.any() or dcf[w].max() <= 0: return (np.nan,)*3
    i50 = np.argmax(dcf[w]); x50 = float(dm[w][i50])
    cflam = float(cc[xx <= x50].min())
    cfturb_idx = int(np.argmax(cc[xx >= x50])); cfturb = float(cc[xx >= x50][cfturb_idx])
    if cfturb <= cflam: return (np.nan, x50, np.nan)
    lo_thr = cflam + 0.1*(cfturb - cflam); hi_thr = cflam + 0.9*(cfturb - cflam)
    xdip = float(xx[xx <= x50][int(np.argmin(cc[xx <= x50]))])
    xpk = float(xx[xx >= x50][cfturb_idx])
    rise = (xx >= xdip) & (xx <= xpk); xr, cr = xx[rise], cc[rise]
    cx = lambda l: float(xr[np.where(cr >= l)[0][0]]) if (cr >= l).any() else float('nan')
    return (cx(lo_thr), x50, cx(hi_thr))

F = "/home/qiqi/flexcompute/aft-sa/flow360"
CASES = [
    ('proper_cav_L0', 'cavity L0',  'k',  '-', None),
    ('proper_cav_L1', 'cavity L1',  'C0', '-', None),
    ('proper_cav_L2', 'cavity L2',  'C2', '-', None),
    ('proper_cav_L3', 'cavity L3',  'C1', '-', None),
    ('proper_str_L0', 'O-grid L0',  'k',  '--', None),
    ('proper_str_L1', 'O-grid L1',  'C0', '--', None),
    ('proper_str_L2', 'O-grid L2',  'C2', '--', None),
]

# pick up cav L3 cell count from mesh metadata if needed
def find_ncell(d):
    import json
    p = f"{d}/mesh.cgns.json"
    if os.path.exists(p):
        m = json.load(open(p))
        if 'nNodes' in m: return int(m['nNodes'])
    return None

# also use json mesh info if available
for i, (d, label, c, ls, ncell) in enumerate(CASES):
    if ncell is None:
        ncell = find_ncell(f"{F}/{d}")
        CASES[i] = (d, label, c, ls, ncell)
        print(f"resolved {d} ncell -> {ncell}")

def forces(d):
    tr = list(csv.reader(open(f"{d}/total_forces_v2.csv")))
    th = [x.strip() for x in tr[0]]
    last = [x.strip() for x in tr[-1] if x.strip() != '']
    g = lambda k: float(last[th.index(k)])
    return dict(
        CL=g('CL'), CD=g('CD'), CMy=g('CMy'),
        CLp=g('CLPressure'), CDp=g('CDPressure'), CMyp=g('CMyPressure'),
        CLf=g('CLSkinFriction'), CDf=g('CDSkinFriction'), CMyf=g('CMySkinFriction'),
    )

fig, axs = plt.subplots(2, 2, figsize=(13, 8))
ax_cp_up, ax_cp_lo = axs[0, 0], axs[0, 1]
ax_cf_up, ax_cf_lo = axs[1, 0], axs[1, 1]

print(f"{'case':>16s}  {'CL':>7s}  {'CD':>8s}  {'CMy':>8s}  | up: {'x10':>5s} {'x50':>5s} {'x90':>5s}  | lo: {'x10':>5s} {'x50':>5s} {'x90':>5s}")
trans_table = {}
forces_table = {}
for d, label, c, ls, ncell in CASES:
    fpath = f"{F}/{d}"
    if not os.path.exists(f"{fpath}/total_forces_v2.csv"):
        print(f"  {d}: SKIP (no forces csv)"); continue
    (xu, cfu, cpu), (xl, cfl, cpl) = contour_walk_both(fpath)
    ax_cp_up.plot(xu, -cpu, color=c, ls=ls, lw=1.4, label=label)
    ax_cp_lo.plot(xl, -cpl, color=c, ls=ls, lw=1.4, label=label)
    ax_cf_up.plot(xu, cfu, color=c, ls=ls, lw=1.4, label=label)
    ax_cf_lo.plot(xl, cfl, color=c, ls=ls, lw=1.4, label=label)
    tu = transition_band(xu, cfu); tl = transition_band(xl, cfl)
    f = forces(fpath); forces_table[d] = (f, ncell); trans_table[d] = (tu, tl, ncell)
    print(f"  {d:14s}  {f['CL']:.4f}  {f['CD']:.5f}  {f['CMy']:.4f}  |     {tu[0]:.3f} {tu[1]:.3f} {tu[2]:.3f}  |     {tl[0]:.3f} {tl[1]:.3f} {tl[2]:.3f}")

for ax, title, ylim in [(ax_cp_up, '(a) $-C_p$, upper', None),
                         (ax_cp_lo, '(b) $-C_p$, lower', None),
                         (ax_cf_up, '(c) $C_f$, upper', (0, 0.012)),
                         (ax_cf_lo, '(d) $C_f$, lower', (0, 0.012))]:
    ax.set_xlabel(r'$x/c$'); ax.set_xlim(0, 1)
    if ylim: ax.set_ylim(*ylim)
    ax.grid(alpha=0.3); ax.set_title(title, fontsize=11)
ax_cp_up.set_ylabel(r'$-C_p$'); ax_cp_lo.set_ylabel(r'$-C_p$')
ax_cf_up.set_ylabel(r'$C_f$'); ax_cf_lo.set_ylabel(r'$C_f$')
ax_cp_up.legend(fontsize=8, frameon=False, ncol=2, loc='upper right')

plt.tight_layout(pad=0.5)
plt.savefig('/tmp/cp_cf_L0L1L2L3.png', dpi=140)
plt.savefig('/home/qiqi/flexcompute/aft-sa/paper/figs/proper_cp_cf_upper_lower_L3.pdf')
print("wrote /tmp/cp_cf_L0L1L2L3.png + paper figs/proper_cp_cf_upper_lower_L3.pdf")

# === force-coefficient grid convergence ===
fig2, axs2 = plt.subplots(1, 3, figsize=(13, 4))
h_cav = []; CL_cav = []; CD_cav = []; CM_cav = []
h_str = []; CL_str = []; CD_str = []; CM_str = []
for d, label, c, ls, ncell in CASES:
    if d not in forces_table or ncell is None: continue
    f, _ = forces_table[d]
    h = 1.0/np.sqrt(ncell)
    if 'cav' in d:
        h_cav.append(h); CL_cav.append(f['CL']); CD_cav.append(f['CD']); CM_cav.append(f['CMy'])
    else:
        h_str.append(h); CL_str.append(f['CL']); CD_str.append(f['CD']); CM_str.append(f['CMy'])
# sort by h for clean lines
def sort_by(xs, *ys):
    idx = np.argsort(xs); return [np.array(xs)[idx]] + [np.array(y)[idx] for y in ys]
h_cav, CL_cav, CD_cav, CM_cav = sort_by(h_cav, CL_cav, CD_cav, CM_cav)
h_str, CL_str, CD_str, CM_str = sort_by(h_str, CL_str, CD_str, CM_str)
for ax, vals_c, vals_s, ylab, title in [
    (axs2[0], CL_cav, CL_str, r'$C_L$', '(a) lift coefficient'),
    (axs2[1], CD_cav, CD_str, r'$C_D$', '(b) drag coefficient'),
    (axs2[2], CM_cav, CM_str, r'$C_{M,y}$', '(c) pitching moment')]:
    ax.plot(h_cav, vals_c, 'ko-', ms=6, label='cavity', lw=1.4)
    ax.plot(h_str, vals_s, '^--', color='C3', ms=6, label='O-grid', lw=1.4)
    ax.set_xlabel(r'$h\sim 1/\sqrt{N_{\rm cell}}$'); ax.set_ylabel(ylab); ax.set_title(title, fontsize=10)
    ax.set_xlim(0, None); ax.grid(alpha=0.3); ax.legend(fontsize=9, frameon=False)
plt.tight_layout(pad=0.5)
plt.savefig('/tmp/force_convergence_L3.png', dpi=140)
plt.savefig('/home/qiqi/flexcompute/aft-sa/paper/figs/proper_force_convergence_L3.pdf')
print("wrote /tmp/force_convergence_L3.png + paper figs/proper_force_convergence_L3.pdf")

print(f"\n=== cavity series: ===")
for h, cl, cd, cm in zip(h_cav, CL_cav, CD_cav, CM_cav):
    print(f"  h={h:.5f}  CL={cl:.5f}  CD={cd:.5f}  CMy={cm:.5f}")
print(f"=== O-grid series: ===")
for h, cl, cd, cm in zip(h_str, CL_str, CD_str, CM_str):
    print(f"  h={h:.5f}  CL={cl:.5f}  CD={cd:.5f}  CMy={cm:.5f}")
