"""Plot Cp and Cf along the upper surface for L0, L1, L2 on both grid families.
Identify transition location via Cf rise."""
import numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt

def contour_walk(d, af='naca0012'):
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
    up = b1 if zo[b1].mean() >= zo[b2].mean() else b2
    xu, cu, pu = xo[up], cfo[up], cpo[up]
    ix = np.argsort(xu)
    return xu[ix], cu[ix], pu[ix]

def find_transition_band(x, cf):
    """Return (x10, x50, x90) of the 10–90% Cf-rise zone, the laminar-to-turbulent transition band."""
    # smooth
    k = np.ones(7)/7
    cm = np.convolve(cf, k, 'same')
    m = (x > 0.03) & (x < 0.9)
    xx, cc = x[m], cm[m]
    dcf = np.diff(cc); dm = 0.5*(xx[1:] + xx[:-1])
    w = (dm > 0.04) & (dm < 0.7)
    if not w.any() or dcf[w].max() <= 0: return (np.nan, np.nan, np.nan)
    i50 = np.argmax(dcf[w])
    x50 = float(dm[w][i50])
    cflam = float(cc[xx <= x50].min())
    cfturb_idx = int(np.argmax(cc[xx >= x50]))
    cfturb = float(cc[xx >= x50][cfturb_idx])
    if cfturb <= cflam: return (np.nan, x50, np.nan)
    lo = cflam + 0.1*(cfturb - cflam); hi = cflam + 0.9*(cfturb - cflam)
    xdip = float(xx[xx <= x50][int(np.argmin(cc[xx <= x50]))])
    xpk = float(xx[xx >= x50][cfturb_idx])
    rise = (xx >= xdip) & (xx <= xpk); xr, cr = xx[rise], cc[rise]
    cx = lambda l: float(xr[np.where(cr >= l)[0][0]]) if (cr >= l).any() else float('nan')
    return (cx(lo), x50, cx(hi))

F = "/home/qiqi/flexcompute/aft-sa/flow360"
CASES = [
    ('proper_cav_L0', 'cavity L0',  'k', '-'),
    ('proper_cav_L1', 'cavity L1',  'C0', '-'),
    ('proper_cav_L2', 'cavity L2',  'C2', '-'),
    ('proper_str_L0', 'O-grid L0',  'k', '--'),
    ('proper_str_L1', 'O-grid L1',  'C0', '--'),
    ('proper_str_L2', 'O-grid L2',  'C2', '--'),
]

print(f"{'case':>20s}  {'CD':>8s}  {'x10':>6s}  {'x50':>6s}  {'x90':>6s}  {'band width':>10s}")
fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
import json
res = json.load(open(f"{F}/proper_refinement_results.json"))
for d, label, c, ls in CASES:
    x, cf, cp = contour_walk(f"{F}/{d}")
    axs[0].plot(x, -cp, color=c, ls=ls, lw=1.4, label=label)
    axs[1].plot(x, cf, color=c, ls=ls, lw=1.4, label=label)
    x10, x50, x90 = find_transition_band(x, cf)
    print(f"  {d:18s}  {res[d]['CD']:.5f}  {x10:.3f}  {x50:.3f}  {x90:.3f}  {x90-x10:.3f}")
    # mark x50 with a dot
    cf_at_x50 = float(np.interp(x50, x, cf))
    axs[1].plot([x50], [cf_at_x50], 'o', color=c, ms=4, alpha=0.7)

axs[0].set_xlabel(r'$x/c$'); axs[0].set_ylabel(r'$-C_p$'); axs[0].set_xlim(0,1)
axs[0].set_title(r'(a) $C_p$, upper surface, $\alpha=4^\circ$', fontsize=11)
axs[0].grid(alpha=0.3); axs[0].legend(fontsize=8, frameon=False, ncol=2, loc='upper right')

axs[1].set_xlabel(r'$x/c$'); axs[1].set_ylabel(r'$C_f$'); axs[1].set_xlim(0,1); axs[1].set_ylim(0,0.012)
axs[1].set_title(r'(b) $C_f$, upper surface, $\alpha=4^\circ$', fontsize=11)
axs[1].grid(alpha=0.3); axs[1].legend(fontsize=8, frameon=False, ncol=2, loc='upper right')

plt.tight_layout(pad=0.5)
plt.savefig('/tmp/cp_cf_L0L1L2.png', dpi=140)
plt.savefig('/home/qiqi/flexcompute/aft-sa/paper/figs/proper_cp_cf_L0L1L2.pdf')
print("\nwrote /tmp/cp_cf_L0L1L2.png + paper figs/proper_cp_cf_L0L1L2.pdf")
