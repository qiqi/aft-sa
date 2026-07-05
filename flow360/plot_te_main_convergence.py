"""Primary grid convergence figure for the paper: TE-refined cavity (L0-L3, monotonic) vs
O-grid (L0-L2). 8 panels: CL, CD, CD_pressure, CD_friction, CMy, L/D, x_tr upper, x_tr lower.
"""
import numpy as np, vtk, csv, os, json
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt

def contour_walk_both(d, af='naca0012'):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{d}/surface_fluid_{af}.pvtu"); r.Update()
    g = r.GetOutput(); p = vtk_to_numpy(g.GetPoints().GetData()); pd = g.GetPointData()
    arrs = {pd.GetArrayName(i): vtk_to_numpy(pd.GetArray(i)) for i in range(pd.GetNumberOfArrays())}
    cf = arrs['Cf']; cf = np.linalg.norm(cf, axis=1) if cf.ndim > 1 else cf
    x, y, z = p[:,0], p[:,1], p[:,2]
    s = np.abs(y - y.min()) < 1e-6
    X, Z, CF = x[s], z[s], cf[s]
    n = len(X); pts = np.column_stack([X, Z])
    st = int(np.argmin(X)); o = [st]; u = np.zeros(n, bool); u[st] = True
    for _ in range(n-1):
        c = o[-1]; dd = np.sum((pts - pts[c])**2, 1); dd[u] = 1e9
        nx = int(np.argmin(dd)); o.append(nx); u[nx] = True
    o = np.array(o); xo, zo, cfo = X[o], Z[o], CF[o]
    # For closed-TE contours (both upper/lower endpoints at (1, 0)), the global
    # max-x sits at the very last walked point — that's not a useful TE split.
    # The TE is the walked-distance midpoint: ~n/2. Find the local-max-x near
    # the midpoint robustly.
    mid = n // 2
    window = slice(max(0, mid - n//8), min(n, mid + n//8))
    te = window.start + int(np.argmax(xo[window]))
    b1, b2 = slice(0, te), slice(te+1, n)   # exclude TE-shared point
    up, lo = (b1, b2) if zo[b1].mean() >= zo[b2].mean() else (b2, b1)
    def srt(sl):
        xs = xo[sl]; oo = np.argsort(xs)
        return xs[oo], cfo[sl][oo]
    return srt(up), srt(lo)

def transition_band(x, cf):
    k = np.ones(7)/7; cm = np.convolve(cf, k, 'same')
    m = (x > 0.03) & (x < 0.95); xx, cc = x[m], cm[m]
    if len(xx) < 5: return (np.nan,)*3
    dcf = np.diff(cc); dm = 0.5*(xx[1:] + xx[:-1])
    w = (dm > 0.04) & (dm < 0.9)
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
def get_ncell(d):
    p = f"{d}/mesh.cgns.json"
    return int(json.load(open(p))['nNodes']) if os.path.exists(p) else None
def forces(d):
    tr = list(csv.reader(open(f"{d}/total_forces_v2.csv")))
    th = [x.strip() for x in tr[0]]
    last = [x.strip() for x in tr[-1] if x.strip() != '']
    g = lambda k: float(last[th.index(k)])
    return dict(CL=g('CL'), CD=g('CD'), CMy=g('CMy'),
                CDp=g('CDPressure'), CDf=g('CDSkinFriction'))

SERIES = {
    'cavity (TE-proper metric)':
        [('proper_cav_L0_TEprop','cav L0'),('proper_cav_L1_TEprop','cav L1'),
         ('proper_cav_L2_TEprop','cav L2')],
    'O-grid':
        [('proper_str_L0','str L0'),('proper_str_L1','str L1'),('proper_str_L2','str L2')],
}

rows = {fam: [] for fam in SERIES}
for fam, cases in SERIES.items():
    for d, lbl in cases:
        full = f"{F}/{d}"
        if not os.path.exists(f"{full}/total_forces_v2.csv"):
            print(f"  skip {d}"); continue
        nc = get_ncell(full)
        f = forces(full)
        (xu, cfu), (xl, cfl) = contour_walk_both(full)
        tu = transition_band(xu, cfu)
        tl = transition_band(xl, cfl)
        h = 1.0/np.sqrt(nc)
        rows[fam].append((h, nc, f['CL'], f['CD'], f['CDp'], f['CDf'], f['CMy'],
                          tu[0], tu[1], tu[2], tl[0], tl[1], tl[2]))
        print(f"  {d}: nNodes={nc} CL={f['CL']:.4f} CD={f['CD']:.5f} CDp={f['CDp']:.5f} CDf={f['CDf']:.5f} CMy={f['CMy']:.4f} "
              f"x_tr_up=({tu[0]:.3f},{tu[1]:.3f},{tu[2]:.3f}) x_tr_lo=({tl[0]:.3f},{tl[1]:.3f},{tl[2]:.3f})")
    rows[fam].sort(key=lambda r: r[0])

# Style — monotone B&W: both series black, distinguished by linestyle + marker only.
# Legend conventions are described in the figure caption (no in-figure legend).
styles = {'cavity (TE-proper metric)': dict(c='k', marker='o', ls='-',  label='unstructured'),
          'O-grid':                     dict(c='k', marker='^', ls='--', label='O-grid')}

matplotlib.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 10,
                            'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 9})
fig, axs = plt.subplots(2, 4, figsize=(7.5, 4.2))

def plot_panel(ax, idx, ylab, title, ylim=None):
    for fam, items in rows.items():
        if not items: continue
        st = styles[fam]
        h_ = [r[0] for r in items]
        v_ = [r[idx] for r in items]
        ax.plot(h_, v_, marker=st['marker'], ls=st['ls'], color=st['c'],
                ms=4, lw=1.2, label=st['label'])
    ax.set_xlabel(r'$h\sim 1/\sqrt{N_{\rm nodes}}$')
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.set_xlim(0, None); ax.grid(alpha=0.3)
    if ylim: ax.set_ylim(*ylim)

plot_panel(axs[0,0], 2, '$C_L$', '(a) lift')
plot_panel(axs[0,1], 3, '$C_D$', '(b) total drag')
plot_panel(axs[0,2], 4, r'$C_{D,p}$', '(c) pressure drag')
plot_panel(axs[0,3], 5, r'$C_{D,v}$', '(d) viscous drag')
plot_panel(axs[1,0], 6, '$C_{M,y}$', '(e) pitching moment')

# (f) L/D ratio — derived
for fam, items in rows.items():
    if not items: continue
    st = styles[fam]
    h_ = [r[0] for r in items]
    LoD = [r[2]/r[3] for r in items]
    axs[1,1].plot(h_, LoD, marker=st['marker'], ls=st['ls'], color=st['c'],
                  ms=4, lw=1.2, label=st['label'])
axs[1,1].set_xlabel(r'$h\sim 1/\sqrt{N_{\rm nodes}}$')
axs[1,1].set_ylabel(r'$L/D$')
axs[1,1].set_title('(f) lift-to-drag ratio')
axs[1,1].set_xlim(0, None); axs[1,1].grid(alpha=0.3)

# (g,h) transition bands upper/lower with error bars
for k, idx_band, ylab, title in [(2, (7,8,9), '$x_{tr}/c$ upper', '(g) suction-side transition'),
                                  (3, (10,11,12), '$x_{tr}/c$ lower', '(h) pressure-side transition')]:
    ax = axs[1, k]
    for fam, items in rows.items():
        if not items: continue
        st = styles[fam]
        h_ = [r[0] for r in items]
        x10 = [r[idx_band[0]] for r in items]
        x50 = [r[idx_band[1]] for r in items]
        x90 = [r[idx_band[2]] for r in items]
        lo_err = [(a-b) if not (np.isnan(a) or np.isnan(b)) else 0 for a,b in zip(x50, x10)]
        hi_err = [(b-a) if not (np.isnan(a) or np.isnan(b)) else 0 for a,b in zip(x50, x90)]
        ax.errorbar(h_, x50, yerr=[lo_err, hi_err], fmt=st['marker'], ls=st['ls'], color=st['c'],
                    ms=4, lw=1.2, capsize=2, capthick=1.0, label=st['label'], alpha=0.85)
    ax.set_xlabel(r'$h\sim 1/\sqrt{N_{\rm nodes}}$')
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.set_xlim(0, None); ax.grid(alpha=0.3)

# No in-figure legend; line style described in the figure caption.

plt.tight_layout(pad=0.5)
plt.savefig('/tmp/te_main_convergence.png', dpi=140)
plt.savefig('/home/qiqi/flexcompute/aft-sa/paper/figs/proper_grid_convergence_L3.pdf')
print('wrote /tmp/te_main_convergence.png + paper figs/proper_grid_convergence_L3.pdf')

print("\n=== finest-level summary ===")
for fam, items in rows.items():
    if items:
        i = items[0]
        print(f"  {fam}: nNodes={i[1]} CL={i[2]:.4f} CD={i[3]:.5f} CDp={i[4]:.5f} CDf={i[5]:.5f} L/D={i[2]/i[3]:.2f} CMy={i[6]:.4f}")
