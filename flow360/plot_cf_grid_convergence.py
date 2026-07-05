"""Per-grid Cf at alpha=4 for the L0-L2 refinement series, two panels:
left = unstructured (proper_cav_L*_TEprop), right = O-grid (proper_str_L*).
Color = surface (upper/lower); linestyle = mesh level (L0/L1/L2).
"""
import vtk, numpy as np, os, json
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

B = '/home/qiqi/flexcompute/aft-sa/flow360'
PD = '/home/qiqi/flexcompute/aft-sa/paper'

UNSTRUCT = [('L0', f'{B}/proper_cav_L0_TEprop'),
            ('L1', f'{B}/proper_cav_L1_TEprop'),
            ('L2', f'{B}/proper_cav_L2_TEprop')]
STRUCT   = [('L0', f'{B}/proper_str_L0'),
            ('L1', f'{B}/proper_str_L1'),
            ('L2', f'{B}/proper_str_L2')]

UP, LO = 'C0', 'C3'   # upper / lower colors
LINESTYLES = {'L0': '-', 'L1': '--', 'L2': ':'}

def get_ncell(d):
    try: return int(json.load(open(f"{d}/mesh.cgns.json"))['nNodes'])
    except: return None

def cf_upper_lower(d, af='naca0012'):
    """Walk the airfoil contour, return ((x_u, cf_u), (x_l, cf_l)) sorted by x."""
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{d}/surface_fluid_{af}.pvtu"); r.Update()
    g = r.GetOutput(); p = vtk_to_numpy(g.GetPoints().GetData()); pd = g.GetPointData()
    cf = vtk_to_numpy(pd.GetArray(next(pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())
                                       if pd.GetArrayName(i).lower().startswith('cf'))))
    cf = np.linalg.norm(cf, axis=1) if cf.ndim > 1 else cf
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    s = np.abs(y - y.min()) < 1e-6
    X, Z, CF = x[s], z[s], cf[s]; n = len(X); pts = np.column_stack([X, Z])
    st = int(np.argmin(X)); o = [st]; u = np.zeros(n, bool); u[st] = True
    for _ in range(n-1):
        c = o[-1]; dd = np.sum((pts - pts[c])**2, 1); dd[u] = 1e9
        nx = int(np.argmin(dd)); o.append(nx); u[nx] = True
    o = np.array(o); xo, zo, cfo = X[o], Z[o], CF[o]
    mid = n // 2
    window = slice(max(0, mid - n//8), min(n, mid + n//8))
    te = window.start + int(np.argmax(xo[window]))
    b1, b2 = slice(0, te), slice(te+1, n)
    up, lo = (b1, b2) if zo[b1].mean() >= zo[b2].mean() else (b2, b1)
    def srt(sl):
        xs = xo[sl]; oo = np.argsort(xs); xs, cfs = xs[oo], cfo[sl][oo]
        if len(xs) > 2 and xs[0] < 0.001 and xs[1] > 2*xs[0]: xs, cfs = xs[1:], cfs[1:]
        return xs, cfs
    return srt(up), srt(lo)

import matplotlib
matplotlib.rcParams.update({'font.size': 11, 'axes.titlesize': 11, 'axes.labelsize': 11,
                            'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9})
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True, sharex=True)

for ax, family, dlist in [(axs[0], 'unstructured', UNSTRUCT),
                          (axs[1], 'O-grid',       STRUCT)]:
    for (lvl, d) in dlist:
        nc = get_ncell(d)
        (xu, cfu), (xl, cfl) = cf_upper_lower(d)
        ls = LINESTYLES[lvl]
        ax.plot(xu, cfu, ls, color=UP, lw=1.1)
        ax.plot(xl, cfl, ls, color=LO, lw=1.1)
    ax.set_xlim(0, 1); ax.set_xlabel('$x/c$')
    ax.set_title(family)
    ax.set_xticks(np.arange(0, 1.01, 0.1), minor=True)
    ax.grid(which='major', alpha=0.45, lw=0.6)
    ax.grid(which='minor', alpha=0.25, lw=0.4)

axs[0].set_ylabel('$C_f$')
axs[0].set_ylim(0, 0.012)

# Composite legend on the right panel
surfh = [Line2D([], [], color=UP, lw=2, label='upper'),
         Line2D([], [], color=LO, lw=2, label='lower')]
lvlh  = [Line2D([], [], color='0.3', ls=LINESTYLES[l], lw=1.4, label=l) for l in ('L0', 'L1', 'L2')]
axs[0].legend(handles=surfh, frameon=False, loc='upper center', ncol=2)
axs[1].legend(handles=lvlh,  frameon=False, loc='upper right')

plt.tight_layout(pad=0.4)
plt.savefig('/tmp/cf_grid_convergence.png', dpi=140)
plt.savefig(f'{PD}/figs/cf_grid_convergence.pdf')
print(f'wrote {PD}/figs/cf_grid_convergence.pdf')
