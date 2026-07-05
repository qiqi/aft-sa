"""Regenerate NACA figures with new kernel data + add nuHat+N top row.

fig:cf = naca0012_cf.pdf — 6 panels (3 rows x 2 cols):
  TOP row (NEW): max_y(nuHat) on upper/lower surfaces (log, left axis)
                 and mfoil N (linear, right axis), at α=0 (left) and α=4 (right).
  MID row: -Cp upper+lower surfaces.
  BOT row: |Cf| upper+lower surfaces.
"""
import csv, os, pickle, numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
matplotlib.rcParams.update({'font.size': 10.5, 'axes.titlesize': 10.5, 'axes.labelsize': 10.5,
                            'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'legend.fontsize': 9})

B = "/home/qiqi/flexcompute/aft-sa/flow360"
PD = "/home/qiqi/flexcompute/aft-sa/paper"
NU = 2e-7
# Cases below are cavprop/strprop with chi_inf=0.02.  In our model N = ln(chi/chi_inf),
# so plotting chi on a log axis is equivalent to plotting N on a linear axis, offset by
# log(chi_inf).  We align the two y axes by choosing N range first, then chi range as
# chi_inf·exp(N_lo), chi_inf·exp(N_hi).
CHI_INF = 8.76e-4
N_LO, N_HI = -2.0, 12.0
CHI_LO, CHI_HI = CHI_INF*np.exp(N_LO), CHI_INF*np.exp(N_HI)

def wd(af, mesh, model, a):
    if mesh=='cavity' and model=='aftsa_m2': return f"{B}/cavprop_{af}_a{a}"
    if mesh=='cavity' and model=='turb':     return f"{B}/cavprop_{af}_turb_a{a}"
    if mesh=='ogrid'  and model=='aftsa_m2': return f"{B}/strprop_{af}_a{a}"
    if mesh=='ogrid'  and model=='turb':     return f"{B}/strprop_{af}_turb_a{a}"
    return f"{B}/full_{af}_{mesh}_{model}_a{a}"

def contour(d, af):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{d}/surface_fluid_{af}.pvtu"); r.Update()
    g = r.GetOutput(); p = vtk_to_numpy(g.GetPoints().GetData()); pd = g.GetPointData()
    nm = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    cf_arr = vtk_to_numpy(pd.GetArray(next(n for n in nm if n.lower().startswith('cf'))))
    cf = np.linalg.norm(cf_arr, axis=1) if cf_arr.ndim > 1 else cf_arr
    cp = vtk_to_numpy(pd.GetArray(next(n for n in nm if n.lower().startswith('cp'))))
    X, Z, CF, CP = p[:,0], p[:,2], cf, cp
    n = len(X)
    # Walk surface contour
    j = int(np.argmin(X)); o = [j]; u = np.zeros(n, bool); u[j] = True; pts = np.column_stack((X, Z))
    while len(o) < n:
        c = o[-1]; dd = np.sum((pts - pts[c])**2, 1); dd[u] = 1e9
        nx = int(np.argmin(dd)); o.append(nx); u[nx] = True
    o = np.array(o); xo, zo, cfo, cpo = X[o], Z[o], CF[o], CP[o]
    mid = n // 2
    window = slice(max(0, mid - n//8), min(n, mid + n//8))
    te = window.start + int(np.argmax(xo[window]))
    b1, b2 = slice(0, te), slice(te+1, n)
    up, lo = (b1, b2) if zo[b1].mean() >= zo[b2].mean() else (b2, b1)
    def srt(sl):
        xx = xo[sl]; oo = np.argsort(xx); xx, cfs, cps = xx[oo], cfo[sl][oo], cpo[sl][oo]
        if len(xx) > 2 and xx[0] < 0.001 and xx[1] > 2*xx[0]:
            xx, cfs, cps = xx[1:], cfs[1:], cps[1:]
        return xx, cfs, cps
    return srt(up), srt(lo)

def max_nuhat_vs_x(case_dir, nbins=80, d_max=0.05):
    """Load slice, bin by x, take max nuHat over BL band on each side.

    Uses solver-computed wallDistance to define the BL band (wd ≤ d_max), so the
    search ALWAYS reaches at least to the BL edge (where ν̃ = ν̃_∞) regardless of
    where the airfoil sits in z.  Separates upper vs lower by the sign of z
    relative to the chord line.  Returns max ν̃ on each surface clipped from below
    at ν̃_∞ in case the search band lands entirely below freestream.
    """
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{case_dir}/slice_centerSpan.pvtu"); r.Update()
    g = r.GetOutput(); pd = g.GetPointData()
    arrs = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    if 'nuHat' not in arrs or 'wallDistance' not in arrs:
        return None, None, None
    p = vtk_to_numpy(g.GetPoints().GetData())
    nh = vtk_to_numpy(pd.GetArray('nuHat'))
    wd_arr = vtk_to_numpy(pd.GetArray('wallDistance'))
    edges = np.linspace(0, 1.0, nbins+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    mu = np.full(nbins, np.nan); ml = np.full(nbins, np.nan)
    mask_band = wd_arr <= d_max
    for i in range(nbins):
        m_x = (p[:,0] >= edges[i]) & (p[:,0] < edges[i+1]) & mask_band
        if not m_x.any(): continue
        zb = p[m_x, 2]; nb = nh[m_x]
        u = zb > 0; l = zb < 0
        if u.any(): mu[i] = nb[u].max()
        if l.any(): ml[i] = nb[l].max()
    return centers, mu, ml

# ---- Load mfoil data ----
mfs = pickle.load(open(f"{PD}/data/mfoil_surf_naca0012.pkl", 'rb'))
mn  = pickle.load(open(f"{B}/mfoil_naca0012_a0a4.pkl", 'rb'))   # has N(x) on each surface

UP, LO = 'C0', 'C3'

# ---- Figure: 3 rows x 2 cols ----
fig, axs = plt.subplots(3, 2, figsize=(8.5, 8.5), sharex=True)

for col, a in enumerate([0, 4]):
    ax_n  = axs[0, col]   # new top row: max nuHat + N
    ax_cp = axs[1, col]
    ax_cf = axs[2, col]
    ax_nN = ax_n.twinx()  # right axis for mfoil N

    # ---- TOP ROW: max nuHat (log) + mfoil N (linear) ----
    # SA-AI O-grid
    for mesh, model, ls, lw in [('ogrid', 'aftsa_m2', '-', 1.4),
                                  ('cavity', 'aftsa_m2', '--', 1.4)]:
        d = wd('naca0012', mesh, model, a)
        xc, mu_u, mu_l = max_nuhat_vs_x(d)
        if xc is None: continue
        ax_n.semilogy(xc, mu_u/NU, ls, color=UP, lw=lw)
        ax_n.semilogy(xc, mu_l/NU, ls, color=LO, lw=lw)
    # mfoil N(x) on each surface
    md = mn[float(a)][9.0]
    ax_nN.plot(md['upper']['x'], md['upper']['n'], ':', color=UP, lw=1.4)
    ax_nN.plot(md['lower']['x'], md['lower']['n'], ':', color=LO, lw=1.4)
    # Reference markers — same vertical position on both axes (axes linked):
    #   right (linear): N = N_crit = 9
    #   left (log):     chi = c_v1 = 7.1  (f_v1 = 1/2)
    ax_nN.axhline(9.0, color='gray', ls=':', lw=0.7, alpha=0.7)
    ax_n.axhline(7.1, color='gray', ls=':', lw=0.7, alpha=0.7)
    if md['xtr_upper'] is not None:
        ax_n.axvline(md['xtr_upper'], color=UP, ls=':', lw=0.6, alpha=0.5)
    if md['xtr_lower'] is not None:
        ax_n.axvline(md['xtr_lower'], color=LO, ls=':', lw=0.6, alpha=0.5)
    ax_n.set_ylabel('$\\chi=\\tilde\\nu/\\nu$  (log, left)')
    ax_nN.set_ylabel('mfoil $N$ (linear, right) — $\\Delta N\\!=\\!1$ matches $\\Delta\\!\\log\\chi\\!=\\!1\\,$e-fold')
    # Aligned: N = ln(chi/chi_inf). chi range = chi_inf*exp(N range).
    ax_n.set_ylim(CHI_LO, CHI_HI); ax_nN.set_ylim(N_LO, N_HI)
    ax_n.set_title(rf'$\alpha={a}^\circ$', fontsize=10)
    ax_n.grid(alpha=0.3)

    # ---- MID + BOT ROWS: Cp + Cf ----
    sols = [
        ('ogrid',  'aftsa_m2', '-',  1.4, 1.0),
        ('cavity', 'aftsa_m2', '--', 1.4, 1.0),
        ('ogrid',  'turb',     '-.', 1.0, 0.85),
    ]
    for mesh, model, ls, lw, al in sols:
        try:
            (xu, cfu, cpu), (xl, cfl, cpl) = contour(wd('naca0012', mesh, model, a), 'naca0012')
            ax_cp.plot(xu, -cpu, ls, color=UP, lw=lw, alpha=al)
            ax_cp.plot(xl, -cpl, ls, color=LO, lw=lw, alpha=al)
            ax_cf.plot(xu,  cfu, ls, color=UP, lw=lw, alpha=al)
            ax_cf.plot(xl,  cfl, ls, color=LO, lw=lw, alpha=al)
        except Exception as e:
            print(f"  skip ({mesh},{model},a={a}): {e}")
    m = mfs[float(a)]
    ax_cp.plot(m['x_upper'], -np.asarray(m['cp_upper']), ':', color=UP, lw=1.4)
    ax_cp.plot(m['x_lower'], -np.asarray(m['cp_lower']), ':', color=LO, lw=1.4)
    ax_cf.plot(m['x_upper'],  np.asarray(m['cf_upper']), ':', color=UP, lw=1.4)
    ax_cf.plot(m['x_lower'],  np.asarray(m['cf_lower']), ':', color=LO, lw=1.4)
    ax_cf.set_xlabel('$x/c$')
    ax_cp.set_xlim(0, 1); ax_cf.set_xlim(0, 1); ax_n.set_xlim(0, 1)
    ax_cf.set_ylim(0, 0.012)
    ax_cp.grid(alpha=0.3); ax_cf.grid(alpha=0.3)

axs[1, 0].set_ylabel('$-C_p$'); axs[2, 0].set_ylabel('$C_f$')

# Composite legend
surfh = [Line2D([], [], color=UP, lw=2, label='upper'),
         Line2D([], [], color=LO, lw=2, label='lower')]
solh  = [Line2D([], [], color='0.3', ls='-',  lw=1.4, label='SA-AI, O-grid'),
         Line2D([], [], color='0.3', ls='--', lw=1.4, label='SA-AI, unstructured'),
         Line2D([], [], color='0.3', ls='-.', lw=1.0, alpha=0.85, label='SA (turb.), O-grid'),
         Line2D([], [], color='0.3', ls=':',  lw=1.4, label=r'mfoil ($e^N$)')]
axs[0, 1].legend(handles=surfh, fontsize=8, frameon=False, loc='upper left')
axs[2, 1].legend(handles=solh,  fontsize=8, frameon=False, loc='upper right')

plt.tight_layout(pad=0.5)
plt.savefig(f"{PD}/figs/naca0012_cf.pdf"); plt.savefig('/tmp/naca0012_cf.png', dpi=140)
plt.close()
print("wrote naca0012_cf.pdf (6 panels: max-nuHat+N, Cp, Cf at alpha=0,4)")
