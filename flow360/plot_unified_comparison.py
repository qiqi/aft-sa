"""Comprehensive comparison: baseline (a_max=0.05, chi-linear σ_t, χ_∞=0.02) vs
unified (a_max=0.2, chi^{1/4} σ_t, χ_∞=1.6e-7), on NACA0012 α=0°,
across 2 grids × 2 refinement levels.
"""
import vtk, numpy as np, json, csv, os
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

F = "/home/qiqi/flexcompute/aft-sa/flow360"
PD = "/home/qiqi/flexcompute/aft-sa/paper"

# Layout: 2 cols (grid), 2 rows (level)
MESHES = [
    ('cav_L0', 'unstructured L0'),
    ('cav_L1', 'unstructured L1'),
    ('str_L0', 'O-grid L0'),
    ('str_L1', 'O-grid L1'),
]

UP, LO = 'C0', 'C3'

def walk_contour(d, af='naca0012'):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f'{d}/surface_fluid_{af}.pvtu'); r.Update()
    g = r.GetOutput(); p = vtk_to_numpy(g.GetPoints().GetData()); pd = g.GetPointData()
    cf = vtk_to_numpy(pd.GetArray('Cf')); cp = vtk_to_numpy(pd.GetArray('Cp'))
    if cf.ndim > 1: cf = np.linalg.norm(cf, axis=1)
    x, y, z = p[:,0], p[:,1], p[:,2]
    s = np.abs(y - y.min()) < 1e-6
    X, Z, CF, CP = x[s], z[s], cf[s], cp[s]; n = len(X); pts = np.column_stack([X, Z])
    st = int(np.argmin(X)); o = [st]; u = np.zeros(n, bool); u[st] = True
    for _ in range(n-1):
        c = o[-1]; dd = np.sum((pts - pts[c])**2, 1); dd[u] = 1e9
        nx = int(np.argmin(dd)); o.append(nx); u[nx] = True
    o = np.array(o); xo, zo, cfo, cpo = X[o], Z[o], CF[o], CP[o]
    mid = n // 2
    window = slice(max(0, mid - n//8), min(n, mid + n//8))
    te = window.start + int(np.argmax(xo[window]))
    b1, b2 = slice(0, te), slice(te+1, n)
    up, lo = (b1, b2) if zo[b1].mean() >= zo[b2].mean() else (b2, b1)
    def srt(sl):
        xs = xo[sl]; oo = np.argsort(xs); return xs[oo], cfo[sl][oo], cpo[sl][oo]
    return srt(up), srt(lo)

def forces(d):
    tr = list(csv.reader(open(f'{d}/total_forces_v2.csv')))
    th = [x.strip() for x in tr[0]]; tl = [x for x in tr[-1] if x.strip()!='']
    g = lambda k: float(tl[th.index(k)])
    return dict(CL=g('CL'), CD=g('CD'), CDp=g('CDPressure'), CDf=g('CDSkinFriction'))

def resid(d):
    rows = list(csv.reader(open(f'{d}/nonlinear_residual_v2.csv')))
    h = [x.strip() for x in rows[0]]
    ip, ic = h.index('pseudo_step'), h.index('0_cont')
    s, c = [], []
    for r in rows[1:]:
        r = [x for x in r if x.strip() != '']
        if len(r) > ic:
            try: s.append(float(r[ip])); c.append(float(r[ic]))
            except: pass
    return np.array(s), np.array(c)

import matplotlib
matplotlib.rcParams.update({'font.size': 9.5, 'axes.titlesize': 10, 'axes.labelsize': 10,
                            'legend.fontsize': 8, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5})

# ============== FIGURE 1: Cf (upper + lower) ==============
fig, axs = plt.subplots(2, 2, figsize=(8.5, 6.8), sharex=True, sharey=True)
for ax, (mkey, label) in zip(axs.flat, MESHES):
    for tag, ls, lw, alpha in [('base', '-', 1.4, 1.0), ('uni', '--', 1.4, 1.0)]:
        d = f'{F}/{mkey}_{tag}_a0'
        if not os.path.exists(f'{d}/surface_fluid_naca0012.pvtu'):
            continue
        try:
            (xu, cfu, _), (xl, cfl, _) = walk_contour(d)
            ax.plot(xu, cfu, ls, color=UP, lw=lw, alpha=alpha)
            ax.plot(xl, cfl, ls, color=LO, lw=lw, alpha=alpha)
        except Exception as e:
            print(f"skip {d}: {e}")
    ax.set_title(label)
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.012)
    ax.grid(alpha=0.3)
for ax in axs[1]: ax.set_xlabel('$x/c$')
for ax in axs[:,0]: ax.set_ylabel('$C_f$')

# Composite legend
surfh = [Line2D([], [], color=UP, lw=2, label='upper'),
         Line2D([], [], color=LO, lw=2, label='lower')]
solh  = [Line2D([], [], color='0.3', ls='-',  label='baseline (a$_\\max$=0.05, χ-linear σ$_t$, χ$_\\infty$=0.02)'),
         Line2D([], [], color='0.3', ls='--', label='unified  (a$_\\max$=0.2, χ$^{1/4}$ σ$_t$, χ$_\\infty$=1.6×10$^{-7}$)')]
axs[0,0].legend(handles=surfh, fontsize=8, frameon=False, loc='upper center', ncol=2)
axs[0,1].legend(handles=solh, fontsize=7.5, frameon=False, loc='upper center')
plt.suptitle(r'NACA 0012, $\alpha=0^\circ$, $Re=10^6$, $M=0.2$  —  $C_f$ comparison: baseline vs unified $a_\max=0.2$', fontsize=11, y=0.99)
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.savefig('/tmp/cf_compare.png', dpi=140); plt.savefig(f'{PD}/figs/unified_vs_baseline_cf.pdf')
plt.close()
print("wrote cf_compare.png")

# ============== FIGURE 2: Cp ==============
fig, axs = plt.subplots(2, 2, figsize=(8.5, 6.8), sharex=True, sharey=True)
for ax, (mkey, label) in zip(axs.flat, MESHES):
    for tag, ls, lw in [('base', '-', 1.4), ('uni', '--', 1.4)]:
        d = f'{F}/{mkey}_{tag}_a0'
        if not os.path.exists(f'{d}/surface_fluid_naca0012.pvtu'): continue
        try:
            (xu, _, cpu), (xl, _, cpl) = walk_contour(d)
            ax.plot(xu, -cpu, ls, color=UP, lw=lw)
            ax.plot(xl, -cpl, ls, color=LO, lw=lw)
        except: pass
    ax.set_title(label)
    ax.set_xlim(0, 1); ax.grid(alpha=0.3)
for ax in axs[1]: ax.set_xlabel('$x/c$')
for ax in axs[:,0]: ax.set_ylabel('$-C_p$')
axs[0,0].legend(handles=surfh, fontsize=8, frameon=False, loc='lower center', ncol=2)
axs[0,1].legend(handles=solh, fontsize=7.5, frameon=False, loc='lower center')
plt.suptitle(r'NACA 0012, $\alpha=0^\circ$  —  $C_p$ comparison', fontsize=11, y=0.99)
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.savefig('/tmp/cp_compare.png', dpi=140); plt.savefig(f'{PD}/figs/unified_vs_baseline_cp.pdf')
plt.close()
print("wrote cp_compare.png")

# ============== FIGURE 3: Drag bar chart ==============
fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.0))
labels = ['cav L0', 'cav L1', 'str L0', 'str L1']
x = np.arange(len(labels))
width = 0.20
metrics = ['CDp', 'CDf', 'CD']
colors = {'CDp': '#4472C4', 'CDf': '#ED7D31', 'CD': 'k'}
for i, m in enumerate(metrics):
    vals_b = []
    vals_u = []
    for mkey, _ in MESHES:
        try: vals_b.append(forces(f'{F}/{mkey}_base_a0')[m])
        except: vals_b.append(np.nan)
        try: vals_u.append(forces(f'{F}/{mkey}_uni_a0')[m])
        except: vals_u.append(np.nan)
    offset = (i - 1) * width
    ax.bar(x + offset - width/4, vals_b, width*0.45, color=colors[m], alpha=0.85,
           label=f'{m} (base)' if i == 0 else None)
    ax.bar(x + offset + width/4, vals_u, width*0.45, color=colors[m], alpha=0.4,
           label=f'{m} (uni)' if i == 0 else None, hatch='//')
    # text annotations
    for j, (b, u) in enumerate(zip(vals_b, vals_u)):
        if not np.isnan(b): ax.text(x[j] + offset - width/4, b, f'{b:.4f}', ha='center', va='bottom', fontsize=6.5, rotation=0)
        if not np.isnan(u): ax.text(x[j] + offset + width/4, u, f'{u:.4f}', ha='center', va='bottom', fontsize=6.5, rotation=0)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('Drag coefficient')
ax.set_title('NACA 0012 $\\alpha=0^\\circ$: drag breakdown — baseline (solid) vs unified $a_\\max{=}0.2$ (hatched)')
# Custom legend
from matplotlib.patches import Patch
handles = [Patch(facecolor=colors['CDp'], label='$C_{D,p}$ (pressure)'),
           Patch(facecolor=colors['CDf'], label='$C_{D,f}$ (friction)'),
           Patch(facecolor='k',           label='$C_D$ (total)'),
           Patch(facecolor='gray', alpha=0.85, label='baseline'),
           Patch(facecolor='gray', alpha=0.4, hatch='//', label='unified')]
ax.legend(handles=handles, fontsize=8.5, frameon=False, ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.10))
ax.set_ylim(0, 0.012)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/drag_compare.png', dpi=140); plt.savefig(f'{PD}/figs/unified_vs_baseline_drag.pdf')
plt.close()
print("wrote drag_compare.png")

# ============== FIGURE 4: Residual histories ==============
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
LVL_LW = {'L0': 0.9, 'L1': 1.6}
for ax, fam in [(axs[0], 'cav'), (axs[1], 'str')]:
    for level in ['L0', 'L1']:
        for tag, ls in [('base', '-'), ('uni', '--')]:
            d = f'{F}/{fam}_{level}_{tag}_a0'
            try:
                s, c = resid(d)
                ax.semilogy(s, c, ls, color='k', lw=LVL_LW[level], alpha=1.0 if tag == 'base' else 0.6,
                            label=f'{level} {tag}')
            except: pass
    title = 'unstructured' if fam == 'cav' else 'O-grid'
    ax.set_title(title); ax.set_xlabel('pseudo-time step')
    ax.set_ylim(1e-12, None); ax.grid(alpha=0.3)
axs[0].set_ylabel('continuity residual')
axs[1].legend(fontsize=8, frameon=False, ncol=2, loc='upper right')
plt.suptitle('Continuity residual: baseline vs unified', fontsize=11, y=0.99)
plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.savefig('/tmp/resid_compare.png', dpi=140); plt.savefig(f'{PD}/figs/unified_vs_baseline_resid.pdf')
plt.close()
print("wrote resid_compare.png")

# ============== Summary table ==============
print("\n=== Drag summary table ===")
print(f"{'mesh':<12} {'CL_base':>10} {'CD_base':>10} {'CDf_base':>10}   {'CL_uni':>10} {'CD_uni':>10} {'CDf_uni':>10}")
for mkey, lbl in MESHES:
    try: b = forces(f'{F}/{mkey}_base_a0')
    except: b = {'CL':np.nan,'CD':np.nan,'CDf':np.nan}
    try: u = forces(f'{F}/{mkey}_uni_a0')
    except: u = {'CL':np.nan,'CD':np.nan,'CDf':np.nan}
    print(f"{lbl:<12} {b['CL']:>+10.5f} {b['CD']:>10.5f} {b['CDf']:>10.5f}   {u['CL']:>+10.5f} {u['CD']:>10.5f} {u['CDf']:>10.5f}")
