"""3-panel comparison: baseline Flow360 / unified (variant B) Flow360 / mfoil at two Ncrit.

  Cp(x) upper surface
  Cf(x) upper surface
  N(x)  amplification factor — for Flow360 = ln(max_y ν̃/ν̃_∞), for mfoil = U[2,:]
"""
import vtk, numpy as np, pickle
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import maximum_filter1d

F  = "/home/qiqi/flexcompute/aft-sa/flow360"
PD = "/home/qiqi/flexcompute/aft-sa/paper"
NU_LAM = 2e-7

FLOW360 = [
    ('cav_L1_base_a0', 'baseline ($a_{\\max}\\!=\\!0.05$, χ$_\\infty\\!=\\!0.02$, N$_{\\max}\\!=\\!3.91$)',  4e-9,    'C0'),
    ('cav_L1_varB_a0', 'unified ($a_{\\max}\\!=\\!0.20$, χ$_\\infty\\!=\\!1.6\\times10^{-7}$, N$_{\\max}\\!=\\!15.65$)', 3.2e-14, 'C2'),
]
MFOIL_PKL = f"{F}/mfoil_naca0012_compare.pkl"
MFOIL_CASES = [
    (9.0, 'mfoil, N$_\\mathrm{crit}\\!=\\!9$', 'C1'),
    (15.65, 'mfoil, N$_\\mathrm{crit}\\!=\\!15.65$', 'C3'),
]

def load_surf(d):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f'{d}/surface_fluid_naca0012.pvtu'); r.Update()
    g = r.GetOutput(); pd = g.GetPointData()
    p = vtk_to_numpy(g.GetPoints().GetData())
    cp = vtk_to_numpy(pd.GetArray('Cp'))
    cf = vtk_to_numpy(pd.GetArray('Cf'))
    cf_mag = np.linalg.norm(cf, axis=1) if cf.ndim == 2 else np.abs(cf)
    return p[:,0], p[:,2], cp, cf_mag

def load_slice_nh(d):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f'{d}/slice_centerSpan.pvtu'); r.Update()
    g = r.GetOutput(); pd = g.GetPointData()
    nh = vtk_to_numpy(pd.GetArray('nuHat'))
    p  = vtk_to_numpy(g.GetPoints().GetData())
    return p[:,0], p[:,2], nh

x_edges = np.linspace(-0.05, 1.05, 70)
def bin_max(x, z, nh, bl_y=0.06):
    mask = np.abs(z) < bl_y
    x, z, nh = x[mask], z[mask], nh[mask]
    cs = 0.5*(x_edges[:-1] + x_edges[1:])
    out = np.full(len(cs), np.nan)
    for i in range(len(cs)):
        m = (x >= x_edges[i]) & (x < x_edges[i+1])
        if m.any(): out[i] = nh[m].max()
    return cs, out

matplotlib.rcParams.update({'font.size':11,'axes.titlesize':11.5,'axes.labelsize':11,
                            'legend.fontsize':9.5,'xtick.labelsize':10,'ytick.labelsize':10})

fig, axs = plt.subplots(1, 3, figsize=(16, 5))
ax_cp, ax_cf, ax_N = axs

# --- Flow360 cases ---
for case, label, nu_inf, color in FLOW360:
    d = f'{F}/{case}'
    try:
        xs, zs, cp, cf = load_surf(d)
        up = zs > 0
        order = np.argsort(xs[up])
        ax_cp.plot(xs[up][order], cp[up][order], '-', color=color, lw=1.6, label=label)
        ax_cf.plot(xs[up][order], cf[up][order], '-', color=color, lw=1.6, label=label)
    except Exception as e: print(f"surf {case}: {e}")
    try:
        xs, zs, nh = load_slice_nh(d)
        xc, max_nh = bin_max(xs, zs, nh)
        max_nh = maximum_filter1d(np.where(np.isnan(max_nh), -np.inf, max_nh), size=3)
        max_nh = np.where(max_nh == -np.inf, np.nan, max_nh)
        N = np.log(np.maximum(max_nh, 1e-35) / nu_inf)
        ax_N.plot(xc, N, '-', color=color, lw=1.6, label=label)
    except Exception as e: print(f"slice {case}: {e}")

# --- mfoil cases ---
mdata = pickle.load(open(MFOIL_PKL,'rb'))
for ncrit, label, color in MFOIL_CASES:
    if ncrit not in mdata: continue
    md = mdata[ncrit]
    # mfoil x already starts from LE.  Order by x:
    order = np.argsort(md['x'])
    x_m = md['x'][order]
    cp_m = md['cp'][order]
    cf_m = md['cf'][order]
    n_m = md['n_amp'][order]
    ax_cp.plot(x_m, cp_m, '--', color=color, lw=1.6, label=label)
    ax_cf.plot(x_m, np.abs(cf_m), '--', color=color, lw=1.6, label=label)
    ax_N.plot(x_m, n_m, '--', color=color, lw=1.6, label=label)
    # mark transition x
    if md['xtr_upper'] is not None:
        ax_cf.axvline(md['xtr_upper'], color=color, ls=':', lw=0.8, alpha=0.7)
        ax_N.axvline(md['xtr_upper'], color=color, ls=':', lw=0.8, alpha=0.7)
        ax_N.text(md['xtr_upper']+0.005, ncrit-0.5, f'$x_\\mathrm{{tr}}\\!=\\!{md["xtr_upper"]:.2f}$',
                  color=color, fontsize=9, ha='left')
    # mark N_crit horizontal
    ax_N.axhline(ncrit, color=color, ls=':', lw=0.8, alpha=0.5)

# axes
for ax in [ax_cp, ax_cf, ax_N]:
    ax.set_xlabel('$x/c$'); ax.set_xlim(-0.02, 1.02); ax.grid(alpha=0.4)
ax_cp.set_ylabel('$C_p$ (upper surface)'); ax_cp.invert_yaxis()
ax_cf.set_ylabel('$|C_f|$ (upper surface)'); ax_cf.set_ylim(0, 0.012)
ax_N.set_ylabel('$N$ (amplification factor)'); ax_N.set_ylim(-1, 22)
ax_N.set_title('amplification $N$:  Flow360 = $\\ln(\\max_y \\tilde\\nu / \\tilde\\nu_\\infty)$;  mfoil = e$^N$ envelope', fontsize=10)

ax_cf.legend(fontsize=8.5, frameon=False, loc='upper left', bbox_to_anchor=(0.0, 1.0))

plt.suptitle('NACA 0012, $\\alpha\\!=\\!0^\\circ$, Re$\\!=\\!10^6$ — Flow360 (baseline & unified Variant B) vs mfoil',
             fontsize=12, y=0.995)
plt.tight_layout(rect=(0,0,1,0.96))
plt.savefig('/tmp/unified_vs_mfoil.png', dpi=140)
plt.savefig(f'{PD}/figs/unified_vs_mfoil.pdf')
plt.close()
print('wrote /tmp/unified_vs_mfoil.png')
