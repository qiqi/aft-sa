"""Compare baseline / variant A / variant B on cav_L1.

  baseline      : a_max=0.05, chi-linear σ_t, χ_∞ = 0.02      → reference (transitions)
  variant A     : a_max=0.20, chi^{1/4}  σ_t, χ_∞ = 1.6e-7   → "proper unified" (μ_t feedback caps?)
  variant B     : a_max=0.20, chi-linear σ_t, χ_∞ = 1.6e-7   → drop the 1/4 power

Panels: Cf(x), Cp(x), max_y ν̃(x), N(x) = ln(ν̃/ν̃_∞).
"""
import vtk, numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter1d

F  = "/home/qiqi/flexcompute/aft-sa/flow360"
PD = "/home/qiqi/flexcompute/aft-sa/paper"
NU_LAM = 2e-7

CASES = [
    ('cav_L1_base_a0', 'baseline ($a_{\\max}\\!=\\!0.05$, χ-linear σ_t, χ$_\\infty\\!=\\!0.02$)', 4e-9, 'C0'),
    ('cav_L1_varA_a0', 'variant A ($a_{\\max}\\!=\\!0.20$, χ$^{1/4}$ σ_t, χ$_\\infty\\!=\\!1.6\\!\\times\\!10^{-7}$)', 3.2e-14, 'C3'),
    ('cav_L1_varB_a0', 'variant B ($a_{\\max}\\!=\\!0.20$, χ-linear σ_t, χ$_\\infty\\!=\\!1.6\\!\\times\\!10^{-7}$)', 3.2e-14, 'C2'),
]

def load_surface(d):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f'{d}/surface_fluid_naca0012.pvtu'); r.Update()
    g = r.GetOutput(); pd = g.GetPointData()
    p = vtk_to_numpy(g.GetPoints().GetData())
    cp = vtk_to_numpy(pd.GetArray('Cp'))
    cf_arr = vtk_to_numpy(pd.GetArray('Cf'))
    cf_mag = np.linalg.norm(cf_arr, axis=1) if cf_arr.ndim == 2 else np.abs(cf_arr)
    return p[:,0], p[:,2], cp, cf_mag

def load_slice(d):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f'{d}/slice_centerSpan.pvtu'); r.Update()
    g = r.GetOutput(); pd = g.GetPointData()
    nh = vtk_to_numpy(pd.GetArray('nuHat'))
    p  = vtk_to_numpy(g.GetPoints().GetData())
    return p[:,0], p[:,2], nh

x_edges = np.linspace(-0.1, 1.1, 70)

def bin_max(x, z, nh, x_edges, bl_y=0.06):
    mask = np.abs(z) < bl_y
    x, z, nh = x[mask], z[mask], nh[mask]
    centers = 0.5*(x_edges[:-1] + x_edges[1:])
    out = np.full(len(centers), np.nan)
    for i in range(len(centers)):
        m = (x >= x_edges[i]) & (x < x_edges[i+1])
        if m.any(): out[i] = nh[m].max()
    return centers, out

import matplotlib
matplotlib.rcParams.update({'font.size':10.5,'axes.titlesize':11,'axes.labelsize':10.5,
                            'legend.fontsize':9,'xtick.labelsize':9.5,'ytick.labelsize':9.5})

fig, axs = plt.subplots(2, 2, figsize=(14, 9))
ax_cf, ax_cp, ax_nu, ax_N = axs.flatten()

for case, label, nu_inf, color in CASES:
    d = f'{F}/{case}'
    # surface Cf / Cp
    try:
        xs, zs, cp, cf_mag = load_surface(d)
        upper = zs > 0
        order = np.argsort(xs[upper])
        ax_cf.plot(xs[upper][order], cf_mag[upper][order], '-', color=color, lw=1.4, label=label)
        ax_cp.plot(xs[upper][order], cp[upper][order], '-', color=color, lw=1.4, label=label)
    except Exception as e:
        print(f"surface {case}: {e}")
    # slice nuHat
    try:
        xb, zb, nh = load_slice(d)
        xc, max_nh = bin_max(xb, zb, nh, x_edges)
        # gap-fill
        max_nh = maximum_filter1d(np.where(np.isnan(max_nh), -np.inf, max_nh), size=3)
        max_nh = np.where(max_nh == -np.inf, np.nan, max_nh)
        ax_nu.semilogy(xc, max_nh, '-', color=color, lw=1.5, label=label)
        N = np.log(np.maximum(max_nh, 1e-35) / nu_inf)
        ax_N.plot(xc, N, '-', color=color, lw=1.5, label=label)
    except Exception as e:
        print(f"slice {case}: {e}")

# Add χ=1 reference
for ax in [ax_nu]:
    ax.axhline(NU_LAM, color='gray', ls=':', lw=0.8, alpha=0.6)
    ax.text(1.05, NU_LAM*1.3, r'$\chi\!=\!1$', fontsize=9, color='gray', ha='right')
    ax.set_ylim(1e-14, 1e6); ax.set_ylabel(r'$\max_y\,\tilde\nu$')
    ax.set_xlabel('$x/c$'); ax.set_xlim(-0.05, 1.05)
    ax.grid(which='major', alpha=0.4); ax.grid(which='minor', alpha=0.15)

ax_N.set_ylim(-1, 25); ax_N.set_ylabel(r'$N\!=\!\ln(\tilde\nu/\tilde\nu_\infty)$')
ax_N.axhline(np.log(NU_LAM/3.2e-14), color='gray', ls=':', lw=0.8, alpha=0.6)
ax_N.text(1.02, np.log(NU_LAM/3.2e-14)+0.3, r'$\chi\!=\!1$ (var A/B)', fontsize=8.5, color='gray', ha='right')
ax_N.axhline(np.log(NU_LAM/4e-9), color='gray', ls='-.', lw=0.8, alpha=0.6)
ax_N.text(1.02, np.log(NU_LAM/4e-9)+0.3, r'$\chi\!=\!1$ (base)', fontsize=8.5, color='gray', ha='right')
ax_N.set_xlabel('$x/c$'); ax_N.set_xlim(-0.05, 1.05); ax_N.grid(alpha=0.4)

for ax in [ax_cf, ax_cp]:
    ax.set_xlabel('$x/c$'); ax.set_xlim(-0.02, 1.02); ax.grid(alpha=0.4)
ax_cf.set_ylabel('$|C_f|$ upper surface'); ax_cf.set_ylim(0, 0.012)
ax_cp.set_ylabel('$C_p$ upper surface'); ax_cp.invert_yaxis()

ax_cf.legend(fontsize=8.5, frameon=False, loc='upper right')

plt.suptitle('cav L1, NACA 0012, α=0°.  Effect of (a) χ_∞ and (b) χ$^{1/4}$ vs χ-linear σ_t handover',
             fontsize=11, y=0.995)
plt.tight_layout(rect=(0,0,1,0.97))
plt.savefig('/tmp/var_compare.png', dpi=130)
plt.savefig(f'{PD}/figs/variant_compare.pdf')
plt.close()
print('wrote /tmp/var_compare.png')
