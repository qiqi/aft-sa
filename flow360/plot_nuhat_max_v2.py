"""max_y(ν̃)(x) and max_y(μ_t/μ)(x) for unified vs baseline, with the linear-regime
envelope-theory prediction ν̃_uni = ν̃_base^4 / ν̃_∞^3 plotted ONLY where it is valid
(χ_base = ν̃_base/ν < 0.1, i.e. before the eddy-viscosity feedback breaks linearity).

Aggregation: for each x-bin, take the MAX of ν̃ over all slice points in that bin.
Suppress noise by using LARGER bins where there are few points (median over neighbors
to fill gaps).
"""
import vtk, numpy as np, os
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import maximum_filter1d

F = "/home/qiqi/flexcompute/aft-sa/flow360"
PD = "/home/qiqi/flexcompute/aft-sa/paper"

MESHES = [('cav_L0','unstructured L0'),('cav_L1','unstructured L1'),
          ('str_L0','O-grid L0'),       ('str_L1','O-grid L1')]

# Constants from the case JSON: muRef/rhoRef = 1/Re·M·... → ν_nondim ≈ 2e-7 in Flow360 units
# At ν̃ = 4e-9 (the actual freestream Flow360 honored), χ_∞ = 4e-9/2e-7 = 0.02.
NU_LAM = 2e-7                     # laminar ν in Flow360 non-dim units
NU_INF = 4e-9                     # actual freestream ν̃ observed (both runs got the same)

def load_slice(d):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f'{d}/slice_centerSpan.pvtu'); r.Update()
    g = r.GetOutput(); pd = g.GetPointData()
    nh = vtk_to_numpy(pd.GetArray('nuHat'))
    p = vtk_to_numpy(g.GetPoints().GetData())
    return p[:,0], p[:,2], nh

def max_along_x(x, z, nh, x_edges, *, near_bl_only=False, bl_y=0.05):
    """Max of nuHat per x-bin. Optionally restrict to a near-wall band."""
    if near_bl_only:
        mask = np.abs(z) < bl_y
        x, z, nh = x[mask], z[mask], nh[mask]
    centers = 0.5*(x_edges[:-1] + x_edges[1:])
    out = np.full(len(centers), np.nan)
    for i in range(len(centers)):
        m = (x >= x_edges[i]) & (x < x_edges[i+1])
        if m.any(): out[i] = nh[m].max()
    return centers, out

# x-binning: coarser bins, plus a smoothing maximum filter to plug gaps
x_edges = np.linspace(-0.15, 1.15, 100)

import matplotlib
matplotlib.rcParams.update({'font.size': 10.5, 'axes.titlesize': 11, 'axes.labelsize': 10.5,
                            'legend.fontsize': 9, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5})

fig, axs = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
for ax, (mkey, label) in zip(axs.flat, MESHES):
    try:
        x_b, z_b, nh_b = load_slice(f'{F}/{mkey}_base_a0')
        x_u, z_u, nh_u = load_slice(f'{F}/{mkey}_uni_a0')
    except Exception as e:
        print(f"skip {mkey}: {e}"); continue
    # Restrict to BL region (|z| < 0.05) and ahead-of-TE x window
    xc, max_b = max_along_x(x_b, z_b, nh_b, x_edges, near_bl_only=True, bl_y=0.06)
    _,  max_u = max_along_x(x_u, z_u, nh_u, x_edges, near_bl_only=True, bl_y=0.06)
    # Forward-fill NaNs with neighbor max (single-bin gaps from sparse slice points)
    def fill_gaps(y):
        y2 = y.copy()
        for _ in range(3):
            y2 = maximum_filter1d(np.where(np.isnan(y2), -np.inf, y2), size=3)
        y2[y2 == -np.inf] = np.nan
        return y2
    max_b = fill_gaps(max_b)
    max_u = fill_gaps(max_u)

    # Plot baseline and unified
    ax.semilogy(xc, max_b, '-',  color='C0', lw=1.7, label='baseline')
    ax.semilogy(xc, max_u, '-',  color='C3', lw=1.7, label='unified')

    # Envelope theory: ν̃_uni = ν̃_base^4 / ν̃_∞^3.  Valid ONLY in linear regime (χ_base ≪ 1).
    chi_base = max_b / NU_LAM
    pred = max_b**4 / NU_INF**3
    valid_linear = chi_base < 0.1  # mask where theory holds
    pred_lin = np.where(valid_linear, pred, np.nan)
    pred_full = np.where(~np.isnan(max_b), pred, np.nan)
    ax.semilogy(xc, pred_lin, '--', color='k', lw=2.0,
                label=r'theory  $\tilde\nu_b^4/\tilde\nu_\infty^3$  (χ$_b\!<\!0.1$)')
    ax.semilogy(xc, pred_full, ':', color='0.5', lw=1.0,
                label=r'theory (extrapolated past linear regime)')

    # χ=1 reference: ν̃ = ν_lam = 2e-7
    ax.axhline(NU_LAM, color='gray', ls=':', lw=0.8, alpha=0.7)
    ax.text(1.02, NU_LAM*1.4, r'$\chi\!=\!1$', fontsize=9, color='gray', ha='left', va='bottom')
    # ν̃_∞ reference
    ax.axhline(NU_INF, color='gray', ls=':', lw=0.5, alpha=0.4)
    ax.text(1.02, NU_INF*1.4, r'$\tilde\nu_\infty$', fontsize=8, color='gray', ha='left', va='bottom')

    ax.set_title(label)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(1e-9, 1e6)
    ax.grid(which='major', alpha=0.4); ax.grid(which='minor', alpha=0.15)

for ax in axs[1]: ax.set_xlabel('$x/c$')
for ax in axs[:,0]: ax.set_ylabel(r'$\max_y \tilde\nu$  (Flow360 units)')
# Single composite legend on top-right
h = [Line2D([],[],color='C0',lw=2,label=r'baseline ($a_\max\!=\!0.05$, $\chi$-linear $\sigma_t$)'),
     Line2D([],[],color='C3',lw=2,label=r'unified  ($a_\max\!=\!0.2$, $\chi^{1/4}$ $\sigma_t$)'),
     Line2D([],[],color='k',ls='--',lw=2,label=r'envelope theory $\tilde\nu_b^4/\tilde\nu_\infty^3$ (linear regime $\chi_b\!<\!0.1$)'),
     Line2D([],[],color='0.5',ls=':',lw=1.5,label=r'envelope theory extrapolated past linear regime'),
     Line2D([],[],color='gray',ls=':',lw=1.0,label=r'$\chi\!=\!1$ / $\tilde\nu_\infty$ reference')]
axs[0,1].legend(handles=h, frameon=False, loc='lower right', fontsize=8.5)

plt.suptitle(r'$\max_y \tilde\nu(x)$ in the BL — unified vs baseline & envelope theory  '
             r'(both runs got the same $\tilde\nu_\infty\!=\!4\!\times\!10^{-9}$; the JSON $\chi_\infty\!=\!1.6\!\times\!10^{-7}$ for unified did not propagate)',
             fontsize=10, y=0.995)
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.savefig('/tmp/nuhat_v2.png', dpi=140); plt.savefig(f'{PD}/figs/unified_vs_baseline_nuhat.pdf')
plt.close()
print("wrote /tmp/nuhat_v2.png")
