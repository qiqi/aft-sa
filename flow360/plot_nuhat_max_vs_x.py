"""Plot max_y(ν̃) along x for unified vs baseline^4 on log scale.

The linear-regime change-of-variable predicts ν̃_unified(x) = ν̃_baseline(x)^4,
so plotting max_y(ν̃_uni) and (max_y(ν̃_base))^4 on the same log axis tests
where that prediction holds (overlap) and where it breaks (divergence).

Sample the centerSpan slice — bin in x, take max over y in each bin.
"""
import vtk, numpy as np, os
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

F = "/home/qiqi/flexcompute/aft-sa/flow360"
PD = "/home/qiqi/flexcompute/aft-sa/paper"

MESHES = [('cav_L0','unstructured L0'), ('cav_L1','unstructured L1'),
          ('str_L0','O-grid L0'),       ('str_L1','O-grid L1')]

def load_slice_nuHat(d):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f'{d}/slice_centerSpan.pvtu'); r.Update()
    g = r.GetOutput(); pd = g.GetPointData()
    p = vtk_to_numpy(g.GetPoints().GetData())
    nh = vtk_to_numpy(pd.GetArray('nuHat'))
    return p[:,0], p[:,2], nh  # x, z, nuHat

def max_along_x(x, z, nh, x_edges):
    """For each bin in x, max(nuHat) over all (z, points in that x-bin)."""
    out = np.zeros(len(x_edges)-1)
    for i in range(len(x_edges)-1):
        m = (x >= x_edges[i]) & (x < x_edges[i+1])
        if m.any(): out[i] = nh[m].max()
        else: out[i] = np.nan
    return 0.5*(x_edges[:-1] + x_edges[1:]), out

# x bins (focus on -0.2..1.2 to capture LE and TE neighborhood)
x_edges = np.linspace(-0.2, 1.2, 200)

import matplotlib
matplotlib.rcParams.update({'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10,
                            'legend.fontsize': 8.5, 'xtick.labelsize': 9, 'ytick.labelsize': 9})

# Both baseline and unified have the same effective freestream nuHat (Flow360 floor):
NU_INF = 4e-9
# Linear-regime envelope theory:  ν̃_uni(x) = ν̃_∞ * exp(4·N) = ν̃_base(x)^4 / ν̃_∞^3
# so the comparable quantity is ν̃_base(x)^4 / ν̃_∞^3.

fig, axs = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
for ax, (mkey, label) in zip(axs.flat, MESHES):
    try:
        x_b, z_b, nh_b = load_slice_nuHat(f'{F}/{mkey}_base_a0')
        xc_b, max_b = max_along_x(x_b, z_b, nh_b, x_edges)
    except Exception as e:
        print(f"base {mkey} fail: {e}"); continue
    try:
        x_u, z_u, nh_u = load_slice_nuHat(f'{F}/{mkey}_uni_a0')
        xc_u, max_u = max_along_x(x_u, z_u, nh_u, x_edges)
    except Exception as e:
        print(f"uni {mkey} fail: {e}"); continue

    # Plot max_y(nuHat) for unified
    ax.semilogy(xc_u, np.maximum(max_u, 1e-30), '-', color='C3', lw=1.6,
                label=r'unified  $\max_y \tilde\nu$')
    # Plot baseline itself
    ax.semilogy(xc_b, np.maximum(max_b, 1e-30), '-', color='C0', lw=1.6,
                label=r'baseline  $\max_y \tilde\nu$')
    # Plot envelope prediction ν̃_base^4 / ν̃_∞^3
    pred = np.where(np.isnan(max_b), np.nan, max_b**4 / NU_INF**3)
    ax.semilogy(xc_b, np.maximum(pred, 1e-30), '--', color='k', lw=1.4,
                label=r'envelope theory  $\tilde\nu_b^4 / \tilde\nu_\infty^3$')
    # χ=1 reference
    ax.axhline(2e-7, color='gray', ls=':', lw=0.8, alpha=0.6)
    ax.text(1.08, 2.2e-7, r'$\chi\!=\!1$', fontsize=8, color='gray', ha='right')

    ax.set_title(label)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(1e-9, 1e6)
    ax.grid(which='major', alpha=0.4); ax.grid(which='minor', alpha=0.15)

for ax in axs[1]: ax.set_xlabel('$x/c$')
for ax in axs[:,0]: ax.set_ylabel(r'$\max_y \tilde\nu$')

# Single legend
h = [Line2D([],[],color='C3',ls='-',lw=2,label=r'unified  $\max_y \tilde\nu$  (a$_\max\!=\!0.2$, $\chi^{1/4}\sigma_t$)'),
     Line2D([],[],color='C0',ls='-',lw=2,label=r'baseline  $\max_y \tilde\nu$  (a$_\max\!=\!0.05$, $\chi$-linear $\sigma_t$)'),
     Line2D([],[],color='k',ls='--',lw=2,label=r'envelope theory  $\tilde\nu_b^4 / \tilde\nu_\infty^3$'),
     Line2D([],[],color='gray',ls=':',lw=1.5,label=r'$\chi\!=\!1$ reference')]
axs[0,1].legend(handles=h, frameon=False, loc='lower right', fontsize=8)

plt.suptitle(r'$\max_y \tilde\nu(x)$  —  unified vs baseline & baseline$^4/\tilde\nu_\infty^3$  (NACA 0012, $\alpha\!=\!0^\circ$, common $\tilde\nu_\infty\!=\!4\!\times\!10^{-9}$)', fontsize=10.5)
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.savefig('/tmp/nuhat_max_vs_x.png', dpi=140)
plt.savefig(f'{PD}/figs/unified_vs_baseline_nuhat.pdf')
plt.close()
print("wrote /tmp/nuhat_max_vs_x.png")
