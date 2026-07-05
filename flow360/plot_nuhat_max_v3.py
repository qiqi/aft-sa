"""Amplification-factor (N = ln(ν̃/ν̃_∞)) comparison.

Theory in the linear regime (σ_t=0, μ_t feedback negligible):
   dN/dx = a(Re_Ω, Γ) · ω_peak / U
With a_uni = 4·a_base, we expect  N_uni(x) = 4·N_base(x).  This is the cleanest
way to see the factor-of-4 relationship; once χ approaches O(1) the eddy-viscosity
feedback on the mean flow drops ω and the prediction breaks.

Both runs got the SAME freestream ν̃_∞ = 4×10⁻⁹ (the unified-case χ_∞=1.6×10⁻⁷ in
the JSON did not propagate through the SDK to the solver — both effectively used
χ_∞=0.02). With a common ν̃_∞ the factor-of-4 in N is the only signature of the
a_max change.
"""
import vtk, numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

F = "/home/qiqi/flexcompute/aft-sa/flow360"
PD = "/home/qiqi/flexcompute/aft-sa/paper"

MESHES = [('cav_L0','unstructured L0'),('cav_L1','unstructured L1'),
          ('str_L0','O-grid L0'),       ('str_L1','O-grid L1')]

NU_LAM = 2e-7        # ν_lam in Flow360 internal units
NU_INF = 4e-9        # actual freestream ν̃ (chi_inf = 0.02 in both runs)

def load_slice(d):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f'{d}/slice_centerSpan.pvtu'); r.Update()
    g = r.GetOutput(); pd = g.GetPointData()
    nh = vtk_to_numpy(pd.GetArray('nuHat'))
    p = vtk_to_numpy(g.GetPoints().GetData())
    return p[:,0], p[:,2], nh

def bin_max(x, z, nh, x_edges, *, bl_y=0.06):
    mask = np.abs(z) < bl_y
    x, z, nh = x[mask], z[mask], nh[mask]
    centers = 0.5*(x_edges[:-1] + x_edges[1:])
    out = np.full(len(centers), np.nan)
    for i in range(len(centers)):
        m = (x >= x_edges[i]) & (x < x_edges[i+1])
        if m.any(): out[i] = nh[m].max()
    return centers, out

x_edges = np.linspace(-0.1, 1.1, 70)

import matplotlib
matplotlib.rcParams.update({'font.size': 10.5, 'axes.titlesize': 11, 'axes.labelsize': 10.5,
                            'legend.fontsize': 9, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5})

# 2-row layout: top = max ν̃ (log), bottom = N = ln(ν̃/ν̃_∞)
fig, axs = plt.subplots(4, 2, figsize=(13, 13), sharex=True)

for col, (mkey, label) in enumerate(MESHES[:2]):  # this only works for 2 cols
    pass

# Simpler: 2 cols (grid family) × 4 rows (L0 max, L0 N, L1 max, L1 N)?
# Let's do 4×2: cols = grid (cav, str), rows = (L0 ν̃, L0 N, L1 ν̃, L1 N).
plt.close()
fig, axs = plt.subplots(4, 2, figsize=(13, 14), sharex=True)

for col, fam in enumerate([('cav', 'unstructured'), ('str', 'O-grid')]):
    famkey, famlab = fam
    for row_pair, level in enumerate(['L0', 'L1']):
        mkey = f'{famkey}_{level}'
        try:
            xb, zb, nb = load_slice(f'{F}/{mkey}_base_a0')
            xu, zu, nu = load_slice(f'{F}/{mkey}_uni_a0')
        except Exception as e:
            print(f'skip {mkey}: {e}'); continue
        xc, mb = bin_max(xb, zb, nb, x_edges)
        _,  mu = bin_max(xu, zu, nu, x_edges)
        valid = (~np.isnan(mb)) & (~np.isnan(mu))
        xc_v, mb_v, mu_v = xc[valid], mb[valid], mu[valid]
        N_b = np.log(np.maximum(mb_v, 1e-30) / NU_INF)
        N_u = np.log(np.maximum(mu_v, 1e-30) / NU_INF)

        # Row 2·row_pair: max ν̃ on log scale
        ax_n = axs[2*row_pair, col]
        ax_n.semilogy(xc_v, mb_v, '-',  color='C0', lw=1.7, label='baseline')
        ax_n.semilogy(xc_v, mu_v, '-',  color='C3', lw=1.7, label='unified')
        # Theory: ν̃_uni = ν̃_∞·(ν̃_base/ν̃_∞)^4 — only in linear regime χ_b<0.5
        chi_b = mb_v / NU_LAM
        linear_mask = chi_b < 0.5
        pred = NU_INF * (mb_v / NU_INF)**4
        pred_lin = np.where(linear_mask, pred, np.nan)
        ax_n.semilogy(xc_v, pred_lin, '--', color='k', lw=1.7,
                      label=r'theory $\tilde\nu_\infty(\tilde\nu_b/\tilde\nu_\infty)^4$  (χ$_b\!<\!0.5$)')
        ax_n.axhline(NU_LAM, color='gray', ls=':', lw=0.8, alpha=0.6)
        ax_n.text(1.06, NU_LAM*1.3, r'$\chi=1$', fontsize=8.5, color='gray', ha='right')
        ax_n.set_ylim(NU_INF*0.5, 1e6)
        ax_n.set_ylabel(r'$\max_y\,\tilde\nu$', fontsize=10.5)
        ax_n.set_title(f'{famlab} {level} — $\\tilde\\nu(x)$', fontsize=11)
        ax_n.grid(which='major', alpha=0.4); ax_n.grid(which='minor', alpha=0.15)

        # Row 2·row_pair+1: N = ln(ν̃/ν̃_∞) on linear scale, with 4·N_base prediction
        ax_N = axs[2*row_pair + 1, col]
        ax_N.plot(xc_v, N_b,        '-',  color='C0', lw=1.7, label=r'$N_{\rm base}\!=\!\ln(\tilde\nu_b/\tilde\nu_\infty)$')
        ax_N.plot(xc_v, N_u,        '-',  color='C3', lw=1.7, label=r'$N_{\rm uni}\!=\!\ln(\tilde\nu_u/\tilde\nu_\infty)$')
        ax_N.plot(xc_v, 4 * N_b,    '--', color='k',  lw=1.7, label=r'$4\,N_{\rm base}$  (theory)')
        ax_N.axhline(0, color='0.6', lw=0.6)
        ax_N.axhline(np.log(NU_LAM/NU_INF), color='gray', ls=':', lw=0.8, alpha=0.6)
        ax_N.text(1.06, np.log(NU_LAM/NU_INF)+0.2, r'$\chi=1$', fontsize=8.5, color='gray', ha='right')
        ax_N.set_ylim(-1, 16)
        ax_N.set_ylabel(r'$N\!=\!\ln(\tilde\nu/\tilde\nu_\infty)$', fontsize=10.5)
        ax_N.set_title(f'{famlab} {level} — amplification factor', fontsize=11)
        ax_N.grid(alpha=0.4)

for ax in axs[-1, :]: ax.set_xlabel('$x/c$')

# Legend (top right)
h_top = [Line2D([],[],color='C0',lw=2,label='baseline ($a_\\max{=}0.05$)'),
         Line2D([],[],color='C3',lw=2,label='unified  ($a_\\max{=}0.2$)'),
         Line2D([],[],color='k',ls='--',lw=2,label=r'theory $\tilde\nu_\infty(\tilde\nu_b/\tilde\nu_\infty)^4$ (linear regime)'),
         Line2D([],[],color='gray',ls=':',lw=1.0,label=r'$\chi=1$ reference')]
h_bot = [Line2D([],[],color='C0',lw=2,label='$N_{\\rm base}$'),
         Line2D([],[],color='C3',lw=2,label='$N_{\\rm uni}$'),
         Line2D([],[],color='k',ls='--',lw=2,label='$4 N_{\\rm base}$ (linear-regime theory)')]
axs[0,0].legend(handles=h_top, fontsize=8.5, frameon=False, loc='lower right')
axs[1,0].legend(handles=h_bot, fontsize=8.5, frameon=False, loc='upper left')

plt.suptitle(r'Amplification comparison — NACA 0012, $\alpha\!=\!0^\circ$, both runs $\tilde\nu_\infty\!=\!4\!\times\!10^{-9}$.  '
             r'Top rows: $\max_y\,\tilde\nu$ (log).  Bottom rows: amplification factor $N$.  '
             r'Theory: $N_{\rm uni}\!=\!4N_{\rm base}$ in linear regime.', fontsize=10.5, y=0.995)
plt.tight_layout(rect=(0, 0, 1, 0.97))
plt.savefig('/tmp/nuhat_N_compare.png', dpi=130)
plt.savefig(f'{PD}/figs/unified_vs_baseline_nuhat.pdf')
plt.close()
print('wrote /tmp/nuhat_N_compare.png')
