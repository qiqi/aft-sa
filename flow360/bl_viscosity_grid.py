"""Plot BL profiles d vs (u_mag, omega, total_visc) at multiple x.
Total viscosity = mu_lam * (1 + chi * f_v1(chi)) where f_v1 = chi^3/(chi^3+c_v1^3).
Look at how viscosity changes through transition.
"""
import vtk, numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

NU = 2.5e-8
C_V1 = 7.1

def f_v1(chi):
    chi = np.asarray(chi); chi3 = chi**3
    return chi3 / (chi3 + C_V1**3)

def column(case, x_t, dx=0.005, z_min=0.025, d_max=0.01):
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(f'/home/qiqi/flexcompute/aft-sa/flow360/{case}/slice_centerSpan.pvtu'); r.Update()
    g = r.GetOutput(); pd = g.GetPointData()
    p = vtk_to_numpy(g.GetPoints().GetData())
    wd_all = vtk_to_numpy(pd.GetArray('wallDistance'))
    m = (np.abs(p[:,0]-x_t) < dx) & (p[:,2] > z_min) & (wd_all < d_max) & (wd_all > 1e-7)
    if m.sum() < 5: return None
    wd = wd_all[m]
    om = vtk_to_numpy(pd.GetArray('vorticityMagnitude'))[m]
    nh = vtk_to_numpy(pd.GetArray('nuHat'))[m]
    vel = vtk_to_numpy(pd.GetArray('velocity'))[m]
    um = np.linalg.norm(vel, axis=1)
    order = np.argsort(wd)
    return wd[order], um[order], om[order], nh[order]

xs = [0.28, 0.30, 0.32, 0.33, 0.34, 0.36, 0.38, 0.40, 0.42, 0.50]
colors = plt.cm.viridis(np.linspace(0, 0.95, len(xs)))

matplotlib.rcParams.update({'font.size': 11})
fig, axs = plt.subplots(2, 4, figsize=(20, 11))

for irow, (case, label) in enumerate([('cavprop_nlf0416_Re4M_a2p5', 'cav (unstructured)'),
                                       ('strprop_nlf0416_Re4M_a2p5', 'str (O-grid)')]):
    ax_u = axs[irow, 0]; ax_om = axs[irow, 1]; ax_chi = axs[irow, 2]; ax_nu = axs[irow, 3]
    for xi_, x_t in enumerate(xs):
        r = column(case, x_t)
        if r is None: continue
        d, u, om, nh = r
        chi = nh / NU
        # total viscosity ratio: nu_total / nu_lam = 1 + chi * f_v1(chi)
        nu_ratio = 1.0 + chi * f_v1(chi)
        ax_u.plot(u, d, '-', color=colors[xi_], lw=1.4, label=f'x={x_t}')
        ax_om.plot(om, d, '-', color=colors[xi_], lw=1.4)
        ax_chi.plot(chi, d, '-', color=colors[xi_], lw=1.4)
        ax_nu.plot(nu_ratio, d, '-', color=colors[xi_], lw=1.4)
    ax_u.set_xlabel('|V|'); ax_u.set_ylabel('d')
    ax_u.set_title(f'{label}: u(d)')
    ax_u.set_xlim(0, 0.18); ax_u.set_ylim(0, 0.003)
    ax_u.grid(alpha=0.3)
    ax_u.legend(fontsize=8, ncol=2, frameon=False, loc='lower right')
    ax_om.set_xlabel('|ω|'); ax_om.set_title(f'{label}: ω(d)')
    ax_om.set_xscale('log'); ax_om.set_xlim(1, 1e4); ax_om.set_ylim(0, 0.003); ax_om.grid(alpha=0.3, which='both')
    ax_chi.set_xlabel('χ = ν̃/ν'); ax_chi.set_title(f'{label}: χ(d)')
    ax_chi.set_xscale('log'); ax_chi.set_xlim(1e-5, 200); ax_chi.set_ylim(0, 0.003); ax_chi.grid(alpha=0.3, which='both')
    ax_chi.axvline(1.0, color='gray', ls=':', lw=1)
    ax_chi.axvline(C_V1, color='red', ls=':', lw=1)
    ax_nu.set_xlabel(r'$\nu_{\rm tot}/\nu_{\rm lam} = 1+\chi f_{v1}(\chi)$')
    ax_nu.set_title(f'{label}: total viscosity ratio')
    ax_nu.set_xscale('log'); ax_nu.set_xlim(0.5, 200); ax_nu.set_ylim(0, 0.003); ax_nu.grid(alpha=0.3, which='both')
    ax_nu.axvline(1.0, color='gray', ls=':', lw=1, label='laminar (no μ_t)')
    ax_nu.axvline(2.0, color='red', ls=':', lw=0.7, alpha=0.7)
    ax_nu.text(2.05, 0.0028, '2×', color='red', alpha=0.7, fontsize=8)

plt.suptitle('NLF(1)-0416 α=2.5° upper-surface BL: u, ω, χ, total ν profiles', fontsize=12)
plt.tight_layout()
plt.savefig('/tmp/bl_viscosity.png', dpi=130)
plt.close()
print('wrote /tmp/bl_viscosity.png')
