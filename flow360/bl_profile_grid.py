"""Plot BL profiles d vs u_mag at multiple x locations on the upper surface of
NLF(1)-0416 alpha=2.5, comparing cav and str meshes side by side.
"""
import vtk, numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

NU = 2.5e-8

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

xs = [0.28, 0.30, 0.32, 0.33, 0.34, 0.36, 0.38, 0.40, 0.42]
colors = plt.cm.viridis(np.linspace(0, 0.95, len(xs)))

matplotlib.rcParams.update({'font.size': 10.5})
fig, axs = plt.subplots(1, 4, figsize=(18, 6))

for col_idx, (case, label) in enumerate([('cavprop_nlf0416_Re4M_a2p5', 'cav (unstr)'),
                                          ('strprop_nlf0416_Re4M_a2p5', 'str (O-grid)')]):
    ax_u = axs[2*col_idx + 0]
    ax_om = axs[2*col_idx + 1]
    for xi_, x_t in enumerate(xs):
        r = column(case, x_t)
        if r is None: continue
        d, u, om, nh = r
        ax_u.plot(u, d, '-', color=colors[xi_], lw=1.4, label=f'x={x_t}')
        ax_om.plot(om, d, '-', color=colors[xi_], lw=1.4, label=f'x={x_t}')
    ax_u.set_xlabel('|V|'); ax_u.set_ylabel('d (wall distance)')
    ax_u.set_title(f'{label}: u(d) profiles')
    ax_u.set_xlim(0, 0.18); ax_u.set_ylim(0, 0.003)
    ax_u.grid(alpha=0.3); ax_u.legend(fontsize=8, frameon=False, ncol=2)
    ax_om.set_xlabel('|ω|'); ax_om.set_ylabel('d')
    ax_om.set_title(f'{label}: ω(d) profiles')
    ax_om.set_xlim(1, 5000); ax_om.set_xscale('log')
    ax_om.set_ylim(0, 0.003)
    ax_om.grid(alpha=0.3, which='both')

plt.suptitle('NLF(1)-0416 α=2.5° upper-surface BL profiles at several x', fontsize=12)
plt.tight_layout()
plt.savefig('/tmp/bl_profile_grid.png', dpi=130)
plt.close()
print('wrote /tmp/bl_profile_grid.png')
