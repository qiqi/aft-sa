"""Compare Cf vs x along NLF α=9 surface across pseudo-step snapshots and
mfoil reference.  Shows whether long-step runs converge to mfoil's laminar
lower-BL prediction.
"""
import os, sys, pickle, glob, numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt

NU = 2.5e-8
SD = "/home/qiqi/flexcompute/aft-sa/flow360/cavL2prop_nlf0416_Re4M_a9/snapshots"
B = "/home/qiqi/flexcompute/aft-sa/flow360"

def cf_from_pvtu(pvtu):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(pvtu); r.Update()
    g = r.GetOutput(); pd = g.GetPointData()
    pts = vtk_to_numpy(g.GetPoints().GetData())
    Cf = vtk_to_numpy(pd.GetArray('Cf'))
    Cfmag = np.linalg.norm(Cf, axis=1) if Cf.ndim>1 else np.abs(Cf)
    return pts, Cfmag

steps = sorted([int(f.split('_s')[-1].split('.')[0])
                for f in glob.glob(f"{SD}/surface_fluid_nlf0416_s*.pvtu")])
print(f"Available steps: {steps}")

# Load mfoil α=9 reference
mn = pickle.load(open(f"{B}/mfoil_nlf0416_Re4M.pkl", 'rb'))
ma9 = mn[9.0]
xtr_lo = ma9['xtr_lower']

fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
colors = plt.cm.viridis(np.linspace(0.0, 0.95, len(steps)))

for ax, side, key in [(axes[0], 'upper', 'upper'), (axes[1], 'lower', 'lower')]:
    for si, step in enumerate(steps):
        pvtu = f"{SD}/surface_fluid_nlf0416_s{step}.pvtu"
        pts, Cf = cf_from_pvtu(pvtu)
        m = (pts[:,2] > 0) if side == 'upper' else (pts[:,2] < 0)
        if m.sum() < 50: continue
        order = np.argsort(pts[m,0])
        ax.plot(pts[m,0][order], Cf[m][order], '-', color=colors[si],
                label=f'step {step}', lw=1.0, alpha=0.85)
    # mfoil reference
    ax.plot(ma9[key]['x'], np.abs(ma9[key]['cf']), 'k-', lw=2.0,
            label='mfoil α=9', alpha=0.8)
    # mfoil xtr line
    xtr = ma9['xtr_upper'] if side == 'upper' else ma9['xtr_lower']
    ax.axvline(xtr, ls=':', color='k', alpha=0.5, label=f'mfoil x_tr={xtr:.2f}')
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.025)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.set_ylabel(f'|Cf|, {side}')
axes[-1].set_xlabel('x/c')
fig.suptitle('NLF α=9 cavL2 — Cf vs pseudo-step (front recession)')
plt.tight_layout()
plt.savefig('/tmp/cavL2_a9_cf_convergence.png', dpi=130, bbox_inches='tight')
print('wrote /tmp/cavL2_a9_cf_convergence.png')
