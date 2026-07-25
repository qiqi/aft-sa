"""Why does max_y(S_hat*g) show a double peak inside the Eppler bubble with a
sharp dip near the chi handover?  (annotated round 6; RESPONSES.md
2026-07-25 ~13:15 UTC)

Probes the full y-resolved P = S_hat*g field through the bubble on the
structured L2 alpha=5 solution.  Verdict: the two peaks are two DIFFERENT
shear layers -- the lifted laminar layer over the recirculation core
(peak 1) and the turbulent reattachment layer (peak 2) -- and the dip is
the handover between them: the eddy viscosity arriving with the chi =
1 -> c_v1 crossing diffuses exactly the inflection/shear that carries the
outer branch before the reattachment branch has formed, and the y-max of a
decaying and a growing hump has a sharp V at their crossing.

  python3 diag_sg_doublepeak.py   -> /tmp/sg_doublepeak.png
"""
import os
import sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'repro', 'cfd'))
import regen_eppler_v2 as R

FR = os.environ.get("SAAI_CFD_ROOT", "/home/qiqi/flexcompute/sa-ai/flow360_fr")
case = f'{FR}/strL2prop_eppler387_Re200k_a5'
NU = 0.1 / 2e5

r = vtk.vtkXMLPUnstructuredGridReader()
r.SetFileName(f'{case}/volume.pvtu'); r.Update()
vol = r.GetOutput()
zup, zlo, _ = R.airfoil_surfaces(case)
xs = np.linspace(0.30, 0.85, 140)
zs = zup(xs)
dz = np.gradient(zs, xs)
tx, tz = 1 / np.sqrt(1 + dz**2), dz / np.sqrt(1 + dz**2)
nx, nz = -tz, tx
d = np.linspace(1e-5, 0.012, 120)
pts = np.stack([xs[:, None] + nx[:, None] * d[None, :],
                np.zeros((len(xs), len(d))),
                zs[:, None] + nz[:, None] * d[None, :]], axis=-1).reshape(-1, 3)
pd = vtk.vtkPolyData(); vp = vtk.vtkPoints()
vp.SetData(numpy_to_vtk(pts, deep=1)); pd.SetPoints(vp)
pf = vtk.vtkProbeFilter(); pf.SetInputData(pd); pf.SetSourceData(vol); pf.Update()
out = pf.GetOutput().GetPointData()
vel = vtk_to_numpy(out.GetArray('velocity')).reshape(len(xs), len(d), 3)
nuh = vtk_to_numpy(out.GetArray('nuHat')).reshape(len(xs), len(d))
ut = vel[:, :, 0] * tx[:, None] + vel[:, :, 2] * tz[:, None]
P = np.zeros((len(xs), len(d)))
for i in range(len(xs)):
    u = ut[i]; dud = np.gradient(u, d); d2u = np.gradient(dud, d)
    X = np.abs(u); Y = d * dud; Z = 0.5 * d**2 * d2u
    Rr = np.sqrt(X * X + Y * Y + Z * Z) + 1e-30
    P[i] = (Y / np.sqrt(X * X + Y * Y + 1e-30)) * ((Y - X - Z) / Rr)
chi = nuh / NU
Pmax = P.max(1); yarg = d[P.argmax(1)]
chimax = chi.max(1)
i1 = int(np.argmax(chimax >= 1.0))

fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
im = axs[0].pcolormesh(xs, d, P.T, cmap='magma', vmin=0, vmax=0.6)
axs[0].plot(xs, yarg, 'c-', lw=1)
axs[0].set_ylabel('$d$ (heat: $P$; cyan: argmax)')
plt.colorbar(im, ax=axs[0])
axs[1].semilogy(xs, np.clip(Pmax, 1e-3, None))
axs[1].axvline(xs[i1], color='r', ls='--', label='$\\chi=1$ (band max)')
axs[1].set_ylabel('$\\max_y P$'); axs[1].grid(alpha=0.3); axs[1].legend()
axs[2].semilogy(xs, np.clip(chimax, 1e-4, None))
axs[2].axhline(1, color='r', ls='--'); axs[2].axhline(7.1, color='g', ls=':')
axs[2].set_ylabel('band $\\max\\chi$'); axs[2].grid(alpha=0.3)
axs[2].set_xlabel('$x/c$')
plt.tight_layout()
plt.savefig('/tmp/sg_doublepeak.png', dpi=110)
print('wrote /tmp/sg_doublepeak.png')
