"""Closed-streamline re-seeding measurement (Sec. epphandover): inside the
low-Re Eppler bubbles the reversed 'dead air' carries chi far ABOVE the
incoming amplification envelope -- nuHat diffuses across the shear layer
into the recirculation and convects back upstream, re-entering the
amplifying band elevated. The effect is absent at high Re.

Two measurements per Reynolds number (structured L2, alpha=5 sweep):
1. seed surplus = median chi in the reversed region (first third of the
   bubble) / max chi in the attached BL just upstream of separation;
2. the kernel's own production in that dead-air region (sphere indicators
   + onset gate evaluated from the slice), to discriminate
   amplification-driven vs advection-diffusion-driven.

RESULT (2026-07-24): surplus 16x/29x at Re=60k/100k, 0.03x/0.01x at
300k/460k -- the flip sits at the bursting boundary; dead-air production
is onset-gated to ~zero everywhere (median effective rate <= 1e-10), so
the low-Re reservoir is advection-diffusion fed, not amplification fed.
Contour signature: at low Re the chi front is bottom-led (cold front,
reversed flow carries chi upstream along the wall); at high Re top-led
(warm front) -- readable in the Appendix C chi sheets.

  python3 measure_lsb_reseeding.py
"""
import json
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

FR = '/home/qiqi/flexcompute/sa-ai/flow360_fr'
AMAX, C, A, B, K, W = 0.19, 2600.0, 175.0, 2.0, 0.712, 0.35


def softmin(x1, x2):
    return x1 * x2 / np.sqrt(x1**2 + x2**2)


def load(p):
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(p)
    r.Update()
    return r.GetOutput()


def grad(g0, field, name):
    gf = vtk.vtkGradientFilter()
    gf.SetInputData(g0)
    gf.SetInputScalars(vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, field)
    gf.SetResultArrayName(name)
    gf.Update()
    return vtk_to_numpy(gf.GetOutput().GetPointData().GetArray(name))


def measure(Rk):
    case = f"{FR}/sweep_strL2_Re{Rk}k_a5"
    g0 = load(f"{case}/slice_centerSpan.pvtu")
    pd = g0.GetPointData()
    pts = vtk_to_numpy(g0.GetPoints().GetData())
    vel = vtk_to_numpy(pd.GetArray('velocity'))
    om = vtk_to_numpy(pd.GetArray('vorticityMagnitude')).astype(float)
    d = vtk_to_numpy(pd.GetArray('wallDistance')).astype(float)
    nu = vtk_to_numpy(pd.GetArray('nuHat')).astype(float)
    mu = json.load(open(f"{case}/Flow360.json"))['freestream']['muRef']
    chi = nu / mu
    gw = grad(g0, 'vorticityMagnitude', 'gradW')
    nhat = grad(g0, 'wallDistance', 'gradD')
    nhat = nhat / np.maximum(np.linalg.norm(nhat, axis=1), 1e-12)[:, None]
    X = np.linalg.norm(vel, axis=1)
    Y = om * d
    Z = 0.5 * d**2 * np.einsum('ij,ij->i', nhat, gw)
    R = np.sqrt(X**2 + Y**2 + Z**2) + 1e-30
    P = (Y / np.sqrt(X**2 + Y**2 + 1e-60)) * (Y - X - Z) / R
    ReOm = d**2 * om / mu
    ReOmC = K * softmin(C, A + B / np.maximum(P, 1e-6)**2)
    rate = AMAX * np.clip(P, 0, 1) * 0.5 * (1 + np.tanh((ReOm / ReOmC - 1) / W))

    x = pts[:, 0]
    up = pts[:, 1] if abs(pts[:, 1]).max() > abs(pts[:, 2]).max() else pts[:, 2]
    u = vel[:, 0]
    band = (x > 0) & (x < 1) & (up > 0) & (up < 0.2)
    rev = band & (u < 0)
    xs, xe = x[rev].min(), x[rev].max()
    dead = rev & (x < xs + (xe - xs) / 3)
    pre = band & (x > xs - 0.06) & (x < xs - 0.02)
    surplus = float(np.median(chi[dead]) / chi[pre].max())
    return xs, xe, surplus, float(np.median(rate[dead])), float(np.percentile(rate[dead], 95))


if __name__ == '__main__':
    print(f"{'Re':>6} {'x_sep':>6} {'x_rev_end':>9} {'seed surplus':>13} "
          f"{'dead-air rate med':>18} {'95th':>10}")
    for Rk in (60, 100, 300, 460):
        xs, xe, s, rm, r95 = measure(Rk)
        print(f"{Rk:>5}k {xs:>6.3f} {xe:>9.3f} {s:>12.2f}x {rm:>18.2e} {r95:>10.2e}")
