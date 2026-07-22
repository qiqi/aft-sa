"""Which term of Gamma_g = (d^2 |lap u|)^2 / (|u|^2 + (omega d)^2) carries the
cavL2 sliver noise?

Term-by-term audit at matched wall distance, a4-upper trio (a3 fleet), stations
x in [0.04, 0.18]:
  1. |u|(d), omega(d), |lap u|(d): medians across stations per grid at each
     height. Which ingredient breaks away on cavL2?
  2. Physical reference from each station's OWN probed profile: u_t(d)
     differentiated smoothly in 1D -> du_t/dn (should match omega in a BL) and
     d2u_t/dn2 (should match |lap u|). Ratios probe/1D = per-derivative noise
     factor per grid.
  3. Substitution test: recompute Gamma_g and Q with the 1D-clean curvature in
     place of |lap u| (everything else probed). If Q collapses across grids,
     the noise enters through the second derivative alone.
"""
import sys, json
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper")
sys.path.insert(0, "/tmp/claude-1006/-home-qiqi-flexcompute/15845519-8cb3-4677-8f3c-47bcc8951d95/scratchpad")
import regen_nlf_v2 as R
from diag_cavL2_breakaway import load_with_laplacian

B = "/home/qiqi/flexcompute/sa-ai/flow360_a3"
CA = 4.0
X_LO, X_HI, NSTA = 0.04, 0.18, 36
dists = np.linspace(2e-6, 6.0e-4, 160)          # uniform for clean 1D FD
REPORT_D = [100e-6, 140e-6, 200e-6, 300e-6]


def probe_case(case):
    cd = f"{B}/{case}"
    Xm, Zm, up_idx, _ = R.walk_contour_xz(cd)
    xs = Xm[up_idx]; zs = Zm[up_idx]
    tx = np.gradient(xs); tz = np.gradient(zs)
    s_ = np.hypot(tx, tz) + 1e-30
    tx, tz = tx/s_, tz/s_
    nx, nz = tz, -tx
    if np.mean(nz) < 0:
        nx, nz = -nx, -nz
    keep = np.linspace(X_LO, X_HI, NSTA)
    ii = np.array([np.argmin(np.abs(xs - x0)) for x0 in keep])
    xs, zs, nx, nz, tx, tz = xs[ii], zs[ii], nx[ii], nz[ii], tx[ii], tz[ii]
    M, K = len(xs), len(dists)
    y0 = R.slice_y_plane(cd)
    pts = np.empty((M*K, 3))
    for j in range(K):
        pts[j*M:(j+1)*M, 0] = xs + dists[j]*nx
        pts[j*M:(j+1)*M, 1] = y0
        pts[j*M:(j+1)*M, 2] = zs + dists[j]*nz
    vp = vtk.vtkPoints(); vp.SetData(numpy_to_vtk(pts, deep=True))
    poly = vtk.vtkPolyData(); poly.SetPoints(vp)
    g = load_with_laplacian(cd)
    pr = vtk.vtkProbeFilter(); pr.SetInputData(poly); pr.SetSourceData(g)
    pr.SetTolerance(1e-3); pr.ComputeToleranceOff(); pr.Update()
    pd = pr.GetOutput().GetPointData()
    valid_ids = vtk_to_numpy(pr.GetValidPoints())
    mask = np.zeros(M*K, bool); mask[valid_ids] = True

    def F(nm):
        a = vtk_to_numpy(pd.GetArray(nm))
        if a.ndim > 1:
            return np.where(mask[:, None], a, np.nan).reshape(K, M, -1)
        return np.where(mask, a, np.nan).reshape(K, M)

    vel = F('velocity')
    omg = F('vorticityMagnitude')
    lap = F('lap_u_mag')
    U = np.sqrt((vel**2).sum(-1))
    # tangential velocity per station
    ut = vel[..., 0]*tx[None, :] + vel[..., 2]*tz[None, :]
    # 1D smooth derivatives along the probe line (uniform spacing)
    h = dists[1] - dists[0]
    dut = np.gradient(ut, h, axis=0)
    d2ut = np.gradient(dut, h, axis=0)
    return dict(x=xs, U=U, omg=omg, lap=lap, ut=ut,
                dut=np.abs(dut), d2ut=np.abs(d2ut))


def med(a):
    return float(np.nanmedian(a))


def p90(a):
    return float(np.nanpercentile(a, 90))


cases = [('strL2', 'strL2prop_nlf0416_Re4M_a4'),
         ('cavL1', 'cavL1prop_nlf0416_Re4M_a4'),
         ('cavL2', 'cavL2prop_nlf0416_Re4M_a4')]
D = {}
for name, case in cases:
    D[name] = probe_case(case)
    print(f"{name} probed", flush=True)

ji = [np.argmin(np.abs(dists - dv)) for dv in REPORT_D]

print("\n=== 1. Term-by-term, medians across stations at matched height ===")
print(f"{'field':>14} {'d(e-6)':>7} {'strL2':>10} {'cavL1':>10} {'cavL2':>10} {'cavL2/str':>10}")
for fname, key in [('|u|', 'U'), ('omega', 'omg'), ('|lap u|', 'lap')]:
    for j, dv in zip(ji, REPORT_D):
        vals = [med(D[n][key][j]) for n in ('strL2', 'cavL1', 'cavL2')]
        print(f"{fname:>14} {dv*1e6:7.0f} {vals[0]:10.3g} {vals[1]:10.3g} {vals[2]:10.3g} {vals[2]/vals[0]:10.2f}")

print("\n=== 2. Noise factor vs each station's own 1D profile ===")
print("  r1 = omega/|du_t/dn|, r2 = |lap u|/|d2u_t/dn2|; per grid: median [P90] over stations x heights d in [80,300]e-6")
band = (dists >= 80e-6) & (dists <= 300e-6)
for n in ('strL2', 'cavL1', 'cavL2'):
    o = D[n]
    r1 = o['omg'][band]/np.maximum(o['dut'][band], 1e-12)
    r2 = o['lap'][band]/np.maximum(o['d2ut'][band], 1e-12)
    print(f"  {n}: r1 = {med(r1):6.3f} [{p90(r1):7.3f}]   r2 = {med(r2):7.2f} [{p90(r2):9.2f}]")

print("\n=== 3. Substitution test: Q with probed lap vs 1D-clean curvature ===")
print(f"{'grid':>7} {'d(e-6)':>7} {'Q(probed lap)':>14} {'Q(clean 1D)':>12}")
for n in ('strL2', 'cavL1', 'cavL2'):
    o = D[n]
    d2 = dists[:, None]**2
    den = o['U']**2 + (o['omg']*dists[:, None])**2
    # Gamma for the pinch factor sqrt(G(2-G)) -- from probed fields
    Gam = 2.0*(o['omg']*dists[:, None])**2/np.maximum(den, 1e-30)
    pinch = np.sqrt(np.maximum(Gam*(2.0-Gam), 0.0))
    Gg_probe = (d2*o['lap'])**2/np.maximum(den, 1e-30)
    Gg_clean = (d2*o['d2ut'])**2/np.maximum(den, 1e-30)
    Qp = 1.0 - pinch/(1.0 + CA*Gg_probe)
    Qc = 1.0 - pinch/(1.0 + CA*Gg_clean)
    for j, dv in zip(ji[:3], REPORT_D[:3]):
        print(f"{n:>7} {dv*1e6:7.0f} {med(Qp[j]):14.4f} {med(Qc[j]):12.4f}")

print("\n=== 4. Does refinement converge the Laplacian noise? (cavL1 -> cavL2) ===")
for j, dv in zip(ji, REPORT_D):
    l1, l2 = med(D['cavL1']['lap'][j]), med(D['cavL2']['lap'][j])
    s2 = med(D['strL2']['lap'][j])
    print(f"  d={dv*1e6:4.0f}e-6: median |lap u|  str={s2:9.3g}  cavL1={l1:9.3g}  cavL2={l2:9.3g}  L2/L1={l2/l1:6.2f}")
