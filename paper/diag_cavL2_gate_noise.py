"""Why does cavL2 (only) ignite the marginal NLF fronts early?

For (strL2, cavL1, cavL2) x (a0 lower, a4 upper), probe wall-normal lines along
the chord (same construction as regen_nlf_v2.wallnormal_max_metrics) and, at
the chi-peak of each station, extract every factor of the SA-AI rate kernel at
the a3 constants:
  Gamma  -> sigmoid  sig = 1/(1+exp(-s(Gamma-g_c)))
  ReO,lp -> cliff    1-(Re_c/Re_O)^4  (0 below cliff)
  Gg (VTK double-gradient |lap u|) -> gate Q = 1-sqrt(G(2-G))/(1+cA*Gg)
  chi, and the measured local growth rate dln(chi_max)/dx.
Reports window means per grid to identify the factor carrying the breakaway.
"""
import os, sys, json
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper")
import regen_nlf_v2 as R

B = "/home/qiqi/flexcompute/sa-ai/flow360_a3"
A_MAX, G_C, S, FLOOR, KL, P, CA = 0.19, 1.005, 11.0, 254.0, 6.1, 4.0, 4.0
SCRATCH = os.path.dirname(os.path.abspath(__file__))


def load_with_laplacian(case_d):
    g = R.load_slice_derived(case_d)
    gf = vtk.vtkGradientFilter()
    gf.SetInputData(g)
    gf.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, 'velocity')
    gf.SetResultArrayName('grad_u'); gf.Update()
    g = gf.GetOutput()
    gu = vtk_to_numpy(g.GetPointData().GetArray('grad_u'))
    lap = np.zeros((gu.shape[0], 3))
    for i in range(3):
        comp = np.ascontiguousarray(gu[:, 3*i:3*i+3])
        va = numpy_to_vtk(comp, deep=True); va.SetName(f'g{i}')
        g.GetPointData().AddArray(va)
        gf2 = vtk.vtkGradientFilter()
        gf2.SetInputData(g)
        gf2.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, f'g{i}')
        gf2.SetResultArrayName('gg'); gf2.Update()
        gg = vtk_to_numpy(gf2.GetOutput().GetPointData().GetArray('gg'))
        lap[:, i] = gg[:, 0] + gg[:, 4] + gg[:, 8]
    lm = np.sqrt((lap**2).sum(1)).astype(np.float32)
    va = numpy_to_vtk(lm, deep=True); va.SetName('lap_u_mag')
    g.GetPointData().AddArray(va)
    return g


def station_metrics(case_d, side, n_probe=140, L_probe=0.012):
    nu = float(json.load(open(f"{case_d}/Flow360.json"))['freestream']['muRef'])
    Xm, Zm, up_idx, lo_idx = R.walk_contour_xz(case_d)
    idx = up_idx if side == 'upper' else lo_idx
    xs = Xm[idx]; zs = Zm[idx]
    tx_raw = np.gradient(xs); tz_raw = np.gradient(zs)
    s_ = np.sqrt(tx_raw**2 + tz_raw**2) + 1e-30
    tx, tz = tx_raw/s_, tz_raw/s_
    nx, nz = tz, -tx
    if side == 'upper' and np.mean(nz) < 0:
        nx, nz = -nx, -nz
    elif side == 'lower' and np.mean(nz) > 0:
        nx, nz = -nx, -nz
    M = len(xs)
    dists = np.linspace(1e-6, L_probe, n_probe)
    y0 = R.slice_y_plane(case_d)
    pts_arr = np.empty((M*n_probe, 3))
    for j in range(n_probe):
        d = dists[j]
        pts_arr[j*M:(j+1)*M, 0] = xs + d*nx
        pts_arr[j*M:(j+1)*M, 1] = y0
        pts_arr[j*M:(j+1)*M, 2] = zs + d*nz
    vpts = vtk.vtkPoints(); vpts.SetData(numpy_to_vtk(pts_arr, deep=True))
    poly = vtk.vtkPolyData(); poly.SetPoints(vpts)
    slice_g = load_with_laplacian(case_d)
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(poly); probe.SetSourceData(slice_g); probe.Update()
    pdd = probe.GetOutput().GetPointData()
    valid_ids = vtk_to_numpy(probe.GetValidPoints())
    mask = np.zeros(M*n_probe, bool); mask[valid_ids] = True

    def F(nm):
        a = vtk_to_numpy(pdd.GetArray(nm))
        if a.ndim > 1:
            return np.where(mask[:, None], a, np.nan).reshape(n_probe, M, -1)
        return np.where(mask, a, np.nan).reshape(n_probe, M)

    chi = F('nuHat')/nu
    Gam = F('Gamma'); ReO = F('Re_Omega'); lp = F('lambda_p')
    lapm = F('lap_u_mag'); omg = F('vorticityMagnitude')
    vel = F('velocity'); U2 = (vel**2).sum(-1)
    d = np.broadcast_to(dists[:, None], (n_probe, M))
    Gg = (d*d*lapm)**2/np.maximum(U2 + (omg*d)**2, 1e-30)
    Q = 1.0 - np.sqrt(np.maximum(Gam*(2.0-Gam), 0.0))/(1.0 + CA*Gg)
    Re_c = FLOOR*np.exp(KL*np.maximum(lp, 0.0))
    cliff = np.where(ReO > Re_c, 1.0 - (Re_c/np.maximum(ReO, 1e-9))**P, 0.0)
    sig = 1.0/(1.0 + np.exp(-S*(Gam - G_C)))
    rate_w = A_MAX*sig*Q*np.maximum(cliff, 0.0)*omg

    ipk = np.nanargmax(np.where(np.isnan(chi), -1e30, chi), axis=0)
    cols = np.arange(M)
    out = dict(x=xs, chi_max=chi[ipk, cols], d_peak=dists[ipk])
    for nm, arr in [('Gam', Gam), ('sig', sig), ('Q', Q), ('cliff', cliff),
                    ('Gg', Gg), ('ReO', ReO), ('lp', lp), ('rate_w', rate_w)]:
        out[nm + '_pk'] = arr[ipk, cols]
    out['rate_w_bandmax'] = np.nanmax(rate_w, axis=0)
    out['Gam_bandmax'] = np.nanmax(Gam, axis=0)
    o = np.argsort(xs)
    return {k: v[o] for k, v in out.items()}


CASES = {
    'a0-lower': [('strL2', 'strL2prop_nlf0416_Re4M_a0', 'lower'),
                 ('cavL1', 'cavL1prop_nlf0416_Re4M_a0', 'lower'),
                 ('cavL2', 'cavL2prop_nlf0416_Re4M_a0', 'lower')],
    'a4-upper': [('strL2', 'strL2prop_nlf0416_Re4M_a4', 'upper'),
                 ('cavL1', 'cavL1prop_nlf0416_Re4M_a4', 'upper'),
                 ('cavL2', 'cavL2prop_nlf0416_Re4M_a4', 'upper')],
}

results = {}
for tag, lst in CASES.items():
    results[tag] = {}
    for name, case, side in lst:
        results[tag][name] = station_metrics(f"{B}/{case}", side)
        print(f"[{tag}] {name} probed", flush=True)

WINDOWS = {'a0-lower': (0.15, 0.42), 'a4-upper': (0.04, 0.18)}
for tag in CASES:
    lo, hi = WINDOWS[tag]
    print(f"\n===== {tag}: window x in [{lo}, {hi}] (pre-front) =====")
    print(f"{'grid':>6} {'<Gam_pk>':>9} {'<sig_pk>':>9} {'<Q_pk>':>8} {'<cliff>':>8} "
          f"{'<Gg_pk>':>9} {'<a*w_pk>':>9} {'<a*w_bmax>':>10} {'dlnchi/dx':>10}")
    for name in ['strL2', 'cavL1', 'cavL2']:
        o = results[tag][name]
        m = (o['x'] >= lo) & (o['x'] <= hi) & np.isfinite(o['chi_max'])
        gr = np.polyfit(o['x'][m], np.log(o['chi_max'][m]), 1)[0]
        print(f"{name:>6} {np.nanmean(o['Gam_pk'][m]):9.4f} {np.nanmean(o['sig_pk'][m]):9.4f} "
              f"{np.nanmean(o['Q_pk'][m]):8.4f} {np.nanmean(o['cliff_pk'][m]):8.4f} "
              f"{np.nanmean(o['Gg_pk'][m]):9.4f} {np.nanmean(o['rate_w_pk'][m]):9.4f} "
              f"{np.nanmean(o['rate_w_bandmax'][m]):10.4f} {gr:10.3f}")

np.savez(os.path.join(SCRATCH, 'diag_breakaway.npz'),
         **{f"{tag}/{name}/{k}": v for tag in results for name in results[tag]
            for k, v in results[tag][name].items()})
print("\nsaved diag_breakaway.npz")
