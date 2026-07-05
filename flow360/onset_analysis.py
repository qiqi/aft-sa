"""Detailed at-d* analysis at a series of x stations spanning the onset and the
maximum-gap region. We pick:
  x = 0.040  pre-onset (both grids still at freestream seed)
  x = 0.045  earliest onset on each grid (when activation just turns on)
  x = 0.050  the activation switch
  x = 0.060  saturation begins
  x = 0.070  early-divergence (gap forming)
  x = 0.100  near-max ν̂ gap (~13×)
"""
import numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy

NU_REF = 0.2 / 1e6
A_MAX = 0.05; S_PARAM = 35.0; RE0 = 1000.0; K_PARAM = 50.0; G_C = 1.04
RE_NA = 5.0; M_EXP = 2.0

def activation(Re, G):
    if Re <= RE_NA: return 0.0
    z = S_PARAM*(np.log10(Re/RE0)/K_PARAM + G - G_C) + M_EXP*np.log(max(1-RE_NA/Re,1e-30))
    return A_MAX / (1.0 + np.exp(-z))

def load_with_vorticity(d):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{d}/slice_centerSpan.pvtu"); r.Update()
    g = r.GetOutput()
    gf = vtk.vtkGradientFilter(); gf.SetInputData(g); gf.SetInputScalars(0, "velocity")
    gf.SetComputeVorticity(True); gf.SetVorticityArrayName("vorticity"); gf.Update()
    out = gf.GetOutput(); pd = out.GetPointData()
    arrs = {pd.GetArrayName(i): vtk_to_numpy(pd.GetArray(i)) for i in range(pd.GetNumberOfArrays())}
    return vtk_to_numpy(out.GetPoints().GetData()), arrs

def airfoil_polyline(d):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{d}/surface_fluid_naca0012.pvtu"); r.Update()
    g = r.GetOutput(); p = vtk_to_numpy(g.GetPoints().GetData())
    x, y, z = p[:,0], p[:,1], p[:,2]
    s = np.abs(y - y.min()) < 1e-6; X, Z = x[s], z[s]
    n = len(X); pts2 = np.column_stack([X, Z])
    st = int(np.argmin(X)); o = [st]; u = np.zeros(n, bool); u[st] = True
    for _ in range(n-1):
        c = o[-1]; dd = np.sum((pts2 - pts2[c])**2, 1); dd[u] = 1e9; nx = int(np.argmin(dd)); o.append(nx); u[nx] = True
    o = np.array(o)
    return X[o], Z[o]

def project_signed(pts, x_contour, z_contour):
    N = len(pts); xz = np.column_stack([pts[:,0], pts[:,2]])
    Ncont = len(x_contour)
    A = np.column_stack([x_contour, z_contour]); B = np.roll(A, -1, axis=0); seg = B - A
    seg_len2 = (seg**2).sum(axis=1); seg_len2[seg_len2 == 0] = 1e-30
    d_min = np.full(N, np.inf); proj_x = np.empty(N); proj_z = np.empty(N)
    chunk = 1000
    for s in range(0, Ncont, chunk):
        e = min(s+chunk, Ncont)
        AP = xz[:, None, :] - A[None, s:e, :]
        t = (AP * seg[None, s:e, :]).sum(axis=2) / seg_len2[None, s:e]; t = np.clip(t, 0.0, 1.0)
        proj = A[None, s:e, :] + t[:, :, None] * seg[None, s:e, :]
        dd2 = ((xz[:, None, :] - proj)**2).sum(axis=2)
        i_loc = np.argmin(dd2, axis=1); d_loc = dd2[np.arange(N), i_loc]
        better = d_loc < d_min
        d_min[better] = d_loc[better]
        proj_x[better] = proj[better, i_loc[better], 0]; proj_z[better] = proj[better, i_loc[better], 1]
    d = np.sqrt(d_min); sign = np.where(pts[:,2] > proj_z, 1.0, -1.0)
    return proj_x, d * sign

def at_x(pts, arrs, xw, dw, xt, dx=0.003, max_d=0.012):
    vel = arrs['velocity']; vort = arrs['vorticity']; nuHat = arrs['nuHat']
    u_mag = np.sqrt(vel[:,0]**2 + vel[:,2]**2); omega = np.abs(vort[:,1])
    m = (np.abs(xw - xt) < dx) & (dw > 0) & (dw < max_d)
    if m.sum() < 5: return None
    d = dw[m]; u = u_mag[m]; om = omega[m]; nu = nuHat[m]
    ix = np.argsort(d)
    return d[ix], u[ix], om[ix], nu[ix]

def bin_smooth(d, val, nbins=40, max_d=0.01):
    bins = np.linspace(0, max_d, nbins+1); db = 0.5*(bins[:-1]+bins[1:])
    out = np.full(nbins, np.nan)
    for j in range(nbins):
        m = (d >= bins[j]) & (d < bins[j+1])
        if m.any(): out[j] = np.median(val[m])
    return db, out

DCAV = "proper_refineA4_cavity_xxfine"; DSTR = "refineA4_struct_sxxfine"
pc, ac = load_with_vorticity(DCAV); ps, as_ = load_with_vorticity(DSTR)
xc_cav, zc_cav = airfoil_polyline(DCAV); xc_str, zc_str = airfoil_polyline(DSTR)
xw_cav, d_cav = project_signed(pc, xc_cav, zc_cav)
xw_str, d_str = project_signed(ps, xc_str, zc_str)

def analyze_at(xt):
    rows = {}
    for gname, pts, arrs, xw, dw in [('cav', pc, ac, xw_cav, d_cav),
                                     ('str', ps, as_, xw_str, d_str)]:
        prof = at_x(pts, arrs, xw, dw, xt, dx=0.003)
        if prof is None: continue
        d, u, om, nu = prof
        db, ub = bin_smooth(d, u); _, ob = bin_smooth(d, om); _, nb = bin_smooth(d, nu)
        u_e = np.nanmax(ub) if np.isfinite(np.nanmax(ub)) else 1.0
        above = (ub/u_e >= 0.99) & (db > 1e-4)
        d99 = db[np.where(above)[0][0]] if above.any() else 0.005
        # find d* where ν̂ peaks inside the BL
        bl = (db > 5e-5) & (db < 1.5*d99) & np.isfinite(nb)
        if not bl.any(): continue
        i = int(np.argmax(nb[bl]))
        d_s = db[bl][i]; nu_s = float(nb[bl][i])
        u_s = float(np.interp(d_s, db, ub))
        om_s = float(np.interp(d_s, db, ob))
        Re = d_s*d_s * om_s / NU_REF
        od = om_s * d_s
        G = 2*od*od / (u_s*u_s + od*od + 1e-30)
        a = activation(Re, G)
        rows[gname] = dict(d=d_s, d99=d99, nu=nu_s, u=u_s, u_e=u_e, om=om_s, od=od,
                           Re=Re, G=G, a=a, aw=a*om_s,
                           dN_dx=(a*om_s)/u_s if u_s > 0 else 0)
    return rows

X_STATIONS = [0.040, 0.045, 0.050, 0.060, 0.070, 0.080, 0.100]
print(f"\n{'='*100}")
print(f"{'detailed comparison at d* (where ν̂ peaks in BL)':^100}")
print(f"{'='*100}\n")
for xt in X_STATIONS:
    r = analyze_at(xt)
    if 'cav' not in r or 'str' not in r:
        print(f"x={xt:.3f}: incomplete data"); continue
    c, s = r['cav'], r['str']
    nu_ratio = c['nu']/s['nu']
    print(f"=== x/c = {xt:.3f}  (ν̂* ratio cav/O-grid = {nu_ratio:.2f}) ===")
    print(f"{'quantity':<14s}  {'cavity':>10s}  {'O-grid':>10s}  {'ratio':>7s}")
    print(f"{'-'*52}")
    print(f"{'d*':<14s}  {c['d']:>10.5f}  {s['d']:>10.5f}  {c['d']/s['d']:>7.3f}")
    print(f"{'d99':<14s}  {c['d99']:>10.5f}  {s['d99']:>10.5f}  {c['d99']/s['d99']:>7.3f}")
    print(f"{'d*/δ99':<14s}  {c['d']/c['d99']:>10.3f}  {s['d']/s['d99']:>10.3f}  --")
    print(f"{'ω at d*':<14s}  {c['om']:>10.2f}  {s['om']:>10.2f}  {c['om']/s['om']:>7.3f}")
    print(f"{'ω·d':<14s}  {c['od']:>10.4f}  {s['od']:>10.4f}  {c['od']/s['od']:>7.3f}")
    print(f"{'Re_Ω':<14s}  {c['Re']:>10.1f}  {s['Re']:>10.1f}  {c['Re']/s['Re']:>7.3f}")
    print(f"{'|u| at d*':<14s}  {c['u']:>10.4f}  {s['u']:>10.4f}  {c['u']/s['u']:>7.3f}")
    print(f"{'u_edge':<14s}  {c['u_e']:>10.4f}  {s['u_e']:>10.4f}  {c['u_e']/s['u_e']:>7.3f}")
    print(f"{'Γ':<14s}  {c['G']:>10.4f}  {s['G']:>10.4f}  {c['G']/s['G']:>7.3f}")
    print(f"{'Γ - g_c':<14s}  {c['G']-G_C:>10.4f}  {s['G']-G_C:>10.4f}  {(c['G']-G_C)/(s['G']-G_C+1e-30):>7.3f}")
    print(f"{'a (activation)':<14s}  {c['a']:>10.4f}  {s['a']:>10.4f}  {c['a']/(s['a']+1e-30):>7.3f}")
    print(f"{'a·ω':<14s}  {c['aw']:>10.2f}  {s['aw']:>10.2f}  {c['aw']/(s['aw']+1e-30):>7.3f}")
    print(f"{'dN/dx = a·ω/u':<14s}  {c['dN_dx']:>10.2f}  {s['dN_dx']:>10.2f}  {c['dN_dx']/(s['dN_dx']+1e-30):>7.3f}")
    # primary driver diagnosis
    if abs(c['om']/s['om'] - 1) > 0.05:
        print(f"  → DRIVER: ω differs by {100*(c['om']/s['om']-1):.0f}%")
    if abs(c['od']/s['od'] - 1) > 0.05:
        print(f"  → DRIVER: ω·d differs by {100*(c['od']/s['od']-1):.0f}%")
    if abs(c['u']/s['u'] - 1) > 0.03:
        print(f"  → DRIVER: |u| differs by {100*(c['u']/s['u']-1):.0f}%")
    print()
