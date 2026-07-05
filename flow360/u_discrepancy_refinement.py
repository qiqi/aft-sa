"""Trace the |u| at d* and u_edge across the full refinement series.
If the discrepancy comes from inside-BL discretization, it should decrease with
refinement. If it's an inviscid LE-shape issue, it stays put.
"""
import numpy as np, vtk, os
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt

NU_REF = 0.2 / 1e6

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

def at_x_profile(pts, arrs, xw, dw, xt, dx=0.003, max_d=0.012):
    vel = arrs['velocity']
    u_mag = np.sqrt(vel[:,0]**2 + vel[:,2]**2)
    m = (np.abs(xw - xt) < dx) & (dw > 0) & (dw < max_d)
    if m.sum() < 5: return None
    ix = np.argsort(dw[m])
    return dw[m][ix], u_mag[m][ix]

def bin_smooth(d, val, nbins=60, max_d=0.01):
    bins = np.linspace(0, max_d, nbins+1); db = 0.5*(bins[:-1]+bins[1:])
    out = np.full(nbins, np.nan)
    for j in range(nbins):
        m = (d >= bins[j]) & (d < bins[j+1])
        if m.any(): out[j] = np.median(val[m])
    return db, out

# Refinement series
CAV_LEVELS = [('coarse', 35912), ('med', 69772), ('fine', 136288),
              ('xfine', 271564), ('xxfine', 537348)]
STR_LEVELS = [('coarse', 8791), ('med', 24651), ('fine', 63441),
              ('sxfine', 124657), ('sxxfine', 254881)]

# x stations to probe — upstream + onset zone
X_PROBE = [0.01, 0.02, 0.03, 0.04, 0.05]
D_PROBE = 0.00038   # the active depth at x=0.05

def extract_one(d, label, ncell):
    """Returns dict: per x station, (u_edge, |u| at d=0.00038)."""
    if not os.path.exists(f"{d}/slice_centerSpan.pvtu"): return None
    pts, arrs = load_with_vorticity(d)
    xc, zc = airfoil_polyline(d)
    xw, dw = project_signed(pts, xc, zc)
    out = {'label': label, 'ncell': ncell, 'profile_x05': None, 'data': {}}
    for xt in X_PROBE:
        prof = at_x_profile(pts, arrs, xw, dw, xt)
        if prof is None: continue
        d_arr, u_arr = prof
        db, ub = bin_smooth(d_arr, u_arr)
        u_edge = float(np.nanmax(ub)) if np.isfinite(np.nanmax(ub)) else None
        u_at_dprobe = float(np.interp(D_PROBE, db, ub))
        out['data'][xt] = dict(u_edge=u_edge, u_at_d=u_at_dprobe)
    # store full u profile at x=0.05 for plotting
    prof = at_x_profile(pts, arrs, xw, dw, 0.05)
    if prof is not None:
        d_arr, u_arr = prof
        db, ub = bin_smooth(d_arr, u_arr, nbins=80, max_d=0.008)
        out['profile_x05'] = (db, ub)
    return out

print("Extracting cavity refinement series...", flush=True)
cav_results = []
for tag, nc in CAV_LEVELS:
    d = f"proper_refineA4_cavity_{tag}"
    r = extract_one(d, f"cav-{tag}", nc)
    if r is not None: cav_results.append(r)
    print(f"  {tag}: done", flush=True)

print("\nExtracting struct refinement series...", flush=True)
str_results = []
for tag, nc in STR_LEVELS:
    d = f"refineA4_struct_{tag}"
    r = extract_one(d, f"str-{tag}", nc)
    if r is not None: str_results.append(r)
    print(f"  {tag}: done", flush=True)

# Table: |u| and u_edge at x=0.05 for each refinement level
print(f"\n{'='*88}")
print(f"  |u| at d=0.00038 AND u_edge at x=0.05, across refinement levels")
print(f"{'='*88}")
print(f"\n{'mesh':<14s} {'cells':>8s}  {'u_edge(0.05)':>14s}  {'|u|@d*':>10s}  {'|u|/u_edge':>11s}")
print("-"*60)
for r in cav_results:
    d = r['data'].get(0.05)
    if d is None: continue
    print(f"{r['label']:<14s} {r['ncell']:>8d}  {d['u_edge']:>14.5f}  {d['u_at_d']:>10.5f}  {d['u_at_d']/d['u_edge']:>11.4f}")
print()
for r in str_results:
    d = r['data'].get(0.05)
    if d is None: continue
    print(f"{r['label']:<14s} {r['ncell']:>8d}  {d['u_edge']:>14.5f}  {d['u_at_d']:>10.5f}  {d['u_at_d']/d['u_edge']:>11.4f}")

# Upstream tracing — u_edge across x for each refinement level
print(f"\n{'='*88}")
print(f"  u_edge upstream of x=0.05, across grid refinement (does it converge?)")
print(f"{'='*88}")
print(f"\n{'x':>5s}", end="")
for r in cav_results: print(f"  {r['label']:>10s}", end="")
print()
for xt in X_PROBE:
    print(f"{xt:5.3f}", end="")
    for r in cav_results:
        d = r['data'].get(xt)
        print(f"  {(d['u_edge'] if d else float('nan')):>10.5f}", end="")
    print()
print()
print(f"{'x':>5s}", end="")
for r in str_results: print(f"  {r['label']:>10s}", end="")
print()
for xt in X_PROBE:
    print(f"{xt:5.3f}", end="")
    for r in str_results:
        d = r['data'].get(xt)
        print(f"  {(d['u_edge'] if d else float('nan')):>10.5f}", end="")
    print()

# Plot: u(d) profiles at x=0.05 for all refinement levels, overlaid
fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))
colors_cav = plt.cm.Greys(np.linspace(0.4, 1.0, len(cav_results)))
colors_str = plt.cm.Reds(np.linspace(0.4, 1.0, len(str_results)))

ax = axs[0]
for r, c in zip(cav_results, colors_cav):
    if r['profile_x05'] is not None:
        db, ub = r['profile_x05']
        ax.plot(ub, db, '-', color=c, lw=1.2, label=f"cav-{r['label'].split('-')[1]} ({r['ncell']//1000}k)")
for r, c in zip(str_results, colors_str):
    if r['profile_x05'] is not None:
        db, ub = r['profile_x05']
        ax.plot(ub, db, '--', color=c, lw=1.2, label=f"str-{r['label'].split('-')[1]} ({r['ncell']//1000}k)")
ax.set_xlim(0, 0.32); ax.set_ylim(0, 0.005)
ax.axhline(D_PROBE, color='b', ls=':', lw=0.8); ax.text(0.005, D_PROBE+1e-4, 'd*=3.8e-4', fontsize=8, color='b')
ax.set_xlabel(r'$|u|$'); ax.set_ylabel(r'$d$ (wall-normal)')
ax.set_title(r'(a) velocity profile at $x/c=0.05$', fontsize=10)
ax.grid(alpha=0.3); ax.legend(fontsize=6.5, frameon=False, ncol=2, loc='upper left')

# Panel 2: u_edge vs ncells^(-1/2) at x=0.05 (and at x=0.03, x=0.02)
ax = axs[1]
for xt, marker in [(0.02, 's'), (0.03, '^'), (0.05, 'o')]:
    h_cav = np.array([1/np.sqrt(r['ncell']) for r in cav_results if xt in r['data']])
    ue_cav = np.array([r['data'][xt]['u_edge'] for r in cav_results if xt in r['data']])
    h_str = np.array([1/np.sqrt(r['ncell']) for r in str_results if xt in r['data']])
    ue_str = np.array([r['data'][xt]['u_edge'] for r in str_results if xt in r['data']])
    ax.plot(h_cav, ue_cav, marker+'-', color='k', ms=5, label=f'cav x={xt}')
    ax.plot(h_str, ue_str, marker+'--', color='r', ms=5, label=f'str x={xt}')
ax.set_xlabel(r'$h \sim 1/\sqrt{N_{\rm cell}}$'); ax.set_ylabel(r'$u_e$ at this $x$')
ax.set_title('(b) edge velocity convergence', fontsize=10)
ax.legend(fontsize=7, frameon=False, ncol=2); ax.grid(alpha=0.3); ax.set_xlim(0, None)

# Panel 3: |u| at d=0.00038 vs h at x=0.05
ax = axs[2]
h_cav = np.array([1/np.sqrt(r['ncell']) for r in cav_results if 0.05 in r['data']])
u_cav = np.array([r['data'][0.05]['u_at_d'] for r in cav_results if 0.05 in r['data']])
h_str = np.array([1/np.sqrt(r['ncell']) for r in str_results if 0.05 in r['data']])
u_str = np.array([r['data'][0.05]['u_at_d'] for r in str_results if 0.05 in r['data']])
ax.plot(h_cav, u_cav, 'ko-', ms=6, label='cavity')
ax.plot(h_str, u_str, 'r^--', ms=6, label='O-grid')
ax.set_xlabel(r'$h \sim 1/\sqrt{N_{\rm cell}}$'); ax.set_ylabel(r'$|u|$ at $d^*\!=\!0.00038$, $x=0.05$')
ax.set_title('(c) inner-BL $|u|$ vs refinement', fontsize=10)
ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.3); ax.set_xlim(0, None)

plt.tight_layout(pad=0.5)
plt.savefig('/tmp/u_discrepancy_refinement.png', dpi=140)
print("\nwrote /tmp/u_discrepancy_refinement.png")
