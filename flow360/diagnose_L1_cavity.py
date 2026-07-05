"""Zoom in on Cp near LE/TE and on upper-vs-lower asymmetry for the L1 cavity outlier."""
import numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt

def contour_walk_both(d, af='naca0012'):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{d}/surface_fluid_{af}.pvtu"); r.Update()
    g = r.GetOutput(); p = vtk_to_numpy(g.GetPoints().GetData()); pd = g.GetPointData()
    arrs = {pd.GetArrayName(i): vtk_to_numpy(pd.GetArray(i)) for i in range(pd.GetNumberOfArrays())}
    cf = arrs['Cf']; cf = np.linalg.norm(cf, axis=1) if cf.ndim > 1 else cf
    cp = arrs['Cp']; cp = cp if cp.ndim == 1 else cp[:,0]
    x, y, z = p[:,0], p[:,1], p[:,2]
    s = np.abs(y - y.min()) < 1e-6
    X, Z, CF, CP = x[s], z[s], cf[s], cp[s]
    n = len(X); pts = np.column_stack([X, Z])
    st = int(np.argmin(X)); o = [st]; u = np.zeros(n, bool); u[st] = True
    for _ in range(n-1):
        c = o[-1]; dd = np.sum((pts - pts[c])**2, 1); dd[u] = 1e9
        nx = int(np.argmin(dd)); o.append(nx); u[nx] = True
    o = np.array(o); xo, zo, cfo, cpo = X[o], Z[o], CF[o], CP[o]
    te = int(np.argmax(xo)); b1, b2 = slice(0, te+1), slice(te, n)
    up = b1 if zo[b1].mean() >= zo[b2].mean() else b2
    lo = b2 if up == b1 else b1
    def srt(sl):
        xs = xo[sl]; oo = np.argsort(xs)
        return xs[oo], cfo[sl][oo], cpo[sl][oo], zo[sl][oo]
    return srt(up), srt(lo)

F = "/home/qiqi/flexcompute/aft-sa/flow360"
CASES = [
    ('proper_cav_L0', 'cavity L0',  'k',  '-'),
    ('proper_cav_L1', 'cavity L1',  'C0', '-'),
    ('proper_cav_L2', 'cavity L2',  'C2', '-'),
    ('proper_str_L0', 'O-grid L0',  'k',  '--'),
    ('proper_str_L1', 'O-grid L1',  'C0', '--'),
    ('proper_str_L2', 'O-grid L2',  'C2', '--'),
]

# Compute analytical CDp / CDf / CL by trapezoidal integration of surface Cp and Cf
# CD = ∮ (-Cp n_x + Cf t_x) ds; CL = ∮ (-Cp n_z + Cf t_z) ds; n is outward normal
# For each upper/lower side, the surface points are ordered by x. n_x = -dz/ds, n_z = +dx/ds (upper)
def integrate_forces_one_side(x, z, cp, cf, side='upper'):
    # arc-length param
    dx = np.diff(x); dz = np.diff(z); ds = np.sqrt(dx**2+dz**2)
    xm = 0.5*(x[1:]+x[:-1]); zm = 0.5*(z[1:]+z[:-1])
    cpm = 0.5*(cp[1:]+cp[:-1]); cfm = 0.5*(cf[1:]+cf[:-1])
    # outward unit normal: on upper surface (z>0), n_z > 0 typically
    # tangent = (dx,dz)/ds; outward normal = (-dz, dx)/ds for upper (z>0)
    if side == 'upper':
        nx_unit = -dz/ds; nz_unit = dx/ds
    else:
        # lower surface: outward normal flips
        nx_unit = dz/ds; nz_unit = -dx/ds
    # tangent
    tx_unit = dx/ds; tz_unit = dz/ds
    dCD_pressure = -cpm * nx_unit * ds      # pressure on body in x
    dCD_friction = cfm * tx_unit * ds        # friction tangent in x
    dCL_pressure = -cpm * nz_unit * ds
    dCL_friction = cfm * tz_unit * ds
    # moment about (x_ref, z_ref) = (0.25, 0) (quarter-chord)
    xref, zref = 0.25, 0.0
    rx = xm - xref; rz = zm - zref
    # force per ds in (x, z) at the segment midpoint
    fx = -cpm * nx_unit + cfm * tx_unit
    fz = -cpm * nz_unit + cfm * tz_unit
    # moment around y (out of plane): My = rx*fz - rz*fx
    dCMy = (rx*fz - rz*fx) * ds
    return dict(CDp=dCD_pressure.sum(), CDf=dCD_friction.sum(),
                CLp=dCL_pressure.sum(), CLf=dCL_friction.sum(),
                CMy=dCMy.sum())

# Also compute the running integrals to find where the differences are concentrated
def running_integrals(x, z, cp, cf, side='upper'):
    dx = np.diff(x); dz = np.diff(z); ds = np.sqrt(dx**2+dz**2)
    xm = 0.5*(x[1:]+x[:-1])
    cpm = 0.5*(cp[1:]+cp[:-1]); cfm = 0.5*(cf[1:]+cf[:-1])
    if side == 'upper':
        nx_unit = -dz/ds; nz_unit = dx/ds
    else:
        nx_unit = dz/ds; nz_unit = -dx/ds
    tx_unit = dx/ds; tz_unit = dz/ds
    dCDp = (-cpm * nx_unit * ds); dCDf = (cfm * tx_unit * ds)
    dCLp = (-cpm * nz_unit * ds); dCLf = (cfm * tz_unit * ds)
    return xm, np.cumsum(dCDp), np.cumsum(dCDf), np.cumsum(dCLp), np.cumsum(dCLf)

# panel layout: Cp LE-zoom, Cp TE-zoom, Cf LE-zoom, running CDp(x), running CL(x)
fig, axs = plt.subplots(2, 3, figsize=(15, 8))

print(f"{'case':>16s} {'CDp_up':>9s} {'CDp_lo':>9s} {'CDp_total':>10s} | {'CLp_up':>9s} {'CLp_lo':>9s}")
for d, label, c, ls in CASES:
    (xu, cfu, cpu, zu), (xl, cfl, cpl, zl) = contour_walk_both(f"{F}/{d}")
    fu = integrate_forces_one_side(xu, zu, cpu, cfu, 'upper')
    fl = integrate_forces_one_side(xl, zl, cpl, cfl, 'lower')
    CDp = fu['CDp'] + fl['CDp']; CDf = fu['CDf'] + fl['CDf']
    CLp = fu['CLp'] + fl['CLp']
    print(f"  {d:14s} {fu['CDp']:9.5f} {fl['CDp']:9.5f} {CDp:10.5f} | {fu['CLp']:9.5f} {fl['CLp']:9.5f}")
    # LE zoom: −Cp
    axs[0,0].plot(xu, -cpu, color=c, ls=ls, lw=1.4, label=label)
    axs[0,0].plot(xl, -cpl, color=c, ls=ls, lw=1.0, alpha=0.6)
    # TE zoom: −Cp
    axs[0,1].plot(xu, -cpu, color=c, ls=ls, lw=1.4, label=label)
    axs[0,1].plot(xl, -cpl, color=c, ls=ls, lw=1.0, alpha=0.6)
    # Cf upper near LE
    axs[0,2].plot(xu, cfu, color=c, ls=ls, lw=1.4, label=label)
    # running cumulative CDp(x) - upper surface only
    xm, cCDp_u, cCDf_u, cCLp_u, cCLf_u = running_integrals(xu, zu, cpu, cfu, 'upper')
    xm_l, cCDp_l, cCDf_l, cCLp_l, cCLf_l = running_integrals(xl, zl, cpl, cfl, 'lower')
    axs[1,0].plot(xm, cCDp_u, color=c, ls=ls, lw=1.4, label=label)
    axs[1,1].plot(xm_l, cCDp_l, color=c, ls=ls, lw=1.4, label=label)
    axs[1,2].plot(xm, cCLp_u, color=c, ls=ls, lw=1.4)
    axs[1,2].plot(xm_l, cCLp_l, color=c, ls=ls, lw=1.0, alpha=0.6)

# format
axs[0,0].set_xlim(0, 0.1); axs[0,0].set_xlabel('x/c'); axs[0,0].set_ylabel('-Cp'); axs[0,0].set_title('(a) Cp, LE zoom (solid=upper, faint=lower)'); axs[0,0].grid(alpha=0.3); axs[0,0].legend(fontsize=7, frameon=False, ncol=2)
axs[0,1].set_xlim(0.85, 1.01); axs[0,1].set_xlabel('x/c'); axs[0,1].set_ylabel('-Cp'); axs[0,1].set_title('(b) Cp, TE zoom'); axs[0,1].grid(alpha=0.3)
axs[0,2].set_xlim(0, 0.04); axs[0,2].set_xlabel('x/c'); axs[0,2].set_ylabel('Cf'); axs[0,2].set_title('(c) Cf upper, LE zoom'); axs[0,2].grid(alpha=0.3)
axs[1,0].set_xlim(0, 1); axs[1,0].set_xlabel('x/c'); axs[1,0].set_ylabel(r'cumulative $C_{D,p}$ upper'); axs[1,0].set_title('(d) running CDp(x), upper'); axs[1,0].grid(alpha=0.3); axs[1,0].legend(fontsize=7, frameon=False, ncol=2)
axs[1,1].set_xlim(0, 1); axs[1,1].set_xlabel('x/c'); axs[1,1].set_ylabel(r'cumulative $C_{D,p}$ lower'); axs[1,1].set_title('(e) running CDp(x), lower'); axs[1,1].grid(alpha=0.3)
axs[1,2].set_xlim(0, 1); axs[1,2].set_xlabel('x/c'); axs[1,2].set_ylabel(r'cumulative $C_{L,p}$'); axs[1,2].set_title('(f) running CLp(x), upper(solid)+lower(faint)'); axs[1,2].grid(alpha=0.3)

plt.tight_layout(pad=0.5)
plt.savefig('/tmp/diagnose_L1.png', dpi=140)
plt.savefig('/home/qiqi/flexcompute/aft-sa/paper/figs/diagnose_L1_cavity.pdf')
print("\nwrote /tmp/diagnose_L1.png + paper figs/diagnose_L1_cavity.pdf")
