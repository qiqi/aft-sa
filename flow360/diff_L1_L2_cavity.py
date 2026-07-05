"""Compute Cp(x) and Cf(x) differences between cavity L1, cavity L2, and O-grid L2,
plus running integral of Cp(x)*dx_ds to localize where pressure-drag accumulates."""
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

# load three reference cases
(xL1u, cfL1u, cpL1u, zL1u), (xL1l, cfL1l, cpL1l, zL1l) = contour_walk_both(f"{F}/proper_cav_L1")
(xL2u, cfL2u, cpL2u, zL2u), (xL2l, cfL2l, cpL2l, zL2l) = contour_walk_both(f"{F}/proper_cav_L2")
(xSu,  cfSu,  cpSu,  zSu),  (xSl,  cfSl,  cpSl,  zSl)  = contour_walk_both(f"{F}/proper_str_L2")

# Interpolate L1 and S_L2 onto L2's x-grid for both upper and lower
def interp_to(xref, x, v): return np.interp(xref, x, v)

# Use cavity L2 upper x-grid as reference for the upper-surface comparison
xref = xL2u
cpL1u_i = interp_to(xref, xL1u, cpL1u); cfL1u_i = interp_to(xref, xL1u, cfL1u)
cpSu_i  = interp_to(xref, xSu,  cpSu);  cfSu_i  = interp_to(xref, xSu,  cfSu)
# lower
xref_lo = xL2l
cpL1l_i = interp_to(xref_lo, xL1l, cpL1l); cfL1l_i = interp_to(xref_lo, xL1l, cfL1l)
cpSl_i  = interp_to(xref_lo, xSl,  cpSl);  cfSl_i  = interp_to(xref_lo, xSl,  cfSl)

# Differences  L1_cav - L2_cav  (cavity-mesh effect)  and  L2_cav - L2_str (residual cav vs O-grid)
dCp_u_L1   = cpL1u_i - cpL2u
dCp_u_S    = cpL2u   - cpSu_i
dCp_l_L1   = cpL1l_i - cpL2l
dCp_l_S    = cpL2l   - cpSl_i

# Running cumulative Cp·dx (this contributes to body-x force from pressure)
def running_Cp_dx(x, cp, z, side='upper'):
    # The contribution to body-frame -CFx from pressure: dCFx = -Cp * (-dz) = +Cp * dz on upper, -Cp*dz on lower
    # but actually -CFx is -∫ -Cp n_x ds, where n_x = -dz/ds (upper). So
    # dCFx_pressure_upper = Cp * dz (with dz traveling along surface)
    # The CHORD-WISE projection: just integrate Cp * dz along the surface
    dz = np.diff(z); dx = np.diff(x); cpm = 0.5*(cp[1:]+cp[:-1])
    # For pressure force in body x: integrand = -Cp * n_x ds = -Cp * (-dz) = Cp * dz on upper
    # on lower, n_x = +dz/ds, integrand = -Cp * dz
    if side == 'upper':
        d_fx = cpm * dz
    else:
        d_fx = -cpm * dz
    return 0.5*(x[1:]+x[:-1]), np.cumsum(d_fx)

xmL2u, cumU_L2 = running_Cp_dx(xL2u, cpL2u, zL2u, 'upper')
xmL1u, cumU_L1 = running_Cp_dx(xL1u, cpL1u, zL1u, 'upper')
xmSu,  cumU_S  = running_Cp_dx(xSu,  cpSu,  zSu,  'upper')
xmL2l, cumL_L2 = running_Cp_dx(xL2l, cpL2l, zL2l, 'lower')
xmL1l, cumL_L1 = running_Cp_dx(xL1l, cpL1l, zL1l, 'lower')
xmSl,  cumL_S  = running_Cp_dx(xSl,  cpSl,  zSl,  'lower')

# Total body-x pressure force per case
print("=== body-frame -CFx_pressure (= ∫ Cp dz with sign, both surfaces) ===")
for d, name in [('proper_cav_L0','cav L0'),('proper_cav_L1','cav L1'),('proper_cav_L2','cav L2'),
                ('proper_str_L0','str L0'),('proper_str_L1','str L1'),('proper_str_L2','str L2')]:
    (xu, cfu, cpu, zu), (xl, cfl, cpl, zl) = contour_walk_both(f"{F}/{d}")
    _, cum_u = running_Cp_dx(xu, cpu, zu, 'upper')
    _, cum_l = running_Cp_dx(xl, cpl, zl, 'lower')
    Fx_total = cum_u[-1] + cum_l[-1]
    print(f"  {name}: -CFx_p_upper = {cum_u[-1]:+.5f}  -CFx_p_lower = {cum_l[-1]:+.5f}  total = {Fx_total:+.5f}")

# Plot
fig, axs = plt.subplots(2, 3, figsize=(15, 8))

axs[0,0].plot(xref, dCp_u_L1, 'C0-', lw=1.4, label='cav L1 − cav L2')
axs[0,0].plot(xref, dCp_u_S,  'C2-', lw=1.4, label='cav L2 − str L2')
axs[0,0].axhline(0, color='0.5', lw=0.5)
axs[0,0].set_xlim(0, 1); axs[0,0].set_xlabel('x/c'); axs[0,0].set_ylabel(r'$\Delta C_p$ upper')
axs[0,0].set_title('(a) $\Delta C_p$ upper'); axs[0,0].grid(alpha=0.3); axs[0,0].legend(fontsize=9, frameon=False)

axs[0,1].plot(xref, dCp_u_L1, 'C0-', lw=1.4)
axs[0,1].plot(xref, dCp_u_S,  'C2-', lw=1.4)
axs[0,1].axhline(0, color='0.5', lw=0.5)
axs[0,1].set_xlim(0, 0.1); axs[0,1].set_xlabel('x/c'); axs[0,1].set_ylabel(r'$\Delta C_p$ upper')
axs[0,1].set_title('(b) $\Delta C_p$ upper, LE zoom'); axs[0,1].grid(alpha=0.3)

axs[0,2].plot(xref_lo, dCp_l_L1, 'C0-', lw=1.4, label='cav L1 − cav L2')
axs[0,2].plot(xref_lo, dCp_l_S,  'C2-', lw=1.4, label='cav L2 − str L2')
axs[0,2].axhline(0, color='0.5', lw=0.5)
axs[0,2].set_xlim(0, 1); axs[0,2].set_xlabel('x/c'); axs[0,2].set_ylabel(r'$\Delta C_p$ lower')
axs[0,2].set_title('(c) $\Delta C_p$ lower'); axs[0,2].grid(alpha=0.3); axs[0,2].legend(fontsize=9, frameon=False)

# running -CFx_p (body frame contribution to drag from pressure)
axs[1,0].plot(xmL1u, cumU_L1, 'C0-', lw=1.4, label='cav L1')
axs[1,0].plot(xmL2u, cumU_L2, 'C2-', lw=1.4, label='cav L2')
axs[1,0].plot(xmSu, cumU_S,  'C3--', lw=1.4, label='str L2')
axs[1,0].set_xlim(0, 1); axs[1,0].set_xlabel('x/c'); axs[1,0].set_ylabel(r'$\int_0^x C_p\,dz$ upper')
axs[1,0].set_title('(d) running pressure thrust on upper'); axs[1,0].grid(alpha=0.3); axs[1,0].legend(fontsize=9, frameon=False)

axs[1,1].plot(xmL1u, cumU_L1, 'C0-', lw=1.4)
axs[1,1].plot(xmL2u, cumU_L2, 'C2-', lw=1.4)
axs[1,1].plot(xmSu, cumU_S,  'C3--', lw=1.4)
axs[1,1].set_xlim(0, 0.1); axs[1,1].set_xlabel('x/c'); axs[1,1].set_ylabel(r'$\int_0^x C_p\,dz$ upper')
axs[1,1].set_title('(e) running pressure thrust upper, LE zoom'); axs[1,1].grid(alpha=0.3)

axs[1,2].plot(xmL1l, cumL_L1, 'C0-', lw=1.4, label='cav L1')
axs[1,2].plot(xmL2l, cumL_L2, 'C2-', lw=1.4, label='cav L2')
axs[1,2].plot(xmSl, cumL_S,  'C3--', lw=1.4, label='str L2')
axs[1,2].set_xlim(0, 1); axs[1,2].set_xlabel('x/c'); axs[1,2].set_ylabel(r'$-\int_0^x C_p\,dz$ lower')
axs[1,2].set_title('(f) running pressure drag on lower'); axs[1,2].grid(alpha=0.3); axs[1,2].legend(fontsize=9, frameon=False)

plt.tight_layout(pad=0.5)
plt.savefig('/tmp/diff_L1_L2.png', dpi=140)
plt.savefig('/home/qiqi/flexcompute/aft-sa/paper/figs/diff_L1_L2.pdf')
print("\nwrote /tmp/diff_L1_L2.png + paper figs/diff_L1_L2.pdf")
