"""Deep trace of the Gamma_g spike at one node: geometry, wiggle, gradient
oscillation, smooth/wiggle decomposition, and per-edge accounting.
Run after replay_gate3_kernel.py (imports its module state)."""
import numpy as np
import replay_gate3_kernel as RP   # executes the replay (fast enough) and exposes state

X, U2d, wd, om = RP.X, RP.U2d, RP.wd, RP.om
adj_n, adj_e, indptr = RP.adj_n, RP.adj_e, RP.indptr
EI, EJ, S, Vdual = RP.EI, RP.EJ, RP.S, RP.Vdual
cells, areaT = RP.cells, RP.areaT
lsq_gradient, gate3_lap = RP.lsq_gradient, RP.gate3_lap

i0 = 167965

# ---- local wall frame ----
nb1 = adj_n[indptr[i0]:indptr[i0+1]]
ring2 = np.unique(np.concatenate([adj_n[indptr[j]:indptr[j+1]] for j in nb1] + [nb1, [i0]]))
ring3 = np.unique(np.concatenate([adj_n[indptr[j]:indptr[j+1]] for j in ring2] + [ring2]))
# normal from LSQ of wallDistance over ring2
E = X[ring2] - X[i0]; dwd = wd[ring2] - wd[i0]
gn, *_ = np.linalg.lstsq(E, dwd, rcond=None)
nrm = gn/np.linalg.norm(gn)
tan = np.array([nrm[1], -nrm[0]])
if tan @ U2d[i0] < 0:
    tan = -tan
T = lambda P: np.stack([(P - X[i0]) @ tan, (P - X[i0]) @ nrm], axis=-1)

def ut(idx):
    return U2d[idx] @ tan

print(f"node {i0}: x={X[i0,0]:.4f}, d={wd[i0]:.3e}, u_t={ut(i0):.5f}, om={om[i0]:.1f}")
print(f"frame: tangent={tan}, normal={nrm}")

# ---- smooth quadratic reference over ring3, fit in wall frame ----
loc = T(X[ring3])
scale_t, scale_n = np.abs(loc[:, 0]).max(), np.abs(loc[:, 1]).max()
tt, nn = loc[:, 0]/scale_t, loc[:, 1]/scale_n
A = np.stack([np.ones_like(tt), tt, nn, tt*tt, tt*nn, nn*nn], axis=1)
fitU = np.zeros((RP.N, 2))
coef = np.zeros((2, 6))
for comp in range(2):
    c, *_ = np.linalg.lstsq(A, U2d[ring3, comp], rcond=None)
    coef[comp] = c
    fitU[ring3, comp] = A @ c
resid = U2d[ring3] - fitU[ring3]
res_t = resid @ tan
print(f"\nquadratic fit over {len(ring3)}-node 3-ring:")
print(f"  rms tangential residual (the wiggle) = {np.sqrt((res_t**2).mean()):.3e}"
      f"  ({np.sqrt((res_t**2).mean())/np.abs(ut(i0)):.1%} of local u_t)")
lap_fit_analytic = [2*coef[c, 3]/scale_t**2 + 2*coef[c, 5]/scale_n**2 for c in range(2)]
print(f"  analytic lap of fit: ({lap_fit_analytic[0]:.3e}, {lap_fit_analytic[1]:.3e})"
      f"  |.| = {np.hypot(*lap_fit_analytic):.3e}")

# ---- run the kernel on raw, fit, and wiggle fields ----
wigU = np.zeros((RP.N, 2)); wigU[ring3] = U2d[ring3] - fitU[ring3]
gr_raw = {int(n): g for n, g in zip(ring2, lsq_gradient(ring2, U2d))}
gr_fit = {int(n): g for n, g in zip(ring2, lsq_gradient(ring2, fitU))}
gr_wig = {int(n): g for n, g in zip(ring2, lsq_gradient(ring2, wigU))}
lap_raw, parts_raw = gate3_lap([i0], U2d, gr_raw)
lap_fit, parts_fit = gate3_lap([i0], fitU, gr_fit)
lap_wig, parts_wig = gate3_lap([i0], wigU, gr_wig)
print(f"\nkernel at node (per component, then magnitude):")
for nm, l, p in [('raw   u', lap_raw, parts_raw), ('smooth fit', lap_fit, parts_fit),
                 ('wiggle', lap_wig, parts_wig)]:
    print(f"  {nm:>10}: lap=({l[0,0]:+.3e},{l[0,1]:+.3e}) |lap|={np.hypot(*l[0]):.3e}"
          f"   [gAvg part |.|={np.hypot(*p[0,:,0]):.3e}, corr part |.|={np.hypot(*p[0,:,1]):.3e}]")

# ---- the stencil: geometry + values ----
print(f"\none-ring ({len(nb1)} nbrs), wall-frame offsets (x1e6), u_t, wiggle_t (x1e4):")
lt = T(X[nb1])
for k in np.argsort(np.arctan2(lt[:, 1], lt[:, 0])):
    j = nb1[k]
    w_t = (U2d[j] - fitU[j]) @ tan
    print(f"  nbr {j}: (t,n)=({lt[k,0]*1e6:+8.1f},{lt[k,1]*1e6:+8.1f})e-6  L={np.hypot(*lt[k])*1e6:7.1f}e-6"
          f"  u_t={ut(j):.5f}  wig_t={w_t*1e4:+.2f}")

# nodal LSQ gradients across the ring: d(u_t)/dn ~ omega; show oscillation
print(f"\nnodal LSQ gradient (du_t/dn) at center + one-ring: raw | fit | wiggle")
for j in [i0] + list(nb1):
    graw = gr_raw[int(j)]; gfit = gr_fit[int(j)]; gwig = gr_wig[int(j)]
    dn_raw = (graw @ nrm) @ tan if False else (tan @ graw) @ nrm
    dn_fit = (tan @ gfit) @ nrm
    dn_wig = (tan @ gwig) @ nrm
    print(f"  node {j}: {dn_raw:+9.1f} | {dn_fit:+9.1f} | {dn_wig:+9.1f}")

# ---- per-edge accounting for the wiggle field ----
print(f"\nper-edge contributions to lap(wiggle) (t-component of gFace.S/V):")
tot = np.zeros(2)
for s in range(indptr[i0], indptr[i0+1]):
    j, e = adj_n[s], adj_e[s]
    sgn = 1.0 if EI[e] == i0 else -1.0
    areaOut = sgn*S[e]
    eV = X[j] - X[i0]; L = np.hypot(*eV); eH = eV/L
    contrib = np.zeros(2)
    for comp in range(2):
        gA = 0.5*(gr_wig[i0][comp] + gr_wig[int(j)][comp])
        du = wigU[j, comp] - wigU[i0, comp]
        corr = du/L - gA @ eH
        contrib[comp] = ((gA + corr*eH) @ areaOut)/Vdual[i0]
    tot += contrib
    ltj = T(X[j])
    print(f"  -> nbr {j} (t,n)=({ltj[0]*1e6:+7.1f},{ltj[1]*1e6:+7.1f})e-6"
          f"  |S|={np.hypot(*areaOut):.2e}  du_t={(wigU[j]-wigU[i0])@tan*1e4:+6.2f}e-4"
          f"  contrib=({contrib[0]:+.2e},{contrib[1]:+.2e})")
print(f"  SUM = ({tot[0]:+.3e},{tot[1]:+.3e})  vs lap_wig = ({lap_wig[0,0]:+.3e},{lap_wig[0,1]:+.3e})")

# ---- triangle quality around the node ----
tris = np.where((cells == i0).any(axis=1))[0]
print(f"\nincident triangles (aspect = longest edge / height):")
for t in tris:
    Pt = X[cells[t]]
    Ls = [np.hypot(*(Pt[(a+1) % 3] - Pt[a])) for a in range(3)]
    Lmax = max(Ls)
    aspect = Lmax/(2*areaT[t]/Lmax)
    print(f"  tri {t}: edges (x1e6) {sorted([L*1e6 for L in Ls])}, aspect {aspect:8.1f}")
