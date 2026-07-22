"""Bit-faithful 2D replay of Flow360's AI_VG_GATE=3 compact Laplacian on the
cavL2 slice (which IS the quasi-2D triangulation: 193,476 nodes = mesh nodes).

Scheme, from source:
  nodal gradient  g_i = per-direction SVD-weighted LSQ over the one-ring with a
                  constant column (MeshProcessor/LeastSquareCoefficients.cpp,
                  exponent=0.5, ctrWeight=0.1)  [Gradient op, PDESolver.h]
  gate-3 lap      lap_i = (1/V_i) sum_edges [ gAvg + ((u_j-u_i)/L - gAvg.e) e ] . S_f
                  (SpalartAllmaras.h, AI_VG_GATE=3 branch)
  dual geometry   median dual: S_f = sum_{T adj edge} rot90(centroid_T - midpoint),
                  V_i = sum_{T adj i} area_T / 3   (MeshProcessor/EdgeProcessing.h)

Outputs: Gamma_g over an ROI, the max-noise node, and a per-edge trace with the
lap decomposed into the gAvg (smooth) part and the (du/L) correction part, for
the raw field and for a smooth quadratic fit (u = fit + wiggle).
"""
import sys, json
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

CASE = "/home/qiqi/flexcompute/sa-ai/flow360_a3/cavL2prop_nlf0416_Re4M_a4"
ROI_X = (0.05, 0.20)
ROI_D = (5e-5, 4e-4)

# ---------------- load slice = 2D mesh with nodal values ----------------
r = vtk.vtkXMLPUnstructuredGridReader()
r.SetFileName(f"{CASE}/slice_centerSpan.pvtu"); r.Update()
g = r.GetOutput()
P3 = vtk_to_numpy(g.GetPoints().GetData()).astype(np.float64)
X = P3[:, [0, 2]]                     # (x, z) plane
pd = g.GetPointData()
vel = vtk_to_numpy(pd.GetArray('velocity')).astype(np.float64)
U2d = vel[:, [0, 2]]                  # in-plane velocity (u, w)
wd = vtk_to_numpy(pd.GetArray('wallDistance')).astype(np.float64)
om = vtk_to_numpy(pd.GetArray('vorticityMagnitude')).astype(np.float64)
cells = vtk_to_numpy(g.GetCells().GetConnectivityArray()).reshape(-1, 3)
N = len(X); NT = len(cells)
print(f"mesh: {N} nodes, {NT} triangles")

# ---------------- edges + median-dual geometry ----------------
e0 = np.concatenate([cells[:, 0], cells[:, 1], cells[:, 2]])
e1 = np.concatenate([cells[:, 1], cells[:, 2], cells[:, 0]])
tri_of = np.tile(np.arange(NT), 3)
lo, hi = np.minimum(e0, e1), np.maximum(e0, e1)
key = lo.astype(np.int64)*N + hi
order = np.argsort(key)
key_s, lo_s, hi_s, tri_s = key[order], lo[order], hi[order], tri_of[order]
uniq, first = np.unique(key_s, return_index=True)
NE = len(uniq)
EI, EJ = lo_s[first], hi_s[first]
print(f"{NE} unique edges")

cent = X[cells].mean(axis=1)                     # triangle centroids
v1 = X[cells[:, 1]] - X[cells[:, 0]]
v2 = X[cells[:, 2]] - X[cells[:, 0]]
areaT = 0.5*np.abs(v1[:, 0]*v2[:, 1] - v1[:, 1]*v2[:, 0])

# dual area vector per unique edge: sum over adjacent triangles of
# rot90(centroid - midpoint), oriented so S . (xj - xi) > 0
S = np.zeros((NE, 2))
edge_slot = np.searchsorted(uniq, key_s)          # which unique edge each (halfedge, tri) belongs to
mid = 0.5*(X[lo_s] + X[hi_s])
seg = cent[tri_s] - mid
rot = np.stack([seg[:, 1], -seg[:, 0]], axis=1)   # rot90
evec_pair = X[hi_s] - X[lo_s]
flip = (rot*evec_pair).sum(1) < 0
rot[flip] *= -1.0
np.add.at(S, edge_slot, rot)

Vdual = np.zeros(N)
np.add.at(Vdual, cells[:, 0], areaT/3.0)
np.add.at(Vdual, cells[:, 1], areaT/3.0)
np.add.at(Vdual, cells[:, 2], areaT/3.0)

# closure check on interior nodes (sum of outward dual normals = 0)
acc = np.zeros((N, 2))
np.add.at(acc, EI, S)
np.add.at(acc, EJ, -S)
bnd_edges = np.bincount(edge_slot, minlength=NE) == 1   # edges with 1 triangle = boundary
is_bnd = np.zeros(N, bool)
is_bnd[EI[bnd_edges]] = True; is_bnd[EJ[bnd_edges]] = True
closure = np.linalg.norm(acc, axis=1)/np.maximum(np.sqrt(Vdual), 1e-30)
print(f"dual closure (interior): max |sum S|/sqrt(V) = {closure[~is_bnd].max():.2e}")

# ---------------- adjacency (CSR) ----------------
deg = np.zeros(N, np.int64)
np.add.at(deg, EI, 1); np.add.at(deg, EJ, 1)
indptr = np.zeros(N+1, np.int64); indptr[1:] = np.cumsum(deg)
adj_n = np.empty(indptr[-1], np.int64)    # neighbor node
adj_e = np.empty(indptr[-1], np.int64)    # unique-edge id
cur = indptr[:-1].copy()
for arrI, arrJ in ((EI, EJ), (EJ, EI)):
    for k in range(NE):
        i = arrI[k]
        adj_n[cur[i]] = arrJ[k]; adj_e[cur[i]] = k; cur[i] += 1
print("adjacency built")

# ---------------- solver-style nodal gradient (per-direction weighted LSQ) ----
CTR_W, EXPO = 0.1, 0.5
def lsq_gradient(nodes, field2):
    """field2: (N,2) in-plane vector field -> gradients (len(nodes),2,2):
    grad[n][comp][dim]. Implements LeastSquareCoefficients.cpp in 2D."""
    out = np.zeros((len(nodes), 2, 2))
    for n_out, i in enumerate(nodes):
        nb = adj_n[indptr[i]:indptr[i+1]]
        E = X[nb] - X[i]                       # (m,2)
        m = len(nb)
        # SVD of edge matrix -> rotation V, singular values sigma
        _, sig, Vt = np.linalg.svd(E, full_matrices=False)
        Erot = E @ Vt.T
        emag = np.sqrt((E**2).sum(1))
        du = field2[nb] - field2[i]            # (m,2)
        grot = np.zeros((2, 2))                # [comp][rot-dim]
        for d in range(2):
            w = np.maximum(sig[d]/np.sqrt(m), np.r_[0.0, emag])**EXPO
            w[0] = max(sig[d]/np.sqrt(m), 0.0)**EXPO   # center row weight base
            # design: row0 = [1/ctrW, 0, 0]; row_j = [1, Erot_j]
            D = np.zeros((m+1, 3))
            D[0, 0] = 1.0/CTR_W
            D[1:, 0] = 1.0
            D[1:, 1:] = Erot
            Dw = D/w[:, None]
            # coefficient extraction: beta = pinv(Dw) rhs_w, rhs = [0, du]/w
            Pinv = np.linalg.pinv(Dw)
            for comp in range(2):
                rhs = np.r_[0.0, du[:, comp]]/w
                grot[comp, d] = (Pinv @ rhs)[1 + d]
        out[n_out] = grot @ Vt                 # rotate back
    return out

# ---------------- gate-3 Laplacian ----------------
def gate3_lap(nodes, field2, grad_of):
    """lap vector (len(nodes),2) via the SpalartAllmaras.h AI_VG_GATE=3 loop.
    grad_of: dict node -> (2,2) gradient."""
    out = np.zeros((len(nodes), 2))
    parts = np.zeros((len(nodes), 2, 2))       # [:, 0]=gAvg part, [:,1]=corr part
    for n_out, i in enumerate(nodes):
        lap = np.zeros(2); lap_g = np.zeros(2); lap_c = np.zeros(2)
        for s in range(indptr[i], indptr[i+1]):
            j, e = adj_n[s], adj_e[s]
            sgn = 1.0 if EI[e] == i else -1.0
            areaOut = sgn*S[e]
            eV = X[j] - X[i]
            L = np.sqrt((eV**2).sum()); eH = eV/L
            fA = np.sqrt((areaOut**2).sum())
            if fA <= 0 or L <= 0:
                continue
            for comp in range(2):
                gAvg = 0.5*(grad_of[i][comp] + grad_of[j][comp])
                du = field2[j, comp] - field2[i, comp]
                corr = du/L - gAvg @ eH
                gFace = gAvg + corr*eH
                lap[comp] += gFace @ areaOut
                lap_g[comp] += gAvg @ areaOut
                lap_c[comp] += (corr*eH) @ areaOut
        out[n_out] = lap/Vdual[i]
        parts[n_out, :, 0] = lap_g/Vdual[i]
        parts[n_out, :, 1] = lap_c/Vdual[i]
    return out, parts

# ---------------- ROI: compute Gamma_g, find the spike ----------------
roi = np.where((X[:, 0] > ROI_X[0]) & (X[:, 0] < ROI_X[1]) &
               (wd > ROI_D[0]) & (wd < ROI_D[1]) & (X[:, 1] > 0) & ~is_bnd)[0]
ring = np.unique(np.concatenate([adj_n[indptr[i]:indptr[i+1]] for i in roi] + [roi]))
print(f"ROI: {len(roi)} nodes (+ring {len(ring)})")

grads = lsq_gradient(ring, U2d)
gmap = {int(n): grads[k] for k, n in enumerate(ring)}
lap, parts = gate3_lap(roi, U2d, gmap)
lapMag = np.sqrt((lap**2).sum(1))
u2 = (U2d[roi]**2).sum(1)
Gg = (wd[roi]**2*lapMag)**2/np.maximum(u2 + (om[roi]*wd[roi])**2, 1e-30)
k = int(np.argmax(Gg))
i0 = int(roi[k])
print(f"\nSPIKE node {i0}: x={X[i0,0]:.4f} z={X[i0,1]:.5f} d={wd[i0]:.3e}")
print(f"  Gamma_g={Gg[k]:.3f}  lapMag={lapMag[k]:.3e}  |u|={np.sqrt(u2[k]):.4f} om={om[i0]:.1f}")
print(f"  lap = {lap[k]}  (gAvg part {parts[k,:,0]}, corr part {parts[k,:,1]})")
print(f"  Gamma_g stats over ROI: median {np.median(Gg):.4f}  P90 {np.percentile(Gg,90):.3f}  max {Gg.max():.3f}")

np.savez('/tmp/claude-1006/-home-qiqi-flexcompute/15845519-8cb3-4677-8f3c-47bcc8951d95/scratchpad/replay_state.npz',
         i0=i0, roi=roi, Gg=Gg, lap=lap, parts=parts, lapMag=lapMag)
print("state saved")
