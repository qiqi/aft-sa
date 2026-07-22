"""Regenerate NLF figures for one alpha-pair, showing L0/L1/L2 for both
cav (unstructured) and str (O-grid) meshes.

Usage:
    python regen_nlf_v2.py low     # alpha = 0, 4
    python regen_nlf_v2.py high    # alpha = 9, 15
"""
import csv, os, sys, pickle, numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
matplotlib.rcParams.update({'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10,
                            'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8})

B  = os.environ.get("SAAI_CFD_ROOT", "/home/qiqi/flexcompute/sa-ai/flow360_fr")
PD = "/home/qiqi/flexcompute/sa-ai/paper"
NU = 2.5e-8

# Kernel constants: imported from the repro-local single source of truth
# (identical to Flow360 ModelConstants.h; enforced by
# sa-ai/tests/test_constants_consistency.py). A stale local copy sat at the
# RETIRED kernel until 2026-07-13 — never copy digits here.
import os as _os, sys as _sys
_REPRO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPRO not in _sys.path:
    _sys.path.insert(0, _REPRO)
from lib.calibrate_kernel import (A_MAX, RE_OMEGA_FLOOR, BARRIER_POWER,
                                  SIGMOID_SLOPE as SLOPE, SIGMOID_CENTER as G_C)
TILT_SLOPE = 1.0e6  # effectively disabled (cliff is pure floor)
C_V1 = 7.1
CHI_INF = C_V1 * np.exp(-9.0)
CHI_BEGIN = 2.0 * CHI_INF; CHI_MID = np.sqrt(CHI_INF); CHI_END = 1.0; CHI_TAIL = C_V1
N_LO, N_HI = -2.0, 12.0
CHI_LO, CHI_HI = CHI_INF*np.exp(N_LO), CHI_INF*np.exp(N_HI)

UP_COLOR = 'C0'
LO_COLOR = 'C3'
# Per-level line THICKNESS (L0 thin, L1 medium, L2 thick).
LEVEL_LW = {'L0': 0.8, 'L1': 1.6, 'L2': 2.4}
# Per-mesh line STYLE (str = solid, cav = dashed).
MESH_LS = {'str': '-', 'cav': '--'}

def rate(Re_O, Gamma):
    """Kernel over the (Re_Ω, Γ) plane at λ_p = 0 (cliff at the floor, f_λ = 1);
    landscape background only. Matches SAAiTransition.h::__aiRate at λ_p = 0."""
    Re_O = np.asarray(Re_O); Gamma = np.asarray(Gamma)
    log_floor = np.log10(RE_OMEGA_FLOOR)
    log_extra = np.where(Gamma < 1.0, (1.0 - Gamma) / TILT_SLOPE, 0.0)
    Re_cliff = 10.0 ** (log_floor + log_extra)
    safe = np.maximum(Re_O, Re_cliff + 1e-12)
    ratio = Re_cliff / safe
    bar_inside = np.maximum(1.0 - np.power(ratio, BARRIER_POWER), 1e-20)
    barr = np.where(Re_O > Re_cliff, np.log(bar_inside), -np.inf)
    z = SLOPE*(Gamma - G_C) + barr
    a = A_MAX/(1.0+np.exp(-z))
    return np.where(Re_O > Re_cliff, a, 0.0)

def case_dir(mesh, level, alpha):
    """mesh in {'cav','str'}, level in {'L0','L1','L2'}, alpha int."""
    a_int = int(alpha)
    return f"{B}/{mesh}{level}prop_nlf0416_Re4M_a{a_int}"

def load_slice(d):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{d}/slice_centerSpan.pvtu"); r.Update()
    g = r.GetOutput(); pd = g.GetPointData()
    arrs = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    return g, pd, arrs

def airfoil_walk_contour(case_d, af='nlf0416', z_te_half=0.00125):
    """Return (x_upper, cf_upper, cp_upper), (x_lower, cf_lower, cp_lower) for
    the AIRFOIL SURFACE only, using the mesh connectivity.

    Algorithm:
      1. Read all surface QUAD cells; each quad spans the y direction with
         two pairs of nodes sharing (x,z). Dedup by (x,z) → 2D-merged graph.
         For each quad, its airfoil-direction edge connects two distinct
         (x,z) nodes; its span edges become self-loops and are dropped.
      2. Walk the edges to form the closed contour loop (every node has
         exactly two neighbors so this is unambiguous).
      3. Compute signed area; if CCW (positive), reverse so the loop is CW.
      4. Find LE (min x) and TE (max x) in the loop. Split at those indices.
         Upper = the arc with higher mean z. Lower = the other.
      5. Filter blunt-face interior nodes (x≈1, |z|<z_TE_half) from each arc.
    """
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(f"{case_d}/surface_fluid_{af}.pvtu"); r.Update()
    g = r.GetOutput(); pd = g.GetPointData()
    P = vtk_to_numpy(g.GetPoints().GetData())
    nm = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    cf_arr = vtk_to_numpy(pd.GetArray(next(n for n in nm if n.lower().startswith('cf'))))
    cf3d = np.linalg.norm(cf_arr, axis=1) if cf_arr.ndim > 1 else cf_arr
    cp3d = vtk_to_numpy(pd.GetArray(next(n for n in nm if n.lower().startswith('cp'))))

    # (1) dedup by (x,z); build mapping orig_id → merged_id
    key = np.stack([np.round(P[:, 0], 9), np.round(P[:, 2], 9)], axis=1)
    _, first_idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    M = len(first_idx)
    Xm = P[first_idx, 0]; Zm = P[first_idx, 2]
    cf_m = cf3d[first_idx]; cp_m = cp3d[first_idx]

    # Read cells from VTK and collect airfoil-direction edges
    cells_data = vtk_to_numpy(g.GetCells().GetData())
    cell_types = vtk_to_numpy(g.GetCellTypesArray())
    edges = set()
    pos = 0
    for ct in cell_types:
        n_pts = cells_data[pos]; pos += 1
        pids = cells_data[pos:pos + n_pts]; pos += n_pts
        mids = inv[pids]
        for k in range(n_pts):
            a, b = mids[k], mids[(k+1) % n_pts]
            if a == b:
                continue  # span edge
            edges.add((min(int(a), int(b)), max(int(a), int(b))))

    # (2) Walk the closed loop. Each merged node has degree 2.
    adj = {i: [] for i in range(M)}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    bad = [i for i in range(M) if len(adj[i]) != 2]
    if bad:
        raise RuntimeError(f"non-loop topology: {len(bad)} nodes with degree != 2 "
                           f"(first few: {bad[:5]} adj counts {[len(adj[i]) for i in bad[:5]]})")
    walk = [0]
    visited = {0}
    while True:
        last = walk[-1]
        nxt = None
        for n in adj[last]:
            if n not in visited:
                nxt = n; break
        if nxt is None:
            break
        walk.append(nxt); visited.add(nxt)
    if len(walk) != M:
        raise RuntimeError(f"walked {len(walk)} of {M} nodes")
    walk = np.array(walk)

    # (3) Signed area to decide orientation. CW (negative) is canonical
    # because we want LE-on-the-left and z-up = "upper" on TOP.
    Xw = Xm[walk]; Zw = Zm[walk]
    area = 0.5 * np.sum(Xw * np.roll(Zw, -1) - np.roll(Xw, -1) * Zw)
    if area > 0:                     # CCW → reverse to CW
        walk = walk[::-1]
        Xw = Xm[walk]; Zw = Zm[walk]

    # (4) LE = min x, TE = max x; split the loop
    le = int(np.argmin(Xw))
    te = int(np.argmax(Xw))
    n = len(walk)
    if le < te:
        seg_a = walk[le:te+1]
        seg_b = np.concatenate([walk[te:], walk[:le+1]])
    else:
        seg_a = walk[te:le+1]
        seg_b = np.concatenate([walk[le:], walk[:te+1]])
    if Zm[seg_a].mean() > Zm[seg_b].mean():
        up_idx, lo_idx = seg_a, seg_b
    else:
        up_idx, lo_idx = seg_b, seg_a

    # (5) drop blunt-face interior nodes from each surface
    def is_blunt(i):
        return (abs(Xm[i] - 1.0) < 1e-4) and (abs(Zm[i]) < z_te_half - 1e-6)
    up_idx = np.array([i for i in up_idx if not is_blunt(i)])
    lo_idx = np.array([i for i in lo_idx if not is_blunt(i)])

    def srt(idx):
        xs = Xm[idx]; cs = cf_m[idx]; ps = cp_m[idx]
        order = np.argsort(xs)
        return xs[order], cs[order], ps[order]
    return srt(up_idx), srt(lo_idx)

def walk_contour_xz(case_d, af='nlf0416', z_te_half=0.00125):
    """Same connectivity walk as airfoil_walk_contour, but returns the merged
    (x,z) arrays and ordered index lists for upper/lower in WALK ORDER (not
    sorted by x) so consecutive points are neighbors along the contour. This
    is the ordering needed to compute tangents and outward normals.
    Returns: Xm, Zm, up_idx_walk, lo_idx_walk.
    """
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(f"{case_d}/surface_fluid_{af}.pvtu"); r.Update()
    g = r.GetOutput()
    P = vtk_to_numpy(g.GetPoints().GetData())
    key = np.stack([np.round(P[:, 0], 9), np.round(P[:, 2], 9)], axis=1)
    _, first_idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    M = len(first_idx)
    Xm = P[first_idx, 0]; Zm = P[first_idx, 2]
    cells_data = vtk_to_numpy(g.GetCells().GetData())
    cell_types = vtk_to_numpy(g.GetCellTypesArray())
    edges = set(); pos = 0
    for ct in cell_types:
        n_pts = cells_data[pos]; pos += 1
        pids = cells_data[pos:pos + n_pts]; pos += n_pts
        mids = inv[pids]
        for k in range(n_pts):
            a, b = int(mids[k]), int(mids[(k+1) % n_pts])
            if a == b: continue
            edges.add((min(a,b), max(a,b)))
    adj = {i: [] for i in range(M)}
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
    walk = [0]; visited = {0}
    while True:
        last = walk[-1]; nxt = None
        for n in adj[last]:
            if n not in visited: nxt = n; break
        if nxt is None: break
        walk.append(nxt); visited.add(nxt)
    walk = np.array(walk)
    Xw = Xm[walk]; Zw = Zm[walk]
    area = 0.5 * np.sum(Xw * np.roll(Zw, -1) - np.roll(Xw, -1) * Zw)
    if area > 0:
        walk = walk[::-1]; Xw = Xm[walk]; Zw = Zm[walk]
    le = int(np.argmin(Xw)); te = int(np.argmax(Xw))
    n = len(walk)
    if le < te:
        seg_a = walk[le:te+1]
        seg_b = np.concatenate([walk[te:], walk[:le+1]])
    else:
        seg_a = walk[te:le+1]
        seg_b = np.concatenate([walk[le:], walk[:te+1]])
    if Zm[seg_a].mean() > Zm[seg_b].mean():
        up_idx, lo_idx = seg_a, seg_b
    else:
        up_idx, lo_idx = seg_b, seg_a
    def filt(idx):
        return np.array([i for i in idx
                         if not ((abs(Xm[i]-1.0) < 1e-4) and (abs(Zm[i]) < z_te_half - 1e-6))])
    return Xm, Zm, filt(up_idx), filt(lo_idx)

def slice_y_plane(case_d):
    """Median y-coordinate of slice_centerSpan.pvtu (the cut plane)."""
    g, _, _ = load_slice(case_d)
    p = vtk_to_numpy(g.GetPoints().GetData())
    return float(np.median(p[:, 1]))

def load_slice_derived(case_d):
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(f"{case_d}/slice_with_derived.pvtu"); r.Update()
    return r.GetOutput()

def wallnormal_max_metrics(case_d, side='upper', L_probe=0.01, n_probe=80,
                            return_sigma_fpg=False, return_lambda_p=False,
                            return_P=False, return_components=False):
    """For each surface point on `side`, shoot a probe line of length L_probe
    in the outward wall-normal direction, interpolate Re_Omega and Gamma on
    n_probe equally spaced points along the line (from slice_with_derived.pvtu),
    and take the max of each along the line. Returns (x_surf, Re_Omega_max,
    Gamma_max). All metrics are PURELY KINEMATIC — they do not depend on
    nuHat or the amplification rate, so they are valid inputs for calibrating
    the rate kernel.

    If return_lambda_p=True, also returns the representative streamwise
    pressure-gradient parameter lambda_p = -d^2 (u.grad p)/(rho nu |u|^2),
    taken as the mean over the kernel-active band (Re_Omega > 0.5*max_y Re_Omega,
    the theta-scale layer around the Re_Omega peak) and lightly smoothed.
    lambda_p is plotted directly for the NLF (favorable-PG calibration); it is
    smoother than sigma_FPG(lambda_p) because the sigmoid steepens noise near
    lambda_*. lambda_p ~ d^2 blows up in the free stream, so the band-mean (not a
    probe extremum) is used.  If return_sigma_fpg=True, returns
    sigma_FPG = 1/(1+exp(s_lambda(lambda_p-lambda_*))) instead (s_lambda=4.56,
    lambda_*=0.64, matching ModelConstants.h).
    """
    Xm, Zm, up_idx, lo_idx = walk_contour_xz(case_d)
    idx = up_idx if side == 'upper' else lo_idx
    xs = Xm[idx]; zs = Zm[idx]
    # Tangent via central differences in walk order
    tx_raw = np.gradient(xs); tz_raw = np.gradient(zs)
    s = np.sqrt(tx_raw**2 + tz_raw**2) + 1e-30
    tx = tx_raw / s; tz = tz_raw / s
    # Outward normal: CW walk → outward is right perpendicular = (tz, -tx)
    nx = tz; nz = -tx
    # Sanity-flip per side (z-up for upper, z-down for lower)
    if side == 'upper' and np.mean(nz) < 0:
        nx, nz = -nx, -nz
    elif side == 'lower' and np.mean(nz) > 0:
        nx, nz = -nx, -nz
    # Build probe points: M_surf × n_probe in (x, y_slice, z)
    M = len(xs)
    dists = np.linspace(1e-6, L_probe, n_probe)  # start just off the wall
    y0 = slice_y_plane(case_d)
    pts_arr = np.empty((M * n_probe, 3))
    for j in range(n_probe):
        d = dists[j]
        pts_arr[j*M:(j+1)*M, 0] = xs + d*nx
        pts_arr[j*M:(j+1)*M, 1] = y0
        pts_arr[j*M:(j+1)*M, 2] = zs + d*nz
    vpts = vtk.vtkPoints()
    vpts.SetData(numpy_to_vtk(pts_arr, deep=True))
    poly = vtk.vtkPolyData(); poly.SetPoints(vpts)
    slice_g = load_slice_derived(case_d)
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(poly); probe.SetSourceData(slice_g)
    probe.Update()
    out = probe.GetOutput(); pdd = out.GetPointData()
    ReO_raw = vtk_to_numpy(pdd.GetArray('Re_Omega'))
    _gam_arr = pdd.GetArray('Gamma')  # retired; kept robust for legacy return slots
    Gam_raw = vtk_to_numpy(_gam_arr) if _gam_arr is not None else np.full(M*n_probe, np.nan)
    P_arr = pdd.GetArray('P')
    P_raw = vtk_to_numpy(P_arr) if P_arr is not None else np.full(M*n_probe, np.nan)
    valid = vtk_to_numpy(probe.GetValidPoints())
    mask = np.zeros(M * n_probe, bool); mask[valid] = True
    # Reshape (n_probe, M) and mask invalid probes as NaN
    ReO = np.where(mask, ReO_raw, np.nan).reshape(n_probe, M)
    Gam = np.where(mask, Gam_raw, np.nan).reshape(n_probe, M)
    Psl = np.where(mask, P_raw, np.nan).reshape(n_probe, M)
    ReO_max = np.nanmax(ReO, axis=0)
    Gam_max = np.nanmax(Gam, axis=0)
    P_max = np.nanmax(Psl, axis=0)
    order = np.argsort(xs)
    if return_components:
        def _rs(nm):
            arr = pdd.GetArray(nm)
            raw = vtk_to_numpy(arr) if arr is not None else np.full(M*n_probe, np.nan)
            return np.where(mask, raw, np.nan).reshape(n_probe, M)
        Xs=_rs('sph_X'); Ys=_rs('sph_Y'); Zs=_rs('sph_Z'); Ss=_rs('Shat'); Gs=_rs('g_coord')
        istar = np.argmax(np.where(np.isfinite(Psl), Psl, -1e30), axis=0); cols=np.arange(M)
        pick = lambda A: A[istar, cols]
        o = order
        return (xs[o], pick(Xs)[o], pick(Ys)[o], pick(Zs)[o],
                pick(Ss)[o], pick(Gs)[o], P_max[o], ReO_max[o])
    if return_P:
        return xs[order], ReO_max[order], Gam_max[order], P_max[order]
    if not (return_sigma_fpg or return_lambda_p):
        return xs[order], ReO_max[order], Gam_max[order]
    # Representative lambda_p = mean over the kernel-active band
    # (Re_Omega > 0.5*max_y Re_Omega, the theta-scale layer around the Re_Omega
    # peak), lightly smoothed. Band-mean (not a probe extremum) because
    # lambda_p = -d^2(u.grad p)/(rho nu |u|^2) ~ d^2 blows up in the free stream.
    lam_arr = pdd.GetArray('lambda_p')
    if lam_arr is None:
        extra = np.zeros_like(ReO_max) if return_lambda_p else np.ones_like(ReO_max)
        return xs[order], ReO_max[order], Gam_max[order], extra[order]
    lam_raw = vtk_to_numpy(lam_arr)
    lam = np.where(mask, lam_raw, np.nan).reshape(n_probe, M)
    band = ReO > 0.5 * ReO_max[None, :]
    lam_rep = np.nanmean(np.where(band, lam, np.nan), axis=0)[order]
    finite = np.isfinite(lam_rep)
    if finite.sum() > 3:
        lam_rep = gaussian_filter1d(
            np.interp(np.arange(len(lam_rep)), np.where(finite)[0], lam_rep[finite]), 1.5)
    if return_lambda_p:
        return xs[order], ReO_max[order], Gam_max[order], lam_rep
    LAMBDA_STAR, LAMBDA_SLOPE = 0.64, 4.56   # matches ModelConstants.h
    sigFPG = 1.0 / (1.0 + np.exp(LAMBDA_SLOPE * (lam_rep - LAMBDA_STAR)))
    return xs[order], ReO_max[order], Gam_max[order], sigFPG

def airfoil_surfaces(case_d, af='nlf0416'):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{case_d}/surface_fluid_{af}.pvtu"); r.Update()
    p = vtk_to_numpy(r.GetOutput().GetPoints().GetData())
    X, Z = p[:,0], p[:,2]
    n = len(X)
    j = int(np.argmin(X)); o = [j]; u = np.zeros(n, bool); u[j] = True
    pts = np.column_stack((X, Z))
    while len(o) < n:
        c = o[-1]; dd = np.sum((pts - pts[c])**2, 1); dd[u] = 1e9
        nx = int(np.argmin(dd)); o.append(nx); u[nx] = True
    xo, zo = X[np.array(o)], Z[np.array(o)]
    mid = n // 2
    window = slice(max(0, mid - n//8), min(n, mid + n//8))
    te = window.start + int(np.argmax(xo[window]))
    b1, b2 = slice(0, te), slice(te+1, n)
    up, lo = (b1, b2) if zo[b1].mean() > zo[b2].mean() else (b2, b1)
    xu_, zu_ = xo[up], zo[up]; oo = np.argsort(xu_); xu_, zu_ = xu_[oo], zu_[oo]
    xl_, zl_ = xo[lo], zo[lo]; oo = np.argsort(xl_); xl_, zl_ = xl_[oo], zl_[oo]
    def z_up(x): return np.interp(x, xu_, zu_)
    def z_lo(x): return np.interp(x, xl_, zl_)
    def z_camber(x): return 0.5*(z_up(x) + z_lo(x))
    return z_up, z_lo, z_camber

def _side_mask(p, z_up_fn, z_lo_fn):
    """For each point, decide if it's on the upper or lower side of the airfoil.

    The previous z_camber classification breaks down for cells near the TE
    blunt face (where upper/lower surface heights collapse to ±0.00125) and
    for off-chord cells (x<0 or x>1). Use a NARROW airfoil-thickness band:
    upper = z_lower(x) < z < z_upper(x) + delta with z > z_camber(x);
    lower = the symmetric. Cells outside x∈[0,1] or outside the airfoil
    thickness band get classified as neither (excluded).
    """
    x, z = p[:,0], p[:,2]
    in_chord = (x >= 0.0) & (x <= 1.0)
    zu = z_up_fn(np.clip(x, 0.0, 1.0))
    zl = z_lo_fn(np.clip(x, 0.0, 1.0))
    zc = 0.5 * (zu + zl)
    # The cell counts as upper-side BL if z > z_camber AND z is reasonably close
    # to the upper surface (within a chord-fraction band so we don't pick up
    # mid-stream points). 0.10 chord above the upper surface is the standard.
    upper = in_chord & (z > zc) & (z < zu + 0.10)
    lower = in_chord & (z < zc) & (z > zl - 0.10)
    return upper, lower

def max_chi_vs_x(case_d, nbins=40, d_max=0.01):
    """Per-x max chi on the upper and lower BL. Cells limited to wallDistance
    within d_max (default 0.01 chord — tighter than before to stay within BL
    and avoid free-stream contamination of the max).
    """
    g, pd, arrs = load_slice(case_d)
    if not all(n in arrs for n in ['nuHat', 'wallDistance']):
        raise RuntimeError('missing nuHat or wallDistance')
    p = vtk_to_numpy(g.GetPoints().GetData())
    nh = vtk_to_numpy(pd.GetArray('nuHat'))
    wd_arr = vtk_to_numpy(pd.GetArray('wallDistance'))
    band = (wd_arr > 0) & (wd_arr <= d_max)
    z_up_fn, z_lo_fn, _ = airfoil_surfaces(case_d)
    upper_mask, lower_mask = _side_mask(p, z_up_fn, z_lo_fn)
    edges = np.linspace(0.0, 1.0, nbins+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    mu = np.full(nbins, np.nan); ml = np.full(nbins, np.nan)
    for i in range(nbins):
        m = (p[:,0] >= edges[i]) & (p[:,0] < edges[i+1]) & band
        mu_ = m & upper_mask
        ml_ = m & lower_mask
        if mu_.any(): mu[i] = nh[mu_].max()
        if ml_.any(): ml[i] = nh[ml_].max()
    return centers, mu, ml

def trajectory(case_d, side='upper', nbins=40, d_max=0.01):
    """Sample point per x-bin: argmax of the amplification kernel
    a(Re_Omega, Gamma) * |omega| * nuHat. The previous argmax(nuHat) locked
    onto freestream cells in the laminar region where nuHat is essentially
    uniform at chi_inf*nu, giving Gamma ≈ 0 (the freestream value, not a BL
    sample). The kernel weights toward cells where amplification is actually
    producing -- always inside the BL.
    """
    g, pd, arrs = load_slice(case_d)
    if not all(n in arrs for n in ['nuHat','vorticityMagnitude','wallDistance']):
        return None
    p = vtk_to_numpy(g.GetPoints().GetData())
    nh = vtk_to_numpy(pd.GetArray('nuHat'))
    vm = vtk_to_numpy(pd.GetArray('vorticityMagnitude'))
    vel_arr = pd.GetArray('velocity')
    if vel_arr is None: return None
    vel = vtk_to_numpy(vel_arr)
    wd = vtk_to_numpy(pd.GetArray('wallDistance'))
    band = (wd > 0) & (wd <= d_max)
    z_up_fn, z_lo_fn, _ = airfoil_surfaces(case_d)
    upper_mask, lower_mask = _side_mask(p, z_up_fn, z_lo_fn)
    side_mask = upper_mask if side == 'upper' else lower_mask
    # Precompute the kernel a*|omega|*nuHat for every cell in the (band & side)
    # so the argmax in each x-bin actually picks the cell that contributes
    # to amplification.
    U = np.linalg.norm(vel, axis=1)
    ome_d = vm * wd
    Re_O_all = wd*wd*vm / NU
    Gam_all = 2.0*ome_d**2 / (U**2 + ome_d**2 + 1e-30)
    a_all = rate(Re_O_all, Gam_all)
    kernel = a_all * vm * nh
    edges = np.linspace(0.0, 1.0, nbins+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    Re_O = np.full(nbins, np.nan); Gam = np.full(nbins, np.nan); chi = np.full(nbins, np.nan)
    for i in range(nbins):
        m = (p[:,0] >= edges[i]) & (p[:,0] < edges[i+1]) & band & side_mask
        if not m.any(): continue
        idx = np.argmax(kernel[m])
        sel = np.where(m)[0][idx]
        chi[i] = nh[sel] / NU
        Re_O[i] = Re_O_all[sel]
        Gam[i] = Gam_all[sel]
    return centers, Re_O, Gam, chi

def find_x_at_chi(xs, chi, chi_target):
    valid = ~np.isnan(chi)
    if not valid.any(): return None
    idx = np.argmax(chi >= chi_target)
    if not (chi[idx] >= chi_target): return None
    if idx == 0: return xs[0]
    f = (chi_target - chi[idx-1]) / (chi[idx] - chi[idx-1] + 1e-30)
    return xs[idx-1] + f * (xs[idx] - xs[idx-1])

# Load mfoil reference
mn = pickle.load(open(f"{B}/mfoil_nlf0416_Re4M.pkl", 'rb'))

_ffn_path = "/home/qiqi/flexcompute/sa-ai/flow360_ai/flexfoil_nlf0416_Re4M.pkl"
ffn = pickle.load(open(_ffn_path, 'rb')) if os.path.exists(_ffn_path) else {}


def _ff_fill(alpha):
    """FlexFoil N envelope, drawn only where mfoil is unusable."""
    return (float(alpha) in ffn) and not (float(alpha) in mn and _mfoil_usable(mn[float(alpha)]))


def _mfoil_usable(md):
    """mfoil is a valid e^N reference only where its coupled solve converged AND
    the upper surface stays attached. It fails to converge at alpha=9 deg and
    returns a stalled solution (upper H>>1) at alpha=15 deg; neither is usable."""
    try:
        return bool(md.get('conv', True)) and float(np.nanmax(md['upper']['H'])) < 4.0
    except Exception:
        return bool(md.get('conv', True))

# xfoil reference: used ONLY to fill Cp/Cf/xtr where mfoil is unusable
# (non-converged / stalled). mfoil's N(x) is kept everywhere it exists.
try:
    xf = pickle.load(open(f"{B}/xfoil_nlf0416_Re4M.pkl", 'rb'))
except Exception:
    xf = {}
def _xfoil_fill(alpha):
    return (float(alpha) in xf) and not (float(alpha) in mn and _mfoil_usable(mn[float(alpha)]))


LEVELS_CF = ['L0', 'L1', 'L2']   # configurable: restrict the refinement levels drawn

# Measured transition onsets (Somers TP-1861 Fig. 9d transition orifices,
# read as xtr(cl) at the computed section cl — the Exp column of the paper's
# NLF transition table). None = front ahead of the instrumented region.
# Entries marked extrapolated are beyond the recorded orifices (the paper's
# double-dagger): upper at alpha=4, lower at alpha=15.
EXP_XTR = {0: {'upper': (0.38, False), 'lower': (0.55, False)},
           4: {'upper': (0.31, True),  'lower': (0.62, False)},
           9: {'upper': None,          'lower': (0.64, False)},
           15: {'upper': None,         'lower': (0.66, True)}}


def make_cf_figure(alphas, out_name, title, meshes=None, L_probe=0.01, n_probe=80):
    """5 rows × len(alphas) cols. Rows top-to-bottom:
       (1) max Re_Omega along 0.01c wall-normal probe (purely kinematic);
       (2) min sigma_FPG along 0.01c wall-normal probe — the strongest
           favorable-PG suppression encountered in the BL (the parameter
           directly tuned in this section);
       (3) chi(x) (log) + mfoil N(x) (linear) on twin axes;
       (4) -Cp;
       (5) signed Cf.
    `meshes` filters which mesh families to draw: None = both ['cav','str'],
    or pass e.g. ['str'] / ['cav'] for a single-family figure.
    """
    if meshes is None: meshes = ['cav','str']
    # Is a usable mfoil reference available for ANY alpha in this figure? If not
    # (e.g. the high-alpha figure, where mfoil doesn't converge / stalls), drop
    # the mfoil legend entry and the mfoil N right-axis label.
    any_mfoil = any(float(a) in mn and _mfoil_usable(mn[float(a)]) for a in alphas)
    any_xfoil = any(_xfoil_fill(a) for a in alphas)
    any_ff = any(_ff_fill(a) for a in alphas)
    fig, axs = plt.subplots(5, len(alphas), figsize=(5*len(alphas), 13), sharex=True)
    if len(alphas) == 1: axs = axs[:, None]
    for col, alpha in enumerate(alphas):
        ax_reo = axs[0, col]; ax_P = axs[1, col]
        ax_n   = axs[2, col]; ax_cp  = axs[3, col]; ax_cf = axs[4, col]
        ax_nN = ax_n.twinx()
        # ROW 1+2: wall-normal-probe max Re_Omega (onset gate) and max P = Shat*g
        # (rate coordinate) --- the two quantities the sphere kernel actually uses.
        for mesh in meshes:
            for level in LEVELS_CF:  # configurable level set
                d = case_dir(mesh, level, alpha)
                if not os.path.exists(f"{d}/slice_with_derived.pvtu"): continue
                lw = LEVEL_LW[level]; ls = MESH_LS[mesh]
                try:
                    xs_u, ReO_u, Gam_u, P_u = wallnormal_max_metrics(
                        d, side='upper', L_probe=L_probe, n_probe=n_probe,
                        return_P=True)
                    xs_l, ReO_l, Gam_l, P_l = wallnormal_max_metrics(
                        d, side='lower', L_probe=L_probe, n_probe=n_probe,
                        return_P=True)
                except Exception as e:
                    print(f"  skip probe ({mesh}/{level}, α={alpha}): {e}"); continue
                ax_reo.semilogy(xs_u, ReO_u, ls=ls, lw=lw, color=UP_COLOR)
                ax_reo.semilogy(xs_l, ReO_l, ls=ls, lw=lw, color=LO_COLOR)
                ax_P.plot(xs_u, P_u, ls=ls, lw=lw, color=UP_COLOR)
                ax_P.plot(xs_l, P_l, ls=ls, lw=lw, color=LO_COLOR)
        ax_reo.axhline(RE_OMEGA_FLOOR, color='gray', ls='--', lw=0.6, alpha=0.5)
        ax_reo.set_ylim(1e2, 1e4); ax_reo.grid(alpha=0.3, which='both')
        ax_reo.set_title(rf'$\alpha={alpha}^\circ$', fontsize=10)
        if col == 0: ax_reo.set_ylabel(r'$\max Re_\Omega$ (log)')
        # max P = Shat*g, the sphere rate coordinate: P>0 (above the dotted line)
        # is the amplifying, inflectional side; the rate is a_max clip<P>.
        ax_P.axhline(0.0, color='gray', ls=':', lw=0.6, alpha=0.5)
        ax_P.set_ylim(-0.3, 1.0); ax_P.grid(alpha=0.3)
        if col == 0: ax_P.set_ylabel(r'$\max \hat S g$')
        # ROW 3: chi + N
        for mesh in meshes:
            for level in LEVELS_CF:  # configurable level set
                d = case_dir(mesh, level, alpha)
                if not os.path.exists(d) or not os.path.exists(f"{d}/slice_centerSpan.pvtu"): continue
                lw = LEVEL_LW[level]; ls = MESH_LS[mesh]
                try:
                    xc, mu, ml = max_chi_vs_x(d)
                except Exception as e:
                    print(f"  skip chi ({mesh}/{level}, α={alpha}): {e}"); continue
                ax_n.semilogy(xc, mu/NU, ls=ls, lw=lw, color=UP_COLOR)
                ax_n.semilogy(xc, ml/NU, ls=ls, lw=lw, color=LO_COLOR)
        if float(alpha) in mn and _mfoil_usable(mn[float(alpha)]):
            md = mn[float(alpha)]
            ax_nN.plot(md['upper']['x'], md['upper']['n'], ':', color=UP_COLOR, lw=1.2, alpha=0.5)
            ax_nN.plot(md['lower']['x'], md['lower']['n'], ':', color=LO_COLOR, lw=1.2, alpha=0.5)
            ax_nN.axhline(9.0, color='gray', ls=':', lw=0.6, alpha=0.6)
            if md.get('xtr_upper') is not None: ax_n.axvline(md['xtr_upper'], color=UP_COLOR, ls=':', lw=0.6, alpha=0.5)
            if md.get('xtr_lower') is not None: ax_n.axvline(md['xtr_lower'], color=LO_COLOR, ls=':', lw=0.6, alpha=0.5)
        elif _ff_fill(alpha):
            fd = ffn[float(alpha)]
            ax_nN.plot(fd['upper']['x'], fd['upper']['n'], ':', color=UP_COLOR, lw=1.2, alpha=0.5)
            ax_nN.plot(fd['lower']['x'], fd['lower']['n'], ':', color=LO_COLOR, lw=1.2, alpha=0.5)
            ax_nN.axhline(9.0, color='gray', ls=':', lw=0.6, alpha=0.6)
            if _xfoil_fill(alpha):
                xd = xf[float(alpha)]
                if xd.get('xtr_upper') is not None: ax_n.axvline(xd['xtr_upper'], color=UP_COLOR, ls='-.', lw=0.7, alpha=0.6)
                if xd.get('xtr_lower') is not None: ax_n.axvline(xd['xtr_lower'], color=LO_COLOR, ls='-.', lw=0.7, alpha=0.6)
        elif _xfoil_fill(alpha):
            xd = xf[float(alpha)]
            if xd.get('xtr_upper') is not None: ax_n.axvline(xd['xtr_upper'], color=UP_COLOR, ls='-.', lw=0.7, alpha=0.6)
            if xd.get('xtr_lower') is not None: ax_n.axvline(xd['xtr_lower'], color=LO_COLOR, ls='-.', lw=0.7, alpha=0.6)
        ax_n.axhline(C_V1, color='gray', ls=':', lw=0.6, alpha=0.6)
        ax_n.set_ylim(CHI_LO, CHI_HI); ax_nN.set_ylim(N_LO, N_HI)
        ax_n.grid(alpha=0.3)
        if col == 0: ax_n.set_ylabel('$\\chi$ (log)')
        if col == len(alphas)-1 and (any_mfoil or any_ff): ax_nN.set_ylabel('$e^9$ $N$ (linear)')
        # ROWS 4+5: Cp/Cf
        for mesh in meshes:
            for level in LEVELS_CF:  # configurable level set
                d = case_dir(mesh, level, alpha)
                if not os.path.exists(f"{d}/surface_fluid_nlf0416.pvtu"): continue
                lw = LEVEL_LW[level]; ls = MESH_LS[mesh]
                try:
                    (xu, cfu, cpu), (xl, cfl, cpl) = airfoil_walk_contour(d)
                    ax_cp.plot(xu, -cpu, ls=ls, lw=lw, color=UP_COLOR)
                    ax_cp.plot(xl, -cpl, ls=ls, lw=lw, color=LO_COLOR)
                    ax_cf.plot(xu, cfu, ls=ls, lw=lw, color=UP_COLOR)
                    ax_cf.plot(xl, cfl, ls=ls, lw=lw, color=LO_COLOR)
                except Exception as e:
                    print(f"  skip surf ({mesh}/{level}, α={alpha}): {e}")
        if float(alpha) in mn and _mfoil_usable(mn[float(alpha)]):
            md = mn[float(alpha)]
            ax_cp.plot(md['upper']['x'], -md['upper']['cp'], ':', color=UP_COLOR, lw=1.2, alpha=0.5)
            ax_cp.plot(md['lower']['x'], -md['lower']['cp'], ':', color=LO_COLOR, lw=1.2, alpha=0.5)
            ax_cf.plot(md['upper']['x'], md['upper']['cf'], ':', color=UP_COLOR, lw=1.2, alpha=0.5)
            ax_cf.plot(md['lower']['x'], md['lower']['cf'], ':', color=LO_COLOR, lw=1.2, alpha=0.5)
        elif _xfoil_fill(alpha):
            xd = xf[float(alpha)]
            ax_cp.plot(xd['upper']['x'], -np.asarray(xd['upper']['cp']), '-.', color=UP_COLOR, lw=1.1, alpha=0.6)
            ax_cp.plot(xd['lower']['x'], -np.asarray(xd['lower']['cp']), '-.', color=LO_COLOR, lw=1.1, alpha=0.6)
            ax_cf.plot(xd['upper']['x'], xd['upper']['cf'], '-.', color=UP_COLOR, lw=1.1, alpha=0.6)
            ax_cf.plot(xd['lower']['x'], xd['lower']['cf'], '-.', color=LO_COLOR, lw=1.1, alpha=0.6)
        # Measured onsets (Somers orifices): triangles on the x-axis of the Cf
        # row; open marker = extrapolated beyond the recorded orifices.
        for side, color in (('upper', UP_COLOR), ('lower', LO_COLOR)):
            e = EXP_XTR.get(int(alpha), {}).get(side)
            if e is None: continue
            x_e, extrap = e
            ax_cf.plot(x_e, 0.0, marker='^', ms=9, color=color,
                       mfc='none' if extrap else color, mew=1.5,
                       clip_on=False, zorder=6, ls='none')
        for a in [ax_reo, ax_P, ax_n, ax_cp, ax_cf]: a.set_xlim(0, 1)
        ax_cf.set_ylim(0, 0.025 if max(alphas) > 6 else 0.008)
        ax_cp.grid(alpha=0.3); ax_cf.grid(alpha=0.3)
        ax_cf.set_xlabel('$x/c$')
    axs[3,0].set_ylabel('$-C_p$'); axs[4,0].set_ylabel('$C_f$')
    # legend
    surfh = [Line2D([],[],color=UP_COLOR, lw=2, label='upper'),
             Line2D([],[],color=LO_COLOR, lw=2, label='lower')]
    legend_items = []
    if 'str' in meshes:
        legend_items += [Line2D([],[],color='0.3', ls='-',  lw=0.8, label='L0 str'),
                         Line2D([],[],color='0.3', ls='-',  lw=1.6, label='L1 str'),
                         Line2D([],[],color='0.3', ls='-',  lw=2.4, label='L2 str')]
    if 'cav' in meshes:
        legend_items += [Line2D([],[],color='0.3', ls='--', lw=0.8, label='L0 cav'),
                         Line2D([],[],color='0.3', ls='--', lw=1.6, label='L1 cav'),
                         Line2D([],[],color='0.3', ls='--', lw=2.4, label='L2 cav')]
    if any_mfoil:
        legend_items += [Line2D([],[],color='0.3', ls=':',  lw=1.2, alpha=0.5, label='mfoil ($e^9$)')]
    if any_xfoil:
        legend_items += [Line2D([],[],color='0.3', ls='-.', lw=1.1, alpha=0.6, label='xfoil ($e^9$)')]
    if any_ff:
        legend_items += [Line2D([],[],color='0.3', ls=':', lw=1.2, alpha=0.5, label='FlexFoil $N$ ($e^9$)')]
    if any(EXP_XTR.get(int(a), {}).get(s) for a in alphas for s in ('upper','lower')):
        legend_items += [Line2D([],[],color='0.3', marker='^', ls='none', ms=8,
                                label='expt. onset'),
                         Line2D([],[],color='0.3', marker='^', ls='none', ms=8,
                                mfc='none', mew=1.5, label='expt. onset (extrap.)')]
    axs[0,-1].legend(handles=surfh, fontsize=8, frameon=False, loc='upper left')
    axs[3,-1].legend(handles=legend_items, fontsize=8, frameon=False, loc='upper right')
    plt.tight_layout(rect=(0,0,1,0.97))
    plt.savefig(f'{PD}/figs/{out_name}.pdf'); plt.savefig(f'/tmp/{out_name}.png', dpi=130)
    plt.close()
    print(f'wrote {out_name}.pdf')

def make_landscape_figure(alphas, out_name, title):
    """1 row × len(alphas) cols: landscape with upper/lower trajectories at each α."""
    fig, axs = plt.subplots(1, len(alphas), figsize=(6.0*len(alphas), 5.5), sharey=True)
    if len(alphas) == 1: axs = [axs]
    ReO_grid = np.logspace(-0.5, 5, 200); G_grid = np.linspace(0, 2.0, 160)
    RG, GG = np.meshgrid(ReO_grid, G_grid)
    A_grid = rate(RG, GG)
    LEVELS_PLT = [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.18, 0.195]
    for col, alpha in enumerate(alphas):
        ax = axs[col]
        cs = ax.contour(RG, GG, A_grid, levels=LEVELS_PLT, colors='k', linewidths=0.8, alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=7, fmt='%.3f')
        ax.axvline(RE_OMEGA_FLOOR, color='gray', ls='--', lw=0.7, alpha=0.4)
        ax.axhline(G_C, color='gray', ls=':', lw=0.5, alpha=0.4)
        # Trajectories from L1 only (production level)
        for mesh in ['cav', 'str']:
            d = case_dir(mesh, 'L1', alpha)
            if not os.path.exists(f"{d}/slice_centerSpan.pvtu"): continue
            for side, color in [('upper', UP_COLOR), ('lower', LO_COLOR)]:
                traj = trajectory(d, side=side)
                if traj is None: continue
                xs, ReO, Gamma, chi = traj
                valid = ~np.isnan(ReO) & ~np.isnan(Gamma) & (ReO > 0)
                if not valid.any(): continue
                idx = np.arange(len(xs))
                ReO_filled = np.interp(idx, idx[valid], np.log(np.maximum(ReO[valid], 1e-30)))
                G_filled = np.interp(idx, idx[valid], Gamma[valid])
                ReO_s = np.exp(gaussian_filter1d(ReO_filled, sigma=2.0))
                G_s = gaussian_filter1d(G_filled, sigma=2.0)
                ax.plot(ReO_s[valid], G_s[valid], MESH_LS[mesh], color=color, lw=LEVEL_LW['L1'])
        ax.set_xscale('log'); ax.set_xlim(0.5, 1e5); ax.set_ylim(0, 2.05)
        ax.set_xlabel('$Re_\\Omega = d^2|\\omega|/\\nu$')
        ax.set_title(rf'$\alpha={alpha}^\circ$', fontsize=10)
        ax.grid(alpha=0.3, which='major')
        if col == 0: ax.set_ylabel('$\\Gamma$')
    handles = [Line2D([],[],color=UP_COLOR, lw=2, label='upper'),
               Line2D([],[],color=LO_COLOR, lw=2, label='lower'),
               Line2D([],[],color='0.3', ls='-',  lw=1.6, label='str (O-grid)'),
               Line2D([],[],color='0.3', ls='--', lw=1.6, label='cav (unstructured)')]
    axs[-1].legend(handles=handles, fontsize=8, frameon=False, loc='lower right')
    plt.tight_layout(rect=(0,0,1,0.96))
    plt.savefig(f'{PD}/figs/{out_name}.pdf'); plt.savefig(f'/tmp/{out_name}.png', dpi=130)
    plt.close()
    print(f'wrote {out_name}.pdf')

def make_convergence_figure(alphas, out_name, title, meshes=None):
    """1 row × 2 wide columns per α: (residuals: cont solid + ν̃ dashed in one plot)
    and (C_L on left axis + C_D on right axis). C_L/C_D y-limits are computed
    from the latter HALF of each run so converged-value oscillations dominate
    the scale rather than the early transient. `meshes` filters mesh families
    (None=both, ['str'], or ['cav']).
    """
    if meshes is None: meshes = ['cav','str']
    fig, axs = plt.subplots(len(alphas), 2, figsize=(13, 4.5*len(alphas)), squeeze=False)
    for col_idx, alpha in enumerate(alphas):
        ax_res = axs[col_idx, 0]
        ax_cl  = axs[col_idx, 1]
        ax_cd  = ax_cl.twinx()
        cl_tail = []; cd_tail = []
        for mesh in meshes:
            for level in LEVELS_CF:  # configurable level set
                d = case_dir(mesh, level, alpha)
                if not os.path.exists(f"{d}/nonlinear_residual_v2.csv"): continue
                lw = LEVEL_LW[level]; ls = MESH_LS[mesh]
                color = ('C2' if mesh == 'cav' else 'C9')
                label = f'{mesh}-{level}'
                try:
                    rows = list(csv.reader(open(f'{d}/nonlinear_residual_v2.csv')))
                    h = [x.strip() for x in rows[0]]
                    ip = h.index('pseudo_step'); ic = h.index('0_cont'); inu = h.index('5_nuHat')
                    s, c, n = [], [], []
                    for r in rows[1:]:
                        if r and len(r) > inu:
                            try:
                                s.append(int(float(r[ip])))
                                c.append(float(r[ic])); n.append(float(r[inu]))
                            except: pass
                    s, c, n = np.array(s), np.array(c), np.array(n)
                    m = s >= 50
                    ax_res.semilogy(s[m], c[m], ls='-',  lw=lw, color=color, label=label)
                    ax_res.semilogy(s[m], n[m], ls='--', lw=lw, color=color, alpha=0.7)
                except Exception as e:
                    print(f'  res {mesh}/{level} α={alpha}: {e}')
                try:
                    rows = list(csv.reader(open(f'{d}/total_forces_v2.csv')))[1:]
                    sf, cl, cd = [], [], []
                    for r in rows:
                        if r and len(r) > 3:
                            try:
                                sf.append(int(float(r[1]))); cl.append(float(r[2])); cd.append(float(r[3]))
                            except: pass
                    sf, cl, cd = np.array(sf), np.array(cl), np.array(cd)
                    m = sf >= 50
                    ax_cl.plot(sf[m], cl[m], ls='-',  lw=lw, color=color, label=label)
                    ax_cd.plot(sf[m], cd[m], ls='--', lw=lw, color=color, alpha=0.7)
                    # Collect the LATER HALF of each run for y-axis scaling
                    if len(sf) > 4:
                        half = len(sf) // 2
                        cl_tail.append(cl[half:]); cd_tail.append(cd[half:])
                except Exception as e:
                    print(f'  force {mesh}/{level} α={alpha}: {e}')
        ax_res.set_xlabel('pseudo-step')
        ax_res.set_ylabel('residual (cont solid, $\\tilde\\nu$ dashed)')
        ax_res.set_yscale('log'); ax_res.set_ylim(1e-11, 1e-2)
        ax_res.set_title(f'α={alpha}°: residuals')
        ax_res.grid(alpha=0.3, which='both')
        ax_cl.set_xlabel('pseudo-step')
        ax_cl.set_ylabel('$C_L$ (solid)')
        ax_cd.set_ylabel('$C_D$ (dashed)')
        ax_cl.set_title(f'α={alpha}°: $C_L$, $C_D$')
        ax_cl.grid(alpha=0.3)
        if cl_tail:
            cl_all = np.concatenate(cl_tail); cd_all = np.concatenate(cd_tail)
            cl_lo, cl_hi = np.percentile(cl_all, [1, 99])
            cd_lo, cd_hi = np.percentile(cd_all, [1, 99])
            cl_pad = max(0.02*(cl_hi - cl_lo), 1e-4)
            cd_pad = max(0.02*(cd_hi - cd_lo), 1e-6)
            ax_cl.set_ylim(cl_lo - cl_pad, cl_hi + cl_pad)
            ax_cd.set_ylim(cd_lo - cd_pad, cd_hi + cd_pad)
        if col_idx == 0:
            ax_res.legend(fontsize=7, loc='upper right', ncol=2, frameon=False)
    plt.tight_layout(rect=(0,0,1,0.94))
    plt.savefig(f'{PD}/figs/{out_name}.pdf'); plt.savefig(f'/tmp/{out_name}.png', dpi=130)
    plt.close()
    print(f'wrote {out_name}.pdf')


def make_landscape_normal_figure(alphas, out_name, title, L_probe=0.01, n_probe=80):
    """Same landscape as make_landscape_figure, but each trajectory is computed
    from a WALL-NORMAL PROBE at every surface point: for each surface node,
    interpolate Re_Omega and Gamma along an L_probe-long line normal to the
    wall and take the max of each. The metrics are purely kinematic (no
    nuHat/amp_rate dependence) — valid as calibration inputs for the rate
    kernel.
    """
    fig, axs = plt.subplots(1, len(alphas), figsize=(6.0*len(alphas), 5.5), sharey=True)
    if len(alphas) == 1: axs = [axs]
    ReO_grid = np.logspace(-0.5, 5, 200); G_grid = np.linspace(0, 2.0, 160)
    RG, GG = np.meshgrid(ReO_grid, G_grid)
    A_grid = rate(RG, GG)
    LEVELS_PLT = [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.18, 0.195]
    for col, alpha in enumerate(alphas):
        ax = axs[col]
        cs = ax.contour(RG, GG, A_grid, levels=LEVELS_PLT, colors='k', linewidths=0.8, alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=7, fmt='%.3f')
        ax.axvline(RE_OMEGA_FLOOR, color='gray', ls='--', lw=0.7, alpha=0.4)
        ax.axhline(G_C, color='gray', ls=':', lw=0.5, alpha=0.4)
        for mesh in ['cav', 'str']:
            d = case_dir(mesh, 'L1', alpha)
            if not os.path.exists(f"{d}/slice_with_derived.pvtu"):
                print(f"  skip {mesh}/L1 α={alpha}: no slice_with_derived"); continue
            for side, color in [('upper', UP_COLOR), ('lower', LO_COLOR)]:
                try:
                    xs, ReO, Gam = wallnormal_max_metrics(d, side=side,
                                                          L_probe=L_probe, n_probe=n_probe)
                except Exception as e:
                    print(f"  skip {mesh}/L1 α={alpha} {side}: {e}"); continue
                valid = (ReO > 0) & np.isfinite(ReO) & np.isfinite(Gam)
                if not valid.any(): continue
                # Tiny smoothing to suppress probe-jitter noise without blurring
                # the LE endpoint (where upper and lower walks meet) into each
                # side's rooftop neighbors. Keep σ small enough that the LE
                # point is barely averaged.
                ReO_s = gaussian_filter1d(np.log(np.maximum(ReO, 1e-30)), sigma=0.5)
                Gam_s = gaussian_filter1d(Gam, sigma=0.5)
                ax.plot(np.exp(ReO_s[valid]), Gam_s[valid], MESH_LS[mesh],
                        color=color, lw=LEVEL_LW['L1'])
        ax.set_xscale('log'); ax.set_xlim(0.5, 1e5); ax.set_ylim(0, 2.05)
        ax.set_xlabel('$Re_\\Omega = d^2|\\omega|/\\nu$')
        ax.set_title(rf'$\alpha={alpha}^\circ$', fontsize=10)
        ax.grid(alpha=0.3, which='major')
        if col == 0: ax.set_ylabel('$\\Gamma$')
    handles = [Line2D([],[],color=UP_COLOR, lw=2, label='upper'),
               Line2D([],[],color=LO_COLOR, lw=2, label='lower'),
               Line2D([],[],color='0.3', ls='-',  lw=1.6, label='str (O-grid)'),
               Line2D([],[],color='0.3', ls='--', lw=1.6, label='cav (unstructured)')]
    axs[-1].legend(handles=handles, fontsize=8, frameon=False, loc='lower right')
    plt.tight_layout(rect=(0,0,1,0.96))
    plt.savefig(f'{PD}/figs/{out_name}.pdf'); plt.savefig(f'/tmp/{out_name}.png', dpi=130)
    plt.close()
    print(f'wrote {out_name}.pdf')

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if mode in ('low', 'all'):
        alphas = [0, 4]
        title_all = 'NLF(1)-0416 at $\\alpha\\in\\{0^\\circ,4^\\circ\\}$ — L0/L1/L2 × cav/str'
        title_str = 'NLF(1)-0416 at $\\alpha\\in\\{0^\\circ,4^\\circ\\}$ — structured (O-grid) L0/L1/L2'
        title_cav = 'NLF(1)-0416 at $\\alpha\\in\\{0^\\circ,4^\\circ\\}$ — unstructured (cavity) L0/L1/L2'
        make_cf_figure(alphas, 'nlf_cf_lowalpha',     title_all)
        make_cf_figure(alphas, 'nlf_cf_lowalpha_str', title_str, meshes=['str'])
        make_cf_figure(alphas, 'nlf_cf_lowalpha_cav', title_cav, meshes=['cav'])
        make_landscape_figure(alphas, 'nlf_landscape_lowalpha',
                              'NLF(1)-0416 amplification landscape, $\\alpha\\in\\{0^\\circ,4^\\circ\\}$')
        make_landscape_normal_figure(alphas, 'nlf_landscape_normal_lowalpha',
                              'NLF(1)-0416 amplification landscape (wall-normal probe), '
                              '$\\alpha\\in\\{0^\\circ,4^\\circ\\}$')
        make_convergence_figure(alphas, 'nlf_convergence_lowalpha',
                                'NLF(1)-0416 convergence, $\\alpha\\in\\{0^\\circ,4^\\circ\\}$ — cav + str')
        make_convergence_figure(alphas, 'nlf_convergence_lowalpha_str',
                                'NLF(1)-0416 convergence, $\\alpha\\in\\{0^\\circ,4^\\circ\\}$ — structured (O-grid)',
                                meshes=['str'])
        make_convergence_figure(alphas, 'nlf_convergence_lowalpha_cav',
                                'NLF(1)-0416 convergence, $\\alpha\\in\\{0^\\circ,4^\\circ\\}$ — unstructured (cavity)',
                                meshes=['cav'])
    if mode in ('high', 'all'):
        alphas = [9, 15]
        title_all = 'NLF(1)-0416 at $\\alpha\\in\\{9^\\circ,15^\\circ\\}$ — L0/L1/L2 × cav/str'
        title_str = 'NLF(1)-0416 at $\\alpha\\in\\{9^\\circ,15^\\circ\\}$ — structured (O-grid) L0/L1/L2'
        title_cav = 'NLF(1)-0416 at $\\alpha\\in\\{9^\\circ,15^\\circ\\}$ — unstructured (cavity) L0/L1/L2'
        make_cf_figure(alphas, 'nlf_cf_highalpha',     title_all)
        make_cf_figure(alphas, 'nlf_cf_highalpha_str', title_str, meshes=['str'])
        make_cf_figure(alphas, 'nlf_cf_highalpha_cav', title_cav, meshes=['cav'])
        make_landscape_normal_figure(alphas, 'nlf_landscape_normal_highalpha',
                              'NLF(1)-0416 amplification landscape (wall-normal probe), '
                              '$\\alpha\\in\\{9^\\circ,15^\\circ\\}$')
        make_landscape_figure(alphas, 'nlf_landscape_highalpha',
                              'NLF(1)-0416 amplification landscape, $\\alpha\\in\\{9^\\circ,15^\\circ\\}$')
        make_convergence_figure(alphas, 'nlf_convergence_highalpha',
                                'NLF(1)-0416 convergence, $\\alpha\\in\\{9^\\circ,15^\\circ\\}$ — cav + str')
        make_convergence_figure(alphas, 'nlf_convergence_highalpha_str',
                                'NLF(1)-0416 convergence, $\\alpha\\in\\{9^\\circ,15^\\circ\\}$ — structured (O-grid)',
                                meshes=['str'])
        make_convergence_figure(alphas, 'nlf_convergence_highalpha_cav',
                                'NLF(1)-0416 convergence, $\\alpha\\in\\{9^\\circ,15^\\circ\\}$ — unstructured (cavity)',
                                meshes=['cav'])
