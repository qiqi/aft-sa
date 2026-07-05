"""Generate NACA0012 contour with explicit TE refinement: chord-direction spacing
ramps geometrically from h_te at TE outward to standard cosine spacing in the
mid-chord region, on BOTH upper and lower surfaces. The transition is smooth.
"""
import numpy as np
from scipy.interpolate import CubicSpline

NACA_DAT = '/home/qiqi/flexcompute/aft-sa/external/construct2d/naca0012.dat'

def load_naca_dat(path=NACA_DAT):
    """Load an airfoil dat file (selig format). For sharp-TE airfoils (NACA0012)
    the original dat file may end at slightly inset (e.g. x=0.9999) on the lower
    side — snap that endpoint to exactly (1, 0) so the contour closes geometrically.
    For finite-thickness TE airfoils (NLF0416, z_TE ≈ ±0.001), leave endpoints alone.
    """
    pts = []
    for ln in open(path).readlines()[1:]:
        a, b = ln.split()
        pts.append((float(a), float(b)))
    pts = np.array(pts)
    # Snap only if x is near 1 AND z is also near zero (sharp TE).
    if abs(pts[0, 0] - 1.0) < 1e-3 and abs(pts[0, 1]) < 1e-3:  pts[0]  = [1.0, 0.0]
    if abs(pts[-1, 0] - 1.0) < 1e-3 and abs(pts[-1, 1]) < 1e-3: pts[-1] = [1.0, 0.0]
    return pts

def parameterize_by_arclength(pts):
    diffs = np.diff(pts, axis=0)
    ds = np.sqrt((diffs**2).sum(axis=1))
    s = np.concatenate([[0], np.cumsum(ds)])
    return s / s[-1]

def generate_contour_te(N, h_te=None, r_te=1.15, close_te=True, dat_path=None, **kwargs):
    """N total contour points ordered TE_upper → LE → TE_lower.
    If h_te is set, the chord-direction spacing at BOTH TE endpoints (s=0 and s=1)
    ramps geometrically from h_te outward with growth ratio r_te, transitioning
    smoothly to the standard piecewise-cosine clustering in the bulk.
    dat_path: airfoil dat file (default: NACA0012). Pass NLF0416 etc. by overriding.
    """
    pts = load_naca_dat(dat_path if dat_path else NACA_DAT)
    s_arc = parameterize_by_arclength(pts)
    spline_x = CubicSpline(s_arc, pts[:, 0], bc_type='natural')
    spline_z = CubicSpline(s_arc, pts[:, 1], bc_type='natural')

    if h_te is None:
        # Standard piecewise-cosine fallback
        half = N // 2
        k = np.arange(half + 1)
        s1 = 0.25 * (1 - np.cos(np.pi * k / half))             # [0, 0.5]
        s2 = 0.5 + 0.25 * (1 - np.cos(np.pi * k[1:] / half))   # (0.5, 1]
        s_new = np.concatenate([s1, s2])
        if len(s_new) > N: s_new = s_new[:N]
    else:
        # Build half on [0, 0.5] with TE refinement at s=0
        N_half = N // 2 + 1
        s_upper = _build_half(N_half, h_te, r_te, spline_x, spline_z)
        # Mirror to [0.5, 1] with TE refinement at s=1
        s_lower_flipped = 1.0 - s_upper[::-1]   # now in [0.5, 1], TE refinement at s=1
        # Concatenate, dropping the duplicate point at s=0.5
        s_new = np.concatenate([s_upper, s_lower_flipped[1:]])
        if len(s_new) > N: s_new = s_new[:N]
        elif len(s_new) < N:
            s_new = np.linspace(s_new[0], s_new[-1], N)

    x_new = spline_x(s_new)
    z_new = spline_z(s_new)
    if close_te:
        # Snap both endpoints to exact (1, 0). For h_te-refined contours,
        # the natural spline values at s=0 and s=1 already approach (1, 0)
        # closely, so this is a small snap.
        x_new[0] = 1.0; z_new[0] = 0.0
        x_new[-1] = 1.0; z_new[-1] = 0.0
    return np.column_stack([x_new, z_new])

def _build_half(N_half, h_te, r_te, spline_x, spline_z):
    """Build s-values in [0, 0.5] with TE refinement at s=0.

    Algorithm:
      1. Geometric ramp from s=0 outward: chord-distance from TE goes
         h_te, h_te(1+r), h_te(1+r+r²), ... until step matches cosine baseline.
      2. Fill remainder of [s_ramp_end, 0.5] with cosine (dense at both ends).
    """
    # Map arc-parameter s near 0 to chord distance from TE
    eps = 1e-5
    dxds_0 = (spline_x(eps) - spline_x(0)) / eps   # negative (x decreases as s grows from 0)
    # Note: actual chord movement per unit s near TE is |dxds_0|

    # Build geometric ramp in CHORD coordinate (distance from x=1)
    ramp_chord = [0.0]
    step = h_te
    # Estimate the "cosine baseline" step size at the TE-end of a half-cosine:
    # standard half-cosine first step: 0.5·(1 - cos(π/N_half)) ≈ π²/(4·N_half²)
    # which corresponds to chord distance baseline ≈ |dxds_0| · π²/(4·N_half²)
    # We stop the geometric ramp when the geometric step matches or exceeds this baseline
    cos_first_ds = 0.5 * (1 - np.cos(np.pi / N_half))      # first ds in cosine
    chord_baseline = abs(dxds_0) * cos_first_ds              # first chord step under pure cosine
    while step < 3.0 * chord_baseline:
        ramp_chord.append(ramp_chord[-1] + step)
        step *= r_te
        if ramp_chord[-1] + step > 0.3 * abs(dxds_0) * 0.5:
            break
    ramp_chord = np.array(ramp_chord)

    # Convert chord distance from TE → s parameter (linear approx near TE)
    s_ramp = ramp_chord / abs(dxds_0)
    s_ramp = s_ramp[s_ramp < 0.45]

    n_ramp = len(s_ramp)
    n_remaining = N_half - n_ramp

    # Fill the remaining s-range [s_ramp[-1], 0.5] with cosine clustering
    if n_remaining > 1:
        k = np.arange(n_remaining + 1)  # +1 because cos endpoint shared with ramp
        s_rest = s_ramp[-1] + (0.5 - s_ramp[-1]) * 0.5 * (1 - np.cos(np.pi * k / n_remaining))
        # Drop the first point of s_rest (duplicate with s_ramp[-1])
        s_rest = s_rest[1:]
        s_half = np.concatenate([s_ramp, s_rest])
    else:
        s_half = np.append(s_ramp, [0.5])

    # Pad if short
    while len(s_half) < N_half:
        s_half = np.append(s_half, 0.5)
    return s_half[:N_half]


if __name__ == '__main__':
    print(f"{'N':>5s} {'h_te target':>12s} {'dx_TE upper':>13s} {'dx_TE lower':>13s} {'ratio':>8s}")
    for N, h_te in [(200, 8e-6), (400, 4e-6), (800, 2e-6), (1600, 1e-6)]:
        c = generate_contour_te(N, h_te=h_te)
        x = c[:, 0]
        # last few upper points (closest to TE upper, at the start of contour)
        # contour goes TE_upper(0) → LE → TE_lower(N-1)
        dx_upper = abs(c[1, 0] - c[0, 0])
        dx_lower = abs(c[-1, 0] - c[-2, 0])
        # find largest jump near TE
        max_dx_te = max(dx_upper, dx_lower)
        ratio = max_dx_te / h_te
        print(f"{N:>5d} {h_te:>12.2e} {dx_upper:>13.2e} {dx_lower:>13.2e} {ratio:>8.2f}")

    print("\n=== checking continuity near TE ===")
    c = generate_contour_te(400, h_te=21e-6)
    print("First 6 points (TE upper region):")
    for i in range(6):
        print(f"  [{i}] x={c[i,0]:.8f} z={c[i,1]:+.6e}")
    print("Last 6 points (TE lower region):")
    for i in range(-6, 0):
        print(f"  [{i}] x={c[i,0]:.8f} z={c[i,1]:+.6e}")
    # consecutive distances
    dd = np.linalg.norm(np.diff(c, axis=0), axis=1)
    print(f"\nFirst 5 inter-point distances: {dd[:5]}")
    print(f"Last 5 inter-point distances: {dd[-5:]}")
