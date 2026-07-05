"""NLF(1)-0416 contour with blunt-TE-aware refinement.

The NLF(1)-0416 has a finite-thickness blunt TE (z_TE ≈ ±0.00125, total face
0.0025 c). Unlike sharp-TE airfoils (NACA 0012), the TE first-step h_te is set
by the geometric thickness, not by the wall-normal first-cell size:

  h_te_L0 = 1/2 · z_TE_thickness = 0.5 · 0.0025 = 0.00125 c
  h_te halves per level along with every other length scale

The blunt face is split into TWO segments meeting at the geometric mid-point
(1, 0). The contour goes:

   (1, +z_TE) → upper surface → LE → lower surface → (1, -z_TE)
                                                          ↓
                                                 (1, 0)        (blunt face)
                                                          ↑
                                                   (close back to start)

Both blunt-face segments are walls and get the same chord-direction h_te ramp
as the airfoil surface ends. r_te varies per level: ~2.0 at L0, decays to ~1.
"""
import numpy as np
from scipy.interpolate import CubicSpline

NLF_DAT = '/home/qiqi/flexcompute/aft-sa/external/construct2d/nlf0416.dat'

def load_nlf_dat(path=NLF_DAT):
    """Load nlf0416.dat. Returns (251, 2) array. First/last rows give blunt TE z."""
    pts = []
    for ln in open(path).readlines()[1:]:
        a, b = ln.split()
        pts.append((float(a), float(b)))
    return np.array(pts)

def parameterize_by_arclength(pts):
    diffs = np.diff(pts, axis=0)
    ds = np.sqrt((diffs**2).sum(axis=1))
    s = np.concatenate([[0], np.cumsum(ds)])
    return s / s[-1]

def _build_half(N_half, h_te, r_te, dxds_0):
    """Build N_half s-values in [0, 0.5]: first chord step h_te, grow at r_te
    from TE, then transition to a quarter-sine distribution that clusters at LE
    ONLY (s=0.5). The previous half-cosine approach clustered at both s_anchor
    and s=0.5, causing a sudden 5-30× drop in cell size right after the ramp
    (the "double growth pattern" — fine cells reappearing at x ≈ 0.987 for NLF
    L1). The quarter-sine s = s_anchor + (0.5 - s_anchor) · sin(π/2 · k/K)
    is monotone: ds is largest at s_anchor (matching the ramp's tail) and
    smallest at s=0.5 (LE-clustering for the leading-edge BL).
    """
    ds_0 = h_te / abs(dxds_0)
    s_vals = [0.0]
    step_s = ds_0
    while len(s_vals) < N_half:
        next_s = s_vals[-1] + step_s
        if next_s >= 0.5:
            break
        n_remain = N_half - len(s_vals)
        avg_remaining = (0.5 - s_vals[-1]) / max(n_remain, 1)
        if step_s > avg_remaining:
            break
        s_vals.append(next_s)
        step_s *= r_te
    s_anchor = s_vals[-1]
    K = N_half - len(s_vals)
    if K > 0:
        k = np.arange(1, K + 1)
        # sin(π/2 · k/K) gives s monotone in [s_anchor, 0.5]
        # ds(k) ∝ cos(π/2 · k/K): max at k=0 (smooth blend w/ ramp tail), 0 at k=K (LE)
        s_le = s_anchor + (0.5 - s_anchor) * np.sin(np.pi/2 * k / K)
        s_vals.extend(s_le.tolist())
    return np.array(s_vals[:N_half])

def generate_contour_te(N, h_te, r_te=1.5):
    """Generate NLF(1)-0416 contour with blunt-TE refinement.

    Args:
        N: total number of points on the AIRFOIL SURFACE (upper + lower; excludes
           the blunt-face traversal segments that close the contour).
        h_te: chord-direction step at the TE corner (= 0.5 · z_TE_thickness at L0,
           halves per level). For NLF(1)-0416 z_TE=0.0025, so L0: 1.25e-3, L1:
           6.25e-4, L2: 3.125e-4, etc.
        r_te: growth ratio along the TE ramp (~2.0 at L0, decays toward 1 at fine).

    Returns:
        (M, 2) ndarray, closed-contour order:
            (1, +z_TE) → upper surface → LE → lower surface → (1, -z_TE)
            → (1, 0) → (1, +z_TE) [closes back to start via blunt face]

        Total length M = N + n_blunt + 1 where n_blunt = ceil(z_TE / h_te) per
        side (rounded so the last cell exactly hits (1, 0) midpoint).
    """
    pts = load_nlf_dat()
    s_arc = parameterize_by_arclength(pts)
    sp_x = CubicSpline(s_arc, pts[:, 0], bc_type='natural')
    sp_z = CubicSpline(s_arc, pts[:, 1], bc_type='natural')

    # Chord movement per s near s=0 (small step into upper surface from TE)
    eps = 1e-5
    dxds_0 = (sp_x(eps) - sp_x(0)) / eps

    # Build the airfoil surface s-distribution: TE-refined on both halves
    N_half = N // 2 + 1
    s_upper = _build_half(N_half, h_te, r_te, dxds_0)
    s_lower_flipped = 1.0 - s_upper[::-1]
    s_air = np.concatenate([s_upper, s_lower_flipped[1:]])
    if len(s_air) > N:
        s_air = s_air[:N]
    elif len(s_air) < N:
        s_air = np.linspace(s_air[0], s_air[-1], N)

    x_air = sp_x(s_air)
    z_air = sp_z(s_air)

    # Force airfoil-endpoint values to the exact blunt-TE coordinates
    z_te_upper = float(pts[0, 1])    # +0.00125 for NLF
    z_te_lower = float(pts[-1, 1])   # -0.00125
    x_air[0] = 1.0;  z_air[0] = z_te_upper
    x_air[-1] = 1.0; z_air[-1] = z_te_lower

    # Blunt-face traversal: split into 2·n_per_side segments (per Principle 3 in
    # memory:airfoil-mesh-refinement-practices), with the midpoint (1, 0) always
    # present and a uniform spacing of h_te between consecutive nodes on each
    # half of the face. The contour goes:
    #   x_air[N-1] = (1, z_te_lower)
    #   then interior nodes z_blunt_low (going from z_te_lower up to 0)
    #   then the midpoint (1, 0)
    #   then interior nodes z_blunt_up (going from 0 up to z_te_upper)
    #   closure back to x_air[0] = (1, z_te_upper) is implicit (added by writer).
    n_per_side = max(1, int(round(abs(z_te_upper) / h_te)))   # cells per half-face
    z_blunt_low = np.linspace(z_te_lower, 0.0, n_per_side + 1)[1:-1]
    z_blunt_up = np.linspace(0.0, z_te_upper, n_per_side + 1)[1:-1]
    z_blunt = np.concatenate([z_blunt_low, [0.0], z_blunt_up])
    x_blunt = np.ones_like(z_blunt)
    return np.column_stack([np.concatenate([x_air, x_blunt]),
                            np.concatenate([z_air, z_blunt])])

if __name__ == '__main__':
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    LEVELS = [
        # (tag, N, h_te, r_te)  — proper refinement ladder for NLF(1)-0416
        ('L0',  200, 1.25e-3, 2.00),
        ('L1',  400, 6.25e-4, 1.50),
        ('L2',  800, 3.125e-4, 1.25),
        ('L3', 1600, 1.5625e-4, 1.125),
        ('L4', 3200, 7.8125e-5, 1.0625),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    for tag, N, h_te, r_te in LEVELS:
        c = generate_contour_te(N, h_te, r_te)
        axs[0,0].plot(c[:,0], c[:,1], '-', lw=0.8, label=f'{tag}: N={N}, h_te={h_te:.2e}, r_te={r_te}')
    axs[0,0].set_aspect('equal'); axs[0,0].legend(fontsize=7); axs[0,0].grid(alpha=0.3)
    axs[0,0].set_title('NLF(1)-0416 contour with blunt-TE refinement')

    # LE zoom
    for tag, N, h_te, r_te in LEVELS:
        c = generate_contour_te(N, h_te, r_te)
        axs[0,1].plot(c[:,0], c[:,1], 'o-', ms=2, lw=0.7, label=tag)
    raw = load_nlf_dat()
    axs[0,1].plot(raw[:,0], raw[:,1], 'x', color='red', ms=3, label='raw .dat')
    axs[0,1].set_xlim(-0.005, 0.02); axs[0,1].set_ylim(-0.015, 0.015)
    axs[0,1].set_aspect('equal'); axs[0,1].legend(fontsize=7); axs[0,1].grid(alpha=0.3)
    axs[0,1].set_title('LE zoom (raw .dat in red ×)')

    # TE zoom — blunt face visible
    for tag, N, h_te, r_te in LEVELS:
        c = generate_contour_te(N, h_te, r_te)
        axs[1,0].plot(c[:,0], c[:,1], 'o-', ms=3, lw=0.7, label=tag)
    axs[1,0].plot(raw[:,0], raw[:,1], 'x', color='red', ms=3, label='raw .dat')
    axs[1,0].set_xlim(0.985, 1.005); axs[1,0].set_ylim(-0.003, 0.003)
    axs[1,0].set_aspect('equal'); axs[1,0].legend(fontsize=7); axs[1,0].grid(alpha=0.3)
    axs[1,0].set_title('TE zoom: blunt face split at (1, 0), with TE-corner refinement')

    # Print first/last cell sizes at TE
    print(f"{'tag':>4s} {'N':>5s} {'h_te tgt':>10s} {'r_te':>6s} {'1st upper':>11s} {'1st blunt':>11s} {'Mfinal':>7s}")
    for tag, N, h_te, r_te in LEVELS:
        c = generate_contour_te(N, h_te, r_te)
        # Distance from first point along upper surface (s=0+ direction)
        d_upper = np.linalg.norm(c[1] - c[0])
        # Distance from last airfoil point to first blunt point
        # Airfoil ends at index N-1; blunt starts at N
        d_blunt = np.linalg.norm(c[N] - c[N-1])  # from (1, z_te_lower) to first blunt interior
        axs[1,1].text(0.02, 0.85 - 0.15*LEVELS.index((tag, N, h_te, r_te)),
                      f'{tag}: 1st upper-cell={d_upper:.3e}, 1st blunt-cell={d_blunt:.3e}',
                      transform=axs[1,1].transAxes, fontsize=10)
        print(f"{tag:>4s} {N:>5d} {h_te:>10.3e} {r_te:>6.2f} {d_upper:>11.3e} {d_blunt:>11.3e} {len(c):>7d}")
    axs[1,1].axis('off')
    plt.tight_layout()
    plt.savefig('/tmp/nlf_contour_te.png', dpi=140)
    print('wrote /tmp/nlf_contour_te.png')
