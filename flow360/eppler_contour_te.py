"""Eppler 387 contour with blunt-TE-aware refinement.

The Eppler 387 in NASA TM 4062 has its trailing edge thickened from 0 to
0.01" on a 6" chord = 0.001667 c TE thickness, blended starting at x/c=0.95
(see Table I). z_TE_upper = +0.00083 c, z_TE_lower = -0.00083 c.

TE-refinement rule (per memory:airfoil-mesh-refinement-practices, blunt-TE
case — same as NLF(1)-0416):

  h_te_L0 = 1/2 · z_TE_thickness = 0.5 · 0.001667 = 0.000833 c
  h_te halves per level along with every other length scale.
  r_te varies per level (~2 at L0, decays toward 1 at fine levels).

The blunt face is split into TWO segments meeting at the geometric mid-point
(1, 0). The contour goes:

   (1, +z_TE) → upper surface → LE → lower surface → (1, -z_TE)
                                                          ↓
                                                 (1, 0)        (blunt face)
                                                          ↑
                                                   (close back to start)
"""
import numpy as np
from scipy.interpolate import CubicSpline

E387_DAT = '/home/qiqi/flexcompute/sa-ai/external/construct2d/eppler387.dat'

def load_e387_dat(path=E387_DAT):
    """Load eppler387.dat. Returns (56, 2) array. First/last rows give blunt TE z."""
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
    from TE, then quarter-sine blend to LE clustering at s=0.5.  See NLF
    nlf_contour_te._build_half for the rationale (quarter-sine avoids the
    "double growth pattern" of a piecewise full cosine).
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
        s_le = s_anchor + (0.5 - s_anchor) * np.sin(np.pi/2 * k / K)
        s_vals.extend(s_le.tolist())
    return np.array(s_vals[:N_half])

def generate_contour_te(N, h_te, r_te=1.5):
    """Generate Eppler 387 contour with blunt-TE refinement.

    Args:
        N: total points on the AIRFOIL SURFACE (excludes blunt-face segments).
        h_te: chord-direction step at the TE corner.  For E387 with
              z_TE_thickness=0.001667 c, L0 uses h_te = 0.000833.
        r_te: growth ratio along the TE ramp.

    Returns:
        Closed contour (1, +z_TE) → upper → LE → lower → (1, -z_TE) →
        blunt-face midpoint (1, 0) → blunt-face upper ramp → back to start.
    """
    pts = load_e387_dat()
    s_arc = parameterize_by_arclength(pts)
    sp_x = CubicSpline(s_arc, pts[:, 0], bc_type='natural')
    sp_z = CubicSpline(s_arc, pts[:, 1], bc_type='natural')

    eps = 1e-5
    dxds_0 = (sp_x(eps) - sp_x(0)) / eps

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

    z_te_upper = float(pts[0, 1])    # +0.00083 for E387
    z_te_lower = float(pts[-1, 1])   # -0.00083
    x_air[0] = 1.0;  z_air[0] = z_te_upper
    x_air[-1] = 1.0; z_air[-1] = z_te_lower

    n_per_side = max(1, int(round(abs(z_te_upper) / h_te)))
    z_blunt_low = np.linspace(z_te_lower, 0.0, n_per_side + 1)[1:-1]
    z_blunt_up = np.linspace(0.0, z_te_upper, n_per_side + 1)[1:-1]
    z_blunt = np.concatenate([z_blunt_low, [0.0], z_blunt_up])
    x_blunt = np.ones_like(z_blunt)
    return np.column_stack([np.concatenate([x_air, x_blunt]),
                            np.concatenate([z_air, z_blunt])])

if __name__ == '__main__':
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    LEVELS = [
        # (tag, N, h_te, r_te) — proper refinement ladder for Eppler 387
        ('L0',  200, 8.33e-4, 2.00),
        ('L1',  400, 4.17e-4, 1.50),
        ('L2',  800, 2.08e-4, 1.25),
        ('L3', 1600, 1.04e-4, 1.125),
        ('L4', 3200, 5.21e-5, 1.0625),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    for tag, N, h_te, r_te in LEVELS:
        c = generate_contour_te(N, h_te, r_te)
        axs[0,0].plot(c[:,0], c[:,1], '-', lw=0.8,
                      label=f'{tag}: N={N}, h_te={h_te:.2e}, r_te={r_te}')
    axs[0,0].set_aspect('equal'); axs[0,0].legend(fontsize=7); axs[0,0].grid(alpha=0.3)
    axs[0,0].set_title('Eppler 387 contour with blunt-TE refinement')

    for tag, N, h_te, r_te in LEVELS:
        c = generate_contour_te(N, h_te, r_te)
        axs[0,1].plot(c[:,0], c[:,1], 'o-', ms=2, lw=0.7, label=tag)
    raw = load_e387_dat()
    axs[0,1].plot(raw[:,0], raw[:,1], 'x', color='red', ms=3, label='raw .dat')
    axs[0,1].set_xlim(-0.005, 0.025); axs[0,1].set_ylim(-0.015, 0.020)
    axs[0,1].set_aspect('equal'); axs[0,1].legend(fontsize=7); axs[0,1].grid(alpha=0.3)
    axs[0,1].set_title('LE zoom (raw .dat in red ×)')

    for tag, N, h_te, r_te in LEVELS:
        c = generate_contour_te(N, h_te, r_te)
        axs[1,0].plot(c[:,0], c[:,1], 'o-', ms=3, lw=0.7, label=tag)
    axs[1,0].plot(raw[:,0], raw[:,1], 'x', color='red', ms=3, label='raw .dat')
    axs[1,0].set_xlim(0.94, 1.005); axs[1,0].set_ylim(-0.003, 0.012)
    axs[1,0].set_aspect('equal'); axs[1,0].legend(fontsize=7); axs[1,0].grid(alpha=0.3)
    axs[1,0].set_title('TE zoom: blunt face split at (1, 0)')

    print(f"{'tag':>4s} {'N':>5s} {'h_te tgt':>10s} {'r_te':>6s} {'1st upper':>11s} {'1st blunt':>11s} {'Mfinal':>7s}")
    for tag, N, h_te, r_te in LEVELS:
        c = generate_contour_te(N, h_te, r_te)
        d_upper = np.linalg.norm(c[1] - c[0])
        d_blunt = np.linalg.norm(c[N] - c[N-1])
        axs[1,1].text(0.02, 0.85 - 0.15*LEVELS.index((tag, N, h_te, r_te)),
                      f'{tag}: 1st upper-cell={d_upper:.3e}, 1st blunt-cell={d_blunt:.3e}',
                      transform=axs[1,1].transAxes, fontsize=10)
        print(f"{tag:>4s} {N:>5d} {h_te:>10.3e} {r_te:>6.2f} {d_upper:>11.3e} {d_blunt:>11.3e} {len(c):>7d}")
    axs[1,1].axis('off')
    plt.tight_layout()
    plt.savefig('/tmp/eppler_contour_te.png', dpi=140)
    print('wrote /tmp/eppler_contour_te.png')
