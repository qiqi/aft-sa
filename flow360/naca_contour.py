"""Generate NACA0012 contour points using EXACT same shape as Construct2D consumes
(the 319-point sharp-TE naca0012.dat), spline-interpolated to N cosine-clustered points.

For mesh refinement, both cavity and Construct2D should use the same underlying shape
sampled at the same N at each level.
"""
import numpy as np
from scipy.interpolate import CubicSpline

NACA_DAT = '/home/qiqi/flexcompute/aft-sa/external/construct2d/naca0012.dat'

def load_naca_dat(path=NACA_DAT):
    """Load Construct2D's naca0012.dat (319 sharp-TE points, ordered TE→LE→TE on upper→lower)."""
    pts = []
    for ln in open(path).readlines()[1:]:  # skip header
        a, b = ln.split()
        pts.append((float(a), float(b)))
    return np.array(pts)

def parameterize_by_arclength(pts):
    """Parameterize the contour by cumulative arclength s in [0, 1]."""
    diffs = np.diff(pts, axis=0)
    ds = np.sqrt((diffs**2).sum(axis=1))
    s = np.concatenate([[0], np.cumsum(ds)])
    return s / s[-1]

def cosine_distribution(N):
    """Generate N cosine-clustered parameter values in [0,1], densest at endpoints (TE)
    AND at the middle (LE). For an airfoil ordered TE→LE→TE, that means dense at LE and TE.
    Use double cosine: spacing dense at s=0, s=0.5, s=1."""
    k = np.arange(N)
    # double cosine: cos(π·k/(N-1)) gives one cosine; we want clustering at both ends
    # and at midpoint -> use cos(2π·k/(N-1)) component
    # Simpler: full-cycle cosine in [0, 2π] mapped to s in [0,1]
    theta = np.pi * k / (N - 1)            # 0..π
    s = 0.5 * (1 - np.cos(2*theta))         # full cycle: 0 at k=0, peaks at k=(N-1)/2 (LE), 0 at k=N-1 (TE again)
    # That's wrong — gives non-monotonic. Use a different mapping.
    # For airfoil parameterized 0=TE_upper, 0.5=LE, 1=TE_lower, we want clustering at 0, 0.5, 1
    # Use: s = (1 - cos(π·k/(N-1)))/2 first to cluster at endpoints, then re-cluster the middle
    # Simpler and standard: use clustering function that puts higher density near s=0, 0.5, 1
    # Construct via PIECEWISE cosine — each half is cosine-clustered
    half = N // 2
    k1 = np.arange(half + 1)
    s1 = 0.5 * (1 - np.cos(np.pi * k1 / half)) * 0.5  # s ∈ [0, 0.5], dense at 0 and 0.5
    k2 = np.arange(half + 1)
    s2 = 0.5 + 0.5 * (1 - np.cos(np.pi * k2 / half)) * 0.5  # s ∈ [0.5, 1]
    # combine; remove duplicate at 0.5
    s_out = np.concatenate([s1, s2[1:]])
    if len(s_out) > N:
        s_out = s_out[:N]
    elif len(s_out) < N:
        # pad with linear at start
        s_out = np.concatenate([s_out, np.linspace(s_out[-1], 1, N - len(s_out) + 1)[1:]])
    return s_out

def generate_contour(N, plot=False, close_te=False):
    """Generate N contour points by spline interpolation of naca0012.dat with cosine clustering
    around both LE and TE. close_te=False forces TE closure (start=end=(1,0))."""
    pts = load_naca_dat()
    s = parameterize_by_arclength(pts)
    spline_x = CubicSpline(s, pts[:, 0], bc_type='natural')
    spline_z = CubicSpline(s, pts[:, 1], bc_type='natural')
    s_new = cosine_distribution(N)
    x_new = spline_x(s_new)
    z_new = spline_z(s_new)
    if close_te:
        # force endpoints to exactly (1, 0) for sharp-TE closure
        x_new[0] = 1.0; z_new[0] = 0.0
        x_new[-1] = 1.0; z_new[-1] = 0.0
    return np.column_stack([x_new, z_new])

if __name__ == '__main__':
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for N in [200, 400, 800, 1600, 3200]:
        c = generate_contour(N)
        axs[0].plot(c[:,0], c[:,1], '-', label=f'N={N}')
        # density: spacing along chord on upper surface
        up = c[c.shape[0]//2:]  # last half is upper (TE→LE goes negative-to-positive z)... actually need to check
        axs[1].plot(c[:,0], np.gradient(np.arange(len(c[:,0])), c[:,0]), label=f'N={N}')
    axs[0].set_aspect('equal'); axs[0].set_title('NACA0012 contours at each level'); axs[0].legend(fontsize=8)
    axs[0].set_xlim(-0.02, 0.05); axs[0].set_ylim(-0.04, 0.04)
    axs[1].set_xlim(0, 0.05); axs[1].set_yscale('log')
    axs[1].set_title('point density vs x'); axs[1].set_xlabel('x/c')
    plt.tight_layout()
    plt.savefig('/tmp/contour_test.png', dpi=120)
    print(f"wrote /tmp/contour_test.png")
    # quick check: print first 6 points at each level
    for N in [200, 400, 800]:
        c = generate_contour(N)
        # find points with smallest x (LE region)
        idx = np.argsort(c[:,0])[:6]
        print(f"\n=== N={N}, 6 smallest-x points ===")
        for i in idx:
            print(f"  x={c[i,0]:.6f}  z={c[i,1]:.6f}")
