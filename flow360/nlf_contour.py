"""Generate NLF(1)-0416 contour points by spline-interpolating the 251-point
nlf0416.dat, with cosine clustering at LE+TE. Single shared source of truth
for both the in-house cavity mesher and Construct2D — same N(level) → same
analytical geometry sampled at the same density on each grid level.

This is the analog of naca_contour.py for NLF. Both must be plumbed through
the L0..L4 mesh ladder; see memory:airfoil-mesh-refinement-practices.
"""
import numpy as np
from scipy.interpolate import CubicSpline

NLF_DAT = '/home/qiqi/flexcompute/aft-sa/external/construct2d/nlf0416.dat'

def load_nlf_dat(path=NLF_DAT):
    """Load nlf0416.dat (251 points, ordered TE → upper → LE → lower → TE)."""
    pts = []
    for ln in open(path).readlines()[1:]:  # skip header line
        a, b = ln.split()
        pts.append((float(a), float(b)))
    return np.array(pts)

def parameterize_by_arclength(pts):
    diffs = np.diff(pts, axis=0)
    ds = np.sqrt((diffs**2).sum(axis=1))
    s = np.concatenate([[0], np.cumsum(ds)])
    return s / s[-1]

def cosine_distribution(N):
    """Piecewise cosine: dense at s=0 (TE), s=0.5 (LE), s=1 (TE)."""
    half = N // 2
    k1 = np.arange(half + 1)
    s1 = 0.5 * (1 - np.cos(np.pi * k1 / half)) * 0.5
    k2 = np.arange(half + 1)
    s2 = 0.5 + 0.5 * (1 - np.cos(np.pi * k2 / half)) * 0.5
    s_out = np.concatenate([s1, s2[1:]])
    if len(s_out) > N:
        s_out = s_out[:N]
    elif len(s_out) < N:
        s_out = np.concatenate([s_out, np.linspace(s_out[-1], 1, N - len(s_out) + 1)[1:]])
    return s_out

def generate_contour(N, close_te=False):
    """N-point smooth spline contour. close_te=True forces TE closure to (1,0)."""
    pts = load_nlf_dat()
    s = parameterize_by_arclength(pts)
    spline_x = CubicSpline(s, pts[:, 0], bc_type='natural')
    spline_z = CubicSpline(s, pts[:, 1], bc_type='natural')
    s_new = cosine_distribution(N)
    x = spline_x(s_new); z = spline_z(s_new)
    if close_te:
        x[0] = 1.0; z[0] = 0.0
        x[-1] = 1.0; z[-1] = 0.0
    return np.column_stack([x, z])

def write_construct2d_dat(N, path):
    """Write a Construct2D-style .dat (header + N rows of 'x z') for the spline contour."""
    c = generate_contour(N, close_te=False)
    with open(path, 'w') as f:
        f.write('nlf0416\n')
        for x, z in c:
            f.write(f'{x:.7f}  {z:.7f}\n')
    return c

if __name__ == '__main__':
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))

    # Full airfoil at multiple N
    for N in [200, 400, 800, 1600, 3200]:
        c = generate_contour(N)
        axs[0,0].plot(c[:,0], c[:,1], '-', lw=0.8, label=f'N={N}')
    axs[0,0].set_aspect('equal'); axs[0,0].set_title('NLF(1)-0416 smooth contour at each N')
    axs[0,0].legend(fontsize=8); axs[0,0].grid(alpha=0.3)

    # LE zoom: show 45-degree problem of raw .dat vs smooth spline
    raw = load_nlf_dat()
    axs[0,1].plot(raw[:,0], raw[:,1], 'o-', color='C3', ms=3, lw=0.8, label=f'raw .dat (N={len(raw)})')
    for N in [200, 800, 3200]:
        c = generate_contour(N)
        axs[0,1].plot(c[:,0], c[:,1], '-', lw=0.8, label=f'spline N={N}')
    axs[0,1].set_xlim(-0.005, 0.015); axs[0,1].set_ylim(-0.012, 0.012)
    axs[0,1].set_aspect('equal'); axs[0,1].set_title('Leading-edge zoom: raw vs spline')
    axs[0,1].legend(fontsize=8); axs[0,1].grid(alpha=0.3)

    # Point density vs x
    for N in [200, 400, 800, 1600]:
        c = generate_contour(N)
        # bin into x bins, count points per bin (rough density measure)
        edges = np.linspace(0, 1, 50)
        h, _ = np.histogram(c[:,0], bins=edges)
        axs[1,0].plot(0.5*(edges[:-1]+edges[1:]), h, '-', label=f'N={N}')
    axs[1,0].set_xlabel('x/c'); axs[1,0].set_ylabel('# points per x-bin')
    axs[1,0].set_title('Surface point density along chord')
    axs[1,0].legend(fontsize=8); axs[1,0].grid(alpha=0.3)

    # TE detail
    for N in [200, 800, 3200]:
        c = generate_contour(N)
        axs[1,1].plot(c[:,0], c[:,1], 'o-', ms=2, lw=0.6, label=f'N={N}')
    axs[1,1].plot(raw[:,0], raw[:,1], 'x', color='C3', ms=4, label='raw .dat')
    axs[1,1].set_xlim(0.95, 1.005); axs[1,1].set_ylim(-0.005, 0.005)
    axs[1,1].set_aspect('equal'); axs[1,1].set_title('Trailing-edge zoom')
    axs[1,1].legend(fontsize=8); axs[1,1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/nlf_contour_test.png', dpi=140)
    print('wrote /tmp/nlf_contour_test.png')

    # Sanity: 6 smallest-x points at each level
    for N in [200, 400, 800]:
        c = generate_contour(N)
        idx = np.argsort(c[:,0])[:6]
        print(f'\n=== N={N}, 6 smallest-x points ===')
        for i in idx:
            print(f'  x={c[i,0]:.6f}  z={c[i,1]:.6f}')
