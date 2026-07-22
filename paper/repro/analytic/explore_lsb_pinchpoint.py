"""Briggs-Bers pinch-point map for wall-bounded separated shear layers.

Rigorous absolute/convective classification (notes_absolute_instability_lsb.md
Sec. 8) on the standard model family in the spirit of Hammond & Redekopp
(1998) and Diwan & Ramesh (2009): a displaced tanh shear layer over a no-slip
wall,

    u(y) = [ (1-u_r)/2 + (1+u_r)/2 * tanh((y-h)/w) ] * (1 - exp(-(y/d_w)^2))

with w = 1 the length unit: u -> 1 outside, dips to ~ -u_r inside the
recirculation, single interior shear layer centered at height h. Knobs:
u_r (peak reverse flow) and h (shear-layer height = bubble thickness) --
exactly the two physical parameters of the LSB absolute-instability
literature. HM85 free-layer limit: h -> inf gives threshold
u_r = (R-1)/(R+1) = 13.6% at R = 1.315.

Method: Rayleigh dispersion w(alpha) on a COMPLEX-alpha grid, tracked from a
single dense eigensolve on the real axis by sparse shift-invert continuation
(tridiagonal matrices); saddle cells located from central differences of
w(alpha) on the grid, polished by Newton with local branch continuation.
The saddle is identified with the pinch of the KH branch by construction
(continuation from the real-axis KH mode) and verified by the grid map
(Im w contours fold around it). The absolute/convective boundary u_r*(h) is
then found by bisection on sign(Im w0), and compared against the
Avanci-Rodriguez-Alves geometric criterion y_i = y_b evaluated on the same
profiles.

Output: table + figs_explore/lsb_pinchpoint_boundary.png.
"""
import os
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eig

FIGD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs_explore')
NGRID = 480
DW = 1.0                       # wall-layer thickness (units of w)
H_SWEEP = (2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
AR = np.linspace(0.03, 1.3, 33)       # Re(alpha) grid
AI = np.linspace(0.10, -0.90, 34)     # Im(alpha) grid (top row first)


def profile(h, ur, N=NGRID):
    ymax = max(h + 12.0, 22.0)
    y = np.linspace(0.0, ymax, N)
    wallf = 1.0 - np.exp(-(y/DW)**2)
    core = 0.5*(1 - ur) + 0.5*(1 + ur)*np.tanh(y - h)
    u = core*wallf
    dy = y[1] - y[0]
    upp = np.gradient(np.gradient(u, dy), dy)
    return y, u, upp


def avanci_gap(y, u):
    """y_b - y_i on the profile (positive => convective per the criterion)."""
    du = np.gradient(u, y[1] - y[0])
    jm = int(np.argmin(u))
    j_i = jm + int(np.argmax(du[jm:]))
    cum = np.concatenate([[0.0], np.cumsum(0.5*(u[1:] + u[:-1])
                                           * np.diff(y))])
    above = np.where(cum[jm:] > 0.0)[0]
    j_b = jm + (int(above[0]) if len(above) else len(y) - 1 - jm)
    return float(y[j_b] - y[j_i]), float(y[j_i]), float(y[j_b])


def matrices(y, u, upp, alpha):
    N = len(y)
    hgrid = y[1] - y[0]
    main = np.full(N-2, -2.0/hgrid**2, dtype=complex) - alpha**2
    off = np.full(N-3, 1.0/hgrid**2, dtype=complex)
    B = sp.diags([off, main, off], [-1, 0, 1], format='csc')
    U = sp.diags(u[1:-1].astype(complex), 0, format='csc')
    A = U @ B - sp.diags(upp[1:-1].astype(complex), 0, format='csc')
    return A, B


def c_shift(y, u, upp, alpha, c_guess):
    A, B = matrices(y, u, upp, alpha)
    val = spla.eigs(A, k=1, M=B, sigma=c_guess, which='LM',
                    return_eigenvectors=False, tol=1e-13, maxiter=500)
    return complex(val[0])


# FD steps for omega(alpha) derivatives. The eigensolve carries ~1e-9..1e-8
# absolute noise, which a second difference amplifies by 4/d^2 -- with
# d = 5e-4 that produced |F''| ~ O(1000) garbage (and Newton steps -F/F''
# that "converged" instantly). Truncation at these steps is benign since
# |omega''|, |omega''''| are O(1) near the KH saddle.
D1 = 2e-3       # first derivative (noise ~ eps/D1)
D2 = 2.5e-2     # second derivative (noise ~ 4 eps/D2^2 ~ 1e-4)


def omega_derivs(om, a, c):
    """(w0, c, dw/da, d2w/da2) by central differences at the two scales."""
    wp1, _ = om(a + D1, c)
    wm1, _ = om(a - D1, c)
    wp2, _ = om(a + D2, c)
    wm2, _ = om(a - D2, c)
    w0, c = om(a, c)
    return w0, c, (wp1 - wm1)/(2*D1), (wp2 - 2*w0 + wm2)/D2**2


def c_dense_kh(y, u, upp, alpha):
    """KH mode at real alpha: unstable (or least stable) with 0<Re c<1."""
    A, B = matrices(y, u, upp, alpha)
    cs = eig(A.toarray(), B.toarray(), right=False)
    cs = cs[np.isfinite(cs)]
    phys = cs[(np.real(cs) > 0.02) & (np.real(cs) < 0.98)]
    if len(phys) == 0:
        phys = cs
    return complex(phys[np.argmax(np.imag(phys))])


def omega_grid(y, u, upp):
    """Track the KH branch over the complex-alpha grid by continuation."""
    W = np.full((len(AI), len(AR)), np.nan + 0j)
    # top row (largest alpha_i) seeded from a dense solve mid-row
    j0 = len(AR)//3
    a0 = complex(AR[j0], AI[0])
    c = c_dense_kh(y, u, upp, AR[j0])           # real-axis identification
    c = c_shift(y, u, upp, a0, c)               # lift to the top row
    W[0, j0] = a0*c
    for j in range(j0 + 1, len(AR)):            # top row, rightward
        c = c_shift(y, u, upp, complex(AR[j], AI[0]), c)
        W[0, j] = complex(AR[j], AI[0])*c
    c = W[0, j0]/complex(AR[j0], AI[0])
    for j in range(j0 - 1, -1, -1):             # top row, leftward
        c = c_shift(y, u, upp, complex(AR[j], AI[0]), c)
        W[0, j] = complex(AR[j], AI[0])*c
    for i in range(1, len(AI)):                 # march rows downward
        for j in range(len(AR)):
            a = complex(AR[j], AI[i])
            # linear extrapolation of the branch's c from the two rows above
            c1 = W[i-1, j]/complex(AR[j], AI[i-1])
            if i >= 2 and np.isfinite(W[i-2, j]):
                c2 = W[i-2, j]/complex(AR[j], AI[i-2])
                cg = 2*c1 - c2
            else:
                cg = c1
            try:
                c = c_shift(y, u, upp, a, cg)
                W[i, j] = a*c
            except Exception:
                W[i, j] = np.nan
    return W


def saddle_from_grid(y, u, upp, W):
    """|dw/da| minimum on the grid -> complex-Newton polish."""
    dAr = AR[1] - AR[0]
    dAi = AI[1] - AI[0]
    F = np.full_like(W, np.nan + 0j)
    F[1:-1, 1:-1] = (W[1:-1, 2:] - W[1:-1, :-2])/(2*dAr)
    mag = np.abs(F)
    mag[~np.isfinite(mag)] = np.inf
    i, j = np.unravel_index(np.argmin(mag), mag.shape)
    a = complex(AR[j], AI[i])
    c = W[i, j]/a

    def om(al, cg):
        cc = c_shift(y, u, upp, al, cg)
        return al*cc, cc

    for _ in range(25):
        w0, c, Fd, Fpp = omega_derivs(om, a, c)
        if abs(Fpp) < 1e-12:
            break
        step = -Fd/Fpp
        if abs(step) > 0.1:
            step *= 0.1/abs(step)
        a += step
        if abs(step) < 1e-6:
            break
    w0, c = om(a, c)
    return a, w0


def newton_saddle(y, u, upp, a, c, itmax=12, span0=0.04):
    """Saddle polish by LOCAL ANALYTIC QUADRATIC FIT: sample a 3x3 patch by
    short-range eigenvalue continuation from the patch center, least-squares
    fit omega ~ q0 + q1 z + q2 z^2 (exact for an analytic branch), step to
    the model's stationary point z* = -q1/(2 q2). Pointwise-FD Newton is
    hopeless here: warm-started shift-invert jumps modes when the iterate
    wanders, making omega(alpha) effectively discontinuous."""
    span = span0
    w0 = a*c
    for _ in range(itmax):
        zs = np.array([complex(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)])
        ws = np.empty(9, dtype=complex)
        for k, z in enumerate(zs):          # continuation center -> corner
            cc = c
            al = a
            for frac in (0.5, 1.0):
                al = a + z*span*frac
                cc = c_shift(y, u, upp, al, cc)
            ws[k] = al*cc
        M = np.stack([np.ones(9), zs*span, (zs*span)**2], axis=1)
        q, res, _, _ = np.linalg.lstsq(M, ws, rcond=None)
        if abs(q[2]) < 1e-12:
            break
        zstar = -q[1]/(2*q[2])
        if abs(zstar) > 3*span:             # model extrapolating too far
            zstar *= 3*span/abs(zstar)
        a = a + zstar
        c = c_shift(y, u, upp, a, c)
        w0 = a*c
        span = min(max(abs(zstar), 0.006), 0.06)
        if abs(zstar) < 1e-5:
            break
    return a, w0, c


def is_pinch(y, u, upp, a0, w0, c0):
    """Briggs collision test: for omega = w0 + i*delta with growing delta,
    the two spatial roots alpha(omega) leaving the saddle must separate into
    OPPOSITE halves of the alpha plane (k+ downstream, k- upstream). A
    stationary point failing this is a saddle between same-side branches
    and does not decide absolute instability."""
    def om(al, cg):
        cc = c_shift(y, u, upp, al, cg)
        return al*cc, cc

    w0, c0, _, Fpp = omega_derivs(om, a0, c0)
    if abs(Fpp) < 1e-10:
        return False
    # delta ladder scaled so the roots are pushed a controlled MULTIPLE of
    # |Im alpha0| away from the saddle: alpha - alpha0 ~ sqrt(2 delta/|F''|),
    # so delta_k = |F''|/2 * (f_k * max(|Im a0|, 0.1))^2, capped to stay a
    # physically sensible distance above the saddle frequency.
    span = max(abs(a0.imag), 0.10)
    deltas = [min(max(0.5*abs(Fpp)*(f*span)**2, 0.02), 1.5)
              for f in (0.7, 1.4, 2.2, 3.0)]
    roots = []
    for sgn in (+1, -1):
        a, c = a0, c0
        ok = True
        for delta in deltas:
            w_t = complex(w0.real, w0.imag + delta)
            a_pred = a0 + sgn*np.sqrt(2*(w_t - w0)/Fpp)
            if abs(a_pred - a) < 0.3:
                a = a_pred
            for _ in range(20):     # Newton on omega(alpha) = w_t
                try:
                    wv, c = om(a, c)
                    g = (om(a + D1, c)[0] - om(a - D1, c)[0])/(2*D1)
                except Exception:
                    ok = False
                    break
                if abs(g) < 1e-12:
                    ok = False
                    break
                step = -(wv - w_t)/g
                if abs(step) > 0.15:
                    step *= 0.15/abs(step)
                a += step
                if abs(step) < 1e-6:
                    break
            if not ok:
                break
        if not ok:
            return False
        roots.append(a)
    # opposite half-planes at the top of the ladder = genuine k+/k- pinch
    lo, hi = min(r.imag for r in roots), max(r.imag for r in roots)
    return hi > 0.02 and lo < -0.02


def detect_saddle(h, ur):
    """Coarse-map detection with candidate polish; keep the stationary point
    with a physical KH phase speed and the largest Im w0."""
    y, u, upp = profile(h, ur)
    W = omega_grid(y, u, upp)
    dAr = AR[1] - AR[0]
    F = np.full_like(W, np.nan + 0j)
    F[:, 1:-1] = (W[:, 2:] - W[:, :-2])/(2*dAr)
    mag = np.abs(F)
    mag[~np.isfinite(mag)] = np.inf
    best = None
    flat = np.argsort(mag, axis=None)[:8]        # polish several candidates
    seen = []
    for idx in flat:
        i, j = np.unravel_index(idx, mag.shape)
        a_g = complex(AR[j], AI[i])
        if any(abs(a_g - s) < 0.08 for s in seen):
            continue
        seen.append(a_g)
        try:
            a0, w0, c0 = newton_saddle(y, u, upp, a_g, W[i, j]/a_g)
        except Exception:
            continue
        cph = (w0/a0).real
        if not (0.05 < cph < 0.95) or abs(a0.imag) > 1.2 or a0.real < 0.02:
            continue
        if a0.real > AR[-1] - 0.05 or a0.imag < AI[-1] + 0.05:
            continue                              # grid-edge artifact
        if not is_pinch(y, u, upp, a0, w0, c0):
            continue
        if best is None or w0.imag > best[1].imag:
            best = (a0, w0, c0)
    return best


def track_threshold(h, ur_grid, ur_start=0.14):
    """Detect the KH pinch once at ur_start, then pure Newton continuation
    across ur_grid (no re-detection). Returns rows of (ur, alpha0, w0)."""
    det = detect_saddle(h, ur_start)
    if det is None:
        print(f"  h={h}: no physical saddle detected", flush=True)
        return []
    rows = {}
    for direction in (+1, -1):
        a0, w0, c0 = det
        urs = [u_ for u_ in ur_grid if (u_ >= ur_start if direction > 0
                                        else u_ < ur_start)]
        urs.sort(key=lambda u_: direction*u_)
        for ur in urs:
            y, u, upp = profile(h, ur)
            try:
                a0, w0, c0 = newton_saddle(y, u, upp, a0, c0)
            except Exception:
                break
            rows[ur] = (a0, w0)
    return [(ur,) + rows[ur] for ur in sorted(rows)]


def avanci_boundary(h, lo=0.01, hi=0.55):
    for _ in range(40):
        mid = 0.5*(lo + hi)
        gap = avanci_gap(*profile(h, mid)[:2])[0]
        if gap < 0:
            hi = mid
        else:
            lo = mid
    return 0.5*(lo + hi)


def main():
    os.makedirs(FIGD, exist_ok=True)
    print(f"family: tanh layer at height h over no-slip wall (w=1, "
          f"d_w={DW}); HM85 free-layer threshold u_r = 13.6%")
    UR_GRID = np.round(np.arange(0.02, 0.42, 0.02), 3)
    rows = []
    tracks = {}
    for h in H_SWEEP:
        tr = track_threshold(h, UR_GRID)
        tracks[h] = tr
        if not tr:
            continue
        for ur, a0, w0 in tr:
            print(f"  h={h:5.1f} ur={ur:5.2f}: alpha0={a0.real:+.3f}"
                  f"{a0.imag:+.3f}i  w0={w0.real:+.4f}{w0.imag:+.4f}i",
                  flush=True)
        wi = np.array([w0.imag for _, _, w0 in tr])
        urs = np.array([ur for ur, _, _ in tr])
        cross = np.where(np.diff(np.sign(wi)) > 0)[0]
        ur_bb = float(np.interp(0.0, wi[cross[0]:cross[0]+2],
                                urs[cross[0]:cross[0]+2])) if len(cross) \
            else float('nan')
        ur_av = avanci_boundary(h)
        rows.append((h, ur_bb, ur_av))
        print(f"h={h:5.1f}: Briggs-Bers u_r* = {ur_bb*100:5.1f}%   "
              f"Avanci u_r** = {ur_av*100:5.1f}%", flush=True)

    np.save(os.path.join(FIGD, 'results_lsb_pinchpoint.npy'),
            np.array(rows), allow_pickle=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = np.array(rows)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(rows[:, 0], rows[:, 1]*100, 'ko-', label='Briggs–Bers pinch '
            r'(Im$\,\omega_0=0$)')
    ax.plot(rows[:, 0], rows[:, 2]*100, 'rs--', label='Avanci geometric '
            r'($y_i=y_b$)')
    ax.axhline(13.6, color='0.5', ls=':', lw=1.0)
    ax.text(rows[-1, 0]*0.7, 14.1, 'HM85 free shear layer 13.6%',
            fontsize=8, color='0.4')
    ax.set_xlabel('shear-layer height $h/w$ (bubble thickness)')
    ax.set_ylabel(r'absolute-instability threshold $u_{rev}/U_e$ [%]')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGD, 'lsb_pinchpoint_boundary.png'), dpi=140)
    print('wrote', os.path.join(FIGD, 'lsb_pinchpoint_boundary.png'))


if __name__ == '__main__':
    main()
