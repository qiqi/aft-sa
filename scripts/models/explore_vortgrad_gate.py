"""Exploration: vorticity-gradient band gate Q for the SA-AI amplification rate.

Idea (2026-07-08): on Blasius, the model's local dN/dRe_theta keeps rising
with Re_theta because the onset cliff (Re_Omega = eta^2 f'' sqrt(Re_x), NOT
self-similar) releases ever-smaller eta, where u is slow and the spatial rate
a*omega/u ~ a/eta diverges. Multiply the rate by the nondimensional gate

    Q = (dOmega/dn * d^2)^2 / ((dOmega/dn * d^2)^2 + (Omega*d)^2 + u^2)

which in FS similarity variables is a PURE function of eta,

    Q(eta) = (f''' eta^2)^2 / ((f''' eta^2)^2 + (f'' eta)^2 + f'^2),

with Q ~ eta^6 at a ZPG wall (f'''(0)=0): the near-wall region is suppressed
regardless of how far the cliff descends, so the amplifying band stays
self-similar and the envelope slope should plateau.

Phases:
  A. Q(eta) profiles on FS members (band location/value; free-shear-notch check
     near separation where dOmega/dn = 0 at the inflection).
  B. Blasius march to high Re_theta on a stretched grid: local slope
     s(Re_theta), baseline vs gated (uncalibrated).
  C. Recalibrate g_c (bisection) so the gated model recovers N=14 at
     Re_x = 4e6 on Blasius; check the free-shear anchor still saturates.
  D. FS two-measure sweep (onset-mean + late rate) with the recalibrated gated
     kernel vs Drela-Giles, side by side with the baseline model's numbers.
"""
import sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, '/home/qiqi/flexcompute/sa-ai')
from src.physics.boundary_layer import FalknerSkanWedge

C_NU_AI, SIGMA_SA = 1.0/12.0, 2.0/3.0
A_MAX, S_SLOPE, G_C = 0.15, 5.263, 1.572
FLOOR, PBAR = 100.0, 4.0
OUT = '/tmp/vortgrad'


def drela(H):
    return 0.01*np.sqrt((2.4*H - 3.7 + 2.5*np.tanh(1.5*H - 4.65))**2 + 0.25)


def rate_kernel(ReO, Gam, s=S_SLOPE, gc=G_C, amax=A_MAX):
    safe = np.maximum(ReO, FLOOR + 1e-12)
    barrier = np.log(np.maximum(1.0 - (FLOOR/safe)**PBAR, 1e-20))
    x = s*(Gam - gc) + barrier
    r = amax/(1.0 + np.exp(-np.clip(x, -300, 300)))
    return np.where(ReO > FLOOR, r, 0.0)


def q_gate(dwdn, w, u, d):
    num = (dwdn*d*d)**2
    return num/(num + (w*d)**2 + u*u)


def stretch_grid(ny, y_top, h0):
    """Geometric cell faces: h0*(r^ny-1)/(r-1) = y_top."""
    from scipy.optimize import brentq
    f = lambda r: h0*(r**ny - 1)/(r - 1) - y_top
    r = brentq(f, 1.0 + 1e-9, 1.5)
    faces = h0*(np.power(r, np.arange(ny + 1)) - 1)/(r - 1)
    return faces, r


def march(fs, x_max, nx=600, ny=400, use_q=False, s=S_SLOPE, gc=G_C,
          y_top_fac=8.0, h0_fac=1.0/40.0, seed=1.0):
    """Eq.-11 march on the FS field, stretched y grid, optional Q gate.
    Returns xs, N(x), plus the last station's band diagnostics."""
    eta99 = np.interp(0.99, fs.u, fs.eta)
    Ue_end = fs.inviscid_at(x_max)
    d99_end = eta99*np.sqrt(x_max/Ue_end)
    x_early = 0.02*x_max
    d99_early = eta99*np.sqrt(x_early/fs.inviscid_at(x_early))
    h0 = min(h0_fac*d99_early, y_top_fac*d99_end/(2.0*ny))
    faces, _ = stretch_grid(ny, y_top_fac*d99_end, h0)
    yc = 0.5*(faces[1:] + faces[:-1])
    dyv = faces[1:] - faces[:-1]                     # cell heights
    dyd = np.empty(ny); dyd[0] = yc[0]; dyd[1:] = yc[1:] - yc[:-1]
    dx = x_max/nx
    nu = np.ones(ny)*seed
    N = [0.0]; xs = [0.0]
    kcoef = C_NU_AI/SIGMA_SA
    for i in range(nx):
        x = (i + 0.5)*dx
        _, u, dudy, v = fs.at(x, faces, cellCentered=True)
        u = np.maximum(u, 1e-12)
        w = np.abs(dudy)
        ReO = yc**2*w
        Gam = 2*(w*yc)**2/(u**2 + (w*yc)**2)
        r = rate_kernel(ReO, Gam, s=s, gc=gc)
        if use_q:
            dwdn = np.gradient(dudy, yc)
            r = r*q_gate(np.abs(dwdn), w, u, yc)
        b = r*w
        # tridiagonal: advection (sign-aware upwind, cell-centered v) + FV diffusion
        vp = np.clip(v, 0, None); vm = np.clip(-v, 0, None)
        di = (vp + vm)/dyv
        lo = -vp[1:]/dyv[1:]
        up = -vm[:-1]/dyv[:-1]
        Dfl = kcoef/(dyv*dyd)                        # lower-face conductance /cell
        Dfu = np.empty(ny); Dfu[:-1] = kcoef/(dyv[:-1]*dyd[1:]); Dfu[-1] = 0.0
        di = di + Dfl + Dfu
        lo = lo - Dfl[1:]
        up = up - Dfu[:-1]
        di[0] += Dfl[0]                              # wall ghost nu=-nu0 (Dirichlet 0)
        # top: zero gradient (Dfu[-1]=0 already); freestream seed advected in if v<0
        rhs = u/dx*nu + b*nu
        rhs[-1] += vm[-1]/dyv[-1]*seed
        A = sp.diags([lo, u/dx + di, up], [-1, 0, 1], format='csc')
        nu = spla.spsolve(A, rhs)
        xs.append((i + 1)*dx)
        N.append(float(np.log(max(nu.max()/seed, 1e-300))))
    return np.array(xs), np.array(N)


def profile_ints(fs):
    I_th = np.trapezoid(fs.u*(1 - fs.u), fs.eta)
    return I_th, np.trapezoid(1 - fs.u, fs.eta)/I_th


# ---------------- Phase A: Q(eta) profiles ----------------
def phase_A():
    print('=== Phase A: Q(eta) band shapes ===')
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for beta in [-0.198, -0.1, 0.0, 0.3, 1.0]:
        fs = FalknerSkanWedge(beta)
        f3 = np.gradient(fs.dudeta, fs.eta)
        Q = (f3*fs.eta**2)**2/((f3*fs.eta**2)**2 + (fs.dudeta*fs.eta)**2
                               + fs.u**2 + 1e-300)
        Gam = 2*(fs.dudeta*fs.eta)**2/(fs.u**2 + (fs.dudeta*fs.eta)**2 + 1e-300)
        ax[0].plot(fs.eta, Q, label=f'beta={beta}')
        ax[1].plot(fs.eta, Q*rate_kernel(1e9*np.ones_like(Q), Gam)/A_MAX, '--')
        ipk = np.argmax(fs.eta**2*fs.dudeta)
        print(f'  beta={beta:+.3f}: Q at Re_Omega peak = {Q[ipk]:.3f}, '
              f'max Q = {Q.max():.3f} at eta={fs.eta[np.argmax(Q)]:.2f} '
              f'(Re_Omega peak at eta={fs.eta[ipk]:.2f})')
    ax[0].set_xlabel('eta'); ax[0].set_ylabel('Q'); ax[0].set_xlim(0, 8)
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].set_xlabel('eta'); ax[1].set_ylabel('Q * S(Gamma) (barrier off)')
    ax[1].set_xlim(0, 8); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f'{OUT}_phaseA.png', dpi=130)
    print(f'  wrote {OUT}_phaseA.png')


# ---------------- Phase B: Blasius slope drift ----------------
def slope_curve(fs, x_max, use_q, s=S_SLOPE, gc=G_C, nx=600, ny=400):
    xs, N = march(fs, x_max, nx=nx, ny=ny, use_q=use_q, s=s, gc=gc)
    I_th, H = profile_ints(fs)
    Rt = I_th*np.sqrt(xs*fs.inviscid_at(np.maximum(xs, 1e-12)))
    sl = np.gradient(N, Rt)
    return Rt, N, sl


def phase_B():
    print('=== Phase B: Blasius local slope vs Re_theta (drift test) ===')
    fs = FalknerSkanWedge(0.0)
    out = {}
    for tag, use_q in [('baseline', False), ('gated', True)]:
        Rt, N, sl = slope_curve(fs, 2.0e8, use_q)   # Re_theta up to ~9400
        out[tag] = (Rt, N, sl)
        samp = [500, 1000, 2000, 4000, 8000]
        vals = [float(np.interp(r, Rt, sl)) for r in samp]
        print(f'  {tag:9s}: N_end={N[-1]:7.1f};  s_loc at Re_th ' +
              ', '.join(f'{r}:{v:.4f}' for r, v in zip(samp, vals)))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for tag, sty in [('baseline', 'k-'), ('gated', 'C0-')]:
        Rt, N, sl = out[tag]
        m = Rt > 300
        ax.plot(Rt[m], sl[m], sty, label=tag)
    ax.axhline(drela(2.591), color='0.5', ls='--', label='Drela Blasius rate')
    ax.set_xlabel('Re_theta'); ax.set_ylabel('local dN/dRe_theta')
    ax.set_ylim(0, 0.05); ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(f'{OUT}_phaseB.png', dpi=130)
    print(f'  wrote {OUT}_phaseB.png')


# ---------------- Phase C: recalibrate g_c for the gated model ----------------
def phase_C():
    print('=== Phase C: recalibrate g_c (gated) for N(Re_x=4e6)=14 on Blasius ===')
    fs = FalknerSkanWedge(0.0)
    def n_end(gc):
        _, N = march(fs, 4.0e6, use_q=True, gc=gc)
        return N[-1]
    lo, hi = 0.8, G_C           # lowering g_c raises the attached rate
    nlo, nhi = n_end(lo), n_end(hi)
    print(f'  bracket: g_c={lo} -> N={nlo:.1f};  g_c={hi} -> N={nhi:.1f}')
    for _ in range(20):
        mid = 0.5*(lo + hi)
        nm = n_end(mid)
        if nm > 14.0:
            lo = mid
        else:
            hi = mid
        if abs(nm - 14.0) < 0.05:
            break
    gc_new = mid
    sat = 1.0/(1.0 + np.exp(-S_SLOPE*(2.0 - gc_new)))
    print(f'  g_c* = {gc_new:.3f} (N_end={nm:.2f}); free-shear saturation '
          f'S(s(2-g_c*)) = {sat:.3f} (baseline {1/(1+np.exp(-S_SLOPE*(2-G_C))):.3f})')
    return gc_new


# ---------------- Phase D: FS sweep with the recalibrated gated kernel ----------
def measures(fs, use_q, gc, x0):
    I_th, H = profile_ints(fs)
    ratio_pk = np.max(fs.eta**2*np.abs(fs.dudeta))/I_th
    Rt_c = FLOOR/ratio_pk
    x_max = x0
    for it in range(12):
        xs, N = march(fs, x_max, use_q=use_q, gc=gc)
        if not np.all(np.isfinite(N)) or N[-1] > 60.0:
            x_max *= 0.15; continue
        if N[-1] > 14.0:
            x_max = 1.1*float(np.interp(14.0, N, xs))
            xs, N = march(fs, x_max, use_q=use_q, gc=gc)
            break
        x_max *= 3.0
    Rt = I_th*np.sqrt(xs*fs.inviscid_at(np.maximum(xs, 1e-12)))
    Rt5, Rt9 = (float(np.interp(v, N, Rt)) for v in (5.0, 9.0))
    return H, 9.0/(Rt9 - Rt_c), 4.0/(Rt9 - Rt5)


def phase_D(gc_new):
    print('=== Phase D: FS sweep, gated+recalibrated vs Drela ===')
    print(f"{'beta':>7} {'H':>6} {'mean':>9} {'(x)':>6} {'late':>9} {'(x)':>6}   Drela")
    for beta in [-0.195, -0.15, -0.09, -0.03, 0.0, 0.1, 0.35, 1.0]:
        fs = FalknerSkanWedge(beta)
        x0 = 4e6 if beta == 0.0 else (3e5 if beta > 0 else 1.2e6)
        H, m, l = measures(fs, True, gc_new, x0)
        d = drela(H)
        print(f'{beta:+7.3f} {H:6.3f} {m:9.2e} {m/d:6.2f} {l:9.2e} {l/d:6.2f}   {d:.2e}',
              flush=True)


# ---------------- Phase E: joint (s, g_c) refit with the gate ----------------
def gc_for_blasius(s, tol=0.08):
    fs = FalknerSkanWedge(0.0)
    def n_end(gc):
        _, N = march(fs, 4.0e6, use_q=True, s=s, gc=gc)
        return N[-1]
    lo, hi = 0.3, 2.0
    for _ in range(24):
        mid = 0.5*(lo + hi)
        nm = n_end(mid)
        if nm > 14.0:
            lo = mid
        else:
            hi = mid
        if abs(nm - 14.0) < tol:
            break
    return mid, nm


def phase_E():
    print('=== Phase E: joint (s, g_c) refit with the gate ===')
    print(f"{'s':>6} {'g_c*':>7} | late-rate ratio vs Drela at "
          f"H=2.59, 2.64, 3.02, 3.64 | sat(2)")
    best = None
    for s in [5.263, 8.0, 12.0, 16.0, 20.0]:
        gc, nm = gc_for_blasius(s)
        rats = []
        for beta, x0 in [(0.0, 4e6), (-0.03, 1.2e6), (-0.15, 1.2e6), (-0.195, 1.2e6)]:
            fs = FalknerSkanWedge(beta)
            H, m, l = measures(fs, True, gc, x0)
            rats.append((H, m/drela(H), l/drela(H)))
        sat = 1.0/(1.0 + np.exp(-s*(2.0 - gc)))
        err = sum(np.log(r[2])**2 for r in rats)
        line = ' '.join(f'{r[2]:.2f}' for r in rats)
        print(f'{s:6.2f} {gc:7.3f} | {line} | {sat:.3f}   (logerr {err:.3f})', flush=True)
        if best is None or err < best[0]:
            best = (err, s, gc, rats)
    err, s, gc, rats = best
    print(f'  best: s={s}, g_c={gc:.3f}')
    print('  full two-measure sweep at best (s, g_c):')
    print(f"{'beta':>7} {'H':>6} {'mean(x)':>8} {'late(x)':>8}")
    for beta in [-0.195, -0.15, -0.09, -0.03, 0.0, 0.1, 0.35, 1.0]:
        fs = FalknerSkanWedge(beta)
        x0 = 4e6 if beta == 0.0 else (3e5 if beta > 0 else 1.2e6)
        try:
            H, m, l = measures(fs, True, gc, x0)
            print(f'{beta:+7.3f} {H:6.3f} {m/drela(H):8.2f} {l/drela(H):8.2f}', flush=True)
        except Exception as e:
            print(f'{beta:+7.3f}  FAIL {e}', flush=True)


# ------- Phase F: small-s scan (unsaturated sigmoid) + band diagnostics -------
def band_diag(fs, use_q, s, gc, x0):
    """nuHat-weighted Gamma and Q at the final station of a gated march."""
    eta99 = np.interp(0.99, fs.u, fs.eta)
    x_max = x0
    xs, N = march(fs, x_max, use_q=use_q, s=s, gc=gc)
    # recompute fields at the final station on the same grid as march used
    Ue_end = fs.inviscid_at(x_max)
    d99_end = eta99*np.sqrt(x_max/Ue_end)
    x_early = 0.02*x_max
    d99_early = eta99*np.sqrt(x_early/fs.inviscid_at(x_early))
    h0 = min((1.0/40.0)*d99_early, 8.0*d99_end/(2.0*400))
    faces, _ = stretch_grid(400, 8.0*d99_end, h0)
    # rerun the march capturing the final nu profile
    yc = 0.5*(faces[1:] + faces[:-1])
    return None  # diagnostics folded into phase_F via march2


def march2(fs, x_max, use_q, s, gc, nx=400, ny=300):
    """march() clone that also returns the final nu profile and fields."""
    eta99 = np.interp(0.99, fs.u, fs.eta)
    Ue_end = fs.inviscid_at(x_max)
    d99_end = eta99*np.sqrt(x_max/Ue_end)
    x_early = 0.02*x_max
    d99_early = eta99*np.sqrt(x_early/fs.inviscid_at(x_early))
    h0 = min(d99_early/40.0, 8.0*d99_end/(2.0*ny))
    faces, _ = stretch_grid(ny, 8.0*d99_end, h0)
    yc = 0.5*(faces[1:] + faces[:-1])
    dyv = faces[1:] - faces[:-1]
    dyd = np.empty(ny); dyd[0] = yc[0]; dyd[1:] = yc[1:] - yc[:-1]
    dx = x_max/nx
    nu = np.ones(ny)
    N = [0.0]; xs = [0.0]
    kcoef = C_NU_AI/SIGMA_SA
    for i in range(nx):
        x = (i + 0.5)*dx
        _, u, dudy, v = fs.at(x, faces, cellCentered=True)
        u = np.maximum(u, 1e-12)
        w = np.abs(dudy)
        Gam = 2*(w*yc)**2/(u**2 + (w*yc)**2)
        r = rate_kernel(yc**2*w, Gam, s=s, gc=gc)
        Q = q_gate(np.abs(np.gradient(dudy, yc)), w, u, yc) if use_q else np.ones(ny)
        b = r*Q*w
        vp = np.clip(v, 0, None); vm = np.clip(-v, 0, None)
        di = (vp + vm)/dyv
        lo = -vp[1:]/dyv[1:]; up = -vm[:-1]/dyv[:-1]
        Dfl = kcoef/(dyv*dyd)
        Dfu = np.empty(ny); Dfu[:-1] = kcoef/(dyv[:-1]*dyd[1:]); Dfu[-1] = 0.0
        di = di + Dfl + Dfu; lo = lo - Dfl[1:]; up = up - Dfu[:-1]
        di[0] += Dfl[0]
        rhs = u/dx*nu + b*nu
        rhs[-1] += vm[-1]/dyv[-1]
        A = sp.diags([lo, u/dx + di, up], [-1, 0, 1], format='csc')
        nu = spla.spsolve(A, rhs)
        xs.append((i + 1)*dx)
        N.append(float(np.log(max(nu.max(), 1e-300))))
    wgt = b*nu*dyv          # local growth contribution at the final station
    G_eff = float(np.sum(Gam*wgt)/max(np.sum(wgt), 1e-300))
    Q_eff = float(np.sum(Q*wgt)/max(np.sum(wgt), 1e-300))
    return np.array(xs), np.array(N), G_eff, Q_eff


def gc_for_blasius_s(s, tol=0.12):
    fs = FalknerSkanWedge(0.0)
    lo, hi = 0.2, 2.4
    nm = None
    for _ in range(22):
        mid = 0.5*(lo + hi)
        _, N, _, _ = march2(fs, 4.0e6, True, s, mid)
        nm = N[-1]
        if nm > 14.0: lo = mid
        else: hi = mid
        if abs(nm - 14.0) < tol: break
    return mid, nm


def measures2(fs, s, gc, x0):
    I_th, H = profile_ints(fs)
    ratio_pk = np.max(fs.eta**2*np.abs(fs.dudeta))/I_th
    Rt_c = FLOOR/ratio_pk
    x_max = x0
    for it in range(12):
        xs, N, Ge, Qe = march2(fs, x_max, True, s, gc)
        if not np.all(np.isfinite(N)) or N[-1] > 60.0:
            x_max *= 0.15; continue
        if N[-1] > 14.0:
            x_max = 1.1*float(np.interp(14.0, N, xs))
            xs, N, Ge, Qe = march2(fs, x_max, True, s, gc)
            break
        x_max *= 3.0
    Rt = I_th*np.sqrt(xs*fs.inviscid_at(np.maximum(xs, 1e-12)))
    Rt5, Rt9 = (float(np.interp(v, N, Rt)) for v in (5.0, 9.0))
    return H, 9.0/(Rt9 - Rt_c), 4.0/(Rt9 - Rt5), Ge, Qe


def phase_F():
    print('=== Phase F: small-s scan (unsaturated regime) ===')
    ADV = [(0.0, 4e6), (-0.03, 1.2e6), (-0.09, 1.2e6), (-0.15, 1.2e6),
           (-0.18, 1.2e6), (-0.195, 1.2e6)]
    best = None
    for s in [1.5, 2.0, 2.5, 3.0, 4.0, 5.26, 8.0]:
        gc, nm = gc_for_blasius_s(s)
        rows = []
        for beta, x0 in ADV:
            fs = FalknerSkanWedge(beta)
            H, m, l, Ge, Qe = measures2(fs, s, gc, x0)
            rows.append((H, l/drela(H), m/drela(H), Ge, Qe))
        sat = 1.0/(1.0 + np.exp(-s*(2.0 - gc)))
        err = sum(np.log(r[1])**2 for r in rows)
        print(f's={s:5.2f} g_c={gc:6.3f} sat={sat:.3f} | late-x: ' +
              ' '.join(f'{r[1]:.2f}' for r in rows) +
              f' | logerr={err:.3f}', flush=True)
        if best is None or err < best[0]:
            best = (err, s, gc, rows)
    err, s, gc, rows = best
    print(f'\n  best: s={s}, g_c={gc:.3f} (late-branch logerr {err:.3f})')
    print(f"{'H':>6} {'late(x)':>8} {'mean(x)':>8} {'Gam_eff':>8} {'Q_eff':>7}")
    for H, lr, mr, Ge, Qe in rows:
        print(f'{H:6.3f} {lr:8.2f} {mr:8.2f} {Ge:8.3f} {Qe:7.3f}')
    print('  favorable branch at best (s, g_c):')
    for beta in [0.1, 0.35, 1.0]:
        fs = FalknerSkanWedge(beta)
        try:
            H, m, l, Ge, Qe = measures2(fs, s, gc, 3e5)
            print(f'  beta={beta:+.2f} H={H:.3f} mean={m/drela(H):5.2f}x '
                  f'late={l/drela(H):5.2f}x Gam_eff={Ge:.3f} Q_eff={Qe:.3f}', flush=True)
        except Exception as e:
            print(f'  beta={beta:+.2f} FAIL {e}')


# ------- Phase G: exponential-in-Gamma rate, capped at a_max, with the gate ----
def rate_exp(ReO, Gam, a0, k, amax=A_MAX):
    r = np.minimum(amax, a0*np.exp(np.clip(k*Gam, -300, 60)))
    barrier = np.where(ReO > FLOOR,
                       np.maximum(1.0 - (FLOOR/np.maximum(ReO, FLOOR + 1e-12))**PBAR,
                                  0.0), 0.0)
    return r*barrier


def march3(fs, x_max, a0, k, nx=400, ny=300):
    """march with the exponential-capped rate * Q gate."""
    eta99 = np.interp(0.99, fs.u, fs.eta)
    Ue_end = fs.inviscid_at(x_max)
    d99_end = eta99*np.sqrt(x_max/Ue_end)
    x_early = 0.02*x_max
    d99_early = eta99*np.sqrt(x_early/fs.inviscid_at(x_early))
    h0 = min(d99_early/40.0, 8.0*d99_end/(2.0*ny))
    faces, _ = stretch_grid(ny, 8.0*d99_end, h0)
    yc = 0.5*(faces[1:] + faces[:-1])
    dyv = faces[1:] - faces[:-1]
    dyd = np.empty(ny); dyd[0] = yc[0]; dyd[1:] = yc[1:] - yc[:-1]
    dx = x_max/nx
    nu = np.ones(ny)
    N = [0.0]; xs = [0.0]
    kcoef = C_NU_AI/SIGMA_SA
    for i in range(nx):
        x = (i + 0.5)*dx
        _, u, dudy, v = fs.at(x, faces, cellCentered=True)
        u = np.maximum(u, 1e-12)
        w = np.abs(dudy)
        Gam = 2*(w*yc)**2/(u**2 + (w*yc)**2)
        r = rate_exp(yc**2*w, Gam, a0, k)
        Q = q_gate(np.abs(np.gradient(dudy, yc)), w, u, yc)
        b = r*Q*w
        vp = np.clip(v, 0, None); vm = np.clip(-v, 0, None)
        di = (vp + vm)/dyv
        lo = -vp[1:]/dyv[1:]; up = -vm[:-1]/dyv[:-1]
        Dfl = kcoef/(dyv*dyd)
        Dfu = np.empty(ny); Dfu[:-1] = kcoef/(dyv[:-1]*dyd[1:]); Dfu[-1] = 0.0
        di = di + Dfl + Dfu; lo = lo - Dfl[1:]; up = up - Dfu[:-1]
        di[0] += Dfl[0]
        rhs = u/dx*nu + b*nu
        rhs[-1] += vm[-1]/dyv[-1]
        A = sp.diags([lo, u/dx + di, up], [-1, 0, 1], format='csc')
        nu = spla.spsolve(A, rhs)
        xs.append((i + 1)*dx)
        N.append(float(np.log(max(nu.max(), 1e-300))))
    return np.array(xs), np.array(N)


def a0_for_blasius(k, tol=0.12):
    fs = FalknerSkanWedge(0.0)
    lo, hi = 1e-6, A_MAX
    nm = None
    for _ in range(28):
        mid = np.sqrt(lo*hi)
        _, N = march3(fs, 4.0e6, mid, k)
        nm = N[-1]
        if nm > 14.0: hi = mid
        else: lo = mid
        if abs(nm - 14.0) < tol: break
    return mid, nm


def measures3(fs, a0, k, x0):
    I_th, H = profile_ints(fs)
    ratio_pk = np.max(fs.eta**2*np.abs(fs.dudeta))/I_th
    Rt_c = FLOOR/ratio_pk
    x_max = x0
    for it in range(12):
        xs, N = march3(fs, x_max, a0, k)
        if not np.all(np.isfinite(N)) or N[-1] > 60.0:
            x_max *= 0.15; continue
        if N[-1] > 14.0:
            x_max = 1.1*float(np.interp(14.0, N, xs))
            xs, N = march3(fs, x_max, a0, k)
            break
        x_max *= 3.0
    Rt = I_th*np.sqrt(xs*fs.inviscid_at(np.maximum(xs, 1e-12)))
    Rt5, Rt9 = (float(np.interp(v, N, Rt)) for v in (5.0, 9.0))
    return H, 9.0/(Rt9 - Rt_c), 4.0/(Rt9 - Rt5)


def phase_G():
    print('=== Phase G: capped exponential rate a=min(a_max, a0 e^{k Gamma}) * Q ===')
    ADV = [(0.0, 4e6), (-0.03, 1.2e6), (-0.09, 1.2e6), (-0.15, 1.2e6),
           (-0.18, 1.2e6), (-0.195, 1.2e6)]
    best = None
    for k in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        a0, nm = a0_for_blasius(k)
        rows = []
        for beta, x0 in ADV:
            fs = FalknerSkanWedge(beta)
            H, m, l = measures3(fs, a0, k, x0)
            rows.append((H, l/drela(H), m/drela(H)))
        err = sum(np.log(r[1])**2 for r in rows)
        cap_gamma = np.log(A_MAX/a0)/k     # Gamma at which the cap engages
        print(f'k={k:4.1f} a0={a0:.2e} cap@Gam={cap_gamma:5.2f} | late-x: ' +
              ' '.join(f'{r[1]:.2f}' for r in rows) + f' | logerr={err:.3f}',
              flush=True)
        if best is None or err < best[0]:
            best = (err, k, a0, rows)
    err, k, a0, rows = best
    print(f'\n  best: k={k}, a0={a0:.3e}')
    print(f"{'H':>6} {'late(x)':>8} {'mean(x)':>8}")
    for H, lr, mr in rows:
        print(f'{H:6.3f} {lr:8.2f} {mr:8.2f}')
    print('  favorable branch at best (k, a0):')
    for beta in [0.1, 0.35, 1.0]:
        fs = FalknerSkanWedge(beta)
        try:
            H, m, l = measures3(fs, a0, k, 3e5)
            print(f'  beta={beta:+.2f} H={H:.3f} mean={m/drela(H):5.2f}x '
                  f'late={l/drela(H):5.2f}x', flush=True)
        except Exception as e:
            print(f'  beta={beta:+.2f} FAIL {e}')
    print('  Blasius drift check at best (k, a0): local slope at high Re_theta')
    fs = FalknerSkanWedge(0.0)
    xs, N = march3(fs, 2.0e8, a0, k, nx=600, ny=400)
    I_th, _ = profile_ints(fs)
    Rt = I_th*np.sqrt(xs*np.maximum(xs, 1e-12)**0)   # Ue=1 for Blasius
    Rt = I_th*np.sqrt(xs)
    sl = np.gradient(N, Rt)
    for r in [500, 1000, 2000, 4000, 8000]:
        print(f'    Re_theta={r}: s_loc={float(np.interp(r, Rt, sl)):.4f}')


# ------- Phase H: gate-variant comparison Q1 vs Q2 -------------------------
def q_gate2(dwdn, w, u, d):
    """Variant: ((dw/dn d^2)^2 + (w d)^2) / ((dw/dn d^2)^2 + (w d)^2 + u^2).
    Near an attached wall (w d ~ u ~ tau_w y): -> 1/2, NOT 0; in free shear
    (d large): -> 1 even at the inflection (no notch)."""
    num = (dwdn*d*d)**2 + (w*d)**2
    return num/(num + u*u)


def march4(fs, x_max, variant, s, gc, nx=400, ny=300):
    eta99 = np.interp(0.99, fs.u, fs.eta)
    Ue_end = fs.inviscid_at(x_max)
    d99_end = eta99*np.sqrt(x_max/Ue_end)
    x_early = 0.02*x_max
    d99_early = eta99*np.sqrt(x_early/fs.inviscid_at(x_early))
    h0 = min(d99_early/40.0, 8.0*d99_end/(2.0*ny))
    faces, _ = stretch_grid(ny, 8.0*d99_end, h0)
    yc = 0.5*(faces[1:] + faces[:-1])
    dyv = faces[1:] - faces[:-1]
    dyd = np.empty(ny); dyd[0] = yc[0]; dyd[1:] = yc[1:] - yc[:-1]
    dx = x_max/nx
    nu = np.ones(ny)
    N = [0.0]; xs = [0.0]
    kcoef = C_NU_AI/SIGMA_SA
    for i in range(nx):
        x = (i + 0.5)*dx
        _, u, dudy, v = fs.at(x, faces, cellCentered=True)
        u = np.maximum(u, 1e-12)
        w = np.abs(dudy)
        Gam = 2*(w*yc)**2/(u**2 + (w*yc)**2)
        r = rate_kernel(yc**2*w, Gam, s=s, gc=gc)
        dwdn = np.abs(np.gradient(dudy, yc))
        Q = q_gate(dwdn, w, u, yc) if variant == 1 else q_gate2(dwdn, w, u, yc)
        b = r*Q*w
        vp = np.clip(v, 0, None); vm = np.clip(-v, 0, None)
        di = (vp + vm)/dyv
        lo = -vp[1:]/dyv[1:]; up = -vm[:-1]/dyv[:-1]
        Dfl = kcoef/(dyv*dyd)
        Dfu = np.empty(ny); Dfu[:-1] = kcoef/(dyv[:-1]*dyd[1:]); Dfu[-1] = 0.0
        di = di + Dfl + Dfu; lo = lo - Dfl[1:]; up = up - Dfu[:-1]
        di[0] += Dfl[0]
        rhs = u/dx*nu + b*nu
        rhs[-1] += vm[-1]/dyv[-1]
        A = sp.diags([lo, u/dx + di, up], [-1, 0, 1], format='csc')
        nu = spla.spsolve(A, rhs)
        xs.append((i + 1)*dx)
        N.append(float(np.log(max(nu.max(), 1e-300))))
    return np.array(xs), np.array(N)


def phase_H():
    print('=== Phase H: Q1 (A/(A+B+u2)) vs Q2 ((A+B)/(A+B+u2)) ===')
    fs0 = FalknerSkanWedge(0.0)
    for variant in (1, 2):
        # recalibrate g_c at s=8 for this variant (Blasius N(4e6)=14)
        lo, hi = 0.2, 2.4
        for _ in range(20):
            mid = 0.5*(lo + hi)
            _, N = march4(fs0, 4.0e6, variant, 8.0, mid)
            if N[-1] > 14.0: lo = mid
            else: hi = mid
            if abs(N[-1] - 14.0) < 0.15: break
        gc = mid
        # drift
        xs, N = march4(fs0, 2.0e8, variant, 8.0, gc, nx=600, ny=400)
        I_th, _ = profile_ints(fs0)
        Rt = I_th*np.sqrt(xs)
        sl = np.gradient(N, Rt)
        drift = [float(np.interp(r, Rt, sl)) for r in [500, 1000, 2000, 4000, 8000]]
        print(f'  Q{variant}: g_c*={gc:.3f}; drift s_loc(500..8000) = '
              + ', '.join(f'{v:.4f}' for v in drift)
              + f'  (ratio {drift[-1]/max(drift[0],1e-9):.1f}x)', flush=True)
        # mini-sweep
        for beta, x0 in [(0.0, 4e6), (-0.09, 1.2e6), (-0.15, 1.2e6), (-0.195, 1.2e6)]:
            fs = FalknerSkanWedge(beta)
            I_th, H = profile_ints(fs)
            ratio_pk = np.max(fs.eta**2*np.abs(fs.dudeta))/I_th
            Rt_c = FLOOR/ratio_pk
            x_max = x0
            for it in range(12):
                xs, N = march4(fs, x_max, variant, 8.0, gc)
                if not np.all(np.isfinite(N)) or N[-1] > 60.0:
                    x_max *= 0.15; continue
                if N[-1] > 14.0:
                    x_max = 1.1*float(np.interp(14.0, N, xs))
                    xs, N = march4(fs, x_max, variant, 8.0, gc)
                    break
                x_max *= 3.0
            Rt = I_th*np.sqrt(xs*fs.inviscid_at(np.maximum(xs, 1e-12)))
            Rt5, Rt9 = (float(np.interp(v, N, Rt)) for v in (5.0, 9.0))
            m, l = 9.0/(Rt9 - Rt_c), 4.0/(Rt9 - Rt5)
            d = drela(H)
            print(f'    beta={beta:+.3f} H={H:.3f} mean={m/d:5.2f}x late={l/d:5.2f}x',
                  flush=True)


# ------- Phase I: full FS two-measure curve, gated (Q1, recalibrated) vs baseline
def measures_cfg(fs, use_q, s, gc, x0):
    I_th, H = profile_ints(fs)
    ratio_pk = np.max(fs.eta**2*np.abs(fs.dudeta))/I_th
    Rt_c = FLOOR/ratio_pk
    x_max = x0
    for it in range(12):
        xs, N, _, _ = march2(fs, x_max, use_q, s, gc)
        if not np.all(np.isfinite(N)) or N[-1] > 60.0:
            x_max *= 0.15; continue
        if N[-1] > 14.0:
            x_max = 1.1*float(np.interp(14.0, N, xs))
            xs, N, _, _ = march2(fs, x_max, use_q, s, gc)
            break
        x_max *= 3.0
    Rt = I_th*np.sqrt(xs*fs.inviscid_at(np.maximum(xs, 1e-12)))
    Rt5, Rt9 = (float(np.interp(v, N, Rt)) for v in (5.0, 9.0))
    return H, 9.0/(Rt9 - Rt_c), 4.0/(Rt9 - Rt5)


GC_GATED = 0.862  # onset-anchored on coupled flat plate (transport N=14 anchor gives 0.795)
def phase_I():
    print(f'=== Phase I: FS two-measure curves, gated(Q1,s=8,gc={GC_GATED}) vs baseline ===')
    adverse = [-0.195, -0.18, -0.15, -0.12, -0.09, -0.06, -0.03, 0.0]
    favorable = [0.05, 0.1, 0.2, 0.35, 0.55, 1.0]
    def sweep(use_q, s, gc, tag):
        A, F = [], []
        for beta in adverse + favorable:
            fs = FalknerSkanWedge(beta)
            x0 = 4e6 if beta == 0.0 else (3e5 if beta > 0 else 1.2e6)
            try:
                H, m, l = measures_cfg(fs, use_q, s, gc, x0)
            except Exception as e:
                print(f'  {tag} beta={beta}: FAIL {e}'); continue
            (A if beta <= 0 else F).append((H, m, l))
            print(f'  {tag} beta={beta:+.3f} H={H:.3f} mean={m:.3e} ({m/drela(H):.2f}x) '
                  f'late={l:.3e} ({l/drela(H):.2f}x)', flush=True)
        return A, F
    gA, gF = sweep(True, 8.0, GC_GATED, 'gated')
    bA, bF = sweep(False, S_SLOPE, G_C, 'base ')
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    Hg = np.linspace(2.2, 3.66, 300)
    ax.semilogy(Hg, drela(Hg), 'k--', lw=1.9, label='Drela--Giles $dn/dRe_\\theta(H)$')
    def band(data, color, label, open_sym=False):
        data = sorted(data)
        H = np.array([d[0] for d in data]); m = np.array([d[1] for d in data])
        l = np.array([d[2] for d in data])
        mfc = 'white' if open_sym else None
        ls = ':' if open_sym else '-'
        if not open_sym:
            ax.fill_between(H, m, l, color=color, alpha=0.16, lw=0)
        ax.semilogy(H, l, ls + 'o', color=color, lw=1.6, ms=4, mfc=mfc, label=label)
        ax.semilogy(H, m, ls + 's', color=color, lw=1.6, ms=4, mfc=mfc)
    band(bA, '0.45', 'current model (mean, late)')
    band(gA, 'C0', 'gated Q1, recalibrated (mean, late)')
    band(bF, '0.45', None, open_sym=True)
    band(gF, 'C0', None, open_sym=True)
    ax.set_xlabel('$H$'); ax.set_ylabel('$dN/dRe_\\theta$ (absolute)')
    ax.set_ylim(3e-3, 2e-1); ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8, loc='upper left')
    plt.tight_layout(); plt.savefig('/tmp/vortgrad_phaseI.png', dpi=140)
    print('  wrote /tmp/vortgrad_phaseI.png')


if __name__ == '__main__':
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == 'E':
        phase_E()
    elif len(_sys.argv) > 1 and _sys.argv[1] == 'F':
        phase_F()
    elif len(_sys.argv) > 1 and _sys.argv[1] == 'G':
        phase_G()
    elif len(_sys.argv) > 1 and _sys.argv[1] == 'H':
        phase_H()
    elif len(_sys.argv) > 1 and _sys.argv[1] == 'I':
        phase_I()
    else:
        phase_A()
        phase_B()
        gc_new = phase_C()
        phase_D(gc_new)


