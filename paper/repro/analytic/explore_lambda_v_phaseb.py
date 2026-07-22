"""explore-lambda-v, step B: given a candidate (cV, floor, g_c, s) from the
anchor scan (explore_lambda_v_anchors.py), re-derive the two pressure-gradient
constants and verify the full Falkner-Skan family with the Lambda_v gate:

  1. K_lambda: most-amplified-point self-consistency fixed point
     (fig05_06_klambda.py protocol, Q4 -> Q_v with analytic f''').
  2. K_r: one-point fit at beta=0.35 (fit_fpg_rate_slope.py protocol,
     R2 form), with the new K_lambda feeding the cliff.
  3. Family: attached branch (incl. separation limit), favorable branch
     pre/post K_r, Stewartson lower branch (H>4, ufrac=0.03) -- mean-rate
     ratios vs Drela, side by side with the canonical Q4 kernel.

Exploratory output only (figs_explore/, no paper figs).

Usage: explore_lambda_v_phaseb.py CV FLOOR GC S  (defaults: cV=1 candidate)
"""
import os
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
# JAX (imported via lib.aft_sources) is multithreaded: fork-after-JAX
# deadlocks Pool workers, so spawn fresh interpreters instead.
import multiprocessing as mp
_MP = mp.get_context('spawn')

sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/analytic")
import _saai  # noqa: F401
from _saai import (SIGMA_SA, C_NU_AI, S_SLOPE, G_C, FLOOR as FLOOR_Q4,
                   K_LAMBDA, K_R)
from lib.boundary_layer import FalknerSkanWedge
from lib.correlations import dN_dRe_theta, Re_theta0
from lib.aft_sources import compute_aft_amplification_rate, compute_q4_gate
from explore_lambda_v_anchors import (gate_lambda_v, gate_composite,
                                      gate_composite_band)

PBAR = 4.0
FIGD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs_explore')


# ---------------------------------------------------------------- K_lambda
def fs_bvp(beta, n=1400):
    from scipy.integrate import solve_bvp
    m = beta/(2 - beta)

    def ode(x, y):
        return np.vstack([y[1], y[2], -0.5*(m + 1)*y[0]*y[2] - m*(1 - y[1]**2)])

    def bc(a, b):
        return np.array([a[0], a[1], b[1] - 1])
    x = np.linspace(0, 10, n)
    y0 = np.vstack([x - 1 + np.exp(-x), 1 - np.exp(-x), np.exp(-x)])
    s = solve_bvp(ode, bc, x, y0, max_nodes=500000, tol=1e-9)
    Y = s.sol(x)
    return x, Y[0], Y[1], Y[2]


def Sz(r, G, gc, s):
    r = np.asarray(r, float)
    bar = np.where(r > 1, np.maximum(1 - (1/np.maximum(r, 1 + 1e-9))**PBAR,
                                     1e-300), 1e-300)
    return np.where(r > 1, 1/(1 + np.exp(-(s*(G - gc) + np.log(bar)))), 0.0)


def prof(beta, cV, gate='lv', band_pow=0.5):
    eta, f, fp, fpp = fs_bvp(beta)
    m = beta/(2 - beta)
    fppp = -0.5*(m + 1)*f*fpp - m*(1 - fp**2)
    th = np.trapezoid(fp*(1 - fp), eta)
    H = np.trapezoid(1 - fp, eta)/th
    Rc = float(Re_theta0(np.clip(H, 1.05, None)))
    sqRex = Rc/th
    ReO = eta**2*fpp*sqRex
    den_core = (fpp*eta)**2 + fp**2 + 1e-30
    G = 2*(eta*fpp)**2/den_core
    lam = m*eta**2/np.maximum(fp, 1e-6)
    if gate == 'lv3':
        Q = np.asarray(gate_composite_band(fpp, fppp, fp, eta, cV))
    elif gate in ('lv2', 'lv2b'):
        # composite two-pinch gate on similarity variables (analytic f''')
        Q = np.asarray(gate_composite(fpp, fppp, fp, eta, cV,
                                      band_pow=band_pow))
    else:
        # Q_v on similarity variables: u'u''d^3 = U^2 f''f''' eta^3, so
        # Q_v = 1 - 2|f''eta||f'| / (cV|f''f'''|eta^3 + (f''eta)^2 + f'^2).
        den = cV*np.abs(fpp*fppp)*eta**3 + den_core
        Q = 1.0 - 2.0*np.abs(fpp*eta)*np.abs(fp)/den
    return dict(ReO=ReO, G=G, lam=lam, Q=Q, H=H, Rc=Rc)


def worst_lam(pr, KL, floor, gc, s):
    S = pr['Q']*Sz(pr['ReO']/(floor*np.exp(KL*np.maximum(pr['lam'], 0))),
                   pr['G'], gc, s)
    if S.max() <= 0:
        r = pr['ReO']/(floor*np.exp(KL*np.maximum(pr['lam'], 0)))
        return int(np.argmax(r))
    return int(np.argmax(S))


def solve_klambda(cV, floor, gc, s, gate='lv', band_pow=0.5):
    bs = np.array([0.0, 0.02, 0.05, 0.10])
    prs = [prof(b, cV, gate=gate, band_pow=band_pow) for b in bs]
    dlnRc_db = np.polyfit(bs, np.log([p['Rc'] for p in prs]), 1)[0]
    Ku = np.linspace(2.0, 14.0, 61)
    Ki = []
    for K in Ku:
        lw = np.array([prs[k]['lam'][worst_lam(prs[k], K, floor, gc, s)]
                       for k in range(len(bs))])
        Ki.append(dlnRc_db/np.polyfit(bs, lw, 1)[0])
    Ki = np.array(Ki)
    d = Ki - Ku
    sgn = np.where(np.diff(np.sign(d)))[0]
    if len(sgn) == 0:
        print(f"  [K_lambda] NO fixed point in [2,14]; implied range "
              f"{Ki.min():.2f}..{Ki.max():.2f}", flush=True)
        return None, dlnRc_db
    i = sgn[0]
    Kfix = Ku[i] + (Ku[i+1] - Ku[i])*(-d[i])/(d[i+1] - d[i])
    return float(Kfix), float(dlnRc_db)


# ------------------------------------------------------------ march + family
def march(fs, x_max, cnst, beta=None, k_r=0.0, ufrac=0.0, nx=800, ny=600,
          seed=1.0):
    """cnst = dict(gate 'q4'|'lv', gw, floor, gc, s, KL)."""
    m = None if beta is None else beta/(2.0 - beta)
    eta99 = np.interp(0.99, np.maximum.accumulate(fs.u), fs.eta)
    y_top = 8.0*eta99*np.sqrt(x_max/fs.inviscid_at(x_max))
    dy = y_top/ny
    yc = (np.arange(ny) + 0.5)*dy
    dx = x_max/nx
    nu = np.ones(ny)*seed
    N = [0.0]
    xs = [0.0]
    k = (C_NU_AI/SIGMA_SA)/dy**2
    for i in range(nx):
        x = (i + 0.5)*dx
        _, u, dudy, v = fs.at(x, np.arange(ny + 1)*dy, cellCentered=True)
        u = np.maximum(u, max(1e-12, ufrac*fs.inviscid_at(x)))
        vp = np.clip(v, 0, None)/dy
        vm = np.clip(-v, 0, None)/dy
        di = vp + vm + 2*k
        lo = -(vp[1:] + k)
        up = -(vm[:-1] + k)
        di[0] += k
        di[-1] -= k
        lam = 0.0 if m is None else m*yc**2*fs.inviscid_at(x)**2/(x*u)
        rate = np.asarray(compute_aft_amplification_rate(
            yc**2*np.abs(dudy), 2*(dudy*yc)**2/(u**2 + (dudy*yc)**2),
            lambda_p=lam, sigmoid_center=cnst['gc'], sigmoid_slope=cnst['s'],
            re_omega_floor=cnst['floor'], barrier_power=PBAR,
            cliff_lambda_slope=cnst['KL'], fpg_rate_slope=k_r))
        upp = np.gradient(dudy, yc)
        if cnst['gate'] == 'q4':
            q = compute_q4_gate(upp, np.abs(dudy), u, yc, cA=cnst['gw'])
        elif cnst['gate'] == 'lv3':
            q = gate_composite_band(dudy, upp, u, yc, cnst['gw'])
        elif cnst['gate'] in ('lv2', 'lv2b'):
            q = gate_composite(dudy, upp, u, yc, cnst['gw'],
                               band_pow=cnst.get('bpow', 0.5))
        else:
            q = gate_lambda_v(dudy, upp, u, yc, cnst['gw'])
        b = rate*q*np.abs(dudy)
        main = u/dx + di
        rhs = u/dx*nu + b*nu
        rhs[-1] += vm[-1]*seed
        A = sp.diags([lo, main, up], [-1, 0, 1], format='csc')
        nu = spla.spsolve(A, rhs)
        xs.append((i + 1)*dx)
        N.append(float(np.log(max(nu.max()/seed, 1e-300))))
    return np.array(xs), np.array(N)


def measures_for_beta(beta, cnst, k_r=0.0, wedge_lambda=True, guess=None,
                      ufrac=0.0, verbose=True):
    fs = FalknerSkanWedge(beta, guess=guess)
    I_th = np.trapezoid(fs.u*(1 - fs.u), fs.eta)
    H = np.trapezoid(1 - fs.u, fs.eta)/I_th
    bb = beta if wedge_lambda else None
    x_max = 4e6 if beta == 0.0 else (3e5 if beta > 0 else 1.2e6)
    for _ in range(12):
        xs, N = march(fs, x_max, cnst, beta=bb, k_r=k_r, ufrac=ufrac)
        if not np.all(np.isfinite(N)) or N[-1] > 60.0:
            x_max *= 0.15
            continue
        if N[-1] > 14.0:
            x_max = 1.1*float(np.interp(14.0, N, xs))
            xs, N = march(fs, x_max, cnst, beta=bb, k_r=k_r, ufrac=ufrac)
            break
        x_max *= 3.0
    Ue = fs.inviscid_at(np.maximum(xs, 1e-12))
    Rt = I_th*np.sqrt(xs*Ue)
    Rt1 = float(np.interp(1.0, N, Rt))
    Rt5 = float(np.interp(5.0, N, Rt))
    Rt9 = float(np.interp(9.0, N, Rt))
    d = float(dN_dRe_theta(H))
    if not np.isfinite(N).all() or N[-1] < 9.0 or Rt9 <= Rt1:
        print(f"  [{cnst['tag']}] beta={beta:+.3f} H={H:6.3f}: march never "
              f"reached N=9 (N_end={N[-1]:.2f}) -- NaN row", flush=True)
        return dict(beta=beta, H=H, mean=float('nan'), late=float('nan'),
                    Rt1=Rt1, Rt1_D=float(Re_theta0(H)) + 1.0/d)
    s_mean = 8.0/(Rt9 - Rt1)
    s_late = 4.0/(Rt9 - Rt5)
    if verbose:
        print(f"  [{cnst['tag']}] beta={beta:+.3f} H={H:6.3f} Rt1={Rt1:6.0f} "
              f"mean={s_mean/d:5.2f}x late={s_late/d:5.2f}x", flush=True)
    return dict(beta=beta, H=H, mean=s_mean/d, late=s_late/d, Rt1=Rt1,
                Rt1_D=float(Re_theta0(H)) + 1.0/d)


def fit_kr(cnst, beta_fit=0.35):
    def mean_at(K):
        return measures_for_beta(beta_fit, cnst, k_r=K, verbose=False)['mean']
    k0, k1 = 3.0, 8.0
    f0, f1 = mean_at(k0) - 1.0, mean_at(k1) - 1.0
    for _ in range(10):
        if abs(f1 - f0) < 1e-9:
            break
        k2 = min(max(k1 - f1*(k1 - k0)/(f1 - f0), 0.01), 40.0)
        k0, f0 = k1, f1
        k1, f1 = k2, mean_at(k2) - 1.0
        if abs(f1) < 0.005:
            break
    return k1, f1


ATT = [-0.1988, -0.15, -0.09, -0.05, 0.0]
FAV = [0.05, 0.10, 0.20, 0.35, 0.55, 1.00]
LOW = [(-0.19, -0.03), (-0.17, -0.06), (-0.15, -0.08), (-0.12, -0.10)]


def one_row(args):
    kind, beta, cnst, k_r, guess, ufrac = args
    o = measures_for_beta(beta, cnst, k_r=k_r, guess=guess, ufrac=ufrac)
    o['kind'] = kind
    return o


def family(cnst, k_r, pool):
    jobs = ([('att', b, cnst, 0.0, None, 0.0) for b in ATT]
            + [('fav0', b, cnst, 0.0, None, 0.0) for b in FAV]
            + [('fav', b, cnst, k_r, None, 0.0) for b in FAV]
            + [('low', b, cnst, 0.0, g, 0.03) for b, g in LOW])
    return pool.map(one_row, jobs)


def main():
    cV, floor, gc, s = (float(a) for a in (sys.argv[1:5] or
                                           [1.0, 177.5, 0.9676, 13.62]))
    gate = sys.argv[7] if len(sys.argv) >= 8 else 'lv'
    bpow = 1.0 if gate == 'lv2b' else 0.5
    os.makedirs(FIGD, exist_ok=True)
    print(f"candidate: gate={gate} cV={cV} floor={floor} gc={gc} s={s}",
          flush=True)

    if len(sys.argv) >= 7 and sys.argv[5] != '-':   # precomputed KL/KR
        KL, KR = float(sys.argv[5]), float(sys.argv[6])
        print(f"K_lambda = {KL} (given), K_r = {KR} (given)", flush=True)
        lv = dict(tag=gate, gate=gate, gw=cV, floor=floor, gc=gc, s=s,
                  KL=KL, bpow=bpow)
    else:
        KL, dln = solve_klambda(cV, floor, gc, s, gate=gate, band_pow=bpow)
        print(f"K_lambda fixed point = {KL}  (d ln Re_theta_c/d beta = "
              f"{dln:.2f}; canonical Q4: {K_LAMBDA})", flush=True)
        if KL is None:
            KL = 0.0
        lv = dict(tag=gate, gate=gate, gw=cV, floor=floor, gc=gc, s=s,
                  KL=KL, bpow=bpow)
        KR, res = fit_kr(lv)
        print(f"K_r one-point fit at beta=0.35: {KR:.3f} (residual {res:+.3f};"
              f" canonical Q4: {K_R})", flush=True)

    q4 = dict(tag='q4', gate='q4', gw=4.0, floor=FLOOR_Q4, gc=G_C, s=S_SLOPE,
              KL=K_LAMBDA)

    with _MP.Pool(12) as pool:
        fam_lv = family(lv, KR, pool)
        fam_q4 = family(q4, K_R, pool)

    np.save(os.path.join(FIGD, 'lambda_v_family.npy'),
            {'lv': fam_lv, 'q4': fam_q4, 'KL': KL, 'KR': KR,
             'cand': (cV, floor, gc, s)}, allow_pickle=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for fam, col, lab in ((fam_q4, 'C0', 'Q4 (canonical)'),
                          (fam_lv, 'C3', f'$\\Lambda_v$ (cV={cV:g})')):
        for kind, ls, mk in (('att', '-', 'o'), ('fav0', ':', '^'),
                             ('fav', '--', 's'), ('low', '-.', 'v')):
            rows = sorted([o for o in fam if o['kind'] == kind],
                          key=lambda o: o['H'])
            ax.plot([o['H'] for o in rows], [o['mean'] for o in rows],
                    ls, marker=mk, ms=4, color=col, lw=1.3,
                    label=f'{lab} {kind}')
    ax.axhline(1.0, color='0.5', lw=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('H')
    ax.set_ylabel('mean rate / Drela')
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGD, 'lambda_v_family.png'), dpi=140)
    print('wrote', os.path.join(FIGD, 'lambda_v_family.png'), flush=True)

    print("\n===== FAMILY SUMMARY (mean rate / Drela) =====")
    print(f"{'kind':>5} {'beta':>7} {'H':>7} | {'Q4':>6} {'Lv':>6}")
    for o4, ol in zip(fam_q4, fam_lv):
        print(f"{o4['kind']:>5} {o4['beta']:+7.3f} {o4['H']:7.3f} | "
              f"{o4['mean']:6.2f} {ol['mean']:6.2f}")


if __name__ == '__main__':
    main()
