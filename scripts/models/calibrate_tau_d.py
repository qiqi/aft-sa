"""Calibrate a separate destruction-gate width tau_D so the SA-AI fully
turbulent log law matches standard SA EXACTLY (zero intercept footprint).

Model (constant-stress wall layer in wall units, same BVP as
paper/regen_wall_layer.py):
    momentum   (1 + nu_t+) du+/dy+ = 1,   nu_t+ = N f_v1(N),  N = chi
    nu~-eqn    0 = sigma_tP * P - sigma_tD * D
                   + [((cnu_ai) + N) N']'/sigma + (cb2/sigma) N'^2
with
    sigma_tP(N) = max[1 - exp(-(N-1)/TAU_P), 0],  TAU_P = 4   (production gate)
    sigma_tD(N) = max[1 - exp(-(N-1)/TAU_D), 0]                (destruction gate)

Current model: TAU_D = TAU_P = 4 -> dB ~ -0.34 (paper Fig. 4 / Table 1).
Rationale: on the log-law solution P/D = cb1/(cw1 kappa^2) ~ 0.25, so gating
P and D by the SAME sigma_t leaves a net positive residual
(1-sigma_t)(1+cb2)kappa^2/sigma (excess nu~ in the buffer, dB<0).  A faster
destruction gate (TAU_D < TAU_P) flips the sign; a root TAU_D* with dB = 0
exists in (0, 4).  This script brackets and bisects it.

Outputs: dB(TAU_D) table + bisection to |dB| < 5e-4, and a verification solve
at TAU_D* printing B_SA, B_SA-AI, dB, and the buffer nu_t footprint.
"""
import numpy as np
from scipy.integrate import solve_bvp, cumulative_trapezoid

# --- constants: match paper Sec. III.E / ModelConstants.h ---
cb1, sig, cb2, kap = 0.1355, 2.0/3.0, 0.622, 0.41
cw1 = cb1/kap**2 + (1 + cb2)/sig
cw2, cw3 = 0.3, 2.0
cv1, cv2, cv3 = 7.1, 0.7, 0.9
CNU, TAU_P = 1.0/12.0, 4.0

fv1 = lambda c: c**3 / (c**3 + cv1**3)
fv2 = lambda c: 1.0 - c / (1.0 + c*fv1(c))

def sigt(N, tau):
    return np.maximum(1.0 - np.exp(-(np.maximum(N - 1.0, 0.0))/tau), 0.0)

def Stilde(N, eta):
    N = np.maximum(N, 1e-9)
    nut = N*fv1(N); Sp = 1.0/(1.0 + nut)          # Omega+ = du+/dy+ = 1/(1+nu_t+)
    Sbar = N*fv2(N)/(kap**2*eta**2)
    lim = Sp + Sp*(cv2**2*Sp + cv3*Sbar)/((cv3 - 2*cv2)*Sp - Sbar)
    return np.maximum(np.where(Sbar >= -cv2*Sp, Sp + Sbar, lim), 1e-8), Sp

def fw(N, eta, St):
    r = np.minimum(N/(St*kap**2*eta**2), 10.0)
    g = r + cw2*(r**6 - r)
    return g*((1 + cw3**6)/(g**6 + cw3**6))**(1.0/6.0)

E0, EMAX = 0.3, 400.0
eta = np.linspace(E0, EMAX, 700)
bc = lambda ya, yb: np.array([ya[0] - kap*E0, yb[0] - kap*EMAX])

def rhs(e, y, lam, tauD):
    N, Np = y; N = np.maximum(N, 1e-9)
    St, Sp = Stilde(N, e)
    P = cb1*St*N
    D = cw1*fw(N, e, St)*(N/e)**2
    sP = 1.0 - lam*(1.0 - sigt(N, TAU_P))          # production gate (tau = 4)
    sD = 1.0 - lam*(1.0 - sigt(N, tauD))           # destruction gate (tau_D)
    dc = (1.0 + N) - lam*(1.0 - CNU)               # reduced molecular diffusion
    return np.vstack([Np, (-sig*(sP*P - sD*D) - (1 + cb2)*Np**2)/dc])

def solve(lam_target, tauD, guess=None, tol=1e-7):
    if guess is None:
        sol = solve_bvp(lambda e, y: rhs(e, y, 0.0, tauD), bc, eta,
                        np.vstack([kap*eta, kap*np.ones_like(eta)]),
                        max_nodes=300000, tol=1e-9)
        for lam in [0.15, 0.3, 0.45, 0.6, 0.75, 0.85, 0.92, 0.97, lam_target]:
            if lam > lam_target:
                break
            sol = solve_bvp(lambda e, y: rhs(e, y, lam, tauD), bc, eta,
                            np.vstack([sol.sol(eta)[0], sol.sol(eta)[1]]),
                            max_nodes=500000, tol=tol)
    else:
        sol = solve_bvp(lambda e, y: rhs(e, y, lam_target, tauD), bc, eta,
                        np.vstack([guess.sol(eta)[0], guess.sol(eta)[1]]),
                        max_nodes=500000, tol=tol)
    assert sol.success, f"BVP failed at tauD={tauD}"
    return sol

yp = np.logspace(np.log10(E0), np.log10(EMAX), 900)

def intercept(sol):
    N = sol.sol(yp)[0]
    nut = N*fv1(N)
    up = cumulative_trapezoid(1.0/(1.0 + nut), yp, initial=0.0) + E0
    m = (yp > 80) & (yp < 250)
    return float(np.mean(up[m] - np.log(yp[m])/kap)), nut

sol0 = solve(0.0, 4.0)                 # standard SA (lam=0; tauD irrelevant)
B0, nut0 = intercept(sol0)
print(f"standard SA:              B = {B0:.4f}")

# --- bracket: dB(tauD) at a few values, continuing from the previous sol ---
sol_cache = solve(1.0, TAU_P)          # current model, tauD = 4
B44, _ = intercept(sol_cache)
print(f"SA-AI tauD=4 (current):   B = {B44:.4f}   dB = {B44-B0:+.4f}")

def dB(tauD, guess):
    s = solve(1.0, tauD, guess=guess)
    b, _ = intercept(s)
    return b - B0, s

scan = [2.0, 1.0, 0.5, 0.25]
g = sol_cache
hist = [(4.0, B44 - B0)]
for t in scan:
    d, g = dB(t, g)
    hist.append((t, d))
    print(f"SA-AI tauD={t:<5g}          B = {B0+d:.4f}   dB = {d:+.4f}")

# --- bisect on the bracketing interval ---
hist.sort(key=lambda x: x[0])
lo = hi = None
for (t1, d1), (t2, d2) in zip(hist, hist[1:]):
    if d1 == 0:
        lo = hi = (t1, d1); break
    if d1*d2 < 0:
        lo, hi = (t1, d1), (t2, d2); break
assert lo is not None, f"no sign change in scan: {hist}"

a, da = lo; b, db_ = hi
for _ in range(40):
    mid = 0.5*(a + b)
    d, g = dB(mid, g)
    print(f"  bisect tauD={mid:.5f}   dB = {d:+.5f}")
    if abs(d) < 5e-4:
        break
    if (d < 0) == (da < 0):
        a, da = mid, d
    else:
        b, db_ = mid, d
tauD_star = mid

# --- verification at tauD* ---
solv = solve(1.0, tauD_star, guess=g, tol=1e-8)
Bv, nutv = intercept(solv)
peak = np.max(np.abs((nutv - nut0)/(nut0 + 1.0)))*100
print(f"\ntau_D* = {tauD_star:.4f}")
print(f"verify: B_SA = {B0:.4f}, B_SA-AI(tau_D*) = {Bv:.4f}, dB = {Bv-B0:+.5f}")
print(f"buffer footprint max |d nu_t+/(nu_t+ + nu+)| = {peak:.2f}%  (was 4.6% at tauD=4)")
