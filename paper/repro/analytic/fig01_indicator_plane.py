"""fig:indicatorplane -> paper/figs/indicator_plane.pdf   (paper v2)

(a) Falkner-Skan profiles in the local-indicator plane (Re_Omega/Re_theta,
    Gamma) -- unchanged from v1;
(b) the same profiles in the (Lambda_v, Gamma) plane, with the UNIVERSAL
    THIN-BUBBLE PARABOLA overlaid: any quadratic profile with a wall zero,
    u = y(b + a y), traces one parameter-free curve whose two stagnation
    points are the gate's two pinches -- the wall zero at (1, 0) and the
    recirculation zero crossing at (2, -2);
(c) contour lines of the composite two-pinch gate Q = Q1 Q2 of eq:qgate,
    with the Q2 pocket boundary (P = 0, Gamma > 1) drawn bold.

Constants from the canonical set (_saai <- lib.aft_sources)."""
import numpy as np
from scipy.integrate import solve_bvp, cumulative_trapezoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _saai  # noqa: F401
from _saai import C_V, C_2, K_LAMBDA

CUT = 1.0/K_LAMBDA if K_LAMBDA else 0.16   # lambda_p style split, as in v1


def fs(beta, n=2400):
    m = beta/(2-beta)
    def ode(x, y): return np.vstack([y[1], y[2], -0.5*(m+1)*y[0]*y[2] - m*(1-y[1]**2)])
    def bc(a, b): return np.array([a[0], a[1], b[1]-1])
    x = np.linspace(0, 12, n); y0 = np.vstack([x-1+np.exp(-x), 1-np.exp(-x), np.exp(-x)])
    s = solve_bvp(ode, bc, x, y0, max_nodes=800000, tol=1e-9); Y = s.sol(x)
    return x, Y[0], Y[1], Y[2]


def fs_reverse(beta_target=-0.15, eta_max=30.0, n=3000):
    """Stewartson lower-branch solution; see the v1 script for the
    continuation strategy (wall shear imposed, beta solved)."""
    x = np.linspace(0, eta_max, n)

    def solve_s(s_wall, guess, beta_g):
        def ode(x_, y, p):
            b = p[0]; m = b/(2-b)
            return np.vstack([y[1], y[2], -0.5*(m+1)*y[0]*y[2] - m*(1-y[1]**2)])
        def bc(a, b_, p):
            return np.array([a[0], a[1], a[2]-s_wall, b_[1]-1])
        return solve_bvp(ode, bc, x, guess, p=[beta_g], max_nodes=800000, tol=1e-9)

    fp0 = 1 - (1 + 0.04*x)*np.exp(-0.9*x)
    guess = np.vstack([cumulative_trapezoid(fp0, x, initial=0.0), fp0,
                       np.gradient(fp0, x)])
    beta_g, betas_s = -0.19, {}
    for s_wall in np.arange(-0.02, -0.0901, -0.01):
        sol = solve_s(s_wall, guess, beta_g)
        assert sol.status == 0 and -0.1989 < sol.p[0] < 0
        guess, beta_g = sol.sol(x), float(sol.p[0])
        betas_s[round(s_wall, 4)] = beta_g
    s0, s1 = -0.08, -0.09
    b0, b1 = betas_s[-0.08], betas_s[-0.09]
    for _ in range(12):
        if abs(b1 - beta_target) < 1e-4 or abs(b1 - b0) < 1e-12:
            break
        s2 = s1 + (beta_target - b1)*(s1 - s0)/(b1 - b0)
        sol = solve_s(s2, guess, b1)
        assert sol.status == 0
        guess = sol.sol(x)
        s0, b0, s1, b1 = s1, b1, s2, float(sol.p[0])
    Y = guess
    assert abs(b1 - beta_target) < 1e-3 and Y[2][0] < 0 and Y[1].min() < -0.03
    return x, Y[0], Y[1], Y[2], b1


def trim(G, key):
    ipk = int(np.argmax(key)); outer = np.where(G[ipk:] < 0.008)[0]
    return (ipk + outer[0]) if len(outer) else len(G)


def prof(beta):
    eta, f, fp, fpp = fs(beta)
    th = np.trapezoid(fp*(1-fp), eta); m = beta/(2-beta)
    x = eta**2*fpp/th                                  # Re_Omega / Re_theta
    G = 2*(eta*fpp)**2/(fp**2 + (eta*fpp)**2 + 1e-30)
    lam = m*eta**2/np.maximum(fp, 1e-6)
    e = trim(G, x)
    return x[1:e], G[1:e], lam[1:e]


def prof_lv(beta):
    eta, f, fp, fpp = fs(beta); m = beta/(2-beta)
    fppp = -0.5*(m+1)*f*fpp - m*(1-fp**2)
    den = fp**2 + (eta*fpp)**2 + 1e-30
    G = 2*(eta*fpp)**2/den
    Lv = -fp*fppp*0  # placeholder replaced below (keep flake quiet)
    Lv = -fpp*fppp*eta**3/den
    e = trim(G, eta**2*fpp)
    return Lv[1:e], G[1:e]


def prof_reverse():
    eta, f, fp, fpp, beta_solved = fs_reverse()
    m = beta_solved/(2-beta_solved)
    fppp = -0.5*(m+1)*f*fpp - m*(1-fp**2)
    th = np.trapezoid(fp*(1-fp), eta)
    den = fp**2 + (eta*fpp)**2 + 1e-30
    x = eta**2*np.abs(fpp)/th
    G = 2*(eta*fpp)**2/den
    Lv = -fpp*fppp*eta**3/den
    e = trim(G, eta**2*np.abs(fpp))
    return x[1:e], G[1:e], Lv[1:e]


def parabola():
    t = np.concatenate([np.linspace(1e-4, 0.499, 300),
                        np.linspace(0.501, 40.0, 2000)])
    r = (t - 1)/(2*t - 1)
    return -2*t*(2*t - 1)/((2*t - 1)**2 + (t - 1)**2), 2/(1 + r*r)


def gate(GA, LV):
    band = np.sqrt(np.clip(GA*(2 - GA), 0, None))
    q1 = 1.0 - band/np.sqrt(1 + (C_V*LV)**2)
    Pp = (LV + GA)**2 - GA*(2 - GA)
    q2 = 1.0 - np.clip(GA - 1, 0, None)**2/(1 + C_2*np.clip(Pp, 0, None))
    return q1*q2


betas = [-0.198, -0.10, 0.0, 0.30, 1.00]
fig, (ax, axL, axQ) = plt.subplots(1, 3, figsize=(15.6, 4.9))

# ------------------------- (a) unchanged from v1 -------------------------
xr, Gr, Lvr = prof_reverse()
for b in betas:
    x, G, lam = prof(b); soft = lam < CUT; strong = lam >= CUT
    ax.plot(np.where(soft, x, np.nan), np.where(soft, G, np.nan), '-', color='k', lw=2.2)
    ax.plot(np.where(strong, x, np.nan), np.where(strong, G, np.nan), ls=':', color='k', lw=2.6)
    i = int(np.argmax(x)) if b != 1.00 else int(np.argmax(x)) + int(np.argmin(np.abs(G[int(np.argmax(x)):]-0.15)))
    ax.annotate(fr'$\beta={b}$', (x[i], G[i]), fontsize=8.5, ha='left', va='center',
                xytext=(5, 0), textcoords='offset points')
ax.plot(xr, Gr, ls=(0, (5, 1.6, 1, 1.6)), color='0.45', lw=2.0)
ir = int(np.argmax(Gr))
ax.annotate('reverse-flow branch,\n' r'$\beta=-0.15$ (Stewartson)',
            (xr[ir], Gr[ir]), fontsize=8, color='0.3', ha='left', va='top',
            xytext=(10, -8), textcoords='offset points')
ax.axhline(1, color='0.6', ls=(0, (4, 3)), lw=0.8)
ax.set_xscale('log'); ax.set_xlim(1e-2, 15.0); ax.set_ylim(0, 2.02)
ax.set_xlabel(r'$Re_\Omega/Re_\theta$'); ax.set_ylabel(r'$\Gamma$')
ax.text(0.03, 0.97, '(a)', transform=ax.transAxes, fontsize=11, va='top', fontweight='bold')

# --------------- (b) (Lambda_v, Gamma) + universal parabola ---------------
for b in betas:
    Lv, G = prof_lv(b)
    axL.plot(Lv, G, '-', color='k', lw=2.2)
    j = int(np.argmax(np.abs(Lv)))
    ha = 'left' if Lv[j] > 0 else 'right'
    axL.annotate(fr'$\beta={b}$', (Lv[j], G[j]), fontsize=8.5, ha=ha,
                 va='center', xytext=(6 if Lv[j] > 0 else -6, 0),
                 textcoords='offset points')
axL.plot(Lvr, Gr, ls=(0, (5, 1.6, 1, 1.6)), color='0.45', lw=2.0)
Lp, Gp = parabola()
axL.plot(Lp, Gp, '-', color='crimson', lw=2.6, alpha=0.85)
axL.annotate('universal parabola\n' r'$u\propto y\,(y-y_0)$',
             (-2.35, 1.15), fontsize=8, color='crimson')
for x0, y0, mk in ((0, 1, 'o'), (-2, 2, 's')):
    axL.plot(x0, y0, mk, ms=9, mfc='w', mec='k', mew=1.6, zorder=6)
axL.annotate('wall pinch $(1,0)$', (0, 1), fontsize=8,
             textcoords='offset points', xytext=(8, -4))
axL.annotate('recirculation pinch $(2,-2)$', (-2, 2), fontsize=8,
             textcoords='offset points', xytext=(6, -12))
axL.axhline(1, color='0.6', ls=(0, (4, 3)), lw=0.8)
axL.set_xlim(-3.0, 1.7); axL.set_ylim(0, 2.02)
axL.set_xlabel(r'$\Lambda_v$')
axL.text(0.03, 0.97, '(b)', transform=axL.transAxes, fontsize=11, va='top', fontweight='bold')

# ------------------- (c) composite-gate contour lines -------------------
lv = np.linspace(-3.0, 1.7, 801); ga = np.linspace(0.0, 2.0, 401)
LV, GA = np.meshgrid(lv, ga)
cs = axQ.contour(LV, GA, gate(GA, LV),
                 levels=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
                 colors='k', linewidths=0.8)
axQ.clabel(cs, fmt='%.2g', fontsize=7)
Pp = (LV + GA)**2 - GA*(2 - GA)
axQ.contour(LV, GA, np.where(GA > 1.0, Pp, np.nan), levels=[0.0],
            colors='crimson', linewidths=2.2)
for x0, y0, mk in ((0, 1, 'o'), (-2, 2, 's')):
    axQ.plot(x0, y0, mk, ms=9, mfc='w', mec='k', mew=1.6, zorder=6)
axQ.text(-2.0, 1.5, '$Q_2$ pocket', fontsize=8.5, color='crimson', ha='center')
axQ.set_xlim(-3.0, 1.7); axQ.set_ylim(0, 2.02)
axQ.set_xlabel(r'$\Lambda_v$')
axQ.text(0.03, 0.97, '(c)', transform=axQ.transAxes, fontsize=11, va='top', fontweight='bold')

plt.tight_layout(); plt.savefig('figs/indicator_plane.pdf')
print("wrote figs/indicator_plane.pdf")
