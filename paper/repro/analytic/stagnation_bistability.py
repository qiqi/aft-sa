"""Standard SA (no AI terms, no ft2) on a frozen Hiemenz stagnation-point
field with EXACTLY ZERO freestream nuHat: does an attachment-anchored
turbulent state self-sustain, and above what leading-edge-radius Reynolds
number?

Model problem (the paper's Appendix on the convergence protocol): plane
stagnation flow U_e = k x -- the prescribed pressure distribution of a
leading edge with strain rate k ~ U_inf/r_LE. The velocity field is the
exact Hiemenz similarity solution (favorable beta=1 Falkner--Skan), FROZEN;
only the SA transport equation is solved on it. Nondimensionalized on the
stagnation-layer thickness delta = sqrt(nu/k) and time 1/k, the steady SA
equation is parameter-free except for the domain half-extent
L = x_max/delta: with k = U/r and x_max = r (the nose), L = sqrt(U r / nu)
= sqrt(Re_r). Sweeping L sweeps the LE-radius Reynolds number.

chi = nuHat/nu obeys (hat coordinates, d = y):
  chi_t + u chi_x + v chi_y = cb1 * Stilde * chi
      + (1/sigma) [ d((1+chi) chi_x)/dx + d((1+chi) chi_y)/dy
                    + cb2 (chi_x^2 + chi_y^2) ]
      - cw1 fw (chi/y)^2
  u = x f'(y), v = -f(y), S = x f''(y),
  Stilde = max(S + chi fv2/(kappa^2 y^2), 0.3 S)   [Flow360's floor]
BCs: chi(wall)=0, chi(top)=0 (zero freestream seed -- exactly),
symmetry at x=0, zero-gradient outflow at x=L.

chi = 0 is an exact fixed point for every L (production is multiplicative
and the seed is zero). The experiment initializes the TURBULENT state
(chi = CHI0 in the layer) and asks whether it decays or self-sustains:
production along near-wall streamlines banks N ~ cb1 * const * L e-folds
(dwell time diverges logarithmically at the attachment point) and lateral
self-diffusion re-seeds the root where advection vanishes, so above a
critical L the turbulent branch is a second stable fixed point.

Numerics: fully explicit update with LOCAL pseudo-time steps (fixed
points unchanged; only the transient path differs): first-order upwind
advection, conservative (1+chi) diffusion in both directions, cb2
gradient-squared, Patankar-style rate limit folded into the local step;
stretched y grid. March to steady state; classify sustained vs
collapsed; bisect the critical L.

-> data/stagnation_bistability.json  + prints a summary table.
Run from paper/: python3 repro/analytic/stagnation_bistability.py
"""
import json
import os

import numpy as np
from scipy.integrate import solve_ivp

_H = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(_H, '..', '..'))

CB1, SIGMA, CB2, KAPPA = 0.1355, 2.0/3.0, 0.622, 0.41
CW1 = CB1/KAPPA**2 + (1.0+CB2)/SIGMA
CW2, CW3, CV1 = 0.3, 2.0, 7.1
CHI0 = 50.0          # turbulent-init level
SUSTAIN = 1.0        # sustained if max chi > this at steady state


def hiemenz():
    """f, f', f'' on a dense eta grid (shooting; f''(0)=1.232588)."""
    def rhs(t, y):
        return [y[1], y[2], -(y[0]*y[2] + 1.0 - y[1]**2)]
    sol = solve_ivp(rhs, [0, 20], [0.0, 0.0, 1.2325876568], dense_output=True,
                    rtol=1e-10, atol=1e-12)
    return sol.sol


def grids(L, nx, ny, H):
    x = np.linspace(0.0, L, nx)
    a = 4.0
    j = np.arange(ny)
    y = H*(np.exp(a*j/(ny-1))-1.0)/(np.exp(a)-1.0)
    y[0] = 0.0
    return x, y


def run_case(L, nx=192, ny=140, t_end=400.0, chi_init=None, verbose=False,
             return_field=False):
    H = max(40.0, 0.12*L)
    x, y = grids(L, nx, ny, H)
    dx = x[1]-x[0]
    F = hiemenz()
    fy = F(np.minimum(y, 20.0))
    f, fp, fpp = fy[0], fy[1], fy[2]
    # frozen field on the (x,y) grid
    U = np.outer(x, fp)              # u(x,y) = x f'(y)
    V = -np.tile(f, (nx, 1))         # v(y) = -f
    S = np.abs(np.outer(x, fpp))     # |omega| = x f''
    yw = np.tile(y, (nx, 1))
    yw[:, 0] = y[1]*0.5              # avoid /0 at the wall row (chi=0 there)

    if chi_init is None:
        # turbulent init: a saturated layer over the wall (interior only;
        # the zero-seed BCs stay exact)
        chi = CHI0*np.exp(-(np.tile(y, (nx, 1))/(0.5*H))**2)
        chi[:, 0] = 0.0
        chi[:, -1] = 0.0
    else:
        chi = chi_init.copy()

    # steady-state solver: explicit update with LOCAL pseudo-time steps
    # (fixed points of the SA equation are unchanged; only the transient
    # path differs). Fully vectorized.
    dyc = np.gradient(y)
    dym = np.diff(y)                 # y_{j+1}-y_j, len ny-1
    dyc2 = np.tile(dyc, (nx, 1))

    niter = 60000
    check = 500
    hist = []
    CFLLOC = 0.7
    for step in range(niter):
        chi = np.clip(chi, 0.0, 1e7)
        nut = 1.0+chi
        # upwind advection (u>=0, v<=0)
        chix = np.zeros_like(chi)
        chix[1:, :] = (chi[1:, :]-chi[:-1, :])/dx
        chiy_up = np.zeros_like(chi)
        chiy_up[:, :-1] = (chi[:, 1:]-chi[:, :-1])/dym
        adv = U*chix + V*chiy_up
        # conservative diffusion, both directions
        fxp = 0.5*(nut[1:, :]+nut[:-1, :])*(chi[1:, :]-chi[:-1, :])/dx
        xdiff = np.zeros_like(chi)
        xdiff[1:-1, :] = (fxp[1:, :]-fxp[:-1, :])/dx
        xdiff[0, :] = 2.0*fxp[0, :]/dx                    # symmetry
        fyp = 0.5*(nut[:, 1:]+nut[:, :-1])*(chi[:, 1:]-chi[:, :-1])/dym
        ydiff = np.zeros_like(chi)
        ydiff[:, 1:-1] = (fyp[:, 1:]-fyp[:, :-1])/dyc2[:, 1:-1]
        gx = np.gradient(chi, dx, axis=0)
        gy = np.gradient(chi, y, axis=1)
        grad2 = gx**2 + gy**2
        # production / destruction
        fv1 = chi**3/(chi**3+CV1**3)
        fv2 = 1.0 - chi/(1.0+chi*fv1)
        St = S + chi*fv2/(KAPPA**2*yw**2)
        St = np.maximum(St, 0.3*S)
        St = np.maximum(St, 1e-12)
        prod = CB1*St*chi
        r = np.minimum(chi/(St*KAPPA**2*yw**2), 10.0)
        g = r + CW2*(r**6-r)
        fw = g*((1.0+CW3**6)/(g**6+CW3**6))**(1.0/6.0)
        dest = CW1*fw*(chi/yw)**2

        res = -adv + prod - dest + (xdiff + ydiff + CB2*grad2)/SIGMA
        # local pseudo-time step from the fastest local rate
        rate = (np.abs(U)/dx + np.abs(V)/dyc2
                + 2.0*nut/SIGMA*(1.0/dx**2 + 1.0/dyc2**2)
                + CB1*St + 2.0*CW1*fw*chi/yw**2)
        chi = chi + (CFLLOC/rate)*res
        chi = np.maximum(chi, 0.0)
        chi[:, 0] = 0.0
        chi[:, -1] = 0.0
        chi[-1, :] = chi[-2, :]                           # outflow
        if step % check == 0:
            m = float(chi.max())
            hist.append(m)
            if verbose and step % (10*check) == 0:
                print(f'  L={L:7.1f} it={step:6d} maxchi={m:10.3f}')
            if m < 1e-3:
                return dict(L=L, sustained=False, maxchi=m, iters=step)
            if len(hist) > 40 and abs(hist[-1]-hist[-10]) \
               < 1e-4*max(hist[-1], 1.0):
                break
    m = float(chi.max())
    out = dict(L=L, sustained=bool(m > SUSTAIN), maxchi=m, iters=step,
               chi_wallmax=float(chi[:, 1:6].max()))
    if return_field:
        out['field'] = (x, y, chi)
    return out


def main():
    turb = None
    results = []
    for L in (30, 100, 300, 1000, 3000):
        res = run_case(float(L), verbose=True)
        res['Re_r'] = L*L
        results.append(res)
        print(f"L={L:6d}  Re_r={L*L:.1e}  sustained={res['sustained']}"
              f"  maxchi={res['maxchi']:.2f}")
    # bisect the critical L between the last collapse and first sustain
    Ls = sorted(r['L'] for r in results if not r['sustained'])
    Hs = sorted(r['L'] for r in results if r['sustained'])
    crit = None
    if Ls and Hs and Ls[-1] < Hs[0]:
        lo, hi = Ls[-1], Hs[0]
        for _ in range(6):
            mid = np.sqrt(lo*hi)
            res = run_case(float(mid))
            res['Re_r'] = mid*mid
            results.append(res)
            print(f"bisect L={mid:8.1f}  sustained={res['sustained']}"
                  f"  maxchi={res['maxchi']:.2f}")
            if res['sustained']:
                hi = mid
            else:
                lo = mid
        crit = dict(L_lo=lo, L_hi=hi, Re_r_lo=lo*lo, Re_r_hi=hi*hi)
        print(f"critical L in [{lo:.0f}, {hi:.0f}]  "
              f"(Re_r {lo*lo:.2e}..{hi*hi:.2e})")
    out = dict(model='standard SA, no ft2, 0.3*Omega Stilde floor',
               seed='chi_freestream = 0 exactly; chi=0 is an exact fixed '
                    'point at every L; turbulent init chi=50',
               parameter='L = x_max/delta = sqrt(Re_r), delta=sqrt(nu/k)',
               results=sorted(results, key=lambda r: r['L']),
               critical=crit)
    p = os.path.join(PAPER, 'data', 'stagnation_bistability.json')
    json.dump(out, open(p, 'w'), indent=1)
    print('wrote', p)


if __name__ == '__main__':
    import sys
    if '--field' in sys.argv:
        # regenerate the L=3000 wedge snapshot for the appendix figure
        # (data/stagnation_field_L3000.npz is gitignored -- rebuild it here)
        r = run_case(3000.0, return_field=True)
        x, y, chi = r['field']
        np.savez_compressed(os.path.join(PAPER, 'data',
                                         'stagnation_field_L3000.npz'),
                            x=x, y=y, chi=chi, maxchi=r['maxchi'])
        print('wrote field snapshot, maxchi', r['maxchi'])
    else:
        main()
