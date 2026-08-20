"""Instrumented clone of stagnation_bistability.run_case.

Supports the term-budget and ablation results of Sec. bistability.

Adds: (a) ablation switches for x- and y-diffusive transport,
      (b) term-by-term budget export at the converged state.
With all switches off it is a line-by-line copy of the original and is
asserted bit-identical against it.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stagnation_bistability as sb
from stagnation_bistability import (CB1, SIGMA, CB2, KAPPA, CW1, CW2, CW3,
                                    CV1, CHI0, grids, hiemenz, PAPER,
                                    TAU_AI, C_NU_AI)


def run_case2(L, nx=384, ny=140, chi_init=None, niter=60000,
              no_xdiff=False, no_ydiff=False, wall_neumann=False, saai=False,
              return_terms=False,
              verbose=False, check=500):
    H = max(40.0, 0.12*L)
    x, y = grids(L, nx, ny, H)
    dx = x[1]-x[0]
    F = hiemenz()
    fy = F(np.minimum(y, 20.0))
    f, fp, fpp = fy[0], fy[1], fy[2]
    U = np.outer(x, fp)
    V = -np.tile(f, (nx, 1))
    S = np.abs(np.outer(x, fpp))
    yw = np.tile(y, (nx, 1))
    yw[:, 0] = y[1]*0.5

    if saai:
        sys.path.insert(0, os.path.join(PAPER, 'repro'))
        from lib.sphere_kernel import sphere_rate
        fppp = -(f*fpp + 1.0 - fp**2)
        a_ai = sphere_rate(U, np.outer(x, fpp), np.outer(x, fppp), yw, nu=1.0)
        R_TIE_AI = CB1/(KAPPA**2*CW1)
    else:
        a_ai = None

    if chi_init is None:
        chi = CHI0*np.exp(-(np.tile(y, (nx, 1))/(0.5*H))**2)
        chi[:, 0] = 0.0
        chi[:, -1] = 0.0
    else:
        chi = chi_init.copy()

    dyc = np.gradient(y)
    dyc2 = np.tile(dyc, (nx, 1))
    dym = np.diff(y)
    niter = int(niter)
    hist = []
    CFLLOC = 0.7
    for step in range(niter):
        chi = np.clip(chi, 0.0, 1e7)
        nut = (C_NU_AI if saai else 1.0) + chi
        chib = np.zeros_like(chi); chib[1:, :] = (chi[1:, :]-chi[:-1, :])/dx
        chif = np.zeros_like(chi); chif[:-1, :] = (chi[1:, :]-chi[:-1, :])/dx
        chix = np.where(U >= 0.0, chib, chif)
        chiy_up = np.zeros_like(chi)
        chiy_up[:, :-1] = (chi[:, 1:]-chi[:, :-1])/dym
        adv = U*chix + V*chiy_up
        fxp = 0.5*(nut[1:, :]+nut[:-1, :])*(chi[1:, :]-chi[:-1, :])/dx
        xdiff = np.zeros_like(chi)
        xdiff[1:-1, :] = (fxp[1:, :]-fxp[:-1, :])/dx
        fyp = 0.5*(nut[:, 1:]+nut[:, :-1])*(chi[:, 1:]-chi[:, :-1])/dym
        ydiff = np.zeros_like(chi)
        ydiff[:, 1:-1] = (fyp[:, 1:]-fyp[:, :-1])/dyc2[:, 1:-1]
        gx = np.gradient(chi, dx, axis=0)
        gy = np.gradient(chi, y, axis=1)
        gx2, gy2 = gx**2, gy**2
        if no_xdiff:
            xdiff = np.zeros_like(chi); gx2 = np.zeros_like(chi)
        if no_ydiff:
            ydiff = np.zeros_like(chi); gy2 = np.zeros_like(chi)
        grad2 = gx2 + gy2
        fv1 = chi**3/(chi**3+CV1**3)
        fv2 = 1.0 - chi/(1.0+chi*fv1)
        St = S + chi*fv2/(KAPPA**2*yw**2)
        St = np.maximum(St, 0.3*S)
        St = np.maximum(St, 1e-12)
        r = np.minimum(chi/(St*KAPPA**2*yw**2), 10.0)
        g = r + CW2*(r**6-r)
        fw = g*((1.0+CW3**6)/(g**6+CW3**6))**(1.0/6.0)
        if saai:
            sP = np.maximum(1.0 - np.exp(-(chi - 1.0)/TAU_AI), 0.0)
            sD = 1.0 - R_TIE_AI*(1.0 - sP)
            prod_lam = (1.0 - sP)*a_ai*S*chi
            prod_turb = sP*CB1*St*chi
            prod = np.maximum(prod_lam, prod_turb)
            dest = sD*CW1*fw*(chi/yw)**2
        else:
            sP = sD = prod_lam = prod_turb = None
            prod = CB1*St*chi
            dest = CW1*fw*(chi/yw)**2
        res = -adv + prod - dest + (xdiff + ydiff + CB2*grad2)/SIGMA
        rate = (np.abs(U)/dx + np.abs(V)/dyc2
                + 2.0*nut/SIGMA*(1.0/dx**2 + 1.0/dyc2**2)
                + CB1*St + 2.0*CW1*fw*chi/yw**2)
        chi = chi + (CFLLOC/rate)*res
        chi = np.maximum(chi, 0.0)
        chi[:, 0] = chi[:, 1] if wall_neumann else 0.0
        chi[:, -1] = 0.0
        chi[0, :] = chi[1, :]
        chi[-1, :] = chi[-2, :]
        if step % check == 0:
            m = float(chi.max())
            hist.append(m)
            if verbose and step % (20*check) == 0:
                print(f'  L={L:7.1f} it={step:6d} maxchi={m:12.5f}', flush=True)
            if m < 1e-3:
                break
            if len(hist) > 40 and abs(hist[-1]-hist[-10]) \
               < 1e-4*max(hist[-1], 1.0):
                break
    m = float(chi.max())
    out = dict(L=L, sustained=bool(m > sb.SUSTAIN), maxchi=m, iters=step,
               x=x, y=y, chi=chi)
    if return_terms:
        t = dict(adv=adv, prod=prod, dest=dest,
                 xdiff=xdiff/SIGMA, ydiff=ydiff/SIGMA,
                 cb2x=CB2*gx2/SIGMA, cb2y=CB2*gy2/SIGMA,
                 U=U, V=V, res=res)
        if saai:
            t.update(sP=sP, sD=sD, prod_lam=prod_lam, prod_turb=prod_turb,
                     a_ai=a_ai, S=S, St=St, fw=fw)
        out['terms'] = t
    return out


if __name__ == '__main__':
    # bit-identity check against the untouched repro solver
    for sa in (False, True):
        a = sb.run_case(700.0, return_field=True, niter=3000, saai=sa)
        b = run_case2(700.0, niter=3000, saai=sa)
        ca, cb = a['field'][2], b['chi']
        same = np.array_equal(ca, cb)
        tag = 'SA-AI' if sa else 'std SA'
        print(f'{tag:>7}: bit-identical={same} maxabsdiff={np.max(np.abs(ca-cb)):.3e}'
              f' maxchi orig={a["maxchi"]:.6f} clone={b["maxchi"]:.6f}')
        assert same
