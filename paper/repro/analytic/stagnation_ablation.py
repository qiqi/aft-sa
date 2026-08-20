"""Ablation, refinement, and term-budget campaign on the Hiemenz wedge.

Every case runs the instrumented solver of ``stagnation_terms`` under the
chained continuation protocol of ``stagnation_bistability``: a case is called
sustained only when two consecutive 60k-iteration chunks agree to 2%, since a
fixed-cap "still alive" is not evidence of sustainment.

Usage:
    python3 stagnation_ablation.py case  <name>     one ablation/refinement case
    python3 stagnation_ablation.py local            local y-only SA-AI equilibrium
    python3 stagnation_ablation.py budget <file>    term budget from a saved field
    python3 stagnation_ablation.py roots            chi=1 root of every saved field

Cases (all L = 3000 unless the name says otherwise):
    baseline noxdiff noydiff wallneu ny280 ny420 nx768 nx1536 nx3072
    ai3000 ai1500 ai700 ainx768 ainx1536
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stagnation_bistability import (CB1, SIGMA, CB2, KAPPA, CW1, CW2, CW3,
                                    CV1, CHI0, PAPER, hiemenz, TAU_AI, C_NU_AI)
from stagnation_terms import run_case2

sys.path.insert(0, os.path.join(PAPER, 'repro'))
from lib.sphere_kernel import sphere_rate

R_TIE = CB1/(KAPPA**2*CW1)

CASES = {
    'baseline':  dict(L=3000., ny=140),
    'noxdiff':   dict(L=3000., ny=140, no_xdiff=True),
    'noydiff':   dict(L=3000., ny=140, no_ydiff=True),
    'wallneu':   dict(L=3000., ny=140, wall_neumann=True),
    'ny280':     dict(L=3000., ny=280),
    'ny420':     dict(L=3000., ny=420),
    'nx768':     dict(L=3000., nx=768),
    'nx1536':    dict(L=3000., nx=1536),
    'nx3072':    dict(L=3000., nx=3072),
    'ai3000':    dict(L=3000., saai=True),
    'ai1500':    dict(L=1500., saai=True),
    'ai700':     dict(L=700.,  saai=True),
    'ainx768':   dict(L=3000., nx=768,  saai=True),
    'ainx1536':  dict(L=3000., nx=1536, saai=True),
}


def run_chained(name, nchunk=10, out_dir='.'):
    """Chained continuation to a verdict. Saves the field and its terms."""
    cfg = dict(CASES[name])
    L = cfg.pop('L')
    chi, prev, verdict, m = None, None, 'unresolved', 0.0
    for k in range(nchunk):
        r = run_case2(L, chi_init=chi, niter=60000, return_terms=True, **cfg)
        m = r['maxchi']
        print(f'{name} chunk {k+1}: maxchi={m:12.5f} iters={r["iters"]}',
              flush=True)
        if m < 1e-3:
            verdict = 'collapsed'
            break
        chi = r['chi']
        np.savez_compressed(os.path.join(out_dir, f'{name}.npz'),
                            x=r['x'], y=r['y'], chi=r['chi'], **r['terms'])
        if prev is not None and m > 2.0 and abs(m-prev) < 0.02*m:
            verdict = 'sustained'
            break
        prev = m
    print(f'{name}: VERDICT={verdict} maxchi={m:.5f} chunks={k+1}', flush=True)
    return verdict, m


def local_eq(x, ny=600, H=360.0, niter=400000, saai=True):
    """Local y-only equilibrium at fixed x: no x-advection, no x-diffusion.

    The lowest x carrying a nonzero state is the continuum quenching point,
    i.e. where the ignited branch ceases to exist independently of whether
    anything can reach it.
    """
    a = 4.0
    j = np.arange(ny)
    y = H*(np.exp(a*j/(ny-1))-1.0)/(np.exp(a)-1.0)
    y[0] = 0.0
    F = hiemenz()
    fy = F(np.minimum(y, 20.0))
    f, fp, fpp = fy[0], fy[1], fy[2]
    fppp = -(f*fpp + 1.0 - fp**2)
    V = -f
    S = np.abs(x*fpp)
    yw = y.copy()
    yw[0] = y[1]*0.5
    a_ai = sphere_rate(x*fp, x*fpp, x*fppp, yw, nu=1.0) if saai else None
    chi = CHI0*np.exp(-(y/(0.5*H))**2)
    chi[0] = 0.0
    chi[-1] = 0.0
    dyc = np.gradient(y)
    dym = np.diff(y)
    for it in range(niter):
        chi = np.clip(chi, 0.0, 1e7)
        nut = (C_NU_AI if saai else 1.0) + chi
        chiy = np.zeros_like(chi)
        chiy[:-1] = (chi[1:]-chi[:-1])/dym
        adv = V*chiy
        fyp = 0.5*(nut[1:]+nut[:-1])*(chi[1:]-chi[:-1])/dym
        ydiff = np.zeros_like(chi)
        ydiff[1:-1] = (fyp[1:]-fyp[:-1])/dyc[1:-1]
        gy = np.gradient(chi, y)
        fv1 = chi**3/(chi**3+CV1**3)
        fv2 = 1.0 - chi/(1.0+chi*fv1)
        St = np.maximum(np.maximum(S + chi*fv2/(KAPPA**2*yw**2), 0.3*S), 1e-12)
        r = np.minimum(chi/(St*KAPPA**2*yw**2), 10.0)
        g = r + CW2*(r**6-r)
        fw = g*((1.0+CW3**6)/(g**6+CW3**6))**(1.0/6.0)
        if saai:
            sP = np.maximum(1.0-np.exp(-(chi-1.0)/TAU_AI), 0.0)
            sD = 1.0 - R_TIE*(1.0-sP)
            prod = np.maximum((1.0-sP)*a_ai*S*chi, sP*CB1*St*chi)
            dest = sD*CW1*fw*(chi/yw)**2
        else:
            prod = CB1*St*chi
            dest = CW1*fw*(chi/yw)**2
        res = -adv + prod - dest + (ydiff + CB2*gy**2)/SIGMA
        rate = (np.abs(V)/dyc + 2.0*nut/SIGMA/dyc**2 + CB1*St
                + 2.0*CW1*fw*chi/yw**2)
        chi = np.maximum(chi + (0.7/rate)*res, 0.0)
        chi[0] = 0.0
        chi[-1] = 0.0
        if it % 2000 == 0 and chi.max() < 1e-3:
            return 0.0
    return float(chi.max())


def budget(fn, stations, saai=True):
    """Term budget normalized by chi (so, rates) at the local chi peak."""
    d = np.load(fn)
    x, chi = d['x'], d['chi']
    i0 = int(np.argmin(np.abs(x)))
    xs, c = x[i0:], chi[i0:]
    keys = ['adv', 'prod', 'dest', 'xdiff', 'ydiff', 'cb2x', 'cb2y']
    T = {k: d[k][i0:] for k in keys}
    sP = d['sP'][i0:] if saai and 'sP' in d else None
    head = f'{"x":>8}{"chi":>11}' + ('' if sP is None else f'{"sP":>7}')
    print(head + ''.join(f'{k:>11}' for k in keys))
    for xt in stations:
        i = int(np.argmin(np.abs(xs-xt)))
        j = int(np.argmax(c[i]))
        ch = max(c[i, j], 1e-300)
        row = ''.join(f'{T[k][i, j]/ch:11.3e}' for k in keys)
        tag = '' if sP is None else f'{sP[i, j]:7.3f}'
        print(f'{xs[i]:8.1f}{c[i, j]:11.3e}{tag}{row}')


def roots(files):
    """Streamwise station of the near-wall chi = 1 root, x > 0 half only."""
    print(f'{"case":>12}{"nx":>6}{"ny":>5}{"dx":>9}{"x_root":>10}{"maxchi":>10}')
    for fn in files:
        d = np.load(fn)
        x, y, chi = d['x'], d['y'], d['chi']
        i0 = int(np.argmin(np.abs(x)))
        xs, m = x[i0:], chi[i0:].max(axis=1)
        i = np.where(m >= 1.0)[0]
        xr = xs[i[0]] if len(i) else float('nan')
        name = os.path.basename(fn)[:-4]
        print(f'{name:>12}{len(x):6d}{len(y):5d}{x[1]-x[0]:9.3f}'
              f'{xr:10.1f}{m.max():10.3f}')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'local'
    if what == 'case':
        run_chained(sys.argv[2])
    elif what == 'local':
        print('SA-AI local equilibrium (no x-transport):')
        for x in (400., 800., 1200., 1600., 2000., 2400., 3000.):
            print(f'   x={x:7.0f}   max chi = {local_eq(x):10.4f}', flush=True)
    elif what == 'budget':
        budget(sys.argv[2], (1527, 1543, 1559, 1574, 1590, 1606, 1621,
                             1637, 1653, 1684, 1903, 2405))
    elif what == 'roots':
        roots(sys.argv[2:])
