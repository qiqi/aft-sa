"""Transition-front sensitivity ("difficulty score") dx_tr/dN_crit via XFOIL.

For each airfoil case of Secs. V/VI, re-converge XFOIL at N_crit = 8, 9, 10
(full viscous-inviscid coupling, same Mach/Re as the RANS cases) and report
the central difference

    dx_tr/dN = [x_tr(N_crit=10) - x_tr(N_crit=8)] / 2      (per surface)

A front pinned by the pressure distribution (designed recovery corner,
suction peak) barely moves when the critical amplification changes by a
whole e-fold, so its location tests little of the amplification model; a
free front on a gently-adverse rooftop moves substantially. High incidences
are reached by walking alpha up within one XFOIL session.

Usage: python3 xtr_sensitivity.py [nlf|eppler|all]
"""
import os, subprocess, sys
import numpy as np

DATDIR = '/home/qiqi/flexcompute/sa-ai/external/construct2d'
CASES = {
    'nlf':    dict(dat=f'{DATDIR}/nlf0416.dat',   mach=0.1, re=4_000_000,
                   alphas=[0, 4, 9, 15]),
    'eppler': dict(dat=f'{DATDIR}/eppler387.dat', mach=0.1, re=200_000,
                   alphas=[0, 2, 5, 7]),
}
NCRITS = [8, 9, 10]


def walkup(alpha):
    """Alpha sequence easing XFOIL to high incidence."""
    ladder = [a for a in (0, 4, 8, 12) if a < alpha]
    return ladder + [alpha]


def run_xfoil(dat, mach, re, alpha, ncrit):
    """Returns (cl, xtr_upper, xtr_lower) at the final alpha, or Nones."""
    cmds = [f'LOAD {dat}', 'OPER', f'MACH {mach}', f'VISC {re}',
            'VPAR', f'N {ncrit}', '', 'ITER 400']
    for a in walkup(alpha):
        cmds += [f'ALFA {a}', f'ALFA {a}']
    cmds += ['', 'QUIT', '']
    p = subprocess.run(['xvfb-run', '-a', 'xfoil'], input='\n'.join(cmds),
                       capture_output=True, text=True, timeout=600)
    cl, xtr_u, xtr_l = None, None, None
    for line in p.stdout.splitlines():
        s = line.strip()
        if s.startswith('a =') and 'CL =' in s:
            try:
                cl = float(s.split('CL =')[1].split()[0])
            except Exception:
                pass
        low = s.lower()
        if 'transition at x/c' in low:
            try:
                x = float(s.split('=')[-1].split()[0])
            except Exception:
                continue
            if 'side 1' in low:
                xtr_u = x
            elif 'side 2' in low:
                xtr_l = x
    return cl, xtr_u, xtr_l


def score(af):
    c = CASES[af]
    print(f"\n=== {af}: Re={c['re']:.0e}, M={c['mach']} ===")
    print(f"{'alpha':>5} {'side':>5} | " +
          ' '.join(f'xtr(N={n})' for n in NCRITS) + ' |  dx_tr/dN')
    rows = {}
    for alpha in c['alphas']:
        res = {}
        for n in NCRITS:
            cl, xu, xl = run_xfoil(c['dat'], c['mach'], c['re'], alpha, n)
            res[n] = (cl, xu, xl)
        for side, k in (('upper', 1), ('lower', 2)):
            xs = [res[n][k] for n in NCRITS]
            d = (None if (xs[0] is None or xs[2] is None)
                 else (xs[2] - xs[0]) / (NCRITS[2] - NCRITS[0]))
            rows[(alpha, side)] = (xs, d)
            fx = ' '.join('   --   ' if x is None else f'{x:8.4f}' for x in xs)
            fd = '    --' if d is None else f'{d:+8.4f}'
            print(f"{alpha:>5} {side:>5} | {fx} | {fd}")
        cls = [f'{res[n][0]:.3f}' if res[n][0] is not None else '--' for n in NCRITS]
        print(f"      cl(N=8,9,10) = {', '.join(cls)}")
    return rows


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    for af in (['nlf', 'eppler'] if which == 'all' else [which]):
        score(af)
