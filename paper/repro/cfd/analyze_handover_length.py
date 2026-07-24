"""Handover-length ladder for the Eppler 387 Reynolds sweep (Sec. eppresweep).

For each Reynolds number of the alpha=5 sweep, measures on the upper
surface (L2 grids, both mesh families):

  x(chi=1)     first station where the near-wall max chi crosses unity
               (transition onset; the sigma_t blend begins),
  x(chi=c_v1)  half-saturation of the eddy viscosity,
  x(chi=30)    effectively full SA production (sigma_P ~ 0.9993 at
               tau=4; f_v1 ~ 0.99),
  x_R          turbulent reattachment (last upward zero crossing of the
               signed C_f),

and prints the distances the paper's handover table quotes:
Delta x(1->c_v1), Delta x(1->30), Delta x(1->x_R). The point (confirming
the mechanism behind the raised bursting boundary): the number of chi
e-folds needed for handover is FIXED, so the physical handover length
grows as the Reynolds number falls, and by Re=1e5 it is comparable to the
room the bubble has to close -- the eddy viscosity arrives too late and
the burst branch is selected, one Reynolds step above the experimentally
bistable 6e4 (TM-4062 Fig. 11a).

Run from paper/: SAAI_CFD_ROOT=... python3 repro/cfd/analyze_handover_length.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import regen_eppler_v2 as R

B = os.environ.get("SAAI_CFD_ROOT", "/home/qiqi/flexcompute/sa-ai/flow360_fr")
C_V1 = 7.1
CHI_FULL = 30.0


def cases(fam):
    yield 60, f'{B}/sweep_{fam}L2_Re60k_a5'
    yield 100, f'{B}/sweep_{fam}L2_Re100k_a5'
    yield 200, f'{B}/{fam}L2prop_eppler387_Re200k_a5'
    yield 300, f'{B}/sweep_{fam}L2_Re300k_a5'
    yield 460, f'{B}/sweep_{fam}L2_Re460k_a5'


def chi_station(d, NU, thr):
    xc, mu, _ = R.max_chi_vs_x(d)
    j = np.where(mu/NU > thr)[0]
    return float(xc[j[0]]) if len(j) else None


def reattach(d):
    (xu, cfu, _), _ = R.airfoil_walk_contour(d)
    o = np.argsort(xu)
    x, cf = xu[o], cfu[o]
    neg = np.where(cf < 0)[0]
    if not len(neg):
        return None, None
    i = neg[-1]
    xr = x[i] + (0 - cf[i])*(x[i+1] - x[i])/(cf[i+1] - cf[i]) \
        if i + 1 < len(x) else x[i]
    return float(x[neg[0]]), float(xr)


def main():
    for fam in ('str', 'cav'):
        print(f"== {fam} L2 ==")
        print(f"{'Re':>5} {'x_sep':>6} {'x_chi1':>7} {'x_cv1':>6} "
              f"{'x_chi30':>7} {'x_R':>6} {'d1->cv1':>8} {'d1->30':>7} "
              f"{'d1->R':>7}")
        for Rk, d in cases(fam):
            NU = 0.1/(Rk*1000.0)
            s1 = chi_station(d, NU, 1.0)
            s2 = chi_station(d, NU, C_V1)
            s3 = chi_station(d, NU, CHI_FULL)
            xs, xr = reattach(d)
            f = lambda v: f"{v:.3f}" if v is not None else "  -- "
            dd = lambda a, b: f"{b-a:.3f}" if (a is not None and b is not None) else "  -- "
            print(f"{Rk:>4}k {f(xs):>6} {f(s1):>7} {f(s2):>6} {f(s3):>7} "
                  f"{f(xr):>6} {dd(s1, s2):>8} {dd(s1, s3):>7} {dd(s1, xr):>7}",
                  flush=True)


if __name__ == '__main__':
    main()
