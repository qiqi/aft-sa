"""Sec. II.D stage (c), the SIMPLE refit: one-parameter anchor of the onset
threshold for a given laminar-diffusion reduction c_nu,ai.

The shape constants (C, A, B) = (2600, 175, 2) are FIXED by the LST
neutral-point graze (fig02_onset_graze.py) and never refit. The residual
laminar diffusion drains the just-ignited disturbance, so inception must be
permitted earlier to land on the Drela-Giles N=1 stations; the drain scales
like 1/Re_theta per e-fold, so it acts on the thin-layer (low-threshold)
side and vanishes on the thick-layer (ceiling) side. Hence ONE compensation
factor k on the low branch only:

    Re_Omega_crit(S_hat*g) = softmin_2(C, k*[A + B*(S_hat*g)^-2]),

with k anchored by a 1-D solve: the marched Blasius envelope must cross
N=1 at the Drela-Giles station Re_theta = 338. (The earlier free 3-parameter
fit at c=1/12 confirmed this structure by itself returning the ceiling to
its unscaled graze value.)

For each c: bisection on k (Blasius marches only), then one 9-wedge family
pass reporting the N=1 ratios and rms. Seconds per candidate, not minutes.

Run from paper/:  python3 repro/analytic/explore_k_anchor.py [c ...]
Default sweep: 1 1/2 1/4 1/6 1/12 1/24.
Out: figs_explore/results_k_anchor.json + stdout table.
"""
import json
import sys
import numpy as np
from scipy.optimize import brentq

import _saai  # noqa: F401
from explore_reomc_retune import Wedge, BETAS

C_SHAPE, A_SHAPE, B_SHAPE = 2600.0, 175.0, 2.0
RT1_TARGET = 338.0   # Drela-Giles Blasius N=1 station


def consts_for_k(k):
    return (C_SHAPE, k*A_SHAPE, k*B_SHAPE)


def anchor_k(c_nu):
    bl = Wedge(0.0, c_nu)

    def f(k):
        r = bl.Rt1(consts_for_k(k))
        return (np.log(r/RT1_TARGET) if np.isfinite(r) else 1.0)

    lo, hi = 0.02, 4.0
    flo, fhi = f(lo), f(hi)
    if flo > 0.0:
        # Even with the low branch nearly removed the Blasius inception stays
        # past the anchor: diffusion-limited, no k exists at this c_nu.
        return None, np.exp(flo)*RT1_TARGET
    if fhi < 0.0:
        return None, np.exp(fhi)*RT1_TARGET
    return brentq(f, lo, hi, xtol=1e-3), None


def main():
    cvals = [float(eval(a)) for a in sys.argv[1:]] or \
        [1.0, 0.5, 0.25, 1.0/6.0, 1.0/12.0, 1.0/24.0]
    out = {"shape": [C_SHAPE, A_SHAPE, B_SHAPE], "target": RT1_TARGET, "fits": []}
    for c in cvals:
        k, rt1_limit = anchor_k(c)
        if k is None:
            out["fits"].append(dict(c_nu=c, k=None, rms=None,
                                    rt1_limit=rt1_limit))
            print(f"c={c:8.5f}: UNANCHORABLE (diffusion-limited); Blasius "
                  f"N=1 at k->limit: Re_theta={rt1_limit:.0f} vs target "
                  f"{RT1_TARGET:.0f}", flush=True)
            continue
        consts = consts_for_k(k)
        wedges = [Wedge(b, c) for b in BETAS]
        rows, errs = [], []
        for b, w in zip(BETAS, wedges):
            r = w.Rt1(consts)
            ratio = r/w.N1_drela
            errs.append(np.log(ratio))
            rows.append(dict(beta=b, H=w.H, N1_drela=w.N1_drela, Rt1=r,
                             ratio=ratio))
        rms = float(np.sqrt(np.mean(np.square(errs))))
        out["fits"].append(dict(c_nu=c, k=k, rms=rms, rows=rows))
        rats = " ".join(f"{r['ratio']:.3f}" for r in rows)
        print(f"c={c:8.5f}: k={k:.4f}  rms={rms:.4f}  ratios: {rats}", flush=True)
    jpath = 'repro/analytic/figs_explore/results_k_anchor.json'
    json.dump(out, open(jpath, 'w'), indent=1)
    print(f'wrote {jpath}')


if __name__ == '__main__':
    main()
