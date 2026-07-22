"""Upper-surface turbulent-REATTACHMENT table for the Eppler 387 (Sec. VI, Table 6).

For a laminar-separation bubble the physically observable quantity is the turbulent
reattachment point (the aft end of the bubble), not the amplification onset. We
define reattachment in the computation as the most-downstream location where the
signed C_f recovers through +1e-3 (the final return to the turbulent branch after
the separated / near-separated region). The experiment is the oil-flow turbulent-
reattachment location (McGhee et al., NASA TM 4062, Table III), which is also the
aft edge of the grey band drawn on the C_f rows of Figs. eppcflow/eppcfhigh.

Prints the Table~\ref{tab:eppxtr} rows for cav/str x L0/L1/L2, mfoil e^9, and Exp.
"""
import os
import sys, pickle
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper")
import numpy as np
import regen_eppler_v2 as R

B = os.environ.get("SAAI_CFD_ROOT", "/home/qiqi/flexcompute/sa-ai/flow360_ai")
# NASA TM 4062, Table III oil-flow turbulent reattachment (aft edge of the grey
# LSB band in the C_f figures), R=2e5.
EXP_REATTACH = {0: 0.74, 2: 0.67, 5: 0.59, 7: 0.48}
mn = pickle.load(open(f"{B}/mfoil_eppler387_Re200k.pkl", "rb"))


def reattach(x, cf, thr=1e-3, xlo=0.05, xhi=0.85):
    """Most-downstream upward crossing of C_f = thr within the LSB window: the
    final recovery to the turbulent branch. Robust to the marginal double-graze
    at alpha=7 (where C_f only just reaches zero). None if never (near-)separates."""
    x = np.asarray(x, float); cf = np.asarray(cf, float)
    m = (x > xlo) & (x < xhi); x, cf = x[m], cf[m]
    o = np.argsort(x); x, cf = x[o], cf[o]
    if cf.min() >= thr:
        return None
    xs = []
    for i in range(1, len(x)):
        if cf[i] >= thr and cf[i - 1] < thr:
            f = (thr - cf[i - 1]) / (cf[i] - cf[i - 1])
            xs.append(float(x[i - 1] + f * (x[i] - x[i - 1])))
    return xs[-1] if xs else None


fmt = lambda v: f"{v:7.3f}" if v is not None else f"{'--':>7}"
print(f"{'a':>3} |{'cavL0':>7}{'cavL1':>7}{'cavL2':>7} |{'strL0':>7}{'strL1':>7}{'strL2':>7} |"
      f"{'mfoil':>7} |{'exp':>6}")
for a in [0, 2, 5, 7]:
    row = {}
    for mesh in ("cav", "str"):
        for lev in ("L0", "L1", "L2"):
            d = f"{B}/{mesh}{lev}prop_eppler387_Re200k_a{a}"
            (xu, cfu, _), _ = R.airfoil_walk_contour(d)
            row[f"{mesh}{lev}"] = reattach(xu, cfu)
    e = mn[float(a)]
    mf = reattach(e["upper"]["x"], e["upper"]["cf"])
    # flag mfoil where it is at the edge of convergence near stall (cl non-monotone)
    flag = "*" if a == 7 else " "
    print(f"{a:>3} |" + "".join(fmt(row[f'cav{l}']) for l in ("L0", "L1", "L2")) + " |"
          + "".join(fmt(row[f'str{l}']) for l in ("L0", "L1", "L2")) + " |"
          + fmt(mf) + flag + f" |{EXP_REATTACH[a]:6.2f}   cl_mf={e['cl']:.3f}")
print("* mfoil at alpha=7 (Re=2e5) is at the edge of its convergence regime; unreliable.")
