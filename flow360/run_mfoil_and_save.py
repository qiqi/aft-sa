"""Run mfoil on NACA 0012, Re=1e6, alpha=0 with multiple Ncrit values and save
Cp/Cf/N for the upper surface to compare against variant B (unified).

We save:
  - x, Cp, Cf at panel midpoints (upper surface)
  - x, N (amplification factor) at panel midpoints (upper surface, laminar region)
  - transition location

For consistency with variant B, primary Ncrit = 15.65 (= 4×ln(50), matching the
unified's log-space runway from chi_inf=1.6e-7 to chi=1).
Also run Ncrit=9 (XFOIL default) for reference.
"""
import sys, pickle, numpy as np
sys.path.insert(0, '/home/qiqi/flexcompute/aft-sa/src/validation')
import mfoil as M
from contextlib import redirect_stdout, redirect_stderr
import os

OUT = "/home/qiqi/flexcompute/aft-sa/flow360/mfoil_naca0012_compare.pkl"

def run_one(ncrit, Re=1e6, alpha=0.0, npanel=199):
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        from loguru import logger; logger.disable("mfoil")
        m = M.mfoil(naca='0012', npanel=npanel)
        m.setoper(alpha=alpha, Re=Re)
        m.param.ncrit = ncrit
        m.solve()
    # Identify upper surface: panels with positive z (after going around).
    Geom = m.geom
    Isol = m.isol
    Vsol = m.vsol
    Glob = m.glob
    Post = m.post
    # mfoil node coordinates: m.foil.x of shape (2, N+1) with surface going from
    # lower TE around to upper TE.
    xy = m.foil.x   # shape (2, N+1) — node coords on the airfoil
    # Panel-midpoint Cp/Cf already stored on each side
    # m.post.cp and m.post.cf are at airfoil nodes (size N+1)
    cp = m.post.cp      # shape (N+1)
    cf = m.post.cf      # shape (N+1)
    # Stagnation point index
    Ist = m.isol.Istag
    # Split into upper/lower based on isol identification
    # m.isol.sgnue tells direction (1 on upper, -1 on lower) - use that
    # Easier: walk from stag forward. Upper goes from stag toward TE going up.
    # The 'mfoil' geometry order: 0 = lower TE, around LE, to upper TE = N
    # Stagnation point index Istag is on lower or upper. For α=0 NACA0012 it's at LE.
    N = xy.shape[1] - 1
    # Determine upper-surface index range
    xc = xy[0]
    zc = xy[1]
    # Upper surface: zc > 0 (after the LE)
    upper_idx = np.where(zc > 1e-8)[0]
    # Order along the surface
    # mfoil's amplification factor n stored per-node? Look at Vsol.turb / state
    # state vector U has [theta, delta*, ue, n_or_cteq] for each node
    U = m.glob.U  # shape (4, N+wake_nodes)
    # n is in row 3 BEFORE transition (when turb=False), cteq after
    turb_flag = m.vsol.turb  # bool per node
    # Build per-upper-surface-node arrays sorted by x
    order = np.argsort(xc[upper_idx])
    iup = upper_idx[order]
    xu = xc[iup]
    cp_u = cp[iup]
    cf_u = cf[iup]
    # Amplification: row 3 of U gives n where laminar (turb=False), cteq where turb
    # mfoil stores amplification factor in U[2, :] (laminar) or sqrt(c_tau) (turb)
    nfac = np.full_like(xu, np.nan)
    for k, i in enumerate(iup):
        if not turb_flag[i]:
            nfac[k] = U[2, i]
    xtr_upper = m.vsol.Xt[1, 1] if hasattr(m.vsol, 'Xt') else None  # transition x upper
    return {
        'x': xu, 'cp': cp_u, 'cf': cf_u, 'n': nfac,
        'cl': float(m.post.cl), 'cd': float(m.post.cd),
        'cdf': float(m.post.cdf), 'cdp': float(m.post.cdp),
        'xtr_upper': xtr_upper, 'ncrit': ncrit,
    }

data = {}
for ncrit in [9.0, 15.65]:
    print(f"=== mfoil with N_crit = {ncrit} ===")
    d = run_one(ncrit)
    print(f"   CL={d['cl']:+.5f}  CD={d['cd']:.5f}  CDf={d['cdf']:.5f}  CDp={d['cdp']:.5f}  x_tr_upper={d['xtr_upper']}")
    print(f"   N range: {np.nanmin(d['n']):.2f} to {np.nanmax(d['n']):.2f}")
    data[ncrit] = d

pickle.dump(data, open(OUT, 'wb'))
print(f"\nsaved -> {OUT}")
