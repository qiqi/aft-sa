"""Re-run mfoil and extract Cp/Cf/N using vsol.Is to map surface-ordered nodes
to global state-vector indices.
"""
import sys, pickle, numpy as np, os
sys.path.insert(0, '/home/qiqi/flexcompute/aft-sa/src/validation')
import matplotlib as mpl; mpl.use('Agg'); mpl.rcParams['text.usetex'] = False
from contextlib import redirect_stdout, redirect_stderr
import mfoil as M
from loguru import logger; logger.disable('mfoil')

OUT = "/home/qiqi/flexcompute/aft-sa/flow360/mfoil_naca0012_compare.pkl"

def run_one(ncrit, Re=1e6, alpha=0.0, npanel=199):
    m = M.mfoil(naca='0012', npanel=npanel)
    m.setoper(alpha=alpha, Re=Re)
    m.param.ncrit = ncrit
    with open(os.devnull,'w') as nf:
        with redirect_stdout(nf), redirect_stderr(nf):
            m.solve()
    # Geometry
    xc = m.foil.x[0]   # global x
    zc = m.foil.x[1]
    U  = m.glob.U      # shape (4, N+wake_nodes)
    turb = m.vsol.turb # bool per global node
    # mfoil tracks 2 surfaces on the airfoil: si=0 (lower) and si=1 (upper).
    # M.vsol.Is[si] gives the ordered list of GLOBAL indices for that surface.
    Is = m.vsol.Is
    # Pick upper-surface (si=1) ordered from stag toward TE
    Iup = Is[1]
    x_up   = xc[Iup]
    cp_up  = m.post.cp[Iup]
    cf_up  = m.post.cf[Iup]
    n_up   = U[2, Iup]                # amplification factor (laminar) / sqrt(c_tau) (turb)
    turb_up = turb[Iup]
    # Replace N with NaN past transition (turb=True)
    n_lam = np.where(turb_up, np.nan, n_up)
    # Transition x for upper side
    xtr_upper = float(m.vsol.Xt[1, 1]) if hasattr(m.vsol, 'Xt') else None
    return {
        'x': x_up, 'cp': cp_up, 'cf': cf_up,
        'n_amp': n_lam,        # only meaningful in laminar region
        'n_raw': n_up,         # raw U[2] including turbulent (sqrt(c_tau))
        'turb': turb_up,
        'cl': float(m.post.cl), 'cd': float(m.post.cd),
        'cdf': float(m.post.cdf), 'cdp': float(m.post.cdp),
        'xtr_upper': xtr_upper, 'ncrit': ncrit,
    }

data = {}
for ncrit in [9.0, 15.65]:
    print(f"=== mfoil N_crit = {ncrit} ===", flush=True)
    d = run_one(ncrit)
    print(f"  CL={d['cl']:+.5f}  CD={d['cd']:.5f}  CDf={d['cdf']:.5f}  CDp={d['cdp']:.5f}  x_tr_upper={d['xtr_upper']}", flush=True)
    print(f"  N_amp (laminar) range: {np.nanmin(d['n_amp']):.2f} to {np.nanmax(d['n_amp']):.2f}", flush=True)
    print(f"  x range upper surface: {d['x'].min():.4f} to {d['x'].max():.4f}, len={len(d['x'])}", flush=True)
    data[ncrit] = d

pickle.dump(data, open(OUT, 'wb'))
print(f"\nsaved -> {OUT}", flush=True)
