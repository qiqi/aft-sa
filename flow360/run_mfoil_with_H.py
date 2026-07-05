"""Re-run mfoil and save H_k(x), N(x), x_tr, plus theta(x) so we can compute
dN/dRe_theta along the BL. Save to mfoil_naca0012_full.pkl.
"""
import sys, pickle, numpy as np, os
sys.path.insert(0, '/home/qiqi/flexcompute/aft-sa/src/validation')
import matplotlib as mpl; mpl.use('Agg'); mpl.rcParams['text.usetex'] = False
from contextlib import redirect_stdout, redirect_stderr
import mfoil as M
from loguru import logger; logger.disable('mfoil')

OUT = "/home/qiqi/flexcompute/aft-sa/flow360/mfoil_naca0012_full.pkl"

def run_one(ncrit, Re=1e6, alpha=0.0, npanel=199):
    m = M.mfoil(naca='0012', npanel=npanel)
    m.setoper(alpha=alpha, Re=Re)
    m.param.ncrit = ncrit
    with open(os.devnull, "w") as nf:
        with redirect_stdout(nf), redirect_stderr(nf):
            m.solve()
    xc = m.foil.x[0]
    U = m.glob.U  # (4, N+wake): [theta, delta_star, N_or_ctau, u_e]
    turb = m.vsol.turb
    Is = m.vsol.Is
    # Upper surface (si=1)
    Iup = Is[1]
    x_up = xc[Iup]
    theta = U[0, Iup]
    delta_star = U[1, Iup]
    N_or = U[2, Iup]
    u_e = U[3, Iup]
    turb_up = turb[Iup]
    H = delta_star / np.maximum(theta, 1e-30)
    # Re_theta = u_e · theta / nu  (mfoil-internal)
    # In mfoil incompressible solver, nu = 1/Re (chord-based)
    nu_m = 1.0 / Re
    Re_theta = u_e * theta / nu_m
    # N is meaningful only in laminar region
    N_lam = np.where(turb_up, np.nan, N_or)
    return {
        'x': x_up, 'theta': theta, 'delta_star': delta_star, 'H': H,
        'N': N_lam, 'u_e': u_e, 'Re_theta': Re_theta, 'turb': turb_up,
        'cp': m.post.cp[Iup], 'cf': m.post.cf[Iup],
        'cl': float(m.post.cl), 'cd': float(m.post.cd),
        'cdf': float(m.post.cdf), 'cdp': float(m.post.cdp),
        'xtr_upper': float(m.vsol.Xt[1, 1]) if hasattr(m.vsol, 'Xt') else None,
        'ncrit': ncrit,
    }

data = {}
for ncrit in [9.0, 15.65]:
    print(f"--- ncrit={ncrit} ---", flush=True)
    d = run_one(ncrit)
    print(f"  CDf={d['cdf']:.5f}, xtr={d['xtr_upper']:.4f}", flush=True)
    H_lam = d['H'][~d['turb']]
    print(f"  H(laminar) range: {H_lam.min():.3f} to {H_lam.max():.3f}", flush=True)
    Rt_lam = d['Re_theta'][~d['turb']]
    print(f"  Re_theta(laminar) range: {Rt_lam.min():.1f} to {Rt_lam.max():.1f}", flush=True)
    N_lam = d['N'][~d['turb']]
    print(f"  N range: {np.nanmin(N_lam):.3f} to {np.nanmax(N_lam):.3f}", flush=True)
    data[ncrit] = d

pickle.dump(data, open(OUT, 'wb'))
print(f"saved -> {OUT}")
