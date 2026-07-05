"""Run mfoil on NLF(1)-0416 at Re = 4×10⁶, M = 0.1, α = 0° and 2.5°, N_crit = 9.
Saves Cp/Cf/N data on upper and lower surfaces.
"""
import sys, pickle, os, numpy as np
sys.path.insert(0, '/home/qiqi/flexcompute/aft-sa/src/validation')
import matplotlib as mpl; mpl.use('Agg'); mpl.rcParams['text.usetex'] = False
from contextlib import redirect_stdout, redirect_stderr
import mfoil as M
from loguru import logger; logger.disable('mfoil')

AIRFOIL_DAT = '/home/qiqi/flexcompute/aft-sa/external/construct2d/nlf0416.dat'

def run(alpha, ncrit=9.0, Re=4e6):
    # Read airfoil coordinates from .dat
    coords = []
    with open(AIRFOIL_DAT) as f:
        for line in f:
            try:
                parts = line.split()
                if len(parts) >= 2:
                    x, y = float(parts[0]), float(parts[1])
                    coords.append((x, y))
            except: pass
    coords = np.array(coords).T  # shape (2, N)
    m = M.mfoil(coords=coords)
    m.setoper(alpha=alpha, Re=Re, Ma=0.1)
    m.param.ncrit = ncrit
    with open(os.devnull,'w') as nf:
        with redirect_stdout(nf), redirect_stderr(nf): m.solve()
    # Classify by GEOMETRY (not by stagnation): merge mfoil's two surfaces, then
    # split by sign of (z - z_camber(x)) using the raw airfoil .dat shape.
    # mfoil's `Is[0]/Is[1]` are stagnation-relative; at high α the LE-tip belongs
    # to mfoil-"upper" (above stagnation) even though it's geometrically lower —
    # which makes mfoil overlays drift across the LE in the Cp plot.
    Is_all = np.concatenate([np.asarray(m.vsol.Is[0]), np.asarray(m.vsol.Is[1])])
    Is_all = np.unique(Is_all)
    x_all = m.foil.x[0][Is_all]; z_all = m.foil.x[1][Is_all]
    cp_all = m.post.cp[Is_all]; cf_all = m.post.cf[Is_all]
    n_all = m.glob.U[2, Is_all]; turb_all = m.vsol.turb[Is_all]
    # BL integral quantities from the global state vector:
    # m.glob.U[0,:] = theta (momentum thickness)
    # m.glob.U[1,:] = delta* (displacement thickness)
    # m.glob.U[3,:] = ue (edge velocity)
    theta_all = m.glob.U[0, Is_all]
    dstar_all = m.glob.U[1, Is_all]
    ue_all = m.glob.U[3, Is_all]
    # Avoid div-by-zero for H
    H_all = np.where(theta_all > 1e-12, dstar_all / theta_all, np.nan)
    Reth_all = ue_all * theta_all * Re  # mfoil θ/c × U_e/U∞ × Re_∞ = Re_θ
    # Geometric camber from the raw .dat (NOT from the sampled point cloud).
    # NLF coords go TE→upper→LE→lower→TE, so coords[0,:] is x and the .dat's
    # natural "upper half" is the first half of the file.
    dat_x = coords[0]; dat_z = coords[1]
    le_idx = int(np.argmin(dat_x))
    upper_dat_x = dat_x[:le_idx+1][::-1]; upper_dat_z = dat_z[:le_idx+1][::-1]
    lower_dat_x = dat_x[le_idx:]; lower_dat_z = dat_z[le_idx:]
    z_up_fn = lambda x: np.interp(np.clip(x, 0, 1), upper_dat_x, upper_dat_z)
    z_lo_fn = lambda x: np.interp(np.clip(x, 0, 1), lower_dat_x, lower_dat_z)
    z_camber = lambda x: 0.5 * (z_up_fn(x) + z_lo_fn(x))
    upper_mask = z_all > z_camber(x_all)
    out = {}
    for side, mask in [('upper', upper_mask), ('lower', ~upper_mask)]:
        order = np.argsort(x_all[mask])
        x = x_all[mask][order]
        cp = cp_all[mask][order]
        cf = cf_all[mask][order]
        n_val = n_all[mask][order]
        turb = turb_all[mask][order]
        theta = theta_all[mask][order]
        dstar = dstar_all[mask][order]
        H = H_all[mask][order]
        ue = ue_all[mask][order]
        Reth = Reth_all[mask][order]
        out[side] = {'x': x, 'cp': cp, 'cf': cf,
                     'n': np.where(turb, np.nan, n_val), 'turb': turb,
                     'theta': theta, 'delta_star': dstar, 'H': H,
                     'ue': ue, 'Re_theta': Reth}
    out['ncrit'] = ncrit
    out['Re'] = Re
    out['xtr_upper'] = float(m.vsol.Xt[1, 1]) if hasattr(m.vsol, 'Xt') else None
    out['xtr_lower'] = float(m.vsol.Xt[0, 1]) if hasattr(m.vsol, 'Xt') else None
    out['cl'] = float(m.post.cl); out['cd'] = float(m.post.cd)
    out['cdf'] = float(m.post.cdf); out['cdp'] = float(m.post.cdp)
    # Whether mfoil's coupled inviscid-viscous Newton iteration actually
    # converged. It fails (limit-cycles) for alpha >= 8 deg on this airfoil at
    # Re=4e6 as the flow separates near stall; those iterates are NOT usable.
    out['conv'] = bool(m.glob.conv)
    return out

data = {}
for alpha in [0.0, 2.5, 4.0, 6.0, 7.0, 9.0, 15.0]:
    print(f'alpha={alpha} ...', flush=True)
    data[alpha] = run(alpha)
    d = data[alpha]
    print(f'  conv={d["conv"]} CL={d["cl"]:+.4f} CD={d["cd"]:.5f} CDf={d["cdf"]:.5f} xtr_u={d["xtr_upper"]} xtr_l={d["xtr_lower"]}')

pickle.dump(data, open('/home/qiqi/flexcompute/aft-sa/flow360/mfoil_nlf0416_Re4M.pkl', 'wb'))
print('saved mfoil_nlf0416_Re4M.pkl')
