"""Build the e^9 strip-theory surface maps for the Daedalus 3x3 figures:
for each alpha in {4,5,6} and ~30 spanwise stations, run FlexFoil
(faithful XFOIL path) at the AVL local cl, local chord Reynolds number and
N_crit=13.6 on the local blended section, dumping per-station upper-surface
cf(x) and amplification N(x). Cached to
flow360_ai/flexfoil_daedalus_strips.pkl as
{alpha: {'eta': [...], 'stations': [{'xc','cf','n','cl','conv'}...]}}."""
import json
import os
import pickle
import subprocess
import sys
import numpy as np

D = '/home/qiqi/flexcompute/sa-ai/scripts/daedalus'
sys.path.insert(0, D)
sys.path.insert(0, '/home/qiqi/flexcompute/sa-ai/paper')
from polar_compare import run_avl                     # noqa: E402
from section_chi_diag import section_coords           # noqa: E402
from wing_geometry import chord, HALF_SPAN, C_ROOT    # noqa: E402

RF = '/home/qiqi/flexcompute/flexfoil/bin_rustfoil'
RE_ROOT, NCRIT, MACH = 5.0e5, 13.6, 0.1
ETAS = np.linspace(0.03, 0.985, 30)
TMP = '/tmp/dae_strip'
OUT = '/home/qiqi/flexcompute/sa-ai/flow360_ai/flexfoil_daedalus_strips.pkl'


def station(eta, cl_loc):
    c = section_coords(eta)
    dat = f'{TMP}_{int(eta*1000):04d}.dat'
    with open(dat, 'w') as f:
        f.write('sec\n')
        for x_, z_ in c.T:
            f.write(f' {x_:.6f} {z_:.6f}\n')
    dump = dat.replace('.dat', '_dump.json')
    re_loc = RE_ROOT * float(chord(eta)) / C_ROOT

    def solve(spec):
        p = subprocess.run([RF, 'faithful-viscous', dat] + spec +
                           ['--re', f'{re_loc:.0f}', '--mach', str(MACH),
                            '--ncrit', str(NCRIT), '--max-iterations', '300',
                            '--dump-surface', dump],
                           capture_output=True, text=True, timeout=600)
        try:
            return json.loads(p.stdout), json.load(open(dump))
        except Exception:
            return None, None

    res, d = solve(['--target-cl', f'{cl_loc:.4f}'])
    # the TargetCl mode sometimes locks onto a low-lift branch; fall back
    # to an alpha secant when the achieved cl misses the target
    if res is None or res.get('cl') is None or abs(res['cl'] - cl_loc) > 0.05:
        a2d = 4.0
        for _ in range(5):
            r2, d2 = solve(['--alpha', f'{a2d:.3f}'])
            if r2 is None or r2.get('cl') is None:
                break
            res, d = r2, d2
            if abs(res['cl'] - cl_loc) < 0.02:
                break
            a2d += (cl_loc - res['cl']) / 0.11
    if res is None or d is None:
        return None
    rows = [r for r in d['upper'] if not r['wake']]
    x = np.array([r['x'] for r in rows])
    o = np.argsort(x)
    return {'xc': x[o],
            'cf': np.array([r['cf'] for r in rows])[o],
            'ue': np.array([r['ue'] for r in rows])[o],
            'n': np.array([r['n'] if not r['turbulent'] else np.nan
                           for r in rows])[o],
            'cl': res.get('cl'), 'cd': res.get('cd'),
            'conv': res.get('converged')}


def main():
    out = {}
    for a in (4.0, 5.0, 6.0):
        _, _, strips = run_avl(a)
        st = []
        for eta in ETAS:
            cl_loc = float(np.interp(eta * HALF_SPAN, strips[:, 0], strips[:, 2]))
            s = station(eta, cl_loc)
            st.append(s)
            ok = s is not None and s['cl'] is not None
            print(f'a={a} eta={eta:.3f} cl_t={cl_loc:.3f} -> '
                  f'{"cl=%.3f" % s["cl"] if ok else "FAIL"}', flush=True)
        out[a] = {'eta': np.array(ETAS), 'stations': st}
    pickle.dump(out, open(OUT, 'wb'))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
