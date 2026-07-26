"""FlexFoil (faithful-XFOIL) bubble-station sweep for fig:eppbubble: an
independent e^9 implementation across the full incidence range, stations by
the same signed-Cf zero-crossing convention as the figure.  Finding worth the
caption: at alpha >= 7 (and 8.5-9) the closure transitions AHEAD of laminar
separation -- min Cf stays positive, no bubble exists to plot -- while the
oil flow still shows one at 7 deg (0.33/0.48).

  python3 run_flexfoil_bubble_sweep.py -> data/flexfoil_eppler_bubble_sweep.json
"""
import os, json, subprocess
import numpy as np

RF = '/home/qiqi/flexcompute/flexfoil/target/release/rustfoil'
DAT = '/home/qiqi/flexcompute/sa-ai/external/construct2d/eppler387.dat'
_H = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_H, '..', '..', 'data', 'flexfoil_eppler_bubble_sweep.json')

out = {}
for a in np.arange(-2.0, 9.01, 0.5):
    dump = f'/tmp/ff_bubble_{a:.1f}.json'
    # --alpha=<v> form: clap rejects a separate "-2.00" token as a flag,
    # which is why every negative incidence of the first sweep errored
    p = subprocess.run([RF, 'faithful-viscous', DAT, f'--alpha={a:.2f}',
                        '--re', '200000', '--mach', '0.1', '--ncrit', '9',
                        '--max-iterations', '300', '--dump-surface', dump],
                       capture_output=True, text=True, timeout=600)
    rec = dict(conv=False, ls=None, tr=None, xtr=None)
    try:
        d = json.load(open(dump))
        u = [q for q in d['upper'] if not q['wake']]
        x = np.array([q['x'] for q in u]); cf = np.array([q['cf'] for q in u])
        tb = np.array([q['turbulent'] for q in u])
        o = np.argsort(x); x, cf, tb = x[o], cf[o], tb[o]
        m = (x > 0.05) & (x < 0.9)
        xm, cm = x[m], cf[m]
        rec['conv'] = True
        rec['xtr'] = float(x[tb][0]) if tb.any() else None
        neg = np.where(cm < 0)[0]
        if len(neg):
            i0, i1 = neg[0], neg[-1]
            if i0 > 0:
                f = (0 - cm[i0-1]) / (cm[i0] - cm[i0-1])
                rec['ls'] = float(xm[i0-1] + f * (xm[i0] - xm[i0-1]))
            if i1 + 1 < len(xm):
                f = (0 - cm[i1]) / (cm[i1+1] - cm[i1])
                rec['tr'] = float(xm[i1] + f * (xm[i1+1] - xm[i1]))
    except Exception as e:
        rec['err'] = str(e)[:60]
    out[f'{a:.1f}'] = rec
    print(a, rec, flush=True)
json.dump(out, open(OUT, 'w'), indent=1)
print('wrote', OUT)
