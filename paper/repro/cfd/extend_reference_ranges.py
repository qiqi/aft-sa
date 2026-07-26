"""Extend the e^9 reference pickles to the full range of the computed data
(fig:epppolar now spans alpha = -2..8.5; the Re sweep figures span
Re = 6e4..4.6e5 with xfoil/flexfoil previously at the endpoints only).

Adds (never overwrites existing keys):
  mfoil_eppler387_Re200k.pkl : alpha -2, 1, 3, 4, 6 (and tries 8, 8.5)
  xfoil_eppler387_Re200k.pkl : alpha -2, 1, 3, 4, 6, 8, 8.5
  xfoil_eppler387_sweep_a5.pkl    : Re 100k, 200k, 300k
  flexfoil_eppler387_sweep_a5.pkl : Re 100k, 200k, 300k

Run: python3 repro/cfd/extend_reference_ranges.py
"""
import os, sys, json, pickle, subprocess
import numpy as np
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _H)
import regen_reference_pickles as RP

B = RP.B
RF = '/home/qiqi/flexcompute/flexfoil/target/release/rustfoil'


def add(fname, keys, entry_fn):
    p = f'{B}/{fname}'
    d = pickle.load(open(p, 'rb')) if os.path.exists(p) else {}
    for k in keys:
        if k in d and (d[k] or {}).get('conv', True) and d[k].get('cl') is not None:
            print(f'  {fname} {k}: exists, kept', flush=True)
            continue
        e = entry_fn(k)
        if e is None:
            print(f'  {fname} {k}: FAILED', flush=True)
            continue
        d[k] = e
        print(f"  {fname} {k}: cl={e.get('cl')} cd={e.get('cd')} "
              f"conv={e.get('conv')}", flush=True)
    pickle.dump(d, open(p, 'wb'))
    print(f'wrote {p}: keys {sorted(k for k in d)}', flush=True)


def flexfoil_entry(Re, alpha):
    dump = f'/tmp/ff_ext_{Re:g}_{alpha:g}.json'
    p = subprocess.run([RF, 'faithful-viscous', RP.EPP_DAT,
                        f'--alpha={alpha:.2f}', '--re', f'{Re:g}',
                        '--mach', '0.1', '--ncrit', '9',
                        '--max-iterations', '300', '--dump-surface', dump],
                       capture_output=True, text=True, timeout=600)
    cl = cd = None
    for ln in p.stdout.splitlines():
        for tok, name in (('CL', 'cl'), ('CD', 'cd')):
            if f'{tok} =' in ln or f'{tok}=' in ln:
                try:
                    v = float(ln.replace('=', ' = ').split(f'{tok} =')[1].split()[0])
                    if name == 'cl':
                        cl = v
                    else:
                        cd = v
                except Exception:
                    pass
    if cl is None and os.path.exists(dump):
        try:
            d = json.load(open(dump))
            cl, cd = d.get('cl'), d.get('cd')
        except Exception:
            pass
    if cl is None:
        print('    rustfoil stdout tail:', p.stdout.strip()[-200:], flush=True)
        return None
    return dict(cl=cl, cd=cd, conv=True)


print('== mfoil_eppler387_Re200k.pkl ==')
add('mfoil_eppler387_Re200k.pkl', [-2.0, 1.0, 3.0, 4.0, 6.0, 8.0, 8.5],
    lambda a: RP.mfoil_entry(RP.EPP_DAT, 2e5, a))
print('== xfoil_eppler387_Re200k.pkl ==')
add('xfoil_eppler387_Re200k.pkl', [-2.0, 1.0, 3.0, 4.0, 6.0, 8.0, 8.5],
    lambda a: RP.xfoil_entry(RP.EPP_DAT, 2e5, a))
print('== xfoil_eppler387_sweep_a5.pkl ==')
add('xfoil_eppler387_sweep_a5.pkl', [100, 200, 300],
    lambda Rk: RP.xfoil_entry(RP.EPP_DAT, Rk * 1000.0, 5.0))
print('== flexfoil_eppler387_sweep_a5.pkl ==')
add('flexfoil_eppler387_sweep_a5.pkl', [100, 200, 300],
    lambda Rk: flexfoil_entry(Rk * 1000.0, 5.0))
print('EXTEND-REFERENCES-DONE')
