"""Dense e^9 (mfoil) alpha sweep of the Eppler 387 bubble stations at
Re=2e5: laminar separation = first signed-Cf zero crossing into negative on
the upper surface, turbulent reattachment = recovery crossing back positive
(the tab:eppxtr definition).  -> data/mfoil_eppler_bubble_sweep.json
"""
import os, sys, json
import numpy as np
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _H)
import regen_reference_pickles as RP

DAT = RP.EPP_DAT

out = {}
for a in np.arange(-2.0, 7.51, 0.5):
    e = RP.mfoil_entry(DAT, 2e5, a)
    rec = dict(cl=e.get('cl'), conv=e.get('conv', False), ls=None, tr=None)
    if e.get('conv') and 'upper' in e:
        x, cf = e['upper']['x'], e['upper']['cf']
        o = np.argsort(x); x, cf = x[o], cf[o]
        neg = np.where(cf < 0)[0]
        if len(neg):
            rec['ls'] = float(x[neg[0]])
            j = neg[-1]
            if j + 1 < len(x):     # interpolate the upward zero crossing
                f = (0.0 - cf[j]) / (cf[j + 1] - cf[j])
                rec['tr'] = float(x[j] + f * (x[j + 1] - x[j]))
    out[f'{a:.1f}'] = rec
    print(a, rec, flush=True)
p = f'{os.path.dirname(os.path.dirname(_H))}/data/mfoil_eppler_bubble_sweep.json'
if os.path.exists(p):              # never clobber the xfoil block
    old = json.load(open(p))
    old.update(out)
    out = old
json.dump(out, open(p, 'w'), indent=1)
print('wrote', p)


def xfoil_highalpha():
    """XFOIL e^9 bubble stations where mfoil is past its convergence edge
    (alpha >= 6.5): appended to the same JSON under 'xfoil'."""
    import json as _json
    out = {}
    for a in np.arange(6.5, 9.01, 0.5):
        e = RP.xfoil_entry(DAT, 2e5, a)
        rec = dict(cl=None, conv=False, ls=None, tr=None)
        if e and e.get('upper') is not None:
            rec['cl'] = e.get('cl'); rec['conv'] = True
            x, cf = np.asarray(e['upper']['x']), np.asarray(e['upper']['cf'])
            o = np.argsort(x); x, cf = x[o], cf[o]
            m = (x > 0.02) & (x < 0.9); x, cf = x[m], cf[m]
            neg = np.where(cf < 0)[0]
            if len(neg):
                rec['ls'] = float(x[neg[0]])
                rec['tr'] = float(x[neg[-1] + 1]) if neg[-1] + 1 < len(x) else None
        out[f'{a:.1f}'] = rec
        print('xfoil', a, rec, flush=True)
    p = f'{os.path.dirname(os.path.dirname(_H))}/data/mfoil_eppler_bubble_sweep.json'
    j = _json.load(open(p)); j_x = dict(j) if isinstance(j, dict) else j
    j_x['xfoil'] = out
    _json.dump(j_x, open(p, 'w'), indent=1)
    print('updated', p)


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'xfoil':
    xfoil_highalpha()
