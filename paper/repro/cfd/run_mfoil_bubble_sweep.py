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
            rec['tr'] = float(x[neg[-1] + 1]) if neg[-1] + 1 < len(x) else None
    out[f'{a:.1f}'] = rec
    print(a, rec, flush=True)
p = f'{os.path.dirname(os.path.dirname(_H))}/data/mfoil_eppler_bubble_sweep.json'
json.dump(out, open(p, 'w'), indent=1)
print('wrote', p)
