"""Repair strip-cache stations whose target-cl solve locked onto the wrong
branch: re-solve them with the alpha-secant fallback now in
regen_daedalus_strips_cache.station()."""
import pickle
import sys
import numpy as np

sys.path.insert(0, '/home/qiqi/flexcompute/sa-ai/paper')
sys.path.insert(0, '/home/qiqi/flexcompute/sa-ai/scripts/daedalus')
import regen_daedalus_strips_cache as C
from polar_compare import run_avl
from wing_geometry import HALF_SPAN

FF = pickle.load(open(C.OUT, 'rb'))
for a in (4.0, 5.0, 6.0):
    _, _, strips = run_avl(a)
    d = FF[a]
    cls = np.array([st['cl'] if st and st.get('cl') is not None else np.nan
                    for st in d['stations']], float)
    med = np.nanmedian(cls)
    for i, eta in enumerate(d['eta']):
        if np.isfinite(cls[i]) and abs(cls[i] - med) <= 0.15:
            continue
        cl_t = float(np.interp(eta * HALF_SPAN, strips[:, 0], strips[:, 2]))
        st = C.station(float(eta), cl_t)
        if st is not None and st.get('cl') is not None \
                and abs(st['cl'] - cl_t) < 0.05:
            d['stations'][i] = st
            print(f'a={a} eta={eta:.3f} repaired -> cl={st["cl"]:.3f} '
                  f'(target {cl_t:.3f})', flush=True)
        else:
            got = None if st is None else st.get('cl')
            print(f'a={a} eta={eta:.3f} STILL BAD (got {got})', flush=True)
pickle.dump(FF, open(C.OUT, 'wb'))
print('repaired pkl saved')
