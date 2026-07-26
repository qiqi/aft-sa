"""Add the derived kernel fields (Omega_hat/I_hat/OmegaI/Re_Omega) to every
fv1 case dir that has a center-span slice — required by the five-row suite
figures' rows 1-2 before the root flip."""
import os
import sys

sys.path.insert(0, '/home/qiqi/flexcompute/sa-ai/paper/repro/cfd')
import add_derived_to_slice as A

ROOT = '/home/qiqi/flexcompute/sa-ai/flow360_fv1'
done = skip = fail = 0
for c in sorted(os.listdir(ROOT)):
    d = os.path.join(ROOT, c)
    p = os.path.join(d, 'slice_centerSpan.pvtu')
    if not os.path.isdir(d) or not os.path.exists(p):
        continue
    if os.path.exists(os.path.join(d, 'slice_with_derived.pvtu')):
        skip += 1
        continue
    try:
        ok, msg = A.augment(p)
        print(('OK  ' if ok else 'FAIL') + f' {c}: {msg}', flush=True)
        done += ok
        fail += (not ok)
    except Exception as e:
        print(f'FAIL {c}: {e}', flush=True)
        fail += 1
print(f'AUGMENT-FV1-DONE done={done} skip={skip} fail={fail}', flush=True)
