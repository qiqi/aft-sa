"""Export the OpenFOAM flat-plate sweep to a compact JSON the paper's
figure script overlays (paper/data/openfoam_flatplate_summary.json):
per Tu, the binned x centers, in-BL max chi, integrated Re_theta, and Cf
from the shared cf_and_retheta machinery of regen_flatplate_compare.

Run from openfoam/scripts/: python3 export_flatplate_summary.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regen_flatplate_compare import (TU_LIST, cf_and_retheta,
                                     extract_volume_openfoam)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data',
                    'cases')
OUT = '/home/qiqi/flexcompute/sa-ai/paper/data/openfoam_flatplate_summary.json'

out = {}
for tu in TU_LIST:
    cd = os.path.join(ROOT, f"flatplate_Tu{int(round(tu*1000)):04d}")
    pts, u, chi, wd = extract_volume_openfoam(cd)
    x, reth, cf, chimax = cf_and_retheta(pts, u, chi, wd)
    out[f"{tu:.2f}"] = {
        "x": np.round(x, 5).tolist(),
        "chi_max": [None if not np.isfinite(v) else float(f"{v:.5g}")
                    for v in chimax],
        "Re_theta": [None if not np.isfinite(v) else float(f"{v:.6g}")
                     for v in reth],
        "cf": [None if not np.isfinite(v) else float(f"{v:.5g}") for v in cf],
    }
    print(f"Tu={tu}: {np.isfinite(chimax).sum()} chi stations", flush=True)
json.dump({"source": "OpenFOAM v2412 SpalartAllmarasAI (cell-centered, "
                     "incompressible simpleFoam), same 320x80 grid spec; "
                     "see openfoam/README.md",
           "cases": out}, open(OUT, 'w'))
print('wrote', OUT)
