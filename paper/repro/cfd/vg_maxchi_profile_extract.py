"""Cache the full radial-ray max-chi(theta) profiles for the vg (low-H)
cylinder cases used by fig:newkernelvgresults panel (c).

Reuses dragcrisis_transition_angle.ray_profiles (the SAME radial-ray
extractor the paper uses); stores log10(maxchi) on the shared 0..180 deg
(step 0.5) theta grid. Canon profiles are already cached elsewhere
(dragcrisis_theta_tr.json for 2e6; systematic_Tu0.2_summary.jsonl for the
ultra ladder), so only the vg profiles are (re)extracted here.

Pure CPU/VTK post-processing. Writes figs_explore/data/vg_maxchi_profiles.json.
Run from paper/: python3 repro/cfd/vg_maxchi_profile_extract.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dragcrisis_transition_angle import ray_profiles, THETA  # noqa: E402

ROOT = "/local_data/qiqi/sa-ai/dragcrisis_matrix"
OUT = os.path.join(HERE, "figs_explore", "data", "vg_maxchi_profiles.json")

# Re_D -> vg case dir (up ladder, Tu=0.2%)
CASES = {
    2.0e6:  "cyl_Re2000000_Tu0.2_up_highre_vg",
    1.0e8:  "cyl_Re100000000_Tu0.2_up_ultra_vg",
    1.0e9:  "cyl_Re1000000000_Tu0.2_up_ultra_vg",
    1.0e10: "cyl_Re10000000000_Tu0.2_up_ultra_vg",
}


def main():
    out = {"meta": {"theta_deg": "0..180 step 0.5 from forward stagnation",
                    "value": "log10_maxchi = log10(max chi over radial ray), "
                             "two-side mean; chi = nuHat/muRef",
                    "extractor": "dragcrisis_transition_angle.ray_profiles",
                    "kernel": "vg (low-H)"},
           "cases": {}}
    for re, case in CASES.items():
        cd = os.path.join(ROOT, case)
        prof, sides, reach, fv = ray_profiles(cd)
        maxchi = np.clip(prof["maxchi"], 1e-30, None)
        out["cases"][case] = {
            "re": re,
            "log10_maxchi": [float(round(x, 3)) for x in np.log10(maxchi)],
            "maxchi_global": float(f"{prof['maxchi'].max():.4g}"),
            "theta_at_maxchi": float(THETA[int(np.argmax(prof['maxchi']))]),
        }
        print(f"{case}: maxchi_global={out['cases'][case]['maxchi_global']:.4g} "
              f"theta@max={out['cases'][case]['theta_at_maxchi']} "
              f"reach={reach:.1f}D frac_valid={fv:.3f}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
