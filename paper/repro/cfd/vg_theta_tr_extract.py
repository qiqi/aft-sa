"""Targeted radial-ray theta_tr for the vg cylinder cases and their canon
counterparts, reusing dragcrisis_transition_angle.ray_profiles/first_cross
(same conventions). Reads each case's OWN summary.json for meta; does NOT
touch matrix_summary.jsonl. Writes vg_theta_tr.json to figs_explore/data."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dragcrisis_transition_angle import (ray_profiles, first_cross, separations,
                                         C_V1, THETA)  # noqa: E402

ROOT = "/local_data/qiqi/sa-ai/dragcrisis_matrix"
OUT = os.path.join(HERE, "figs_explore", "data", "vg_theta_tr.json")


def do(case):
    cd = os.path.join(ROOT, case)
    if not os.path.exists(os.path.join(cd, "slice_centerSpan.pvtu")):
        return None
    prof, sides, reach, fv = ray_profiles(cd)
    row = json.load(open(os.path.join(cd, "summary.json"))) \
        if os.path.exists(os.path.join(cd, "summary.json")) else {}
    t1 = first_cross(prof["maxchi"], 1.0)
    imax = int(np.argmax(prof["maxchi"]))
    ent = {"re": row.get("re"), "kernel": row.get("kernel", "canon"),
           "dir": row.get("dir"), "Cd": row.get("Cd"),
           "theta_tr_chi1": t1,
           "theta_tr_cv1": first_cross(prof["maxchi"], C_V1),
           "theta_at_maxchi": float(THETA[imax]),
           "maxchi_global": float(f"{prof['maxchi'][imax]:.4g}"),
           "nearwall_chi1_front": min(
               [v for v in (row.get("chi1_front_upper"),
                            row.get("chi1_front_lower")) if v is not None],
               default=None)}
    ent.update(separations(row))
    return ent


if __name__ == "__main__":
    cases = sys.argv[1:]
    out = {}
    for c in cases:
        e = do(c)
        if e:
            out[c] = e
            print(f"{c}: theta_tr(chi1)={e['theta_tr_chi1']} "
                  f"theta_tr(cv1)={e['theta_tr_cv1']} "
                  f"theta@maxchi={e['theta_at_maxchi']} "
                  f"maxchi={e['maxchi_global']} Cd={e['Cd']} "
                  f"sep_first={e['sep_first']} sep_final={e['sep_final']}",
                  flush=True)
        else:
            print(f"{c}: no slice", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
    prev.update(out)
    json.dump(prev, open(OUT, "w"), indent=1, default=float)
    print(f"wrote {OUT}", flush=True)
