"""explore-lambda-v, step C summary: transition locations + forces of the
Lambda_v-gate CFD matrix vs the committed Q4 fleet (flow360_g4, identical
meshes/protocol) and the e^9 references (xfoil Eppler Re200k, mfoil NLF Re4M).

x_tr from the converge_by_xtr history (near-wall chi>1 walk, last batch);
forces from the driver's extract_forces (both trees)."""
import csv
import json
import pickle
import sys
from pathlib import Path

_DRIVER = Path(__file__).resolve().parent
sys.path.insert(0, str(_DRIVER))
_RANS = _DRIVER.parent.parent.parent.parent / "flexfoil" / "rans"
sys.path.insert(0, str(_RANS))
from solve import extract_forces  # noqa: E402

G4 = Path("/home/qiqi/flexcompute/sa-ai/flow360_g4")
LV = Path(sys.argv[1]) if len(sys.argv) > 1 else _DRIVER / "out_lambda_v"
AI = Path("/home/qiqi/flexcompute/sa-ai/flow360_ai")

E9 = {"eppler387": pickle.load(open(AI / "xfoil_eppler387_Re200k.pkl", "rb")),
      "nlf0416": pickle.load(open(AI / "mfoil_nlf0416_Re4M.pkl", "rb"))}

CASES = [(f"{fam}L1prop_{af}_{re}_a{a}", af, float(a))
         for af, re, alphas in (("nlf0416", "Re4M", (0, 4)),
                                ("eppler387", "Re200k", (0, 2, 5, 7)))
         for a in alphas for fam in ("cav", "str")]


def last_xtr(cd: Path):
    p = cd / "xtr_history.csv"
    if not p.exists():
        return None, None
    rows = list(csv.reader(open(p)))
    r = rows[-1]
    return float(r[1]), float(r[2])


def forces(cd: Path):
    try:
        f = extract_forces(cd)
        return f["CL"], f["CD"]
    except Exception:
        return float("nan"), float("nan")


def main():
    print(f"{'case':>32} | {'xtr_up':>28} | {'xtr_lo':>28} | {'CL Lv/Q4':>17} | {'CD Lv/Q4':>19}")
    print(f"{'':>32} | {'Lv':>8} {'Q4':>8} {'e9':>8} | {'Lv':>8} {'Q4':>8} {'e9':>8} |")
    for name, af, a in CASES:
        lu, ll = last_xtr(LV / name)
        gu, gl = last_xtr(G4 / name)
        e = E9[af].get(a, {})
        eu, el = e.get("xtr_upper"), e.get("xtr_lower")
        clv, cdv = forces(LV / name)
        clq, cdq = forces(G4 / name)

        def s(v):
            return f"{v:8.3f}" if v is not None else "     -- "
        print(f"{name:>32} | {s(lu)} {s(gu)} {s(eu)} | {s(ll)} {s(gl)} {s(el)}"
              f" | {clv:+.3f}/{clq:+.3f} | {cdv:.5f}/{cdq:.5f}")
    r = json.load(open(LV / "results.json"))
    err = {k: v for k, v in r.items() if "error" in v}
    if err:
        print("\nERRORED CASES:", list(err))


if __name__ == "__main__":
    main()
