"""Phase-A CFD gate check for the vg kernel port: SHORT restart run from the
canon converged 2e6 dn state, one stage, ~few hundred pseudo steps.

--mode canon : canon env (AI_A_VISC/AI_REOMC_BC unset -> default 0). Gate (a):
               solver.log echoes ai_aVisc=0, ai_reOmBc=0 and Cd holds the canon
               2e6 dn value (0.2049) -- the default path is unchanged canon.
--mode vg    : vg env (AI_A_VISC=0.0276, AI_REOMC_BC=130). Gate (c): echoes
               ai_aVisc=0.0276, ai_reOmBc=130.

Writes to a throwaway dir under OUT; does NOT touch matrix_summary.jsonl.
Usage: python verify_vg_gates.py --mode canon --gpu 2
"""
from __future__ import annotations
import argparse, json, os, sys, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_dragcrisis_matrix import (OUT, SEEDS, make_case, solve_stage,      # noqa
                                   read_cd_trace)
from run_dragcrisis_pilot import canonical_env, make_env                    # noqa
from run_dragcrisis_extension import TEMPLATES                              # noqa

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["canon", "vg"], required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()
    chi = SEEDS["0.2"]; fslow = 0.01
    seed = OUT / "cyl_Re2000000_Tu0.2_dn_highre"
    case = OUT / f"_verify_{args.mode}"
    make_case(case, 2e6, chi, tmpl=TEMPLATES["highre"])
    # warm restart from the canon converged state
    for f in os.listdir(seed / "restartOutput"):
        shutil.copy2(seed / "restartOutput" / f, case / f)
    # patch: short single stage, restart on, seed
    p = case / "Flow360.json"; d = json.loads(p.read_text())
    s = chi * fslow
    d["freestream"]["turbulenceQuantities"] = {
        "modelType": "ModifiedTurbulentViscosityRatio",
        "modifiedTurbulentViscosityRatio": s}
    for bc in d.get("boundaries", {}).values():
        if bc.get("type") == "Freestream":
            bc["turbulenceQuantities"] = {
                "modelType": "ModifiedTurbulentViscosityRatio",
                "modifiedTurbulentViscosityRatio": s}
    d["runControl"]["restart"] = True
    d["timeStepping"]["maxPseudoSteps"] = args.steps
    p.write_text(json.dumps(d, indent=1))

    env, find = make_env(); env["_find"] = find
    e = dict(env); e.update(canonical_env(chi, fslow))
    if args.mode == "vg":
        e["AI_A_VISC"] = "0.0276"; e["AI_REOMC_BC"] = "130"
    v = solve_stage(case, e, args.gpu, args.steps - 50, args.steps, 1e-3)
    ps, cl, cd = read_cd_trace(case)
    print(f"MODE={args.mode} verdict={v} Cd={cd[-1] if cd else None:.6f} "
          f"CL={cl[-1] if cl else None:+.2e} steps={ps[-1] if ps else 0}")
    print("--- ai_constants echo from solver.log ---")
    for line in open(case / "solver.log"):
        if any(k in line for k in ("ai_rateScale", "ai_reOmCeil", "ai_reOmA",
                                   "ai_reOmB:", "ai_aVisc", "ai_reOmBc",
                                   "SA-AI transition constants")):
            print(line.rstrip())

if __name__ == "__main__":
    main()
