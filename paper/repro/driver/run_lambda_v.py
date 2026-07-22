"""explore-lambda-v, step C: run the exploration CFD matrix with the Lambda_v
band gate (C++ AI_VG_GATE=5) and its re-solved constants.

Reuses the paper driver end to end (case build, convergence protocol,
extraction) -- ONLY the model env changes: canonical_env() is wrapped so every
case gets the Lambda_v overrides on top of the canonical per-case env
(chi seed + slowdown conventions unchanged).

Matrix (17 cases): flat plate Tu sweep (5), NLF0416 Re=4M a={0,4} x
{cav,str} L1 (4), Eppler 387 Re=200k a={0,2,5,7} x {cav,str} L1 (8).

Usage:
    run_lambda_v.py --cv CV --floor F --gc GC --s S --kl KL --kr KR \
                    [--gpus 0,1,...,7] [--outdir .../out_lambda_v] [--cases ...]
"""
from __future__ import annotations

import argparse
import json
import traceback
from multiprocessing import Process, Queue
from pathlib import Path

_DRIVER = Path(__file__).resolve().parent

MATRIX = [
    "flatplate_ags_Tu0040", "flatplate_ags_Tu0080", "flatplate_ags_Tu0160",
    "flatplate_ags_Tu0300", "flatplate_ags_Tu0600",
    "cavL1prop_nlf0416_Re4M_a0", "cavL1prop_nlf0416_Re4M_a4",
    "strL1prop_nlf0416_Re4M_a0", "strL1prop_nlf0416_Re4M_a4",
    "cavL1prop_eppler387_Re200k_a0", "cavL1prop_eppler387_Re200k_a2",
    "cavL1prop_eppler387_Re200k_a5", "cavL1prop_eppler387_Re200k_a7",
    "strL1prop_eppler387_Re200k_a0", "strL1prop_eppler387_Re200k_a2",
    "strL1prop_eppler387_Re200k_a5", "strL1prop_eppler387_Re200k_a7",
]


def lambda_v_overrides(a) -> dict[str, str]:
    env = {
        "AI_VG_GATE": repr(float(a.gate)),       # 5: Lambda_v; 6: composite Q1s*Q2
        "AI_VG_GATE_WEIGHT": repr(a.cv),         # cV
        "AI_REOMEGA_FLOOR": repr(a.floor),       # re-solved three-anchor kernel
        "AI_GCRIT": repr(a.gc),
        "AI_SIGMOIDSLOPE": repr(a.s),
        "AI_CLIFF_LAMBDA_SLOPE": repr(a.kl),     # re-derived K_lambda
        "AI_FPG_RATE_SLOPE": repr(a.kr),         # re-fit K_r
    }
    if a.gate >= 6:
        env["AI_VG_GATE_L0"] = repr(a.l0)        # Q2 pinch center
        env["AI_VG_GATE_C2"] = repr(a.c2)        # Q2 pinch width
        env["AI_VG_GATE_BPOW"] = repr(a.bpow)    # Q1s band exponent
    if getattr(a, "nulam", None) is not None:
        env["AI_NULAMSCALE"] = repr(a.nulam)     # c_nu,ai diffusion scale
    return env


def worker(gpu: int, q: Queue, outroot: Path, over: dict[str, str], res_q: Queue):
    import run as _run
    import saai_env as _se
    orig = _se.canonical_env

    def patched(chi, *, laminar_slowdown=None):
        env = orig(chi, laminar_slowdown=laminar_slowdown)
        env.update(over)
        return env
    _se.canonical_env = patched
    _run.canonical_env = patched

    import cases as _cases
    while True:
        name = q.get()
        if name is None:
            return
        try:
            cfg = _cases.get(name)
            r = _run.run_case(cfg, outroot / name, gpu=gpu)
            res_q.put((name, {k: r.get(k) for k in
                              ("CL", "CD", "L_over_D", "xtr", "chi_inf")}))
        except Exception:
            traceback.print_exc()
            res_q.put((name, {"error": traceback.format_exc(limit=3)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", type=float, required=True)
    ap.add_argument("--floor", type=float, required=True)
    ap.add_argument("--gc", type=float, required=True)
    ap.add_argument("--s", type=float, required=True)
    ap.add_argument("--kl", type=float, required=True)
    ap.add_argument("--kr", type=float, required=True)
    ap.add_argument("--gate", type=int, default=5)
    ap.add_argument("--l0", type=float, default=-1.8)
    ap.add_argument("--c2", type=float, default=2.0)
    ap.add_argument("--bpow", type=float, default=0.5)
    ap.add_argument("--nulam", type=float, default=None,
                    help="override c_nu,ai (AI_NULAMSCALE); default C++ 1/12")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--outdir", default=str(_DRIVER / "out_lambda_v"))
    ap.add_argument("--cases", nargs="*", default=None,
                    help="subset of MATRIX (default: all 17)")
    ap.add_argument("--full-fleet", action="store_true",
                    help="run the ENTIRE paper SA-AI matrix from cases.py "
                         "(69 minus the 8 gate-independent turbulent "
                         "baselines) instead of the 17-case explore MATRIX")
    a = ap.parse_args()
    if a.full_fleet:
        import cases as _c
        a.cases = [n for n in _c.all_cases()
                   if not getattr(_c.get(n), "turbulent", False)]

    over = lambda_v_overrides(a)
    outroot = Path(a.outdir)
    outroot.mkdir(parents=True, exist_ok=True)
    (outroot / "model_env.json").write_text(json.dumps(over, indent=2))

    todo = a.cases if a.cases else MATRIX
    gpus = [int(g) for g in a.gpus.split(",")]
    q: Queue = Queue()
    res_q: Queue = Queue()
    for name in todo:
        q.put(name)
    for _ in gpus:
        q.put(None)
    procs = [Process(target=worker, args=(g, q, outroot, over, res_q))
             for g in gpus]
    for p in procs:
        p.start()
    results = {}
    for _ in todo:
        name, r = res_q.get()
        results[name] = r
        (outroot / "results.json").write_text(json.dumps(results, indent=2))
        print(f"== {name}: {r}", flush=True)
    for p in procs:
        p.join()
    print(f"ALL DONE ({len(results)}/{len(todo)}); results.json in {outroot}")


if __name__ == "__main__":
    main()
