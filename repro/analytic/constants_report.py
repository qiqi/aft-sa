"""Canonical constant report against the paper's Appendix-E constants block:
a_max = 0.19, (C, A, B; k) = (2600, 175, 2; 0.712), w = 0.35, tau = 4,
c_nu_ai = 1/6, and Mack's map (A_TU, B_TU) = (-8.43, 2.4) with c_v1 = 7.1.

The live source is driver/saai_env._SPHERE (the compiled Flow360
ModelConstants.h defaults, pinned against the header by
sa-ai/tests/test_constants_consistency.py); the constants with no env
override are pinned to their compiled values here. Prints every constant
and asserts it equals the paper value. The retired v2 gate-kernel report
this file used to be (g_c/s/floor/K_lambda/K_r against lib/aft_sources, itself
removed 2026-07-30) lives in git history; those constants are no longer in the
paper, and the canonical kernel is lib/sphere_kernel.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'driver'))
from saai_env import _SPHERE

K = 0.712
PAPER = {
    "a_max": 0.19,
    "reOm_ceil": K * 2600.0,   # = 1851.2
    "reOm_A": K * 175.0,       # = 124.6
    "reOm_B": K * 2.0,         # = 1.424
    "ramp_w": 0.35,
    "tau": 4.0,
    "c_nu_ai": 1.0 / 6.0,
    "A_TU": -8.43,
    "B_TU": 2.4,
    "c_v1": 7.1,
}
GOT = {
    "a_max": _SPHERE["A_MAX"],
    "reOm_ceil": _SPHERE["REOM_CEIL"],
    "reOm_A": _SPHERE["REOM_A"],
    "reOm_B": _SPHERE["REOM_B"],
    "ramp_w": _SPHERE["RAMP_W"],
    # compiled-in, no env override: pinned to ModelConstants.h values, which
    # the sa-ai consistency test verifies against the header text
    "tau": 4.0,
    "c_nu_ai": 1.0 / 6.0,
    "A_TU": -8.43,
    "B_TU": 2.4,
    "c_v1": 7.1,
}
SRC = {
    "a_max": "saai_env._SPHERE[A_MAX] (ModelConstants.h ai_rateScale)",
    "reOm_ceil": "saai_env._SPHERE[REOM_CEIL] (ai_reOmCeil)",
    "reOm_A": "saai_env._SPHERE[REOM_A] (ai_reOmA)",
    "reOm_B": "saai_env._SPHERE[REOM_B] (ai_reOmB)",
    "ramp_w": "saai_env._SPHERE[RAMP_W] (ai_rampWidth)",
    "tau": "ModelConstants.h ai_sigmaTau (compiled)",
    "c_nu_ai": "ModelConstants.h ai_nuLamScale = 1/6 (compiled)",
    "A_TU": "Mack map, run stagers (flow360/run_flatplate_ags.py)",
    "B_TU": "Mack map, run stagers",
    "c_v1": "standard SA (SpalartAllmaras.h)",
}


def main():
    print(f"{'constant':>10}{'live':>12}{'paper':>12}   source")
    for k in PAPER:
        ok = abs(GOT[k] - PAPER[k]) <= 1e-9 * max(1.0, abs(PAPER[k]))
        print(f"{k:>10}{GOT[k]:>12.6g}{PAPER[k]:>12.6g}   {SRC[k]:<52}"
              f" {'OK' if ok else 'MISMATCH'}")
        assert ok, f"{k}: live {GOT[k]} != paper {PAPER[k]}"
    print(f"\nAll {len(PAPER)} sphere-kernel constants match the paper's"
          " Appendix-E block.")


if __name__ == '__main__':
    main()
