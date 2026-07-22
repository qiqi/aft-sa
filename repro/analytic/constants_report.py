"""Canonical constant report (paper Table). Imports every SA-AI constant from its
src/ home, prints it, and asserts each equals the paper Table value. No formula or
constant is restated -- only the paper Table targets appear here, for the check."""
import _saai
from _saai import (A_MAX, G_C, S_SLOPE, FLOOR, K_LAMBDA, P, C_A,
                   TAU, TAU_D, C_NU_AI, A_TU, B_TU, C_V1)
from src.numerics.aft_sources import AFT_GAMMA_COEFF

# Paper Table (main.tex) target values -- the ONLY place numbers are written here.
PAPER = {
    "a_max": 0.19, "g_c": 1.005, "s": 11.0, "reOmegaFloor": 254.0,
    "K_lambda": 6.1, "p": 4.0, "c_A": 4.0, "gammaCoeff": 2.0,
    "tau": 4.0, "tau_D": 1.36, "c_nu_ai": 1.0/12.0,
    "A_TU": -8.43, "B_TU": 2.4, "c_v1": 7.1,
}
GOT = {
    "a_max": A_MAX, "g_c": G_C, "s": S_SLOPE, "reOmegaFloor": FLOOR,
    "K_lambda": K_LAMBDA, "p": P, "c_A": C_A, "gammaCoeff": float(AFT_GAMMA_COEFF),
    "tau": TAU, "tau_D": TAU_D, "c_nu_ai": C_NU_AI,
    "A_TU": A_TU, "B_TU": B_TU, "c_v1": C_V1,
}
SRC = {
    "a_max": "aft_sources.AFT_RATE_SCALE", "g_c": "aft_sources.AFT_SIGMOID_CENTER",
    "s": "aft_sources.AFT_SIGMOID_SLOPE", "reOmegaFloor": "aft_sources.AFT_RE_OMEGA_FLOOR",
    "K_lambda": "aft_sources.AFT_CLIFF_LAMBDA_SLOPE", "p": "aft_sources.AFT_BARRIER_POWER",
    "c_A": "aft_sources.AFT_Q4_CA", "gammaCoeff": "aft_sources.AFT_GAMMA_COEFF",
    "tau": "regen_wall_layer.TAU", "tau_D": "regen_wall_layer.TAU_D",
    "c_nu_ai": "boundary_layer_solvers.NuHatBlasiusSolver.aft_nuLamScale",
    "A_TU": "calibrate_kernel.A_TU", "B_TU": "calibrate_kernel.B_TU", "c_v1": "calibrate_kernel.C_V1",
}


def main():
    print(f"{'constant':>12}{'imported':>14}{'paper':>12}   source")
    bad = []
    for k in PAPER:
        ok = abs(GOT[k] - PAPER[k]) <= 1e-9
        flag = 'OK' if ok else 'MISMATCH'
        if not ok: bad.append(k)
        print(f"{k:>12}{GOT[k]:>14.6g}{PAPER[k]:>12.6g}   {SRC[k]:<48} {flag}")
        assert ok, f"{k}: imported {GOT[k]} != paper {PAPER[k]}"
    print(f"\nAll {len(PAPER)} canonical constants match the paper Table.")


if __name__ == '__main__':
    main()
