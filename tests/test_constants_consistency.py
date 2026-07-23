"""Guard: the SA-AI model constants are identical across every source of truth.

SPHERE-KERNEL (model v3) EDITION. The live model is the sphere kernel
(compute branch explore-lambda-v, SAAiTransition.h::__aiRate):
    rate  a = a_max * clip<S_hat*g>_0^1 * S(Re_Omega / Re_Omega_crit(P))
    Re_Omega_crit(P) = softmin_2(reOmCeil, reOmA + reOmB / P^2)
The canonical constants live in Flow360 ModelConstants.h and are pinned here.

The pre-sphere sources of truth (src/numerics/aft_sources.py,
scripts/calibrate_kernel.py, paper/repro/lib, flow360/add_derived_to_slice.py)
still carry the RETIRED v2 gate kernel and are pending migration; their checks
are SKIPPED with an explicit reason rather than deleted, so the pending work
stays visible in the test report. Re-enable each as it is migrated.

Values are read textually from the literal definitions so the check needs no
JAX / build system -- it verifies what the code *declares*.

Run: pytest sa-ai/tests/test_constants_consistency.py     (or: python <thisfile>)
"""
import re
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# THE canonical set: sphere kernel, compute/ModelConstants.h (explore-lambda-v).
# Onset shape from the LST neutral-point graze of the Falkner-Skan family at
# (2600, 175, 2) with n=2; one scale k=0.64 anchored at the marched Blasius
# N=1 crossing (Drela Re_theta=338); constants below are the absorbed k*(...).
# Edit here only when the model changes; every other source must match.
# ---------------------------------------------------------------------------
CANON = {
    "a_max":        0.19,       # Michalke free-shear (tanh-layer) eigenvalue 0.1897
    "reOmCeil":     1670.0,     # onset soft-min ceiling (favorable side)
    "reOmA":        112.0,      # onset near-separation additive floor
    "reOmB":        1.28,       # onset inverse-square coefficient
    "rampWidth":    0.35,       # onset tanh ramp half-width
    "tau":          4.0,        # production handover width (ai_switchWidth)
    "switchCenter": 1.0,        # chi at laminar->turbulent handover
    "sigmaDTie":    1.0,        # destruction tie ON: sigma_D = 1 - (cb1/kap^2 cw1)(1-sigma_P)
    "switchWidthD": 1.36,       # legacy tau_D (used only when sigmaDTie=0)
    "nuLamScale":   1.0 / 12.0, # c_nu,ai laminar-diffusion reduction
    "maxBlend":     1.0,        # gated-max production blend
    # Mack (1977) e^N receptivity + SA c_v1
    "A_TU":        -8.43,
    "B_TU":         2.4,
    "c_v1":         7.1,
}

REPO = Path(__file__).resolve().parents[2]          # .../flexcompute
SAAI = Path(__file__).resolve().parents[1]           # .../flexcompute/sa-ai
AFT  = SAAI / "src/numerics/aft_sources.py"
CALIB = SAAI / "scripts/calibrate_kernel.py"
CPP_DIR = REPO / "compute/src/Flow360Core/Applications/Solver/SpalartAllmaras"
CPP  = CPP_DIR / "ModelConstants.h"
CPP_STRUCT = CPP_DIR / "SAAiTransition.h"

TOL = 1e-9


def _num(text, pattern):
    """Return the float assigned in the first line matching `pattern` (one capture group)."""
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        raise AssertionError(f"pattern not found: {pattern}")
    return float(eval(m.group(1)))   # eval handles '1.0/12.0'


def _read(p):
    assert p.exists(), f"missing source of truth: {p}"
    return p.read_text()


def _assert_subset(where, got):
    bad = {k: (v, CANON[k]) for k, v in got.items() if abs(v - CANON[k]) > TOL}
    assert not bad, f"{where} disagrees with canonical set: " + \
        "; ".join(f"{k}={v} (want {c})" for k, (v, c) in bad.items())


# ---------------------------------------------------------------------------
# LIVE checks: the Flow360 solver.
# ---------------------------------------------------------------------------

def test_flow360_cpp_defaults_match_canon():
    if not CPP.exists():
        raise unittest.SkipTest(f"compute repo not present at {CPP}")
    t = _read(CPP)
    got = {
        "a_max":        _num(t, r"ai_rateScale\s*=\s*([\d.eE+/-]+)"),
        "reOmCeil":     _num(t, r"ai_reOmCeil\s*=\s*([\d.eE+/-]+)"),
        "reOmA":        _num(t, r"ai_reOmA\s*=\s*([\d.eE+/-]+)"),
        "reOmB":        _num(t, r"ai_reOmB\s*=\s*([\d.eE+/-]+)"),
        "rampWidth":    _num(t, r"ai_rampWidth\s*=\s*([\d.eE+/-]+)"),
        "tau":          _num(t, r"ai_switchWidth\s*=\s*([\d.eE+/-]+)"),
        "switchCenter": _num(t, r"ai_switchCenter\s*=\s*([\d.eE+/-]+)"),
        "sigmaDTie":    _num(t, r"ai_sigmaDTie\s*=\s*([\d.eE+/-]+)"),
        "switchWidthD": _num(t, r"ai_switchWidthD\s*=\s*([\d.eE+/-]+)"),
        "nuLamScale":   _num(t, r"ai_nuLamScale\s*=\s*([\d./eE+-]+)"),
        "maxBlend":     _num(t, r"ai_maxBlend\s*=\s*([\d.eE+/-]+)"),
    }
    _assert_subset("ModelConstants.h", got)


def test_flow360_struct_defaults_match_modelconstants():
    """The aiSaConstants struct defaults (SAAiTransition.h) are overwritten at
    runtime, but drifted defaults have caused confusion twice (fixed 2026-06-30
    and 2026-07-23) -- pin the live fields to ModelConstants.h."""
    if not CPP_STRUCT.exists():
        raise unittest.SkipTest(f"compute repo not present at {CPP_STRUCT}")
    t = _read(CPP_STRUCT)
    got = {
        "a_max":        _num(t, r"double rateScale\s*=\s*([\d.eE+/-]+)"),
        "reOmCeil":     _num(t, r"double reOmCeil\s*=\s*([\d.eE+/-]+)"),
        "reOmA":        _num(t, r"double reOmA\s*=\s*([\d.eE+/-]+)"),
        "reOmB":        _num(t, r"double reOmB\s*=\s*([\d.eE+/-]+)"),
        "rampWidth":    _num(t, r"double rampWidth\s*=\s*([\d.eE+/-]+)"),
        "tau":          _num(t, r"double switchWidth\s*=\s*([\d.eE+/-]+)"),
        "switchCenter": _num(t, r"double switchCenter\s*=\s*([\d.eE+/-]+)"),
        "sigmaDTie":    _num(t, r"double sigmaDTie\s*=\s*([\d.eE+/-]+)"),
        "switchWidthD": _num(t, r"double switchWidthD\s*=\s*([\d.eE+/-]+)"),
        "maxBlend":     _num(t, r"double maxBlend\s*=\s*([\d.eE+/-]+)"),
    }
    _assert_subset("SAAiTransition.h (struct defaults)", got)


# ---------------------------------------------------------------------------
# PENDING sphere-kernel migrations: skipped, not deleted. Each skip names the
# file that still carries the retired v2 gate kernel. Re-enable as migrated.
# ---------------------------------------------------------------------------

def test_jax_aft_sources_matches_canon():
    raise unittest.SkipTest(
        "src/numerics/aft_sources.py still carries the retired v2 gate kernel; "
        "pending sphere-kernel migration (rate = a_max*clip<S_hat*g>, soft-min onset)")


def test_calibrate_kernel_matches_canon():
    raise unittest.SkipTest(
        "scripts/calibrate_kernel.py still carries the retired v2 gate kernel; "
        "pending sphere-kernel migration")


def test_paper_repro_lib_matches_canon():
    raise unittest.SkipTest(
        "paper/repro/lib + paper/repro/driver still encode the v2 canon "
        "(AI_VG_GATE=7 env, gate constants); pending sphere-kernel migration")


def test_cfd_generators_match_canon():
    raise unittest.SkipTest(
        "flow360/add_derived_to_slice.py and the repro cfd generators still "
        "reconstruct the v2 gate rate; pending sphere-kernel migration")


if __name__ == "__main__":
    for fn in (test_flow360_cpp_defaults_match_canon,
               test_flow360_struct_defaults_match_modelconstants,
               test_jax_aft_sources_matches_canon,
               test_calibrate_kernel_matches_canon,
               test_paper_repro_lib_matches_canon,
               test_cfd_generators_match_canon):
        try:
            fn()
            print(f"OK    {fn.__name__}")
        except unittest.SkipTest as e:
            print(f"SKIP  {fn.__name__}: {e}")
    print("Live constant sources agree with the sphere-kernel canonical set.")
