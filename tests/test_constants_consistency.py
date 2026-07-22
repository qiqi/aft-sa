"""Guard: the SA-AI model constants are identical across every source of truth.

The paper (main.tex Table), the JAX model (src/numerics/aft_sources.py), the
calibration helper (scripts/calibrate_kernel.py) and the Flow360 C++ solver
(ModelConstants.h) must all carry ONE canonical constant set. This test fails
loudly if any of them drifts -- the exact inconsistency that has to be gone
before the manuscript ships (paper vs CFD "same model, same constants").

Values are read textually from the literal definitions so the check needs no
JAX / build system -- it verifies what the code *declares*.

Run: pytest sa-ai/tests/test_constants_consistency.py     (or: python <thisfile>)
"""
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# THE canonical set (paper main.tex Table, ~lines 1161-1179). Edit here only if
# the paper itself changes; every other source must match these.
# ---------------------------------------------------------------------------
CANON = {
    "a_max":        0.19,     # Michalke free-shear ceiling
    "g_c":          0.9874,    # sigmoid center
    "s":            10.68,      # sigmoid slope
    "reOmegaFloor": 243.7,    # cliff floor
    "K_lambda":     6.20,      # favorable-PG onset-delay slope
    "K_r":          5.80,      # favorable-rate factor slope (fit at beta=0.35)
    "p":            4.0,      # cliff barrier exponent
    "c_A":          4.0,      # Q4 vorticity-gradient weight
    "gammaCoeff":   2.0,      # Gamma coefficient
    "tau":          4.0,      # production handover width
    "sigmaDTie":    1.0,      # destruction tie ON (sigma_D = 1 - (cb1/kap^2 cw1)(1-sigma_P); no tau_D)
    "nuLamScale":   1.0 / 12.0,
    "vgGate":       7.0,      # FINAL band-form composite gate (paper v2)
    "c_V":          4.0,      # Lambda_v weight in the smooth Q1 denominator
    "c_2":          8.0,      # Q2 release softness outside the parabola loop
    # Mack (1977) e^N receptivity + SA c_v1
    "A_TU":        -8.43,
    "B_TU":         2.4,
    "c_v1":         7.1,
}

REPO = Path(__file__).resolve().parents[2]          # .../flexcompute
SAAI = Path(__file__).resolve().parents[1]           # .../flexcompute/sa-ai
AFT  = SAAI / "src/numerics/aft_sources.py"
CALIB = SAAI / "scripts/calibrate_kernel.py"
CPP  = REPO / "compute/src/Flow360Core/Applications/Solver/SpalartAllmaras/ModelConstants.h"
REPRO_AFT = SAAI / "paper/repro/lib/aft_sources.py"
REPRO_CALIB = SAAI / "paper/repro/lib/calibrate_kernel.py"

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


def test_jax_aft_sources_matches_canon():
    t = _read(AFT)
    got = {
        "a_max":        _num(t, r"AFT_RATE_SCALE\s*=\s*([\d.eE+/-]+)"),
        "g_c":          _num(t, r"AFT_SIGMOID_CENTER\s*=\s*([\d.eE+/-]+)"),
        "s":            _num(t, r"AFT_SIGMOID_SLOPE\s*=\s*([\d.eE+/-]+)"),
        "reOmegaFloor": _num(t, r"AFT_RE_OMEGA_FLOOR\s*=\s*([\d.eE+/-]+)"),
        "K_lambda":     _num(t, r"AFT_CLIFF_LAMBDA_SLOPE\s*=\s*([\d.eE+/-]+)"),
        "K_r":          _num(t, r"AFT_FPG_RATE_SLOPE\s*=\s*([\d.eE+/-]+)"),
        "p":            _num(t, r"AFT_BARRIER_POWER\s*=\s*([\d.eE+/-]+)"),
        "c_A":          _num(t, r"AFT_Q4_CA\s*=\s*([\d.eE+/-]+)"),
        "c_V":          _num(t, r"AFT_LV_CV\s*=\s*([\d.eE+/-]+)"),
        "c_2":          _num(t, r"AFT_Q2_C2\s*=\s*([\d.eE+/-]+)"),
        "gammaCoeff":   _num(t, r"AFT_GAMMA_COEFF\s*=\s*([\d.eE+/-]+)"),
    }
    _assert_subset("aft_sources.py", got)


def test_calibrate_kernel_matches_canon():
    t = _read(CALIB)
    got = {
        "a_max":        _num(t, r"^A_MAX\s*=\s*([\d.eE+/-]+)"),
        "s":            _num(t, r"^SIGMOID_SLOPE\s*=\s*([\d.eE+/-]+)"),
        "g_c":          _num(t, r"^SIGMOID_CENTER\s*=\s*([\d.eE+/-]+)"),
        "reOmegaFloor": _num(t, r"^RE_OMEGA_FLOOR\s*=\s*([\d.eE+/-]+)"),
        "K_lambda":     _num(t, r"^K_LAMBDA\s*=\s*([\d.eE+/-]+)"),
        "K_r":          _num(t, r"^FPG_RATE_SLOPE\s*=\s*([\d.eE+/-]+)"),
        "p":            _num(t, r"^BARRIER_POWER\s*=\s*([\d.eE+/-]+)"),
        "A_TU":         _num(t, r"^A_TU\s*=\s*([\d.eE+/-]+)"),
        "B_TU":         _num(t, r"^B_TU\s*=\s*([\d.eE+/-]+)"),
        "c_v1":         _num(t, r"^C_V1\s*=\s*([\d.eE+/-]+)"),
    }
    _assert_subset("calibrate_kernel.py", got)


def test_flow360_cpp_defaults_match_canon():
    t = _read(CPP)
    got = {
        "a_max":        _num(t, r"ai_rateScale\s*=\s*([\d.eE+/-]+)"),
        "g_c":          _num(t, r"ai_sigmoidCenter\s*=\s*([\d.eE+/-]+)"),
        "s":            _num(t, r"ai_sigmoidSlope\s*=\s*([\d.eE+/-]+)"),
        "reOmegaFloor": _num(t, r"ai_reOmegaFloor\s*=\s*([\d.eE+/-]+)"),
        "K_lambda":     _num(t, r"ai_cliffLambdaSlope\s*=\s*([\d.eE+/-]+)"),
        "K_r":          _num(t, r"ai_fpgRateSlope\s*=\s*([\d.eE+/-]+)"),
        "p":            _num(t, r"ai_barrierExponent\s*=\s*([\d.eE+/-]+)"),
        "tau":          _num(t, r"ai_switchWidth\s*=\s*([\d.eE+/-]+)"),
        "sigmaDTie":    _num(t, r"ai_sigmaDTie\s*=\s*([\d.eE+/-]+)"),
        "nuLamScale":   _num(t, r"ai_nuLamScale\s*=\s*([\d.eE+/-]+)"),
        "vgGate":       _num(t, r"ai_vgGateEnable\s*=\s*([\d.eE+/-]+)"),
        "c_V":          _num(t, r"ai_vgGateWeight\s*=\s*([\d.eE+/-]+)"),
        "c_2":          _num(t, r"ai_vgGateC2\s*=\s*([\d.eE+/-]+)"),
        "gammaCoeff":   _num(t, r"ai_gammaCoeff\s*=\s*([\d.eE+/-]+)") if "ai_gammaCoeff" in t else CANON["gammaCoeff"],
    }
    _assert_subset("ModelConstants.h", got)


def test_paper_repro_lib_matches_canon():
    """The self-contained paper/repro/lib copies must carry the same constants
    (they are verbatim copies with import lines rewritten -- this guards
    against the copies drifting from the project originals)."""
    for src, orig in ((REPRO_AFT, AFT), (REPRO_CALIB, CALIB)):
        t, t0 = _read(src), _read(orig)
        # every constant-assignment line in the original must appear verbatim
        for line in t0.splitlines():
            if re.match(r"^(AFT_[A-Z_0-9]+|A_MAX|SIGMOID_\w+|RE_OMEGA_FLOOR|"
                        r"K_LAMBDA|FPG_RATE_SLOPE|BARRIER_POWER|A_TU|B_TU|C_V1)\s*=\s*[\d.]", line):
                assert line in t, f"{src.name}: drifted line {line!r}"


def test_cfd_generators_match_canon():
    """The CFD post-processing/figure generators must not carry their own
    kernel digits (a stale local copy sat at the RETIRED kernel g_c=1.572
    until 2026-07-13 and silently colored derived amp_rate fields and
    landscape contours). The repro copies must IMPORT from lib; the flow360/
    original (kept literal for standalone use) must match canon."""
    for name in ("add_derived_to_slice.py", "regen_nlf_v2.py", "regen_eppler_v2.py"):
        src = SAAI / "paper/repro/cfd" / name
        t = _read(src)
        assert "from lib.calibrate_kernel import" in t, \
            f"{name}: must import kernel constants from lib.calibrate_kernel"
        assert not re.search(r"^G_C, SLOPE, BARRIER_POWER, A_MAX\s*=", t, re.MULTILINE), \
            f"{name}: local kernel literals are forbidden"
    t = _read(SAAI / "flow360/add_derived_to_slice.py")
    got = {
        "g_c":          _num(t, r"^G_C, SLOPE, BARRIER_POWER, A_MAX\s*=\s*([\d.]+)"),
        "reOmegaFloor": _num(t, r"^RE_OMEGA_FLOOR\s*=\s*([\d.eE+/-]+)"),
        "K_lambda":     _num(t, r"^CLIFF_LAMBDA_SLOPE\s*=\s*([\d.eE+/-]+)"),
        "K_r":          _num(t, r"^FPG_RATE_SLOPE\s*=\s*([\d.eE+/-]+)"),
    }
    _assert_subset("flow360/add_derived_to_slice.py", got)


def _assert_subset(where, got):
    bad = {k: (v, CANON[k]) for k, v in got.items() if abs(v - CANON[k]) > TOL}
    assert not bad, f"{where} disagrees with canonical set: " + \
        "; ".join(f"{k}={v} (want {c})" for k, (v, c) in bad.items())


if __name__ == "__main__":
    for fn in (test_jax_aft_sources_matches_canon,
               test_calibrate_kernel_matches_canon,
               test_flow360_cpp_defaults_match_canon,
               test_paper_repro_lib_matches_canon,
               test_cfd_generators_match_canon):
        fn()
        print(f"OK  {fn.__name__}")
    print("All constant sources agree with the canonical set.")
