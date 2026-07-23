"""Single source of truth for the canonical SA-AI transition-model environment.

Every SA-AI CFD run in the paper is pinned to ONE model configuration: the
SPHERE KERNEL (model v3, compute branch explore-lambda-v):
    rate  a = a_max * clip<S_hat*g>_0^1 * S(Re_Omega / Re_Omega_crit(P)),
    Re_Omega_crit(P) = softmin_2(reOmCeil, reOmA + reOmB / P^2).
The C++ Flow360 solver reads the KERNEL constants from the process
environment, so the entire job of this module is to construct that environment
dict in exactly ONE place. (Freestream chi is the exception -- see the caveat
below.)

The numeric kernel constants are the compiled defaults of Flow360
``ModelConstants.h`` (the authoritative source for the sphere kernel), pinned
textually by ``sa-ai/tests/test_constants_consistency.py``; we export them
explicitly anyway so every case records its kernel even if the compiled
defaults later change. NOTE: ``lib/calibrate_kernel.py`` still carries the
retired v2 gate kernel (pending sphere migration) -- only its Tu<->chi_inf
Mack map (unchanged in v3) is imported here.

Freestream chi -- how it ACTUALLY reaches the solver (verified against
compute/src/Flow360Core):
    The C++ solver does NOT read any chi env var -- neither ``AI_CHI_INF`` nor
    ``AFT_CHI_INF`` occurs in the solver source. It reads chi only from the case
    JSON field ``freestream.turbulenceQuantities.modifiedTurbulentViscosityRatio``.
    That JSON field is patched at build time by ``flexfoil/rans/rans/case.py``
    (from the legacy ``AFT_CHI_INF`` env) AND directly by this driver's case
    builder (``case.py::_patch_chi_seed``). So the JSON is the real channel;
    ``AFT_CHI_INF`` is load-bearing only insofar as flexfoil.rans reads it.
    We still export ``AI_CHI_INF`` alongside for forward-compat / to match the
    paper run scripts' convention, but it is NOT read by the solver today.
"""
from __future__ import annotations

import sys
from pathlib import Path

# calibrate_kernel.py lives in the repro-local lib/ (self-contained package;
# textual consistency with the project original is enforced by
# sa-ai/tests/test_constants_consistency.py).
_REPRO = Path(__file__).resolve().parent.parent
if str(_REPRO) not in sys.path:
    sys.path.insert(0, str(_REPRO))

from lib import calibrate_kernel as _kernel  # noqa: E402  (Mack Tu<->chi map only; see docstring)

# Re-export the Tu -> chi_inf map so callers use ONE definition (Mack 1977 e^N).
chi_inf_from_Tu_pct = _kernel.chi_inf_from_Tu_pct
Tu_pct_from_chi_inf = _kernel.Tu_pct_from_chi_inf

# Sphere-kernel constants = the compiled defaults of Flow360 ModelConstants.h
# (explore-lambda-v). Pinned against that header by
# sa-ai/tests/test_constants_consistency.py::CANON -- edit BOTH together.
_SPHERE = {
    "A_MAX":     0.19,    # Michalke free-shear (tanh-layer) eigenvalue
    "REOM_CEIL": 1851.2,  # k*2600 (whole-equation drain compensation, k=0.712, c=1/6)
    "REOM_A":    124.6,   # k*175
    "REOM_B":    1.424,   # k*2
    "RAMP_W":    0.35,    # onset tanh ramp width scale
}


def canonical_ai_env() -> dict[str, str]:
    """The canonical SA-AI (sphere kernel) model env.

    These are the model constants that define the transition kernel; they are
    identical for every SA-AI run in the paper. Per-case quantities (chi_inf,
    laminar slowdown) are added separately by ``canonical_env()``.

    They coincide with the compiled ModelConstants.h defaults, so exporting
    them is redundant TODAY -- it is done anyway so the solver log's resolved-
    constants echo and the case env both prove which kernel a run used, and so
    a future default change cannot silently reinterpret old case dirs.

    The v2 gate env vars (AI_VG_GATE*, AI_GCRIT, AI_SIGMOIDSLOPE,
    AI_REOMEGA_FLOOR, AI_CLIFF_LAMBDA_SLOPE, AI_FPG_RATE_SLOPE) are DEAD in the
    sphere kernel (qGate is hardcoded 1.0; no lambda_p anywhere) and are no
    longer exported.
    """
    return {
        "AI_SA": "1",                              # SA-AI model ON
        "AI_RATESCALE": repr(_SPHERE["A_MAX"]),    # 0.19
        "AI_REOMC_CEIL": repr(_SPHERE["REOM_CEIL"]),  # 1851.2
        "AI_REOMC_A": repr(_SPHERE["REOM_A"]),     # 124.6
        "AI_REOMC_B": repr(_SPHERE["REOM_B"]),     # 1.424
        "AI_RAMPWIDTH": repr(_SPHERE["RAMP_W"]),   # 0.35
    }


def canonical_env(chi_inf: float, *, laminar_slowdown: float | None = None) -> dict[str, str]:
    """Full per-case SA-AI env: the model constants + freestream chi + slowdown.

    ``chi_inf`` is the freestream SA modified-viscosity ratio seed (nuHat/nu).
    The solver reads it ONLY from the case JSON (patched by the case builder and
    by flexfoil.rans.case from ``AFT_CHI_INF``). ``AI_CHI_INF`` is exported for
    forward-compat but is NOT read by the solver -- see the module docstring.

    ``laminar_slowdown`` (if given) sets ``AI_LAMINAR_SLOWDOWN``, which damps the
    natural-transition limit cycle. The paper uses 0.01 for the harder cases
    (NLF, Eppler, flat plate). NOTE the paired convention used in the paper's
    build scripts: when AI_LAMINAR_SLOWDOWN < 1, the freestream BC seed written
    into the case JSON is pre-multiplied by that same factor, so the effective
    chi the model sees is ``seed / slowdown`` = the requested ``chi_inf``. This
    driver keeps the two consistent: pass the physical ``chi_inf`` here, and the
    case builder applies the slowdown compensation to the JSON seed.
    """
    env = canonical_ai_env()
    env["AI_SIGMAD_TIE"] = "1"                  # canonical destruction tie (paper Sec. III.E); C++ default is also on
    env["AFT_CHI_INF"] = repr(float(chi_inf))   # LOAD-BEARING: flexfoil.rans.case patches JSON seed from this
    env["AI_CHI_INF"] = repr(float(chi_inf))    # forward-compat only; solver does NOT read a chi env var
    if laminar_slowdown is not None:
        env["AI_LAMINAR_SLOWDOWN"] = repr(float(laminar_slowdown))
    return env


def classical_sa_env(chi_inf: float = 3.0) -> dict[str, str]:
    """Fully-turbulent CLASSICAL-SA baseline env (AI_SA=0), for the paper's
    ``*_turb_*`` polar baselines (run_turb_baselines.py). No transition kernel,
    no laminar slowdown; the SA freestream seed is a turbulent nuTilde/nu ratio
    (the paper uses chi_inf=3, the TMR standard). The seed reaches the solver via
    the case JSON (patched from AFT_CHI_INF); AI_CHI_INF is forward-compat only.
    """
    return {
        "AI_SA": "0",                            # SA-AI kernel OFF -> classical SA
        "AI_CHI_INF": repr(float(chi_inf)),
        "AFT_CHI_INF": repr(float(chi_inf)),
    }


if __name__ == "__main__":
    import json
    print("canonical SA-AI model env (VARIANT='ai'):")
    print(json.dumps(canonical_ai_env(), indent=2))
    print("\nexample per-case env (Tu=0.08% -> chi_inf, slowdown=0.01):")
    chi = chi_inf_from_Tu_pct(0.08)
    print(f"  chi_inf(Tu=0.08%) = {chi:.4e}")
    print(json.dumps(canonical_env(chi, laminar_slowdown=0.01), indent=2))
