"""Single source of truth for the canonical SA-AI transition-model environment.

Every SA-AI CFD run in the paper is pinned to ONE model configuration: the
"ai" variant (final, ring-averaged compact-Laplacian gate ``AI_VG_GATE=4``) from
``sa-ai/flow360/run_vg_all.py``. The C++ Flow360 solver reads the KERNEL
constants (AI_RATESCALE, AI_GCRIT, ...) from the process environment, so the
entire job of this module is to construct that environment dict in exactly ONE
place. (Freestream chi is the exception -- see the caveat below.)

The numeric kernel constants are IMPORTED from
``sa-ai/scripts/calibrate_kernel.py`` (the paper's single source of truth,
identical to Flow360 ``ModelConstants.h`` and the JAX
``src/numerics/aft_sources.py``). We do NOT re-type them here -- if the kernel
is recalibrated, editing ``calibrate_kernel.py`` flows through automatically.

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

# calibrate_kernel.py lives in sa-ai/scripts. Make it importable without
# requiring the caller to have set up sys.path (config/cases must import
# standalone). driver/ -> repro/ -> sa-ai/ -> scripts/
_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import calibrate_kernel as _kernel  # noqa: E402  (single source of truth)

# Re-export the Tu -> chi_inf map so callers use ONE definition (Mack 1977 e^N).
chi_inf_from_Tu_pct = _kernel.chi_inf_from_Tu_pct
Tu_pct_from_chi_inf = _kernel.Tu_pct_from_chi_inf


def canonical_ai_env() -> dict[str, str]:
    """The canonical SA-AI (VARIANT='a3') model env, built from calibrate_kernel.

    These are the model constants that define the transition kernel; they are
    identical for every SA-AI run in the paper. Per-case quantities (chi_inf,
    laminar slowdown) are added separately by ``canonical_env()``.

    Mapping to calibrate_kernel constants:
        AI_RATESCALE       <- A_MAX          (Michalke free-shear ceiling a_max)
        AI_GCRIT           <- SIGMOID_CENTER (attached-asymptote center g_c)
        AI_SIGMOIDSLOPE    <- SIGMOID_SLOPE  (sigmoid slope s)
        AI_REOMEGA_FLOOR   <- RE_OMEGA_FLOOR (cliff floor)
        AI_CLIFF_LAMBDA_SLOPE <- K_LAMBDA    (favorable-PG onset-delay slope)
    Fixed structural switches (not calibrated numbers): AI_SA=1 turns the model
    on; AI_VG_GATE=4 selects the final ring-averaged compact-Laplacian gate.
    """
    return {
        "AI_SA": "1",                                        # SA-AI model ON
        "AI_VG_GATE": "4",                                   # ring-averaged compact-Laplacian gate (final)
        "AI_RATESCALE": repr(_kernel.A_MAX),                 # 0.19
        "AI_GCRIT": repr(_kernel.SIGMOID_CENTER),            # 1.005
        "AI_SIGMOIDSLOPE": repr(_kernel.SIGMOID_SLOPE),      # 11.0
        "AI_REOMEGA_FLOOR": repr(_kernel.RE_OMEGA_FLOOR),    # 254.0
        "AI_CLIFF_LAMBDA_SLOPE": repr(_kernel.K_LAMBDA),     # 6.1
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
