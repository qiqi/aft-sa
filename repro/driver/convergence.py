"""The paper's convergence protocols, in one place.

A single fixed-length solve does NOT reproduce the paper's numbers on the
marginal cases. Every CFD result in the paper was produced under one of three
protocols, selected purely by case family (``protocol_for(cfg)``):

``plain``   -- flat plates and the fully-turbulent baselines: one solve of the
               case's own ``maxPseudoSteps`` budget (they converge directly).

``ladder``  -- the 48 NLF/Eppler refinement-ladder cases: adaptive batching by
               ``flow360/converge_by_xtr.py`` (the paper's tool, reused, not
               reimplemented) -- 5000-step batches, stop when the upper AND
               lower transition locations drift < 0.01c over consecutive
               batches. The minimum-batch floors exist because impulsive
               starts sit on sentinel plateaus that fake convergence; the
               marginal cases (NLF alpha=9, near-stall Eppler alpha=7, the
               breathing NLF lower surfaces) genuinely need the adaptive
               extension.  Parameters per family (from run_vg_all.py):
                 NLF    : --consec 2 --min-batches 8
                 Eppler : --consec 3 --min-batches 5
               (common: --batch 5000 --tol 0.01 --max-batches 24)

``staged``  -- the 8 Eppler Reynolds-sweep cases: the staged-fSlow protocol
               (run_vg_all.py::run_sweep). Impulsive starts at slowdown 0.01
               limit-cycle; instead run stage 1 at AI_LAMINAR_SLOWDOWN=0.1
               with the JSON seed pre-multiplied by 0.1 for 30000 steps, then
               restart stage 2 at 0.01 / seed*0.01 for 10000 more. The seed
               pre-multiplication keeps the effective freestream chi equal to
               the requested chi_inf at every stage (seed/slowdown = chi_inf).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_FLOW360 = Path(__file__).resolve().parent.parent.parent / "flow360"
_PY = sys.executable

_LADDER_ARGS = ["--batch", "5000", "--tol", "0.01", "--max-batches", "24"]
_LADDER_FAMILY = {  # geometry -> (consec, min_batches)
    "nlf0416": ("2", "8"),
    "eppler387": ("3", "5"),
}
_SWEEP_STAGES = [(0.1, 30000), (0.01, 10000)]  # (AI_LAMINAR_SLOWDOWN, extra steps)


def protocol_for(cfg) -> str:
    """plain | ladder | staged, decided by case family (see module docstring)."""
    if cfg.geometry == "flatplate" or getattr(cfg, "turbulent", False):
        return "plain"
    if cfg.name.startswith("sweep_"):
        return "staged"
    return "ladder"


def solve_to_convergence(cfg, case_dir: Path, env: dict, *, gpu: int = 0) -> str:
    """Run ``case_dir`` under the paper protocol for ``cfg``. Returns the protocol."""
    proto = protocol_for(cfg)
    if proto == "plain":
        _plain(case_dir, env, gpu)
    elif proto == "ladder":
        _ladder(case_dir, env, gpu, cfg.geometry)
    else:
        _staged_sweep(case_dir, env, float(cfg.resolved_chi_inf()), gpu)
    return proto


# ---------------------------------------------------------------------------
def _plain(case_dir: Path, env: dict, gpu: int) -> None:
    from rans.env import make_env
    from rans.solve import run_solver
    base, find = make_env()
    base.update(env)
    run_solver(case_dir, find, base, gpu=gpu, timeout=14400)


def _ladder(case_dir: Path, env: dict, gpu: int, geometry: str) -> None:
    consec, min_batches = _LADDER_FAMILY[geometry]
    child = dict(os.environ)
    child.update(env)                       # converge_by_xtr propagates os.environ
    cmd = [_PY, str(_FLOW360 / "converge_by_xtr.py"), str(case_dir),
           *_LADDER_ARGS, "--consec", consec, "--min-batches", min_batches,
           "--gpu", str(gpu)]
    r = subprocess.run(cmd, env=child, timeout=86400)
    if r.returncode not in (0, 1):          # 1 = tolerance not met at max-batches
        raise RuntimeError(f"converge_by_xtr failed (rc={r.returncode}) on {case_dir}")
    if r.returncode == 1:
        print(f"[{case_dir.name}] WARNING: xtr tolerance not met at max-batches "
              "(known breathing cases report their upper-surface value reliably)",
              flush=True)


def _set_stage(case_dir: Path, chi_inf: float, fslow: float, extra_steps: int) -> None:
    """Patch the case for one sweep stage: seed = chi_inf*fslow, restart, +steps."""
    p = case_dir / "Flow360.json"
    d = json.loads(p.read_text())
    seed = chi_inf * fslow
    d["freestream"]["turbulenceQuantities"]["modifiedTurbulentViscosityRatio"] = seed
    for bc in d.get("boundaries", {}).values():
        if "turbulenceQuantities" in bc:
            bc["turbulenceQuantities"]["modifiedTurbulentViscosityRatio"] = seed
    cur = 0
    rp = case_dir / "nonlinear_residual_v2.csv"
    if rp.exists():
        for ln in rp.read_text().splitlines():
            try:
                cur = int(float(ln.split(",")[1]))
            except (ValueError, IndexError):
                pass
    d["runControl"]["restart"] = cur > 0
    d["timeStepping"]["maxPseudoSteps"] = cur + extra_steps
    p.write_text(json.dumps(d, indent=1))
    ro = case_dir / "restartOutput"
    if d["runControl"]["restart"] and ro.is_dir():
        for f in os.listdir(ro):            # solver reads restart from case root
            shutil.copy2(ro / f, case_dir / f)


def _staged_sweep(case_dir: Path, env: dict, chi_inf: float, gpu: int) -> None:
    from rans.env import make_env
    from rans.solve import run_solver
    for fslow, steps in _SWEEP_STAGES:
        _set_stage(case_dir, chi_inf, fslow, steps)
        base, find = make_env()
        base.update(env)
        base["AI_LAMINAR_SLOWDOWN"] = repr(fslow)
        print(f"[{case_dir.name}] sweep stage fSlow={fslow} (+{steps} steps)", flush=True)
        run_solver(case_dir, find, base, gpu=gpu, timeout=14400)
