"""Build a solver-ready Flow360 case directory for an SA-AI CaseConfig.

This is a thin facade over ``flexfoil.rans``:

  * airfoil (cavity-meshed) cases  -> ``rans.pipeline.run(..., solve=False)``
    runs contours -> mesh -> Flow360.json.
  * pre-meshed cases (flat plate)  -> clone an existing base case dir, so we
    reuse the exact reference mesh from the paper's flow360 suite.

In both paths we then patch the freestream turbulence seed in Flow360.json to
the SA-AI chi_inf (with the laminar-slowdown compensation), so the built case is
already pinned to the canonical model before ``run.py`` solves it.

The freestream chi is ALSO exported to the environment (AI_CHI_INF / AFT_CHI_INF)
before calling the rans pipeline, because ``rans.case.preprocess`` reads
AFT_CHI_INF to seed the interior field at build time. We keep the JSON patch as
well so the seed is correct even when a case is cloned rather than freshly built.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from config import CaseConfig
from saai_env import canonical_env

_FLOW360 = Path(__file__).resolve().parent.parent.parent / "flow360"

# State/output files we must NOT copy when cloning a pre-meshed base case.
_SKIP_CLONE = (".pvtu", ".vtu", ".pvd", ".gltf", ".log", ".sock")


def _is_output(f: str) -> bool:
    return (any(f.endswith(s) for s in _SKIP_CLONE) or f.startswith("ipc")
            or f.endswith("_v2.csv") or f in ("progress.csv", "xtr_history.csv")
            or f in ("restartOutput", "restart.json", "restart_rank_1_of_1.dmp")
            or f.startswith("surface_forces") or f.startswith("autovis"))


def _seed_for_json(cfg: CaseConfig) -> float:
    """Freestream nuHat/nu ratio to WRITE INTO Flow360.json.

    Paper convention: when AI_LAMINAR_SLOWDOWN=s<1, the BC seed is pre-multiplied
    by s so the effective chi the model sees is seed/s = the physical chi_inf.
    So the JSON seed = chi_inf * slowdown.
    """
    chi = cfg.resolved_chi_inf()
    s = cfg.laminar_slowdown
    return chi * s if (s is not None and s < 1.0) else chi


def _patch_chi_seed(case_dir: Path, seed: float) -> None:
    """Set the freestream + farfield-boundary turbulence seed in Flow360.json."""
    p = case_dir / "Flow360.json"
    d = json.loads(p.read_text())
    d.setdefault("freestream", {})["turbulenceQuantities"] = {
        "modelType": "ModifiedTurbulentViscosityRatio",
        "modifiedTurbulentViscosityRatio": seed}
    for bn, bc in d.get("boundaries", {}).items():
        if "farfield" in bn or bc.get("type") == "Freestream":
            bc["turbulenceQuantities"] = {
                "modelType": "ModifiedTurbulentViscosityRatio",
                "modifiedTurbulentViscosityRatio": seed}
    d.setdefault("runControl", {})["restart"] = False
    p.write_text(json.dumps(d, indent=1))


def build(cfg: CaseConfig, outdir: str | Path) -> Path:
    """Build the Flow360 case for ``cfg`` under ``outdir``; return the case dir.

    Does NOT solve (that is run.py's job). Airfoil cases are meshed fresh via the
    rans pipeline; pre-meshed cases (base_case_dir set) are cloned.
    """
    outdir = Path(outdir)
    seed = _seed_for_json(cfg)

    if cfg.base_case_dir is not None:
        # ---- clone a pre-meshed reference case (flat plate / structured grid) ----
        src = _FLOW360 / cfg.base_case_dir
        if not src.is_dir():
            raise FileNotFoundError(f"base_case_dir not found: {src}")
        outdir.mkdir(parents=True, exist_ok=True)
        for f in os.listdir(src):
            if _is_output(f) or f == "ipc_data":
                continue
            sp = src / f
            (shutil.copy2 if sp.is_file() else shutil.copytree)(sp, outdir / f)
        _patch_chi_seed(outdir, seed)
        return outdir

    # ---- fresh cavity mesh + case build via the proven rans pipeline ----
    # Export chi so rans.case.preprocess seeds the interior field at build time.
    # AFT_CHI_INF is the load-bearing one (rans.case reads it and patches the case
    # JSON, which is the ONLY channel the solver reads chi from). AI_CHI_INF is
    # exported for forward-compat only; the solver does not read a chi env var.
    os.environ["AFT_CHI_INF"] = repr(seed)
    os.environ["AI_CHI_INF"] = repr(seed)

    from rans.pipeline import run as pipe_run
    rcfg = cfg.to_rans_config()
    # pipeline.run reads a JSON path; dump the lowered rans config next to output.
    outdir.mkdir(parents=True, exist_ok=True)
    cfg_json = outdir / "_rans_config.json"
    rcfg.dump(cfg_json)
    pipe_run(cfg_json, outdir, solve=False)
    _patch_chi_seed(outdir, seed)
    return outdir
