"""Minimal case description for an SA-AI reproduction run.

This is a thin, paper-facing wrapper. A case is described by geometry + flow +
turbulence-seed knobs; ``to_rans_config()`` lowers it to the proven
``flexfoil.rans`` ``CaseConfig`` that the meshing/solving pipeline consumes.

We deliberately reuse ``flexfoil.rans.config`` for the heavy lifting (Element /
Farfield / Flow / Mesh / Solver dataclasses) and only add the SA-AI-specific
bits on top: how the turbulence freestream is specified (Tu% via the Mack map,
or a raw chi_inf) and the laminar-slowdown convention.

The ``flexfoil.rans`` import is GUARDED so this module (and ``cases.py``) import
standalone even when the Flow360 SDK / rans package is not on the path -- the
lowering only needs the rans dataclasses, and if they are unavailable we raise a
clear error only when someone actually calls ``to_rans_config()``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from saai_env import chi_inf_from_Tu_pct

# Reference geometry JSONs shipped with the paper's flow360 suite. The airfoil
# contour + mesh knobs are read from these so we do not re-type 300-point
# contours in the driver.
_FLOW360 = Path(__file__).resolve().parent.parent.parent / "flow360"
BASE_JSONS = {
    "naca0012": _FLOW360 / "naca0012_re1m.json",
    "nlf0416": _FLOW360 / "nlf0416_re1m.json",
}


@dataclass
class CaseConfig:
    """One SA-AI reproduction case.

    Geometry is EITHER an airfoil name with a shipped base JSON (``naca0012`` /
    ``nlf0416``) OR ``flatplate`` (natural-transition flat plate, meshed by a
    pre-built case dir -- see ``base_case_dir``).

    Turbulence: give ``Tu_pct`` (freestream turbulence %, mapped to chi_inf via
    Mack 1977) OR a raw ``chi_inf``; exactly one is required.
    """
    name: str                       # human-readable case id (also the output subdir)
    geometry: str                   # "naca0012" | "nlf0416" | "eppler387" | "flatplate"

    # flow conditions
    reynolds: float = 1.0e6         # based on chord = 1
    mach: float = 0.2
    alpha_deg: float = 0.0
    temperature: float = 288.15     # K

    # turbulence freestream seed (give exactly one of Tu_pct / chi_inf)
    Tu_pct: float | None = None     # freestream turbulence intensity, %
    chi_inf: float | None = None    # raw SA modified-viscosity ratio seed (overrides Tu_pct)
    laminar_slowdown: float | None = 0.01  # AI_LAMINAR_SLOWDOWN (None -> unset)

    # Fully-turbulent classical-SA baseline (AI_SA=0). The paper's *_turb_*
    # polar baselines (run_turb_baselines.py) run standard SA with a turbulent
    # freestream (chi_inf=3) and no transition model. When True, run.py builds
    # the env with the SA-AI kernel OFF.
    turbulent: bool = False

    # mesh knobs (defaults follow the shipped base JSONs; override per case)
    max_steps: int = 5000
    yplus: float = 1.0
    far_radius: float = 50.0

    # For pre-meshed cases (flat plate, structured C-grids): clone this existing
    # Flow360 case dir instead of running the cavity mesher. Relative to flow360/.
    base_case_dir: str | None = None

    # optional overrides forwarded verbatim into flexfoil.rans mesh/solver
    rans_overrides: dict = field(default_factory=dict)

    # ---- derived ----
    def resolved_chi_inf(self) -> float:
        """The physical freestream chi_inf, from a raw value or from Tu% (Mack)."""
        if self.chi_inf is not None:
            return float(self.chi_inf)
        if self.Tu_pct is not None:
            return chi_inf_from_Tu_pct(self.Tu_pct)
        raise ValueError(f"case {self.name!r}: give one of Tu_pct or chi_inf")

    def base_json(self) -> Path | None:
        """Path to the shipped rans base JSON for airfoil geometries, else None."""
        return BASE_JSONS.get(self.geometry)

    def to_rans_config(self):
        """Lower to a ``flexfoil.rans.config.CaseConfig`` for the pipeline.

        Only valid for cavity-meshed airfoil geometries with a base JSON
        (naca0012 / nlf0416). Flat-plate / structured cases are pre-meshed and
        run by cloning ``base_case_dir`` instead (see run.py).
        """
        base = self.base_json()
        if base is None:
            raise ValueError(
                f"case {self.name!r}: geometry {self.geometry!r} has no base JSON; "
                "it must be run from a pre-meshed base_case_dir, not to_rans_config()")
        # Import here so config.py imports without the rans package present.
        from rans.config import CaseConfig as RansConfig
        cfg = RansConfig.load(base)
        cfg.flow.reynolds = self.reynolds
        cfg.flow.mach = self.mach
        cfg.flow.alpha_deg = self.alpha_deg
        cfg.flow.temperature = self.temperature
        cfg.mesh.yplus = self.yplus
        cfg.farfield.radius = self.far_radius
        cfg.solver.max_steps = self.max_steps
        for k, v in self.rans_overrides.items():
            # dotted path, e.g. "mesh.growth": 1.1
            obj, attr = self, k
            target = cfg
            *path, last = k.split(".")
            for p in path:
                target = getattr(target, p)
            setattr(target, last, v)
        return cfg
