"""The paper's SA-AI CFD case matrix, in ONE place -- EXACT to the figures/tables.

Each entry is a ``config.CaseConfig``. This matrix was reconciled against the
paper's figure-regeneration scripts in ``sa-ai/paper/`` and the corresponding
Flow360 case directories in ``sa-ai/flow360_g4/`` (NOT ``flow360/`` -- see the
"base directory" note below). Every case here is one CFD solve the paper's
figures or tables actually read.

IMPORTANT -- base directory
    The paper's *final* figures are generated from ``sa-ai/flow360_g4/`` by:
        flat plate : paper/regen_flatplate_flow360.py   (B = .../flow360_g4)
        NLF0416    : paper/regen_nlf_v2.py, regen_nlf_polar.py
        Eppler387  : paper/regen_eppler_v2.py, regen_epp_L1compare.py,
                     regen_resweep_table.py, regen_epp_reattach.py
    All read ``B = "/home/qiqi/flexcompute/sa-ai/flow360_g4"``.
    (The older ``sa-ai/flow360/`` tree holds superseded ablation variants
    -- *_ai/_vg/_q4/_tauD suffixes, run10k_*, sweep_*_ai -- that no paper figure
    references. ``paper/regen_figs.py`` targets flow360/run10k_* and NACA0012 but
    writes figure names (naca0012_cf.pdf, polar.pdf, ...) that do NOT appear in
    sa-ai.tex; it is superseded. NACA0012 is therefore NOT a paper CFD case.)

Because the shipped Flow360 cases are already meshed, EVERY paper case here is
run by cloning its pre-meshed ``base_case_dir`` (absolute path into flow360_g4)
and patching the freestream chi seed -- none are re-meshed from a base JSON.
``config.CaseConfig`` resolves ``base_case_dir`` relative to ``flow360/``, but an
ABSOLUTE base_case_dir overrides that root (pathlib ``Path('/x') / '/abs'`` ==
'/abs'), so pointing at flow360_g4 needs no change to config.py / case.py.

Four families (see the paper sections around sa-ai.tex L1190-1978):
  A. flat-plate natural-transition Tu sweep   (Sec. flatplate; Fig flatplate_batch)
       Tu = {0.04,0.08,0.16,0.30,0.60}%, M=0.1, Re_unit=1e6, slowdown=0.01;
       seed = chi_inf_from_Tu_pct(Tu).  5 cases.
  B. NLF(1)-0416 refinement ladder            (Sec. nlfval; Figs nlf_cf_*, nlf_polar)
       Re=4e6, M=0.1, chi_inf=c_v1*e^-9=8.76e-4, alpha={0,4,9,15},
       {cav,str} x {L0,L1,L2} = 24 SA-AI cases, + 4 fully-turbulent SA
       baselines (strL2, alpha sweep). 28 cases.
  C. Eppler 387 refinement ladder, Re=2e5      (Sec. eppler; Figs eppler_cf_*, eppler_polar)
       Re=2e5, M=0.1, chi_inf=8.76e-4, alpha={0,2,5,7},
       {cav,str} x {L0,L1,L2} = 24 SA-AI cases, + 4 fully-turbulent SA
       baselines (strL2). 28 cases.
  D. Eppler 387 Reynolds sweep, alpha=5        (Sec. eppler resweep; Figs eppler_L1compare_*,
                                                Table eppresweep, Table eppxtr)
       alpha=5, M=0.1, chi_inf=8.76e-4, L1 both meshes, Re={60,100,300,460}k.
       The Re=200k point of the sweep REUSES the Group-C cavL1/strL1 a5 cases.
       8 NEW cases (Re != 200k).

Model env (all SA-AI cases): the canonical 'fr' variant from run_vg_all.py, which
is exactly ``saai_env.canonical_ai_env()`` (AI_SA=1, AI_VG_GATE=4,
AI_RATESCALE=0.19, AI_GCRIT=1.005, AI_SIGMOIDSLOPE=11.0, AI_REOMEGA_FLOOR=254.0,
AI_CLIFF_LAMBDA_SLOPE=6.1, AI_FPG_RATE_SLOPE=5.5) + per-case
AI_CHI_INF/AFT_CHI_INF + AI_LAMINAR_SLOWDOWN.
Fully-turbulent baselines run CLASSICAL SA: AI_SA=0, chi_inf=3.0, no slowdown
(see run_turb_baselines.py); flagged here with ``turbulent=True``.

Distinct paper CFD cases: 5 + 28 + 28 + 8 = 69.
"""
from __future__ import annotations

from config import CaseConfig

# Absolute root of the shipped, pre-meshed paper cases that the FINAL figures use.
_AI = "/home/qiqi/flexcompute/sa-ai/flow360_g4"

# Anchor seeds (from calibrate_kernel / run_vg_all.py / run_turb_baselines.py):
_CHI_EN9 = 8.76e-4   # c_v1 * exp(-9); the N_crit=9 anchor (NLF + Eppler)
_CHI_TURB = 3.0      # classical-SA turbulent freestream (TMR ~3); baselines only


# ---- A. flat-plate natural-transition Tu sweep -----------------------------
# regen_flatplate_flow360.py: TU_LIST=[0.04,0.08,0.16,0.30,0.60], MACH=0.1,
# NU=1e-6 (Re_unit=1e6); dirs flatplate_ags_Tu{int(round(Tu*1000)):04d}.
# run_flatplate_ags.py: AI_SA=1, AI_LAMINAR_SLOWDOWN=0.01, BC seed =
# chi_inf_from_Tu_pct(Tu)*0.01 (the driver applies the *0.01 slowdown
# compensation itself in case.py::_seed_for_json, so we pass the physical
# chi_inf via Tu_pct here).
_FLATPLATE_TU = [0.04, 0.08, 0.16, 0.30, 0.60]


def _flatplate(Tu: float) -> CaseConfig:
    tag = f"{int(round(Tu * 1000)):04d}"
    return CaseConfig(
        name=f"flatplate_ags_Tu{tag}",
        geometry="flatplate",
        mach=0.1,
        reynolds=1.0e6,          # unit Reynolds number; Re_x = x*1e6
        Tu_pct=Tu,               # -> chi_inf via Mack map (saai_env)
        laminar_slowdown=0.01,
        base_case_dir=f"{_AI}/flatplate_ags_Tu{tag}",
    )


# ---- B. NLF(1)-0416 refinement ladder, Re=4e6 ------------------------------
# regen_nlf_v2.py / regen_nlf_polar.py: case_dir = {mesh}{level}prop_nlf0416_Re4M_a{a}
# with mesh in {cav,str}, level in {L0,L1,L2}, alpha in {0,4,9,15}.
# sa-ai.tex L1296-1304: Re=4e6, M=0.1, chi_inf=c_v1*e^-9~8.76e-4, "twenty-four
# cases total". Fully-turbulent SA baselines: strL2prop_nlf0416_Re4M_turb_a{a}
# (run_turb_baselines.py: AI_SA=0, chi=3, alpha {0,4,9,15}).
_NLF_ALPHAS = [0, 4, 9, 15]
_MESHES = ["cav", "str"]
_LEVELS = ["L0", "L1", "L2"]


def _nlf0416(mesh: str, level: str, alpha: int) -> CaseConfig:
    return CaseConfig(
        name=f"{mesh}{level}prop_nlf0416_Re4M_a{alpha}",
        geometry="nlf0416",
        reynolds=4.0e6,
        mach=0.1,
        alpha_deg=float(alpha),
        chi_inf=_CHI_EN9,
        laminar_slowdown=0.01,
        base_case_dir=f"{_AI}/{mesh}{level}prop_nlf0416_Re4M_a{alpha}",
    )


def _nlf0416_turb(alpha: int) -> CaseConfig:
    return CaseConfig(
        name=f"strL2prop_nlf0416_Re4M_turb_a{alpha}",
        geometry="nlf0416",
        reynolds=4.0e6,
        mach=0.1,
        alpha_deg=float(alpha),
        chi_inf=_CHI_TURB,        # classical turbulent freestream
        laminar_slowdown=None,    # no slowdown for the turbulent baseline
        turbulent=True,           # -> AI_SA=0 (see run.py)
        base_case_dir=f"{_AI}/strL2prop_nlf0416_Re4M_turb_a{alpha}",
    )


# ---- C. Eppler 387 refinement ladder, Re=2e5 -------------------------------
# regen_eppler_v2.py: case_dir = {mesh}{level}prop_eppler387_Re200k_a{a},
# mesh {cav,str}, level {L0,L1,L2}, alpha {0,2,5,7} (low={0,2}, high={5,7},
# polar={0,2,5,7}). sa-ai.tex L1656-1657: Re=2e5, M=0.1. chi_inf=8.76e-4 (same
# N_crit=9 anchor as NLF). Fully-turbulent baselines strL2prop_eppler387_Re200k_turb_a{a}.
_EPP_ALPHAS = [0, 2, 5, 7]


def _eppler387(mesh: str, level: str, alpha: int) -> CaseConfig:
    return CaseConfig(
        name=f"{mesh}{level}prop_eppler387_Re200k_a{alpha}",
        geometry="eppler387",
        reynolds=2.0e5,
        mach=0.1,
        alpha_deg=float(alpha),
        chi_inf=_CHI_EN9,
        laminar_slowdown=0.01,
        base_case_dir=f"{_AI}/{mesh}{level}prop_eppler387_Re200k_a{alpha}",
    )


def _eppler387_turb(alpha: int) -> CaseConfig:
    return CaseConfig(
        name=f"strL2prop_eppler387_Re200k_turb_a{alpha}",
        geometry="eppler387",
        reynolds=2.0e5,
        mach=0.1,
        alpha_deg=float(alpha),
        chi_inf=_CHI_TURB,
        laminar_slowdown=None,
        turbulent=True,
        base_case_dir=f"{_AI}/strL2prop_eppler387_Re200k_turb_a{alpha}",
    )


# ---- D. Eppler 387 Reynolds sweep, alpha=5, L1 -----------------------------
# regen_epp_L1compare.py / regen_resweep_table.py:
#   cav L1: cav_dir(Rk) = cavL1prop_eppler387_Re200k_a5 (Rk==200) else sweep_Re{Rk}k_a5
#   str L1: str_dir(Rk) = strL1prop_eppler387_Re200k_a5 (Rk==200) else sweep_str_Re{Rk}k_a5
# Re list = {60,100,200,300,460}k; the 200k point is the Group-C cavL1/strL1 a5
# case (NOT re-listed). Only the 4 off-200k Reynolds numbers are NEW cases here
# (x 2 meshes = 8). alpha=5, M=0.1, chi_inf=8.76e-4, slowdown=0.01.
# NU (regen scripts): 60k->1.6667e-6, 100k->1e-6, 300k->3.3333e-7, 460k->2.1739e-7,
# consistent with muRef=M/Re at M=0.1 (=> M=0.1 confirmed for the whole sweep).
_EPP_SWEEP_RE = [60, 100, 300, 460]   # 200k intentionally omitted (== Group C)


def _eppler387_resweep(mesh: str, Rk: int) -> CaseConfig:
    dirname = f"sweep_Re{Rk}k_a5" if mesh == "cav" else f"sweep_str_Re{Rk}k_a5"
    return CaseConfig(
        name=dirname,
        geometry="eppler387",
        reynolds=Rk * 1.0e3,
        mach=0.1,
        alpha_deg=5.0,
        chi_inf=_CHI_EN9,
        laminar_slowdown=0.01,
        base_case_dir=f"{_AI}/{dirname}",
    )


# ---- assembled matrix ------------------------------------------------------
def _build_matrix() -> dict[str, CaseConfig]:
    m: dict[str, CaseConfig] = {}

    def add(c: CaseConfig) -> None:
        m[c.name] = c

    # A. flat plate (5)
    for Tu in _FLATPLATE_TU:
        add(_flatplate(Tu))

    # B. NLF0416 SA-AI ladder (24) + turbulent baselines (4)
    for mesh in _MESHES:
        for level in _LEVELS:
            for a in _NLF_ALPHAS:
                add(_nlf0416(mesh, level, a))
    for a in _NLF_ALPHAS:
        add(_nlf0416_turb(a))

    # C. Eppler387 SA-AI ladder (24) + turbulent baselines (4)
    for mesh in _MESHES:
        for level in _LEVELS:
            for a in _EPP_ALPHAS:
                add(_eppler387(mesh, level, a))
    for a in _EPP_ALPHAS:
        add(_eppler387_turb(a))

    # D. Eppler387 Reynolds sweep, alpha=5, L1 (8; 200k reuses Group C)
    for mesh in _MESHES:
        for Rk in _EPP_SWEEP_RE:
            add(_eppler387_resweep(mesh, Rk))

    return m


MATRIX = _build_matrix()


def all_cases() -> list[str]:
    return list(MATRIX.keys())


def get(name: str) -> CaseConfig:
    if name not in MATRIX:
        raise KeyError(f"unknown case {name!r}; see cases.all_cases()")
    return MATRIX[name]


if __name__ == "__main__":
    print(f"{len(MATRIX)} paper cases:")
    for name, c in MATRIX.items():
        tag = "turb" if getattr(c, "turbulent", False) else "ai"
        print(f"  [{tag:4s}] {name:38s} geom={c.geometry:10s} "
              f"Re={c.reynolds:.2g} M={c.mach:g} a={c.alpha_deg:+g} "
              f"chi={c.resolved_chi_inf():.3e}")
