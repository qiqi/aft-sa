# SA-AI paper CFD re-run manifest

Every Flow360 CFD case the SA-AI paper's figures and tables depend on, reconciled
against `paper/*.py` (the figure regenerators) and the pre-meshed case directories
in `sa-ai/flow360_g4/`. **69 distinct cases.**

- **Do NOT re-run on GPU as part of this task.** This manifest is the plan for a
  human-supervised GPU re-run.
- All base case dirs are absolute paths under
  `/home/qiqi/flexcompute/sa-ai/flow360_g4` (the tree the *final* paper figures
  read). The older `.../flow360/` tree holds superseded ablation variants
  (`*_ai/_vg/_q4/_tauD`, `run10k_*`, `sweep_*_ai`; the `*_a3` clones are the canonical fleet, linked into `flow360_g4/`) referenced by no paper figure.
- Every case is **pre-meshed**: the driver clones its base dir and patches the
  freestream chi seed. None are re-meshed from a base JSON.
- Case name == base dir basename == `cases.py` key == output subdir.

## Canonical model environment

**SA-AI cases** (`saai_env.canonical_env(chi_inf, laminar_slowdown=...)`),
identical to the `g4` variant in `flow360/run_vg_all.py`:

| var | value |
|---|---|
| `AI_SA` | `1` |
| `AI_VG_GATE` | `4` |
| `AI_RATESCALE` | `0.19` |
| `AI_GCRIT` | `1.005` |
| `AI_SIGMOIDSLOPE` | `11.0` |
| `AI_REOMEGA_FLOOR` | `254.0` |
| `AI_CLIFF_LAMBDA_SLOPE` | `6.1` |
| `AFT_CHI_INF` | per-case `chi_inf` (load-bearing: `flexfoil.rans.case` patches the case JSON from it; the solver reads chi ONLY from the JSON) |
| `AI_CHI_INF` | same value, forward-compat only (NOT read by the solver) |
| `AI_LAMINAR_SLOWDOWN` | `0.01` (all paper SA-AI cases) |

Seed convention (`case.py::_seed_for_json`): with slowdown `s<1`, the JSON BC
seed = `chi_inf * s`; the solver sees `seed/s = chi_inf`. Matches
`run_flatplate_ags.py` (`seed = chi_inf_from_Tu_pct(Tu)*0.01`) and the NLF/Eppler
builders.

**Fully-turbulent SA baselines** (`saai_env.classical_sa_env(3.0)`,
`run_turb_baselines.py`): classical SA, no transition model:

| var | value |
|---|---|
| `AI_SA` | `0` |
| `AFT_CHI_INF` (+ `AI_CHI_INF`, forward-compat) | `3.0` |
| `AI_LAMINAR_SLOWDOWN` | (unset) |

Run: `python3 run.py <case_name> --gpu N` (in-session, non-sandboxed shell).
`run.py` selects the env automatically from `cfg.turbulent` and runs each case
under the paper's **convergence protocol** (`convergence.py`, selected by case
family -- a plain fixed-length solve does NOT reproduce the marginal cases):

| protocol | cases | procedure |
|---|---|---|
| `plain` | 5 flat plates, 8 turbulent baselines | one solve of the case's own step budget |
| `ladder` | 48 NLF/Eppler ladder cases | `flow360/converge_by_xtr.py`: 5000-step batches, xtr-drift tol 0.01c, NLF `--consec 2 --min-batches 8` / Eppler `--consec 3 --min-batches 5`, `--max-batches 24` |
| `staged` | 8 Re-sweep cases | staged-fSlow: stage 1 `AI_LAMINAR_SLOWDOWN=0.1`, JSON seed = chi*0.1, 30000 steps; stage 2 `0.01` / chi*0.01, +10000 (restart; seed/slowdown = chi at every stage) |

After the runs: assemble the figure tree and derived slice fields with
`repro/postprocess/prepare.py`, then regenerate all CFD figures/tables with
`repro/postprocess/regenerate_cfd.py` (reads `SAAI_CFD_ROOT`).

---

## Group A — Flat plate, Tu sweep (5 cases)

Feeds **Fig. `flat_plate_batch_flow360.pdf`** (`\label{fig:flatplate_batch}`),
from `paper/regen_flatplate_flow360.py` (`B=flow360_g4`).
Common: geometry `flatplate`, M=0.1, unit Reynolds (Re_x = x·10⁶), model SA-AI,
slowdown 0.01. `chi_inf` from the Mack Tu→chi map; the values below match the
paper text (sa-ai.tex L1230-1231).

| case dir (under flow360_g4/) | Tu % | chi_inf | fig |
|---|---|---|---|
| `flatplate_ags_Tu0040` | 0.04 | 2.28e-4 | flatplate_batch |
| `flatplate_ags_Tu0080` | 0.08 | 1.20e-3 | flatplate_batch |
| `flatplate_ags_Tu0160` | 0.16 | 6.34e-3 | flatplate_batch |
| `flatplate_ags_Tu0300` | 0.30 | 2.87e-2 | flatplate_batch |
| `flatplate_ags_Tu0600` | 0.60 | 1.50e-1 | flatplate_batch |

---

## Group B — NLF(1)-0416, Re=4×10⁶ refinement ladder (28 cases)

Feeds **`nlf_cf_lowalpha.pdf`, `nlf_cf_highalpha.pdf`** (`regen_nlf_v2.py`) and
**`nlf_polar_compare.pdf`** (`regen_nlf_polar.py`).
Common: geometry `nlf0416`, Re=4e6, M=0.1 (sa-ai.tex L1296).

### B1 — SA-AI ladder (24): chi_inf=8.76e-4 (= c_v1·e⁻⁹), slowdown 0.01

Dirs `{mesh}{level}prop_nlf0416_Re4M_a{alpha}`, mesh∈{cav,str}, level∈{L0,L1,L2},
alpha∈{0,4,9,15}. low-alpha fig = {0,4}; high-alpha fig = {9,15}; polar = all four.

| meshes × levels × alphas | count |
|---|---|
| {cav,str} × {L0,L1,L2} × {0,4,9,15} | 24 |

Example dirs: `cavL0prop_nlf0416_Re4M_a0` … `strL2prop_nlf0416_Re4M_a15`.

### B2 — Fully-turbulent SA baselines (4): AI_SA=0, chi_inf=3.0, no slowdown

Only the polar figure uses these (`regen_nlf_polar.py`, "SA, fully turbulent (str L2)").

| case dir | alpha | model |
|---|---|---|
| `strL2prop_nlf0416_Re4M_turb_a0` | 0 | classical SA |
| `strL2prop_nlf0416_Re4M_turb_a4` | 4 | classical SA |
| `strL2prop_nlf0416_Re4M_turb_a9` | 9 | classical SA |
| `strL2prop_nlf0416_Re4M_turb_a15` | 15 | classical SA |

---

## Group C — Eppler 387, Re=2×10⁵ refinement ladder (28 cases)

Feeds **`eppler_cf_lowalpha.pdf`, `eppler_cf_highalpha.pdf`,
`eppler_polar_compare.pdf`** (`regen_eppler_v2.py`) and **Table `tab:eppxtr`**
(reattachment, `regen_epp_reattach.py`, uses cav/str × L0/L1/L2 at alpha {0,2,5,7}).
Common: geometry `eppler387`, Re=2e5, M=0.1 (sa-ai.tex L1656).

### C1 — SA-AI ladder (24): chi_inf=8.76e-4, slowdown 0.01

Dirs `{mesh}{level}prop_eppler387_Re200k_a{alpha}`, mesh∈{cav,str},
level∈{L0,L1,L2}, alpha∈{0,2,5,7}. low-alpha fig = {0,2}; high-alpha fig = {5,7};
polar = all four. **`cavL1prop_eppler387_Re200k_a5` and
`strL1prop_eppler387_Re200k_a5` also serve as the Re=200k point of the Group-D
sweep** (Table `tab:eppresweep`, Figs `eppler_L1compare_*`).

| meshes × levels × alphas | count |
|---|---|
| {cav,str} × {L0,L1,L2} × {0,2,5,7} | 24 |

### C2 — Fully-turbulent SA baselines (4): AI_SA=0, chi_inf=3.0, no slowdown

Used by `regen_eppler_v2.py::make_polar_figure` ("SA, fully turbulent (str L2)").

| case dir | alpha | model |
|---|---|---|
| `strL2prop_eppler387_Re200k_turb_a0` | 0 | classical SA |
| `strL2prop_eppler387_Re200k_turb_a2` | 2 | classical SA |
| `strL2prop_eppler387_Re200k_turb_a5` | 5 | classical SA |
| `strL2prop_eppler387_Re200k_turb_a7` | 7 | classical SA |

---

## Group D — Eppler 387 Reynolds sweep, alpha=5°, L1 (8 NEW cases)

Feeds **`eppler_L1compare_lowRe.pdf`, `eppler_L1compare_highRe.pdf`**
(`regen_epp_L1compare.py`) and **Table `tab:eppresweep`** (`regen_resweep_table.py`).
Common: geometry `eppler387`, alpha=5, M=0.1, chi_inf=8.76e-4, slowdown 0.01,
level L1. Sweep Re = {60,100,200,300,460}k; **the 200k point IS the Group-C
`cavL1/strL1prop_eppler387_Re200k_a5` cases** and is not re-listed. Only the 4
off-200k Reynolds numbers (× 2 meshes = 8) are new.

| case dir | mesh | Re | fig / table |
|---|---|---|---|
| `sweep_Re60k_a5`      | cav | 6e4   | L1compare_lowRe, eppresweep |
| `sweep_Re100k_a5`     | cav | 1e5   | L1compare_lowRe, eppresweep |
| `sweep_Re300k_a5`     | cav | 3e5   | L1compare_highRe, eppresweep |
| `sweep_Re460k_a5`     | cav | 4.6e5 | L1compare_highRe, eppresweep |
| `sweep_str_Re60k_a5`  | str | 6e4   | L1compare_lowRe, eppresweep |
| `sweep_str_Re100k_a5` | str | 1e5   | L1compare_lowRe, eppresweep |
| `sweep_str_Re300k_a5` | str | 3e5   | L1compare_highRe, eppresweep |
| `sweep_str_Re460k_a5` | str | 4.6e5 | L1compare_highRe, eppresweep |

Note: `regen_resweep_table.py` ALSO reads structured `str_dir(Rk)` for the same
Re list; those are the same `sweep_str_*` dirs listed above (200k = `strL1prop...a5`).

---

## Totals

| group | cases | model |
|---|---|---|
| A — flat plate | 5 | SA-AI |
| B — NLF0416 ladder | 24 | SA-AI |
| B — NLF0416 turb baseline | 4 | classical SA |
| C — Eppler387 ladder | 24 | SA-AI |
| C — Eppler387 turb baseline | 4 | classical SA |
| D — Eppler387 Re-sweep (new) | 8 | SA-AI |
| **total distinct** | **69** | 61 SA-AI + 8 turb |

---

## UNCONFIRMED / flagged values — confirm before GPU re-run

Everything below was confirmed directly from a regen script, a `run_*.py`
builder, sa-ai.tex, or the shipped `Flow360.json`. Items 1–2 were verified by
reading the case JSONs and are noted for transparency; items 3–4 are the genuine
open questions for the human before a GPU re-run.

1. **CONFIRMED — Mach = 0.1 for all groups.** Read directly from `freestream.Mach`
   in the shipped JSONs: `cavL1prop_eppler387_Re200k_a5`, `cavL1prop_nlf0416_Re4M_a4`,
   `sweep_str_Re60k_a5`, `flatplate_ags_Tu0080`, `strL2prop_eppler387_Re200k_turb_a5`
   all have `Mach=0.1`. (The experimental LTPT Mach varies 0.03–0.13 per Re, but the
   CFD ran at a single Mach=0.1.)

2. **CONFIRMED — chi_inf.** JSON `freestream.turbulenceQuantities.
   modifiedTurbulentViscosityRatio` (the post-slowdown BC seed) reads:
   Eppler/NLF SA-AI = `8.76e-6` = 8.76e-4 × 0.01 → physical chi_inf = **8.76e-4** ✓;
   flatplate Tu=0.08% = `1.2016e-5` = 1.2016e-3 × 0.01 → Mack-map chi_inf ✓;
   turb baselines = `3.0` (no slowdown) ✓. The driver's `case.py::_seed_for_json`
   reproduces exactly these seeds.

3. **OPEN — Pseudo-step budget (max_steps).** Left at the `CaseConfig` default; NOT
   encoded per case here. Known references: flat plate 10800 s wall-time cap
   (`run_flatplate_ags.py`); turbulent baselines 60000 steps
   (`run_turb_baselines.py`). SA-AI ladder / sweep step counts were not located
   in a single builder — the cloned base JSON's `timeStepping.maxPseudoSteps`
   governs by default. **Confirm each base JSON's step count is the intended
   production budget before relying on it.**

4. **OPEN (naming, low risk) — which `sweep_*` variant is canonical (Group D).** `flow360_g4` also contains
   `sweep_s1_Re*k_a5` and `sweep_str_s1_Re*k_a5` (an alternate `fSlow=1` staging).
   The paper table/figure scripts (`regen_resweep_table.py`, `regen_epp_L1compare.py`)
   read the plain `sweep_Re{Rk}k_a5` / `sweep_str_Re{Rk}k_a5` dirs (NOT `_s1`), so
   those are used here. `regen_epp_resweep.py` has a `MODE=="sweep"` branch that
   also targets `sweep_Re{Rk}k_a5` but reads from `flow360/` — it is a diagnostic
   (writes only to `/tmp/`), not a paper-figure generator; ignore it. Confirmed,
   but flagged because the naming is easy to confuse.

5. **Random seed.** The Flow360 RANS solve is deterministic (steady pseudo-time);
   there is no stochastic seed. No seed to confirm.

6. **NACA0012 is intentionally EXCLUDED.** `paper/regen_figs.py` builds NACA0012
   (and NLF) figures from `flow360/run10k_*`, but the figure filenames it writes
   (`naca0012_cf.pdf`, `convergence.pdf`, `polar.pdf`, `xtr_alpha.pdf`) do **not**
   appear in `sa-ai.tex`. It is a superseded script; NACA0012 is not a paper CFD
   case. If a later sa-ai.tex revision reintroduces NACA0012, add
   `run10k_{aftsa,turb}_naca0012_a{-2,0,2,4,6,8}` (Re=1e6 per `naca0012_re1m.json`).
