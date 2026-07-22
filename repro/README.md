# SA-AI paper reproduction (pre-Flow360)

> **Superseded:** the canonical, self-contained reproduction package now
> lives at `paper/repro/` (no code dependencies outside that folder). This
> directory is kept for history; the constants-consistency test guards both.

Minimal, self-contained scripts that regenerate **every figure, table, and
quantitative result appearing before the Flow360 CFD runs** in the SA-AI
transition-model paper. Each script imports the exact model kernel, physics,
and constants from `sa-ai/src/` (and `sa-ai/scripts/calibrate_kernel.py`) — no
formula or constant is restated here, so "same model, same constants" is
guaranteed by construction.

## Run

From the repo root (`/home/qiqi/flexcompute`):

```bash
python3 sa-ai/repro/regenerate_all.py            # every script; prints PASS/FAIL
python3 sa-ai/repro/analytic/constants_report.py # constant audit against paper Table
```

Figures are written into `sa-ai/paper/figs/` at the exact `\includegraphics`
filenames `main.tex` expects, so the paper picks them up directly. JAX is pulled
in transitively by the `src` imports (expected); a `.jax_cache/` dir may appear
under `sa-ai/paper/`.

## Script -> figure/table -> paper label

| script (`analytic/`)        | paper float                         | output |
|-----------------------------|-------------------------------------|--------|
| `constants_report.py`       | Constant Table (audit + assert)     | stdout |
| `tu_map.py`                 | `eq:tumap` (Tu -> chi_inf -> N_crit)| stdout |
| `amax_rayleigh.py`          | a_max = tanh-layer Rayleigh eigenvalue (asserts 0.1897 vs A_MAX) | stdout |
| `fig01_indicator_plane.py`  | `fig:indicatorplane`                | `figs/indicator_plane.pdf` |
| `fig02_kernel_maps.py`      | `fig:kernel`                        | `figs/kernel_maps.pdf` |
| `fig03_fs_transport_rows.py`| `fig:nuhat`                         | `figs/fs_nuHat_rows.pdf` |
| `fig04_shapefactor.py`      | `fig:shapefactor`                   | `figs/shapefactor_amplification.pdf` |
| `fig05_06_klambda.py`       | `fig:worstpoint`, `fig:klambda_sc`  | `figs/klambda_profiles.pdf`, `figs/klambda_selfconsistent.pdf` |
| `fig07_wall_layer.py`       | `fig:walllayer`                     | `figs/wall_layer.pdf` |
| `tab02_yplus.py`            | `tab:yplus`                         | stdout |

`_saai.py` is the shared helper (sys.path + cwd setup and the single point where
every canonical constant is imported from its `src/` home). It is plumbing, not
a paper float.

## The Flow360 (CFD) leg

- `driver/` — the minimal clarified driver: canonical model env
  (`saai_env.py`, constants imported from `calibrate_kernel.py`), the 69-case
  paper matrix (`cases.py`), case build (`case.py`), the paper's convergence
  protocols (`convergence.py`: plain / ladder / staged-fSlow, selected by case
  family), and the CLI (`run.py`). See `driver/RERUN_MANIFEST.md` for the full
  case ↔ figure map and protocol table.
- `postprocess/` — after runs: `prepare.py` assembles the figure tree (fresh
  results with fallback to the shipped `flow360_a3/`) and computes the derived
  slice fields; `regenerate_cfd.py` regenerates every CFD figure/table via the
  `paper/regen_*` generators, which read the tree from `SAAI_CFD_ROOT`.
- The shipped `flow360_a3/` tree IS the paper's final fleet, produced with
  exactly `saai_env.canonical_env()` values (the `ai` variant of
  `flow360/run_vg_all.py`); the driver was validated end-to-end against it on
  a three-case subset, one per protocol (see VALIDATION below).

## Not reproduced here (no code by design)

- **Table 1 (transition-model taxonomy)** — a literature survey, not computed.
- **`eq:scaling`** — a dimensional-analysis scaling argument, not computed.
- **Mesh generation** — every CFD case clones a pre-meshed dir from
  `flow360_a3/`. The contour -> Construct2D (structured) / cavity (unstructured)
  meshing pipeline behind the mesh figures/tables lives in `flow360/`
  (`*_contour.py`, `build_proper_*.py`) and is not wrapped here.
- **e^N references** — mfoil results are consumed as cached artifacts
  (`flow360_a3/mfoil_*.pkl`; mfoil = Fidkowski 2022, coupled viscous-inviscid
  e^9 panel code, default settings, N_crit=9); XFOIL numbers (v6.99, e^9,
  N_crit=9) are quoted where mfoil fails (documented per-table in the paper).
- **Experimental data** — read from `paper/data/`: McGhee et al. NASA TM-4062
  tabulated Cp / section coefficients / oil-flow (Eppler 387), Somers NASA
  TP-1861 orifice and polar data (NLF(1)-0416), Abu-Ghannam & Shaw and
  Schubauer–Skramstad flat-plate references (digitized).

Everything else pre-Flow360 is covered above.

## VALIDATION (recorded 2026-07-10)

**Figure identity (analytic set).** `regenerate_all.py`: 9/9 pass. The ported
scripts' printed diagnostics are digit-for-digit identical to the last-good
`paper/regen_*` runs the shipped `main.pdf` was built from -- fig04's full
14-beta mean/late table (Blasius mean 1.0427e-2 = 1.00x Drela, late 1.1157e-2 =
1.07x), fig03's three rows (N_end 18.9 / 11.6 / 17.5 at the same x_max), the
K_lambda fixed point 5.90, and fig07's wall-layer footprint (peak 0.80% at
y+=15.3, |dB| < 1e-3). Envelope marches at nx=800, ny=600 (the grid-converged
resolution).

**Driver end-to-end (one case per protocol, vs the shipped fleet).** Fresh
solves through `run.py` (clone -> chi patch -> canonical env -> protocol),
compared to the same case in `flow360_a3/`:

| case | protocol | fresh | shipped |
|---|---|---|---|
| `flatplate_ags_Tu0160` | plain | onset Re_theta 908.9 | 908.9 (0.00%) |
| `cavL1prop_nlf0416_Re4M_a4` | ladder | xtr 0.2733/0.6017, CL 0.9813, CD 0.00827 | identical to all printed digits |
| `sweep_Re300k_a5` | staged | CL 0.9603, CD 0.01110 | identical |

**Post-processing.** `postprocess/prepare.py` assembled a hybrid tree (3 fresh
+ 66 shipped + e^N reference caches) and `regen_epp_reattach.py` reproduced the
paper's reattachment table through `SAAI_CFD_ROOT` unchanged.

## Canonical constants and their `src/` homes

`constants_report.py` imports and asserts all 14 against the paper Table:

| constant | value | imported from |
|---|---|---|
| a_max | 0.19 | `aft_sources.AFT_RATE_SCALE` |
| g_c | 1.055 | `aft_sources.AFT_SIGMOID_CENTER` |
| s | 8.0 | `aft_sources.AFT_SIGMOID_SLOPE` |
| ReOmega floor | 290 | `aft_sources.AFT_RE_OMEGA_FLOOR` |
| K_lambda | 5.9 | `aft_sources.AFT_CLIFF_LAMBDA_SLOPE` |
| p | 4 | `aft_sources.AFT_BARRIER_POWER` |
| c_A | 4 | `aft_sources.AFT_Q4_CA` |
| gammaCoeff | 2 | `aft_sources.AFT_GAMMA_COEFF` |
| tau | 4 | `regen_wall_layer.TAU` (read as literal) |
| tau_D | 1.36 | `regen_wall_layer.TAU_D` (read as literal) |
| c_nu,ai | 1/12 | `boundary_layer_solvers.NuHatBlasiusSolver.aft_nuLamScale` |
| A_TU | -8.43 | `calibrate_kernel.A_TU` |
| B_TU | 2.4 | `calibrate_kernel.B_TU` |
| c_v1 | 7.1 | `calibrate_kernel.C_V1` |

SA wall-layer constants (kappa, cb1, cb2, sigma, cv1, cv2, cw1, cw2, cw3) used
by `fig07`/`tab02` are imported from `src.physics.spalart_allmaras`.
