# paper/repro — self-contained reproduction package

Everything needed to reproduce every figure, table, and numerics claim of the
SA-AI paper, with **no code dependencies outside this folder**: the only
external requirements are pip packages (numpy, scipy, matplotlib, vtk, jax,
loguru) and the Flow360 solver binary (`compute` repo, branch
`favorable-rate`, commit `349162fc26` or later, built to
`compute/install/release/bin/Flow360Solver`).

## Layout

- `lib/` — the model kernel: constants + amplification-rate/gate functions
  (`aft_sources.py`), the Tu→χ∞ map (`calibrate_kernel.py`), Falkner–Skan and
  Blasius solution machinery, the SA wall-layer module. These are verbatim
  copies of the project originals with only import lines rewritten;
  `sa-ai/tests/test_constants_consistency.py` fails if they drift.
- `analytic/` — every non-CFD figure/table (paper Figs. 1–7, Tables 1–2, the
  a_max eigenvalue check, the Tu map). Run all:
  `python analytic/regenerate_all.py` (figures land in `paper/figs/`).
- `cfd/` — the CFD figure/table generators. They read one case tree selected
  by `SAAI_CFD_ROOT` (default: the shipped `sa-ai/flow360_fr/`). Run all:
  `python cfd/regenerate_cfd.py`. `prepare.py` assembles a tree from fresh
  driver output (falling back to the shipped tree) and computes the derived
  slice fields (`add_derived_to_slice.py`).
- `driver/` — re-runs the CFD cases themselves: the 69-case matrix
  (`cases.py`), the canonical model environment (`saai_env.py`,
  `AI_VG_GATE=4` + the paper constants incl. `AI_FPG_RATE_SLOPE=5.5`), the convergence protocols
  (`convergence.py`: plain / converge-by-xtr ladder / staged-fSlow sweep,
  with `converge_by_xtr.py` included), and the case builder + solver launch
  (`case.py`, `config.py`, `env.py`, `solve.py`, `run.py`). Example:
  `python driver/run.py cavL1prop_nlf0416_Re4M_a4 --gpu 0`.

## Reading order (follows the paper)

Read each script alongside the passage it backs; `regenerate_all.py` and
`regenerate_cfd.py` run their scripts in this same order.

| Paper | Claim / float | Script |
|---|---|---|
| §II model | fig:indicatorplane | `analytic/fig01_indicator_plane.py` |
| §II model | fig:kernel | `analytic/fig02_kernel_maps.py` |
| §III.A a_max | eigenvalue 0.19 (asserted) | `analytic/amax_rayleigh.py` |
| §III.B qualitative constants | c_ν,ai plateau; c_A / p brackets, anchors re-solved per candidate (~40 min, on demand) | `analytic/scan_background_constants.py` |
| §III.C triple | (254, 1.005, 11) meets the 3 conditions at the quoted residuals (asserted) | `analytic/verify_three_anchors.py` |
| §III.C N=1 level | departure-anchor sensitivity sweep (on demand) | `analytic/scan_anchor_level.py` |
| §III.A instrument | fig:nuhat | `analytic/fig03_fs_transport_rows.py` |
| §III.C family | fig:shapefactor (cliff-only + factored) | `analytic/fig04_shapefactor.py` |
| §III.D K_λ | fig:worstpoint, fig:klambda_sc (eq:klambda fixed point) | `analytic/fig05_06_klambda.py` |
| §III.D K_r | eq:kr one-point fit at β=0.35 (asserted; `--forms` = E/R1/R2 selection study) | `analytic/fit_fpg_rate_slope.py` |
| §III.E handover | tie exactness: linear nuHat and dB = 0 to roundoff (asserted) | `analytic/verify_wall_layer_tie.py` |
| §III.E coarse mesh | tab:yplus | `analytic/tab02_yplus.py` |
| §III.F receptivity | eq:tumap | `analytic/tu_map.py` |
| §III.G assembled | constants block (asserted vs paper Table) | `analytic/constants_report.py` |
| §IV flat plate | fig:flatplate_batch; `ONSET_DIAG=1` prints the quoted AGS onset numbers (both conventions) | `cfd/regen_flatplate_flow360.py` |
| §V NLF | nlf_cf figures | `cfd/regen_nlf_v2.py` |
| §V NLF | tab:nlftrans | `cfd/regen_nlf_transition.py` |
| §V/§VI | dx_tr/dN sensitivity column of tab:nlftrans + Sec. VI onset numbers (needs xfoil + xvfb-run, on demand) | `cfd/xtr_sensitivity.py` |
| §V NLF | L0-artifact narrative (diagnostic; figure not in main.tex) | `cfd/regen_l0_artifact.py` |
| §V NLF | fig:nlfpolar | `cfd/regen_nlf_polar.py` |
| §VI Eppler | eppler_cf figures, fig:epppolar | `cfd/regen_eppler_v2.py` |
| §VI Eppler | tab:eppxtr | `cfd/regen_epp_reattach.py` |
| §VI α=7° | N_crit sweep behind the shared-e^N discussion (needs xfoil + xvfb-run, on demand) | `cfd/xfoil_ncrit_sweep.py` |
| §VI Re sweep | fig:eppresweep_low/high | `cfd/regen_epp_L1compare.py` |
| Appendix | 18 wall-anchored contour sheets (velocity + log10 chi, 6 grids) | `cfd/regen_chi_sheets.py` |
| §VI Re sweep | tab:eppresweep | `cfd/regen_resweep_table.py` |
| numerics.md | discrete-scheme record (replay, spike trace, operator variants; reads the mode-3 tree) | `numerics/*.py` |

Infrastructure (no single paper anchor): `lib/` (the kernel, consistency-
tested), `analytic/_saai.py` (shared constants plumbing), `driver/` (re-run
any of the 69 CFD cases with the canonical env + convergence protocols).
- `cfd/xfoil_ncrit_sweep.py` — the XFOIL N_crit sweep behind the Sec. VI
  α=7° discussion (needs `xfoil` + `xvfb-run` on PATH).
- `numerics/` — the discrete-scheme studies behind `paper/numerics.md`: the
  bit-faithful replay of the solver's gate kernel on the slice triangulation
  (`replay_gate3_kernel.py`), the spike-node trace (`trace_spike_node.py`),
  and the term-attribution / operator-variant studies (`diag_*.py`). These
  intentionally read the **gate-3 (mode 3)** case tree `sa-ai/flow360_a3/`,
  since they document the pathology that mode 4 removes.

## Data

Case trees (`sa-ai/flow360_fr/`, `flow360_g4/`, `flow360_a3/`) and mesh/restart
binaries are not in git. The digitized experimental references and the
mfoil/XFOIL e^9 caches (`*.pkl`) ship inside the case tree root.

## One-command checks

```
python analytic/regenerate_all.py          # 12/12 must pass
SAAI_CFD_ROOT=... python cfd/regenerate_cfd.py    # 9/9 must pass
python ../../tests/test_constants_consistency.py  # 4/4 must pass
```
