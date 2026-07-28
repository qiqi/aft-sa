# Full-body spheroid discriminator (arms A/B): symmetry sheets exonerated, the azimuthal "staggering" was probe geometry, and the converged O-grid is the march

*2026-07-28 04:13, CFD-agent thread. Task: rerun re72a0 (alpha = 0,
Re_L = 7.2e6, Mach 0.1) on a FULL-circumference L1 O-grid (user directive:
no symmetry sheets), arms A/B = lowMachPreconditioner false/true, harvested
with the 0105-record instruments. Mid-task protocol update adopted from
2026-07-28-0350 (unstructured-family record): runs extended past the 20k
cold-start budget to a settled CD (<3e-5/1k), harvest re-based on the
discrete wall facets (Moller-Trumbore), converged numbers compared against
the unstructured 43k state, the og-ext half-model control, and the 0105 BL
march. Committed: `spheroid/ogrid_spheroid_full.py`,
`spheroid/build_fullbody_case.py`, `paper/repro/cfd/spheroid_fullbody_check.py`.
JSON: `paper/repro/cfd/figs_explore/spheroid_fullbody.json`; figures
`..._rings.png`, `..._H.png` (exploratory). NO tex edits.*

**Headline (three findings):** (1) The half-model symmetry sheets are
EXONERATED: at matched budget and matched convention the full body
reproduces the half-model to +0.001 in H and 2e-4 in front position, and at
convergence both relax identically onto the BL-march mean flow (front
0.8581 vs og-ext 0.8582 vs unstructured 0.8584). (2) The 0105 record's
"cell-locked one-azimuthal-cell staggering" (du/u rms 2.8e-2 at L1,
0.15 delta99) is a PROBE-GEOMETRY ARTIFACT, not a flow feature: it is the
one-cell wall-distance modulation of rays anchored on the ANALYTIC surface
over the faceted discrete wall (sag ~1.6e-5 L = 21 h0 = up to 12% of the
probe height). Re-basing the rings on the actual wall facets collapses it
500x (rms 3.1e-2 -> 5.6e-5), identically on half/full, transient/converged.
(3) The low-Mach preconditioner does NOT change the converged laminar mean
flow (H within 0.003, front -0.0026) — it accelerates convergence ~2.5x
(settled by ~16k steps where arm A needed ~40k) and shifts converged CD
+3.4e-4 (0.01302 vs 0.01268).

## 1. Mesh: full-circumference L1 O-grid (`ogrid_spheroid_full.py`)

Identical meridian half-plane grid as `ogrid_spheroid.py` (meridian nodes,
h0/L = 7.5e-7, growth 1.1927, ds_pole 3.75e-4, R_ff 30 L), revolved over
phi in [0, 2pi) with periodic closure BY NODE IDENTIFICATION (azimuthal
index wraps; no seam, no symmetry boundary), azimuthal count 160 = 2x the
half-model's 80 so every cell size matches (dphi = 2.25 deg both). Pole
prisms all around, as in the half-model.

- nodes 4,353,622; cells 4,320,000 (4,291,200 hex + 28,800 prism) =
  exactly 2x the half-model's 2,160,000;
- validity: min hex/prism corner Jacobians 4.7e-15/4.1e-15 (> 0), wall
  area -0.009% vs analytic FULL spheroid, farfield -0.016%, volume -0.031%
  (watertight => the periodic wrap is seamless), half-plane worst corner
  sine 0.939, layers below 0.3: none;
- flow360gmshtocgns conversion clean:
  `/local_data/qiqi/sa-ai/spheroid_meshes/mesh_re65full_L1.cgns`
  (454,251,775 bytes; ~2x the half-model's 229 MB). Build log:
  `runlogs/spheroid_meshfull_L1.log`.

## 2. Cases + protocol (byte-comparability proof)

`build_fullbody_case.py` stages both arms from the campaign case
`case_ogrid_L1_saai_re72a0`'s Flow360.json verbatim. Full JSON diff vs the
reference contains ONLY: fluid/symmetry boundary removed; refArea doubled
(0.0218166 = pi B^2); volumeOutput removed (disk discipline) and slices
added (meridian z = 0 + constant-x planes at x/L 0.20/0.42/0.70,
primitiveVars + nuHat); surfaceOutput retargeted to fluid/wall (the
campaign file carried a stale `fluid/wing` key). Arm B differs from arm A
in exactly `lowMachPreconditioner: true` +
`lowMachPreconditionerThreshold: 0.1`. Everything else — Mach 0.1, muRef
1.3889e-8, alpha 0, seed 8.76e-6 (physical, fSlow = 1), Roe, kappaMUSCL -1,
adaptive CFL max 200 — is byte-identical. Solver-echo `ai_constants.log`
extracted per case and diffed LINE-IDENTICAL to the campaign's re72a0
(incl. ai_laminarSlowdown 1.0, ai_fv1Bypass 1); solver logs truncated after
extraction.

Two gotchas found (documented in the builder):
- removing `autoVisOutput` aborts the solver (json type_error.305 at
  output-writer init — it is dereferenced unconditionally); keep it and
  retarget its surface;
- `lowMachPreconditioner: true` requires `lowMachPreconditionerThreshold`
  (read unconditionally in `NavierStokesSolver.cpp
  initLowMachPreconditioner`; missing -> nlohmann abort). Convention from
  the flow360translator test refs: threshold = freestream Mach.

Runs (all single-GPU, in-session, no mpirun; occupancy-checked, <= 4
qiqi GPU processes at all times): arm A 20k (55 min) + restart leg 23k
-> 43k total; arm B 20k + 5k confirmation leg -> 25k. Restart legs per the
0350 Sec-9 recipe; arm A's processed mesh dmp had been cleaned for disk and
was regenerated with the case's ORIGINAL partitionerData (MeshProcessor
only) before the leg. Case dirs (repo-symlinked, snap20k/ archives hold the
20k slices): `/local_data/qiqi/sa-ai/spheroid_fv1/
case_ogridfull_L1_saai_re72a0{,_lowmach}`. Run logs:
`runlogs/spheroid_fullbody_arm{A,B}.log`, `..._armA_leg2.log`,
`..._armB_leg2.log`.

## 3. Convergence (the 0350 finding reproduced on the O-grid family)

| state | steps | CD | CD end-drift /1k | CL |
|---|---|---|---|---|
| half-model campaign (20k) | 19999 | 0.01663 | -2.2e-4 (falling) | +5.0e-5 |
| og-ext control (half, 43k; other thread's case) | 43k | 0.01266 | -2.0e-5 | +2.4e-3 |
| arm A full body (43k) | 43k | 0.01268 | -2.1e-5 | +4.7e-4 |
| arm B full body, precond (20k) | 20k | 0.01301 | +8.0e-7 | -1.3e-4 |
| arm B + confirm leg (25k) | 25k | 0.01302 | +5.7e-6 | -5.4e-4 |
| unstructured 43k (0350) | 43k | 0.01531 | +2.3e-5 | +3.7e-3 |

Arm B was settled INSIDE its first 20k (CD flat at ~1e-6/1k by step ~14k):
the preconditioner removes most of the slow transient that costs the
unpreconditioned arms ~40k steps. Note the O-grid families converge to
CD 0.0127 while the unstructured 43k sits at 0.0153 — a real family
difference at converged state (aft prism/tet resolution class), not a
budget artifact; not chased here.

## 4. H and fronts: converged verdict table (facet-re-based origin, phi = 90)

H at x/L = 0.20 / 0.42 / 0.70; front = near-wall chi = 1 crossing
(chi max over y <= 0.02, RAY grid, 0.004 station spacing):

| state | H(0.20) | H(0.42) | H(0.70) | front chi1 | cv1 |
|---|---|---|---|---|---|
| half 20k (transient) | 2.5224 | 2.5334 | 2.5158 | 0.9177 | 0.9246 |
| og-ext half 43k | 2.5237 | 2.5540 | 2.6057 | 0.8582 | 0.8948 |
| arm A full 43k | 2.5248 | 2.5549 | 2.6065 | 0.8581 | 0.8943 |
| arm B full precond 25k | 2.5242 | 2.5545 | 2.6097 | 0.8555 | 0.8947 |
| unstructured 43k (0350) | 2.515 | 2.556 | 2.611 | 0.8584 | — |
| BL march (0105, op-matched) | 2.533 | 2.561 | 2.613 | — | — |

- **Half vs full, matched budget AND matched convention** (raw analytic
  origin, 20k): full 2.5085/2.5258/2.5118, front 0.9175 vs half
  2.5074/2.5248/2.5106, front 0.9177 — identical to +0.001/2e-4. The
  symmetry sheets play no role at either state.
- **Converged**: every variant (half/full, precond on/off, unstructured)
  lands on the march to <= 0.018 in H (aft gap <= 0.007) and front
  0.8555-0.8584. The 0105 "anomalously full mean flow" is confirmed as a
  cold-start transient on the O-grid family too (the og-ext control's
  [OG-EXT-RESULTS] placeholder in the 0350 record can be filled with the
  ext row above).
- lowMachPreconditioner moves the CONVERGED laminar state by <= 0.003 in H
  and -0.0026 in front — no low-Mach-dissipation content in the anomaly.
- Sag-correction footprint on these O-grid meridians is tiny as predicted:
  |t0|max = 3.7e-6 L (node meridian; meridional-chord sag only); raw vs
  corrected H differs by <= +0.009.

Instrument validation: the same script on the committed half-model 20k
volume reproduces the 0105/0350 numbers exactly (raw H 2.5074/2.5248/2.5106
= 0105's 2.507/2.525/2.511; front 0.9177 = committed front-summary median
0.9177; cv1 0.9246 vs 0.9247).

## 5. The azimuthal one-cell "staggering" is probe geometry (key readout)

Rings of u_s at x/L = 0.42, heights 0.15/0.3/0.6/1.2 x delta99, 12 samples
per azimuthal cell (full circle on the full body, 5-175 deg arc on the
half), residual = signal minus 25-sample running mean (the 0105
wiggle_stats convention). du/u rms at 0.15 delta99:

| origin convention | half 20k | arm A 43k | arm B 25k | og-ext 43k |
|---|---|---|---|---|
| analytic surface (0105 convention) | 2.98e-2 | 3.13e-2 | — | — |
| discrete wall facets (Moller-Trumbore re-base) | 5.5e-5 | 5.6e-5 | 5.6e-5 | 5.6e-5 |

- With the 0105 convention the full body reads the SAME one-cell mode as
  the half model (rms 3.1e-2, p2p 10%, wavelength 2.25 deg = exactly the
  cell, periodic-FFT mode number exactly 160 = N_cell, low modes <= 3e-4):
  removing the symmetry sheets changes nothing — and the CONVERGED field
  reads the same as the transient. A state-independent, geometry-locked
  signal.
- Re-based on the actual wall facets the mode collapses 500x at every
  height, on every case (residual mode-160 amplitude 7.9e-5 at
  0.15 delta99, 3.7e-6 above the BL; low modes ~1e-6 — the full-body
  solution is azimuthally uniform to ~1e-5 of u_s).
- Mechanism: the rays' analytic origin sits up to the facet sagitta
  (dc^2 kappa / 8 ~ 1.6e-5 L = 21 h0 at L1) ABOVE the discrete wall between
  azimuthal node lines, so "fixed height" rings sample wall distances
  modulated +-12% at 0.15 delta99 with exactly one-cell periodicity; near
  the wall du/u ~ dh/h. This also quantitatively explains the 0105 L1->L2
  amplitude ratio (sag/delta99 halves per level: their L2 p2p 2.6% vs L1
  ~9-10%) — refinement reduced the artifact, not a flow mode.

**Consequence for the 0105 record:** its Sec-3 azimuthal staggering
(rms 7.7e-3/2.8e-2, "12x the meridional mode", "should be exactly zero")
is the analytic-origin convention reading the faceted wall, not a discrete
solution mode; its meridional wiggle (rms 6.3e-4, locked to the meridional
cell) is much closer to the meridional sag scale (~1.2e-6/8.9e-4 d99 ~
1.3e-3 du/u class) and should be re-examined the same way; and per the 0350
record the near-wall momentum-source instrument was reading the same 20k
transient the H table above closes. The 0105 candidate-(a) conviction
("solver numerics on the axisymmetric O-grid") does not survive
convergence + convention correction at L1: nothing in these arms
distinguishes half from full, or preconditioned from not, in the converged
laminar mean flow. What remains open of 0105 is only the L2 aft deepening
(H 2.44 at 0.70 at 20k) — expected to close at converged budget like L1,
but not measured here (og-L2 ext leg ~5 GPU-h, 0350 caveat 1).

## 6. Honest-findings ledger

1. Arm A's first solve crashed at output-writer init because the builder
   had dropped autoVisOutput (Sec 2 gotcha); rebuilt + rerun (prep reused).
2. Arm B's first solve aborted because lowMachPreconditionerThreshold was
   missing (Sec 2 gotcha); patched to 0.1 = freestream Mach and rerun.
   Neither failure consumed a converged-state result.
3. The 20k harvest in this thread's first pass (raw convention, before the
   0350 protocol update landed) is preserved in the JSON under
   `*_raw_analytic_origin_20k` keys and in the snap20k/ archives; the
   record's tables mark convention explicitly.
4. Ring probes on the constant-x slices had 8/1920 invalid points
   (degenerate slice-triangulation edges); filled by periodic
   interpolation over phi (counted in the JSON: `n_invalid_filled`).
5. Arm B shows a mild CL wander (-5.4e-4 at 25k) like the unstructured
   43k's (+3.7e-3); fronts on the phi = 90 meridian are quoted; not chased
   at alpha = 0.
6. The half-model 20k row in Sec 4's corrected table (2.5224/2.5334/2.5158)
   differs from the raw 0105 L1 row (2.507/2.525/2.511) by the sag
   correction (+0.009 class) — both are recorded; comparisons are always
   convention-matched.
7. Front numbers here are chi = 1 near-wall crossings from settled-CD
   states, NOT converge_by_xtr stability batches; the 0350 flag that the
   campaign's committed a0 fronts (0.908/0.918/0.925) are budget artifacts
   stands — the converged front at this condition is 0.855-0.858 on every
   family, still far aft of Stock's measured 0.438 (the model-content
   discussion now routes through transport realization + seed, per 0350
   Sec 7.2).

## 7. Artifacts

- Mesh: `/local_data/qiqi/sa-ai/spheroid_meshes/mesh_re65full_L1.cgns`
  (+ `runlogs/spheroid_meshfull_L1.log`)
- Cases: `/local_data/qiqi/sa-ai/spheroid_fv1/case_ogridfull_L1_saai_re72a0`
  (arm A, 43k, CD 0.01268), `..._lowmach` (arm B, 25k, CD 0.01302); each
  with `ai_constants.log`, `total_forces_leg1.csv` + `total_forces_v2.csv`
  (leg 2), `snap20k/` (20k slices+surface+forces), converged slices
  `slice_{meridian,x020,x042,x070}.pvtu`, surface `surface_fluid_wall.pvtu`;
  solver logs truncated; mesh/restart dmps cleaned (regen: MeshProcessor
  with the case's partitionerData).
- Scripts (committed): `spheroid/ogrid_spheroid_full.py`,
  `spheroid/build_fullbody_case.py`,
  `paper/repro/cfd/spheroid_fullbody_check.py` (imports the 0350 verdict
  script's facet re-basing; validates against the committed half-model).
- Results: `paper/repro/cfd/figs_explore/spheroid_fullbody.json`,
  `spheroid_fullbody_rings.png`, `spheroid_fullbody_H.png`.
- GPU cost: mesh ~5 min CPU; arm A 20k + 23k ~ 2.1 GPU-h; arm B 20k + 5k
  ~ 1.2 GPU-h; harvest CPU-only.
