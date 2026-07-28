# Converged spheroid alpha=0: circumferential uniformity verified, Cp(x) with the converged suction peak, and the BL profile series IS the march

*2026-07-28 10:30, post-processing thread.  Task (user directive, verbatim
intent): (1) first make sure the flow is circumferentially uniform;
(2) plot pressure versus x; (3) a series of boundary-layer profile plots:
at each x, draw a line from the surface in the surface-normal direction
till it reaches outside the BL, probe/interpolate the velocity along that
line, and plot it at each x station.  Primary case: the CONVERGED
(43k-step) full-body structured O-grid arm A,
`/local_data/qiqi/sa-ai/spheroid_fv1/case_ogridfull_L1_saai_re72a0`
(Re_L = 7.2e6, alpha = 0, M = 0.1); cross-checks with the SAME instruments:
arm B (`..._lowmach`, 25k, settled) and the converged unstructured
`case_unstr_L1_saai_re72a0` (43k).  Pure CPU post-processing, no runs.
Script `paper/repro/cfd/spheroid_uniformity_profiles.py` (rerunnable,
~25 min CPU for all three arms); figures + JSON under
`paper/repro/cfd/figs_explore/` (captions in
`spheroid_uniformity_profiles_captions.md`).  NO tex edits.*

**Headline:** (1) The converged full-body flow is circumferentially
uniform to a few 1e-5 of u_s at every probed station and height once the
rings are re-based on the actual wall facets (raw analytic-origin
convention: rms 3.1e-2 — a 550x fake, shown once for contrast); the
transition front itself is azimuthally uniform to rms 3.9e-4 in x/L.
(2) The converged suction peak sits at x/L = 0.497 (u_e maximum; the
0040/0105 20k field read 0.489), wall-Cp minimum -0.0924 at 0.495;
meridian and azimuthal-average Cp coincide to the wall-Cp azimuthal
scatter (<= 1.4e-4 for x <= 0.85).  (3) The profile series confirms the
0350/0413 verdict AT PROFILE LEVEL: the converged RANS laminar boundary
layer IS the marched laminar solution — |dH| <= 0.008 for x/L <= 0.7,
growing to +0.013 / +0.038 at 0.80 / 0.84 as the front (chi = 1 at 0.858)
is approached; the profiles overlay point-by-point in the figure.

## 0. Instrument validation (gates all passed)

- Arm A meridian sweep reproduces the 0413 verdict table EXACTLY:
  H(0.20/0.42/0.70) = 2.5248 / 2.5549 / 2.6065, front chi = 1 at 0.8581,
  c_v1 at 0.8943 (same RAY grid, edge/integral operator, facet re-basing,
  chi band y <= 0.02).
- The laminar-march reference (spheroid_a0_meanflow.bl_march, axisym) is
  Blasius-validated in-run: H = 2.5906 (2.5905), cf*theta*ue/nu = 0.2206
  (0.2205), grid/IC-shape variants within 0.0008.
- The raw-convention ring reproduces the 0413 numbers: rms 3.13e-2,
  periodic mode-160 amplitude 4.4e-2 at x = 0.42, 0.15 d99.

## 1. Circumferential uniformity (arm A, corrected rings; figure
`spheroid_uniformity_rings.png`, contrast in `..._contrast.png`)

Rings at 12 samples per azimuthal cell in the case's constant-x slice
planes; residual = 25-sample running-mean detrend (0105/0413 convention);
heights 0.15 / 0.5 / 1.0 delta99.  du_s/u_s:

| x/L | n/d99 | rms | p2p | mode-160 amp | low-mode (m<=8) max |
|---|---|---|---|---|---|
| 0.20 | 0.15 | 5.8e-5 | 2.2e-4 | 8.0e-5 | 3.3e-6 |
| 0.20 | 0.5  | 4.1e-5 | 1.4e-4 | 5.8e-5 | 2.5e-6 |
| 0.20 | 1.0  | 4.0e-6 | 1.4e-5 | 5.6e-6 | 9.6e-7 |
| 0.42 | 0.15 | 5.7e-5 | 2.1e-4 | 7.9e-5 | 5.5e-6 |
| 0.42 | 0.5  | 3.7e-5 | 1.3e-4 | 5.2e-5 | 8.1e-6 |
| 0.42 | 1.0  | 8.8e-6 | 2.9e-5 | 1.2e-5 | 1.8e-6 |
| 0.70 | 0.15 | 5.8e-5 | 2.0e-4 | 8.1e-5 | 4.2e-5 |
| 0.70 | 0.5  | 4.4e-5 | 1.5e-4 | 6.2e-5 | 3.4e-5 |
| 0.70 | 1.0  | 5.1e-6 | 1.7e-5 | 7.2e-6 | 1.2e-5 (m=1) |

- RAW vs CORRECTED, shown once (x = 0.42, 0.15 d99): rms 3.13e-2 ->
  5.65e-5 (554x), p2p 0.103 -> 2.1e-4, mode-160 4.4e-2 -> 7.9e-5.  The
  0413 mechanism (facet-sag wall-distance modulation of analytic-origin
  rays) confirmed on the converged state; arm B and unstructured
  reproduce the same collapse (3.13e-2 -> 5.7e-5; 2.16e-2 -> 2.8e-4).
- What remains after correction is (a) the mode-160 probe-interpolation
  ripple at the one-cell scale (7e-5 class, the blue band in the figure)
  and (b) ONE coherent flow mode: an azimuthal mode-1 at x/L = 0.70,
  amplitude 1.2e-5 (edge) to 4e-5 (0.15 d99) — the footprint of arm A's
  residual transient asymmetry (CL = +4.7e-4, CD still drifting
  -2.1e-5/1k).  Nothing else above 1e-5.
- Ring pressure: dCp rms <= 1.4e-6 everywhere (at/below the float32
  storage quantum of p, ~1.7e-5 in Cp — several rings read exactly 0).
- Wall data (solver-native surface output, all 160 azimuth lines):
  Cp azimuthal std <= 1.4e-4 and Cf rms/mean <= 1.2e-4 for all
  x/L <= 0.85.  At x/L = 0.90 Cf rms/mean = 6.3e-2 — this is NOT flow
  nonuniformity but the steep transitional dCf/dx: the Cf-rise front
  (k = 1.5 of the pre-rise minimum) extracted per azimuth line reads
  median 0.8880, rms 3.9e-4, p2p 1.1e-3 — the FRONT is azimuthally
  uniform to 0.04% L rms.  (No off-wall ring at 0.90: the case wrote
  constant-x slices only at 0.20/0.42/0.70; wall rings + front spread
  cover the station.)
- Opposite-meridian symmetry: front 0.8581 (phi = 90) vs 0.8588
  (phi = 270); H(0.42) identical to 4 decimals.
- The largest wall-Cp azimuthal scatter anywhere is 8.8e-3 at
  x = 0.977 (post-transition tail), the mapped image of the 3.9e-4
  front scatter through the steep tail gradients; arm B's settled state
  reads 2.5e-3 there and front rms 6.4e-5 — the scatter is transient
  convergence residue, not a stationary pattern.

**VERDICT: circumferentially uniform.**  Corrected u_s rings uniform to
rms <= 5.8e-5 (p2p <= 2.2e-4) at every station/height; wall Cp uniform to
<= 1.4e-4 pre-front; front position uniform to 3.9e-4 rms (6.4e-5 when
fully settled).

## 2. Pressure vs x (figure `spheroid_cp_x.png`)

- Wall Cp along the phi = 90 meridian (solver surface output, nose
  Cp = 1.00 stagnation sanity), azimuthal average overlaid — the two
  coincide within the scatter quoted above (meridian-minus-average
  <= 1.4e-4 class mid-body; max 1.4e-2 only at the x = 0.977 tail).
  Bernoulli 1 - (u_e/U_inf)^2 from the BL-edge sweep lies on top.
- CONVERGED suction peak: u_e maximum at **x/L = 0.4974** (the
  0040/0105 threads' 0.489 was the 20k transient; the converged peak
  moved aft by 0.008).  Wall-Cp minimum -0.0924 at x/L = 0.4948 (the Cp
  plateau is flat to ~4e-4 over 0.42-0.55, so the u_e peak is the sharper
  locator).  Arm B: peak 0.5006, Cp_min -0.0925 at 0.4948.  Unstructured:
  peak 0.4995, Cp_min -0.0928 (binned-average location 0.452 is
  ill-conditioned on the flat plateau — same plateau, no discrepancy).
- Marked on the figure: suction peak and the chi = 1 front (0.858).
  Front-convention reminder (handover rule 6): chi = 1 at 0.858,
  chi = c_v1 at 0.894, Cf-rise (k = 1.5) at 0.888 — three conventions,
  0.03 L apart, all azimuthally uniform.

## 3. BL profile series (figure `spheroid_bl_profiles.png`)

Surface-normal rays (analytic normal, facet-re-based origin) at
x/L = 0.10, 0.20, 0.30, 0.42, 0.50, 0.60, 0.70, 0.80, 0.84, 0.88 (last
two straddle the converged front 0.858); u_t/u_e vs n/delta99 panels +
physical n/L (log) waterfall; thin dashed overlay = the axisymmetric
implicit laminar-BL march on THIS field's own u_e, operator-matched
(the 0105 record's validated laminar reference).  Arm A:

| x/L | H field | H march | dH | Re_theta | d99/L | maxP (planar) |
|---|---|---|---|---|---|---|
| 0.10 | 2.490 | 2.499 | -0.008 | 399 | 4.1e-4 | 0.074 |
| 0.20 | 2.525 | 2.533 | -0.008 | 587 | 6.0e-4 | 0.077 |
| 0.30 | 2.541 | 2.549 | -0.008 | 743 | 7.5e-4 | 0.087 |
| 0.42 | 2.555 | 2.561 | -0.006 | 920 | 9.1e-4 | 0.094 |
| 0.50 | 2.563 | 2.568 | -0.005 | 1044 | 1.04e-3 | 0.094 |
| 0.60 | 2.579 | 2.581 | -0.002 | 1216 | 1.22e-3 | 0.097 |
| 0.70 | 2.607 | 2.604 | +0.002 | 1440 | 1.45e-3 | 0.108 |
| 0.80 | 2.695 | 2.681 | +0.013 | 1793 | 1.79e-3 | 0.147 |
| 0.84 | 2.793 | 2.755 | +0.038 | 2018 | 2.06e-3 | 0.167 |
| 0.88 | 2.922 | (separated) | — | 2335 | 2.41e-3 | 0.211 |

Observations:
- **The converged RANS laminar BL is the march at profile level**, not
  just in the integrals: |dH| <= 0.008 up to x/L = 0.5, sign change at
  0.6-0.7 (march slightly fuller aft), the profiles indistinguishable in
  the panels.  This closes the 0105 "anomalously full mean flow" on the
  converged field with the strongest instrument yet — the 0350/0413
  expectation (~0.02 in H) is met and bettered mid-body.
- The departure grows to +0.013 (0.80) and +0.038 (0.84): approaching the
  chi = 1 front the transported chi begins to thicken the profile, and
  the march itself is 0.04 L from its separation (0.880 on this u_e) where
  its adverse-gradient H steepens — both effects, small and expected.
- x/L = 0.88 (0.022 aft of the chi = 1 crossing): H = 2.922 — the profile
  is still laminar-shaped (adverse-gradient laminar, maxP = 0.21); the Cf
  rise completes only at 0.888.  The station pair 0.84/0.88 brackets the
  handover as intended.
- maxP (planar kernel, W = 61 estimator) rises monotonically 0.074 ->
  0.21 into the adverse region — the physical expectation (0105 challenge
  finding), no aft collapse anywhere on the converged field.

## 4. Cross-checks (same instruments)

- **Arm B (precond, 25k, settled)**: uniformity identical (ring rms
  5.7e-5 / dCp 0; front spread rms 6.4e-5 — the most uniform state);
  H(0.20/0.42/0.70) = 2.5242/2.5545/2.6097, front 0.8555 (phi = 270:
  0.8554); peak 0.5006; profiles: dH vs march -0.009..+0.006 through 0.7,
  +0.059 at 0.84 (its front is 0.0026 forward, the chi effect starts
  earlier).
- **Unstructured (43k)**: corrected ring rms 2.8e-4 at 0.42/0.15 d99
  (raw 2.16e-2 -> 77x collapse; its facet sag is larger and irregular),
  wall dCp uniform to 9e-6; front 0.8584 (phi = 270: 0.8590); peak
  0.4995; profiles: dH -0.016 (0.20, its H 2.515 = the 0350 table)
  ..+0.009 (0.70), +0.050 (0.84).  One flag: low-azimuthal-mode u_s
  content up to 6.7e-3 (mode 5, 0.15 d99; O-grid reads 8e-5 there) —
  probe-geometry vs the known mild CL = +3.6e-3 asymmetry (0350 caveat 3)
  unresolved, no front/pressure counterpart, not chased at alpha = 0.

## 5. Honest-findings ledger

1. The per-azimuth Cf-front instrument shipped with two bugs in its first
   pass: (a) float azimuth grouping (round to 1e-6 deg) SPLIT node lines
   and decimated each line's x sampling; (b) the search window (0.3,
   0.98) included the near-tail Cf dip (~0.977) which undercuts the
   pre-transition minimum and hijacked the k = 1.5 crossing — together
   they manufactured front spreads of p2p 0.02-0.11.  Fixed (exact
   2.25-deg cell grouping, window 0.5-0.95); the quoted spreads
   (rms 3.9e-4 / 6.4e-5) are from the fixed instrument.  Lesson recorded
   in the script docstring comments.
2. dCp ring statistics are storage-precision-limited: p in the slice
   files is float32, quantum ~1.7e-5 Cp (visible as the green staircase
   in the rings figure); rms values at or below ~1e-6 mean "zero to
   storage precision".
3. forces_tail reads the restart-leg CSV whose step counter renumbers
   from 0: printed "step 22999" = 43k total (A, unstr), "4999" = 25k
   total (B).  CD/CL match the 0413/0350 tables.
4. The march separates at x = 0.880 on this u_e, so 0.88 has no laminar
   reference and 0.84's march value is within 0.04 L of separation
   (steepening H) — the +0.038 there is an upper bound on the physical
   departure.
5. No off-wall ring at x = 0.90 (no slice written there; adding one
   would need a rerun, out of scope) — wall rings + the front-spread
   measure cover that station.
6. Everything here is from settled-CD states, not converge_by_xtr
   stability batches (0413 ledger 7 still applies to front quotes).

## 6. Artifacts

- Script: `paper/repro/cfd/spheroid_uniformity_profiles.py`
  (imports the 0350 facet re-basing, the 0413 ring conventions, the 0105
  march; rerunnable end-to-end, CPU only).
- Figures: `paper/repro/cfd/figs_explore/spheroid_uniformity_rings.png`,
  `spheroid_uniformity_contrast.png`, `spheroid_cp_x.png`,
  `spheroid_bl_profiles.png`; captions:
  `spheroid_uniformity_profiles_captions.md`.
- Numbers: `paper/repro/cfd/figs_explore/spheroid_uniformity_profiles.json`
  (all rings/stats/profiles/cp for the three arms + conventions + the
  0413 gate).
- Cases read (untouched):
  `/local_data/qiqi/sa-ai/spheroid_fv1/case_ogridfull_L1_saai_re72a0`
  (slices + surface), `..._lowmach`, `case_unstr_L1_saai_re72a0`
  (volume + surface); meshes
  `/local_data/qiqi/sa-ai/spheroid_meshes/mesh_re65full_L1.cgns` and the
  unstructured case's own `mesh.cgns` (wall facets for re-basing).
