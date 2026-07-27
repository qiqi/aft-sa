# Drag-crisis PILOT: subcritical cylinder at Re_D=1e5 — URANS + steady twin + frozen-field kernel check

*2026-07-27, CFD pilot thread (executes the "Recommendation" paragraph of
2026-07-27-1540-dragcrisis-cylinder-feasibility.md, plus a user-amended
STEADY pseudo-time twin). All runs on 017-v100-dev (one V100 at a time;
GPUs checked idle before every launch). Scripts committed under
paper/repro/cfd/: build_dragcrisis_pilot.py, run_dragcrisis_pilot.py,
dragcrisis_pilot_forces.py, dragcrisis_kernel_check.py. Case data:
/local_data/qiqi/sa-ai/dragcrisis_pilot/ (both machines). Exploratory
figures: paper/repro/cfd/figs_explore/dragcrisis_* (NOT paper figures).*

## 1. Mesh (local, no GPU)

- D=1 circle centered (0.5, 0); ANALYTIC radial O-grid written in
  Construct2D .p3d convention; everything downstream is byte-identical
  machinery to flow360/build_eppler_struct_cases.py (.p3d -> one-cell-span
  quasi-2D gmsh .msh -> CGNS -> rans.case.preprocess -> Flow360.json).
- Construct2D itself FAILS on a closed circle (its airfoil surface split
  puts wall points at the far field: max skew 90 deg, eta-growth 6e3).
  For a circle the analytic grid IS the ideal O-grid (radial normals =
  exactly the hyperbolic march's answer), with exact y1/growth control.
- GOTCHA (cost a debugging round): the wall loop must be CLOCKWISE
  (Construct2D's convention, verified on proper_struct_eppler_L1.p3d);
  counterclockwise inverts the extruded hexes and MeshProcessor fails
  dual-closedness (ERROR 2492) at a max-area-vector node near the seam.
- Verified statistics (mesh_stats.json in the case dir):
  N_s = 1200 uniform (ds = 2.618e-3 D, uniformity 1+3e-13); y1 = 8.0e-6 D
  exact (spread 4e-16); growth 1.0994; 148 layers; outer radius 100 D
  exact; 177,600 quads (Eppler-L1/L2 class); wall AR 327.
  y+ estimates: 0.057 at Re 1e5 (laminar shoulder Cf ~ 0.01), 0.64 at
  Re 2e6 turbulent — ONE grid family serves the whole campaign matrix.

## 2. URANS case + protocol

- Re_D 1e5, M 0.1, muRef 1e-6, refArea 0.1 (D x span 0.1), alpha 0.
- Dual time: dt = 0.1 solver units = 0.01 D/U (526 steps/period at
  St 0.19); ramp CFL 1->100 over 15 pseudo steps, maxPseudoSteps 24,
  relTol 1e-2; in-step residual drop ~2 orders (verified at step 4700:
  momx 4.0e-6 -> 2.2e-8 by pseudo 23).
  - FINDING: the campaign's ADAPTIVE CFL is a steady policy and does not
    work in dual time — it resets to CFL ~0.1 at the start of EVERY
    physical step and crawls to ~4 within the pseudo budget (residual
    fell only 2-3x/step at 0.54 s/step). Diagnosed on the first launch
    via cfl_v2.csv; switched to ramp CFL (documented in the builder).
- fv1 canon env PROVEN: the solver's resolved-constants echo is identical,
  line by line, to flow360_fv1/strL1prop_eppler387_Re200k_a5/
  ai_constants.log (rateScale 0.19, reOmCeil 1851.2, reOmA 124.6,
  reOmB 1.424, rampWidth 0.35, switchWidth 4, sigmaDTie 1, switchWidthD
  1.36, maxBlend 1, nuLamScale 0.166667, fv1Bypass 1, fv1SwitchWidth 1,
  invariantKernel 0), with ai_laminarSlowdown per stage. Remote solver =
  explore-lambda-v @ 05d05a123b (canon commit), clean install.
- Seed: chi_inf = 8.76e-4 (LTPT-class, c_v1 e^-9); the JSON BC seed is
  ALWAYS chi_inf * fSlow (pre-compensation hard rule); the outer-field chi
  in the results reads 8.76e-4 — compensation verified end-to-end.
- Stages (run_dragcrisis_pilot.py; each solver invocation TRUNCATES the
  force/residual CSVs — observed; the runner now archives per stage):
  * kick    0->800,   alpha=3, fSlow 0.1 — Travin-style symmetry kick
    (ends CL -1.51, CD 1.57);
  * develop 800->4000, alpha=0, fSlow 0.1 — starting-vortex sloshing
    (CL swings +-2), settling toward the shedding limit cycle;
  * prod   4000->22000, fSlow 0.01, seed 8.76e-6 — time averaging from
    step 6000 (surface + slice; 16000 steps = 31.0 shedding periods).
- Time-averaged outputs at the SOLVER level require top-level
  timeAverageSliceOutput / timeAverageSurfaceOutput sections mirroring the
  instantaneous ones (Flow360Solver.cpp); the "computeTimeAverages" flag
  alone is an SDK-preprocessing input and does NOTHING on an
  already-preprocessed Flow360.json. Discovered by source read; the stage
  driver injects the sections (running-average snapshots every 2000 steps
  + final write; averaged fields are restart-persistent).

## 3. STEADY twin (user amendment)

Same mesh, same canon env; the campaign steady solver blocks verbatim
(adaptive CFL, timeStepSize inf, absTol 1e-30, relTol 0); the staged-fSlow
protocol (steady1 fSlow 0.1, steady2 fSlow 0.01 restart, seeds
re-compensated). Graceful stop.json at CD flatness.

RESULT: converges CLEANLY to the SYMMETRIC (time-unstable) solution — no
limit cycle, no divergence: CL tail p2p 5e-8; CD drift 5.8e-4 per 1000
pseudo steps at stop (CD 0.835 -> 0.838 over the last 4000; asymptote
~0.84-0.85). steady1 12130 pseudo steps in 171 s; steady2 7260 in 105 s.
The pseudo-transient acts like heavy selective damping and finds the
unstable symmetric branch — a reproducible, protocol-clean object, but NOT
the physical mean flow.

## 4. Results vs the subcritical experimental band

| quantity | STEADY twin | URANS (mean, steps 6000-22000) | experiment (subcritical) |
|---|---|---|---|
| Cd | 0.836 (tail median; +0.5% drift/1k) | 1.720 (rms 0.29) | 1.2 +/- 0.05 |
| CL mean | 0 (symmetric, 1e-9) | +0.003 | 0 |
| CL rms | 5e-8 | 1.18 | ~0.1-0.5 (3D, facility-dep.) [mem] |
| St | no shedding | 0.194 (spectrum peak; band 0.14-0.25) | 0.19-0.20 |
| laminar separation | 73.8 deg (mean-Cf crossing) | knee of mean Cf 81-84 deg; instantaneous crossing 85.1 deg (windward-phase side) | 78-80 deg |
| shoulder Cp (suction knee) | -0.91 @ 65 deg | -1.33/-1.31 @ ~75-80 deg | ~ -1.2 [mem] |
| Cp base | -0.616 | -2.03 | ~ -1.2 [mem] |
| chi=1 near-wall front | 87 deg | ~113 (lower) / ~140+ (upper) | (wake-SL transition, see below) |

Reading:
- The known 2D-URANS bias is CONFIRMED and quantified on our model/mesh:
  Cd +43% (1.72 vs 1.2), CL rms ~1.2, base suction -2.0 vs -1.2 — at the
  high end of the documented 1.4-1.6 band. St is ON the experimental band
  (0.194). Separation-angle class quantities land within ~5 deg of
  experiment for BOTH solutions (steady 74, URANS mean-knee 81-84,
  instantaneous 85; exp 78-80).
- The steady symmetric branch under-predicts base suction and Cd by
  30-45% (no shedding) — the classical steady-RANS cylinder result. Its
  BL-side quantities (separation angle, front) are however close to the
  URANS mean's, at 1/30 the cost.
- Mean-Cf sign-change convention DEGRADES under shedding: the sweeping
  separation rectifies the mean tangential Cf to slightly positive almost
  everywhere aft of the knee (single crossing at ~175 deg upper, none
  lower). Use the Cf knee (fall below 20% of the laminar peak) or
  phase-resolved crossings on cylinders; noted for the campaign extractor.
- The near-wall chi=1 "front" convention (airfoil-style wall band) is
  also weak on a bluff body: the transitioning shear layer is LIFTED, so
  the wall band goes turbulent only in the base region (~113 deg+ mean),
  while the actual transition happens off-wall (next section). Wall-band
  asymmetry upper/lower reflects the early averaging window (steps
  6000-7500 still relaxing, CL mean -0.16 there).

## 5. Frozen-field kernel check (risk 1 of the 1540 note) — VERDICT

Machinery: add_derived_to_slice.augment() (the paper slice convention,
n = grad wallDistance) + canon onset constants; wake shear-layer ridges =
per-x |omega| maximum per half-plane, 0.6-3.5 D aft of center; run on
THREE fields: steady slice, URANS time-averaged slice, URANS instantaneous
slice (figures dragcrisis_kernel_check_{steady,avg,inst}.png + JSON).

- Along the separated shear layers just aft of separation (x-x_c
  0.6-1.1 D, d 0.17-0.8 D): Y = omega*d dominates (steady: X~0.09,
  Y~0.32-0.35, Z~-0.15, Omega_hat ~ 0.96; URANS mean: X~0.01, Y~0.18-0.36)
  -> P = Omega_hat*I_hat ~ 0.9-1.1 (saturated), Re_Omega = d^2 omega/nu =
  3e4-3e5 vs Re_Omega_crit ~ 126 -> onset margin 10^2.5-10^3.5, gate hard
  open, rate = a_max. chi along the steady ridge rises 190 -> 700 within
  0.5 D of separation: the model transitions the separated shear layer
  essentially AT separation — the free-shear pole of the sphere kernel
  doing exactly what it was designed to do (P > 0 from inflectional
  Y-dominance; the design intent in ONBOARDING Sec. 1).
- Wall-distance semantics DO degrade at wake distances, as feared:
  angle(grad d, layer normal) = 50 deg at 0.6 D rising to 75-80 deg by
  2-3 D (steady; noisier 10-80 on the mean). BUT the degradation enters
  only through Z (the curvature indicator) — and on the free-shear pole Z
  is subdominant: recomputing P with Z taken along the true layer normal
  changes it by ~6% near separation (1.00 vs 0.94 steady; <10% on the
  mean field out to ~2.2 D), and both clip to rate = a_max. The verdict
  "amplify here" is carried by Y-dominance + the enormous Re_Omega, both
  of which use d only as a magnitude, not a direction.
- Sanity of the OFF states: P < 0 in the attached FPG boundary layer and
  in bands flanking the shear layers; onset gate closed in the front
  stagnation region and the immediate near-wall laminar zone; on the
  instantaneous field the P map resolves the vortex-street structure
  (amplifying braids, stabilized cores) and the rate concentrates in the
  braids/shear layers at 0.5-1.5 D — physically sensible.
- CAVEAT for the matrix (model prediction, not a semantics failure): with
  Re_Omega = d^2 omega/nu, the onset gate is trivially open at wake
  distances — the model has effectively NO onset threshold in the free
  wake; the shear layers transition immediately at separation at Re 1e5.
  Experimentally the wake-transition point moves toward the cylinder as
  Re rises through subcritical; at the matrix's low end (6e4) immediate
  transition may be somewhat early. Watch Cp_base/Cd(Re) at 6e4 vs 1e5.

RISK-1 ANSWER: the kernel sees the wake shear-layer instability for the
right geometric reason; wall-distance-direction degradation is real but
confined to the subdominant Z indicator (<10% on P, invisible after the
clip). No blocker.

## 6. Cost calibration

- URANS (V100, 177.6k quads, in-session single GPU): kick 457 s
  (40 pseudo/step), develop 1258 s + prod 7076 s at 24 pseudo/step =
  0.393 s/step. Total 8791 s = 2.44 GPU-h for 22k steps (transient + 31
  averaged periods). One aborted first launch (adaptive-CFL diagnosis)
  ~0.16 GPU-h. Pilot total ~2.7 GPU-h incl. steady twin.
- STEADY twin: 276 s = 0.08 GPU-h (30x-100x cheaper than URANS; and a
  continuation-ladder restart would converge in a fraction of that).
- Matrix extrapolation (12 Re x 2-3 seeds x up/down + checks ~ 60-80
  cases): URANS design ~ 150-190 V100 GPU-h at L1 (plus L0/L2 grid check
  ~ x0.3/x3 per case). STEADY-first design ~ 10-20 GPU-h for the whole
  matrix + ~2.5 GPU-h per URANS spot check.

## 7. Steady-first vs URANS matrix — recommendation

What the pilot showed: at subcritical, NEITHER branch gives the
experimental Cd (steady-symmetric -30%, 2D URANS +43%; experiment falls
between the branches) — absolute subcritical Cd was already conceded to
the 2D bias in the campaign design. The BL-CONTROLLED quantities —
laminar separation angle (74 vs 81-85 deg), shear-layer transition
location (immediate, both), front behavior — agree between steady and
URANS to a few degrees. The quantities ONLY URANS provides: St (0.194,
on-band), CL rms, phase-resolved Cf, and the base-pressure/Cd dynamics.

Recommendation for the matrix (go, with a restructured design):
1. STEADY-FIRST matrix: run the full Re x Tu x up/down continuation
   ladders with the steady protocol (the run_continuation_ladders.py
   machinery unchanged). This is where Re_crit(Tu), separation angles,
   Cp/Cf distributions, and the transition-front family live — at ~10% of
   the URANS cost. The steady branch converges cleanly at subcritical;
   whether it stays clean THROUGH the crisis (where the LSB and one-bubble
   asymmetric states appear) is untested — treat any steady
   non-convergence there as a finding, not a failure.
2. URANS spot checks at 2-4 Re points (subcritical anchor = this pilot;
   1-2 in the critical range; 1 supercritical) to (a) quantify the 2D
   bias per regime, (b) capture St / hysteresis / the one-bubble state,
   which are intrinsically unsteady-branch physics (Schewe's lift jumps
   cannot exist on the symmetric steady branch).
3. GATE before committing the full matrix: ONE steady+URANS pair in the
   critical range (Re ~ 3e5) to test whether the steady branch tracks the
   drag crisis (turbulent reattachment moving the separation to ~140 deg)
   at all. That pair is another ~3 GPU-h — same class as this pilot.

## 8. Reproducibility

- Build: python paper/repro/cfd/build_dragcrisis_pilot.py [--steady]
  (compute venv python; needs flexfoil/rans + compute install).
- Run: python paper/repro/cfd/run_dragcrisis_pilot.py CASE --stage
  kick|develop|prod|steady1|steady2 --gpu N (self-contained; inlines
  driver/env.py + solve.py + the canonical env; ai_constants.log diff
  is the canon proof).
- Analyze: dragcrisis_pilot_forces.py [--steady], dragcrisis_kernel_check.py.
- Case data (mesh + all outputs incl. time averages):
  /local_data/qiqi/sa-ai/dragcrisis_pilot/cylL1_Re100k{,_steady} on both
  017-v100-dev and the local machine.
