# Spheroid alpha=0, CONVERGED field: the three-way N(x) decomposition of the 0.42 L front miss

*2026-07-28 11:22, diagnostics thread.  Question (user, verbatim intent):
with everything environmental now clean — mean flow = march (1030),
circumferential uniformity verified (1030), topology/numerics/convergence
exonerated (0350/0413) — why doesn't the model amplify chi and transition?
Converged front 0.858, measurement 0.438: decompose the remaining 0.42 L.
Method: the cylinder three-way instrument set of 2026-07-28-1041
(fpg_rate_audit.py) transplanted onto the spheroid.  CPU only, no solver
runs, NO tex edits.  Script: `paper/repro/cfd/spheroid_threeway.py`
(committed; every number below regenerates from it).  Primary case: the
converged (43k) full-body arm A
`/local_data/qiqi/sa-ai/spheroid_fv1/case_ogridfull_L1_saai_re72a0`;
cross-check `case_ogrid_L1_saai_re72a0_ext`.  This supersedes the e^N
section of 2026-07-28-0040, which ran on the 20k-transient L2 state.*

**Headline: the model is not rate-starved on this flow, and no e^N-faithful
method transitions at 0.438 at this case's seed.**  On the converged mean
flow the transported chi tracks the Drela-Giles envelope one-for-one to
x/L ~ 0.7; even a perfect envelope tracker at the case's flight-quiet
budget N = ln(1/chi_inf) = 11.65 commits only at **0.735** (solver front
0.858, measured 0.438).  Of the 0.420 L miss, **0.30 L is disturbance
-content class** (seed budget + envelope-fit-vs-TS-method offset) and
**0.12 L is the model's residual vs Drela at the same seed**, which is a
transport-realization loss (3.5 e-folds) net of a frozen-rate surplus.

## 0. Verified inputs and instrument gates (all passed)

- **Seed (verified, not trusted from the brief):** the case's Flow360.json
  freestream and farfield BC both carry
  `modifiedTurbulentViscosityRatio = 8.76e-6`; arm A's `ai_constants.log`
  echoes `ai_laminarSlowdown = 1.0` (NO pre-compensation in this campaign,
  unlike the airfoil JSONs) -> physical chi_inf = 8.76e-6, budget
  **N = ln(1/chi_inf) = 11.645**.  Mack-map equivalent Tu = 0.0103%
  (flight-quiet).  Field verification on BOTH cases: the near-nose
  near-wall chi plateau reads 1.001x (A) / 1.006x (ext) chi_inf.
- Marcher (axisymmetric implicit laminar BL, spheroid_a0_meanflow):
  Blasius H = 2.5906 (2.5905), cf*theta*ue/nu = 0.2206 (0.2205), IC/grid
  variants within 0.0008.  mfoil formula cross-checked against a direct
  get_damp call to 1e-10 at three (Hk, Ret) points.
- Extraction gate: the sweep reproduces the 0413 verdict table (H(0.42)
  2.5549, front chi = 1 at 0.8581 / c_v1 at 0.8943); u_e peak 0.4974 and
  march separation 0.880 match 1030.  Forces settled (CD drift
  -2.1e-5/1k both cases).
- Cross-check case (ext, independent half-body O-grid, own mesh/facets,
  volume probe): front 0.8582/0.8948, transported curve indistinguishable
  (dashed under the solid in the figure).  Its restart-leg stdout carries
  no ai-constants echo (honest-findings 1).
- Measured front **0.4381** and Stock's own e^N (N_TS = 8) front
  **0.4245**: committed digitization
  `paper/data/stock2006_fig14a_digitized.json` (Kreplin/DFVLR hot films
  via Stock AIAA J 44(1) 2006, Fig. 14a).

## 1. The three-way N(x) (figure `paper/figs/spheroid_threeway_N.pdf`)

All three curves on the march of the converged field's own u_e(x), nose to
laminar separation (0.880); budget levels from the case seed.

| curve | x at N = 11.645 | N at measured 0.4381 | N at solver front 0.8581 |
|---|---|---|---|
| Drela-Giles envelope (chain rule on the march) | **0.735** | 5.24 | 20.80 |
| mfoil/XFOIL envelope (cross-check) | 0.744 | 4.98 | — |
| model frozen-profile gated eigenvalue (c_nu = 1/6) | **0.673** | 7.57 | 15.16 |
| model kernel sup bound (unconfined) | 0.401 | 12.57 | 21.64 |
| transported chi (solver field, near-wall max) | **0.8581** (= the front) | 5.62 | 11.645 (= budget, by definition) |

Envelope amplification onset (Rt crosses Rt0(H)): x = 0.100 (Drela) /
0.114 (mfoil) — the layer is envelope-unstable from just aft of the nose;
the transported curve leaves the seed plateau at the same station class
(figure).  The gate is not binding anywhere (2142/1041 confirmed again).

**The striking feature: the transported curve IS the Drela envelope to
x ~ 0.7** (N_tr vs N_DG: 1.9 vs 1.7 at 0.20; 5.4 vs 5.0 at 0.42; 7.2 vs
7.1 at 0.55; 9.1 vs 10.4 at 0.70) — the model books e^N-faithful
amplification on its own converged mean flow through the entire mid-body.

## 2. Rate ratio along the body (model/Drela, per station)

March H = 2.544 -> 2.581 (0.10 -> 0.42), 2.617 (0.70), 2.769 (0.85);
Re_theta 394 -> 2066; kernel maxP 0.060 -> 0.139 (savgol W=61 on the
marched profiles).

| x/L | 0.10 | 0.20 | 0.30 | 0.42 | 0.55 | 0.70 | 0.80 | 0.85 |
|---|---|---|---|---|---|---|---|---|
| eig/Drela | 0.93 | 1.44 | 1.45 | 1.27 | 0.97 | 0.55 | 0.29 | 0.19 |
| sup/Drela | 2.13 | 2.30 | 2.06 | 1.68 | 1.21 | 0.64 | 0.32 | 0.20 |
| Drela dN/ds [/L] | 18.9 | 15.5 | 14.5 | 14.9 | 17.9 | 30.2 | 68.8 | 125.8 |

The FS-ladder expectation for this H class (committed fpg_rate_audit.json:
mean-secant 0.74/0.80/0.84, late-secant 0.92/0.98/1.00 at H = 2.48/2.53/
2.59) is CONFIRMED and, if anything, exceeded: through mid-body the frozen
instrument runs at or above Drela.  The collapse to 0.2-0.3x is confined
to the aft steepening-adverse run (x > 0.7, H -> 2.77 approaching the
0.880 laminar separation), where the envelope's demanded rate explodes
30 -> 126 /L and neither the frozen instrument nor the transported field
keeps pace.  This is a different regime from the 1041 FPG cliff (that was
H <= 2.35 favorable; here the deficit is the steep-ADVERSE blow-up).

## 3. The decomposition of the 0.420 L (front-position bookkeeping)

| segment | Delta x [L] | attribution |
|---|---|---|
| 0.4381 (measured) -> 0.600 (Drela N=8) | **+0.162** | envelope-method offset: Stock's own TS e^N books N=8 by 0.4245 where the Drela-Giles fit (the model's calibration target; mfoil concurs) books only 5.0-5.2 — the H-based envelope family is ~2.8 e-folds slow vs full TS theory on this flow |
| 0.600 -> 0.735 (Drela N=8 -> 11.645) | **+0.135** | seed budget: tunnel-class N=8 vs the case's flight-quiet 11.645 |
| 0.735 -> 0.673 (Drela -> model frozen eig at budget) | **-0.061** | frozen-profile rate SURPLUS (instrument hotter than Drela through mid-body) |
| 0.673 -> 0.8581 (frozen eig -> transported) | **+0.185** | transport realization: 3.5 e-folds eaten by the front |
| total | **+0.420** | |

E-fold bookkeeping at the solver front 0.8581: Drela 20.80 = transported
11.645 + transport gap 3.52 (eig 15.16) + net rate deficit 5.64.  Segment
split (JSON `segments`): nose -> 0.70 books DG 10.44 / eig 12.08 / tr 9.13
(the eig is +1.6 AHEAD of Drela; transport eats 2.95 of it, leaving the
transported curve 1.3 below Drela — the on-calibration pattern: at the
Blasius calibration point the transported march runs 0.84-1.00x Drela
while the frozen eig runs 1.1-1.3x, 1041 tables); the aft run 0.70 ->
0.8581 books DG +10.36 / eig +3.08 / tr +2.52 — ALL of the model's
booking deficit vs Drela accrues here.

## 4. Seed sensitivity (analysis only; the lever, priced)

x(N) crossings; "transported" = the model's own curve re-read at a reduced
budget (first-order seed remap, honest-findings 3):

| budget N | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 11.645 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| x, Drela envelope | 0.354 | 0.422 | 0.487 | 0.547 | 0.600 | 0.646 | 0.685 | 0.718 | 0.735 | 0.745 | 0.767 |
| x, transported (model) | 0.330 | 0.397 | 0.465 | 0.537 | 0.615 | 0.689 | 0.764 | 0.827 | 0.858 | 0.873 | 0.891 |

- At tunnel-class budgets the model's own front sits within 0.03 L of the
  envelope's (N = 7: 0.537 vs 0.547; N = 9: 0.689 vs 0.646) — the seed
  lever works at full strength on the converged field (the 0040-era
  "front barely moves with seed" was a property of the transient stalled
  state, not of this solution).
- **What the measurement implies:** by 0.4381 the Drela envelope on this
  flow accumulates N = 5.24 (mfoil 4.98); within envelope physics the
  measured front therefore corresponds to chi_inf ~ 5.3e-3, Mack
  Tu ~ 0.149%.  The model's transported curve reads N = 5.62 there
  (chi_inf ~ 3.6e-3, Tu ~ 0.127%): **re-seeded to a DFVLR-tunnel-class
  disturbance level, the model's own front lands on the measurement.**
  Stock's assignment of N_TS = 8 to the same station says his TS method
  amplifies ~2.8 e-folds more than the Drela-fit family by mid-body —
  the remainder of the same disturbance-content bucket.

## 5. The answer (one paragraph)

The model does not transition at 0.438 because at this case's seed nothing
e^N-faithful does.  On the converged mean flow — which IS the laminar
march (1030) — the transported chi tracks the Drela-Giles envelope
one-for-one to x/L ~ 0.7, the frozen-profile instrument runs at 0.9-1.45x
the Drela rate through mid-body (at or above the FS-ladder expectation for
this H class), and even a perfect envelope tracker with the case's
flight-quiet budget ln(1/chi_inf) = 11.65 commits only at 0.735 vs the
measured 0.438.  Of the 0.420 L miss, 0.30 L is disturbance-content class
— 0.135 L tunnel-vs-flight seed (N 8 vs 11.65) plus 0.162 L
envelope-method offset (Stock's TS e^N books N = 8 by 0.425 where the
Drela-Giles fit the model is calibrated to books ~5) — and the model's own
residual vs Drela at the same seed is 0.12 L: a 0.185 L transport
-realization loss (3.5 e-folds by the front, 2.9 of them the known
transported-vs-frozen offset already present at calibration) net of a
0.061 L frozen-rate surplus.  The only genuinely deficient model physics
on this flow is confined to the final steepening-adverse run 0.70 -> 0.86
(H 2.62 -> 2.77 near laminar separation), where the envelope demands
30 -> 126 e-folds/L and the model delivers ~20: that is why its front
parks just ahead of laminar separation (0.858, separation 0.880) instead
of at the envelope's 0.735.  Priced levers: a tunnel-class seed moves the
model's front to 0.69 (N = 9) / 0.54 (N = 7) and at the measurement-implied
budget (~5.6 e-folds, Tu ~ 0.13%) onto 0.438 itself; no rate or transport
fix can reach 0.438 at the flight-quiet seed — the correlations forbid it
by 6.4 e-folds.

## 6. Honest-findings ledger

1. The ext case's restart-leg stdout carries NO ai-constants echo (and no
   ai_constants.log); its seed is verified from Flow360.json plus the
   in-field near-nose chi plateau (1.006x chi_inf) rather than an echo.
2. The march runs on the converged RANS u_e, which contains the model's
   own front feedback aft of ~0.83; nose -> 0.7 is insensitive (Cp there
   matches the azimuthal average to 1.4e-4, 1030).  In a flow actually
   tripped at 0.438 the aft u_e/H would differ entirely — the
   decomposition is self-consistent for THIS converged solution, which is
   what the question asks.
3. The transported seed-lever row is a first-order remap of the frozen
   converged curve (chi linear in seed); local slope dN_tr/dx = 13-16/L
   near tunnel budgets -> +-0.5 e-fold reads as +-0.03 L.  Real re-seeded
   fronts would also move the handover/feedback region (not re-run here;
   analysis only per brief).
4. The frozen-profile instrument uses the planar-convention kernel
   (savgol W=61, the calibrated 0040/1041 estimator) on marched — not
   RANS — profiles; 1030 shows the two profile sets coincide
   (|dH| <= 0.008 to x = 0.7, <= 0.038 at 0.84), so the substitution is
   controlled.  The sup row is the unconfined pointwise bound (always
   high); conclusions are keyed to the gated eigenvalue.
5. The e-fold and the front-position decompositions differ by
   construction (Drela keeps booking e-folds past the budget in the aft
   blow-up): the x-table in Sec 3 is the answer to "decompose the 0.42 L";
   the at-front e-fold split (5.6 rate / 3.5 transport) is stated for
   completeness and is aft-concentrated.
6. Supersedes 2026-07-28-0040 Sec 2/5 where they touch e^N: that analysis
   ran on the 20k-transient L2 state (full profiles H ~ 2.49, front
   0.924, "mean-flow fullness dominant") — on the converged field the
   fullness bucket vanishes and the front is 0.858.
7. Fronts quoted are settled-CD states, not converge_by_xtr stability
   batches (0413 ledger 7); the A-vs-ext agreement to 1e-4 and the
   azimuthal front rms 3.9e-4 (1030) bound the residual front motion well
   below the 0.02-0.19 L buckets above.

## 7. Figure caption (for integration; NO tex edits made)

`paper/figs/spheroid_threeway_N.pdf` — 6:1 spheroid, alpha = 0,
Re_L = 7.2e6, converged full-body O-grid: transported near-wall chi
(solid; dashed: independent half-body case, indistinguishable) on the log
left axis, and on the matched linear right axis (N = ln(chi/chi_inf),
chi_inf = 8.76e-6 from the case) the Drela-Giles envelope, the mfoil
envelope, and the model's frozen-profile instrument (kernel sup bound and
gated c_nu = 1/6 eigenvalue), all integrated on the validated
axisymmetric laminar march of the field's own u_e (curves end at laminar
separation, 0.880).  The chi = 1 level and the seed budget N = 11.65
coincide by construction; N = 9 / N = 7 mark tunnel-class budgets.
Triangles: measured transition 0.438 (filled; Kreplin hot films via Stock
Fig. 14a) and Stock's e^N (N_TS = 8) front 0.425 (open).

## Artifacts (full paths)

- Script: `/home/qiqi/flexcompute/sa-ai/paper/repro/cfd/spheroid_threeway.py`
  (rerunnable end-to-end, CPU, ~8 min)
- Figure (paper-quality): `/home/qiqi/flexcompute/sa-ai/paper/figs/spheroid_threeway_N.pdf`;
  preview `/home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/spheroid_threeway_N.png`
- Numbers: `/home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/spheroid_threeway.json`
- Cases read (untouched):
  `/local_data/qiqi/sa-ai/spheroid_fv1/case_ogridfull_L1_saai_re72a0`
  (slices + surface + mesh facets via
  `/local_data/qiqi/sa-ai/spheroid_meshes/mesh_re65full_L1.cgns`),
  `/local_data/qiqi/sa-ai/spheroid_fv1/case_ogrid_L1_saai_re72a0_ext`
  (volume + own mesh.cgns)
- Committed references: `paper/data/stock2006_fig14a_digitized.json`
  (measured 0.4381, TS front 0.4245),
  `paper/repro/cfd/figs_explore/fpg_rate_audit.json` (FS-ladder ratios)
- Cross-referenced records: 2026-07-28-1041 (method), 1030 (mean flow =
  march; uniformity), 0413/0350 (convergence/topology), 0040 (superseded
  where noted), 2026-07-27-2142 (flank feed-limit)
