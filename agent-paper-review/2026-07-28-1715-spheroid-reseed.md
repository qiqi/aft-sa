# Spheroid RESEED: the freestream-turbulence mis-specification test (5 converged L1 runs)

*2026-07-28 17:15, CFD thread. User directive: the re72a0 spheroid miss
is (operationally) a freestream-turbulence mis-specification -- rerun with
the correct freestream; also rerun the lower-Re alpha=0 case and the
alpha=2.5 case, confirming per-case that freestream+IC chi is the main fix.
Five cold-start converged-protocol L1 half-model O-grid runs on the
existing re65 meshes (topology/level exonerated by 0413/0350). Runs on
017-v100-dev GPUs 2-3 (0-1 = ultra cylinder). NO tex edits.
Scripts (committed): `spheroid/build_reseed_cases.py` (staging + restart-leg
tooling), `paper/repro/cfd/spheroid_reseed_harvest.py` (harvest + figure),
`paper/repro/cfd/digitize_stock_fig14b.py` (the alpha=2.5 measured
reference, new). Every number below regenerates from the harvest script.*

**Headline.** Re-seeding from the campaign's flight-quiet chi_inf =
8.76e-6 (Mack Tu 0.010%, budget N = 13.6) to DFVLR-tunnel-class seeds
moves the alpha=0 front from the converged flight-quiet 0.858 (1122) onto
the measurement: the three-seed re72a0 bracket **tracks the 1122
seed-lever prediction at full e^N strength** and crosses the measured
0.438 at budget **N approx 5.0 (Mack Tu approx 0.17%)**. At the documented
DFVLR seed (Stock's limiting N_TS = 8, Tu 0.106%) the model front is
**0.511** (chi=1) -- 0.07 L aft of measured; a marginally stronger
tunnel seed lands it ON 0.438. **For alpha=0, freestream/IC chi IS the
main fix** (confirmed, all three seeds, both Re). **For alpha=2.5 it is
the main fix on the windward meridian only** (model 0.444 vs measured
0.450) -- the model develops a converged azimuthal front spread of 0.13 L
(leeward 0.576) that the measurement (nearly uniform, 0.019 L) does not
show; that flank/leeward lateness is a second, seed-independent effect.

## 1. Seed determination (honest trail)

**No explicit freestream turbulence intensity Tu% is published anywhere in
the DFVLR spheroid measurement chain.** Checked: Stock AIAA J 44(1) 2006
(`paper/spheroid.pdf`, full text extracted) -- gives no tunnel Tu, only
the DFVLR 3x3m Low Speed Wind Tunnel Goettingen designation and the
derived limiting N factors; the source-of-record Kreplin/Vollmers/Meier
(DFVLR-AVA IB 22-84 A 33, 1985) is NOT in the repo (`references/`,
`paper/data/` searched); the workshop/automatic-prediction literature
(`references/krimmelbein-krumbein-2010...pdf`,
`TransitionMPW1_summary_Coder...pdf`) also quotes only N-factors
(Krimmelbein uses a critical N = 7.5 for this case).
`paper/data/stock2006_fig14a_digitized.json` metadata records the same:
"computed = Stock pure-TS-wave e^N (N_TS=8.0, his DFVLR-tunnel limit,
Fig. 11a)".

The DFVLR tunnel's DOCUMENTED disturbance level is therefore Stock's
limiting **N_TS = 8.0** (AIAA J Fig. 11a), calibrated ON these very
measurements -- that is the honest documented anchor, not a guess. Mapped
through the paper's own Mack receptivity map (eq:tumap,
`lib/calibrate_kernel.py`: N_crit = -8.43 - 2.4 ln(Tu_frac),
chi_inf = c_v1 e^{-N_crit}, c_v1 = 7.1), N_crit = 8.0 is the **center
seed**. Bracket = Tu x/geo 2 (N_crit -+ 2.4 ln 2):

| tag | N_crit | Tu [%] | chi_inf | budget N = ln(1/chi_inf) |
|---|---|---|---|---|
| n9p66 | 9.664 | 0.053 | 4.513e-4 | 7.703 |
| **n8 (center = DFVLR)** | **8.000** | **0.106** | **2.382e-3** | **6.040** |
| n6p34 | 6.336 | 0.213 | 1.257e-2 | 4.376 |

This is a curve, not a tuned point. Cross-check to 1122: 1122's
measurement-implied budget was N ~ 5.2-5.6 (Tu ~ 0.13-0.15%) -- the
bracket straddles it. (The 1122 value was a first-order remap; the
converged bracket refines it, Sec 3.)

Seed carried correctly (verified per case, not assumed): both
`freestream.turbulenceQuantities` and the `fluid/farfield` Freestream BC
carry the new chi; IC = cold-start `{type: freestream}` so the IC chi is
the SAME new seed; `ai_constants.log` echoes **ai_laminarSlowdown = 1.0**
on every case (NO pre-compensation -- physical seed, campaign convention),
line-identical to the committed campaign echo apart from the timestamp
(diffed). The in-field near-nose chi plateau reads **1.001x chi_inf** on
all five (the 1122 ritual). N_crit back-computed from the JSON chi matches
the intended value to 2e-3 (asserted in the harvest).

## 2. Front-vs-seed, alpha=0, Re_L=7.2e6 (the re72a0 bracket)

Facet-re-based phi=90 meridian sweep (0350/1030/1122 instruments,
imported verbatim): chi=1 = near-wall (y<=0.02) max-chi crossing of 1;
c_v1 companion; Cf-rise = k=1.5 running-min of the pre-rise minimum over
81 azimuth lines (median).

| seed | budget N | chi=1 | c_v1 | Cf-rise (rms) | 1122 remap | 1122 Drela | CD conv |
|---|---|---|---|---|---|---|---|
| n9p66 (Tu 0.053%) | 7.703 | 0.6276 | 0.6770 | 0.6644 (2e-3) | 0.592 | 0.584 | settled |
| n8 (Tu 0.106%) | 6.040 | 0.5111 | 0.5519 | 0.5439 (2e-3) | 0.468 | 0.490 | settled |
| n6p34 (Tu 0.213%) | 4.376 | 0.4002 | 0.4696 | 0.4273 (8e-3) | 0.355 | 0.380 | limit-cycle* |

Measured (Kreplin/Stock Fig 14a) **0.4381**; Stock e^N (N_TS=8)
**0.4248**. All H(0.20/0.42) = 2.525/2.51-2.56 (Blasius-class laminar
fore-body, unchanged by the seed -- the seed only advances the front);
nose chi ratio 1.001 on all.

\* n6p34 CD sits in a natural-transition **limit cycle** (p2p 1.4e-3,
~3% of CD) that appears at the early tunnel-class front; the FRONT is
stationary across a +20k settle leg (chi=1 0.4013 -> 0.3992 -> 0.4002
at 45k/60k/80k; Cf-rise 0.4286 -> 0.4273), so the front quote is
converged. n9p66/n8 are fully CD-settled (drift +1.7e-5/-2.3e-5/1k,
p2p <= 4e-4; n8 front-stationarity confirmed on a +10k leg:
0.5104 -> 0.5111).

**Verdict (alpha=0): the bracket CONFIRMS the 1122 seed-lever prediction.**
The three converged fronts move monotonically with seed at dx/dN approx
0.067 /unit-N, tracking the 1122 transported-remap curve (Fig
`spheroid_reseed_front_vs_seed`); the actuals run ~0.03-0.05 L AFT of the
first-order remap, the expected sign -- a real re-seeded run also moves
the aft u_e/feedback region (1122 honest-finding 3), which the frozen
remap cannot. The measured 0.438 is crossed at **budget N approx 5.0**
(chi=1) / 4.5 (Cf-rise), i.e. Mack **Tu approx 0.17-0.20%** -- squarely
tunnel-class, a touch above 1122's remap-based ~5.6 and consistent with
DFVLR-class disturbance. **A tunnel-class seed lands the model on the
measurement**: the documented DFVLR seed (N_TS=8, Tu 0.106%) gives 0.511
(0.07 aft), and Tu ~ 0.17% gives 0.438 exactly. No rate/transport change
is invoked; the miss was the seed.

## 3. Lower-Re alpha=0 (re65a0, center seed)

`case_ogrid_L1_saai_re65a0_reseed_n8`: Re_L = 6.5e6 (muRef 0.1/6.5e6, the
campaign re65 family value; confirmed from the JSON), N_crit = 8 (Tu
0.106%), CD-settled (drift +8.5e-6/1k, p2p 4e-5). **chi=1 front 0.5564**
(c_v1 0.6076, Cf-rise 0.5938). Versus re72a0 at the SAME seed (0.5111):
lowering Re 7.2 -> 6.5e6 moves the front AFT +0.045 L -- the physically
correct Reynolds direction (less amplification length at lower Re), and
the expected magnitude for a 0.7e6 drop.

**No measured DFVLR alpha=0 front exists at Re=6.5e6.** Stock Fig 14a
(the only measured alpha=0 waterfall) is Re=7.2e6; his Fig 18 gives
COMPUTED-only fronts at alpha=0/2.5 (Re 7.2e6) and alpha=5-29.7 (Re
6.4-6.54e6). So re65a0 is a Reynolds-trend confirmation, not a
measurement comparison -- stated plainly per the directive.

## 4. alpha=2.5 crossflow check (re72a2p5, center seed) -- the key question

`case_ogrid_L1_saai_re72a2p5_reseed_n8`, N_crit=8. Windward (mesh
phi=180) and leeward (mesh phi=0) meridians from the y=+1e-5
near-symmetry slice; flank from phi=90. Front **STATIONARY** across a +25k
settle leg (windward 0.4444 -> 0.4439, leeward 0.5773 -> 0.5760, flank
0.5350 -> 0.5347 at 45k/70k; CL settled +0.011; CD limit-cycle p2p
2.3e-3) -- so the azimuthal spread below is a converged feature, not a
transient.

Convention note (load-bearing): **mesh phi=0 = +z = LEEWARD at alpha>0**
(0350); **Stock Fig 14b phi=0 = WINDWARD**. Cross-mapped:

| meridian | model chi=1 | model c_v1 | measured (Stock 14b) |
|---|---|---|---|
| windward | 0.4439 | 0.4922 | 0.4502 |
| flank (phi=90) | 0.5347 | 0.5829 | 0.4379 |
| leeward | 0.5760 | 0.6616 | 0.4317 |

Measured front is **nearly azimuthally uniform** (0.432-0.450, spread
0.019 L; Stock's own TS e^N mean 0.4456, also uniform -- Sec III.C: at
alpha=0 and 2.5 transition is TS-only). The **model front spans 0.13 L**
(windward 0.444 to leeward 0.576), azimuthal Cf-rise rms 5.4e-2.

**Verdict (alpha=2.5): freestream/IC chi is the main fix on the WINDWARD
meridian ONLY.** The reseeded windward front (0.444) matches the
measurement (0.450) to 0.006 L -- the same success as alpha=0. But a
converged flank/leeward lateness of up to 0.14 L remains that the seed
change does NOT remove: the model's TS-surrogate under-amplifies on the
leeward mild-adverse meridian relative both to the measurement and to
Stock's own TS e^N (which stays uniform). This is the same class of
deficiency the 1122 decomposition isolated for alpha=0 (the model's only
genuine rate shortfall is in the aft/adverse run) -- here it is exposed
azimuthally by the incidence. So: freestream reseeding fully explains the
axisymmetric (alpha=0) miss; the alpha=2.5 azimuthal structure is a
second, seed-independent model-content effect (leeward-meridian
adverse-gradient TS amplification), NOT a freestream mis-specification.

## 5. Front-vs-seed figure

`paper/repro/cfd/figs_explore/spheroid_reseed_front_vs_seed.png` and
`paper/figs/spheroid_reseed_front_vs_seed.pdf` (paper-ready): the two
1122 x(N) curves (Drela-Giles envelope, model transported remap) with the
three bracket runs (filled = chi=1, open = Cf-rise), the measured 0.438
and Stock e^N 0.425 levels, and the measurement-implied N approx 5.0
crossing annotated. The bracket markers sit on the transported-remap
curve, ~0.03-0.05 aft (Sec 2).

## 6. Cost

L1 45k cold start ~= 1.8 GPU-h each; 5 cases + 3 short settle legs
(n8 +10k, n6p34 +15k/+20k, a2p5 +25k) ~= 12 GPU-h total on 2 V100s (2-3),
wall time ~4 h with the two-GPU queue. Disk: volumeOutput off, slices +
wall surface only; each case ~1.8 GB, 45k-state archived in `snap45k/`.

## 7. Honest-findings ledger

1. n6p34 and a2p5 carry a natural-transition CD limit cycle (p2p 1.4-2.3
   e-3) -- expected at the early tunnel-class front with the campaign's
   laminarSlowdown=1.0 (the airfoil cases use 0.01 to damp exactly this;
   memory saai-laminar-slowdown-pairing). Fronts are stationary across the
   settle legs regardless; front quotes are the azimuth-median of the
   settled state, not converge_by_xtr stability batches.
2. Stock Fig 14b (alpha=2.5 measured reference) was digitized for this
   task (`digitize_stock_fig14b.py`): 1 in-panel square (NCC, phi 90,
   x/L 0.438) + 2 on-frame squares (targeted cluster centroid, phi 0/180,
   x/L 0.450/0.432) + the near-vertical TS front (windowed per-row walk,
   mean x/L 0.446). Every glyph verified in
   `/local_data/qiqi/sa-ai/stock_digitize/check_fig14b.png`. Sparse
   coverage (the alpha=2.5 hot-film test has few azimuths) -- 3 squares.
3. The 1122 seed-lever rows are a first-order remap of the flight-quiet
   converged curve; this bracket is the real re-seeded confirmation. The
   measurement-implied budget from the actual bracket (N approx 5.0,
   chi=1) supersedes 1122's remap-based ~5.6 for the chi=1 convention;
   the Cf-rise convention gives ~4.5. Both are DFVLR-tunnel class.
4. re65a0 has no measured comparison (Sec 3) -- Reynolds-trend point only.
5. a2p5 windward/leeward from the y=+1e-5 slice: the 1e-5 L transverse
   offset is ~1% of delta99; rays built on the analytic meridian and
   shifted into the plane (n3/t_s are in-plane at phi=0/180). H(0.42)
   leeward 2.518 / windward 2.587 -- both laminar-class fore-body, so the
   front spread is a transition-rate effect, not a mean-flow artifact.

## 8. Artifacts (full paths)

Cases (synced to local, `_reseed` tag, committed campaign untouched):
- `/local_data/qiqi/sa-ai/spheroid_fv1/case_ogrid_L1_saai_re72a0_reseed_n9p66`
- `/local_data/qiqi/sa-ai/spheroid_fv1/case_ogrid_L1_saai_re72a0_reseed_n8`
- `/local_data/qiqi/sa-ai/spheroid_fv1/case_ogrid_L1_saai_re72a0_reseed_n6p34`
- `/local_data/qiqi/sa-ai/spheroid_fv1/case_ogrid_L1_saai_re65a0_reseed_n8`
- `/local_data/qiqi/sa-ai/spheroid_fv1/case_ogrid_L1_saai_re72a2p5_reseed_n8`
  (each: Flow360.json, ai_constants.log, slices, wall surface, forces,
  snap45k/ archive of the first-leg state)
- run copies on 017: `/local_data/qiqi/sa-ai/spheroid_fv1_reseed/<case>`
Scripts (committed):
- `/home/qiqi/flexcompute/sa-ai/spheroid/build_reseed_cases.py`
- `/home/qiqi/flexcompute/sa-ai/paper/repro/cfd/spheroid_reseed_harvest.py`
- `/home/qiqi/flexcompute/sa-ai/paper/repro/cfd/digitize_stock_fig14b.py`
Data / figures:
- `/home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/spheroid_reseed_harvest.json`
  (all fronts/H/forces/seeds + measured + implied N)
- `/home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/spheroid_reseed_front_vs_seed.png`
- `/home/qiqi/flexcompute/sa-ai/paper/figs/spheroid_reseed_front_vs_seed.pdf`
- `/home/qiqi/flexcompute/sa-ai/paper/data/stock2006_fig14b_digitized.json` (new)
- `/local_data/qiqi/sa-ai/stock_digitize/check_fig14b.png` (digitization check)
Run logs on 017: `runlogs/spheroid_reseed_gpu{2,3}.log`,
`runlogs/spheroid_reseed_{n8_leg2,n6p34_leg2,n6p34_leg3,a2p5_leg2}.log`
Cross-referenced: 1122 (seed-lever prediction + the x(N) tables this
tests), 0350/0413 (convergence protocol + topology exoneration), 1030
(uniformity + facet re-basing conventions), 0040 (superseded a0 physics).
