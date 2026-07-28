# FPG recalibration study — floored rate coordinate P, three forms, headline P = Ω̂·softmax₂(⟨Î⟩₊, ε(−Z)/R)

*2026-07-28 12:18, model-calibration study thread (follow-up to the 1041 FPG
audit). OFFLINE study on the paper's analytic rigs: NO tex edits, NO solver
edits; the canonical `paper/figs/model_calibrate.pdf` is untouched. The brief
evolved mid-study through two user redesigns; all three forms are fully
evaluated and recorded, newest form last. Script:
`paper/repro/analytic/fpg_recalibration_study.py` (stage-driven, results
persist per-form to `figs_explore/fpg_recalibration_*.json`).*

## The task and its evolution

1. **Directive 1 (additive):** P = Ω̂(Î+ε) everywhere P appears (rate AND
   onset gate), single ε, Fig-4 both panels to match Drela–Giles down to
   stagnation H = 2.216. Headline ε = 0.03, companions 0.01 / trade-off
   optimum; 8-scenario impact table; re-anchoring question.
2. **Redesign 1 (softmax):** P = Ω̂·softmax₂(Î, ε) with the onset softmin
   CEILING REMOVED (Re_Ω^c = 124.6 + 1.424/P², intrinsically capped by the
   floor); both sign conventions evaluated.
3. **Redesign 2 (curvature-keyed, FINAL):** constant floors destroy the
   rate→0-at-the-wall property (Ω̂ → 1/√2 at the wall) — the floor argument
   must be ε(−Z)/R: **P = Ω̂·√(⟨Î⟩₊² + ⟨ε(−Z)/R⟩₊²)**, wall-boundedness a
   first-class regression gate, ceiling-removal re-evaluated, ε recalibrated.

Kernel definitions verified from repo source (fig04_shapefactor.sphere_rate
= the paper's calibration instrument; solver mirror SAAiTransition.h:108-123):
X = u, Y = y·u′, Z = ½y²·u″, R = √(X²+Y²+Z²), Ω̂ = Y/√(X²+Y²),
Î = (Y−X−Z)/R. −Z/R is dimensionless with Î's own /R normalization.
CLIP CONVENTION adopted: both softmax arguments clipped at zero —
⟨Î⟩₊ (raw-Î REJECTED, below) and ⟨−Z⟩₊ (APG walls have u″(0) > 0 → floor
inert there by construction).

## Target choice (DG-vs-mfoil ambiguity, stated up front)

All deviations are against Drela–Giles 1987 Eq. 29/30 as coded in
`paper/repro/lib/correlations.py` — the correlation the model is calibrated
against and the dashed reference of the canonical Fig. 4. The mfoil
(Fidkowski XFOIL-port) fit agrees within ~15% on the attached branch but is
~60% hotter in slope at stagnation H (1.18e-2 vs 7.44e-3 at H = 2.216, 1041
audit). Sub-20% match claims at H < 2.3 are inside the inter-fit ambiguity.

## Why the 1851.2 ceiling existed, and what protects its cargo now

Eq. (reomc): Re_Ω^c = k·softmin₂(C, A + B/P²), (C,A,B) = (2600,175,2),
k = 0.712 (Blasius-march anchor at Drela's Re_θ = 338). The shape is the LST
neutral-point graze envelope (fig02_onset_graze.py, "never refit"): the
ceiling C is the favorable-side saturation, the floor A the near-separation
immediate-ignition limit (the separated Stewartson profile rides ~4x above
it at Re_θ0 ≈ 26 and must keep igniting at once). The paper itself flags
that β ≥ +0.25 members graze 1.27–1.34 HIGH — the shape is least trusted
exactly where this study operates. Under the floored P the ceiling's job
(bounding A + B/P² as P → 0) is done by the floor itself:
B/P² ≤ B/(Ω̂·floor)², and the free-shear/separated side (large P → threshold
→ A) never sees the ceiling. **Verdict: drop the softmin and C = 1851.2**
(quantified below); **net constant count unchanged: −1851.2, +ε.**

## Sign-convention check (redesign 1, instrument: zero-suite + march)

sm2raw (√(Î²+ε²), even in Î) at ε = 0.03: log-layer parasitic production
P_AI/P_SA = 1.4e-1 RAW (y+ ≈ 200, gate open, the Î ≈ −0.8 stable layer
boosted by the even form) — two-plus orders worse than the clipped form —
and β=1 onset at 0.06x DG (catastrophically early). **REJECTED; clip Î.**

## THE CENTERPIECE — trade-off curves ε → {Fig-4 match, Blasius perturbation}

Marched instrument (canonical grid 1600x1200); dev = worst multiplicative
deviation over favorable H ∈ [2.216, 2.591]; panel (a) on the late N∈[5,9]
secant, panel (b) on the N=1 crossing vs DG's Re_θ0 + 1/slope. Blasius
canon at this grid: late secant 1.00x DG, Rt1 = 349.3 (converged anchor 338).

**Form zc (HEADLINE): P = Ω̂·√(⟨Î⟩₊² + ⟨ε(−Z)/R⟩₊²), gate 124.6+1.424/P², no ceiling**

| ε | (a) worst rate | (b) worst onset | max | Blasius late | Blasius Rt1 | Re_x(N=9) |
|---|---|---|---|---|---|---|
| 0.05 | 5.74x low | 4.11x late | 5.74x | −0.1% | +0.2% | +0.4% |
| 0.08 | 2.87x low | 1.81x late | 2.87x | +1.0% | −1.4% | −2.2% |
| **0.12** | **1.67x low** | **1.90x early** | **1.90x** | **+3.2%** | **−4.5%** | **−7.1%** |
| 0.16 | 1.16x | 3.10x early | 3.10x | +6.2% | −8.5% | −13.2% |
| 0.22 | 1.31x | 5.27x early | 5.27x | +12.2% | −15.5% | −23.6% |
| 0.30 | 1.84x hot | 8.63x early | 8.63x | +22.5% | −25.0% | −37.3% |

Minimax ε* = 0.12; stagnation slope-match (late = 1x at H = 2.216) at
ε ≈ 0.17 where onset runs 3x early. Full ladder at ε = 0.12 (late secant x
DG / N=1 x DG-N=1): β=1: 0.60/0.53; 0.55: 0.63/0.72; 0.35: 0.66/0.90;
0.20: 0.81/1.02; 0.10: 0.98/0.80; 0.05: 1.03/0.84; Blasius: 1.04/0.99.
The residual defect is confined to H ≤ 2.29 (β ≥ 0.55); from H = 2.34 on,
BOTH panels are within ±35% and mostly ±20%. Candidate figure:
`figs_explore/model_calibrate_candidate.png` — panel (b) lies ON the DG
N=1 curve across the whole attached+separated family.

**Form sm2clip (constant floor): P = Ω̂·√(⟨Î⟩₊²+ε²), no ceiling**

| ε | (a) | (b) | max | Blasius late | Rt1 | Re_x(N9) |
|---|---|---|---|---|---|---|
| 0.005 | 31.6x | 24.9x | 31.6x | −0.7% | +1.2% | +1.8% |
| 0.01 | 8.8x | 8.2x | 8.8x | −0.2% | +0.8% | +1.0% |
| 0.02 | 2.63x | 2.48x | 2.63x | +1.7% | −0.5% | −2.5% |
| **0.03** | **1.37x** | **1.58x** | **1.58x** | **+5.1%** | **−2.7%** | **−7.8%** |
| 0.05 | 1.86x | 3.77x | 3.77x | +16.6% | −8.9% | −22.8% |

The user's ε = 0.03 is EXACTLY this form's optimum (ladder: β=1 0.73/0.63,
0.55: 0.85/0.93, 0.35: 0.94/1.23, 0.20: 0.99/1.19, 0.10: 1.04/0.86,
Blasius 1.06/1.01). Slightly better Fig-4 minimax than zc — but it fails
the wall-boundedness gate and books massive spurious nose amplification
(below), which is why zc is the recommendation.

**Form add (original additive): P = Ω̂(Î+ε), softmin gate kept**

| ε | (a) | (b) | max | Blasius late | Rt1 | Re_x(N1) |
|---|---|---|---|---|---|---|
| 0.005 | 4.37x | 2.00x | 4.37x | +5.5% | −4.7% | −9.2% |
| 0.01 | 2.47x | 1.35x | 2.47x | +11.1% | −9.0% | −17.3% |
| 0.02 | 1.48x | 1.59x | 1.59x | +22.8% | −16.6% | −30.5% |
| 0.03 | 1.52x | 2.10x | 2.10x | +41.9% | −26.1% | −45.4% |
| 0.06 | 2.60x | 3.72x | 3.72x | +77% | −37% | −61% |

The additive floor's linear leak into every layer moves Blasius by 3-4x the
flat-plate cross-solver agreement band (±7-10%) at any useful ε — the
coordinator's predicted tension, measured. This is what motivated the
softmax redesigns. Joint re-anchoring (a_max' = 0.140 = 0.19x0.739,
k' = 0.889 = 0.712x1.249 at ε = 0.03) restores Blasius exactly and gives
dev (1.43x, 1.64x) — but the k-rescale is global and pushes the SEPARATION
side off its calibration (β = −0.1988 onset 1.17x → 1.48x DG): re-anchoring
does NOT cleanly resolve the additive form. By contrast sm2clip@0.03 needs
only a_max x0.952, k x0.997 (dev 1.39x/1.57x, Blasius exact), and
zc@0.12 needs a_max x0.969 (0.184), k x1.066 (0.759): Blasius restored
(late 1.00x, Rt1 1.03x), dev (1.74x, 1.79x), separation-limit onset only
1.17x → 1.23x DG — the k-rescale stays benign for the quadratic floors.

## Ceiling-removal verdict (redesign 2, item 5)

zc vs zc_ceil (identical P, gate with/without softmin ceiling), panel-(b)
worst-onset factor: ε = 0.08: 1.81x (no ceiling) vs 3.75x (ceiling);
ε = 0.16: 3.10x vs 5.06x. At ε = 0.08, β=1: N=1 at 1.07x DG (no ceiling)
vs 0.27x (ceiling). The ceiling strictly harms panel (b) under a floored P
— **drop it**. What it protected is preserved: LST graze re-check at
ε = 0.12 (no-ceiling threshold, k=1 shape) gives attached-family graze
ratios 0.98–1.24 (canon softmin: 0.93–1.14), adverse side unchanged
(0.98–1.04), separated lower-branch 0.909 vs canon 0.911 — free-shear/
separated immediate ignition intact (threshold → A = 124.6 at large P,
ceiling-independent).

## Wall-boundedness regression gate (redesign 2, item 2) — zc PASSES, constant floors FAIL

FS β=1 profile, b(y) = a·onset·|u′|, Rt ladder 1e3→1e7 (geometric y-grid to
1e-7):

- **zc ε=0.12**: sup b/u′_max saturates at 1.483e-3 with the peak FIXED at
  y*/θ = 2.005 from Rt = 1e5 through 1e7; b(wall)/sup = 6.9e-5.
  Analytically: floor argument ε(−Z)/R ~ ε·y·|u″(0)|/(2√2·u′(0)) → 0
  linearly in y (numerically verified). BOUNDED.
- **sm2clip ε=0.03**: peak migrates wallward without limit (y*/θ = 1.65 →
  0.446 → 0.142 → 0.046 for Rt = 1e4→1e7), sup still growing (2.0→4.0e-3),
  b(wall)/sup = 3.3e-3 finite — the high-Re eigenvalue is wall-captured,
  exactly the failure mode the user predicted for any constant floor.

## Sign map of −Z/R (redesign 2, item 3; `signmap_0.15` in the zc JSON)

| profile class | −Z/R range (shear region) | floor active | floor > ⟨Î⟩₊ (ε=0.1…0.15) |
|---|---|---|---|
| FS β=+1 (stagnation FPG) | [0, +0.35] | 100% | 100% (Î ≡ ~0) |
| FS β=+0.35 (FPG) | [0, +0.41] | 100% | 100% (g_max 0.003) |
| Blasius | [0, +0.53] | 100% | 48% (near-wall/outer only; g_max 0.14 keeps the driving band) |
| FS β=−0.15 (APG) | [−0.11, +0.66] | 73% (inert at the wall) | 35% |
| Stewartson lower (separated) | [−0.82, +0.84] | 62% | 24% |
| tanh free-shear | [−0.99, +0.99] | 50% | 17% |
| freestream | −Z → 0 | inert | — |

The floor is keyed to the favorable curvature that suppresses Î: fully
active precisely on non-inflectional FPG layers, wall-inert under APG,
subordinate wherever the layer is inflected, zero in the freestream.

## Impact table per campaign scenario (frozen-profile instruments; zc = the operative column)

Sup-bound convention: dN/ds = max_y[a·onset·|u′|/u] integrated on the
marched laminar mean flow (1041 Part-B instrument; diffusion drain ignored
— transported N is lower; deltas vs the ε=0 kernel are the signal).

| scenario | zc ε=0.12 | zc ε=0.08/0.16 | sm2clip ε=0.03 | add ε=0.03 |
|---|---|---|---|---|
| (1) Blasius/ZPG: late-secant / Rt1 / Re_x(N9) | +3.2% / −4.5% / −7.1% | +1.0/−1.4/−2.2% ; +6.2/−8.5/−13.2% | +5.1% / −2.7% / −7.8% | +42% / −26% / −49% |
| (2) mild FPG β=0.05/0.1 late-rate vs canon | +5%/+11% (0.98-1.03x DG; onsets 0.80-0.84x DG) | +2/+5% ; +9/+22% | +7%/+18% | +57%/+78% |
| (3) β=0.2 / 0.55 / 1.0 slope vs DG | 0.81x / 0.63x / 0.60x | 0.16: 0.99/0.89/0.86x | 0.99x / 0.85x / 0.73x | 1.31x / 0.83x / 0.92x |
| (4) inflected side H ≥ 3 + LSB (attached-adverse / lower branch) | ≤2.4%/3.2% ; lower ≤0.2%/0.1% | 0.16: ≤4.5%/6.1% ; ≤0.3%/0.2% | ≤3.7%/1.9% ; ≤0.2%/0.0% | ≤29%/19% ; ≤5.6%/2.9% |
| (5) cylinder nose N_sup(80°), Re_D 2e6 / 2e7 / 1e9 | 0.03 / 0.44 / 69.4 (canon 0.00/0.01/1.85) | 0.08: 0.02/0.08/24.2 ; 0.16: 0.05/2.4/120.9 | 21.9 / 21.9 / 121.0 | 21.7 / 21.1 / 255.7 |
| (6) spheroid re72a0 ΔN_sup at front 0.858 (dN/dx ≈ 114/L → front shift) | +1.29 (~0.011 L upstream) | +0.59 / +2.24 | +6.3 (~0.06 L) | +10.7 (~0.08 L) |
| (7) Hiemenz/Sec VIII N banked at Re_r 4.66-4.74e5 | 0.010 (bracket UNAFFECTED) | 0.007 / 0.014 | 8.9 (bracket needs re-verification) | 6.9 (same) |
| (8) Spalart zero-suite raw / σ_t-blended max P_AI/P_SA | 4.4e-4 / 3.0e-4 (canon 4.0e-4/2.8e-4) | 4.1e-4 / 4.8e-4 raw | 4.9e-4 / 3.4e-4 | 7.6e-4 / 5.2e-4 |

Readings:
- **The zc form is the only one whose off-target footprint is small
  everywhere**: subcritical cylinder noses stay clean (2e6/2e7: ≤0.44
  sup e-folds where DG also books zero — the constant floors book ~22
  spurious e-folds, pure bypass content), the Sec VIII bistability bracket
  moves by nothing (0.01 e-folds), the spheroid front shifts ~1% L, the
  separated/LSB branch is untouched, and the turbulent parasitic
  production is a 10% perturbation on the canon's already-negligible
  4.0e-4 (buffer-layer y+ ≈ 6, gate closed; σ_t gated-max blend in the
  solver suppresses further; instrument note: a hard u_e profile cap fakes
  a u″ delta that flips g positive — the C¹ exponential-tail closure
  removed a spurious 0.12-0.18 reading, and the BL-edge verdict deserves a
  solver-field slice cross-check).
- **Ultra-Re cylinder (1e9)**: the nose at Re_D = 1e9 is DG-supercritical
  (Rt ≈ 9000-13000 > Rt0(H≈2.25) ≈ 4500-6500), so forward creep there is
  qualitatively e^N-consistent; zc books 69 sup e-folds by 80° (canon 1.9)
  → the transcritical front collapse arrives on the ultra arm. Potential
  u_e = 2 sinθ used at all three Re_D (coordinator-approved; measured u_e
  runs 3-5% below potential, so these budgets are upper bounds twice over).
- Mack seed map: unchanged by construction (eq:tumap maps Tu → χ_∞ through
  A_TU, B_TU, c_v1 only; no kernel-coordinate dependence).

## Recommendation

**Form zc, clip both softmax arguments, drop the softmin ceiling,
ε = 0.12** (minimax; ε = 0.16 if panel (a) at stagnation is prioritized
over onset — the whole curve is recorded). Constant bookkeeping: drop
1851.2, add ε = net zero. Blasius drift −4.5% in Rt1 (+3.2% late secant)
can be absorbed by the usual k re-anchor if desired (zc reanchor stage in
the JSON); at ε = 0.08 the drift is within ±1.4% with max-dev 2.87x.

Residual defect, stated honestly: at H ≤ 2.29 (β ≥ 0.55) the single-ε zc
form still under-amplifies (0.60-0.63x DG late) while opening early
(0.53-0.72x DG N=1) — the two errors partially cancel in transition
location but do not vanish. Closing panel (b) at the stagnation point
would need the gate to see a coordinate that keeps rising as Î → 0 (e.g.
un-floored P in the gate with its own bounded threshold, or an
Ω̂-conditioned B) — one more constant; out of scope per the directive.

## Solver-implementation sketch (env-gated, default OFF, canon untouched)

`SAAiTransition.h` builds Ŝ (=Ω̂) and g (=Î) at lines ~122-123
(`g = (Y0−X0−Z0)/R; P = Shat*g;`) and a second SIMD/Jacobian variant at
~line 218 — both sites take the same two-line change:

    const ftype gp  = fmax(g, ftype(0));
    const ftype flo = fmax(-aiZFloorEps * Z0 / R, ftype(0));   // eps*(-Z)/R
    const ftype P   = Shat * sqrt(gp*gp + flo*flo);

with `double ai_zFloorEps = 0.0;` in ModelConstants.h (env `AI_ZFLOOR_EPS`,
default 0.0 = bit-identical canon) and the gate switched to the no-ceiling
branch only when the floor is active (`AI_REOMC_CEIL=1e30` handles it with
zero new code — the softmin degenerates to A + B/P²). Campaign canon JSONs
are untouched; the sweep runs export `AI_ZFLOOR_EPS=0.12 AI_REOMC_CEIL=1e30`.

## In-solver validations that would follow (the two live testbeds)

1. **Ultra-Re cylinder sweep** (dragcrisis matrix, Re_D 1e6→2e7→1e9-class):
   expect no front motion at Re_D ≤ 2e7 (zc's clean subcritical nose —
   this is the discriminator vs the constant floors), front collapse toward
   the gate-opening angle on the ultra arm; watch Cd against the 0412
   turbulent-separation ledger.
2. **Spheroid re72a0 front**: expect ≤ 0.01-0.02 L upstream front motion at
   ε = 0.12 (sup bound; transported less) — a regression-friendly case.
3. Flat-plate ladder + NLF/E387 spot-checks for the Blasius −4.5%/−7% Re_x
   drift (or re-anchor k first and expect bit-class agreement).

## Honest ledger

1. DG-vs-mfoil target ambiguity at low H (~60% in slope at H = 2.216):
   sub-20% claims there are inside the inter-fit noise.
2. Sup-bound instrument for rows 5-7 ignores the diffusion drain
   (transported N lower); all values are deltas against the same
   instrument at ε = 0.
3. Cylinder rows use potential-flow u_e (audit Part B used measured Cp;
   its ε=0 numbers differ accordingly: e.g. 2e7 N(90°) sup 8.3 measured-Cp
   vs 1.9-3.0 potential here).
4. Blasius comparisons at the 1600x1200 grid anchor 349.3 (converged 338);
   all shifts are same-instrument ratios.
5. Sec VIII bracket (4.66-4.74e5) taken from the coordinator mid-rebisection;
   the zc conclusion (0.01 e-folds) is insensitive to the exact bracket.
6. The additive-form impact numbers are superseded (kept for the record).
7. sm2clip's Fig-4 minimax (1.58x) is nominally better than zc's (1.90x);
   the recommendation rests on the boundedness gate + off-target footprint,
   not on Fig-4 alone. If a constant floor were ever preferred, ε = 0.03
   is its calibrated value.
8. fig04 RuntimeWarnings (invalid multiply) during extreme-domain searches
   are the known NaN-guarded domain-sizing path; rows are NaN-checked.
9. The paper's ±12% panel-(b) tracking claim uses the converged instrument;
   this study's same-grid canon baseline shows 1.10-1.17x on the adverse
   branch — deltas here are all same-instrument, so the claim is not
   contradicted.

## Artifacts (full paths)

- Script: `/home/qiqi/flexcompute/sa-ai/paper/repro/analytic/fpg_recalibration_study.py`
  (stages: --smoke --baseline --sweep --final --family --impact --reanchor
  --tradeoff --signcheck --boundedness --signmap; `--form {add,sm2raw,sm2clip,zc,zc_ceil}`)
- Numbers: `/home/qiqi/flexcompute/sa-ai/paper/repro/analytic/figs_explore/`
  `fpg_recalibration_study.json` (additive), `fpg_recalibration_sm2clip.json`,
  `fpg_recalibration_sm2raw.json`, `fpg_recalibration_zc.json` (HEADLINE),
  `fpg_recalibration_zc_ceil.json`
- Figures (same dir): `model_calibrate_candidate.png` (zc ε=0.12, HEADLINE),
  `model_calibrate_candidate_sm2clip.png` (ε=0.03),
  `model_calibrate_candidate_additive.png` (ε=0.03),
  `fpg_recal_tradeoff.png` (zc), `fpg_recal_tradeoff_sm2clip.png`,
  `fpg_recal_tradeoff_add.png`
- Run logs (not committed): same dir, `fpg_recal_run.log`, `fpg_recal_sm2.log`,
  `fpg_recal_zc.log`
- Cross-referenced: 2026-07-28-1041 (FPG rate audit), 1033 (field-side FPG),
  0412 (drag-crisis Cd ledger), 0105/1030 (spheroid a0 meanflow marcher +
  u_e; the re72a0 budget reuses the committed a0phys cache chain)

---

# PART II (appended same day) — two-epsilon variant A, rate-only variant B, and the two-branch naming refactor

*USER FOLLOW-UP: the single ε cannot serve both panels (growth wants
~0.16-0.17, onset then runs early). Two variants evaluated with the same
instruments and regression suite; forms `zc2` (A) and `zb` (B) in the study
script; JSONs `fpg_recalibration_zc2.json`, `fpg_recalibration_zb.json`;
logs `fpg_recal_variantA.log`, `fpg_recal_variants.log` (B).*

## Variant A (two epsilons; rate P_r with ε_r, no-ceiling gate on its own P_o with ε_o)

**The as-directed tuning is structurally infeasible at stagnation** — the
central Part-II finding. Tuned gate-first (rate-first diverges: with a weak
gate the late secant is onset-limited and no ε_r reaches Drela's slope —
the tuner ran to ε_r = 1 at late = 0.56x, found the hard way): ε_o = 0.0639
puts the β=1 N=1 crossing ON the DG station (0.995x). But with the gate
there, the threshold (~1.1e4) is the same order as the available
Re_Ω = y²u′/ν over the ENTIRE march — the tanh ramp never saturates — and
the late secant asymptotes far below Drela no matter the rate constant:
late = 0.42x / 0.46x / 0.53x at ε_r = 0.15 / 0.19 / 0.35 (ε_o ≈ 0.055).
The separability expectation ("ε_r for panel a, ε_o for panel b") holds
only where the gate saturates early, i.e. mild β — exactly where the
single-ε form already works.

At the bounded compromise (ε_r = 0.35, ε_o = 0.0639; onset pinned):

- Ladder: β=1: 0.43x/0.99x; 0.55: 0.61x/1.40x; 0.35: 0.80x/1.76x;
  0.20: 1.48x/1.36x; 0.10: 1.60x/0.76x; Blasius: 1.30x/0.88x.
  Family devs (2.35x, 1.76x) — worse than single-ε zc@0.12 on panel (a)
  and barely better on (b); the stagnation-sized ε_r overheats the whole
  mild band.
- Regression: Blasius late +29.8%, Rt1 −14.5%, Re_x(N9) −37.2% (anchor
  destroyed); attached-adverse family 21.9%/12.6%; lower branch 1.4%/1.1%.
- Off-target: cylinder nose clean (0.26 sup e-folds by 80° at 2e7 — the
  small-ε_o no-ceiling gate protects it), Hiemenz 0.030, spheroid +6.6,
  zero-suite raw 7.8e-4; boundedness PASSES (sup saturates 4.3e-3,
  y*/θ = 2.005, wall 2.4e-5).

## Variant B (rate-only floor ε_r; CANON gate untouched — softmin(1851.2, ...) on un-floored P)

Tuned: **ε_r = 0.1455** (β=1 late = 0.994x). Panel (a) is essentially
perfect FAMILY-WIDE: late secants 0.99–1.09x over H ∈ [2.216, 2.591]
(dev 1.09x — the best panel (a) of the whole study). Panel (b), honestly:
exactly the predicted ceiling failure — the un-floored canon P → 0 in
strong FPG saturates the softmin at 1851.2, the gate opens at Rt ≈ 1150–1280
and the floored rate amplifies immediately: N=1 at Rt = 1540/1480/1422 =
**0.23x/0.35x/0.48x DG** at β = 1/0.55/0.35 (dev 4.40x); mild band
0.71–0.86x; Blasius 0.99x (late +5.9%, Rt1 −3.8%, Re_x(N9) −10.0%).

- Regression: attached-adverse 4.1%/2.9%, lower 0.2%/0.2% (isolated);
  boundedness PASSES (rate floor is the zc form; sup saturates 1.80e-3 at
  y*/θ = 2.005 from Rt = 1e4, wall 5.7e-5 — the canon gate is a ≤1 factor
  and cannot unbound it).
- Off-target — two NEW strikes from the kept ceiling: (i) cylinder nose at
  2e7 books 4.0 spurious sup e-folds by 80° (2e6: 0.07; 1e9: 140 — ultra
  arm hottest of the recommended-class forms); (ii) in high-Re_τ log
  layers the canon gate OPENS at y+ ≈ κ·1851 ≈ 760 (Re_Ω = y+/κ crosses
  the saturated ceiling) and the floored rate leaks: zero-suite raw max
  P_AI/P_SA = 3.9e-3 at Re_τ ≥ 1000 — 10x the zc/canon level — though
  σ_t-blending still crushes it to 3.2e-4. Hiemenz 0.013, spheroid +1.28
  (both fine — same rate floor as zc).

## A vs B vs single-ε zc@0.12 — the decision table

| criterion | zc@0.12 (Part I) | A (0.35, 0.0639) | B (0.1455) |
|---|---|---|---|
| panel (a) worst, H ≤ 2.6 | 1.67x | 2.35x | **1.09x** |
| panel (b) worst | **1.90x** | 1.76x (β=0.35; β=1 pinned 0.99x) | 4.40x |
| Blasius (late / Re_x(N9)) | +3.2% / −7.1% | +30% / −37% | +5.9% / −10.0% |
| adverse+lower family | **≤2.4%/3.2% ; 0.2%** | 22%/13% ; 1.4% | 4.1%/2.9% ; 0.2% |
| constants bookkeeping | **net 0** (−1851.2, +ε) | net +1 (−1851.2, +ε_r, +ε_o) | net +1 (+ε_r, ceiling kept) |
| wall-boundedness | PASS | PASS | PASS |
| cylinder nose 2e7 (sup N by 80°) | **0.44** | 0.26 | 3.99 |
| Sec VIII Hiemenz | 0.010 | 0.030 | 0.013 |
| spheroid re72a0 ΔN | **+1.29** | +6.6 | +1.45 |
| Spalart zero-suite raw | **4.4e-4** | 7.8e-4 | 3.9e-3 (log-layer leak) |
| ultra arm 1e9 (80°) | 69 | 53 | 140 |

**Recommendation: keep the single-ε zc@0.12.** A is dominated (its one
win, the pinned β=1 onset, costs the Blasius anchor, the mild band, and a
+1 constant). B is the choice ONLY if panel-(a) exactness family-wide is
the overriding goal — its price is 4.4x-early strong-FPG onset, spurious
nose amplification at practical Re (the bypass content zc avoids), the
log-layer gate leak, and a +1 constant. The single-ε form remains the
only net-zero-constants, both-panels-balanced, clean-off-target option;
its known residual (H ≤ 2.29) is smaller than either variant's worst
defect.

## The two-branch refactor (RECOMMENDED FINAL FORM) and naming

Per the user's naming preference the ε-form should be presented with the
prefactor inside the softmax, so each classical instability branch carries
its own named rate and ε disappears as a symbol:

    Q  = Ω̂ · softmax₂( a_R·⟨Î⟩₊ ,  a_TS·⟨−Z⟩₊/R )
    a  = min( a_R , ⟨Q⟩₊ )                       [rate; gate unchanged on P = Q/a_R]

- **a_R = 0.19** — the Rayleigh (inviscid-inflectional) branch rate: the
  renamed a_max, its Michalke free-shear eigenvalue determination
  untouched. (Alternative if "Rayleigh" overclaims: a_I, inflectional.
  a_invisc rejected as too long — user's own note.)
- **a_TS = a_max·ε = 0.0228** at ε = 0.12 (0.0304 at the slope-priority
  0.16) — the Tollmien–Schlichting (viscous) branch rate, with its own
  determination: the Drela-stagnation trade-off of Part I. The name states
  the mechanism, matching the paper's convention (a_max named for the
  free-shear eigenvalue it caps; c_ν,ai the retained laminar diffusion):
  the floor holds exactly the viscous TS branch that survives where the
  inviscid inflectional coordinate dies, keyed to the favorable curvature
  (−Z) that kills it. c_TS/c_curv/c_FPG were considered and rejected: the
  constant IS a rate (same units and role as a_max), so the a_ prefix is
  the honest one; c_curv names the lever, c_FPG a regime.
- **Identity VERIFIED, composition matters**: by degree-1 homogeneity of
  softmax₂, Q = a_max·P exactly, and a = min(a_R, ⟨Q⟩₊) reproduces
  a_max·min(1, ⟨P⟩₊) to ≤ 5e-18 relative on the full FS family (numerical
  check in the study log). The clip CEILING must be a_R, NOT 1 — writing
  clip(Q, 0, 1) would raise the strongly-inflected ceiling 5x. The onset
  gate keeps the dimensionless coordinate (P = Q/a_R, threshold constants
  unchanged); under variant A the onset branch would carry its own
  a_TS,o = a_max·ε_o = 0.0121, but A is not recommended.
- The rename a_max → a_R (paper-wide + ModelConstants.h + AI_RATESCALE
  env naming) is a single-commit mechanical change at adoption time, not
  performed now.

## Part II erratum (found in Part III)

The decision table's B-spheroid entry should read **+1.28** sup e-folds
(24.43 − 23.15), not +1.45.

## Part II honest ledger additions

1. A's tuning bound: ε_r capped at 0.35 in the secant; the late-ratio
   trend (0.42→0.46→0.53 per 0.15→0.19→0.35) is saturating, so the
   infeasibility is structural, not a bound artifact.
2. B's zero-suite log-layer leak (3.9e-3) is a RAW-kernel number at the
   solver's σ_t-suppressed fringe; the blended value (3.2e-4) is
   comparable to the other forms. It is still a 10x raw regression and
   lives exactly where relaminarizing/transitional fringes have σ_t < 1.
3. A/B impact rows use the same sup-bound instruments as Part I (same
   caveats); B's cylinder ε=0 baseline rows differ slightly from Part I's
   zc rows because the gate form differs at ε=0 (softmin vs no-ceiling).
4. Naming judgment (a_TS vs a_visc) is taste on top of convention; both
   are defensible — a_TS is more specific about WHICH viscous instability
   the floor represents, and pairs with a_R without inventing a_invisc.

---

# PART III (appended same day) — variant B first-class, the joint (a_visc, C) retune, and the adopted notation

*USER CLARIFICATION: variant B is the INTENDED design — rate =
softmax₂(a_inviscid·Ω̂⟨Î⟩₊, a_visc·Ω̂⟨−Z⟩₊/R) · OnsetGate(Re_Ω/Re_Ω^c(Ω̂Î));
the gate never sees the viscous term. The Part-II `zb` form is exactly this
by degree-1 homogeneity (a_visc = a_max·ε_r). USER EXTENSION: within B the
1851.2 ceiling is NOT sacred — joint (a_visc, C) optimization. Both are
delivered here. softmax₂ CONFIRMED as the two-argument 2-norm,
softmax₂(x,y) = √(x²+y²) — the user's "softmax(x+y)" read as shorthand.
Figure: `figs_explore/model_calibrate_candidate_zb.png` (canon-C at
a_visc = 0.0276 and 0.0230, plus the joint optimum). Data keys with a
retuned ceiling carry the suffix `_C<value>` in `fpg_recalibration_zb.json`.*

## B at the canon gate (C = 1851.2) — first-class summary

a_visc = 0.0276 (ε_r = 0.1455, β=1 late secant = 0.994x Drela; lean value
0.0230 also plotted): **panel (a) is the best of the entire study —
late secants 0.99–1.09x family-wide** (dev 1.09x). Panel (b): 0.23x/0.35x/
0.48x DG at β = 1/0.55/0.35 (dev 4.40x), mild band 0.71–0.86x, Blasius
0.99x (late +5.9%, Re_x(N9) −10.0%). Families: attached 4.1%/2.9%, lower
0.2%/0.2%. Boundedness PASSES (sup saturates 1.80e-3, y*/θ = 2.005,
wall fraction 5.7e-5 — the gate is a ≤1 factor and cannot unbound the rate).

**Wart (i), front-shift translation (sup-bound instrument, potential u_e,
Tu 0.2% thresholds N = 4.525 chi=1 / 6.485 handover; baselines: solver
front 82–83° at 2e7, 90° at 7e6; DG chi=1 crossing 91.8° at 2e7, none at
7e6):** at 2e7 canon-C B books N = 4.53 by **82.5°** (handover 88.6°) —
i.e. it can spuriously advance the 2e7 front up to ~9° ahead of the DG
station (coincidentally ON the current solver front, for the wrong reason;
the 4 e-folds booked forward of 80° are where DG books ~0). At **7e6: NO
crossing** (N_sup(90°) = 2.0 < 4.53) — the front stays
separation-controlled, matching the solver's 90°; 2e6 clean. Ultra arm
1e9: crossing at 12.0° — gate opens essentially at the leading edge
(subcritical per DG even at 1e9; zc gives 25.6°, the canon kernel's own
crossing is 77–79°).

**Wart (ii), σ_t-blended leak:** raw 3.9e-3 at Re_τ ≥ 1000 lives at
y+ ≈ Re_τ (gate opens at y+ ≈ κ·1851 ≈ 760, ratio keeps growing outward);
the BLENDED max stays at the buffer-layer value **3.2e-4 at y+ ≈ 6**
(canon 2.8e-4, +14%) because χ_eq(760) = min(0.41·760, 0.08·Re_τ) ≥ 61
gives (1−σ_t) ≤ e^{−15} ≈ 2.6e-9 → blended leak ~1e-11, seven orders below
visibility. Flows exercising y+~760 gate opening: any turbulent layer with
Re_τ ≳ 900 (the suite's Re_τ = 1000 and 5200 rows). Exposure is confined
to σ_t < 1 fringes that simultaneously reach Re_Ω ≈ 1851 — very-high-Re
transitional shoulders, where amplification is the intended behavior.

## The joint (a_visc, C) retune — extension results

C-scan at fixed a_visc = 0.0276 (favorable ladder; onsets as x DG N=1
station, order β = 1/0.55/0.35/0.2):

| C | late secants (β=1…0) | onsets | worst onset | worst late |
|---|---|---|---|---|
| 1851.2 (canon) | 0.99…1.06 | 0.23/0.35/0.48/0.71 | 4.40x | 1.09x |
| 4000 | 0.78…1.06 | 0.41/0.62/0.85/1.20 | 2.46x | 1.28x |
| **8000** | **0.56…1.06** | **0.69/1.05/1.45/1.73** | **1.73x** | **1.79x** |
| 20000 | 0.31…1.06 | 1.37/2.10/2.90/2.11 | 2.90x | 3.2x |

Expectation (1) verified with the predicted residual: raising C delays the
strong-FPG opening toward Drela's Rt0, but a fixed C cannot track the
rising Rt0(H) — at C = 8000 the family straddles it (β=1 0.69x early,
β=0.2 1.73x late; β=0.2's threshold saturates on its own 1/P² branch
(≈6450) above C ≈ 6450, capping what larger C can do there). RATE-GATE
COUPLING (same physics as variant A, milder): the later gate narrows the
open band and the late secant sags (0.99 → 0.56x at β=1); re-tuning ε_r at
C = 8000 SATURATES — even a_visc = a_R = 0.19 reaches only late 0.93x while
dragging the onset back to 0.40x — so "late = 1 at β=1" is not attainable
and the joint optimum is the balanced compromise. Interior probe
(a_visc = 0.0418, C = 6000): devs (1.26x, 1.99x) — the surface is flat;
**joint optimum adopted: (a_visc, C) = (0.0276, 8000)**, devs
**(1.77x, 1.73x)**, max **1.77x**.

At the optimum, full package:
- Blasius intact: late +5.1%, Rt1 −2.6% (Rt1 = 340), Re_x(N9) −8.2%.
- Families: attached-adverse 3.8%/2.3%, lower 0.2%/0.1%.
- **Wart (i) CURED at practical Re**: 2e7 N_sup(80°) = 0.15, N(90°) = 2.42
  < 4.53 — NO spurious crossing at 2e7/7e6/2e6 (the canon-C 82.5° front is
  gone). Ultra arm: crossing at **35.4°** (handover 38.6°) — the
  transcritical forward collapse retained, now landing in the experimental
  25–35° class [mem: Achenbach digitization still gated] rather than at
  the leading edge.
- **Wart (ii) essentially cured**: gate needs y+ ≈ 0.41·C ≈ 3300 — leak
  GONE at Re_τ ≤ ~2700 (Re_τ = 1000 raw back to 4.5e-4, canon level);
  residual edge value 3.2e-3 raw at Re_τ = 5200 (blended 2.9e-4 ≈ canon).
- Graze re-check at C = 8000 (k=1 shape 11236): mild favorable members
  move TOWARD their LST anchors (β=0.10: 1.085→0.923; β=0.05: 1.137→1.082;
  Blasius 1.035→1.017; and the paper's flagged β ≥ 0.25 members, grazing
  1.27–1.34 high at canon C, move toward 1 by construction); β=0.15
  overshoots downward (1.004→0.607 — its closest approach saturates on the
  raised ceiling). Adverse/separated side pinned EXACTLY (0.951/0.991/
  1.034/0.982/0.930; lower branch 0.909) — threshold → 124.6 at large P,
  ceiling-independent, confirmed.
- Boundedness PASSES unchanged (1.80e-3 saturated, y*/θ = 2.005,
  wall 5.7e-5).
- Spheroid ΔN = +1.27 (front shift ~0.011 L); Hiemenz 0.013 e-folds
  (Sec VIII untouched); zero-suite blended 3.1e-4.

## Final three-way table (symmetric rows)

| criterion | zc@0.12 (net 0) | B canon-C (0.0276) | **B joint (0.0276, C=8000)** |
|---|---|---|---|
| panel (a) worst | 1.67x | **1.09x** | 1.79x |
| panel (b) worst | 1.90x | 4.40x | **1.73x** |
| max of both | 1.90x | 4.40x | **1.77x** |
| Blasius late / Re_x(N9) | +3.2% / −7.1% | +5.9% / −10.0% | +5.1% / −8.2% |
| families (att; lower) | 2.4%/3.2%; 0.2% | 4.1%/2.9%; 0.2% | 3.8%/2.3%; 0.2% |
| nose 2e7 (sup N80; front) | 0.44; none | 3.99; 82.5° spurious | **0.15; none** |
| nose 7e6 / 2e6 | none / none | none / none | none / none |
| ultra 1e9 front (sup) | 25.6° | 12.0° | 35.4° |
| Sec VIII / spheroid ΔN | 0.010 / +1.29 | 0.013 / +1.28 | 0.013 / +1.27 |
| zero-suite raw / blended | 4.4e-4 / 3.0e-4 | 3.9e-3 / 3.2e-4 | 3.2e-3 (Re_τ=5200 only) / 3.1e-4 |
| boundedness | PASS | PASS | PASS |
| constants | net 0 (−C, +ε) | net +1 (+a_visc) | net +1 (+a_visc; C retuned) |

**Recommendation: adopt the joint-retuned variant B, (a_visc, C) =
(0.0276, 8000)** — it is the user's intended structure, it now beats the
single-ε zc on the combined Fig-4 metric (1.77x vs 1.90x), both warts are
cured or invisible at the retuned C, and it keeps the onset gate a purely
inviscid-LST object (the gate never sees the viscous branch — the graze
construction survives as a concept, with C's determination changed from
"favorable-side graze saturation" to "centers the Drela–Giles critical
stations of the favorable family"). Cost vs zc: one net extra constant
and the β=0.15 graze undershoot. zc@0.12 remains the net-zero-constants
fallback with nearly the same quality.

## Adopted notation (one block)

    Q      = softmax₂( a_inv·Ω̂·⟨Î⟩₊ ,  a_visc·Ω̂·⟨−Z⟩₊/R ),   softmax₂(x,y) = √(x²+y²)
    rate a = min( a_inv , Q )                                  [ceiling clip RETAINED from canon;
                                                                the user's loose formula omitted it —
                                                                without it free-shear rates overshoot
                                                                a_inv by ~30%]
    gate   = ½[1 + tanh((Re_Ω/Re_Ω^c − 1)/0.35)],
    Re_Ω^c = softmin₂( C , 124.6 + 1.424/⟨Ω̂Î⟩₊² )              [gate BLIND to the viscous term]

- **a_inv = 0.19** — the inviscid (inflectional) branch rate; rename of
  a_max, its Michalke free-shear eigenvalue determination untouched.
  Short forms considered: a_R (Rayleigh; most mechanistic, risks
  overclaiming), a_I (collides visually with Î and reads as "index"),
  **a_inv (RECOMMENDED)** — self-explanatory, three characters, and pairs
  with a_visc as the classical inviscid/viscous instability dichotomy.
- **a_visc = 0.0276** — the viscous-branch rate (a_max·ε_r, promoted to a
  first-class constant). Determination: β=1 (stagnation) late secant =
  Drela–Giles at the canon gate. More specific alternative name: a_TS
  (Tollmien–Schlichting); a_visc adopted per the user's formula.
- **C = 8000** (k-carrying units; 11236 at the k=1 graze scale) —
  determination CHANGED: retuned to center the Drela–Giles critical
  stations of the favorable family (β=1 opens 0.69x early ↔ β=0.2 1.73x
  late), superseding the LST-graze favorable-side saturation (which the
  paper already flagged as 1.27–1.34 high for β ≥ 0.25).
- Constant bookkeeping: C retuned (existing constant), a_visc added —
  net +1.
- Two-epsilon naming (variant A, NOT adopted): the onset branch would
  carry its own a_visc,o; recorded for completeness only.

## Part III honest ledger

1. Front estimates are sup-bounds on potential-flow u_e (no diffusion
   drain): transported fronts sit later; "cured" rows (no crossing) are
   robust to this (the bound overbooks), the 1e9 angles are not.
2. The joint optimum is a flat-surface pick from a coarse (ε_r, C) grid
   plus one interior probe and one failed exact-target retune; ±20% moves
   in either constant change the max dev by < 0.1 in log units.
3. β=0.2's onset lateness (1.73x) at the optimum is branch-limited
   (its own 1/P² threshold ≈ 6450), not C-limited — a C retune cannot
   remove it; it is the fixed-shape residual the extension predicted.
4. The Re_τ = 5200 residual raw leak sits in the outer 20% of the layer
   where the C¹-tail profile closure is least trustworthy (Part I
   instrument note); the blended value is the operative one.
5. The zb ypos diagnostic ("floored P>0 for y+ <= 0.0") is inert for this
   form (its threshold convention doesn't apply); rate/gate values are
   unaffected.

---

# PART IV (appended same day) — enriched onset-graze figure and the low-H flatness diagnosis

*USER DIRECTIVE: panel (b) of `model_calibrate_candidate_zb.png` goes too
flat vs H at low H. First step: enrich the whitepaper's Figure 2 (the
paper's fig:onsetgraze, regen `repro/analytic/fig02_onset_graze.py`) with
FS beta = {0.35, 0.5, 0.7, 1.0}, same method, same styling; deliver a
candidate PDF + a diagnostic companion; quantify. Canon `paper/figs/
onset_graze.pdf` and tex UNTOUCHED. Script:
`repro/analytic/fig02_onset_graze_enriched.py`; artifacts in
`repro/analytic/figs_explore/`: `onset_graze_enriched.pdf` (adoption
candidate), `.png` (preview), `onset_graze_flatness.png` (diagnostic),
`onset_graze_enriched.json` (tables).*

## Method for the new neutral points (requirement 1)

The canonical figure's "LST neutral point" is each profile evaluated at
its **Drela–Giles Eq. 30 critical Re_theta0(H)** (repo fit,
`lib/correlations.py`) — a correlation distilled from Drela's
Orr–Sommerfeld database, not a per-profile OS solve. The new members use
the SAME method: Eq. 30 is smooth down to H = 2.216 (Re_theta0 = 6640 on
the rig's H). Cross-check at the stagnation profile: the published OS
critical for Hiemenz flow (Wazzan–Okamura–Smith 1968, Re_delta*_crit ≈
12490 → Re_theta0 ≈ 5640 at H = 2.216) sits ~15% below the correlation —
same decade, extension defensible [literature memory; citation to be
verified at integration]. NUMERICAL FIX required and applied: the strong-
FPG members' P ~ 1e-4-class is delicate, so the curvature indicator Z is
computed from the repo's FS ODE exactly
(f''' = −[f·f'' + β(1−f'²)]/(2−β), Blasius-consistent eta — NOTE the
repo's normalization; the Hartree form corrupts the Y−X−Z cancellation,
found the hard way). Legacy members verified to reproduce the canon graze
ratios to <2% (printed digits identical).

## The structural finding that reframes the request

With exact curvature, **max_y(Ω̂Î) = 1.7e-3 at β = 0.35 and P ≤ 0 over the
ENTIRE profile for β ≥ 0.5** — the 1041 audit's ~5e-4 readings at β ≥ 0.5
were RANS-field estimator noise; on the clean rig profile the amplifying
coordinate simply does not exist beyond β ≈ 0.4. So β = 0.35 joins the
canon (P, Re_Ω) plane normally, while β = 0.5/0.7/1.0 have NO locus in
the plane: they are drawn as left-edge arrows at their profile-max Re_Ω
and quantified against the P→0 threshold limit (the ceiling).

## Graze-ratio table, old + new (canon shape at k = 1; C8000 = Part-III retuned ceiling, k=1 scale 11236)

| β | H | Re_θ0 (DG) | Re_Ω* | P* | graze vs canon | vs C=8000 |
|---|---|---|---|---|---|---|
| **+1.00** | 2.216 | 6640 | 9624 | P ≤ 0 interior | **3.70** | 0.86 |
| **+0.70** | 2.256 | 4991 | 7630 | P ≤ 0 interior | **2.94** | 0.68 |
| **+0.50** | 2.297 | 3756 | 6053 | P ≤ 0 interior | **2.33** | 0.54 |
| **+0.35** | 2.342 | 2752 | 3312 | 1.5e-4 | **1.27** | 0.30 |
| +0.15 | 2.442 | 1235 | 2177 | 2.3e-2 | 1.004 | 0.59 |
| +0.10 | 2.481 | 827 | 1531 | 3.7e-2 | 1.085 | 0.92 |
| +0.05 | 2.529 | 471 | 933 | 5.4e-2 | 1.137 | 1.08 |
| 0 (Blasius) | 2.591 | 242 | 515 | 7.8e-2 | 1.035 | 1.02 |
| −0.05 | 2.676 | 138 | 320 | 0.111 | 0.959 | 0.95 |
| −0.10 | 2.801 | 97 | 250 | 0.161 | 0.996 | 0.99 |
| −0.15 | 3.021 | 72 | 216 | 0.244 | 1.037 | 1.03 |
| −0.19 | 3.481 | 49 | 185 | 0.390 | 0.984 | 0.98 |
| −0.1988 | 3.982 | 36 | 170 | 0.509 | 0.932 | 0.93 |
| −0.19 lower | 4.922 | 26 | 163 | 0.656 | 0.911 | 0.91 |

(β ≥ 0.5 "graze" convention: Re_Ω*_max / ceiling — the value every
P-threshold presents as P → 0. β = 0.35's 1.27 lands exactly on the
paper's flagged "β ≥ +0.25 graze 1.27–1.34 high", and the miss grows
2.33 → 2.94 → 3.70 toward stagnation. Against the retuned C = 8000 the
strong-FPG members sit at 0.54–0.86 — why the Part-III joint retune
centers their marched onsets — while β = 0.15/0.35 drop to 0.59/0.30,
the mild-favorable price Part III already measured.)

## Flatness diagnosis

Toward stagnation the members' neutral points keep rising —
**Re_Ω* ≈ 1.2–1.6 × Re_θ0(H)**, up to 9624 at H = 2.216 (diagnostic left
panel: the neutral locus vs any flat ceiling) — while beyond β ≈ 0.4 the
gate coordinate Ω̂Î is ≤ 0 across the whole profile: every threshold
Re_Ω^c(P), capped or uncapped, presents its constant P→0 limit to exactly
the layers whose critical Reynolds number grows fastest. The flatness is
therefore NOT a mis-set ceiling value but the death of the gate
coordinate at low H; a constant C (any value, incl. Part III's 8000) can
only split the difference across the family.

## Shape candidates (diagnosis only — nothing adopted, canon constants untouched)

1. **Un-capped 175 + 2/P² branch: NOT viable** — the coordinate is dead
   (P ≤ 0) at β ≥ 0.5; where it exists (β = 0.35, P* = 1.5e-4) the branch
   overshoots the neutral point by ~e4.
2. **Different softmin exponent: NOT viable** — any softmin_n limit as
   P → 0 is still the constant C.
3. **Added rising branch in the Part-III viscous coordinate
   P_o = Ω̂⟨−Z⟩₊/R**: a locus EXISTS there for all members (diagnostic
   right panel of the earlier draft; JSON `viscous_pts`), and a power law
   through the four new neutral points gives Re_Ω* ~ 12·P_o^−3.2 — but
   the fit is ILL-CONDITIONED (P_o spans only 0.129–0.160 across
   β = 0.35→1.0): the viscous coordinate barely varies while Re_Ω* rises
   2.9x, so a P_o-branch alone is a fragile lever.
4. **The observed law is Re_Ω* ≈ 1.45·Re_θ0** (diagnostic right panel):
   a threshold carrying a Re_θ-like integral scale would track trivially,
   but leaves the pointwise-local model class — recorded as the structural
   tension any Part-V gate redesign must resolve.

## Part IV honest ledger

1. Eq. 30's low-H validity is inherited, not established: the one
   available OS cross-check (Hiemenz, Wazzan et al. 1968 ≈ 12490 in
   Re_δ*) is literature memory pending citation; it brackets the DG value
   within ~15%.
2. The audit-vs-rig maxP discrepancy at β ≥ 0.5 (5e-4 vs ≤ 0) is an
   ERRATUM against the 1041 audit's frozen-profile column (its marched
   rates and verdicts are unaffected — they used the transported
   instrument); the "noise floor" footnote there was the right instinct.
3. β = 0.35's neutral point sits at the extreme low-P end of its locus
   (P* = 1.5e-4), i.e. its graze is already ceiling-dominated; its
   in-plane placement is a boundary case.
4. Enriched-figure styling: identical construction/colors/masks; the
   x-range is extended (1e-4 vs 3e-3) and y-range raised (1e5) to admit
   the new members, and the colormap indices shift because the attached
   count grew 9 → 13 — flag for the caption at integration.
5. The exact-curvature refactor changes legacy members' curves by less
   than the line width (graze ratios reproduce to printed digits).

---

# PART V (appended same day) — the blended gate argument: coordinate death cured, ceiling retired

*USER DESIGN, responding to the Part-IV diagnosis: exchange the softmin
ceiling for a BLENDED gate argument
P_gate = ⟨Ω̂Î⟩₊ + c_o·Ω̂⟨−Z⟩₊/R (linear headline; softmax₂ alternative
evaluated), Re_Ω^c = 124.6 + 1.424/P_gate², NO ceiling — net constant
count zero vs canon (C exchanged for c_o). SIGN CONVENTION verified: the
viscous coordinate is the SAME ⟨−Z⟩₊ clip the Part-III rate kernel uses
(−Z > 0 in favorable-curvature layers, u″ < 0); the user's "Ω̂Z/R" read
as loose shorthand. Rate = Part-III two-branch, unchanged
(a_inv = 0.19, a_visc = 0.0276). Forms `vb` (linear) / `vbs` (softmax₂)
in the study script; stage `--partv` = the c_o graze calibration; JSONs
`fpg_recalibration_vb.json`, `fpg_recalibration_vbs.json`; log
`fpg_recal_partv.log`.*

## The user's quantitative check — VERIFIED to ±10%

Tracking Drela's rising critical curve requires the gate argument at the
neutral points to be √(B/(Re_Ω*−A)) ≈ √(1.424/Re_Ω*) (k-carrying;
2/Re_Ω* at the k=1 graze scale). Against the Part-IV members:

| β | Re_Ω* | required P_gate* | measured max P_o | required/measured |
|---|---|---|---|---|
| +0.35 | 4685 | 0.0211 | 0.1648 | 0.128 |
| +0.50 | 6053 | 0.0184 | 0.1556 | 0.118 |
| +0.70 | 7630 | 0.0164 | 0.1488 | 0.110 |
| +1.00 | 9624 | 0.0145 | 0.1435 | 0.101 |

The ratio is constant to ±12% across the strong-FPG family — the viscous
coordinate's narrow variation has almost exactly the right slope, as the
user estimated: the blend buys genuine H-tracking in strong FPG, not just
boundedness. **c_o = 0.1046**, determined by anchoring the stagnation
member's graze at exactly 1 on the no-ceiling k=1 shape (at β = 1 the
inviscid term is zero, so linear and softmax₂ share the anchor).

## Graze calibration across the enriched family (requirement 1)

At c_o = 0.1046, graze ratios (canon values in Part-IV table):

| β | linear blend | softmax₂ blend | canon (ceiling) |
|---|---|---|---|
| +1.00 | 0.996 | 0.996 | 3.70 |
| +0.70 | 0.865 | 0.865 | 2.94 |
| +0.50 | 0.762 | 0.762 | 2.33 |
| +0.35 | 0.668 | 0.668 | 1.27 |
| +0.15 | **1.634** | 0.885 | 1.004 |
| +0.10 | **1.887** | 1.135 | 1.085 |
| +0.05 | **1.759** | 1.204 | 1.137 |
| 0 | **1.384** | 1.072 | 1.035 |
| −0.05 | 1.135 | 0.974 | 0.959 |
| −0.10 | 1.084 | 1.000 | 0.996 |
| −0.15 | 1.074 | 1.037 | 1.037 |
| −0.19 | 0.994 | 0.982 | 0.984 |
| −0.1988 | 0.937 | 0.930 | 0.932 |
| −0.19 lower | 0.912 | 0.909 | 0.911 |

The strong-FPG side is cured by construction (0.67–1.00 both blends, vs
1.27–3.70 over any ceiling). But the LINEAR blend breaks the legacy
mild-favorable band: there the viscous term is 30–81% of the inviscid one
(c_o·P_o/P = 0.81/0.55/0.40/0.30 at β = 0.15/0.10/0.05/0), inflating the
gate argument and collapsing the threshold — graze 1.38–1.89, i.e. those
members would ignite well before their LST stations. The softmax₂ blend
suppresses the cross-term quadratically exactly there and keeps the
legacy family at 0.89–1.20 (canon 0.93–1.14); adverse/separated members
are perturbed < 1% under either blend (c_o·P_o/P = 0.08–0.14).

## Marched Fig-4 family (requirement 2) — flatness resolved; softmax₂ wins

Rate constants unchanged; late secants x DG / N=1 x DG-N=1 station,
β order 1/0.55/0.35/0.2/0.1/0.05/0:

- **linear (vb)**: late [0.61 0.69 0.75 0.96 1.05 1.07 1.06]; onset
  [0.62 0.85 1.05 **0.74 0.61 0.69 0.87**] — devs (1.63x, 1.65x); the
  graze-predicted mild-band earliness materializes and **Blasius Rt1
  −15.7%** (anchor broken). Rejected on legacy grounds despite the good
  minimax.
- **softmax₂ (vbs)**: late [0.61 0.69 0.74 0.91 1.03 1.06 1.05]; onset
  [0.62 0.85 1.05 1.13 0.82 0.84 0.98] — devs **(1.63x, 1.62x), max
  1.63x — the best form of the entire study**, with Blasius late +5.0%,
  Rt1 −5.0%, Re_x(N9) −9.6%. The Part-IV low-H flatness is RESOLVED: the
  onset curve now rises with the members (no flat segment; the residual
  β = 1 earliness (0.62x) is a RAMP-TAIL interaction — with the threshold
  finally grazing AT the DG station, the tanh ramp's partially-open tail
  (gate 2–16% at Re_Ω/Re_Ω^c = 0.5–0.7) lets the floored rate book N=1
  before nominal opening, an effect the canon never exposed because its
  ceiling sat far below the stations. A march-anchored c_o (calibrate on
  the β=1 marched N=1 instead of the frozen graze, the paper's k-anchor
  philosophy) would trade this against mid-band lateness; the surface is
  flat and it was not run.)

## Wall behavior (requirement 3)

P_gate → 0 at the wall (⟨Î⟩₊ ~ y³-class, ⟨−Z⟩₊/R ~ y) → Re_Ω^c → ∞ →
gate CLOSED at the wall. Desirable on all three counts: it reinforces the
rate floor's own wall-vanishing (production doubly zero at the wall), it
keeps the frozen eigenproblem interior-anchored — boundedness gate PASSES
(sup b/u′_max saturates at 1.798e-3, peak fixed at y*/θ = 2.005 from
Rt = 1e5 to 1e7, wall fraction 5.7e-5) — and in the solver the canonical
max(P, tiny) clip keeps the 1/P² arithmetic finite while the tanh onset
→ 0 smoothly; no new singularity.

## Log-layer / zero-suite (requirement 4) — the leak stays closed

In a log layer Î < 0, so P_gate = c_o·Ω̂⟨−Z⟩₊/R alone ≈ 0.105·0.114/(2κu⁺)
~ 9e-4 at y+ = 100 → threshold ≈ 1.7e6, and Re_Ω = y+/κ never reaches it:
**raw max P_AI/P_SA = 4.3–4.6e-4 at ALL Re_τ (180/1000/5200), canon
4.0e-4; σ_t-blended 2.9–3.1e-4** — unlike the canon-ceiling gate (zb raw
3.9e-3 from y+ ≈ 760 opening) and even the C = 8000 retune (3.2e-3
residual at Re_τ = 5200). The no-ceiling blended gate closes the
parasitic path entirely, because the threshold in stable layers is set by
the small viscous coordinate, not by any constant.

## Full regression + impact at (a_inv, a_visc, c_o) = (0.19, 0.0276, 0.1046), softmax₂ blend

- Families: attached-adverse ≤ 3.7%/3.6%, lower branch ≤ 0.2%/0.1%.
- Cylinder noses: 2e6/7e6/2e7 clean (N_sup(80°) = 0.03/0.08/0.28, no
  seed-threshold crossings) — best-in-study alongside zc; ultra arm 1e9:
  crossing at 29.8° (handover 32.5°) — the experimental transcritical
  class, between zc (25.6°) and zb-joint (35.4°).
- Spheroid re72a0: ΔN = +1.29 (~0.011 L); Hiemenz/Sec VIII: 0.013
  e-folds — untouched.
- Mack map: unchanged by construction.

**Four-way summary (max both-panel dev / constants / leak / noses):**
canon-gate B: 4.40x / +1 / 3.9e-3 raw / 2e7 spurious; zb-joint C=8000:
1.77x / +1 / partial / clean; zc single-ε: 1.90x / net 0 / closed /
clean; **vbs blended gate: 1.63x / net 0 / closed / clean — RECOMMENDED,
superseding the Part-III recommendation.** The one deviation from the
user's stated preference: the LINEAR blend is not adoptable (legacy
mild-favorable band and the Blasius anchor break); the softmax₂
alternative the user themselves flagged is the working variant.

## Naming (requirement 6)

**c_o** as the user wrote it — "the viscous branch's weight in the onset
coordinate" — value 0.105 (graze-anchored at the stagnation member).
It is a dimensionless weight inside a coordinate, not a rate, so the
a_-prefix class (a_visc,o) would misclassify it; c_visc,o is verbose.
The adopted trio: **a_inv = 0.19, a_visc = 0.0276, c_o = 0.105**, and the
full form:

    rate  = min(a_inv, softmax₂(a_inv·Ω̂⟨Î⟩₊, a_visc·Ω̂⟨−Z⟩₊/R))
    gate  = ½[1 + tanh((Re_Ω/Re_Ω^c − 1)/0.35)]
    Re_Ω^c = 124.6 + 1.424 / softmax₂(⟨Ω̂Î⟩₊, c_o·Ω̂⟨−Z⟩₊/R)²   [NO ceiling]

Curious observation, recorded without a claim: c_o/ε_r = 0.1046/0.1455 =
0.719 ≈ k = 0.712 — the gate's viscous weight is the rate's times the
drain-compensation scale, to 1%. Different anchors produced it; possibly
coincidence.

## Part V honest ledger

1. The c_o determination is frozen-graze-anchored; the marched β=1 onset
   lands 0.62x from the ramp-tail interaction (quantified above). The
   march-anchored alternative was not run (flat trade against mid-band).
2. The strong-FPG graze slope mismatch (0.67 at β = 0.35 rising to 1.00
   at β = 1) is the residual of the ±12% slope agreement — the blend
   tracks, but not perfectly; β = 0.35's marched onset (1.05x) benefits
   from rate-lag cancellation.
3. Linear-blend results are complete and recorded (vb JSON) for the
   user's inspection despite the rejection.
4. Impact/zero instruments and their caveats identical to Parts I–III.

---

# PART VI (appended same day) — a_visc sweep, physical grounding, the c_o/eps_r=k question, and the P_o(H) flatness bound

*USER QUESTIONS on the Part-V vbs form (rate = the Part-III two-branch,
gate = blended no-ceiling softmax2). Rate constants a_inv=0.19,
a_visc=0.0276, gate c_o=0.1046. All offline on the analytic rigs; no
tex/solver edits. New study stage `--partv`/`--co`; forms vb/vbs already
present. Log `fpg_recal_partvi.log`; numbers in
`fpg_recalibration_vbs.json` (sweep keys eps=a_visc/a_inv).*

## Q1 — sweeping a_visc (a_inv=0.19 fixed, c_o=0.1046; the gate is a_visc-independent)

FIRST, a structural note that simplifies the whole sweep: in the vbs form
a_visc enters ONLY the rate (via eps_r=a_visc/a_inv in the softmax2 rate
coordinate); the gate coordinate uses c_o alone. So **the graze family is
identical for every a_visc** (Part V's table stands unchanged) — only the
marched panels and the rate-side impact move. The late secant is the
rate-controlled panel; the N=1 onset is gate-controlled.

| a_visc | eps_r | stagn. late (β=1) | stagn. early (β=1) | Blasius Rt1 | Blasius Re_x(N9) | mild β=0.10 late | dev_a | dev_b |
|---|---|---|---|---|---|---|---|---|
| **0.0276** | 0.145 | 0.61x | 0.34x | **−5.0%** | −9.6% | 1.03x | 1.63 | 1.62 |
| 0.035 | 0.184 | 0.70x | ~0.37x | −7.0% | −14.8% | 1.12x | 1.43 | 1.71 |
| 0.040 | 0.211 | 0.74x | ~0.38x | −8.4% | −18.5% | 1.19x | 1.35 | 1.77 |
| 0.045 | 0.237 | 0.78x | ~0.39x | −9.8% | −22.3% | 1.27x | 1.29 | 1.82 |
| 0.050 | 0.263 | 0.82x | 0.40x | −11.1% | −26.1% | 1.34x | 1.39 | 1.87 |

**Why a_visc can't simply be raised — three findings:**

1. **The late secant saturates sublinearly and never reaches Drela.**
   0.0276→0.050 is ×1.8 in a_visc but the stagnation late secant rises
   only 0.61→0.82x (×1.34). The pointwise rate a = a_visc·P_o IS linear in
   a_visc (nowhere near the a_inv ceiling: a_visc·P_o ≈ 0.007 << 0.19), but
   raising it pulls the N=1 and N=9 stations to lower Re_θ, compressing
   ΔRe_θ so the secant ΔN/ΔRe_θ grows sublinearly. Linear-in-log
   extrapolation needs a_visc ≳ 0.09 to reach 1.0x — where Blasius is
   already destroyed. **No reachable a_visc matches Drela's stagnation
   slope.**
2. **The EARLY secant is gate/ramp-limited, not rate-limited** — it barely
   moves (0.34→0.40x for ×1.8 a_visc). Near N=1 the disturbance sits in
   the tanh onset ramp's partially-open tail, so the deficit the user
   noted ("~3x low early at H=2.2") cannot be closed by ANY rate constant;
   it is the same ramp-tail interaction Part V flagged.
3. **The cost is a steady Blasius/mild-FPG over-amplification** driven by
   softmax2 bleed. Viscous fraction of the rate coordinate at the driving
   point (100% = pure viscous, i.e. Î≤0):

   | β | H | a_v=0.0276 | 0.035 | 0.04 | 0.05 |
   |---|---|---|---|---|---|
   | +1.0 | 2.216 | 100% | 100% | 100% | 100% |
   | +0.35 | 2.342 | 100% | 100% | 100% | 100% |
   | +0.20 | 2.411 | 78% | 90% | 95% | 99% |
   | +0.10 | 2.481 | 35% | 47% | 55% | 68% |
   | 0 (Blasius) | 2.591 | 14% | 20% | 25% | 35% |
   | −0.10 | 2.801 | 5% | 8% | 11% | 16% |

   Where Î already dominates (Blasius, mild FPG) the softmax2 quadrature
   adds a growing viscous term the inviscid branch does not need: Blasius
   Re_x(N9) drifts −10%→−26%, mild β=0.10 late over-amplifies to 1.34x.

**Impact rows (rate-side, a_visc-dependent), at a_visc=0.04 with the
a_visc=0.0276 / 0.05 zero-suite trend:**
- Cylinder noses 2e6/7e6/2e7 still clean at a_visc=0.04 (N_sup(80°) =
  0.04/0.11/0.40, no seed crossings); ultra 1e9 crossing 27.3° (0.0276:
  29.8°). Spheroid re72a0 ΔN = +2.6 (0.0276: +1.29). Hiemenz 0.018
  (Sec VIII still untouched).
- Zero-suite raw / blended max P_AI/P_SA: 4.6e-4/3.1e-4 (0.0276) →
  5.4e-4/3.5e-4 (0.04) → 6.2e-4/4.0e-4 (0.05) — the log-layer path stays
  closed (no ceiling); the rise is the buffer-layer floor scaling with
  a_visc, still ≤ 0.1% of SA production.

**Verdict on Q1:** no a_visc fixes the stagnation rate without moving
Blasius > 5% — even a_visc=0.035 already drifts Blasius Rt1 −7%
(Re_x −15%) while stagnation is still only 0.70x; **a_visc=0.0276 is the
largest value holding Blasius within ~5%.** The trade is fundamental to
the softmax2 blend, because that norm mixes the viscous term into
low-but-nonzero-(−Z) layers. CANDIDATE (noted, not refit): a sharpened
crossover — softmax_p with p>2, or a hinge that activates the viscous rate
only where Î≤0 — would suppress the Blasius bleed (Î-dominated) while
preserving the pure-viscous stagnation gain, possibly allowing a higher
a_visc at lower Blasius cost. Worth a Part-VII probe if the stagnation
late secant matters more than constant economy.

## Q2 — is a_visc=0.0276 physically grounded? PLAINLY: no, it is a fit

- **a_inv = 0.19 IS a measured eigenvalue**: the Michalke (1964)
  hyperbolic-tangent free-shear layer has most-amplified temporal growth
  ω_i,max = 0.1897 U₀/δ against peak vorticity U₀/δ, a normalization-
  independent ratio with NO free constant (paper Sec. II, cite
  michalke_1964). It is the inviscid Kelvin–Helmholtz / inflectional
  branch's eigenvalue.
- **a_visc = 0.0276 is PURELY EMPIRICAL** — determined by requiring the
  marched β=1 (stagnation) late secant to match Drela–Giles Eq. 29. It is
  a knob, not a measured growth rate. The only physical statement that can
  be made is order-of-magnitude: a_visc/a_inv = 0.145 puts the viscous
  branch at ~1/7 of the inflectional eigenvalue, consistent with the known
  fact that viscous Tollmien–Schlichting growth rates run roughly an order
  below inflectional/KH rates — but 0.0276 itself carries no eigenvalue
  provenance. If the paper adopts it, it must be presented as a
  Drela-anchored calibration constant (like k), NOT as a measured
  eigenvalue (like a_inv). This is the honest distinction the user asked
  for.

## Q3 — c_o/eps_r ≈ k: COINCIDENCE, not a removable constant

Definitions (as the user set them): eps_r = a_visc/a_inv = 0.145 (the
rate-floor viscous weight); k = 0.712 (the onset-gate scale in the CANON
Re_Ω^c = k·softmin2(...), anchored by the Blasius marched N=1 at
Re_θ=338); c_o = 0.1046 (the gate blend's viscous weight). Numerically
c_o/eps_r = 0.1046/0.1453 = 0.720 ≈ k = 0.712 (1%).

**The verdict is coincidence, and the algebra shows why cleanly.** c_o is
GRAZE-anchored: it is fixed by requiring the stagnation member to graze
the no-ceiling shape at 1, which uses only (Re_Ω*, P_o, Drela's Re_θ0) —
**none of which depends on a_visc**. Direct computation: the graze-anchored
c_o = 0.1048 for EVERY a_visc. Therefore c_o/eps_r = c_o·a_inv/a_visc
scales as 1/a_visc:

| a_visc | eps_r | c_o (graze) | c_o/eps_r | k·a_visc/a_inv |
|---|---|---|---|---|
| 0.0276 | 0.145 | 0.1048 | **0.721** | 0.103 |
| 0.035 | 0.184 | 0.1048 | 0.569 | 0.131 |
| 0.040 | 0.211 | 0.1048 | 0.498 | 0.150 |
| 0.050 | 0.263 | 0.1048 | 0.398 | 0.187 |

The proposed identity c_o = k·a_visc/a_inv would require c_o ∝ a_visc;
the graze anchor makes c_o CONSTANT in a_visc, so the identity holds at
exactly one a_visc (≈0.0276) and fails everywhere else. The three
constants are set by three unrelated anchors — β=1 marched RATE (a_visc),
β=1 frozen GRAZE (c_o), Blasius marched ONSET (k) — with no shared
determination; "sharing the Blasius anchor" does not occur (neither a_visc
nor c_o is set at Blasius). **c_o cannot be removed; it is an independent
constant.** (The near-miss is genuinely a numerical accident of the
current a_visc.)

## Q4 — P_o(H) flattens toward stagnation; this BOUNDS any c_o

The gate's viscous coordinate max_y P_o = Ω̂⟨−Z⟩₊/R vs H, with the
Drela-required gate argument √(B/(Re_Ω*−A)):

| β | H | Re_θ0 | Re_Ω* | max P_o | required arg | req/P_o |
|---|---|---|---|---|---|---|
| +1.00 | 2.216 | 6640 | 9624 | 0.1435 | 0.0145 | 0.101 |
| +0.70 | 2.256 | 4991 | 7630 | 0.1488 | 0.0164 | 0.110 |
| +0.50 | 2.297 | 3756 | 6053 | 0.1556 | 0.0184 | 0.118 |
| +0.35 | 2.342 | 2752 | 4685 | 0.1648 | 0.0211 | 0.128 |
| +0.20 | 2.411 | 1641 | 3018 | 0.1810 | 0.0265 | 0.147 |
| +0.10 | 2.481 | 827 | 1633 | 0.1992 | 0.0370 | 0.186 |
| 0 | 2.591 | 242 | 529 | 0.2292 | 0.0752 | 0.328 |

**Confirmed: P_o is nearly H-insensitive at low H.** Over β 0.35→1.0
(H 2.342→2.216) max P_o falls only −13% (0.1648→0.1435) while Re_θ0 rises
+141% and the required gate argument (∝1/√Re_Ω*) must fall −31%. So P_o
supplies only ~13/31 ≈ 42% of the variation Drela demands across the
strong-FPG band. **Residual bound:** a single graze-anchored c_o leaves
the gate argument ~(1−0.13)/(1−0.31) = 1.26× too large at the stagnation
end, i.e. the threshold ~1/1.26² ≈ 0.63× too low — matching the marched
β=1 onset (0.62x, Part V). No c_o can do better than this with P_o as the
carrier: the required per-member weight req/P_o rises monotonically
0.101→0.328 from stagnation to Blasius, so any single c_o matches exactly
one station. Fully tracking Drela's low-H rise would need a gate
coordinate steeper in H than P_o — e.g. the Re_θ-integral scale flagged in
Part IV — which leaves the pointwise-local model class. **The Part-V vbs
form's residual low-H onset earliness is therefore intrinsic to the
viscous coordinate's flatness, not a mis-set c_o.**

## Part VI honest ledger

1. Q1 early-secant values at intermediate a_visc are read to ±0.02 (the
   early secant is noisy near onset); the endpoints (0.34, 0.40) are solid.
2. The "Drela-match a_visc" does not exist in a physical range (late
   secant asymptotes ≈0.82x); the interpolation endpoint is reported as
   such, not as a usable value.
3. Impact cylinder/spheroid rows are stored for a_visc=0.04 only (the
   0.05 call overwrote them; zero-suite is keyed per-a_visc so both
   survive) — the trend 0.0276→0.04 is monotone and sufficient.
4. Q3's graze-anchored c_o=0.1048 vs Part V's marched-context 0.1046
   differ at the 4th digit (brentq vs secant); immaterial.
5. All instruments and caveats identical to Parts I–V.
