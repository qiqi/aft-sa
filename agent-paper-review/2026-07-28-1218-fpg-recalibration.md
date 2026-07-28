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
