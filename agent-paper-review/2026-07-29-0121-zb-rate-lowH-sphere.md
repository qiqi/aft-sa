# Why the C=8000 (green) panel-(a) rate sits BELOW the canon C=1851.2 (blue) at low H — mechanism trace + low-H indicator-sphere degeneracy

*2026-07-29 01:21, model-calibration diagnostic thread. OFFLINE study on the
paper's analytic rigs (variant B / `zb`: rate = softmax2(a_inv·Ω̂⟨Î⟩₊,
a_visc·Ω̂⟨−Z⟩₊/R); onset gate sees ONLY the canon un-floored coordinate
P = Ω̂Î with Re_Ω^c = softmin2(C, 124.6 + 1.424/P²)). NO tex edits, NO solver
edits; the canonical `paper/figs/*` are untouched. Rate constants a_inv = 0.19,
a_visc = 0.0276 (ε_r = 0.1455) are IDENTICAL for both C — only the onset
ceiling C differs (1851.2 blue vs 8000 green in
`figs_explore/model_calibrate_candidate_zb.png`). Follow-up to
2026-07-28-1218 Parts III/IV/VI.*

Scripts:
- `paper/repro/analytic/zb_rate_lowH_trace.py` (Task 1; reuses the study
  kernel `fpg_recalibration_study._P_and_thresh`, marches the fig04 instrument
  instrumented to record the gate along the march).
- `paper/repro/analytic/indicator_sphere_lowH.py` (Task 2; reuses
  `fig01_indicator_sphere` machinery).

Artifacts: `figs_explore/zb_rate_lowH_trace.{png,json}`,
`figs_explore/indicator_sphere_lowH.{png,json}`.

---

## TASK 1 — the rate/gate trace (mechanism, not hand-waving)

FS wedges marched to N > 11 with the zb kernel; along the march we record
Re_θ, the realized dN/dRe_θ, and the onset gate S = ½[1+tanh((Re_Ω/Re_Ω^c−1)/
0.35)] evaluated at the production-peak band. "H=2.2" uses β = 1 (the FS floor,
H = 2.216); "H=2.3" uses β = 0.5 (H = 2.297). Numbers:

| H | C | Rt₁ | [5,9] window Rtθ | gate S over window | s_late | ×DG | Rt[S=0.5] | Rt[S=0.99] | S at N=9 |
|---|---|---|---|---|---|---|---|---|---|
| 2.216 | **1851 (blue)** | 1540 | 2335 → 2875 | 0.918 → 0.963 | 7.40e-3 | **0.99×** | 1367 | (>march) | 0.963 |
| 2.216 | **8000 (green)** | 4676 | 6491 → 7442 | 0.624 → 0.751 | 4.20e-3 | **0.56×** | 5908 | (>march) | 0.751 |
| 2.297 | 1851 (blue) | 1468 | 2400 → 3086 | 0.968 → 0.993 | 5.84e-3 | 1.02× | 1208 | **2835** | 0.993 |
| 2.297 | 8000 (green) | 4431 | 6299 → 7341 | 0.727 → 0.853 | 3.84e-3 | 0.67× | 5193 | (>march) | 0.853 |

(DG dN/dRe_θ = 7.44e-3 at H=2.216, 5.71e-3 at H=2.297.)

**The hypothesis is CONFIRMED, essentially exactly.** Reading the trace figure
`zb_rate_lowH_trace.png`:

1. **The gate-open Re_θ scales with C.** At low H the un-floored canon gate
   coordinate P = Ω̂Î is ≤ 0 through the interior (Part IV: dead for β ≥ 0.5),
   so the softmin sits AT its ceiling — Re_Ω^c ≡ C over the whole march, with
   no P-driven relief. The gate then depends only on Re_Ω/C, so raising C by
   8000/1851 = 4.32× pushes the whole gate curve to 4.32× higher Re_θ:
   Rt[S=0.5] moves 1367 → 5908 (ratio **4.32**, i.e. exactly the C ratio) at
   H=2.216, and 1208 → 5193 at H=2.297. Onset (Rt₁) likewise moves
   1540 → 4676 and 1468 → 4431.

2. **The [5,9] secant window is defined by N, not by gate state, so raising C
   drags the window into the STILL-OPENING part of the gate.** At H=2.216 the
   canon window (Rt 2335→2875) sits on a gate that is 0.92–0.96 open — near
   saturation — so its late secant realizes ~0.99× DG. The C=8000 window
   (Rt 6491→7442) sits on a gate only 0.62–0.75 open; the realized rate =
   (ungated kernel) × S is throttled to 0.56× DG. Same story at H=2.297
   (window gate 0.97–0.99 vs 0.73–0.85; 1.02× vs 0.67×).

3. **The S=1 (saturated) limit rate is the SAME shared value for both C** —
   the rate constants are identical and the gate is blind to the viscous rate
   term, so both curves approach the same dN/dRe_θ plateau (~8.5e-3 at
   H=2.216, visible as the blue curve's top). Raising C does NOT lift that
   ceiling; it only moves the measurement window away from it.

4. **Re_θ where S reaches 0.99 (the task's quantitative ask).** At H=2.216
   neither C reaches S=0.99 within the resolved march, but the N=9 station is
   BEFORE saturation for both — decisively so for green: gate at N=9 is 0.963
   (canon, essentially saturated) vs **0.751** (C=8000, deep in the ramp). At
   H=2.297 the canon gate crosses 0.99 at Rt=2835, INSIDE its own [5,9] window
   (Rt9=3086) — the blue window straddles saturation — whereas C=8000 never
   reaches 0.99 (N=9 gate = 0.853).

### Reconciliation with the user's expectation

The user expected "higher onset reduces the viscosity effect and boosts the
rate toward the high-Re limit." The realized march rate is
dN/dRe_θ = (ungated kernel rate) × S(Re_Ω/Re_Ω^c). In variant B the gate is
blind to the viscous rate term and the rate constants are identical for both
C, so **the gate-saturated (S→1) ceiling is the SAME shared ungated value —
raising C cannot boost the rate above it.** What raising C actually does is
raise the threshold Re_Ω^c ∝ C (at low H the coordinate is dead so the softmin
is pinned at C), so at any matched Re_θ the ratio Re_Ω/Re_Ω^c is 1851/8000
smaller and the gate is less open. Because the [5,9] window is fixed by
integrated N rather than by gate state, delaying onset shoves the window to
~4.3× higher Re_θ but the threshold rose in lockstep, so the window lands on
the sub-saturation (0.6–0.75-open) shoulder of the gate — the measured secant
is LOWER, not higher. The intuition would hold only if the ungated rate itself
climbed with Re_θ (a genuinely rising high-Re limit); here it is flat-to-
declining and the gate is the sole Re-dependent factor, so higher C purely
under-samples it. This is worst at low H precisely because there the coordinate
is dead and the threshold is fully C-proportional; as H rises the coordinate
revives, the softmin drops below C, and the green/blue gap shrinks
(β=0.2/H=2.41: 1.06× canon vs 0.89× C=8000 — a much smaller gap).

---

## TASK 2 — indicator-sphere low-H degeneracy

FS trajectories for H = 2.216 (β=1, "2.2"), 2.30, 2.40, 2.50, 2.59 (Blasius)
drawn on the RP² sphere (`indicator_sphere_lowH.png`), thin = trajectory,
thick = OS amplifying band (production > ½ peak, evaluated at 2×Re_θ0). The two
lowest-H members (H=2.216, 2.30) carry **NO amplifying band** — they are
subcritical/dead in the rate coordinate — while H ≥ 2.40 do; the strong-FPG
curves bunch on the neutral locus Ω̂Î = 0 (zoom panel).

**Separation table** (unit-vector n = (X,Y,Z)/R along the shear region; the
decisive quantity is the amplifying coordinate P = Ω̂Î the gate/rate reads):

| pair | geodesic mean | Euclidean mean | max P=Ω̂Î (A / B) | \|ΔP\| | Drela Re_θ0 ratio |
|---|---|---|---|---|---|
| **H=2.216 vs 2.30** | 12.1° | 0.206 | 3.24e-4 / 1.68e-4 | **1.6e-4** | **1.81×** |
| H=2.30 vs 2.40 | 8.6° | 0.146 | 1.68e-4 / 1.23e-2 | 1.2e-2 | 2.05× |
| H=2.216 vs 2.59 | 26.7° | 0.457 | 3.24e-4 / 7.77e-2 | 7.7e-2 | 27.3× |

(The per-curve geodesic *max* is a shared near-wall pole-crossing artifact —
X=Y=Z→0 at η→0 makes n ill-defined there — so means are the robust metric.)

**Answer — do H=2.216 and 2.30 BLs bunch on the sphere?** YES, decisively in
the coordinate that matters. Both sit at max P = Ω̂Î ≈ 1–3e-4 — i.e. both on
the neutral locus P≈0, DEAD (consistent with Part IV: P ≤ 0 interior for
β ≥ 0.5) — differing by only 1.6e-4, whereas the amplifying coordinate first
comes alive only at H=2.40 (P=1.2e-2, ~40–70× larger) and reaches 7.8e-2 at
Blasius. Their Euclidean/geodesic means differ ~2× less than the span to
Blasius, and that residual difference lives in the near-wall curvature sector
that the clipped gate coordinate ⟨Î⟩₊ discards.

**Does this explain why the gate cannot set distinct onsets at low H?** YES —
this is the geometric root of the P_o-flatness / onset-resolution limit
(Parts IV/VI). The single-branch gate reads P = Ω̂Î, which is degenerate
(≈1e-4 for both H=2.216 and 2.30) exactly where Drela demands a 1.81×
difference in critical Re_θ0. A pointwise-local coordinate that collapses two
physically distinct boundary layers onto the same near-zero value cannot
encode their different onsets; the softmin ceiling C then supplies the same
constant to both. That is why panel (b) goes flat at low H.

---

## PART VIII — the two-branch onset gate (coordinator Task 3)

*USER PROPOSAL: the rate is two-branch (a_inv·Ω̂⟨Î⟩₊ AND a_visc·Ω̂⟨−Z⟩₊/R) but
the canon gate is single-branch — Re_Ω^c = softmin(C, A+B/P_I²), P_I = Ω̂Î only.
Tasks 1–2 CONFIRM the premise: at low H, P_I ≈ 0 for all profiles (they bunch),
so the softmin saturates to the constant C — a non-H-resolving stand-in. FIX:
make the gate two-branch, replacing C with a curvature branch,
Re_Ω^c = softmin(A + B/P_I², A_c + B_c/P_curv²), P_curv = Ω̂⟨−Z⟩₊/R. Form `vg`
in the study script; net-zero by sharing A_c = A and calibrating B_c only.*

### VIII.1 — calibration and the onset-threshold table

B_c = **129.74** (k-carrying), fixed by grazing the stagnation member (β=1) at
1 — the vbs-style frozen anchor. **The curvature branch does break the
constant-C degeneracy**: it now sets DISTINCT, H-resolving thresholds where the
constant C could not (k=1 shape, evaluated at each member's neutral point):

| β | H | Re_θ0 | br_I (inflectional) | br_curv | const C | winner | graze_vg |
|---|---|---|---|---|---|---|---|
| +1.0 | 2.216 | 6640 | ~1e18 (dead) | **7982** | 2600 | curv | 0.999 |
| +0.5 | 2.297 | 3756 | ~1e18 | **6019** | 2600 | curv | 0.765 |
| +0.35 | 2.342 | 2752 | ~1e18 | 5174 | 2600 | curv | 0.670 |
| +0.2 | 2.411 | 1641 | ~1e18 | 4160 | 2600 | curv | 0.523 |
| +0.1 | 2.481 | 827 | 2793 | 3401 | 2600 | infl | 0.953 |
| 0 | 2.591 | 242 | 430 | 2598 | 2600 | infl | 1.024 |
| −0.10 | 2.801 | 97 | 184 | 1764 | 2600 | infl | 0.996 |
| −0.199 | 3.982 | 36 | 130 | 762 | 2600 | infl | 0.944 |
| −0.19 lower | 4.922 | 26 | 128 | 620 | 2600 | infl | 0.929 |

Readings:
- The curvature branch WINS exactly where intended — the strong-FPG members
  (β ≥ 0.2, H ≤ 2.41) whose P_I is dead — giving H=2.216 → 7982 vs H=2.297 →
  6019, i.e. **distinct onsets for H=2.2 vs 2.3** (the constant C gave both
  2600). Direction correct; degeneracy broken.
- On the mild-FPG/Blasius/adverse/separated side the inflectional branch wins
  (br_I 128–430 « br_curv 620–2598), so the curvature branch **does NOT fire
  on the separated/Stewartson branch** (separated graze checked below) — the
  regression concern is clean.

### VIII.2 — the two-branch gate IS vbs at low H (the key structural result)

The β=1-graze-anchored B_c = **129.74** and the Part-V vbs strong-FPG
equivalent B/c_o² = 1.424/0.1046² = **130.15** agree to **0.3%**. This is not a
coincidence: in the strong-FPG limit P_I → 0, vbs's P-blend
Re_Ω^c = A + B/softmax₂(P_I, c_o·P_curv)² → A + (B/c_o²)/P_curv², which is
*exactly* the vg curvature branch with A_c = A, B_c = B/c_o². **The user's
softmin-of-two-thresholds and Part-V's P-blend are the SAME curvature branch at
low H**; they differ only in the moderate-H crossover (softmin picks the lower
threshold; the 2-norm blend always nudges the argument up, lowering the
threshold slightly). Both therefore inherit the identical P_curv-flatness
bound (Part VI Q4): the graze falls 0.999 → 0.52 across β=1 → 0.2 because a
single B_c cannot track Drela's 4× Re_θ0 rise with P_curv that varies only ~26%
— the same residual vbs has.

### VIII.3 — marched Fig-4 (both panels, family-wide) and regression

Controlled comparison — SAME two-branch rate (a_inv=0.19, a_visc=0.0276),
three GATES, canonical 1600×1200 grid (`model_calibrate_candidate_2branchgate.png`,
`zb_2branch_gate.json`). canon-C1851 / C8000 are the zb-figure blue/green
(loaded from the zb JSON); vbs and vg freshly marched:

| gate | panel(a) worst | panel(b) worst | H=2.216 onset (Rt1, ×DG-N1) | H=2.285 | H=2.342 | constants |
|---|---|---|---|---|---|---|
| canon const C=1851 (blue) | **1.09×** | 4.40× | 1540, **0.23×** | 0.35× | 0.48× | +1 (C kept) |
| canon const C=8000 (green) | 1.77× | 1.73× | 4676, 0.69× | 1.05× | 1.45× | +1 |
| vbs (P-blend) | 1.63× | 1.62× | 4190, 0.62× | 0.85× | 1.05× | **net 0** |
| **vg (two-branch gate)** | **1.63×** | **1.62×** | 4180, **0.62×** | 0.84× | 1.05× | **net 0** |

- **The two-branch gate resolves the low-H degeneracy the constant C cannot.**
  Panel (b) worst deviation drops 4.40× (const C=1851) → **1.62×**, and H=2.2 vs
  2.3 now get DISTINCT, direction-correct onsets (Rt1 4180 vs 3583; ×DG 0.62 vs
  0.84) instead of the constant C's degenerate near-identical early crossings
  (0.23× / 0.35×). This is exactly the fix the user's premise predicted.
- **vg IS vbs.** Both-panel deviations are identical (1.63×/1.62×), and the
  low-H onsets agree to <0.3% (Rt1 4180 vs 4190). The B_c ≡ B/c_o² algebra
  (VIII.2) plays out in the march: the user's softmin-of-two-thresholds and
  Part-V's P-blend are the same curvature physics; the softmin-vs-2-norm
  crossover at moderate H is invisible in both panels.
- Residual (shared with vbs): the β=1 marched onset is 0.62× (early) from the
  tanh ramp-tail interaction, and mid-band is slightly late — the P_curv-
  flatness bound (Part VI Q4), not removable by any single B_c/c_o.

**Regression at (a_inv, a_visc, B_c) = (0.19, 0.0276, 129.74), A_c = A shared:**
- Wall-boundedness: PASS — sup b/u′_max saturates at 1.798e-3 at y*/θ = 2.005
  from Rt=1e5→1e7, wall fraction 5.7e-5 (identical to vbs/zc; the rate floor
  is unchanged).
- Attached-adverse family normal (late 0.76–1.04× over H 2.6–3.3); separated/
  Stewartson lower branch (β=−0.19, H=4.92) late 0.76×, Rt1 1.20× DG and graze
  0.929 — the **curvature branch does NOT fire there** (br_I 128 « br_curv 620;
  matches canon 0.911 / vbs 0.909).
- Spalart zero-suite: raw max P_AI/P_SA = 4.3–4.5e-4, σ_t-blended 2.9–3.1e-4 at
  Re_τ 180/1000/5200 — the **log-layer leak is CLOSED** (no ceiling; the −Z
  coordinate keeps the threshold high where Î<0), clean like vbs and canon
  (4.0e-4), and unlike the kept-ceiling zb (raw 3.9e-3) or C=8000 (3.2e-3
  residual).
- Mack seed map: unchanged by construction (no kernel-coordinate dependence).
- Constant bookkeeping: **net 0** with A_c = A shared (B_c replaces the single
  constant C). The two-constant curvature branch (free A_c, B_c) is NOT viable:
  a least-squares fit to the neutral points needs A_c ≈ −8230 (negative floor,
  predicts negative thresholds at Blasius) because P_curv is too flat to span
  Drela's range with a positive floor — the same flatness bound.

### VIII.4 — recommendation

**The user's two-branch onset gate is correct, clean, and net-zero — and it is
equivalent to the already-recommended Part-V vbs form, not better.** It is the
more transparent *presentation* of the same physics: it literally repairs the
inconsistency the user identified (two-branch rate ↔ two-branch gate, each with
an inflectional and a curvature branch), and its curvature branch
A_c + B_c/P_curv² is exactly vbs's strong-FPG limit (B_c = 129.74 ≡ B/c_o² =
130.15). Both give panel-(a)/(b) worst deviations 1.63×/1.62× — the best
combined Fig-4 metric in the study, beating the constant-C forms (1.09×/4.40×
at C=1851; 1.77×/1.73× at C=8000) — and both break the low-H onset degeneracy
into distinct, Drela-tracking H=2.2-vs-2.3 crossings. Neither beats the other:
they share the P_curv-flatness residual (β=1 onset 0.62× early). If the paper
wants the gate to *read* as two-branch (mirroring the rate and making the
curvature mechanism explicit), adopt vg; if it prefers one fewer operation,
vbs is identical in every measured respect. **Verdict: adopt the two-branch
gate OR vbs — same physics, same numbers, net-zero constants; vg is the more
honest presentation.**

---

## Files

- Scripts: `paper/repro/analytic/zb_rate_lowH_trace.py`,
  `paper/repro/analytic/indicator_sphere_lowH.py`,
  `paper/repro/analytic/zb_2branch_gate.py`; kernel form `vg` added to
  `paper/repro/analytic/fpg_recalibration_study.py` (module consts AC/BC).
- Figures: `figs_explore/zb_rate_lowH_trace.png`,
  `figs_explore/indicator_sphere_lowH.png`,
  `figs_explore/model_calibrate_candidate_2branchgate.png`.
- Data: `figs_explore/zb_rate_lowH_trace.json`,
  `figs_explore/indicator_sphere_lowH.json`, `figs_explore/zb_2branch_gate.json`.

## Honest ledger

1. "H=2.2" is β=1 (H=2.216, the FS floor — H<2.216 is unattainable for a FS
   wedge); "H=2.3" is β=0.5 (H=2.297). Numbers labelled with the true H.
2. The sphere geodesic *max* per pair is a near-wall pole-crossing artifact
   (n undefined as X,Y,Z→0 at the wall); means are the reported metric, and the
   decisive quantity is the well-defined amplifying coordinate P=Ω̂Î.
3. B_c is frozen-graze-anchored at β=1 (vbs-style); a march-anchored B_c would
   trade the 0.62× β=1 earliness against mid-band lateness — flat surface, not
   run (same trade Part V flagged for c_o).
4. All marches at the 1600×1200 grid; devs are same-instrument ratios. The
   canon-C1851/C8000 baselines are the committed zb-JSON rows (same grid).
5. Front/impact instruments (cylinder/spheroid) not re-run for vg — its rate is
   the zb/vbs rate and its gate equals vbs at low H, so those rows carry over
   from Parts III/V unchanged; only re-verify if vg is adopted over vbs.
