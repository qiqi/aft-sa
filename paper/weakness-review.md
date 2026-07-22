# SA-AI paper — weakness-of-argument review (open items)

Living catalog of places where the paper's argument is not water-tight.
**Open items only** — resolved items are removed as they close (their record is
this file's git history). Last full re-scan: 2026-07-13 (post
favorable-rate round; found no new argument-level weaknesses — five
mechanical consistency residues of the round were fixed directly: the
§II six-decisions overview and the eq:transport signature/footnote missing
the rate factor, a dangling numerics clause in the c_A bracket, the
assembled-model equation range, and the Eppler §VI transfer sentence). Branch
`favorable-rate`, constants (floor, g_c, s) = (254, 1.005, 11),
K_λ = 6.1, K_r = 5.5, a_max = 0.19, c_A = 4, p = 4, c_ν,ai = 1/12, gate mode 4.

Closed to date (see git history for the full entries): a_max provenance
(eigenvalue), s-insensitivity claim (third anchor), the "merely solve the
anchors" overclaim and the c_ν,ai/c_A/p conditionality (two-tier quotient
structure), c_A circularity in Fig. 4 (two-sided failure bracket),
onset-vs-rates framing tension, cavity marginal-front breakaway (ring-averaged
gate), NLF bucket family ΔC_d, Eppler α=7° early reattachment (shared
e^N-framework limitation, compensating error closed), Blasius interior
residual framing (quoted as the fully-determined kernel's residual),
**O2** the N=1 departure anchor (quantified by the anchor-level sweep and
stated in §III.A: predictions pinned by N=9, the family response selects
N≈1; its evidentiary-concentration residual folded into O9), and **O18**
the calibration/validation separation (made precise in the §III intro:
constants-level separation, forms iterated with results visible, the
Eppler α=7° regression left to stand; the genuine blind test is the
conclusion's proposed community-benchmark comparison). **O1** the solve valley (quantified in §III.A — (g_c,s) slides
(1.005,11)↔(1.020,9.9) within 2.2% anchor residuals, mid-adverse interior
±5–8%; residuals quoted; future work may exploit the valley as a
one-parameter family of near-anchored kernels), **O3** τ (bracketed by
numerics and admitted as such in §III.C: too wide throttles SA production
past c_v1, too narrow outruns streamwise mesh resolution; the §III intro
names it the one non-canonical constant), **O4** the adverse onset
intercept (quantified in §III.A and Fig. 3: permission at 1.0×Rtc
mid-adverse rising to 1.5× at the separation limit; visible N=1 ignition
1.4×/1.7× — this catalog's old "factor 2–3" was stale, measured at the
pre-three-anchor kernel), **O5** the floor claim (now "suffices in giving
the physical trend," quantified, with future-work λ_p modulation of the
adverse cliff; the favorable rate over-estimate likewise quantified at
1.2–2.0×/1.3–2.4× — the λ_p rate modulation then flagged as future work
was subsequently done, closing O10), and **O6**
K_λ's exponential form (reach quantified: ±8% to β≈0.55, 22% early at
β=1; the airfoil cases peak at equivalent β≈0.5). **O7** the ν̃~u′ℓ scaling (reframed in §I as a motivation, not a
representation claim, with the explicit statement that the model
constrains neither the disturbance's amplitude nor its location — only
its effect, the transition location), **O8** the eigenvalue-to-production
bridge (the non-exact translation — temporal↔spatial, envelope↔pointwise,
tanh↔wall-bounded — is admitted in §III.A with room for future modest
adjustment of a_max), and **O9** the gate's double duty (design intent
stated: one job, wall pinch, unity elsewhere the ideal; the c_A bracket
argued from the design statement at both ends; band throttling a declared
side effect; manuscript-wide framing scan completed — the residual
evidentiary-concentration caveat is recorded here: the unfitted
mid-adverse family confirms the c_A/p brackets and the N=1 level, so an
independent discriminating observable would still strengthen the case),
**O10** the favorable-branch rate philosophy (RESOLVED 2026-07-13,
branch `favorable-rate`: the rate factor f_λ = 1/(1+(K_r·max(0,λ_p))²)
of eq:fpgrate, K_r = 5.5 fit at the single worst point β=0.35, brings
the favorable family onto Drela's correlation — means 0.84–1.00×,
lates 0.92–1.17× over β=0.05–1, vs 1.2–2.0×/1.3–2.4× cliff-only; the
form is C¹ at λ_p=0 so anchors and K_λ are untouched; the onset price
is stated (N=1 runs 1.2–1.3× late in mild acceleration); the 61-case
fleet rerun shows the airfoil validation is essentially unmoved — NLF
fronts ≤0.003c, Eppler bit-level, flat plate +2% Re_θ at low Tu from
the plate's residual λ_p≈+0.01, honestly caveated in §IV — and AGS
tracking improves 13%→12%),
and **O11** the mesh-coupled Γ_g evaluation (resolved by scoping,
2026-07-13, per user directive: the manuscript presents the continuum
model and does not discuss numerics at all — the
compact-kernel/neighborhood-average passages were removed from §II, the
§III development-history parenthetical de-numericized, and the §V
"earns its keep" passage cut; `paper/numerics.md` is the sole and
complete record of the discrete scheme, its failure mode, and the
operator-variant studies), and **O12** the unconstrained mid-transition
blend (resolved by admission, 2026-07-13, per user directive: §II now
states the blend is an interpolation with no physics claimed for it ---
no turbulent spots, no intermittency distribution; the target is the
transition location and the macroscopic aerodynamics, the transitional
region being usually short on the geometry's scale, with the
long-transitional-zone limitation flagged; §I scope says the same in one
line), **O13** the Mack-map overstatement (resolved by admission,
2026-07-13: "so the same map applies" removed; §III.D now states the
correspondence is not exact — sharp N_crit vs the χ=1→c_v1 smear, ~2
e-folds — and that adopting Mack wholesale is an approximation whose
flat-plate consequence is quantified under a stated convention), and
**O14** the onset-convention dependence (resolved explicitly,
2026-07-13: §IV states that quoting one number requires choosing a
crossing, gives BOTH readings — χ=1: ~12%, +5/−4..−12/−7%; χ=c_v1:
+12..+19% and +35% at Tu=0.6 — and grounds the χ=1 choice: AGS falls
inside the model's transitional band at every Tu but the lowest, nearer
the χ=1 end; the conclusion names the convention too), and **O15** the
in-bubble observable (resolved by admission, 2026-07-13, per user
directive with literature verification: §VI now flags the deep
reverse-Cf excursion at Re=6e4/1e5 — C_f≈−0.005, 4–5× the
post-reattachment turbulent peak at 1e5 — as a scope limit, not a
prediction. Literature check CORRECTED the initial "unphysical"
hypothesis in one direction: a sharp negative C_f surge near the bubble
rear IS physical (Spalart & Strelets 2000 call it a momentary turbulent
re-separation; Alam & Sandham 2000 bubbles carry 15–20% reverse flow at
the absolute-instability boundary — both now cited), but its computed
DEPTH exceeds anything in DNS, where the surge stays of the order of
the downstream turbulent C_f (mfoil likewise: −0.0006 at 1e5, a fifth
of its turbulent peak). The passage attributes the exaggeration to the
declared scope — no transitional-zone physics; mid-bubble C_f is an
uncalibrated byproduct of the σ_t interpolation and SA's separated-flow
behavior — and names the deliverables: separation, in-bubble transition
onset, reattachment, forces), **O16** the α=7° sweep digits (resolved
by restatement, 2026-07-13: §VI now states the 0.01–0.02c cross-tool
measure noise explicitly — XFOIL's α=7° closure reads 0.40c by its
transition station, ≈0.415c by the paper's C_f-threshold measure — and
quotes the conclusion at the precision the sweep supports: measured
0.48c needs N_crit≈11, about two units above the ≈8–9 that matches
α=5°, the quoted range itself spanning the noise and the two-unit gap
far exceeding it), and **O17** the α=4° extrapolated datum (resolved by
restatement, 2026-07-13: §V now separates what the orifices strictly
establish — the front has advanced to the forward end of their range,
consistent with the computed 0.25c — from the ≈0.31c extrapolation of
the x_tr(c_l) trend, against which the computation reads 0.06c forward;
the entry is explicitly weighted below the located fronts). Item numbers are
stable; removed numbers are not reused.

---

## Open items

None. All catalogued items (O1–O37 and the pre-numbered entries) are
closed; see the ledger above and this file's git history for the full
records. Next accumulation begins at O38.

**O37** (§V, user-found via the appendix sheets, 2026-07-15): the claim
"across a wide α range the transition on one surface or the other is
governed by slow TS accumulation rather than a severe adverse gradient"
overclaimed. Band-λ_p extraction across the matrix shows the design pins
most fronts: α=0 upper at the recovery steepening (λ_p through −1 near
0.35c, front 0.37c); α=9–15 upper just aft of the suction peak; α=9–15
lower at the 0.6c recovery wall behind a cliff-guarded λ_p≈1–1.7
rooftop. Genuinely accumulation-set: α=4 upper (gently adverse
λ_p∼−0.1…−2, front 0.25c, the free window that moves the polar's upper
knee — and where model-vs-e⁹ spread is largest, 0.25 vs 0.33); the
low-α lower surface is intermediate (gentle plateau, ignition just
ahead of the wall). → §V opening rewritten around pinned/free structure
and what the case actually tests (holding the designed runs + prompt
ignition at recovery + the free-window march).

## Adoption sweep of 2026-07-15 (sigma_D tie)

The tie sigma_D = 1 − (c_b1/κ²c_w1)(1−σ_P) replaced the calibrated τ_D
across model, paper, whitepaper, and repro (verification in
scripts/models/sigma_d_tie/RESULTS.md). Post-adoption weakness sweep of
every changed passage; seven items, fixed in place:

**O30** (§II eq:sa list): eq:blend now uses κ, which was missing from
the "constants ... as defined there" list. → κ added.
**O31** (III.A instrument note, arbitrariness): "the instrument omits
the destruction floor" gave no reason — reads as an arbitrary
convenience. → Rationale stated: with the floor the envelope becomes
seed-amplitude-dependent, while the Drela–Giles correlation it anchors
to is amplitude-free; effect quantified in III.E.
**O32** (III.E CFD bound, overclaim): "verified at the most sensitive
case of each family" claimed a superlative not established. → The two
cases are named (NLF rooftop α=4°, Eppler bubble α=5°).
**O33** (coarse-mesh text+caption, overclaim): "the two discretize the
same linear solution" — discretely they differ at truncation level; the
equality claim belongs to the continuous solutions. → Plain statement
of the observed table equality; mechanism claim dropped.
**O34** (§II/§VII recovery statement): "σ_P=σ_D≡1 ... recovers baseline
SA" treats σ_D as independent — under the tie it is not. → "σ_P≡1 (and
with it σ_D=1)". Both documents.
**O35** (§VII): "the gated destruction D" → "the tied destruction D".
**O36** (repro): constants_report.py carried tau_D (now R_tie, marked
derived); README reading-order table's §III letters predated the
five-stage restructure (background→III.B, triple→III.C, K_λ/K_r→III.D,
handover→III.E, receptivity→III.F, assembled→III.G).

Also verified in the sweep: the D/P ≤ 10⁻³ bound is stated identically
in §II, III.E, and whitepaper §1.4/§3.5 (the earlier 2×10⁻⁴ figure was
the band-ideal estimate; the marched value at χ=1 is ~10⁻³); the
negative-branch passage needed no change (the trimmed version carries
only the f_n consistency requirement, which the tie does not touch);
all five constants-consistency tests pass under the new canon
(sigmaDTie=1.0, no tau_D); constants_report prints 15/15 OK.

## Second sweep of 2026-07-15

Deeper pass: whitepaper figure-caption numbers traced to their repro
prints (Stewartson f''(0)=−0.0910 and 6% backflow re-run and confirmed;
Fig. 3/8/11/13/17/19 numbers re-checked against the paper), tone and
300-liang greps re-run on the newly written whitepaper Sec. 1 (clean),
Sec-3.x pointers re-verified. Three more items, fixed in place:

**O27** (whitepaper preamble, reproducibility overclaim): "every number
below is reproducible from paper/repro/" glossed over the Flow360
binary the coupled-RANS results require (the repro README states it;
the summary did not). → Split honestly: analytic figures/tables from
repro/analytic/ alone; RANS results additionally require the Flow360
binary (driver re-runs, cfd regenerates).
**O28** (whitepaper coarse-mesh table caption): said "second-order FD"
without the both-fields-on-the-coarse-grid statement the paper's
caption carries. → Mirrored.
**O29** (repro fig02 docstring, stale): said the operative Γ of the two
anchor profiles are "marked" — the markers were removed from the figure
in the declutter round; the script prints them. → "printed (the map
itself is left unannotated)".

## Sweep of 2026-07-15 (paper + repro + whitepaper, pre-Spalart round)

Focus on the whitepaper, freshly enriched with the paper's §II content.
Checks run: all five constants-consistency tests (pass); whitepaper §3
numbers spot-checked against the paper (envelope anchors 338/1108,
d ln Re_θc/dβ = 12.3, nose margin 2.1×, operating points 0.865/0.059 and
1.575/0.786, K_r solve 5.47→5.5, τ_D root 1.356→1.36, c_ν plateau ±2%,
c_A/p sag percentages, raw-kernel 400–677, NLF fronts ≤0.003c, AGS
percentages, Eppler reattachment digits — all consistent); every §3.x
and Fig-number pointer in the enriched whitepaper §1 verified against
where the content actually lives. Eight items found, all fixed in place:

**O19** (whitepaper §1.1, representation overclaim): "repurposes it to
carry a Tollmien–Schlichting envelope amplitude" claimed the
representation the paper itself declines ("we do not constrain ν̃ to
represent how large the disturbance is"). → Restated as the mechanism:
ln(ν̃/ν̃_∞) accumulates as an e^N-style envelope amplification factor.
**O20** (whitepaper seed subsection, missing caveat): Mack's map was
presented "adopted unchanged" without the approximation the paper
states. → Added: e^N declares a sharp N_crit while the model's threshold
is smeared over the χ=1→c_v1 handover (~2 e-folds); Fig. 8 pointer.
**O21** (whitepaper §1.2, wrong attribution): the shape-factor-map
collapse credited "Wazzan–Okamura–Smith"; the 1981 H-Rx paper is
Wazzan–Gazley–Smith. → Corrected.
**O22** (stale terminology): λ_p^worst in whitepaper §3.4 and "Worst
(triggering) points" in its Fig. 5 caption survived the paper's
most-amplified-point rename; repro docstrings (fig05_06_klambda,
fit_fpg_rate_slope) likewise. → All aligned (λ_p^⋆; "Most-amplified
(triggering) points"; "largest overshoot of the un-factored family").
**O23** (whitepaper Fig. 4 caption, wrong pointer): "favorable as in
§3.5" pointed at the turbulent-wall-layer subsection. → §3.4.
**O24** (whitepaper, misfiled fact): the gated operating points sat
under §3.5 "Turbulent wall layer"; they are kernel facts. → Moved to
§3.2, labeled "at the canonical c_A".
**O25** (whitepaper §1.1, false forward reference): the six-decisions
paragraph said the freestream seed "closes this section", but the
SA-calibrations contact subsection followed it. → Subsections reordered
(contact §1.6, seed §1.7).
**O26** (paper §III roadmap, imprecision): "the handover widths τ and
τ_D, fixed on the fully turbulent wall layer" — only τ_D is; τ is
bracketed by numerics. → Split in place ("τ bracketed by numerics, τ_D
fixed on the fully turbulent wall layer").
