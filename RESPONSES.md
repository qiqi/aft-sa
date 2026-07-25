# Claude session log — important responses (append-only)

Newest entries at the bottom. Each entry is timestamped; skim headers.

---

## 2026-07-24 ~19:50 — backfill of today's key findings

### Daedalus section compiled onto the final-kernel campaign (commit 5509ab0)
- 8 completed canon runs in the paper (both families L1 at α=4,5,6; structured
  L2 at α=4,5); pending runs excluded everywhere; generators fill in
  automatically as runs finish.
- Headline: at η=0.31, α=4° the recomputed bubble spans 0.489–0.661c vs the
  e^N strips' 0.472–0.660c (predecessor kernel: 0.04–0.06c, 3–4x short).
  Reattachment agrees to <0.01c, separation to 0.02c. The second-gate-pinch
  prediction is fulfilled in 3D.
- χ=1 front still leads the strips' N_crit contour by ~0.09c = the expected
  2-e-fold convention gap (χ=1 ≡ N=11.65 vs N_crit=13.6).
- Both spanwise phenomena survive at canon: outboard bubble stretch
  (reattachment 0.66→0.79→0.95c at η=0.31/0.92/0.97, strips 0.66→0.74→0.84)
  and the tip-vortex trip (χ>1 from LE for η≳0.995).

### AVL reference bug — you were right (commit 53d0a40)
- run_avl grepped the NEAR-FIELD CDind (0.0121 at α=4°, implied e=0.66)
  instead of the Trefftz CDff (0.0092, e=0.87) three lines below in the same
  file. Appendix D always said "Trefftz"; the code betrayed it.
- Corrected reference CDs: 0.02028 / 0.02212 / 0.02414 at α=4/5/6°.
  RANS drag now lands within ±3 counts (1.5%) at L2, ±11 at L1.
  Predecessor's true deficit was 11–17 counts (not 40–50); final kernel
  returned 6–14 of them.
- Discretization check (cached 24×80 hi-res AVL run): CDff moves +0.2%,
  CLff +0.04% — Trefftz value converged; the near-field totals NaN at the
  higher resolution (it is the fragile quantity). Stated in Appendix D.
- The +5% lift offset stands: camber-line VLM misses section-thickness lift.

### Blockage (far-field size)
- O-grid far field: cylinder R = 100 root chords = 91.7 m = 2.7 spans.
  Unstructured octree: ±2048 m (60 spans). Families agree on CL to 0.7%.
- Classical wall-interference estimate: Δα = δ(S/C)C_L, C≈4×10^4 m²,
  δ≈0.12 → ΔC_L/C_L ≈ 0.06%. The 0.7% family gap is discretization, not
  confinement. In the section text now.

### Annotated-pdf round (Sec. II.E restructure, commit 53d0a40)
- All 13 annotations addressed: strikeouts deleted; calibration contact is
  its own subsection ordered high-Re-untouched → inner-layer linearity →
  c_ν,ai multiplies zero curvature (QED); τ folded into the blend paragraph;
  coarse-grid table dropped; assembled model → Appendix E (bare reference).
- Your round-off claim VERIFIED: the linear profile is an exact discrete
  solution for both models under any consistent scheme; two unrelated
  solvers (2nd-order FD at first-cell y+ 0.25–8; 4th-order collocation)
  agree to 1e-9–1e-12.
- "e^N growth untouched" now scoped: untouched *to* the χ=1 crossing; past
  it the retained SA terms pace the growth — significant at low Re
  (→ Sec. epphandover).

### fv1-bypass status
- Baseline battery (canon fSlow schedule): 1e5 bubble closes, CL 0.907, but
  violent limit cycle (std 0.12); 2e5 appeared non-inert (std 0.016).
- fSlow-schedule battery at 1e5: wide → std 0.0066; mild (0.1) → 0.0134;
  OFF (fSlow=1) → std 0.0030, CL 0.927 / CD 0.0231.
  ⇒ The limit cycle is a PSEUDO-TIME ARTIFACT of the canon slowdown
  schedule. True bypass state at 1e5: CL≈0.93 (exp 0.873, +6%), CD≈0.023
  (exp 0.0237 — essentially on the measurement). Burst branch was 0.77/0.044.
- Running now (GPU 6): 2e5 pair (bypass on/off, fSlow=1) to re-test guard
  inertness without the schedule artifact.

### Review loop
- Pass 13 (all fixed + gitignore root cause), pass 14 (provenance JSONs
  tracked), pass 15 (P1-1/P1-2 CLOSED; Appendix-D predecessor clause fixed,
  Daedalus repro chain force-added, jargon removed) — all addressed and
  committed; responses in agent-paper-review/.

### 3D validation cases beyond wings (your question; search agent running)
- Scope filter: SA-AI covers TS + separation bubbles, NOT crossflow /
  attachment-line / bypass ⇒ bodies of revolution at zero/low incidence are
  the right family (as you suggested: fuselages/pods).
- My ranking: (1) 6:1 prolate spheroid, DLR Kreplin hot-film data — α=0 is
  axisymmetric TS-dominated, and an α-sweep MEASURES the model's crossflow
  scope boundary (honest, quotable negative); meshing trivial with our
  machinery. (2) Low-drag laminar bodies of revolution (Dolphin,
  Hansen–Hoyt, Lutz–Wagner) — the UUV/airship community quantifies laminar
  extent for a living; 3D analog of the NLF validation. (3) NLF nacelle
  flight tests (NASA/GE '90s) — industrially compelling, less-controlled data.
- Axisymmetric α=0 runs as a thin periodic wedge ≈ 2D cost.

## 2026-07-24 ~20:00 — 3D non-wing validation cases: ranked survey (search agent, sources verified)

1. **DLR 6:1 prolate spheroid, α=0 (Meier & Kreplin, Göttingen 1977–85)** — the
   headline candidate. Hot-film wall-shear transition fronts + Cp; L=2.4 m,
   Re_L 1.6–9.6e6. α=0 is axisymmetric, purely TS — cleanly in scope. Bonus:
   the α=0 transition was measured in THREE tunnels at matched Re specifically
   to quantify freestream-turbulence sensitivity — maps directly onto our Mack
   χ∞(Tu) closure. Crossflow takes over by α=10° (Stock 2006), so an α-sweep
   measures our scope boundary as an honest negative. Caveats: primary DFVLR
   data reports not openly downloadable (digitize from secondary
   transition-model papers or request from DLR); ERCOFTAC's entry covers only
   the FIXED-transition α=10/30° cases, not natural transition at α=0.
2. **NASA Langley MSBS laminar-flow body of revolution (Dress 1988/89,
   Dodbele NLF design)** — the fully-open case: NTRS PDFs downloadable today
   (19880016048, 19890009861). Magnetic suspension = support-interference-free
   drag, free vs fixed transition, liquid-crystal/oil-flow laminar extent.
   Axisymmetric, zero incidence, NLF-designed → in scope. Weakness: integral
   drag rather than precise front coordinates; less famous.
3. **NASA/GE NLF nacelle flight experiment (1986–92)** — flight, full-scale,
   laminar to 50% of nacelle length; open NTRS status report (19900003234).
   Risk: GE1/GE2 contour tables not confirmed open; mild pylon 3D effects.
   The applications/flight case if geometry checks out.
4. (LSB-specific complement) **Hansen & Hoyt 1984** (ASME JFE 106:202) —
   ocean-going body of revolution where the favorable forebody postpones
   transition all the way to laminar separation. Squarely our physics;
   paywalled, disturbance environment hard to map to Tu; geometry unconfirmed.
   Ruled out as validation: Carmichael Dolphin (no transition measurement),
   Lutz–Wagner / Parsons–Goodson (design studies, not experiments), sailplane
   fuselages (plot-level data, CAD-complex shapes — motivation citations only).

Suggested program if we go: spheroid α=0 across Re + the three-tunnel Tu axis
(thin periodic wedge ≈ 2D cost), MSBS body free-vs-fixed drag, spheroid α=10°
as the labeled out-of-scope crossflow contrast.

## 2026-07-24 ~20:20 — fv1-bypass 2e5 inertness verdict (fSlow=1 pair, GPU 6)
- bypass ON : CL 0.9398 / CD 0.01340 (cycle std 0.0079)
- bypass OFF: CL 0.9448 / CD 0.01368 (cycle std 0.0066)
⇒ The earlier "not inert" cycle at 2e5 WAS the slowdown-schedule artifact
  (both runs flutter equally at raw pseudo-time). Remaining bypass footprint
  at 2e5: ΔCL −0.5%, ΔCD −2.8 counts — small, comparable to mesh-family
  spread, and physically expected (2e5 also has a bubble where the bypass
  legitimately acts). Not exactly inert, but no longer disqualifying.
  Guard-tightening (χ>1 ramp, q threshold) can shrink it further if needed.

## 2026-07-24 ~20:45 — AVL CDind vs CDff: they DO match (in double precision)
Rebuilt AVL 3.36 from source (gfortran, -fdefault-real-8, raised limits) at
tools/avl/Avl/bin/avl and ran a lattice ladder on the Daedalus wing, α=4°:
  12x40: CDind 0.009245 | CDff 0.009244   (e = 0.869)
  16x56: CDind 0.009254 | CDff 0.009258
  24x80: CDind 0.009256 | CDff 0.009267
⇒ In double precision the near-field and Trefftz totals agree to FOUR DIGITS
  at every lattice, both grid-converged to ~0.1%. The old cache's CDind
  0.0121 (same 12x40 lattice!) and the 24x80 NaN were SINGLE-PRECISION
  ROUNDOFF — near-field induced drag is a small difference of large terms
  (LE-suction cancellation), hopeless in float32 at AR 38. Not a lattice-
  resolution problem, and the corrected Trefftz reference is rock solid
  (0.009244 identical across precision). Appendix D wording updated to this
  sharper story. (32x120 case abandoned: DP factorization > 10 min, not
  needed.)

## 2026-07-24 ~20:45 — review pass 16 addressed
All pass-15 fixes + the AVL correction + II.E restructure verified by the
reviewer end-to-end (they reran the round-off scripts and read the mesh
bounding boxes; blockage arithmetic confirmed against the campaign's own
lift slope). Fixed this round: predecessor-deficit claim now per-α (12–17
counts below at 4°, ~0 at 6°); signed-algebra paragraph got its antecedent
and moved after the implementation note; "within 3.5 counts (≈1.5%)";
Re_Ω/P_AI wordings tightened; round-off claim now "below 1e-10" (what the
scripts print); repro README + verify docstring updated; tracked
avl_compare.py de-bugged (CDff); NaN stated plainly.

## 2026-07-24 ~21:05 — DLR spheroid: α coverage and TS-vs-crossflow sectors (verified)
Answers to the two questions:
- BETWEEN 0 and 10: YES. Natural-transition hot-film front maps exist at
  α=5° (Re 1.5e6 AND 6.5e6), plus 10° (both Re), 15/20/24° (6.5e6), 30°
  (1.5–8.5e6). Tunnel Tu=0.1–0.2%. Primary: DFVLR IB 22-84 A 33 (Kreplin,
  Vollmers, Meier 1985); all fronts REPRODUCED in Krimmelbein's 2021 DLR
  dissertation (OPEN PDF, elib.dlr.de) — digitization source solved.
- AT α=10°: NOT all crossflow — it is sectored (φ from windward):
  φ≲30° pure TS; 30–60° TS/CF interaction; 60–140/160° CF-dominated;
  leeward tangled in 3D separation (Krimmelbein & Krumbein 2010; Boiko
  2021; Stock 2006).
- THE KEY FACT for us: at Re=1.5e6 the transition is PURE TS at ALL
  incidences (per Krimmelbein) — a full α-ladder (5°→30°) of genuinely
  3D, skewed-boundary-layer transition entirely within SA-AI's scope.
  At α=5°, Re=6.5e6 the CF N-factors are sub-critical (<3.5) — front still
  TS-driven, CF only shapes its curvature.
- Bonus: α=5/10/15° at Re=6.5e6 is AIAA Transition Modeling Workshop
  Case 3 (M=0.13, Tu=0.15%) — community-standard comparison, open case
  definition.
- Profile data at incidence exist (hot-wire surveys at α=10°/6.5e6 with
  natural transition, DFVLR IB 222-84/A10; wall-shear VECTOR maps give
  laminar wall skew everywhere).
⇒ Proposed campaign (addresses the "downgrade" concern): the Re=1.5e6
  incidence ladder — nonzero AoA, twisted 3D profiles, new territory for
  the Ŝg/Re_Ω indicators, yet in-scope so misses are genuine model
  findings; then α=5/10° at 6.5e6 (workshop case) where the measured front
  kink at φ≈30° localizes the model's crossflow boundary on one plot.

## 2026-07-24 ~21:20 — spheroid campaign cost estimate (matched L0/L1/L2 quality)
Meshes (meridional x circumferential-half x normal): L0 ~0.4M, L1 ~2-3M,
L2 ~12-15M nodes (vs Daedalus 0.7/5.1/36.7M — compact body, no 37:1 span).
Wall spacing at Re_L=6.5e6 tightens to h0/L ~ 1-2e-6 (y+<0.5), +~20% normal
nodes; 1.5e6 ladder milder (5e-6). GPU cost from measured Daedalus
throughput (L2 24 GPU-h): spheroid L2 ~8-12 GPU-h/case, L1 1-2. Program =
ladder at anchor (a10/1.5e6) + L2 at a5/a30 (1.5e6) + a5/a10 (6.5e6
workshop) ≈ 60-80 GPU-h ≈ one Daedalus-L2-phase; a day on the 6-GPU pool.
Schedule items: mesh generator adaptation (~a day; two pole closeouts,
symmetry plane through incidence plane), digitizing Kreplin fronts from
Krimmelbein's open dissertation, a30 leeward may be unsteady (ladder to
10-15 deg safely steady).

## 2026-07-24 ~21:50 — AVL CL question resolved: thin-airfoil truncation, with numbers
How the sequence runs (confirmed): AVL solves the CAMBER-surface lattice at
α → CLtot + Trefftz CDff + cl(y); XFOIL/FlexFoil then runs sections at
MATCHED local cl for profile cd. So the reference CL is purely inviscid
thin-camber — and your intuition is right that proper inviscid should sit
HIGH. Station ledger at η=0.31, α=4° (α_eff≈3.5°), DAE-11 at Re 5e5:
  AVL camber-only strip      cl = 1.011
  thick-section INVISCID 2D  cl = 1.119   (+10.7%: thickness; slope 6.92/rad)
  viscous 2D (N=13.6)        cl = 1.053   (−5.9%: decambering)
  RANS strip (canon L2)      cl = 1.066   (+1.2% vs viscous 2D)
⇒ The +5% wing offset = thickness (+11%) minus viscosity (−6%). AVL's
  camber-only CL is the LOW outlier by construction; RANS agrees with
  viscous 2D to ~1%. Lift-curve check: AVL slope 5.73/rad, α0L −5.77°;
  RANS slope 5.84/rad, α0L −6.1° — offset, not slope anomaly. Paper text
  updated with the four-number decomposition.
  (Tools note: mfoil viscous diverges on DAE-11 @5e5; FlexFoil
  faithful-viscous plateaus unconverged but repeatable — treated as ±1%.)

## 2026-07-24 ~22:15 — AVL reference moved to MATCHED-LIFT trim (your suggestion)
New chain (daedalus/avl_fixed_cl_reference.py): AVL trimmed with 'A C <CL>'
to the canon RANS CL (finest completed level per α), CDff at trim + XFOIL
profile drag at the trimmed strip loading:
   α=4: target CL 1.0282 → trim α 4.53°, CD_ref 0.02125 (RANS 0.01997: 13 cts / 6.0% low)
   α=5: target CL 1.1289 → trim 5.55°,  CD_ref 0.02315 (RANS 0.02243:  7 cts / 3.1% low)
   α=6: target CL 1.2317 (L1) → 6.59°,  CD_ref 0.02539 (RANS L1 0.02479: 6 cts low)
The matched-α "lands on the reference" was flattered: the lattice carries 5%
less lift → ~10% less induced drag. Matched-lift is the honest comparison:
RANS 3–6% below strip theory, deficit shrinking with incidence; the
reference's profile term barely moves with trim (0.0110→0.0108), so the
growth is all induced. Predecessor sat 15–26 counts below this reference.
Paper (ledger paragraph, fig caption, tab:daetotals, Appendix D) updated;
the a6 reference refreshes when the L2 run lands (script picks up the
finest completed case automatically).

## 2026-07-24 ~22:40 — spheroid mesher built & validated (agent)
spheroid/ogrid_spheroid.py: 6:1 prolate spheroid half-model O-grid reusing
the Daedalus machinery (growth solver, pole treatment as prisms instead of
the wing's slit-merge, Gmsh->CGNS via the flexfoil/rans pipeline).
Symmetry plane y=0 (incidence in x-z), far field = sphere R=30L.
L0 generated: 372,771 nodes, min Jacobian positive, wall/farfield/symmetry
areas within 0.06% of analytic, volume closure 0.12%. L1: 2,204,111 nodes,
areas within 0.02%. L2 projects 12.63M (not generated yet). Wall-spacing
ladder: --h0 5e-6 halving per level (Daedalus convention), --h0-fixed and
1.5e-6 flag for the Re=6.5e6 cases.
Commands: python3 spheroid/ogrid_spheroid.py L0|L1|L2 [--h0 ...] [--summary]
Next: L2 generation + a Flow360 case template (BCs, Mach 0.13/0.1, chi_inf
from Tu=0.15% via Mack), then the alpha=10/Re=1.5e6 anchor ladder.

## 2026-07-24 ~22:40 — plotting changes in flight
(1) Four Cf-distribution figures (nlf/eppler low/high-alpha) widened
    5->5.76 in/column, tex 0.86->0.99\textwidth — same rendered height,
    near-full page width. Regenerating.
(2) max(Shat*g) row now log scale, ylim 3e-3..1.3 (same range as the
    onset-threshold figure); favorable/suppressed values drop below axis.
(3) e^N envelopes now truncated at their transition point (the
    post-transition reset to N=0 was an artifact of the reference dump,
    not an envelope).

## 2026-07-24 ~23:00 — who is closer to truth inside the LSB: e^N or SA-AI?
Answer: e^N, on both halves — and SA-AI's reattachment agreement is a
compensated error.
- SA-AI over-amplifies inside the bubble via (1) LEDGER SPEED: nuHat peak
  advects at ~half the physical packet (critical-layer) speed → ~2x e-folds
  per unit length (intrinsic to local transport; Appendix-D frozen-profile
  measurement), and (2) RATE CEILING: Shat*g saturates → a_max = detached-
  tanh KH eigenvalue, while a shallow-bubble layer is still wall-damped
  (resolved band 0.9–1.9x Drela–Giles). The in-bubble onset arm fixes only
  the just-separated zone.
- e^N + instant breakdown is closer to DNS truth (linear growth matches
  LST; spot/roll breakdown within a few delta).
- The agreement: too-fast climb + too-slow commit ≈ true climb + instant
  commit. Compensation is Re-dependent: climb excess is a fixed FACTOR,
  handover lag a fixed E-FOLD COUNT whose length grows as Re drops → the
  cancellation breaks at 1e5/6e4 = the bursting-boundary displacement of
  Sec. VI. fv1-bypass corroborates: fixing only the commit side closed the
  1e5 bubble but CL 0.93 vs exp 0.873 (transition effectively early once
  the climb excess is no longer absorbed).
- Implication: truthful fix moves BOTH dials (rate down toward LST in the
  bubble + faster commitment); either alone breaks the pairing. Sharper
  framing of the "model the transition zone" future-work item.

## 2026-07-24 ~21:15 — LSB over-amplification mechanism FOUND: closed-streamline re-seeding
Your two objections (Re-trend; calibration already includes advection) both
hold and they corner the mechanism. Measured on the Eppler strL2 a5 sweep,
median chi in the reversed "dead air" (first third of bubble) vs the max
incoming-envelope chi just upstream of separation:
  60k: 16x ABOVE | 100k: 29x ABOVE | 300k: 0.03x (quiet) | 460k: 0.01x
The flip sits exactly at the bursting boundary. Mechanism: in the elliptic
solution the bubble is a closed cell — nuHat diffuses across the layer into
the reversed stream, convects BACK upstream, and re-enters the amplifying
band elevated (spiral path, recirculating dwell). Absent from the parabolic
frozen-profile calibration BY CONSTRUCTION (one-pass march). Cross-layer
diffusion vs convection strengthens as Re drops -> your observed trend.
- Coder question: he never solved local advection speed — the amplifying
  band is placed by profile geometry (Shat*g>0), production peaks low in
  the layer (nuHat peak 2.4θ @ 0.22Ue vs packet 3.5θ @ 0.45Ue); the
  Blasius N=1 anchor absorbs the family-stable speed ratio for attached
  profiles. Absorbed bias, not a solution; useless against re-seeding
  (not a profile property).
- Nuance: real bubbles DO feed back (absolute instability at ~15-20%
  reversal, Rist/Maucher) — the model's reservoir is an unconditional,
  diffusion-driven caricature that switches on too early in Re.
- Rig note: single-sweep FLARE marching can't reproduce this; multi-sweep
  windward (Davis-Carter/ALESEP style) can — rig #2 design should keep
  the multi-sweep reversed-layer convection for exactly this reason.

## 2026-07-24 ~21:45 — re-seeding mechanism written into the paper; rig #2 cancelled
- Dead-air production check (both Re): onset-gated to ~zero (median rate
  <1e-9 a_max) → the low-Re reservoir is ADVECTION-DIFFUSION fed, exactly
  the contour reading (cold front = bottom-led chi at 60k/100k; warm front
  = top-led at 300k/460k, App C sheets).
- New passage in Sec. epphandover: the two opposite-signed low-Re defects
  (closed-cell re-seeding vs starved handover + SA anemia chi_eq~0.08
  Re_tau), the measured 16x/29x vs 0.03x/0.01x flip at the bursting
  boundary, the cold/warm-front contour signature, why no profile-based
  calibration can see it (one-way march), and the honest-caution paragraph:
  at 2-4.6e5 the model is genuine to the physics; at Re<~1e5 correct
  bubbles/gross aero are the sum of two canceling errors, not resolved
  transition physics. Conclusion updated to match.
- repro/cfd/measure_lsb_reseeding.py committed (surplus + production
  tables). Rig #2 cancelled (memory updated): a parabolic march is
  structurally blind to the mechanism that matters.

## 2026-07-24 ~22:05 — the fortuitous cancellation is now COMPLETE (both halves)
Why re-seeding is severe at low Re and absent at high Re — measured:
the REAR third of the reversed region is loaded at EVERY Re (19x the
incoming envelope at 300k, 2.9x at 460k) — pickup is universal. What
discriminates is CONDUCTION TO THE FRONT, with three controls, all
favoring low Re:
1. The hot-over-reversed OVERLAP is created by the starved handover
   itself: at high Re closure follows transition within the short
   handover, ending the conveyor as soon as the layer gets hot; the
   un-closed low-Re bubble runs its conveyor under an amplifying layer
   for half a chord. The two defects are ONE CAUSAL CHAIN (slow handover
   -> no closure -> long fetch -> reservoir -> re-seeded front) — which
   is why the flip is sharp at the bursting boundary, not gradual.
2. The only sink is the destruction floor ∝ (nuHat/d)²: thin high-Re
   cell returns its leak at small d (floor eats it in transit); thick
   low-Re cell is ~lossless.
3. Loading self-accelerates past chi~1 (nuHat self-diffusion takes over
   from the reduced molecular trickle) — crossed at 60k/100k (maxima
   5–18), never approached at high Re (chi~1e-3).
Paper (Sec. epphandover) and measure_lsb_reseeding.py updated with the
rear-third evidence and the three controls.

============================================================
## 2026-07-24 ~22:15 — DETAILED RESTATEMENT of tonight's LSB physics thread
(The earlier entries above compressed these into digests; full prose below,
as delivered in chat, actions omitted.)

### 1. Who is closer to truth inside the LSB: e^N or SA-AI?
e^N is closer on both halves, and SA-AI's reattachment agreement is a
compensated error whose cancellation we have already watched fail at low
Re.

Why SA-AI over-amplifies inside the bubble — the original two-mechanism
answer (later refined, see §3 below):
(1) Ledger speed: the model banks e-folds along particle paths — dN/dx =
a·ω/u_conv, where u_conv is the speed of the fluid carrying ν̃. The
transported ν̃ peak sits low in the separated shear layer, in slow (even
reversed) fluid at roughly half the physical packet speed (the real
instability travels at the critical-layer phase speed, ~0.4–0.5 U_e).
Half the convection speed = twice the e-folds per unit length. This is
intrinsic to any local one-equation transport: we cannot advect N at the
phase speed without an auxiliary field.
(2) Rate ceiling: once the layer lifts, Ŝg saturates and the rate rides
at a_max = 0.19 — the eigenvalue of a fully detached tanh layer. A
shallow-bubble shear layer a few θ off the wall is still partially
wall-damped and grows slower. The final kernel's in-bubble onset arm
fixes the just-separated zone (the predecessor's 1.3–1.8× excess), but
past onset the resolved band still runs 0.9–1.9× the Drela–Giles
envelope — biased high.

Why reattachment agrees anyway: the model's other economy runs in the
opposite direction. e^N declares transition at N_crit and the physical
breakdown is genuinely fast (spot/roll saturation within a few δ —
Alam–Sandham-type DNS supports the reattaches-in-a-few-BL-thicknesses
picture), while SA-AI must climb χ = 1 → c_v1 → ~30 and then let eddy
viscosity diffuse to closing strength. So: too-fast climb + too-slow
commit ≈ true climb + instant commit. The calibration effectively
accepted this cancellation — it is why the anchors work.

The tell that it is compensation, not correctness: the two errors scale
differently with Reynolds number. The climb excess is a fixed FACTOR
(~1.5–2× in e-folds per length), but the handover lag is a fixed E-FOLD
COUNT whose physical length grows as Re drops. At 4.6e5 both are small
against the bubble; at 1e5–6e4 the handover lag outgrows the climb
surplus, the cancellation breaks, and the bubble starves — that IS the
bursting-boundary displacement of Sec. VI. The fv1-bypass experiment is
the complementary demonstration: fixing only the commit side (instant
eddy viscosity) while keeping the over-amplification closed the 1e5
bubble but landed CL ≈ 0.93 vs the measured 0.873 — transition now
effectively early, because the climb excess is no longer being absorbed
by the handover lag.

Implication: a truthful model has to move both dials together — rate
down toward the LST envelope inside the bubble AND commitment faster
(spot-closure on σ_t, or the fv1-bypass with proper guards). Either
alone breaks the agreement the current pairing buys.

### 2. Your Re-trend objection and the calibration objection — how they
### cornered the real mechanism
Your observation (at high Re SA-AI does NOT over-amplify vs e^N; the
discrepancy grows as Re falls) discriminates between the two mechanisms
above and acquits the rate ceiling: if a_max-on-saturated-Ŝg were the
culprit, the excess factor would be Re-independent and high-Re bubbles
would over-amplify just as visibly.

Your second objection then killed the pure advection-speed story too:
the calibration instrument is the same transport equation marched with
its own advection on frozen profiles, reversed branch included, and it
shows only 0.9–1.9× — so anything the march captures is already priced
into the anchors. The mechanism had to be something a parabolic,
one-pass march cannot represent.

That corner leaves closed-streamline re-seeding: in the elliptic
solution the bubble is a closed cell — ν̃ produced in the shear layer
diffuses across into the reversed stream, is carried BACK upstream to
the separation region, and re-enters the amplifying band already
elevated. The envelope at a station is then not ∫ a·ω/u dx from the
attachment of the layer, but amplification along a spiral path with
recirculating dwell.

Measured on the Eppler strL2 α=5 sweep (median χ in the reversed dead
air, first third of bubble, vs max incoming-envelope χ just upstream of
separation): 60k: 16× ABOVE; 100k: 29× ABOVE; 300k: 0.03× (quiet);
460k: 0.01× (quiet). The flip sits exactly at the bursting boundary.
At 300k/460k the recirculation is QUIETER than the arriving envelope —
one-pass behavior, exactly what the frozen-profile calibration assumes.

Production check (sphere indicators + onset gate evaluated on the
slices): the kernel's effective rate in the dead air is onset-gated to
essentially zero at every swept Re (median < 1e-9 a_max; Ŝg>0 at about
half the dead-air nodes but Re_Ω sits under the threshold). The
reservoir is purely advection–diffusion fed — your contour reading
("pretty sure it's not amplification driven") verified.

Physical nuance: real bubbles do develop feedback — absolute/global
instability when peak reversal exceeds roughly 15–20% of U_e
(Rist/Maucher) — so the model's recirculating reservoir is a caricature
of something real. But the model's version is unconditional
diffusion-driven leakage, not a criterion-gated instability, and it
switches on far too early on the Re axis.

### 3. How did Coder solve the local advection speed? (He didn't.)
He never placed the amplifying region at the wavepacket. The kernel puts
production wherever Ŝg > 0 — a profile-geometry statement (the Rayleigh
coordinate between neutral-circle crossings) — and production ∝ ω peaks
low in the layer, well below the critical layer (the Appendix-D ledger
measured the ν̃ peak at 2.4θ moving at 0.22 U_e vs the physical packet
at 3.5θ, 0.45 U_e). What made this survivable: the calibration is the
same transport equation marched on frozen profiles, and for the attached
Falkner–Skan family the band's convection speed is a roughly
family-stable fraction of U_e — so the single Blasius N=1 anchor (the k
of the onset threshold) silently absorbs the speed ratio along with the
laminar drain. It is an absorbed bias, not a solved problem: it holds
while the speed ratio is universal (attached family, tracked within
±12%), degrades on the reversed branch (0.9–1.9×, checked but never
re-anchored), and offers no defense at all against re-seeding — which
is not a profile property, so no local rate calibration can absorb it.

### 4. How did Coder get his model to "match LSB"? (He didn't either.)
He matched the two things that are cheap to match — transition FRONTS
(his χ=1 contour sat the same ~0.1c ahead of the strips as ours) and
integrated FORCES (few counts, 0.02–0.03c of front) — while the bubble
LENGTHS were wrong the whole time (3–4× too short on the wing:
0.04–0.06c vs the strips' 0.2c, the compressed-bubble systematic the old
section carried as an unexplained disclosure). Front + forces are
exactly the observables that tolerate large internal-structure errors,
because they are set by the SUM of climb and commit, not by either one.

His sum was the same cancellation in stronger doses: his ungated rate
ran the free-shear ceiling from the separation point on — 1.3–1.8× the
correlation immediately beyond separation — an even faster climb than
ours, against the same slow σ_t handover. Faster climb + same lag = the
handover costs less DISTANCE → bubbles too short, fronts right-ish,
forces fine at moderate Re. The final kernel's in-bubble arm slowed the
climb toward the correlation, which is precisely what lengthened the
wing bubble onto the strips — and simultaneously exposed the low-Re
starvation, because with a truthful climb there is nothing left to hide
the lag behind.

His ν̃ recirculates identically (re-seeding is a property of the
transport topology plus the elliptic solver, not of the kernel) and SA's
anemia (χ_eq ≈ 0.08 Re_τ) is kernel-independent. Neither ever showed in
his record because his validation set lived where both are off: wing
sections at Re ≈ 5e5 and Eppler at 2e5 — and the re-seeding flip sits
between 1e5 and 3e5. He never ran 1e5 or 6e4. Did anyone investigate
whether the agreement was fortuitous? No — the audit trail is: the
compressed bubble was flagged as a systematic in the predecessor-era
draft; the frozen-profile ledger (this thread) decomposed the excess
into 1.45× temporal rate × 2.05× ledger speed; the sustainment floor and
re-seeding measurements are from this week. The cancellation was never
identified as such before.

### 5. The completed mechanism: why re-seeding is Re-selective
The missing half (why high-Re bubbles see no recirculation effect) has a
clean answer, and the punchline is that re-seeding is not an independent
defect — it is a corollary of the handover defect.

Decisive measurement: the REAR third of the reversed region is loaded at
EVERY Reynolds number — 19.5× the incoming envelope at 300k, 2.9× at
460k — so pickup is universal. What discriminates the regimes is whether
the load CONDUCTS TO THE FRONT. Three controls, all favoring low Re:

(1) The hot-over-reversed overlap that feeds the conveyor is created by
the starved handover itself. Re-seeding needs a reversed conveyor below
and a hot (high-χ) layer above it simultaneously. At high Re, transition
sits near the bubble's end and closure follows within the (short)
handover length — the reversed flow ends almost as soon as the layer
gets hot; the overlap is only the handover length itself, a sliver. At
low Re the starved handover prevents closure, so the reversed conveyor
persists under a layer that keeps amplifying — half a chord of
continuously-loading fetch. The "two canceling errors" are therefore ONE
CAUSAL CHAIN: slow handover → no closure → long hot-over-reversed
overlap → reservoir loads → front re-seeds. This is also why the flip in
the measurement (and in the contour character: cold front, bottom-led χ
at 60k/100k; warm front, top-led at 300k+) is sharp at the bursting
boundary rather than a smooth trend — the overlap inherits the fold from
the closure failure.

(2) The reservoir's only sink is the destruction floor, which goes as
(ν̃/d)². The high-Re bubble is thin — the reversed region lives at small
wall distance, and any ν̃ that leaks down is eaten during the return
transit. The low-Re bubble is thick: d is an order larger, the 1/d²
floor two orders weaker, and the conveyor runs essentially lossless.

(3) The pickup has a nonlinear switch. Cross-layer loading per unit
fetch scales as D·t/θ² with D ∝ ν(c_ν,ai/6 + χ): while the reservoir
sits at χ ≪ 1 its diffusivity is the reduced molecular trickle, but once
any of it crosses χ ~ 1 the ν̃ SELF-diffusion switches on and the
reservoir spreads itself — crossed by the 60k/100k reservoirs (medians
0.03–0.08, maxima 5–18), never approached at 300k/460k (χ ~ 1e-3).

All of this is now in Sec. epphandover (with the honest-caution
paragraph: genuine physics at 2–4.6e5; below ~1e5 correct bubbles and
gross aerodynamics are the sum of canceling errors, not resolved
transition physics), in the Conclusion, and reproducible via
repro/cfd/measure_lsb_reseeding.py.
============================================================

## 2026-07-24 ~22:30 — mechanism trimmed to (1) in the paper, per your call
You judged that of the three controls only (1) — the hot-over-reversed
overlap created by the starved handover — is strong enough for the paper;
(2) and (3) can be argued the other way or are neutral, and (3) is part
of (1). On reflection I agree, and (2) has a specific problem beyond
arguability: it is in tension with the paper's own earlier claim that the
destruction floor is inert at the disturbance's scale (f_w ≈ 0.7 r with
r ≪ 1 in the laminar range — the same smallness that makes the floor
harmless under the amplifying band makes it a weak sink in the reversed
cell), and its transit-time counterargument (the thin high-Re cell's
return leg is also SHORTER, so exposure to the floor shrinks along with
d) leaves the net effect ambiguous. (3) only fires because the reservoir
is already loaded — the χ~1 self-diffusion crossing happens BECAUSE the
fetch is long — so it is an amplifier inside mechanism (1), not an
independent discriminator.

The paper passage now carries only the causal-chain mechanism: pickup is
universal (rear-third 19×/2.9× at 300k/460k), conduction to the front is
controlled by the overlap length, the overlap is created by the starved
handover (slow handover → no closure → long hot-over-reversed fetch →
loaded reservoir → re-seeded front), the self-diffusion crossing folded
in as a parenthetical amplifier, and the sharp flip at the bursting
boundary explained as the overlap inheriting the fold from the closure
failure. The destruction-floor argument is dropped from paper and
measurement-script docstring both.

## 2026-07-24 ~22:45 — Daedalus anatomy recast as the Sec-VI compensation
Per your instruction, the Daedalus chi-map paragraph no longer dresses the
mid-bubble front as "the expected convention offset" (the 11.7-vs-13.6
e-fold arithmetic is dropped). It now says plainly what the maps show:
the bubble's EXTENT matches the strips station by station, but the
model's transition front (chi=1) sits near mid-bubble, ~0.09c ahead of
the strips' N_crit contour, which sits just ahead of reattachment — the
compensation of Sec. epphandover seen in three dimensions. The model
books its e-folds early and spends the remainder of the bubble bringing
eddy viscosity to closing strength; the e^N reference climbs to the
bubble's rear and commits at once. At this chord Reynolds number (5e5)
both halves of the compensation are small enough that the sum —
separation to reattachment — lands on the reference. (The predecessor
paragraph that follows is untouched: same fronts, bubble 3-4x short =
same compensation at a stronger climb.)

## 2026-07-24 ~22:50 — Sec. II.F retitled
"Contact with Spalart's calibration" -> "Preservation of the
Spalart–Allmaras calibration" — matches what the subsection actually
proves (each calibration stage preserved exactly) and reads as the
respectful claim it is. No residual "contact" phrasing in the body.

## 2026-07-24 ~22:55 — back-reaction paragraph moved II.F -> II.E
The destruction-floor back-reaction quantification (ratio < 1e-3 to the
crossing at the N_crit=9 seed; instrument re-march with the term: anchors
<1%, favorable mean -2.6% mostly past the crossing; coupled bounds:
flat-plate onset +0.8% at lowest Tu, airfoil fronts/forces under line
widths, peak |dCf| 3e-5) now sits in II.E directly after the paragraph
that introduces the tie and claims the floor's inertness — where the
claim is made, not two subsections later. Opening adapted ("The floor's
inertness deserves its numbers, since it is the tie's one back-reaction
on the laminar branch") to avoid repeating the claim sentence; the
blend paragraph's pointer changed from "(quantified in Sec. II.F)" to
"(quantified below)". II.F now runs intro -> high-Re stages -> inner
layer/tie derivation -> linearity/discrete exactness -> Appendix E
pointer, purely about preservation.

## 2026-07-24 ~23:20 — review pass 18 addressed (commit cbec5f7)
Pass 18 verified the entire re-seeding/cancellation analysis by rerunning
the measurement script (every number reproduced digit-for-digit), added
its own 2e5 measurement (front 0.09x, rear 58x, rate 0 — flip lands
square on the bursting boundary), rendered the App C sheets to confirm
the cold-front/warm-front signature is visible as claimed, and confirmed
the plotting round (canvas rescale is height-preserving; e^N truncation
works). Its four majors were all real:
1. My cancellation clause said the re-seeded front lands "roughly where
   the physical envelope would" — falsified by our own data: the chi=1
   front is 0.16c AHEAD of the envelope's equivalent station (0.41 vs
   0.57c at 1e5; 0.44 vs 0.60c at 6e4, vs mfoil at N=ln(1/chi_inf)).
   And the cancellation logic NEEDS the front early (a head start the
   over-long handover spends; if the front were right the two errors
   would add on the reattachment side). Rewritten to the measured
   statement.
2. The "below 1e-9 a_max" production bound was exceeded ~40x by the 1e5
   median (7.3e-9 absolute = 3.9e-8 a_max). Bound now 1e-7 a_max +
   95th-percentile framing; conclusion (advection-diffusion fed)
   unaffected.
3. "Can still predict the bubble" contradicted "never closes" two pages
   earlier. Scoped to what compensation delivers: separation + front +
   lift within 2–12% family-dependently.
4. The Daedalus front offset had been re-badged as "the compensation of
   Sec VI seen in 3D" — but at wing Re (5e5) the re-seeding half is
   measured OFF (0.01x). Restored the seed-convention arithmetic
   (chi=1 = 11.7 e-folds vs N_crit 13.6, ~2 e-folds per 0.09c) as the
   reason the front is early there, keeping only the commit half
   (handover spending the rest of the bubble) as shared with Sec VI.
Minors: 200k added to the committed script table; monotonicity claim
dropped (29x at 1e5 > 16x at 6e4 — both burst, geometry differs);
chi_eq now 0.09 Re_tau (the committed script's number) with Re_tau
defined; Re_theta 220–330 scoped to 6e4 with "would equilibrate";
median-vs-peak statistic convention stated in text; house-style times;
README rows for both new scripts; 28 side-effect figure regenerations +
lsb-zoom exploration committed. Reviewer also suggested (uncounted) an
abstract amendment to mention the second defect — left for your call.

## 2026-07-24 ~23:55 — annotated.pdf round 2 (Sec. III + implementation), all 10 items
1-6 (seed paragraph): the "one empirical closure" sentence, the "used
unchanged / plays the role of the e^N envelope" clause, the "Adopting
Mack's map wholesale is an approximation..." sentence, and the "every
airfoil computation uses N_crit=9" sentence (already said in the airfoil
sections) all deleted; the turbulence intensity is now introduced in
words before Eq. (18) uses the Tu symbol; the handover smear is now
"chi = 1 -> O(c_v1), several e-folds" (was "two").
7 (implementation note): the discretization paragraph (least-squares
double-gradient odd-even modes, viscous-flux Laplacian, ring average)
is REPLACED by a short vector-calculus statement: X=|u| (wall-relative,
Galilean invariant), Y=|omega|d, Z=(1/2)d^2 (lap u . u_hat) with the
divergence-free identity lap u = -curl omega, sign(u) u'' on a parallel
layer, Re_Omega = d^2|omega|/nu; the analytic-Jacobian sentence kept.
The signed-algebra paragraph's antecedent updated accordingly.
8: the "kernel is therefore exact on the sign-consistent region..."
closing sentence deleted (paragraph now ends at the antisymmetric-layer
zero-amplitude remark).
9 (THE SUBSTANTIVE ONE — Re_theta by integration, "don't do this
anymore"): onset_vs_ags() in regen_flatplate_flow360.py now interpolates
the INTEGRATED edge-normalized Re_theta (the machinery cf_and_retheta
already had, which even documents the freestream-overshoot trap) at the
crossings instead of converting via 0.664 sqrt(Re_x). New honest
numbers, quoted in Sec. III + fig caption + Conclusion:
  chi=1 crossing: +6.1% (Tu 0.04), -3.3% (0.08), -10.5/-13.7/-11.2%
  (0.16/0.30/0.60) => "brackets AGS within 14%" (was ±10%). Note the
  +6% at the lowest Tu now agrees BETTER with the 7% retained-drain
  footprint than the old +10% did.
  c_v1 crossing: integrated Re_theta ≈ 2500 nearly uniform across the
  sweep = 2.2-3.5x AGS — the layer is already MID-TRANSITION there; the
  old laminar-conversion reading (+14-36%) badly understated how far
  the layer had left the laminar branch. The text now says the c_v1
  reading measures transitional growth, not onset placement.
  (My first standalone attempt hit the exact trap the regen script
  documents: integrating f(1-f) over the 50-delta-tall column picks up
  percent-level freestream nonuniformity — delta* survives clipping but
  theta is poisoned, H came out 10. Deleted the standalone; the regen
  extension is the single source of truth, run via ONSET_DIAG=1.)
10: Fig. 5 (flat-plate batch) moved to right after the Sec. III opening
paragraph.

## 2026-07-25 ~00:15 — annotated.pdf round 3: floor back-reaction quantified on the march
Four items, all on the back-reaction paragraph (now in II.E):
- The coupled-computation bounds sentence is REMOVED per instruction and
  replaced by the prescribed march quantification
  (repro/analytic/floor_backreaction_table.py): D_floor/P_AI at the nuHat
  peak of the frozen-profile marches, for beta = +0.10 / Blasius /
  separation limit (-0.1988) and N_crit = 7/9/11:
    chi=0.1 : 3e-6 – 1.2e-5
    chi=1   : (2–8)e-4  — below 1e-3 for EVERY wedge and seed
    chi=c_v1: (0.9–2.3)e-2, and at most 1.4% of the sigma_P-weighted SA
              production it is tied against (largest at the separation
              limit)
- This also answers "why specific to airfoil seed?": it isn't — the
  ratio at fixed chi depends on the seed only through Re_Omega at the
  crossing (lower N_crit -> smaller Re_Omega -> larger ratio), and the
  bound holds across N_crit 7–11 generally; the airfoil-seed phrasing is
  gone.
- "has already handed production over" -> "starts handing production
  over" (at chi=1 the handover begins, not completes).
- The instrument re-march sentence (anchors <1%, favorable mean -2.6%)
  stays — it measures a different thing (station shifts with the term
  included) and now follows the ratio table.

## 2026-07-25 ~00:25 — elegant vector-calculus form of Shat*g in 3D
The kernel only ever needs EVEN pairwise products of the signed
indicators, and each has a coordinate-free realization (n_hat = grad d,
u wall-relative):
  Shat*g = (Y^2 - XY - YZ) / (sqrt(X^2+Y^2) R)
  Y^2 = d^2 (omega . omega)
  XY  = d   u . (omega x n_hat)         [omega x n_hat IS the signed
                                          shear direction; = u' x_hat on
                                          a parallel layer]
  YZ  = (1/2) d^3 (lap u) . (omega x n_hat)
  X^2 = |u|^2 ; Z^2 = (1/4) d^4 ((lap u).u_hat)^2   [norms, even anyway]
Consequences:
1. The mixed-sign-layer caveat dissolves: every term is even under the
   global mirror (u,omega)->(-u,-omega), so the antipodal identification
   is realized EXACTLY everywhere, including between the backflow
   extremum and the velocity zero where the magnitude triple deviates.
   The paper's half-page sign-algebra discussion would collapse to the
   table above.
2. Same ingredients (omega, lap u = -curl omega, grad d), no new
   operators or cost — a re-wiring, not new machinery.
3. NOT just elegance: on skewed 3D layers (shear direction rotating with
   height) the current form pairs |omega| with the flow direction
   blindly, while u.(omega x n_hat) picks the shear component aligned
   with the local velocity. The two coincide on parallel layers (all 2D
   validation + the high-AR wing) and differ exactly on twisted
   profiles — the spheroid-at-incidence territory. The spheroid campaign
   could double as the discriminating test between the two 3D closures.
Caution: bit-identical in 2D, different in 3D => a model revision, not a
refactor. Proposal: implement behind AI_INVARIANT_KERNEL alongside the
current form, verify 2D bit-identity, and run both on the spheroid
alpha-ladder.

## 2026-07-25 ~00:40 — Daedalus ogrid L2 a6 landed; section refreshed (all-L2 structured set)
case_ogrid_L2_saai_a6 finished (rc=0, 22:00): CL 1.2279 / CD 0.02509.
- Matched-lift reference refreshed (target now the L2 lift): trim 6.555°,
  CD_ref 0.02530 → RANS 2.1 counts (0.8%) BELOW. Full L2 drag ladder vs
  strip theory: −6.0% / −3.1% / −0.8% at α=4/5/6 — deficit shrinking with
  incidence, now measured on the finest grids at all three points.
- Bubble at η=0.31, α=6°: RANS 0.458–0.630 vs strips 0.458–0.631 —
  separation exact, reattachment within 0.001c. The "separation to 0.02c,
  reattachment under 0.01c" statement now holds at every incidence.
- Front: 0.522 vs strips' N_crit 0.617 (~0.095c ahead — same convention
  offset as a4/a5).
- Paper updated: nine solutions (pending set = three unstructured L2);
  ledger drag numbers; tab:daetotals a6-L2 + refreshed AVL row; bubble
  sentence extended to 6°; the a6 surface figure re-added (str L2 + e^N
  rows) with the Figs. 4–6 range refs restored; polar figure regenerated
  with the full structured-L2 line.
Campaign remainder: cavity L2 a4 started 22:00 on all 8 GPUs (then a5,
a6) — the unstructured-L2 rows and figure panels will slot in the same
way when they land (generators are availability-driven).

## 2026-07-25 ~01:00 — invariant kernel implemented (2D-verify-first plan)
Per your revised plan (verify both forms agree on 2D — one NLF + one
Eppler — then run only the verified invariant form on the spheroid):
- Solver: __aiRateInvariant added to SAAiTransition.h — the triple is
  built through the shear direction s_hat = (omega x n_out)/|omega x
  n_out|: X = u.s_hat, Y = d|omega x n_out|, Z = (1/2)d^2 (lap u).s_hat,
  then the shared __aiRateFromXYZ tail (Shat, g, P, clip, onset — now
  factored out and used by both forms). Re_Omega keeps full |omega|.
  Where tangential vorticity vanishes, Y->0 and the rate extinguishes
  through Shat->0, as in the standard form.
- Wiring: ai_invariantKernel constant, env AI_INVARIANT_KERNEL,
  constants-echo line, rateOverride parameter threaded through
  __aiSaProduction (Jacobian property unchanged: rate independent of
  nuHat); the debug rawRate output uses the override when active.
- Clean rebuild running (BUILD_CONSISTENCY SOP; headers changed).
- Verification runner staged: paper/repro/cfd/run_invariant_kernel_verify.py
  — NLF a4 Re4M strL2 (attached: expect agreement to solver noise; the
  two forms are algebraically identical on parallel layers) + Eppler a5
  Re2e5 strL2 (bubble: differences confined to the mixed-sign
  recirculation layer, expected within line widths). Cold starts, canon
  env + flag. GPU-gated: all 8 GPUs are on cavity-L2 until ~02:00.

## 2026-07-25 ~01:45 — annotated.pdf round 4 (NLF section), all 13 items
The section is restructured around the transition-location figure:
- OPENING GUTTED per the two big strikeouts: the pins-most-fronts claim,
  the lambda_p definition and all per-alpha front narration, the
  what-the-case-tests paragraph, and the finite-difference pinned/free
  quantification (dx_tr/dN discussion) are deleted. The section now
  opens with one sentence + fig:nlfaft MOVED TO THE FRONT, regenerated
  to show ALL SIX GRIDS (L0-L2 by marker size, both families) — the
  coarse-grid scatter is now visible honestly (e.g. the cavity-L0
  lower-front outliers). The lambda_p nomenclature row is dropped too
  (no remaining use).
- AFT ADDED TO THE POLAR (Fig. 7): the digitized dissertation data
  already contained the AFT polar curves (nominal 10.07 dashed,
  recalibrated 7.18 solid, gray) — overlaid with legend entries.
- EPISTEMIC REWRITE of the AFT comparison ending (your note: can't argue
  "we are better"; may be coincidence/compensation): the text now reads
  "We do not read the untuned agreement as a ranking, because it may be
  partly compensatory: the model's handover is least decisive at low
  Re_theta (Sec. VI), the c_v1-based N_crit convention was anchored on
  flat-plate transition at lower Re_theta than these fronts, and at the
  NLF's higher-Re_theta fronts the same convention is effectively
  sharper — while AFT's recalibrated freestream may equally be read as
  absorbing the tunnel's true disturbance level rather than as a
  per-case tuning. The two closures place their empiricism differently;
  this case cannot separate them." ("fixed once for the whole paper" and
  "without a per-case constant to choose" deleted.)
- Also deleted per strikeouts: the families-agree-in-drag block
  (0.4e-3/1.1e-3 counts + L1/L0 scatter recap), the signed/unsigned Cf
  clause + the Throughout-conventions block + the surfaces-walk detail +
  the across-figs convergence recap (all restated in the captions), the
  "three-dimensional" before stall, mfoil's Newton-globalization clause
  ("FlexFoil and XFOIL both converge" per your note), "least-squares"
  (now "computed gradients"), and "triangulates" ->
  "Delaunay-triangulates".
Build clean.

## 2026-07-25 ~02:00 — review pass 20 addressed
Pass 20 audited the NLF restructure deletion-by-deletion (and reran both
figure generators — content-identical). Real catches, all fixed:
the cavity-L1 15° non-convergence disclosure (the only converged:false
case of 24) had been deleted while the same commit plotted that case in
the new headline figure — restored in the fig:nlfaft caption with the
L0-artifact note; the polar caption now describes the two AFT curves
(+ provenance + the cd=0.025 source-axis truncation); and my
"FlexFoil and XFOIL both converge" over-reached the committed evidence
(the pkl is source=flexfoil throughout; packaged xfoil SIGFPEs on this
input — reviewer reproduced it) — scoped to "as does XFOIL in the
N_crit sweep of Table 5", preserving the annotation's intent. Plus the
adversarial-mesh payoff sentence, README row, whitespace.

## 2026-07-25 ~02:20 — invariant kernel: compactness & cancellation retained?
YES on both, by construction:
1. Compactness: the invariant branch consumes lapVecAi = the SAME
   ring-summed, dual-volume-weighted output of the VelocityLaplacianRaw
   pre-pass (per-dual-face compact stencil with the edge-aligned gradient
   component replaced by the two-point difference). No new
   second-derivative discretization anywhere — only the projection
   direction changed (u_hat -> s_hat).
2. Cancellation: the odd–even mesh-scale cancellation lives in the
   one-ring average of the Laplacian VECTOR, applied before any
   projection — identical in both forms.
NEW sensitivity (the honest caveat): s_hat = (omega x n_out)/|.| uses the
NODAL vorticity direction (least-squares gradient, not ring-averaged),
where u_hat used the smooth velocity. Protections: where the kernel acts
(shear-dominated) omega's direction is as robust as the shear itself;
where omega is small (BL edge/freestream) Y = d|omega x n| -> 0 kills the
rate via Shat->0 before direction noise matters. Watch-item: cavity
sliver triangles mid-BL — a cavity-family NLF case added to the 2D
verification battery to measure it; if family scatter grows, the cheap
fix is ring-averaging omega with the same dual-volume weights before
forming s_hat.

## 2026-07-25 ~02:45 — annotated round 5 (NLF): tab:nlftrans deleted; sheets upgraded
- "handover" -> "laminar-to-turbulent handover" at the AFT-paragraph use;
  the closing aphorism ("The two closures place their empiricism
  differently; this case cannot separate them.") deleted per strikeout.
- tab:nlftrans AND its comparison paragraph deleted ("with Fig 7 added
  these are no longer necessary"). Dependents rewired: FlexFoil's
  introduction (independent XFOIL-method implementation, <1%
  cross-check) moved into the polar caption where it first appears; the
  XFOIL-converges claim now points at the digitized e9 curve of
  fig:nlfaft (spans 9/15 deg); the cf-caption onset triangles now cite
  Somers's orifices read at computed cl directly (open = extrapolated
  beyond recorded data). Zero dangling refs; build clean.
- Appendix chi sheets (both the alpha-matrix and Re-sweep sets):
  contour levels EXTENDED beyond c_v1 (chi = 30 — where sigma_P is 97%
  complete and f_v1 90% — and chi = 100, the turbulent interior) to show
  the handover dynamics; Eppler UPPER-surface probe range raised 5x
  (L_up 0.0067 -> 0.0335, inherited by the Re-sweep sheets through
  their sqrt(2e5/Re) scaling) so the handover above the bubble shear
  layer stays in frame; probe resolution 140 -> 320 points to keep
  near-wall sampling. All 16 sheets regenerating in the background.

## 2026-07-25 ~03:00 — chi sheets committed (70ec237)
All 16 sheets regenerated with the new levels (chi = 30 and 100 past
c_v1) and the 5x Eppler upper range. Visual check (Re=1e5 upper): the
handover fan above the bubble shear layer is fully in frame, and the
low-Re RE-SEEDING TONGUE — near-wall chi contours running upstream
along the wall to x~0.15, the cold-front signature — is now directly
readable in the L2 rows, giving the Sec. VI mechanism its visual
counterpart in the appendix. Build clean.

## 2026-07-25 ~03:15 — Coder + Eppler 387: verified answer (search agent, sources checked)
DID CODER RUN E387? NO — not with AFT, ever. Verified against his full
146-page dissertation (downloaded; case list: flat plate, PSU 94-097,
S805, S414, NLF(1)-0416, Discus 2cx, CRM, DLR-F11 — zero E387
computations; "Eppler" appears only as the Eppler transition CRITERION),
his AIAA J. 2014 paper, the AFT-2017/2019 papers, and his 2017 NAS talk.
His two E387 publications (Maughmer & Coder, Army TR 10-D-106 2010;
J. Aircraft 51(1) 2014) used LANGTRY–MENTER, not AFT — E387 at Re=3e5.
The 1st Transition Modeling Workshop did NOT include E387 (cases: flat
plate, NLF(1)-0416, spheroid, CRM-NLF).
THE ONLY PUBLISHED AFT-ON-E387: Lopes, Eça & Vaz (IST/MARIN, ReFRESCO),
ASME J. Fluids Eng. 142(5):051503 (2020) — E387 at two incidences
(almost certainly Re=3e5, α=1° and 7°; paywalled), AFT predicts the
separation-induced transition with higher numerical uncertainty.
Open companion: NuTTS'18 (ResearchGate, login-gated).
BEST OPEN E387 BENCHMARK (no AFT, but quality-controlled): Eça et al.,
34th Symp. Naval Hydrodynamics 2022 — NTRS open PDF, E387 Re=3e5
α=1/7 vs McGhee, multi-code with numerical-uncertainty estimates, PURE
VECTOR figures (trivially digitizable).
DOWNLOADS (saved to sa-ai/references/): coder_diss.pdf (dissertation,
via Wayback mirror — PSU eTDA blocks scripts), snh34.pdf (NTRS),
coder_ams2017.pdf, nutts2019_lopes.pdf, koning_ja.pdf. DTIC Army report:
https://apps.dtic.mil/sti/tr/pdf/ADA532502.pdf (browser only).
COMPARISON OPTIONS: (a) honest framing "AFT as implemented by Lopes et
al., Re=3e5" — one Reynolds point, two incidences; (b) the SNH-34
multi-code γ-family band at Re=3e5 as a community reference envelope on
our sweep's 3e5 station; (c) email Rui Lopes (now Chalmers) for his IST
thesis, which likely has fuller E387/AFT polars. Nearest Coder-authored
low-Re LSB case if we want HIS model on a bubble: PSU 94-097 at
Re=2.4e5 in the dissertation (polars + fronts, mixed raster/vector).

## 2026-07-25 ~03:30 — AFT removed from the Eppler section; CRM-NLF answered below
The AFT-family sentence and the tab:epplit "no published data" row are
deleted per your instruction (the citation survives at its NLF-section
sites). Build clean.

## 2026-07-25 ~03:50 — LM/BCM digitization plan (research agent running)
Figure numbering clarified: Figs 7/8 are the NLF polar + NLF Cf sheets
(NOT Eppler); Eppler polar is Fig 10, Eppler Cf Figs 11-12; the NLF
transition-vs-cl plot (where AFT already lives) is Fig 6.

Current state of the two polar figures:
- Eppler polar (Fig 10, regen_eppler_v2.make_polar_figure): SA-AI L0-L2
  both families, LTPT experiment, mfoil/xfoil e9, fully-turbulent SA.
  NO LM/BCM curves yet — but tab:epplit already has their alpha=5 POINTS
  (gamma-Retheta cd 0.0200, SA-BC 0.0142, gamma-SA 0.0133-0.0140),
  figure-read from shahjahan_2024 / cakmakcioglu_2017 / dalessandro_2025.
- NLF polar (Fig 7): SA-AI, mfoil, FlexFoil, turbulent SA, + AFT (just
  added). NO LM/BCM yet.
Overlay mechanics are identical to the AFT addition (data dict -> plot
-> legend entry) in both generators.

Research agent launched for digitizable LM (gamma-Retheta + variants) and
BCM (SA-BC/BCM) data on BOTH Eppler 387 (Re 2e5) AND NLF(1)-0416
(Re 4e6) — polars + transition locations, with download URLs and
vector/raster assessment. User confirmed they want NLF LM/BCM too if it
exists.
Plan once sources land: digitize polars -> overlay on Eppler Fig 10 +
NLF Fig 7; digitize transition locations -> NLF Fig 6 (transition-vs-cl,
alongside AFT); decide Fig 8 (Cf sheets) treatment (transition markers
only, if at all).

## 2026-07-25 ~04:10 — compressibility & crossflow extensions (physics assessment)
FRAME: the model = instability-agnostic SKELETON (transport of the
amplification var + chi->SA handover + Re_Omega onset switch) + an
instability-specific RATE ATOM (a_max, the inflection/curvature
indicator building Shat*g, the threshold shape Re_Omega^c(Shat*g), and
the N_crit(Tu) map). Skeleton reuses for any convective e^N-describable
mechanism; only the rate atom changes.

COMPRESSIBILITY (transonic / first-mode = TS): three rate-atom edits,
no structural change:
1. a_max -> 0.19 * Lambda(M_c): compressible free-shear growth
   suppression (Papamoschou-Roshko/Sandham; spreading down 2-3x by
   M_c=1). M_c built LOCALLY from shear dU + sound speeds, not flight
   Mach. Transonic bubble M_c~0.4 -> Lambda~0.85 (10-20% ceiling drop).
   Leading-order effect.
2. Curvature indicator Z -> GENERALIZED inflection point d/dy(rho du/dy)
   =0 (Lees-Lin) not u''=0: density enters Z; the kinematic sphere
   indicators acquire a thermodynamic input (T field already carried).
   Modest shift at subsonic-transonic edge Mach, but the correct
   criterion.
3. Re-anchor Re_Omega^c shape, k (Blasius N=1 anchor), and the Mack
   N_crit(Tu) map against a COMPRESSIBLE LST/e^N database (Mach-dep);
   receptivity map gains an acoustic branch.
Plus shocks/SBLI as a separate (unmodeled) receptivity concern.
Hypersonic SECOND mode (M>~4): different animal (trapped acoustic,
opposite wall-T sensitivity) -> separate sub-model, not a mod.

CROSSFLOW: a genuine SECOND amplification channel, not a re-tune.
Physics: crossflow velocity (perp to inviscid streamline) is 0 at wall &
edge -> inflectional -> co-rotating CF vortices (stationary in low-Tu).
Different DIRECTION from TS. Needs:
- CF indicator: the current pure-magnitude kernel is direction-agnostic,
  which CF can't tolerate (CF is defined relative to edge direction e).
  ELEGANT local candidate from the invariant-kernel work: s_hat =
  (omega x n)/|.| ROTATES with height (skin-friction dir at wall ->
  e at edge); that twist IS the crossflow. So CF strength ~ |ds_hat/dy|
  or the wall-shear-vs-edge misalignment angle. Nonlocal wrinkle: e
  (edge dir) from velocity at ~edge distance, or approx by grad-p dir.
- CF rate a_cf + threshold on Re_cf (Arnal C1) or a CF N-factor.
- Two-channel combo: transport N_TS and N_CF, fire when EITHER crosses
  (Stock two-N diagrams, same as the spheroid lit).
Attachment-line contamination = 3rd swept cousin (Re_theta_AL>~100),
own cheap criterion.

BOTTOM LINE: transonic compressibility = bounded 3-part rate-atom edit,
realistic next step, needed by transonic-NLF. Crossflow = bigger
(new directional indicator [best hope: invariant-kernel shear-twist],
new rate+threshold, two-channel combo). CRM-NLF needs BOTH at once
(compressible swept wing, CATNLF suppresses CF leaving residual TS) =
a genuine leap, not an increment.

## 2026-07-25 ~07:35 UTC — cavity-L2 a4 landed; section refreshed (all times UTC)
(Clarified per your question: the server clock and all quoted times are
UTC. Cavity a5 ETA ~09:00 UTC, a6 ~14:30 UTC.)
case_cavity_L2_saai_a4: CL 1.0208 / CD 0.02004.
- L2 FAMILY AGREEMENT at 4 deg: dCL 0.7%, dCD 0.7 counts (0.35%) —
  tighter than L1's 0.9%. At eta=0.31 the two finest grids put the
  transition front 0.002c apart (0.549 vs 0.551), separation 0.008c,
  reattachment 0.016c; cavity bubble 0.481-0.645 vs strips 0.472-0.660.
- vs matched-lift reference (0.02125): cavity 12.1 counts below —
  consistent with the structured family's 12.8.
- Paper updated: totals-table cell filled; "Ten converged RANS
  solutions" with pending = cavity L2 at 5/6 deg; ledger carries the
  L2 agreement; the a4 surface figure now has its promised
  unstructured-L2 row (caption updated; the pending clause moved to
  the a5/a6 captions); bubble-edge agreement now an L2 statement at
  4 deg; polar + sectional figures regenerated (sectional a4 cavity
  trace now L2).

## 2026-07-25 ~08:20 UTC — extension memo published (HTML artifact)
URL: https://claude.ai/code/artifact/7431a96b-50e2-499a-ace1-e77bb5a2b472
Detailed page covering: (1) skeleton-vs-rate-atom framing; (2)
compressibility — Langley-curve suppression Lambda(M_c) with the
transonic-bubble point marked (~0.85), Lees-Lin generalized inflection
d/dy(rho du/dy)=0 as a density-weighted curvature invariant
(1/2)d^2[div(rho grad u)].s_hat/rho, acoustic receptivity, second
mode/shocks fenced off, and the note that Re_Omega already uses local
nu(T); (3) crossflow — profile unstable by construction, stationary
(roughness) vs traveling (Tu) receptivity, criteria/two-N/local-closure
landscape, and the memo's central identity: u.omega = u^2 d/dy
arctan(w/u) — the validated helicity indicators (Langtry-CF,
Grabe-Krumbein) ARE the wall-normal twist rate of the invariant
kernel's s_hat frame; indicator fully local, no edge direction needed,
pseudoscalar parity correct; (4) one-frame-three-instabilities table
(curvature along s_hat / density-weighted curvature / twist of s_hat);
(5) caveats from this week's lessons (compensation risk, nodal-s_hat
discretization -> the queued cavity verification case, roughness input
honesty, falsifiable rate-max combination rule vs the spheroid's
phi~30 deg front kink); (6) sequencing, incl. the cheap next step:
evaluate the twist indicator PASSIVELY on the spheroid solutions
(post-processing only) to check it lights the measured crossflow
sectors before building any rate.

## 2026-07-25 ~08:45 UTC — review pass 21 addressed (+ corrections to earlier entries)
Pass 21 audited the tab:nlftrans and AFT-absence deletions (both clean,
zero orphans), re-verified the chi sheets and the cavity-a4 numbers
(all exact), and caught three staleness items — including one that's a
process lesson: THE ROUND-5 APHORISM DELETION NEVER HAPPENED. My replace
target had a whitespace mismatch, silently no-op'd, and the commit
message + my earlier RESPONSES entry wrongly recorded it as done. It is
actually deleted now, and every tex edit in this round printed a
per-edit OK/FAILED check.
Also fixed: Appendix A/B sheet-conventions text updated for the new
contour levels (solid at 1, c_v1, 30, 10^2) and the 5x Eppler upper
frame (0.0335c, deliberately breaking the 1/sqrt(Re) scaling — now
stated); fig:daepolar panel-b caption now says "finest completed grid
per family and incidence" (durable through the a5/a6 landings);
Somers Fig. 9d citation restored at the cf-caption orifice clause;
regen_chi_sheets default root corrected to flow360_fr (the stale
flow360_tie default was what made the first regen silently skip
everything — can't recur from a fresh checkout now).
Verified for the reviewer's minor 4: the two L2-a4 runs used different
solver builds, but the only delta is the newer build's ai_fv1Bypass=0
constant, which gates on (>0.5) — bit-identical eddy-viscosity path at
0; family agreement unaffected.
CORRECTIONS to my earlier entries per the pass: (i) at 4 deg the L2
family pair is tighter than L1 in DRAG (0.35% vs 0.92%) and marginally
LOOSER in lift (0.72% vs 0.65%) — my "tighter than L1's 0.9%" conflated
the two; (ii) the sheet regeneration was 24 PDFs, not 16.

## 2026-07-25 ~09:00 UTC — Daedalus CD-vs-CL slope: quantified; section sheets building
Your observation quantified (ogrid L2 strips vs AVL-Trefftz-rescaled +
FlexFoil profile):
  eta=0.31: RANS sectional dcd/dcl = 0.0180 vs reference 0.0109 — and
  the reference slope is PURE INDUCED (its XFOIL profile term is flat,
  0.0122->0.0121). Excess = RANS PROFILE drag growing ~6 counts per
  0.1 cl where XFOIL's doesn't grow at all.
  Crossing behavior: inboard RANS goes from -7 counts (a4) to +8 (a6)
  vs reference; outboard (eta=0.6) RANS is above at ALL alpha and
  diverging (+7 -> +21 counts) — worst where the bubble is longest.
Candidate mechanisms the new five-row SECTION SHEETS will separate:
 (a) bubble pressure drag deepening with incidence (-Cp plateau depth &
     recovery deficit vs strip Cp);
 (b) handover post-reattachment Cf overshoot growing with alpha (Cf row
     vs strip cf);
 (c) incipient TE separation/thickened recovery at a6 (Cf droop at
     tail);
 (d) mid-bubble transition lead extending the turbulent run (chi/N row)
     — but that offset is ~alpha-independent, so unlikely to make a
     SLOPE.
Prior: (a)+(b) — the stretched handover's reattachment signature grows
with loading; XFOIL's abrupt closure doesn't.
Generator: repro/cfd/regen_daedalus_section_sheets.py — the 2D
five-row layout (probe-max Re_Omega / probe-max Shat*g / max-chi + strip
N / -Cp / Cf_x) cut at eta = 0.10, 0.31, 0.60; columns alpha=4/5/6;
O-grid L2 solid + cavity L2 dashed where complete; FlexFoil strip
dotted. Probes the volumes along in-plane surface normals (0.01 c_loc,
the 2D convention); chi row from the committed chi_surface.npz. Running
now (~30 min: reloads the 37-55M-node volumes); destined for Appendix D
with a main-text pointer from the slope discussion.

## 2026-07-25 ~09:30 UTC — Daedalus L0: never run at canon; ogrid L0 rebuilt & staged
Answer to your question: NO — the canon campaign only ever had L1/L2
phases. No L0 case dirs exist in the tree, and the predecessor
campaign's L0 solutions AND meshes are gone (the old scripts/daedalus
tree is no longer on disk). The paper is internally consistent about it
(totals table L1/L2 only; y+ caption predecessor-attributed), but the
mesh table describes L0 — a referee bait your question anticipates.
Action taken:
- STRUCTURED L0 IS BACK: the ogrid mesh regenerated from the tracked
  ogrid_wing.py refinement ladder (LEVELS[0]; 0.72M nodes; mesh.cgns
  64 MB via the established gmsh->cgns pipeline), and the three case
  dirs are staged from the L1 siblings. Runner:
  daedalus/build_run_ogrid_L0.py (mesh|stage done; `run <gpu>` queued
  as task #24 for after the cavity-L2 campaign, ~20-40 min/case on one
  GPU).
- CAVITY L0: the 3D unstructured mesher was NOT recovered on disk (the
  2D cavity mesher in flexfoil/rans is not the 3D wing one) — cavity L0
  stays absent unless the mesher resurfaces; the section's per-family
  "where completed" wording already covers asymmetric availability.
When the L0 runs land: totals-table L0 rows, polar L0 line (the
"L0->L2 by weight" legend becomes literal again), and a canon re-test
of the predecessor lore that L0 suppresses the bubble at all but 2 of
36 stations.

## 2026-07-25 ~09:50 UTC — L0 meshes FOUND on 017-v100-dev; all six cases staged
Good call. The predecessor Daedalus tree survives intact on 017-v100-dev
at ~/flexcompute/sa-ai/scripts/daedalus — including case_cavity_saai
(the cavity L0, mesh 202 MB) and case_ogrid_saai (ogrid L0, 63.6 MB),
plus all the predecessor SOLUTIONS (L0-L2 both families), which also
restores provenance for the paper's historical predecessor numbers.
- Both archived L0 meshes fetched. The regenerated ogrid L0 matches the
  archived one in size exactly (63,573,561 bytes; md5 differs only
  through HDF5 creation metadata) — the tracked generator reproduces
  the mesh; the ARCHIVED originals are what's staged, for continuity
  with tab:daemesh's L0 metrics.
- All SIX canon L0 cases staged (both families x a4/5/6). Runner:
  daedalus/build_run_ogrid_L0.py run <gpu> (task #24). Queue: after
  cavity-L2 a6 (~14:30 UTC); invariant-kernel verification takes the
  first free GPU, L0 the second.
- When they land: tab:daetotals gains its full L0 rows (both families),
  the polar recovers the true L0->L2 ladder, and the predecessor lore
  (L0 suppresses the bubble at all but 2 of 36 stations) gets re-tested
  at the canon kernel.

## 2026-07-25 ~09:20 UTC — LM/BCM literature: landed hours ago; recap
You asked whether the Langtry–Menter / BCM literature search landed —
yes (full entry above at ~04:40 UTC, "LM/BCM source matrix"; PDFs in
references/). The short version:
- Eppler 387, Re=2e5, BOTH models in one open source: Shahjahan et al.,
  ICAS 2024 (references/icas2024_0327.pdf), VECTOR figures. Fig 6 =
  drag polars for gamma-Re_theta (OpenFOAM) and SA-BC (SU2) at
  Tu=0.1%; Fig 9 = bubble/transition stations. Backups: Cakmakcioglu
  2017 (original BC paper, exactly Re=2e5, raster Fig 8) and
  D'Alessandro 2025 (gamma-SA).
- NLF(1)-0416, Re=4e6: Langtry–Menter YES — Denison et al. OVERFLOW
  transition-workshop paper (references/overflow_tmw.pdf), Fig 11 =
  transition x/c vs c_l for both surfaces (drops directly into
  fig:nlfaft next to AFT). No clean cd-cl polar in it.
- BCM on NLF: NO usable open data (Tarsia Morisco 2025 is a
  mesh-adaptation study at one condition; the Cakmakcioglu 2020
  candidates are paywalled). Same honest-absence verdict as
  AFT-on-Eppler.
Digitization plan (ready, awaiting your go): Shahjahan Fig 6 -> LM+BCM
curves on the Eppler polar (Fig 10); Denison Fig 11 -> LM transition
curves on fig:nlfaft; text states the BCM-on-NLF absence.

## 2026-07-25 ~09:25 UTC — Daedalus drag-slope diagnosis: the section sheets' verdict
Why does CFD sectional CD grow faster with CL than AVL+XFOIL? The
three new five-row section sheets (eta = 0.10/0.31/0.60, columns
alpha = 4/5/6: probe-max Re_Omega, probe-max S_hat*g, max-chi + strip
e^N, -Cp, Cf_x) now answer it:
- Slopes: mid-span sectional dcd/dcl is ~0.0180 (RANS) vs ~0.0109
  (reference). ALL the reference's growth is induced — its XFOIL
  profile drag is flat in cl. The RANS profile drag climbs ~6 counts
  per 0.1 cl.
- Mechanism (visible in the -Cp row): separation and reattachment both
  match the strips at every incidence — the bubble EDGES are right —
  but inside the bubble the model's handover-stretched pressure
  recovery deepens with incidence: the -Cp plateau persists further
  and recovers more gradually than the strip's abrupt-closure
  recovery, and the area between the curves grows with alpha. That is
  bubble pressure drag an instant-transition closure structurally
  cannot produce — the same Sec. eppresweep handover mechanism, now at
  Re_c ~ 6.6e5.
- Non-mechanisms ruled out: the chi=1 lead over N=N_crit sits at its
  ~0.09c convention offset at every incidence (no slope contribution).
In the paper: diagnosis paragraph added to the Daedalus ledger; the
three sheets are in the Daedalus appendix (D) as figs daesec10/31/60,
per your directive to keep them Daedalus-focused.

## 2026-07-25 ~09:25 UTC — "Sections above, total below": the reconciliation
Your observation — SA-AI has MORE drag than the reference over most of
the span, yet LESS in the total — decomposes into three strands:
1) The strip integrals don't tile the wing. Integrating the sectional
   curves gives Delta(RANS - ref) = -3.7/+1.8/+8.3 counts at
   alpha=4/5/6, but the force totals give -12.8/-7.2/-2.1: a nearly
   incidence-INDEPENDENT ~9-10 count offset between the two
   accountings. The strip bins miss a root sliver and cap the tip, so
   the strip-integrated view sits systematically high; slopes are
   trustworthy where levels are not.
2) The reference's sectional induced-drag SHAPE is Trefftz-exact only
   in its integral (local cl*alpha_i is rescaled to match CDff), so
   pointwise sectional comparisons inherit shape error.
3) At alpha=4 there is no paradox even sectionally (RANS is below the
   reference over most of the span); the crossover builds with
   incidence exactly as the bubble-pressure-drag slope does.
Both accounting notes are now stated in the paper alongside the
diagnosis paragraph.

## 2026-07-25 ~09:30 UTC — Disk-full incident: casualties and repairs
The root filesystem hit 100% (~08:45-08:55 UTC). Full damage list:
- cavity-L2 a6 died at launch (rc=1). Freed 182 GB by deleting
  regenerable partitioner dumps (mesh.cgns_rank_*.dmp) from completed
  cases; a6 relaunched 08:57 on all 8 GPUs (~5.5 h, sentinel
  CAVITY-A6-DONE).
- paper/sa-ai.tex was TRUNCATED TO 0 BYTES: the interrupted edit's
  open(...,'w') succeeded, the write hit ENOSPC. Restored from git
  HEAD (f0d93d9, 07:41 — no tex edits existed between commit and
  truncation, so nothing was lost). Edit scripts now write to a temp
  file and os.replace() so a failed write can never truncate again.
- cavity-L2 a5 (finished rc=0 at 08:55, INTO the full disk): forces
  history and volume/surface outputs intact, but both slicing CSVs
  (X/Y_slicing_forceDistribution.csv) are zero-byte. Sectional strips
  for cav-a5 therefore fall back to L1 in the figure (explicit
  has_strips() gate); the CSVs can be regenerated with a short
  restart run once GPUs free up.

## 2026-07-25 ~09:30 UTC — Cavity-L2 a5 landed; paper refreshed
CL = 1.1219, CD = 0.02259 (final-500 mean; std 4e-4/4e-5). Family
agreement at L2 now holds at both completed incidences: alpha=4:
0.7% CL / 0.7 counts; alpha=5: 0.6% CL / 1.6 counts (0.7%).
ai_constants check: a5's resolved-constants block is IDENTICAL to
a4's except one new line, ai_invariantKernel: 0.000000 — the a5 run
used the rebuilt solver carrying the (default-off) invariant-kernel
code. Constants verified, flag off, as intended.
Paper: tab:daetotals a5 cell filled, "Ten"->"Eleven" solutions,
family-agreement sentence covers both incidences, fig:daesurf5
caption's pending-clause dropped (cavity row regenerates when the a5
chi surface map finishes), polar figure regenerated with the cav-L2
a5 point.

## 2026-07-25 ~09:35 UTC — /local_data migration + appendix restructure + mesh line widths
- Large data now migrates to /local_data/qiqi/sa-ai/... mirroring
  017-v100-dev's /local_disk/qiqi convention, with symlinks left in
  the repo so every path keeps working
  (scripts/migrate_to_local_data.sh: rsync, verify second pass
  transfers nothing, rm, ln -s; skips busy/running dirs). 15 Daedalus
  case dirs (~60 GB) moved so far; paused while figure regens read
  flow360_fr, resumes after. One lesson already paid for: the mover
  yanked a dir mid-read under the a5 chi-map (the lsof guard raced);
  the map was simply rerun through the symlink — data verified intact.
- Appendices, per your directives: the three Daedalus section sheets
  live in the Daedalus appendix (D); the eight Eppler lower-surface
  contour sheets (App B alpha sweep + App C Re sweep) are DELETED —
  upper-only now, prose notes the lower surface stays attached and
  featureless. NLF lower-surface sheets kept (App A).
- Mesh figures: they are VECTOR (matplotlib LineCollection -> PDF),
  so thin lines stay crisp under zoom. Line widths now halve per
  level: L0 0.175, L1 0.0875, L2 0.044 (was uniform 0.35); all four
  mesh figures regenerating from the canonical flow360_fr a0 meshes
  (the script had pointed at the stale flow360 tree).

## 2026-07-25 ~09:45 UTC — CORRECTION: cavity-L2 a5 field output also lost
The 09:30 entry said a5's volume/surface outputs were intact — wrong.
volume.pvtu is zero-byte too (everything the solver wrote at shutdown
hit the full disk): no chi surface map, no surface-map figure row, no
section-sheet dashed lines for cav-a5 until recovery. What IS intact:
the full forces history (the CL/CD numbers stand — they come from
total_forces_v2.csv, written incrementally long before the disk
filled) and the 8-rank restart dumps. Recovery plan: after a6 lands,
a short restart run of a5 (~100 steps, force means move < the 4e-4
history std) regenerates volume.pvtu + slicing CSVs; then chi map,
surface-map cavity row, and section-sheet overlays follow. The
fig:daesurf5 pending-clause is restored until then.

## 2026-07-25 ~09:40 UTC — Mesh figures regenerated and verified
All four mesh figures rebuilt from the canonical flow360_fr a0 meshes
with the per-level widths (L0 0.175 / L1 0.0875 / L2 0.044) and
checked visually: the L2 panels now show grid structure where they
were solid ink; the near-wall boundary-layer stack still saturates to
a dark band (cell heights below line pitch — unavoidable), but the
O-grid fan and TE wake refinement read cleanly at every level. They
are vector PDFs, so zooming in a viewer resolves individual lines.
Committed (a08434b); /local_data migration resumed for the flow360_fr
2D cases now that nothing is reading them.

## 2026-07-25 ~10:15 UTC — Review pass 22: three of my numbers were wrong; all fixed
Pass 22 (agent-paper-review/2026-07-25-0929.md, response alongside)
confirmed the slope diagnosis quantitatively (reproduced my 0.0180 vs
0.0110 exactly, regenerated a committed section sheet bit-for-bit) but
caught three real numerical defects, all mine, all now fixed:
1. WRONG WINDOW on the new a5 table cell: I averaged the last 500 CSV
   rows (= 5000 pseudo-steps); the table's convention is the last 500
   pseudo-steps (51 samples). Canon values: CL 1.1223 / CD 0.02254
   (std 1e-5/2e-6). Family gap at a5 is actually 0.58% CL / 1.09
   counts — a STRONGER agreement than the 1.6 counts I printed. Table,
   sentence, canon_results.json all corrected; the caption now spells
   out "final 500 pseudo-steps (51 logged samples)" so the ambiguity
   is dead. (This supersedes the 09:30 entry's numbers.)
2. WRONG Re: the sections run at Re_c = 5.0e5, not 6.6e5 (I mixed the
   dimensional 0.917 m chord into a unit-chord reference length). The
   sheets' suptitles were always right; caption + slope paragraph
   corrected. (Supersedes "Re_c ~ 6.6e5" in the 09:25 entry.)
3. WRONG ACCOUNTING NOTE: my "strip decomposition sits 9-10 counts
   above the force totals" came from an uncommitted heredoc and is not
   reproducible. The committed repro script
   (paper/repro/cfd/daedalus_strip_accounting.py) integrating exactly
   the figure's own sectional curves shows: RANS strips tile their
   force totals to ~1 count; it is the REFERENCE's plotted sectional
   curve that integrates 18-21 counts BELOW its own ledger total
   (induced shape Trefftz-exact only in the integral; profile on nine
   FlexFoil stations). Tex sentence rewritten to say exactly that.
   Moral unchanged: sectional slopes trustworthy, level differences no.
Also from the pass: Appendix D was missing its figure/table counter
renew (the new sheets printed as C5-C7 while the text said Appendix D
— now D1-D3, and tab:daetotals becomes Table D1); the onset-threshold
overlay the appendix text promised is now actually DRAWN in row 1 of
the sheets (solver-exact form, dash-dotted; sheets regenerating); the
paper's last hardcoded figure number ("as Fig. 2") is now a \ref; the
dashed-cavity availability qualifiers are worded durably so the a5
recovery won't need a third touch.

## 2026-07-25 ~10:15 UTC — /local_data migration COMPLETE + full-page mesh figures
Migration finished: 139 directories (all completed Daedalus cases +
the whole flow360_fr 2D tree) now live under /local_data/qiqi/sa-ai/
with symlinks in place; /home went 100% -> 87% (another ~100 GB
freed), /local_data at 78%. Only the running a6 case remains on /home
(moves after it lands).
Mesh figures per your directive: LE and TE windows are now the same
extents (0.16c x 0.14c) — same aspect ratio AND same physical scale —
and the figure canvas is sized to the full text block (6.5 x 8.3 in)
with the tex include set to width=\textwidth, height-capped with
keepaspectratio. Regenerating all four now.

## 2026-07-25: Digitized LM/SA-BC literature data for paper figures
- paper/data/lmbcm_eppler387_digitized.json — ICAS 2024-0327 (Shahjahan): Fig 6b polar Re=200k (LM 19 pts, SA-BC 19 pts, exp Selig/McGhee/Volkers), Fig 9a transition-vs-alpha Re=100k + Fig 10a Re=200k (triangle markers = transition; alphas land on integers ±0.07°). Vector-path extraction via PyMuPDF; colors matched to legend swatches.
- paper/data/lm_nlf0416_transition_digitized.json — Denison OVERFLOW TMW Fig 11 (raster 1280x960): LM fine-grid xtr/c vs cl upper+lower (7 pts each), Somers exp bounds (30+30 pts, laminar/turbulent pairs).
- Verification overlays: paper/data/digitize_check_{eppler,nlf}.png — visually indistinguishable from originals.

## 2026-07-25 ~10:50 UTC — LM/BCM literature curves ARE IN THE FIGURES now
Took your third ask as a go. The digitization agent extracted, and I
verified against the source pages (check PNGs committed at
paper/data/digitize_check_{eppler,nlf}.png — overlays are
point-for-point):
- Eppler polar (fig:epppolar): gamma-Re_theta and SA-BC curves from
  Shahjahan ICAS 2024 Fig 6b, VECTOR-EXACT (19 points each — the
  polyline vertices are the per-alpha marker centers). The story the
  figure now tells: SA-AI and SA-BC both track the measured low-drag
  bucket (SA-BC slightly better at mid-bucket, but it overshoots the
  bucket top and its cl); gamma-Re_theta runs 40-60% high in drag
  through the whole bucket. Bonus extraction: all three experiment
  symbol sets (Selig/McGhee/Volkers) and the transition-vs-alpha data
  at BOTH Re=1e5 (Fig 9a) and Re=2e5 (Fig 10a) are in the JSON for
  future use.
- NLF transition figure (fig:nlfaft): Langtry-Menter fine-grid curves
  (upper+lower) from Denison's OVERFLOW workshop paper Fig 11
  (raster, color/shape-segmented, calibration verified on tick
  pixels). LM lands nearly on top of the recalibrated-AFT/experiment
  cluster on both surfaces — a strong independent reference right
  where SA-AI's points sit. Caption states plainly: no SA-BCM result
  at this condition exists in the open literature.
- tab:epplit's polar cells upgraded from figure-read +/- ranges to
  matched-cl interpolations of the digitized curves: gamma-Re_theta
  0.0195, SA-BC 0.0140 at cl=0.891 (both inside the old error bars —
  the table's story is unchanged, just tighter).
Committed 09b337b (+ 76d6bfe housekeeping: 98 tracked 2D ai_constants
logs relocated out of the migration symlinks, same as the daedalus
move). New Denison bib entry denison_2021_overflow.
Also in 09b337b: the full-page mesh figures (identical 0.16c x 0.14c
LE/TE windows, equal scale, width=\textwidth) — verified visually.

## 2026-07-25 ~11:00 UTC — Invariant-kernel verification: first pass INVALID (protocol), redoing at matched convergence
The three verification cases landed and at first read looked like a
kernel failure (structured NLF transition 0.28 -> 0.62 aft, cavity
0.30 -> 0.10 forward, Eppler bubble shifted). Diagnosis so far:
- A pure-python replica of BOTH kernels on the converged canon field
  (wall-normal probe at x/c=0.15) gives P_invariant == P_standard to
  all printed digits with the physical vorticity orientation — the
  CONCEPT is exactly 2D-degenerate as claimed. A deliberately flipped
  omega sign inflates P 10x, confirming the s_hat construction is the
  sensitive element and that the correct sign reproduces the standard
  kernel.
- Freestream/ahead-of-LE chi is EXACTLY the seed in all runs — no
  contamination; solver env verified (ai_constants diff: only the
  invariantKernel flag differs).
- The smoking gun against my first comparison: a FLAG-OFF CONTROL run
  through the same clone recipe (cold start, 20k steps) ALSO misses
  the canon record — and its transition (0.60) lands next to the
  invariant run's (0.62). The canon NLF cases ran 35k steps; at 20k
  cold the transition front is still creeping aft. My verification
  battery compared mid-transient snapshots against converged records.
  (The one remaining flag-attributable signal: solver turbulence
  residual floors 5 decades higher with the flag on — s_hat sign
  flutter in weak-shear regions is the suspect; watching whether it
  matters at convergence.)
- Redo in flight at matched protocol: all four runs (str control, str
  invariant, cav invariant, Eppler invariant) extended by restart to
  40k, plus a fresh 40k cavity control. Verdict when they land: the
  invariant kernel passes only if flag-ON == flag-OFF within line
  widths AT CONVERGENCE on both families.

## 2026-07-25 ~11:15 UTC — Table 6 -> figure: yes, and the data is RICHER than NLF
Your question: can the reattachment table (Table 6, tab:eppxtr) become a
figure like the NLF transition figure, with separation AND reattachment
per panel — and is the data as tightly spaced in CL?
Answer: yes, and on the experiment side it is RICHER than the NLF case:
- TM-4062 Table III (paper/eppler.pdf p.15 — the full report was already
  in the repo) tabulates oil-flow separation AND reattachment at R=200k
  for NINE incidences: alpha = -2, 0, 2, 4, 5, 6, 7, 8, 8.5 (alpha=8 is
  "NT at .32" — natural transition, no bubble; 8.5 is the post-stall
  leading-edge bubble .03/.18). The repo had only our four alphas; the
  full set is now in the new figure script.
- Shahjahan Fig 10a carries separation (circles) and reattachment
  (squares) markers for LM and SA-BC at integer alpha at exactly Re=2e5
  — a second digitization agent is extracting those now (we already
  have its transition triangles).
- mfoil e^9 sweeps alpha densely for free: a -2..7.5 x 0.5deg bubble-
  station sweep is running (separation/reattachment from signed Cf,
  same definitions as the table).
- SA-AI: currently the same density as NLF (4 incidences x 6 grids).
  Queued to densify: alpha = -2, 1, 3, 4, 6 on both L2 grids (10 quick
  2D runs) once the current GPU batch lands.
Design: two panels sharing the alpha axis (separation left,
reattachment right), NLF-figure conventions (oil flow black circles,
mfoil dotted, LM/SA-BC in the polar's colors, SA-AI six grids by
marker size). Axis is alpha rather than cl because every reference
here is alpha-native (oil flow, Fig 10a). Script:
repro/cfd/regen_epp_bubble_figure.py; the table's definitions and the
mfoil-at-7deg caveat move into the caption. Will replace tab:eppxtr
once the digitization and sweep land.

## 2026-07-25 ~11:40 UTC — L0 failure modes: NLF vs Eppler are DIFFERENT failures
Your observation quantified (CD counts, canon window):
  NLF a4:    strL0 110.7 vs strL2 74.0 (+37) | cavL0 106.3 vs cavL2 77.6 (+29)
  Eppler a5: strL0 205.1 vs strL2 155.2 (+50) | cavL0 161.0 vs cavL2 148.1 (+13)
Pressure/friction split (surface-integral CD_p; friction = CD - CD_p):
  NLF a4:    strL0 63p/48f vs strL2 27p/47f  -> excess is +36 PRESSURE, friction dead-on
             cavL0 76p/31f vs cavL2 31p/47f  -> +45 pressure MINUS 16 friction
  Eppler a5: strL0 159p/46f vs strL2 105p/51f -> +54 pressure, friction -4
             cavL0 109p/52f vs cavL2 94p/54f  -> +15 pressure only
Mechanisms (diag plot: paper/diag_l0_failure_modes.py):
1. NLF (attached, Re=4e6): NOT a transition failure. The fronts sit
   within 0.06c of L2 on both families (str slightly early 0.22 vs
   0.28, cav slightly late 0.36 vs 0.30) and the Cp distributions
   overlay. The excess is the classic under-resolved-boundary-layer
   FORM-DRAG error: the momentum deficit that should be skin friction
   leaks into pressure drag through the smeared TE recovery. The
   cavity shows it most nakedly — its L0 turbulent Cf is ragged and
   HALF the resolved level (friction 16 counts low) while pressure
   goes 45 high. Model-independent coarse-grid behavior; both
   families converge by L1.
2. Eppler (bubble, Re=2e5): a genuinely different, TRANSITION-ZONE
   failure, and only the O-grid suffers it. strL0's bubble is
   SHORTER and SHALLOWER than converged (reattach 0.587 vs 0.655,
   min Cf -2.6e-3 vs -4.7e-3): the -Cp plateau breaks ~0.05c early
   into a long soft recovery, and that fattened mid-chord pressure
   loop is the +54 counts. The cavity L0 bubble is nearly canonical
   (0.43-0.57). Why the difference: streamwise spacing exactly where
   the bubble lives. The cos-clustered O-grid is COARSEST at
   midchord — L0 Delta_s = 1.7-1.9%c across x=0.4-0.65 (~12 points
   over the bubble, 3-4 over the handover) — while the cavity's
   curvature-driven sizing gives 1.1-1.4%c there (tab:eppmesh rows).
   The O-grid's clustering budget protects LE and TE; the bubble
   pays.
Irony worth noting: strL0's broken short bubble lands nearly ON the
experimental reattachment (0.587 vs oil flow 0.59) — the right answer
for the wrong reason, now visible as the small L0 square hugging the
data in the new fig:eppbubble while the converged grids sit late.

## 2026-07-25 ~12:00 UTC — L0 follow-up: is it y+? And the strL0 LE suction peak
Q1 (NLF: just y+ too high?): No — y+ cannot be the driver, by your own
argument made quantitative. NLF L0 y+ (95th pct): cavity 2.99,
structured 1.90 (tab:nlfmesh) — the cavity is 1.6x WORSE on y+ yet its
drag error is SMALLER (+29 vs +37 counts), i.e. y+ anti-orders the
error. Also the structured family's friction came out dead-on (47.6 vs
47) despite y+~2 — the excess is all in pressure. What orders the
error is BL PROFILE resolution: cells across the boundary layer
(first-cell-to-BL-height ratio; at Re=4e6 the BL is ~0.5-1%c and the
L0 stretching gives ~10-15 cells through it) and streamwise spacing.
The under-resolved profile mispredicts displacement growth ->
form-drag error; on the cavity the same under-resolution additionally
shreds the turbulent Cf itself (visibly ragged, half the resolved
level), trading friction for pressure. y+~2-3 contributes at the
margin but the families' error ordering rules it out as the cause.
Q2 (Eppler strL0 LE): confirmed — good catch, and it revises my
midchord-only story. The strL0 suction peak is clipped ~7% (-Cp 1.35
vs 1.45): the Construct2D O-grid at L0 has Delta_s @ LE = 4.9e-3 c vs
the cavity's 1.1e-3 (4.4x coarser; NLF same pattern, 5.3e-3 vs
0.55e-3) — the L0 O-grid walks around the LE arc in ~8-10 points. So
the structured-L0 Eppler failure is a two-deficit story: (1) clipped
LE suction peak (pressure-drag error at the nose AND a weakened
adverse-gradient onset feeding the bubble), plus (2) the cosine
distribution being coarsest at midchord where the bubble lives
(1.7-1.9%c). The cavity L0 is finer at BOTH ends, which is why it
keeps the bubble nearly canonical and pays only the generic
form-drag tax.

## 2026-07-25 ~12:00 UTC — Invariant-kernel verdict at matched protocol: FAILS
The redo is conclusive and the answer is negative:
- Controls validate the harness EXACTLY: flag-off clones at 40k
  reproduce canon to the last digit (str 0.9796/0.00741 vs canon
  0.9797/0.00740; cav 0.9839/0.00771 vs 0.9843/0.00776).
- Flag-on at the same protocol converges AWAY from canon: str NLF
  +12.4 counts, cav NLF +22.3, Eppler a5 +10.8. Not line widths.
- The concept is NOT what fails: the python replica of both kernels
  on the converged field is identical to all digits (parallel-layer
  degeneracy exact). The freestream/ahead-of-LE chi is exactly the
  seed in flag-on runs too (no contamination), constants verified.
  The remaining suspects are discrete-feedback effects where s_hat's
  sign is noise-driven (BL edge / weak shear): flag-on runs carry a
  turbulence-residual floor 5 DECADES higher — persistent production
  flutter — and small positive P where the magnitude form clips to
  zero can bias the amplification integral once the solution is
  allowed to respond (the python check was frozen-field).
DECISION per the original plan (verify on 2D before 3D): the invariant
form is NOT cleared. The spheroid campaign runs the standard magnitude
kernel. The elegant form stays in the extension memo as future work
with this diagnosis attached (likely fix: blend s_hat to the
magnitude representative where |Y| << |X|, i.e. weak shear, keeping
the signed behavior only where vorticity is coherent).

## 2026-07-25 ~12:05 UTC — Negative-alpha NLF pair: landed and converged (40k)
str/cav L2 at alpha=-4: CL +0.018/+0.036 (zero-lift hit), CD
0.00768/0.00746 vs experiment ~0.0068 at matched CL (+9/+7 counts).
At alpha=-8 (experimental edge): CL -0.476/-0.457 (exp -0.474 at -8),
CD 0.01125/0.01110 vs exp ~0.0105 (+7/+6 counts). Family agreement
1-2 counts everywhere. Transition: at CL~0 the camber makes the LOWER
surface the adverse side — SA-AI lower front 0.08-0.16 (LM's fine-grid
lower curve says 0.18 at the same cl; the experiment sits later), upper
0.46-0.50 (LM 0.48). Campaign JSON updated; the transition figure now
carries the negative-branch points on the finest grids.

## 2026-07-25 ~12:35 UTC — fv1-bypass: conclusion so far, and the full Eppler battery is RUNNING
Conclusion of the disable-fv1-outside-buffer-layers exploration:
1. It does exactly what the bursting-boundary analysis says is needed:
   it commits the handover faster (fv1's chi^3 gate starves young
   turbulence; bypassing it outside buffer layers removes the SA-side
   anemia). At Re=1e5 it flips the model from the burst branch
   (0.77/0.044) to a CLOSED bubble: CL 0.93 / CD 0.0231 against exp
   0.873/0.0237 — drag lands ON the measurement, lift +6%.
2. The "violent limit cycle" scare was an artifact of the laminar-
   slowdown pseudo-time schedule, not of the bypass (fSlow=1 battery,
   both on/off flutter equally under the schedule).
3. At 2e5 its footprint is small but not zero: dCL -0.5%, dCD -2.8
   counts (comparable to mesh-family spread; physically expected —
   2e5 also has a bubble where the bypass legitimately acts).
So: promising as the model's missing "fast transition commit" lever;
the open question is inertness across the validated envelope. Per your
question — yes, and it is now running: 18-run battery (bypass ON+OFF
matched pairs, cold, fSlow=1 with seed decompensation, 40k steps) over
str-L2 a0/2/5/7 at 2e5, cav-L2 a5, and the a5 Reynolds sweep
60k/100k/300k/460k. The sweep rows answer the money question: does the
bypass move the bursting boundary (60k!) without damaging the
converged range (300k/460k)? ~2h. NOTE the OFF controls also test
something independently interesting: the 2026-07-24 pair suggested the
cold/fSlow=1 protocol itself moves the 2e5 state vs the canon
continuation numbers — if confirmed, that is a second bistable-window
symptom at 2e5 worth knowing about.

## 2026-07-25 ~12:55 UTC — fv1-bypass free parameters, and the analytic answer
The bypass blends mu_t from the fv1-damped value to nuHat with weight
w = s(chi) * G(q), where q = Re_u / (y+ U+(y+)), y+ := chi/kappa, and
U+ is the Reichardt law. Free parameters as implemented: the two gate
windows — s(chi) switching on over chi in (1,2), and G(q) over q in
(2,4). The Reichardt/kappa constants are standard, not free.
Do we have an analytic, CFD-independent answer? Largely YES, and it is
the reason this construction was chosen:
1. The q gate's CENTER is analytic. SA's whole design premise is that
   nuHat stays LINEAR through the buffer layer (nuHat = kappa y u_tau;
   fv1 exists precisely to damp nu_t while nuHat stays linear). So in
   any attached equilibrium wall layer, y+ = chi/kappa is exact and
   Re_u = y+ U+(y+) identically -> q = 1 EXACTLY, parameter-free. In a
   lifted/transitional shear layer at the same chi, d and |u| are
   set by the bubble geometry, not the wall unit -> q >> 1 (measured
   O(10-100) in our bubbles). The discriminator is not tuned; it is
   law-of-the-wall self-consistency.
2. The q gate's MARGIN (on over 2->4) must clear how far attached
   layers stray from Reichardt. In the gated band (chi in (2,30), i.e.
   y+ in (5,73)) the deviation is bounded by the pressure-gradient
   distortion of the law of the wall; with a Coles log-wake composite
   the wake contribution at y+ <= 73 stays small for any attached
   Pi (it lives at y/delta = O(1)), and near-separation APG
   half-power distortion reaches factor ~2 at the top of the band.
   So q_on = 2 IS the analytic edge, and full-on at 4 is margin. This
   can be made a one-page repro/analytic calculation (composite
   profiles, no CFD) if we promote the bypass into the paper.
3. The chi ramp s(chi) in (1,2) can be ELIMINATED as a parameter: tie
   it to the model's own sigma_t handover ramp (center 1, width 4),
   which is where turbulence formally begins. Then the bypass carries
   ZERO new tunables beyond the q margin, which is analytic.
Also inert by construction at both ends: chi<1 (mu_t negligible either
way) and chi>30 (fv1 ~= 1 already) — the bypass can only act in the
handover band, which is exactly where the bursting-boundary analysis
says the model is too slow.

## 2026-07-25 ~13:15 UTC — The max(S_hat*g) double-peak inside the bubble: investigated
Your annotated question: row 2 shows a double peak inside the bubble
with a sharp dip between them, roughly at the chi=1 crossing. Why?
I probed the full y-resolved P = S_hat*g field (not just its
wall-normal max) through the bubble on str-L2 a5 (diagnostic heat map:
scratchpad sg_doublepeak.png; the committed row-2 curve is the
y-max of this field). The answer: THE TWO PEAKS ARE TWO DIFFERENT
SHEAR LAYERS, and the dip is the handover between them.
- Peak 1 (x ~ 0.42-0.53) is the LIFTED LAMINAR shear layer over the
  recirculation core — the inflectional profile drives P to its
  ceiling (~1) across the mid-bubble, aided by the low advection
  speed inside the bubble (X small).
- The DIP (x ~ 0.55): the transition the layer itself ignited kills
  it. Once near-wall chi crosses 1 -> c_v1 (the crossing sits at
  0.48-0.53 here), the arriving eddy viscosity diffuses exactly the
  quantities P is built from — the inflection (Z) and the
  concentrated shear (Y) of the outer branch — while the TURBULENT
  reattachment shear has not yet formed. Both branches are
  momentarily weak; the y-max is the upper envelope of a decaying
  curve and a growing one, and such an envelope has a sharp V at the
  crossing — hence the narrow, deep notch (the argmax height jumps
  discontinuously across it: branch switch confirmed numerically).
- Peak 2 (x ~ 0.60-0.66) is the REATTACHMENT layer: the forming
  turbulent wall jet concentrates shear (Y large) while the flow
  near the reattachment line is slow (X small), so S_hat -> 1 and
  g -> 1 saturate P again. Downstream, the attached turbulent BL
  speeds up along the wall (X grows) and P relaxes to the attached
  level (~0.2).
So the dip tracks the chi handover CAUSALLY (the diffusion of branch
1 is done by the handover's eddy viscosity), which is why it lines up
with the chi=1..c_v1 crossing on row 3. It is, rather neatly, the
model's own transition-completion marker expressed in the rate
coordinate. Note this also explains why the second peak often tops
the first: at reattachment the advection-speed X in the denominator
of g is at its smallest. Diagnostic committed as
paper/diag_sg_doublepeak.py.

## 2026-07-25 ~13:45 UTC — fv1-bypass: s(chi) tied to sigma_t (parameter eliminated); PROMOTION battery running
Per your approval, s(chi) in the bypass gate is no longer its own
window: it is now the model's own sigma_t handover ramp
(1 - exp(-(chi - switchCenter)/switchWidth), the committed (1,4)),
so the bypass carries zero new chi parameters; the q gate keeps its
analytic (2,4) law-of-the-wall margin. Clean rebuild done (SOP).
Promotion criteria recorded: canon if the flat plate AND NLF are
unaffected AND Eppler improves. The battery now covers all three:
17 slots x (ON, OFF) at matched protocol — Eppler a0/2/5/7 + cav a5 +
Re sweep 60k/100k/300k/460k; NLF a0/4/9/15 + cav a4; flat plate
Tu0040/Tu0300. All 8 GPUs, ~3-4 h. Verdict table when it lands.

## 2026-07-25 ~14:30 UTC — Review pass 24 + the alpha=-8 front diagnosis
Pass 24 verified the whole pass-23 fix round and the eppbubble
promotion bit-for-bit, and caught two real defects of mine:
1. GHOST LEGEND: fig:eppbubble shipped with an "XFOIL (e9, a>=6.5)"
   legend entry and NO curve (the xfoil fallback SIGFPE'd; the
   committed branch data is all-null) — and my pass-23 response file
   claimed the branch had landed. Fixed: legend gated on drawn data;
   response corrected. Investigated the lost table-footnote number
   too: XFOIL's alpha=7 Cf never crosses zero (min +2.2e-4), so the
   old "0.40c" was a threshold crossing, not a zero-crossing
   reattachment — it stays off the figure (convention mixing).
2. THE ALPHA=-8 FRONTS DISAGREE WITH EVERYTHING, and the worse half
   was invisible (clipped by xlim). Diagnosis on the am8 fields:
   - Suction (lower) surface: a -Cp = 4.7 leading-edge spike whose
     adverse recovery should trip transition at ~0.01c (XFOIL/AFT do).
     SA-AI keeps the surface laminar to 0.89c: the spike bubble
     offers ~0.01c of streamwise room while the chi handover needs
     many times that — the paper's handover-length mechanism at its
     most extreme. A genuinely informative failure.
   - Pressure (upper) surface: front at 0.14-0.22c vs measured ~0.5c,
     sitting just downstream of the stagnation point that has moved
     onto that surface at -8 deg — the rate coordinate's
     low-advection-speed (X->0) inflation region.
   Decision per the reviewer's options: KEEP the markers, disclose
   both misses in the caption, and extend xlim so the lower point is
   visible. The forces stay within 7 counts — the polar is honest —
   but the front misses now say out loud what they are: the boundary
   of applicability at strong negative incidence, outside anything
   the model was calibrated on.
Minors: mfoil sweep now interpolates its reattachment crossing (and
never clobbers the xfoil block); the L0 drag split lives in the
committed diag script; nlfpolar caption covers the negative branch;
campaign JSON schema completed; the invariant-kernel harness moved
out of paper/repro (the paper is silent on the abandoned variant, as
it should be).

## 2026-07-25 ~15:15 UTC — CORRECTION to the double-peak explanation (review pass 25)
The reviewer refuted the CAUSAL half of the double-peak story you
liked, and the data is on its side. What survives and what changes:
- SURVIVES (verified): the two peaks are two different shear layers —
  the lifted laminar layer over the recirculation core (peak 1, at
  the top of the row-2 probe window at its maximum) and the near-wall
  reattachment layer (peak 2 at d=0.0014c). Peak positions, heights,
  chi crossings: all as quoted.
- CORRECTED: the sharp notch between them is the lifted layer
  CLIMBING OUT OF THE FIXED 0.01c PROBE WINDOW, not the handover
  diffusing it. Probed deeper, the layer stays saturated (P~1) just
  above the window through the entire "collapse", until x~0.69; the
  full-depth envelope never drops below 0.85. My "deeper probe shows
  it dying by 0.55" misread an argmax jump between two branches that
  are BOTH at ceiling. The chi=1/c_v1 alignment with the notch is
  correlational (both track the bubble's growth toward reattachment),
  not causal. Appendix B now says exactly this.
- Also fixed: both y+ conversions were 10x too large (u_tau missing
  the xU_inf=M=0.1 in solver units): the window top is y+~42 and
  peak 2 sits at y+~6 — the SUBLAYER of the forming wall layer, which
  actually fits "new near-wall reattachment layer" better than the
  wrong 58 did.

## 2026-07-25 ~15:20 UTC — fv1-bypass PROMOTION BATTERY: verdict table
All 32 runs done (16 slots x ON/OFF, matched cold/fSlow=1/40k
protocol, sigma_t-tied s(chi), analytic q gate). ON-vs-OFF:
  case            OFF CL/CD        ON CL/CD         dCD(counts)
  epp2e5 a0       0.389/0.01059    0.388/0.01047    -1.3
  epp2e5 a2       0.608/0.01206    0.608/0.01191    -1.6
  epp2e5 a5       0.927/0.01529    0.928/0.01498    -3.1
  epp2e5 a7       1.140/0.01452    1.140/0.01444    -0.8
  epp2e5 a5 cav   0.937/0.01477    0.937/0.01455    -2.2
  epp 60k a5      0.622/0.05382    0.633/0.05343    -3.9 (still burst)
  epp100k a5      0.812/0.04048    0.828/0.03847   -20.1 (still open)
  epp300k a5      0.940/0.01116    0.940/0.01096    -2.0
  epp460k a5      0.949/0.00886    0.949/0.00881    -0.5
  nlf a0/4/9/15   ---              ---              -0.1/-0.3/-0.4/-1.3
  nlf a4 cav      ---              ---              -0.4
  flat Tu0040     0.024/0.01181    0.024/0.01166    -1.5
  flat Tu0300     0.034/0.01694    0.034/0.01691    -0.3
Against your promotion criteria:
- NLF: unaffected (<=0.4 counts except -1.3 at the near-stall 15 deg;
  CL moves <=0.08%). PASS.
- Flat plate: Tu0300 untouched; Tu0040 loses 1.5 counts — needs the
  onset-Re_theta check (CD alone conflates onset with the turbulent
  run) before calling it. PROBABLE PASS, verify onset.
- Eppler: improves at EVERY condition, direction always toward the
  measurement: -0.8..-3.1 counts through the 2e5 benchmark (a5 str
  0.01498 vs exp 0.0138 — about a third of the canon gap closed),
  -2.0/-0.5 at 300k/460k, and -20 counts at the 1e5 bursting
  boundary. IMPROVED — but the 1e5 bubble does NOT close (0.828/0.038
  vs the burst 0.812/0.040; experiment 0.873/0.0237).
THE TWIST: the earlier exploration (old s(chi) window (1,2)) CLOSED
the 1e5 bubble outright (0.93/0.0231, drag on the measurement).
Today's sigma_t-tied s is much weaker exactly where fv1 bites
(s=0.22 at chi=2, 0.63 at chi=5, full only ~chi 20), and the 1e5
closure is lost. The parameter elimination has a real cost at the
bursting boundary.
Recommendation: make the s-ramp width an explicit input defaulting to
the sigma_t width (keeps the zero-parameter default), and test width=1
(the old window's strength) at 1e5 + the 2e5 benchmark + one NLF +
one flat plate (8 quick runs). If width=1 closes 1e5 while staying
inert elsewhere, the promotion decision becomes: accept ONE tunable
with an analytic anchor candidate (the fv1 half-saturation scale,
chi~c_v1/2... to be derived) in exchange for moving the bursting
boundary — your call.

## 2026-07-25 ~16:00 UTC — fv1 switch speed: the analytic case for FASTER than sigma_t
Your question: any analytic reason (besides Eppler) for the bypass
switch to be faster than the handover? YES — the two ramps answer
different questions and tying them double-counts:
- sigma_t (width 4) encodes turbulence MATURITY — how gradually nuHat
  earns SA-production authority. Real physics, already applied to
  production.
- fv1 is NOT a maturity function. It is WALL-PROXIMITY damping: its
  derivation is the attached-layer identity chi = kappa*y+, under
  which fv1 enforces mu_t ~ y^3 in the sublayer; its half-saturation
  chi = c_v1 = 7.1 IS y+ ~= 17 (mid buffer layer, by construction).
  In a lifted transitional layer that identity is void at EVERY chi —
  and the q gate is exactly the parameter-free certificate of that.
  Once q fires, fv1 damping is an unjustified error with no
  chi-dependence excuse. s(chi)'s only remaining jobs: smoothness at
  the chi=1 edge, and inertness below chi~1 where mu_t < nu/360
  regardless.
- Ergo: sigma_t-tied s double-counts immaturity (production
  discounted AND mu_t wall-damped in a layer with no wall). The
  principled structure: maturity on production (sigma_t), geometry on
  the viscosity assembly (q), s = fast one-chi-unit smoothing ramp.
The aggressive switch is therefore the analytically preferred form;
the width-1 battery (8 runs, in flight) is confirmation, not
justification. If it closes 1e5 and holds the NLF/flat-plate gates,
proposal: canon = bypass ON, s width ~= 1.

## 2026-07-25 ~16:05 UTC — Fig. 13 FlexFoil branch + the e9 no-bubble finding
FlexFoil (faithful-XFOIL, target/release/rustfoil) swept alpha=-2..9
at Re=2e5 (repro: run_flexfoil_bubble_sweep.py). Branch added to
fig:eppbubble (dash-dot, alpha<=6.5; isolated 8.0 sliver marked).
Finding for the caption/text: at alpha>=7 the e9 integral closure
TRANSITIONS AHEAD OF SEPARATION (min Cf stays positive, +2e-4 at 7 —
identical in XFOIL, same closure) and predicts NO bubble, while the
oil flow still shows one at 7 deg (0.33/0.48). mfoil's convergence
failure at 7 and the missing high-alpha e9 stations are thus the same
phenomenon: the reference family loses the bubble before the
experiment does.
Also queued per your directive: Eppler alpha=-2 and 8.5 canon runs
AFTER the fv1 promotion decision (task list updated).

## 2026-07-25 ~17:20 UTC — fv1 bypass IS CANON; full 2D recomputation running
Your decision executed:
- canonical_ai_env() now exports AI_FV1BYPASS=1 and AI_FV1_SWIDTH=1 —
  the LINEAR s-ramp over chi in (1,2) (the original clip form; the
  exponential width-1 probe showed even it is too soft: 0.840/0.0367
  at 1e5 vs the linear form's closed 0.925/0.0253). Compiled defaults
  stay bypass-OFF so classical-SA comparisons are untouched.
- NEW campaign root flow360_fv1 (on /local_data, symlinked): 93 cases
  staged and RUNNING on all 8 GPUs — NLF 24 + negative pair (both L2
  fams), Eppler 24 + extension alphas (-2, 1, 3, 4, 6, 8.5 on both
  L2s: the oil-flow range you asked for, task #28 folded in), the
  full Re sweep 24 (uniform sweep_{fam}{L}_ names), flat plates 5.
  Old canon flow360_fr untouched for provenance; the paper flips via
  SAAI_CFD_ROOT when we adopt. First-case solver echo verified:
  ai_fv1Bypass 1, ai_fv1SwitchWidth 1. ETA ~8-12 h.
- When it lands: full old-vs-new-canon comparison tables (forces,
  fronts, bubble stations), flat-plate onset-Re_theta check
  (the Tu0040 -1.5 ct needs the onset location, not CD), then the
  paper migration (constants section gains the bypass term, Appendix
  E assembled model updated, every figure/table regenerated from the
  new root) as its own reviewed pass. The 1e5 bistability/fork
  studies of Sec. VII will need re-running at new canon after that.
- NOTE the sigma_t-tied battery + exponential width probe remain on
  disk (fv1b_*, fv1w1_*) as the promotion evidence trail; the
  matched-protocol linear ON/OFF pairs will be pulled from the new
  campaign vs the OFF battery.

## 2026-07-25 ~18:10 UTC — Your alpha=7 catch: the e^N Cf was EDGE-normalized (real bug, 4 figures)
You were right and it was worth chasing: in fig:eppcfhigh's alpha=7
column the dotted e^N Cf is XFOIL's dump (mfoil stalls there), and
XFOIL/FlexFoil dump Cf is normalized by the LOCAL EDGE dynamic
pressure (tau / (rho ue^2 / 2)) while the CFD rows and mfoil use the
freestream convention. On the slow pressure side (ue~0.68 at alpha=7)
that inflates the dotted curve by 1/ue^2 ~ 2.2x — the "many times
higher" you saw. Verified: multiplying by ue^2 (= 1 - cp) drops the
XFOIL lower-surface curve EXACTLY onto the CFD (including the 0.008
peak at x=0.03; overlay in scratchpad cfnorm_check.png).
Affected and fixed (conversion at plot time; all zero-crossing
extractions are sign-invariant and untouched):
- fig:eppcfhigh alpha=7 (XFOIL dotted),
- NLF five-row sheets where the XFOIL fill is used (alpha=9/15),
- the Re-sweep sheets at 60k/460k (XFOIL-primary rows),
- the Daedalus section sheets' strip C_f row (FlexFoil dump; xue^2
  via the stored ue).
mfoil-sourced overlays were always freestream-normalized and correct.
Regens running; caption of fig:eppcfhigh now states the convention
(and corrects "FlexFoil" -> XFOIL — the alpha=7 reference is the
XFOIL pickle; FlexFoil supplies the N envelopes elsewhere).

## 2026-07-25 ~18:15 UTC — CANON CLARIFICATION (important): NOT the invariant kernel
To your "confirm the canon is the vector-calculus implementation +
fv1 bypass": NO on the first half. Canon = the MAGNITUDE (standard)
sphere kernel + fv1 bypass (linear s over (1,2), analytic q gate).
The invariant/vector-calculus kernel FAILED its 2D verification at
matched protocol this morning (controls reproduce old canon exactly;
invariant +12.4/+22.3/+10.8 counts on str-NLF/cav-NLF/Eppler;
turbulence-residual floor 5 decades high — s_hat direction flutter in
weak shear suspected). Per your own gate (verify on 2D first), it is
not cleared; the running 93-case campaign uses the magnitude kernel.
The invariant form remains a debugging project (candidate fix: blend
s_hat to the magnitude representative in weak shear) and lives in the
extension memo, not the paper.
Paper rewrite scope right now (fv1 bypass only): Sec. II gains the
bypass definition + double-counting argument; Appendix E's mu_t
assembly gains the (1-w) fv1 + w blend; the analytic q-margin
composite-profile note goes to repro/analytic. Results
sections/figures flip to flow360_fv1 when the campaign lands. I'll
list every rewritten paragraph as it happens.

## 2026-07-25: AIAA Transition Workshop research (Q1: 2nd workshop? Q2: NLF-0416 open sources)
- NO 2nd Transition Modeling & Prediction Workshop held/announced as of Jul 2026. Follow-ons: 1st AIAA LFC Transition-Prediction Workshop (suction LFC, LST/PSE, 2025-26, Schrauf/Vijgen/Badrya, 3 test cases, flyer+test-case PDF on transitionmodeling.larc.nasa.gov); NATO AVT-313 (prolate spheroid); AIAA APA Transition Modeling DG verification-paper series (OVERFLOW/FUN3D gamma-Re_theta J.Aircraft).
- JACKPOT for NLF-0416 digitization: Coder NAS seminar "Summary of the 1st AIAA CFD TMPW" slides, pp.16-22 = Case-2 grid conv (a=0,5) + alpha-sweep overlays of ALL 14 submittals (x_tr/c vs alpha + polars), raster 1000x750 PNG. Saved: references/TransitionMPW1_summary_Coder_NAS2021.pdf. Also saved proceedings agenda (participant list+deck links) and Piotrowski-Zingg(2020) UTIAS preprint (vector Fig.6 alpha sweep).
- 12+ participant slide decks freely downloadable from transitionmodeling.larc.nasa.gov (ANSYS, Tsinghua k-w-gamma, TAI SA-BCM, NLR, CVUT, MARIN/IST, Siemens, CIRA, ONERA, UMD, KAIST, Toronto).

## 2026-07-25 ~18:45 UTC — Workshop questions answered (agent report, PDFs in references/)
Q1 (2nd workshop?): NO — none held or announced as of Jul 2026; no
summary journal paper either. Follow-ons: the 1st AIAA LFC
Transition-Prediction Workshop (2025-26, suction-LFC/LST-PSE scope,
different beast), Coder's APA discussion group verification-paper
series (Venkatachari et al. J. Aircraft 2024, gamma-Re_theta_t in
OVERFLOW+FUN3D), and NATO AVT-313 (spheroid; headline finding = big
cross-implementation scatter — useful context for our Sec. spheroid).
Q2 (more NLF participants than AFT?): YES. The jackpot:
references/TransitionMPW1_summary_Coder_NAS2021.pdf — Coder's NAS
seminar summary of the workshop, pp. 15-22 = Case 2 (NLF-0416) with
ALL 14 submittals overlaid: grid convergence at alpha=0/5 and full
alpha-sweep x_tr/c (both surfaces) + polars vs experiment. Raster
(lossless) but digitizable. Also saved: Piotrowski & Zingg (VECTOR
alpha-sweep transition, SA-LM2015 + smooth variant) and the workshop
proceedings index (links to 13 participant decks: ONERA elsA, Ansys
Fluent, NLR, CTU OpenFOAM, CIRA, MARIN/IST ReFRESCO, STAR-CCM+,
Tsinghua k-omega-gamma, UMD, SA-BCM (Cakmakcioglu), KAIST...).
PROPOSAL: digitize the summary's alpha-sweep overlays into a shaded
"workshop envelope" band on fig:nlfaft (14 curves would be soup) +
optionally Piotrowski's vector curves as one named line. Awaiting go.

## 2026-07-25 ~19:20 UTC — Canon campaign PAUSED (your call); invariant-kernel debug resumed
Campaign stopped cleanly at ~20/93 (completed cases preserved in
flow360_fv1; resume needs a skip-completed guard added to
run_sphere_campaign). GPUs turned to clearing the vector-calculus
question. First finding, from the existing 40k fields:
- THE LEADING EDGE IS THE CONTAMINATION SITE. Control: chi == seed
  exactly everywhere at x<0.02. Invariant run: p99 = 3x seed, max 6x
  seed in the same window. Spurious amplification at/above the LE
  seeds the whole downstream boundary layer -> earlier transition ->
  the +12/+22/+11 counts. (Wake elevation exists but is modest and
  post-transition.)
- Mechanism (quasi-2D makes it sharp): omega is out-of-plane, so
  s_hat is ALWAYS the in-plane tangent, signed by omega_y. In
  near-irrotational regions (above the LE curvature, outer BL edge)
  omega_y is gradient noise, its sign flutters, and the nodes with
  s_hat anti-parallel to u get X = -|u| -> g ~ +1 -> P ~ +Y/|u|
  (small but positive) where the magnitude form's X = |u| gives g < 0,
  clipped to zero. The onset gate can pass there (large d). This also
  explains the 5-decade residual floor (sign flutter never settles).
- Candidate fix (kernel v2): blend the signed representative toward
  the magnitude form by shear significance, X_eff = W (u.s_hat) +
  (1-W)|u| with W = Y^2/(Y^2 + X_mag^2) = S_hat^2 (and Z likewise).
  Strong shear (attached BL, bubble shear layer, reversed-flow layer:
  the case the invariant form exists for) keeps the signed X; weak
  shear (stagnation, irrotational noise) falls back to the magnitude
  form whose clipping is the guard. No new constants.
- Running now: one-step ON/OFF restart pair from the identical
  converged control state to localize the production difference
  field-by-field before implementing v2.

## 2026-07-25 ~20:10 UTC — Workshop envelope IS IN THE PAPER (fig:nlfworkshop)
Per your go: new figure after fig:nlfaft — the min-max envelope of
all ~14 workshop submittals (transition-vs-alpha, both surfaces,
digitized from Coder's summary deck p.22 with verification overlay
committed), the LTPT points recoverable from under the overlay
clusters, Piotrowski-Zingg's vector SA-LM2015 sweep, and SA-AI's
markers. The story it tells: the envelope is 0.4-0.6 chord WIDE at
fixed incidence (the workshop's headline scatter), and the untuned
SA-AI fronts sit inside it, on the experiment/LM cluster, at every
incidence — including the alpha=-4 lower-surface front, which looked
aggressive against the experiment alone but is within the community
scatter. Two new bib entries (coder_tmpw_summary,
piotrowski_zingg_2021); the three workshop PDFs committed to
references/. Commit c0f3d8f (88 pages, clean build).

## 2026-07-25 ~20:15 UTC — Vector-calculus kernel: the PAIR (division-free) form
Your reformulation adopted, and it is strictly better conditioned.
With m = omega x d_vec (d_vec = d n_hat) and l = (d.d)/2 * lap(u):
  A = u . m   (= [u, omega, d_vec], a signed scalar triple product)
  B = m . m
  C = l . m
  S_hat = B/sqrt(A^2+B^2),  g = (B-A-C)/sqrt(A^2+B^2+C^2),  P = S_hat g.
Divisions only by positive-definite RMS norms of the pair triple; no
unit vector is ever manufactured (the old code's s_hat needed
1e-30-guarded normalization and fabricated a noise DIRECTION at
omega -> 0; here A,B,C all vanish smoothly and P -> 0). Algebraically
identical to the s_hat form wherever omega x d != 0 (the tail is
homogeneous degree 0 — same projective class), so the LE sign-flutter
pathology is a property of the class, not the divisions; the guard is
the shear-significance blend, one line in pair language:
  A_eff = W A + (1-W) |u| sqrt(B),  W = B/(B + u.u).
Plan: implement pair form + W-guard as AI_INVARIANT_KERNEL=2 after
the one-step ON/OFF localization (running) confirms the mechanism.

## 2026-07-25 ~21:00 UTC — Vector kernel: YOUR wall-parallel Gram form agreed and implemented
Final form (AI_INVARIANT_KERNEL=2), exactly as you proposed:
  m = omega x d_vec (automatically wall-parallel),
  u_par = u - (u.n)n,  l_par = (d^2/2)[lap u - (n.lap u)n],
  P = (<m,m> - <m,u_par> - <m,l_par>) /
      sqrt((<m,m>+<u_par,u_par>) (<m,m>+<u_par,u_par>+<l_par,l_par>)).
Why it wins over both my earlier drafts:
- EXACT on parallel layers: reduces to Shat*g when the three vectors
  are collinear — verified on the frozen NLF attached profile to
  1.4e-6.
- Cauchy-Schwarz makes any misalignment only SHRINK P, which kills
  both failure sites the one-step localization found (near-wake
  n-to-TE orthogonality; aft-BL-edge omega sign flutter) with NO
  switch, NO W-blend, NO degenerate limit: the denominator is
  >= <u,u> wherever there is flow (your objection to W->0 as B->0 is
  structurally resolved).
- In the bubble the vectors are collinear, so the signed
  (antipodal-exact) behavior of the recirculation layer is retained;
  the pointwise bubble deviation from the MAGNITUDE kernel is the
  designed one, and the solver battery is the macroscopic test.
One-step localization that motivated the guard (for the record): the
s_hat form's instantaneous production difference concentrates at
x=1.0-1.25 (near-wake, wallDist 0.02-0.25) and the aft BL edge; the
LE was numerically ZERO at one step — the 40k LE chi elevation was
downstream contamination feeding back, not a local source.
Battery running (4 runs: str/cav NLF a4 + Eppler a5 with the new form
vs the existing controls + a fresh Eppler control). Verdict = ON ==
control within line widths, the bar the s_hat form failed.

## 2026-07-25 ~22:30 UTC — Gram form (projected) verdict: FAILS WORSE; diagnosis + repair proposal
Battery result (vs held controls; s_hat form had failed +12.4/+22.3/
+10.8 counts):
  str NLF a4:  control 0.9796/0.00741 -> gram 0.9644/0.00882 (+14.2 ct)
  cav NLF a4:  control 0.9839/0.00771 -> gram 0.9539/0.01057 (+28.5 ct)
  Eppler a5:   control 0.9171/0.01598 -> gram 0.9050/0.01834 (+23.5 ct)
Diagnosis (chi maps, systematic not flutter): the STAGNATION UNDERSIDE
carries chi at 94x seed and the LE at 17x (controls: exactly seed).
Cause: projecting u to the wall-parallel plane removes the advection-
speed guard precisely at impingement points, where u is wall-normal
and u_par ~ 0 -> denominator collapses -> P inflates. The amplification
steadying must come from the FULL advection speed (the same physics as
the advection-speed discussion in the LSB thread).
PROPOSED REPAIR (awaiting the author's agreement): drop the
projections entirely — they are automatic in the numerator (m is
perpendicular to n, so <m,u> = <m,u_par>) and harmful in the
denominator:
  P = (<m,m> - <m,u> - <m,l>) /
      sqrt((<m,m>+<u,u>)(<m,m>+<u,u>+<l,l>)),
  m = omega x d_vec,  l = (d.d)/2 lap(u).
Exact on parallel layers (u_n = 0 there); crossflow wedge H untouched;
wake/BL-edge suppression retained; impingement guarded by the full
<u,u>. Pure dot products of unmodified vectors — simpler than every
prior variant.

## 2026-07-25 ~23:30 UTC — Unprojected Gram verdict: BOTH variants fail identically; root cause is signedness itself
Results (vs controls; ct = dCD counts):
  mode 2 (all raw):    nlf_str +13.7, nlf_cav +23.6, epp +23.8
  mode 3 (l projected): identical to mode 2 within 0.2 ct
  (projected form was +14.2/+28.5/+23.5; s_hat form +12.4/+22.3/+10.8)
So the projection was never the driver. Field forensics on mode 2:
BL edge and outer bands CLEAN (exactly seed — my edge hypothesis was
wrong for this variant); the contamination is the LE (14x seed) and
the STAGNATION UNDERSIDE (88x seed) — despite the full <u,u> guard,
because at a stagnation region the full velocity itself vanishes.
Root cause common to ALL signed variants: in weak-shear regions the
signed <m,u> flutters with the omega-noise sign, clip<P>_0 rectifies
the positive half, and the onset gate passes wherever d^2|omega|/nu
is large. The magnitude kernel is immune BY CONSTRUCTION (X=+|u|
always -> g<0 in weak shear -> clipped). Signedness is the leak, not
the normalization.
Remaining options: (1) the shear-significance blend W=B/(B+uu) —
signed where shear dominates advection (backflow), magnitude where
advection dominates (every leak site); engineered for exactly this
discriminator, never run; zero new constants. (2) accept the
magnitude kernel as final (expressible division-free in Gram
clothing; the crossflow odd-pairing H story is an ADDITION and
survives either way). Proposed: run (1) as mode 4 to close the
question with data. Awaiting the author's go.

## 2026-07-26 ~00:15 UTC — The magnitude kernel in division-free Gram clothing (derivation)
Homogeneity: Shat and g are degree-0 in (X,Y,Z); scale the triple by
X = |u| (positive) instead of Y:
  (X,Y,Z) ~ (X^2, XY, XZ) = (<u,u>, sqrt(<u,u><m,m>), <l,u>)
— the division hidden in Z = <l,u>/|u| is cancelled EXACTLY by the
scaling. The calibrated magnitude kernel is then
  Shat = sqrt(<u,u><m,m>) / sqrt(<u,u>^2 + <u,u><m,m>)
  g    = (sqrt(<u,u><m,m>) - <u,u> - <l,u>)
         / sqrt(<u,u>^2 + <u,u><m,m> + <l,u>^2)
with m = omega x d_vec, l = (d.d)/2 lap(u): two Gram pairings, one
square root of their product, RMS denominators, no unit vectors, and
P -> 0 smoothly in quiescent flow.
Structure worth putting in the paper:
1. DUALITY: signed form = pairings against m (<m,u>,<m,m>,<m,l>);
   magnitude form = pairings against u (<u,u>, sqrt(<u,u><m,m>),
   <u,l>). Same three vectors, different reference leg; coincident on
   parallel layers (u || m) — matching every frozen-field check.
2. CAUCHY-SCHWARZ IS THE BATTERY VERDICT IN ONE LINE: the magnitude
   form's middle entry is the C-S ENVELOPE of the signed form's
   <m,u>. The envelope never flips sign with omega-noise -> weak
   shear always lands on g<0 and clips; every signed variant failed
   because it kept the pairing, the magnitude kernel survives because
   it keeps the envelope. (The W-blend = interpolating between a
   pairing and its own envelope.)
3. Y-CHOICE via the Lagrange identity: |omega x d|^2 =
   <omega,omega><d,d> - <omega,d>^2 — the canon full-|omega| Y is
   also division-free (Y^2 = <d,d><omega,omega>), the tangential
   choice differs only by the wall-normal-vorticity pairing
   <omega,d>^2 (identical in 2D, i.e., the whole calibration domain),
   and that difference term sits naturally next to the crossflow odd
   pairing H = n.(u x m) in the same algebra.
So the paper can present the UNCHANGED calibrated kernel in this form
— same numbers, no recalibration — with <m,u> (signed variant, failed
verification) and H (crossflow channel) documented as the algebra's
other two readings. W-blend run still on offer to data-close the
signed option.

## 2026-07-26 ~01:00 UTC — GRAM FORM ADOPTED (magnitude kernel); campaign resumed; what changed where
Adoption executed (commit b8cac0e):
1. REUSE: everything already computed remains valid — the Gram form
   is algebraically identical to the calibrated magnitude kernel
   (degree-0 homogeneity, rescale by |u|), so the old canon, the fv1
   batteries, and the 20 completed flow360_fv1 cases all carry over
   unchanged. No solver code change: the implemented (X,Y,Z)
   arithmetic IS the Gram form after the rescale.
2. Y-CONVENTION pinned: full-vorticity Y^2 = <d,d><omega,omega>
   (identical to canon everywhere incl. 3D; also division-free). The
   tangential |omega x d| variant differs only by the wall-normal-
   vorticity pairing (Lagrange identity) and is documented as the
   3D-future option next to the crossflow pairing H.
3. PAPER (Sec. III solver-realization passage, new Eq. eq:gram): the
   triple is now stated as (<u,u>, sqrt(<u,u><d,d><omega,omega>),
   <l,u>) with the four vectors defined; parallel-layer reduction
   spelled out; the signed pairing <omega x d, u> recorded WITH its
   verdict (implemented in four variants, rejected — rectifies
   vorticity-direction noise in weak shear; its C-S envelope IS the
   middle Gram entry); the odd crossflow pairing pointed at the
   outlook. The existing signed-vs-magnitude realization paragraph
   (mixed-sign layer discussion) stands unchanged — it was already
   exactly right.
4. ARTIFACT memo updated + republished (same URL): kernel card now
   'adopted' with the full verdict trail.
5. CAMPAIGN resumed with a skip-completed guard (ai_constants.log as
   the completion marker): the 20 finished cases skipped in seconds,
   remaining ~73 running on all 8 GPUs.
Sections to read: Sec. III paragraph beginning 'In the solver the
indicator triple is assembled from four local vector-calculus
objects' + Eq. (eq:gram) (one page); the RESPONSES ~00:15 entry for
the derivation; the artifact kernel card for the memo version.

## 2026-07-26 ~02:00 UTC — Review passes 26 (remainder) + 27 all addressed (commit 0f6c17a)
Headlines (full detail in agent-paper-review/2026-07-25-1613-response.md):
- fig:nlfworkshop: the 9-deg cluster is DROPPED (it sat beyond the
  digitized envelope's -4..8 alpha domain — the caption's "at every
  incidence" could not be supported there); caption states the domain,
  the manuscript-vs-journal Piotrowski provenance, and the c_l->alpha
  replot; band recolored gray so the O-grid markers cannot hide.
- The promised analytic q-margin note is DELIVERED:
  repro/analytic/fv1_qmargin_composite.py (Reichardt + Coles): q < 2
  through the entire gated band for wake parameter Pi <= 3.5 at any
  delta+ >= 100; near-separation wakes (Pi 5-7, low delta+) reach
  q ~ 2.3-2.8 and open G partially — the II.F sentence is scoped to
  exactly this envelope, and the <=1.5-count consequence bound now
  traces to a committed dataset (data/fv1bypass_battery_results.json,
  16 slots x on/off).
- The fv1 bypass weight is renamed b (w stays the onset ramp width);
  Appendix E counts FOUR departures from baseline SA, the recovery
  recipe includes b==0, and s, G, and Reichardt's U+ (cited) are
  restated in the assembled model.
- fig:nlfcfhigh's alpha=9/15 dash-dot overlay correctly attributed to
  XFOIL (with the x ue^2 conversion stated); N envelope remains
  FlexFoil's. A global Cf-convention disclosure now lives in the
  Appendix-A conventions paragraph.
- Status sentence made pause-proof; note the campaign is in fact
  RESUMED (skip guard) since the Gram adoption.

## 2026-07-25 — E387 literature scour for comparison figures (agent)
Downloaded to references/ (4 best, all free):
1. frere2016-e387-ILES-DG-eN-RANS-Re60k.pdf — Frere et al., J.Phys.Conf.Ser 753:022037 (2016), open access. ILES(DG, effectively DNS) + EllipSys2D RANS coupled e^N (Ncrit 7/9) + XFOIL, E387 Re=6e4. Fig 2 CL/CD vs alpha vs 4 experiments (LTPT/Delft/Princeton/Stuttgart); Figs 4-5 Cp AND Cf at alpha=4,8 deg. Covers coupled RANS-e^N + high-fidelity anchor at the bursting Re.
2. carreno-ruiz2022-gammaRetheta-STARCCM-e387-Re60k.pdf — Carreno Ruiz & D'Ambrosio, FTaC 109:279-308 (2022), open access. gamma-Re_theta in STAR-CCM+ (Menter vs Suluksna-Juntasaro correlations, s1 tuning); E387 Fig 19 CL+CD vs alpha at Re=6e4.
3. ghimire2025-openfoam-gammaSST-e387-Re300k.pdf — Ghimire/Ni/Wang, Fluids 10:230 (2025), CC-BY. OpenFOAM gamma-Retheta-SST vs gamma-SST vs Kgamma-SST; E387 Re=3e5 Tu=0.1% alpha=2/4/6: Cp (Figs 3/5/7) + Tables 6-8 of LS/TR/transition x/c vs McGhee (tables — no digitization needed). IDDES at Re=1e5 alpha=11 (Fig 10 Cp).
4. ijsrp2019-fluent-kklw-transSST-e387-polars-Re20k-300k.pdf — Anushka R et al., IJSRP 9(1) 2019. Fluent k-kL-omega + trans-SST polars (Cl-a, Cd-a, Cl-Cd) at Re=2e4/3e4/6e4/1e5/3e5 vs McGhee. Only free k-kL-omega E387 dataset spanning our sweep; low-tier journal, raster.
Not downloaded (ranked): Lopes/Eca/Vaz NuTTS'18 "Assessment of RANS Transition Models" (SST+gamma-Retheta/gamma/AFT + kT-kL-omega Cp at Re=3e5 alpha=1, free: uni-due.de/imperia/md/content/ist/nutts_21_2018_cortona.pdf pp~118-123); Brunelli et al. FTaC 116:27 (2026) VMS-iLES Cp Re=3e5 alpha=1 (open, link.springer.com/content/pdf/10.1007/s10494-025-00727-7.pdf); Sahin/Hall/Mohseni AIAA 2008-0422 DNS Re=6e4-4.6e5 (paywalled, exact sweep match); Catalano&DeRosa AIAA 2020-2990 LES Re~1e5 (paywalled); Salimipour C&F 181:67 (2019) k-kL-omega mod E387 Re 3e4-1.3e5 (paywalled). Bonus experiment: Cole&Mueller 1990 (NTRS) LSB measurements on E387. Checked: NASA LAVA-AFT 2018 has NO E387; arXiv 2508.02547 = D'Alessandro (already held).

## 2026-07-25 16:57 UTC — Eppler literature: Cole–Mueller fetched + proposed insertions

Addendum to the agent's ranked list above. Fetched the bonus experimental source from NTRS:
`references/cole-mueller1990-e387-LSB-LDV-Re100k-NTRS19900006064.pdf` (148 pp., Notre Dame
UNDAS/NASA CR, Jan 1990). It is LDV boundary-layer velocimetry inside the E387 bubble at
**Re = 1e5** (primary case alpha = 2 deg), and its Table 4.1 tabulates laminar-separation /
transition / turbulent-reattachment stations vs alpha — a SECOND experimental bubble-station
set at exactly the Reynolds number where the model's bursting boundary sits (fig:eppbubble
currently carries only the TM-4062 oil-flow stations at 2e5). 1990 raster scan: the table
needs visual reading (one page), not digitization software.

Proposed insertions, in value order (awaiting user pick):
(a) Frere 2016 -> fig:eppresweepforces: the 6e4 literature cluster (ILES ~ DNS-grade, coupled
    RANS-e^N N=7/9, XFOIL N=7/9, four experiments) at the Re where our refinement fan opens —
    lets the bursting-boundary section place truth between us, e^9, and the tunnel.
(b) Ghimire 2025 Tables 6-8 -> fig:eppbubble at 3e5 (three gamma-variants, tabulated, no
    digitization risk).
(c) Carreno Ruiz 2022 (gamma-Retheta at 6e4) + IJSRP k-kL-omega polars -> resweep-forces
    columns at 6e4/1e5/3e5.
(d) Cole-Mueller Table 4.1 -> fig:eppbubble at 1e5 (second experimental set, at the failing Re).

Campaign status at this writing: nlf set on its final four L2 high-alpha batches (a9/a15 both
families); eppler/sweep/extension sets follow; skip-guard working (20 SKIPs logged).

## 2026-07-25 — E387 polar digitization (2 reference PDFs)
- paper/data/carreno2022_e387_re60k_polar.json — Carreno Ruiz 2022 Fig.19 (Re=60k). Legend has ONE gamma-Re_theta curve (not Menter-vs-SJ split; text implies SJ s1=6 recalibration ~= Menter), plus Experimental(McGhee), 2D NS(Sahin), 3D DNS(Sahin), 3D LES(Frere). Raster-embedded MATLAB fig; color-cluster digitization, uncertainty ~+/-0.005 CL, +/-0.0005 CD.
- paper/data/ijsrp2019_e387_polars.json — IJSRP 2019 (Fluent). Re=60k: kklw (-8..10) + trans-SST (1..11) + Expt-as-plotted; Re=100k/300k: trans-SST ONLY (paper never ran kklw there; kklw=null). Markers are per-point image XObjects with vector bboxes + born-digital tick text => vector precision (+/-0.02 CL worst case). Paper's "EXP" does NOT faithfully duplicate McGhee TM-4062 (60k Cl wildly low; Cd high at 100k) — kept as-plotted, use real TM-4062 tables for comparisons.
- Checks: paper/data/digitize_check_carreno2022.png, digitize_check_ijsrp2019.png.
- alpha=5: carreno gRT CL 0.828/CD 0.0334; ijsrp 60k kklw 0.871/0.0471 (interp), transSST 0.029/0.0247 (as-plotted!); 100k transSST 0.834/0.0205 (interp); 300k transSST 0.879/0.0129 (interp).

## 2026-07-25 18:12 UTC — Annotated round 7: 13 of 16 remarks addressed (commit a5eff9d)

Done:
- p23: Sec III keeps only the four-vector summary + pointer; Appendix E now states the
  realization Gram-first (eq:gram moved there, = Eq. E4), followed by the signed-algebra
  correspondence and the two other pairings. Ratio-form disclosure retained.
- p35: "the sphere kernel" purged (1 site, -> "the model"); P-for-Shat-g purged (2 sites:
  benchmark prose "(Shat g ~ 1)", Appendix D caption "Re_Omega^c(Shat g)"); struck the
  inviscid-pressure clause, "The amplification is on time", "deliberately".
- p36: tab:epplit deleted with its pointer paragraph (fig:epppolar + fig:eppbubble carry it);
  SA-BC calibration sentence went with the caption.
- p37: tab:eppresweep moved to Appendix C; figure is now the primary (2 rows x 3+2 panels:
  cl/cd/cm; separation/reattachment). "independently"->"separately" for FlexFoil (5 sites).
  The 1e5-failure prose awaits the fv1 recomputation (campaign running).
- p39: sec:eppbistab dissolved to ONE paragraph (tab:eppfork deleted; fork agreement numbers
  inline); hysteresis now only pointed at the 60k evidence. Labels kept inline so existing
  Sec-refs resolve to the enclosing Reynolds section.
- p40: caption overflow fixed by a pixel audit (text blocks vs page-number band): Fig. 16
  caption trimmed + image 0.92\textwidth; Fig. C2 image 0.82\textwidth. Zero pages flagged.
- p42: sec:epphandover dissolved: one completing paragraph (handover-length numbers inline,
  table deleted), one trimmed re-seeding paragraph, cancellation paragraph untouched.

Literature insertions (user: "do all of them into the figures, skip discussion"):
- fig:eppresweepforces rebuilt 2-row full-width: top cl/cd/cm with Carreno gamma-Retheta
  (0.828/0.0334 at 6e4), IJSRP k-kL-omega (6e4) and transition-SST (1e5, 3e5); bottom
  sep/reattach with oil flow (1/2/3e5), Cole-Mueller alpha=4/6 bracket at 1e5, Ghimire's
  three gamma variants at 3e5. EXCLUDED: IJSRP 6e4 transition-SST + its "experiment"
  (corrupt as plotted: CL(5deg)=0.03, Cd spikes; recorded in JSON meta). Frere markers
  drop in automatically when its digitization agent lands.
- fig:eppbubble gains Cole-Mueller's SECOND experimental set at 2e5 (Figs 4.37-4.39 of the
  1990 report digitized; also 1e5 + 3e5 in the JSON). REVISION (not append): their
  reattachment reads 0.03-0.08c aft of the LTPT oil flow at matched alpha, so the
  "systematically late" sentence is now scoped to the oil flow with the facility-to-facility
  spread stated, and the computed low-incidence reattachments noted to fall within it.

Flat-plate section revised to NEW canon (user directive; plates ran ahead of queue, all 5
DONE): chi=1 onset bracket tightens 14% -> 9% (+8% at Tu=0.04, within 2% at 0.08/0.60,
6-8% early at 0.16-0.30); early-bias attribution moved from "higher Tu" to mid-range;
figure regenerated from flow360_fv1; conclusion updated to match. c_v1-crossing statement
(Re_theta ~ 2500) unchanged.

Pending: p26 workshop all-submittal overlay (digitization agent running) + same treatment
for the drag bucket; Frere 6e4 cluster. Campaign: nlf set on final 4 L2 batches at last
check; standing orders on completion = verify nlf_neg/eppler_ext, then Daedalus L0+L1
new-canon (hold L2).

## 2026-07-25 19:05 UTC — Round 7 complete (commits a5eff9d, 1e86d17, 44e0a83)

Final two items landed:
- Frere 6e4 cluster on fig:eppresweepforces: coupled RANS-e^N N=7/9 at exact alpha=5
  (0.728/0.0602, 0.650/0.0602), ILES at alpha=4 only (0.768/0.0431; its alpha=4/8 pair
  straddles the reattachment jump so an alpha=5 interpolation is meaningless), plus Delft
  (0.590) and Stuttgart (0.732) alpha=5 lift points. Notable: the model's burst-branch
  lift (0.62-0.70) sits amid the community/other-facility values at the bistable
  condition; the LTPT upper branch (0.838) is the high edge of the spread.
- p26 workshop rework: fig:nlfworkshop DELETED; all 14 submittals digitized per-submittal
  (19 transition series, 267 pts, zero violations of the earlier envelope; 18 polar
  series) and underlaid thin on fig:nlfaft (AFT purple, algebraic B-C green, others light
  gray; placed on the lift axis via the computed strL2 lift curve since the deck's polar
  has no per-point alpha — disclosed in caption) and on fig:nlfpolar (native cl-cd).
  Piotrowski-Zingg SA-LM2015 kept at its native cl ordinate. The B-C submittal is exposed
  as the near-LE outlier on both surfaces; the workshop AFT trio tracks the recalibrated
  dissertation AFT and the SA-AI markers.

Build clean, 87 pp., zero caption overflows. Remaining round-7 item ("get ready to revise
the 1e5-failure prose per fv1 results") is the campaign-migration task (#27).

## 2026-07-25 19:35 UTC — Review pass 29: all 3 majors + 7 minors implemented

Pass 29 independently re-digitized every literature source (all committed JSONs confirmed
faithful; Ghimire zero-mismatch; Carreno/Frere within 1-2 px) and verified the flat-plate
new-canon numbers exactly. Its one real catch: the alpha=5 oil-flow constants in the two
sweep scripts were partly TM-4062's alpha=4 row — at 3e5 the plotted reattachment sat
0.03c aft of the actual alpha=5 tabulation, flattering the fan-vs-band comparison.
Fixed to (0.34,0.67)/(0.38,0.59)/(0.39,0.55); three figures regenerated.

Also: II.F status sentence now records that Sec. III is already on the recomputed model;
SA-BC calibration caveat restored to fig:epppolar's caption; gamma-SA (D'Alessandro)
comparison restored as prose (matched-lift vs matched-lift); frere JSON provenance title
fixed; anushka bib completed (6 authors, pp. 204-219); flatplate regen script documents
the SAAI_CFD_ROOT=flow360_fv1 requirement; Cole-Mueller "matched incidence" wording made
exact; eight-vs-nine oil-flow incidences reconciled; helicity sentence scoped to
wall-parallel vorticity; phantomsection anchors + Table C1 numbering.

Full response: agent-paper-review/2026-07-25-1800-response.md. Build: 87 pp, 0 errors,
0 undefined, 0 overflows.
