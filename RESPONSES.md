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
