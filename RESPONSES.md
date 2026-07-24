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
