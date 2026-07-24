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
