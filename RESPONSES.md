# RESPONSES.md — paper agent log (role reversal, 2026-07-27)

*Cleared 2026-07-27 ~15:10 by user order at the role handover. Prior
history (3,476 lines) is in git (`git log -- RESPONSES.md`, last entry
830ceee). From here on: the former reviewer agent (this log's new
author) is the PAPER AGENT — editing the paper, responding to reviews
in `agent-paper-review/`, and mirroring responses to the user's prompts
here in appending style.*

---

## 2026-07-27 ~15:10 — response to the role-reversal order

Acknowledged. Actions taken with this entry:

1. **RESPONSES.md cleared** (this file); prior content preserved in git
   history.
2. **Awaiting the outgoing agent's handover document** — I will not
   duplicate or preempt its in-flight work (known in flight: the
   stagnation-bisection re-emit with verdict-based logic in
   `runlogs/stag_rebisect.log`; the Stock Figs 2–5 digitization /
   spheroid-section overhaul; the just-lifted cavity-L2 hold and
   paused spheroid runs per 830ceee). On receipt I will confirm the
   split of responsibilities: which CFD the outgoing agent continues to
   babysit vs. what transfers to me.
3. **My review loop reverses role**: the standing watch on
   `agent-paper-review/` now triggers me to RESPOND to incoming review
   files (fix the paper, write the `-response.md`) rather than to write
   reviews. The verification standards I enforced as reviewer (verify
   numbers before printing, move-don't-paraphrase, honest disclosures,
   independent digitization, repro hygiene, length/organization
   discipline) remain the house rules I will now be held to.
4. **Known owed items I inherit unless the handover says otherwise**:
   the bisection re-emit completion (update Sec. VII interval + every
   2.8e5-leaning sentence; regenerate JSON/figure), the spheroid
   overlay integration under the digitization gate, the pass-46 cut
   plan's remaining user decisions (O1 supplemental, Sec VIII scoping),
   the Eppler OpenFOAM L2 adoption decision when that ladder lands, and
   the repro-refactor batch.

## 2026-07-27 ~15:10 — handover received; taking up task B (spheroid overlays)

Read `HANDOVER-paper-revision.md` in full. State check at takeover:
- **D (critical-Re re-emit)**: rebisect RUNNING (log at 15:00: L=537.5
  endpoint confirmed collapsed in 2 chunks; L=700 on chunk 2, maxchi
  8.1 and still moving). Merge + the two tex numbers stay with the CFD
  agent per the handover; I will not touch sec:bistability numbers
  until their completion note appears here.
- **A (cavity-L2)**: a6 running, harvest ~17:40; my tex work queues on
  the CFD agent's completion note.
- **C (OF Eppler L2)**: tomorrow morning.
- **Starting now — B**: Stock Figs 2–3 (Cp waterfalls, 13 stations,
  dCp=0.42/station) digitization via the existing
  `digitize_stock_waterfalls.py` pipeline (tick-detected calibrations,
  measured not assumed), then measured-symbol overlays on our
  waterfalls via `regen_spheroid_station_profiles.py`, then the
  main-vs-appendix split per the user's rule ("any figure without an
  experimental comparison on top of it belongs to the appendix") —
  split proposal will be posted here for sign-off per the handover's
  pending-decisions note.
All hard rules acknowledged; the review-thread watch continues (my
monitor is alive and now serves as the response trigger).

---

## 2026-07-27 — Task B step 1-2: Stock Figs 2-3 digitized; measured overlays on all four spheroid waterfalls (implementation agent)

- **Digitized Stock Figs 2-3** (Cp waterfalls, 13 stations each) with the
  existing tracker pipeline: tick-detected calibrations (fig2: phi
  250.5→1655 px for 0→180°, -Cp 28.5→2402.5 px for 6→-1, 339.1 px/unit;
  fig3: 236→1495.5, 25→2153, 304.0 px/unit). New pipeline pieces for the
  Cp figures: blob vertical-extent floor (symbols ~20 px vs 3-5 px
  potential-theory strokes/grid dashes — the thick/thin roles are
  inverted vs Figs 4-5), in-panel legend erase boxes, a stepped erase
  along the dashed free-vortex curve (it hijacked the 0.212 tracker),
  and manual seed rows for fig3 (the -0.9852/-0.9988 chains overlap at
  the left edge). Chain identity verified station-by-station at full
  resolution (per-station check PNGs in
  /local_data/qiqi/sa-ai/stock_digitize/). fig2: all 13 chains clean
  0-180°. fig3: seven chains truncated in the leeward dive tangle
  (0.792@92°, 0.669@94°, 0.462@101°, 0.212@100°, -0.038@108°,
  -0.288@108°, -0.865@122°); six clean over the full range.
- **FINDING — the printed Cp waterfalls carry a +0.14 base displacement
  on top of the 0.42/station step.** Exact potential theory on the 6:1
  spheroid (matches our L2 RANS station Cp to ~0.005 away from
  nose/tail) sits a uniform 0.12-0.15 below the as-printed chains at
  BOTH incidences, and below Stock's own printed potential-theory
  curves by 0.15. Stock's stated "0.14" is evidently the base
  displacement, not a typo of the step (pass-43's 0.42 step stands;
  its "0.14 = 3x typo" interpretation is revised). Digitization removes
  0.14+0.42i; after removal, windward measured-minus-potential is
  -0.01 +/- 0.015 at both incidences. Figs 4-5 carry no base shift
  (gamma_w=0 symmetry anchors at phi=0 confirm).
- **Overlays**: all four Appendix-H waterfalls regenerated with the
  measured chains as open red circles (Cp from figs 2-3, Cft+gamma_w
  from the committed figs 4-5 JSONs). Character: windward Cp and
  gamma_w sit on the computed curves at both incidences; at alpha=10
  the measured Cft transition rise (phi~120-150, aft stations) has no
  computed counterpart — the amplification deficit in the campaign's
  own presentation; at 29.7 the computed fronts track the measured
  dives. Captions + Appendix F provenance entries added (all four
  JSONs). pdflatex x2: 0 errors, 0 undefined, 116 pp.
- **Main-vs-appendix split proposal (NOT executed, per sign-off rule)**:
  the four waterfalls now carry experimental comparisons and qualify
  for the main section. Propose promoting the two Cft/gamma_w
  waterfalls (H2, H4) into Sec. IX — they carry the transition story;
  keep the two Cp waterfalls in Appendix H as supporting validation
  (they qualify; taste call). Per the rule, the three map figures in
  Sec. IX (spheroid_maps_L2, _a29p7_L2, _L2_3d) carry no experimental
  overlay and would move to the appendix — alternative: overlay the
  already-digitized Fig. 15a measured transition points on
  spheroid_maps_L2 so the alpha=10 map can stay.

---

## 2026-07-27 ~15:35 — review-folder audit + response to 1540 (drag-crisis feasibility)

- **Audit (user request): every review through the handover cutoff is
  answered.** The apparent gaps in a filename scan are naming
  conventions (multi-pass responses: 1050 covered passes 3+4, 0930
  covered 10–12, 1130 covered 1020+1031+1130), absorptions (pass 49 and
  the 1315 memo inside the 1250-response addendum), and commit-resolved
  items (pass 17 via 3220fc2; the 2340 follow-up via 060f7c1). The
  cleanup record needs no response. One genuinely open file:
  2026-07-27-1540 (arrived post-cutoff) — now answered
  (1540-response.md).
- **1540 response summary**: the cylinder drag-crisis campaign design is
  sound and matches the outlook paragraph the paper already carries;
  branch protocol (up/down continuation + cold reference) is the right
  instrument; the quasi-2D Cd caveat is correctly front-loaded; risk 1
  (kernel indicator semantics at wake distances) is the real gate.
  **Decision for you**: (a) go/no-go on the cheap pilot pair — the
  one-afternoon frozen-field kernel check + one Re=1e5 quasi-2D URANS
  case — runnable in queue gaps; (b) the full ~60–80-case matrix is an
  Eppler-scale GPU commitment, and (c) whether any results would enter
  this paper or the follow-on. No draft changes made; all [mem] targets
  must be source-verified before any number is quoted.
- Also pending your sign-off (from the Task B entry above): the spheroid
  main-vs-appendix figure split proposal.

---

## 2026-07-27 ~16:10 — pass-46 remainder executed: D9 seed pointers + Sec V language sweep (99a58b5)

- **D9 remainder (both sites)**: the in-text seed restatements are now
  one-clause pointers — Sec V "the same N_crit=9, LTPT-class anchor as
  the NLF (Sec. IV)" and Sec VI "the paper-wide freestream seed
  (Sec. IV)". The value 8.76e-4 lives at its Sec IV derivation plus the
  two condition-stating captions (nlfcfhigh, eppcflow) only.
- **L2 sweep, Sec V only** (5 sites): (1) mesh paragraph — the
  spline/clustering description and "every length scale halving per
  level" deferred to the Sec IV recipe home ("reuses the NLF recipe
  verbatim" already opens the paragraph); the LE-and-TE clustering
  qualifier MOVED into Sec IV's recipe parenthetical so nothing is
  orphaned. (2) Skin-friction paragraph — the 7-deg three-of-six-grids
  collapse telling deleted (homes: eppcfhigh caption + bubble-stations
  paragraph); its unique number, the 0.014c sliver, MOVED verbatim into
  the bubble-stations paragraph. (3) Rate-coordinate paragraph — final
  clause restating its own sentence deleted. (4) "the ninth, alpha=8"
  parenthetical deleted (caption home keeps the 0.32c fact). (5) The
  lower-surface-laminar third telling deleted (full copy: eppcfhigh
  caption; short copy: eppbubble caption); "That low-alpha margin"
  added to keep the Cole-Mueller 0.03-0.08c comparison pinned to the
  0.02-0.04c over-length, not the 7-deg 0.07c.
- **Deleted vs moved**: ~5 sentences/clauses deleted (all with
  surviving full-strength homes), 2 moved (0.014c, LE/TE qualifier),
  0 paraphrased. Honesty disclosures untouched (matched-lift clause,
  canceling-errors class all intact).
- **Numeric-token diff vs pre-edit HEAD**: removals = the two 8.76e-4
  restatements, the alpha=8 parenthetical, and the "L2" absorbed by
  "structured-L2 sliver"; additions = none.
- **Build**: pdflatex x2, 0 errors, 0 undefined, 116 pp -> 116 pp (the
  ~11 saved lines reflowed inside the figure-dominated section; no page
  boundary moved). Not touched per the task fence: sec:bistability,
  tab:daetotals/daggers, Sec IX / Appendix H.

---

## 2026-07-27 ~17:00 — user go-ahead: drag-crisis pilot pair on 017-v100-dev

User approved the cheap pilot pair from the 1540 response, on
017-v100-dev (free GPUs), with the constraint: ONLY cheap runs
remotely. Plan being executed:
1. Cylinder contour + quasi-2D O-grid via the existing airfoil pipeline
   (local, no GPU): D=1 circle, L1-class sizing per the 1540 design
   (~1200 x ~150, y+<=1 at 1e5 with margin), one-cell span.
2. ONE subcritical URANS case at Re_D=1e5 on 017-v100-dev (fv1 canon,
   AI_LAMINAR_SLOWDOWN=0.01 with pre-compensated seed, two-stage
   startup per Sec. VII protocol, dt ~ 0.01 D/U, >=10 flow-throughs
   transient + >=30 shedding periods averaging). Deliverables: cost
   calibration, mean Cd / St vs the subcritical experimental band
   (Cd 1.2 +/- 0.05, St 0.19-0.20; the known 2D-URANS high bias
   expected), separation angles from Cf.
3. Frozen-field kernel check on the converged snapshot: evaluate the
   indicator triple / P = Omega_hat*I_hat at wake distances d ~ O(D)
   (risk 1 of the 1540 note) — does the kernel transition the separated
   wake shear layers for the right reason, and do the wall-distance
   semantics degrade?
Nothing else runs remotely; the full matrix remains ungated pending
pilot results.

---

## 2026-07-27 ~17:15 — response to "why go unsteady at all?" (drag-crisis pilot)

Your point wins for this campaign's actual claims. Decomposition:
- The classical URANS-on-bluff-body justification is that shedding
  shapes the MEAN wake (steady forced solutions give too-long
  recirculation, wrong base suction, subcritical Cd off the other way).
- But the campaign already concedes the wake-dynamics-controlled
  quantities (subcritical absolute Cd, rms CL, St amplitudes) to the
  quasi-2D bias. The surviving claims — separation angles, Cf/Cp,
  Re_crit(Tu), supercritical recovery — are BL-controlled, exactly
  where this project has found steady pseudo-time medians match
  unsteady means. The branch structure (one-bubble state, hysteresis)
  is MEAN-state structure the steady continuation ladders probe more
  cleanly than time marching. Steady is ~10x cheaper.
- Remaining risk: steady non-convergence on the shedding mode, or the
  too-long steady recirculation feeding back (via base suction) into
  the shoulder Cp enough to move separation. Empirical — so the pilot
  now answers it: the running URANS case gets a STEADY twin (same
  mesh/canon, campaign driver; minutes of GPU), and the report compares
  steady-median vs URANS-mean for Cd, separation angle, front location,
  shoulder Cp, plus the steady solve's convergence character.
- If they agree on the BL-controlled quantities: the matrix runs
  STEADY-first with the existing ladder machinery + 2-3 URANS spot
  checks. Pilot agent instructed accordingly; remote budget unchanged.

## 2026-07-27 18:05 — CFD agent: cavity-L2 harvest COMPLETE (successor: go)

All three cavity-L2 fv1 cases done + harvested (committed): a4
1.0217/0.01979, a5 1.1231/0.02226, a6 1.2230/0.02497 (drifts <=
3.4e-4). chi_surface.npz built for all three (17:54-17:59). HANDOVER
sec A is now unblocked: de-dagger, fill tab:daetotals, close the
held-solutions disclosures, regenerate the Daedalus figures (caches
are ready — no race), compute the finest-grid two-family claims
(L2 family gaps: CL 0.42/0.52/0.65%, CD 1.0/1.4/2.3 counts at
a4/a5/a6). Spheroid runs resumed on GPUs 0-3 (re65a10-L2 + re72a2p5
ladder). Stagnation re-bisection: one step from final; corrected
bracket will be ~(677, 689] in L (Re_r ~4.6e5) — I'll re-emit the
JSON/figure and the two sec:bistability numbers when it lands.


---

## 2026-07-27 ~18:40 — paper agent: cavity-L2 integrated; Daedalus de-daggered (HANDOVER A done)

Verified the three cavity-L2 fv1 cases myself before printing anything.
IMPORTANT — harvest-vs-canon discrepancy: the values in
`flow360_fv1/daedalus_fv1_results.json` (a4 1.0217/0.01979, a5
1.1231/0.02226, a6 1.2230/0.02497) are MEDIANS over roughly the last
4,000 pseudo-steps, not the table's stated convention (MEAN over the
final 500 pseudo-steps, 51 samples — which reproduces all nine
previously printed tab:daetotals entries exactly). At the canon window
the cavity-L2 cells are a4 1.0219/0.01976, a5 1.1236/0.02223, a6
1.2238/0.02491 (differences up to 8e-4 in CL / 0.6 counts; final-1000-step
CL peak-to-peak 4.4e-5/1.0e-4/1.8e-4 — converged, so the gap is pure
window convention). The PAPER prints the canon-window values. CFD agent:
please rebuild the JSON at the canon window when convenient so the two
agree.

Done on top of that:
- tab:daetotals dashes filled (canon values above); caption's dash/dagger
  legend removed; trim qualifier now "finest structured level" (matches
  avl_fixed_cl_reference.py).
- De-dagger sweep: II.F "Status" now closes with the 3D deployment
  recomputed in full; Sec VIII opening parenthetical reduced to one line;
  "Fifteen ... held" -> "Eighteen final-kernel RANS solutions"; dagger
  legends dropped from daepolar / daesurf4 captions; daesurf5/6 "row
  follows if the held recomputation lands" dropped; section-sheet
  "where its field output is available" dropped. KEPT (historical/
  diagnostic, per HANDOVER): Appendix D temporal-units ledger, the
  pass-44 like-for-like correction paragraph (15-26 counts / 2-11
  counts / bubble-shortening physics), tab:daemesh y+ provenance note,
  L1 predecessor gap comparisons.
- Two-family finest-grid claims (recomputed, fv1-vs-fv1): L2 totals agree
  to 0.35-0.63% in CL and 0.7-1.7 counts (0.34-0.69%) in CD at a6/a5/a4
  respectively (CL gap tightens, CD gap widens with incidence); bubble
  edges at eta=0.31 via canon_front_metrics.py on the fv1 root:
  separation within 0.008c and reattachment within 0.016c at every
  incidence (replaces the fv1-vs-predecessor 0.008c-at-4-deg placeholder).
- Figures regenerated from daedalus_fv1 (chi caches 17:54-17:59, no race):
  polar/sectional now carries the unstructured-L2 line at all three
  incidences (sectional panel used L2 for all six family/incidence
  combos); surface maps a4/a5/a6 all three rows (a5/a6 gained the
  unstructured-L2 row; a4's replaced the predecessor row); section
  sheets regenerated with fv1 dashed curves at all alphas.
- User directive (received mid-regen): D1-D3 section sheets restyled —
  canvas narrowed from 17.28x13 in to 11.5x12.5 in, exactly the dominant
  2D chi-sheet canvas (828x900 pt; both figure families sit at
  width=0.99\textwidth, so on-page heights now match exactly); suptitle
  and the per-column alpha titles removed (captions carry both). The
  probe results now persist to section_sheet_probes.pkl in each fv1 case
  dir, so future re-styles skip the ~2 h L2 volume probes. NOTE for the
  paper-wide title sweep: daedalus_surface_a4/5/6 still carry their
  suptitle + column titles (directive scoped to the section sheets).
- Build: pdflatex x2, 0 errors, 0 undefined; 116 pp -> 114 pp (the
  removed disclosure text let the [p] floats repack). Not touched per
  the fence: sec:bistability, Sec IX / App H spheroid figures,
  drag-crisis text.

---

## 2026-07-27 ~18:40 — USER CONSTRAINT: 4-GPU cap on all sa-ai work

User: another user (vishal) is using 4 GPUs; restrict our own work to
4 GPUs only. Interpretation and current state:
- Local machine: spheroid ladders on GPUs 0-3 (CFD agent) = exactly at
  the cap; GPUs 4-7 are vishal's (100% util, high mem) — do not touch.
  CFD AGENT: keep all future local scheduling within 4 GPUs total
  (0-3); no 8-GPU chains until the user lifts this.
- 017-v100-dev: pilot on GPU 0 only; same 4-GPU cap applies there
  (pilot agent instructed directly).

---

## 2026-07-27 ~19:05 — user directives batch (execution queued behind item A)

1. **Spheroid split AGREED — executing**: promote the two Cft/gamma_w
   waterfalls to Sec IX; Cp pair stays App H; overlay the digitized
   Fig-15a measured points on the alpha=10 map so it stays in Sec IX;
   the other two maps (a29p7, 3d) move to the appendix.
2. **Daedalus L0**: the data EXISTS and is complete (all six L0 cases,
   both families x a4/a5/a6, full 20k-step histories in daedalus_fv1)
   — it simply never entered tab:daetotals/figures (L1/L2 only).
   Adding L0 rows + ladder discussion with the L0 path-sensitivity
   caveat (the paper's own Sec IV finding) in the next tex round.
3. **Figs 20-22 (daesurf4/5/6) repetitive**: keep ONE in Sec VIII, move
   two to the appendix (choice by text-reference density; executing in
   the next round).
4. **Figures D1-D3**: narrow the plots so on-page height matches the 2D
   sheets — injected into the in-flight section-sheet regeneration.
5. **ALL in-figure titles removed paper-wide** (captions suffice) —
   layout-only regeneration batch in the next round.
6. **CFD AGENT — new runs requested by the user**: Eppler alpha=-2 and
   8.5 deg at L0 and L1, both families (8 quasi-2D cases), fv1 canon,
   converge-by-xtr protocol. Queue under the 4-GPU cap (behind or
   interleaved with the spheroid ladders at your discretion).

7. **D1-D3 sheets: add L1 and L0 curves** (user directive). Data
   availability verified: all six L0 and all six L1 cases complete in
   daedalus_fv1 for both families. Queued for round 2 (needs L0/L1 chi
   caches + one re-probe; rides with the title-strip regeneration so
   the sheets re-render once, not twice).

## 2026-07-27 19:25 — [CFD->successor] sec:bistability numbers corrected in working tree

Verdict-based re-bisection final: critical Re_r = 4.66-4.74e5
(L in (682.9, 688.5]). I edited THREE strings in your working-tree
sa-ai.tex (surgical, non-conflicting): the interval sentence (+ a
clause on the verdict method), the caption band value, and
"L <~ 530" -> 685. JSON + figure + rebisect machinery committed
separately; please include the tex lines in your next commit. Note
the seed-fed opening clause is now stronger (NLF Re_r ~ 7e4 is 6.7x
below threshold). Build checked clean at my edit points.

---

## 2026-07-27 ~19:30 — rebisect data landed (a0678b0); tex half moves to round 2

The CFD agent committed the corrected-bracket JSON/figure/script
(critical Re_r 4.66-4.74e5) WITHOUT the tex numbers — correct
coordination, since item A holds sa-ai.tex uncommitted. I take the tex
half into round 2: update the sec:bistability interval sentence + the
caption band/"L ~ 530" text against the committed JSON, and re-check
the seed-fed clause (which strengthens: NLF Re_r ~ 7e4 is now ~7x below
the corrected threshold, not ~4x).

---

## 2026-07-27 ~19:55 — response to "does Fig 22 show we are completely wrong?" (spheroid a10 waterfall)

Not a broken transition calibration — the figure shows the model's
DISCLOSED weakness at its worst. Per Stock's mechanism reading (cited
in the section), the measured chains at Re=1.52e6, alpha=10 mark
essentially the laminar-separation line: transition happens in/near the
separated layer, plus crossflow on the flanks — exactly the two
mechanisms the model lacks (under-amplified detached shear layer =
the conclusion's named clearest limitation; crossflow = explicit future
work, the kernel's odd pairing). Quantified deficit on record: our
chi=1 arrives 0.13-0.56 L aft of the hot-film line — amplification
happens but far too slowly once the layer lifts/skews. Wrong at
incidence for attributable, disclosed reasons — not silently wrong.

USER'S CONTROL EXPERIMENT ADOPTED — alpha=0 comparison launched:
axisymmetric, no crossflow, pure attached TS = the fair kernel test.
Digitizing Stock Fig 14a (measured front at alpha=0), extracting our
L0/L1/L2 fronts under both conventions, comparing at matched condition.
If the model lands on the measured alpha=0 front, the incidence miss is
demonstrated-attributable; the section gets restructured control-first
so a reader with the user's (referee's) reaction meets the attribution
immediately. Integration sequenced around the round-2 tex work.

## 2026-07-27 20:15 — [CFD->successor] spheroid campaign nearly complete; 8 new harvests

sphere_campaign_spheroid_results.json now carries: re65a5 L0/L1/L2,
re65a10 L0/L1/L2 (the postponed L2 finished 20:04), re72a2p5 L0/L1.
Only re72a2p5-L2 still running (~2/3 done, lands ~21:15) — after it,
EVERY tab:sphtotals dash fills. This also unblocks the Stock Figs
14/16/17 measured+computed front comparisons at 6.5e6/7.2e6 (HANDOVER
sec B literature-overlay list) and the re65/re72 waterfall/map figures
if you want them (surface field exports exist per case).

---

## 2026-07-27 ~20:10 — drag-crisis pilot COMPLETE (Re_D=1e5 cylinder: URANS + steady twin + kernel check)

Full technical record: agent-paper-review/2026-07-27-2006-dragcrisis-pilot.md.
Scripts committed under paper/repro/cfd/ (build/run/forces/kernel-check);
case data on /local_data/qiqi/sa-ai/dragcrisis_pilot/ (both machines).
GPU usage stayed within the constraint (never more than 1 GPU on
017-v100-dev; vishal's 4-GPU allotment untouched).

- **Cost calibration**: URANS 2.44 GPU-h/case (V100, 177.6k-quad O-grid,
  22k dual-time steps = transient + 31 shedding periods, 0.393 s/step);
  STEADY twin 0.08 GPU-h (30x cheaper). Matrix extrapolation: URANS
  design ~150-190 GPU-h; steady-first design ~10-20 GPU-h + ~2.5 GPU-h
  per URANS spot check.
- **URANS vs subcritical band**: St 0.194 (exp 0.19-0.20, ON band);
  Cd_mean 1.72 vs exp 1.2+-0.05 — the known 2D-URANS high bias, here
  +43% (CL rms 1.18, mean Cp_base -2.03 vs -1.2); CL_mean 0.003
  (symmetric limit cycle). Laminar separation: mean-Cf knee 81-84 deg,
  instantaneous crossing 85 deg (exp 78-80). Mean-Cf ZERO-CROSSING and
  the airfoil-style near-wall chi=1 front conventions both degrade on a
  shedding bluff body (rectified mean Cf; lifted shear layer) — knee /
  phase-resolved / off-wall conventions needed in the campaign extractor.
- **STEADY twin** (user amendment): converges CLEANLY to the symmetric
  branch — no limit cycle (CL ~1e-9, CD drift 5.8e-4/1000 pseudo steps at
  stop). Cd 0.836, separation 73.8 deg, shoulder Cp -0.91, Cp_base -0.62.
  BL-controlled quantities agree with URANS to a few degrees; wake-
  controlled quantities (Cd, base suction) bracket the experiment from
  below (-30%) while 2D URANS brackets from above (+43%).
- **Kernel check (risk 1): NO BLOCKER.** In the separated shear layers
  the free-shear pole carries the verdict: Y=omega*d dominance gives
  P ~ 1 (saturated) with onset margin 10^3 (Re_Omega 3e4-3e5 vs crit
  ~126) — transition completes within ~0.5 D of separation, chi 190->700
  along the ridge. Wall-distance DIRECTION degradation is real
  (grad d vs layer normal 50->80 deg with distance) but enters only the
  subdominant Z indicator: P changes <10% when Z is recomputed along the
  true layer normal, and clips to the same rate. P<0 in the attached FPG
  BL; gate closed at the front stagnation zone; instantaneous P map
  resolves amplifying braids vs stabilized cores. Caveat logged: at wake
  distances Re_Omega makes the onset gate trivially open, so shear-layer
  transition is immediate at ALL matrix Re — watch Cd/Cp_base at the
  Re=6e4 end.
- **Matrix go/no-go: GO, restructured steady-first.** (1) Full Re x Tu x
  up/down continuation matrix with the STEADY protocol (~10% of URANS
  cost) for Re_crit(Tu), separation angles, Cp/Cf, front families;
  (2) URANS spot checks at 2-4 Re for St / hysteresis / one-bubble
  physics (intrinsically unsteady-branch); (3) GATE: one steady+URANS
  pair in the critical range (Re~3e5, ~3 GPU-h) to verify the steady
  branch tracks the crisis before committing the matrix.
