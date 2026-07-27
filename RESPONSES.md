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

---

## 2026-07-27 ~20:15 — drag-crisis pilot COMPLETE (2.7 GPU-h); steady-first vindicated; gate pair launched

Pilot record: agent-paper-review/2026-07-27-2006-dragcrisis-pilot.md.
Headlines (chat has the full relay):
- BL-controlled quantities agree steady-vs-URANS to a few degrees
  (separation 74 vs 81-85 deg, exp 78-80); experiment's Cd (1.2) falls
  BETWEEN the branches (steady 0.836, 2D-URANS 1.72) — neither is
  trustworthy for absolute subcritical Cd, as conceded by design. St
  0.194 on-band (URANS-only quantity). Steady is 30-100x cheaper.
- Risk-1 kernel check: NO BLOCKER — free-shear pole transitions the
  separated layer at separation for the right reason; wall-direction
  degradation confined to subdominant Z (<10% on P). Watch-item: no
  effective onset threshold in the free wake (possible early transition
  at Re 6e4).
- Infrastructure findings banked: adaptive CFL is steady-only (ramp CFL
  for dual time), solver-level time-average sections required, analytic
  O-grid for closed contours (Construct2D fails on circles), clockwise
  wall-loop convention.
- Matrix design per recommendation: STEADY-FIRST ladders (~10-20 GPU-h
  total) + 2-4 URANS spot checks, GATED on one steady+URANS pair at
  Re~3e5 (critical range, ~3 GPU-h) — gate pair LAUNCHED on
  017-v100-dev under the cheap-runs envelope (user may hold if
  preferred).

---

## 2026-07-27 ~20:30 — user correction + new hypothesis on the a10 flank quiet zone

USER IS RIGHT, my earlier mechanistic explanation conflated two modes:
the documented 2D separated-zone behavior is fire-EARLY (chi=1 ~0.16c
ahead of the physical envelope) then under-amplify the handover at low
Re_theta — NOT a failure to amplify. The a10 flank shows no
amplification at all, which the 2D story does not explain (and the
cylinder pilot just showed the free-shear pole firing immediately in an
essentially-2D separated layer — the spheroid flank is the one quiet
place).

USER HYPOTHESIS (new, sharper than the pass-41 trio): the
vector-calculus formulation under-estimates the accumulated inflection
under three-dimensionality — in a skewed layer û rotates with height,
so the projected curvature and its wall-accumulation partially cancel,
suppressing even the STREAMWISE inflection (distinct from the known
transverse/crossflow gap).

AUDIT LAUNCHED (no GPU; existing a10 L2 fields): per-station kill-chain
(which link silences the kernel at x/L 0.4-0.8, phi 90-135) + the
three-way Î comparison (implemented vs fixed-edge-direction 2D
equivalent vs rotating projection) vs skew angle + crossflow-inflection
context + a control station where the model does amplify. Verdict
before any tex reframing.

## 2026-07-27 ~21:30 — alpha=0 pure-TS control DONE (Stock Fig 14a): the control does NOT vindicate the model

Digitization (repro/cfd/digitize_stock_fig14a.py, check PNG verified at
full res; provenance JSON data/stock2006_fig14a_digitized.json):
- Condition: alpha=0, Re=7.20e6 — EXACT match to our re72a0 ladder (no
  1.52-vs-1.5 caveat this time). Calibration tick-detected (5 grid
  columns, residual <1.5 px; phi=90 row check 567 vs 566.25).
- Measured (DFVLR hot films): x/L = 0.438 +- 0.001, azimuth-independent
  (17 squares recovered, phi 3-177 deg).
- Stock computed pure-TS e^N (N_TS=8.0, his tunnel limit): x/L = 0.425
  +- 0.001. e^N sits 0.014 L ahead of measurement — excellent.

Our fronts (compare_stock_fronts_a0.py; maps re72a0 L0/L1 built this
session, L2 pre-existing; runs residual-converged to ~1e-10, CL<=6e-5
on L1/L2):
- Cf-rise (k=1.5): 0.897 / 0.908 / 0.914 (L0/L1/L2); criterion band
  and azimuthal spread both ~0.002 at L2. chi=1: 0.924, chi=c_v1:
  0.930 (L2). Refinement moves the front AFT, toward ~0.92.
- VERDICT: the model misses the pure-TS control by ~0.47 L — c_f
  decays laminar straight through the measured front and transitions
  only at the tail near-separation (c_f -> 0 at x/L ~ 0.91). At
  alpha=0 there is no crossflow and no free-vortex layer: this miss
  cannot be attributed to the disclosed unmodeled mechanisms.

TWO LOAD-BEARING FINDINGS for the section (and the in-flight audit):
1. SEED DEFECT IN SEC IX AS PRINTED: the spheroid campaign ran with
   ai_laminarSlowdown=1.0 (daedalus/ai_constants convention, same
   launcher) and JSON seed 8.76e-6 => physical delivered seed 8.76e-6
   (N~13.6, the flight-quiet Daedalus level; ambient chi probed in all
   three re72a0 fields = 8.76e-6). Sec IX's "airfoil-study ... seed
   ... unchanged" is WRONG as printed (airfoil seed = 8.76e-4; the
   DFVLR tunnel itself calibrates to N_TS=8 ~ 2.4e-3).
2. BUT THE SEED IS NOT THE MISS (on the fine grid): linear-remap of
   the as-run L2 envelope (front at seed s = crossing of 8.76e-6/s)
   moves chi=1 only 0.925 -> 0.913 at the tunnel-matched seed; a 270x
   seed raise buys 0.012 L, because the L2 mid-body envelope STALLS at
   chi ~ 1e-3 (only ~5 e-folds by x/L=0.85 where Stock's TS envelope
   reaches N=8 by 0.425). On L0/L1 the same remap reads 0.64/0.54 —
   the mid-body amplification falls ~an order of magnitude L1 -> L2,
   i.e. the AMPLIFICATION is not grid-converged even though the front
   is. AUDIT-RELEVANT: this under-amplification is axisymmetric and
   zero-skew — it exists with NO three-dimensionality, which bounds
   how much the skew-cancellation hypothesis can explain.

Products committed: both scripts, both JSONs, two-panel figure
figs/spheroid_front_compare_a0.pdf/png (front plane + meridional c_f,
no in-figure titles), re72a0 L0/L1 surface-map figures. TEX HELD:
sa-ai.tex is round-2-modified and the kernel-silence audit fences
mechanism reframing; insertion-ready Sec IX paragraph + figure env +
the required "seed unchanged" sentence fix are in my report to the
orchestrator.

---

## 2026-07-27 ~21:00 — user GO: drag-crisis steady-only matrix, local, shared 4-GPU cap

User directive: steady RANS only (URANS half of the gate pair
cancelled); run on the local machine; our TOTAL across all sa-ai work
<= 4 GPUs (0-3; 4-7 are vishal's). Pilot agent redirected: full steady
continuation matrix (12 Re x 3 seeds x up/down ladders + cold
references, L1 grid, Cf-knee extraction convention, honest
non-convergence reporting) interleaving with the spheroid/Eppler runs
on free slots among GPUs 0-3. Budget ~10-20 GPU-h; stop-and-report if
>2x.

ALSO this hour (chat has full detail): the alpha=0 spheroid control
came back NOT vindicating the model — measured front 0.438 x/L vs our
0.914 (L2), mid-body envelope stalls at ~5 e-folds where Stock's e^N
does N=8 by 0.425, amplification NOT grid-converged (10x drop L1->L2),
seed explains ~0.012 L of it. Sec IX "seed unchanged" is factually
wrong as printed (spheroid ran at the flight-quiet 8.76e-6 seed) — fix
queued. The flank kernel audit was extended with alpha=0 zero-skew
control stations so its verdict separates: axisymmetric TS deficit vs
skew cancellation (user hypothesis) vs crossflow gap. Insertion-ready
control-first paragraph + figure drafted and HELD pending the audit
verdict and user read.

---

## 2026-07-27 ~20:40 — Round-2 batch: spheroid split, daesurf move, title strip, Daedalus L0 (3 commits)

- **sec:bistability numbers (item 1): verified, nothing left to do.**
  The corrected bracket rode the previous item-A commit (5545c59); I
  re-read data/stagnation_bistability.json myself — critical block
  L 682.88–688.54, Re_r 4.663–4.741e5 — and confirmed the interval
  sentence (4.66–4.74e5), the caption band and "L<~685" all match; no
  2.78/2.89/530/537.5 stragglers (the 2.78 at tab:nlfmesh is a
  mesh-spacing cell). The seed-fed opening clause carries no numbers
  and only strengthens under the higher threshold.
- **Spheroid split (COMMIT 9523f24)**: the alpha=10 surface map now
  carries the digitized Stock Fig-15a measured transition points (red
  open squares on the c_f and chi panels; new --measured flag in
  spheroid/surface_map.py, replotted from the cached npz) and stays
  in Sec. IX; the two Cft/gamma_w waterfalls moved up into Sec. IX
  with their hot-film-convention text moved (not paraphrased) from
  Appendix H; the 29.7-deg map + 3D rendering moved down to Appendix
  H (retitled "spheroid pressure stations and supplementary maps")
  beside the Cp waterfalls; your placement rule is recorded in one
  line in Sec. IX. Caption fix en route: the cfg-29.7 caption had
  borrowed the phi 90–122 deg truncation range from the Cp figure —
  that range is Stock Fig. 3's, not Fig. 5's — it now says "leeward
  tangle" without a range. daesurf5/6 joined Appendix D beside the
  sectional sheets (daesurf4 stays); all three former range-refs
  rewritten (a main-to-appendix \ref range renders "Figs. 20–D2").
- **Paper-wide title strip (COMMIT 0f71ce9)**: swept every remaining
  paper generator — daedalus_surface_a4/5/6 (suptitle + column
  titles), spheroid maps/waterfalls/front-compare,
  stagnation_bistability, nlf_cf_{low,high,neg}alpha,
  eppler_cf_{low,high}alpha, eppler_resweep_{low,high}Re,
  nlf_aft_transition, mesh_nlf_{str,cav}. Kept: axis labels, in-panel
  data labels (chi-sheet |u|/log-chi quantity headers, flat-plate
  in-axes Tu tags, the 3D view's leeward/windward identity),
  unit-bearing legend entries. Every regenerated PDF text-scanned and
  eyeballed against its committed predecessor. GOTCHA:
  regen_mesh_figures.py still pointed at flow360_fr (meshes deleted)
  and silently wrote EMPTY panels — caught via 0-edge counts,
  re-rooted to flow360_fv1 (same meshes), verified identical content.
- **Daedalus L0 ladder (COMMIT b1a33de)**: extractor re-proven
  against four printed cells (canon window: mean over final 500
  pseudo-steps, 51 samples — exact match), then the six L0 cells
  entered tab:daetotals: str 1.0244/0.01958, 1.1254/0.02219,
  1.2262/0.02505; cav 1.0156/0.02134, 1.1145/0.02427, 1.2171/0.02713
  (a4/a5/a6). Ladder sentence: at L0 the families sit 0.74–0.97%
  apart in CL and 18–21 counts (8–9%) apart in CD, unstructured the
  high-drag member; coarse-grid caveat (Sec.-IV path-sensitive front
  breakaways) cross-referenced, no L0 convergence claim. D1–D3
  section sheets now carry the full L0/L1/L2 ladder, both families,
  2D-suite line weights; 6 L0 chi caches built; probe pkls persist
  per case. **Two probe bugs found and fixed at the source** (both
  invisible in the L2-only sheets): (1) the fixed-count |y| station
  band spans 4–5 duplicate-section spanwise rings at L0 (mixed-chord
  rings at the L1 taper) — x-sorted near-coincident nodes wrecked the
  contour tangents, probe rays entered the body, Re_Omega combed 5–6
  decades; sections now snap to the single nearest ring (L0+L1
  re-probed; L2 caches verified already single-ring-dominated).
  (2) Partition-boundary duplicate wall nodes split the chi_surface
  scatter-max — one twin gets the real column, its twins read
  leftovers (a 1.5-decade square wave on cavity L0, and the TE chi
  bar that was already visible in the PREVIOUSLY COMMITTED eta92
  sheet); duplicates now merge max-wise at plot time. The dedup also
  cleaned the committed sheets' own TE artifact.
- Verification each commit: pdflatex x2 = 0 errors, 0 undefined;
  116 pp final (115 after the split, +1 back with the L0 rows); only
  my own recomputed numbers entered the tex; every regenerated figure
  rendered and inspected.

## 2026-07-27 20:45 — [CFD->successor] SPHEROID CAMPAIGN COMPLETE

re72a2p5-L2 harvested (final case). sphere_campaign_spheroid_results
.json now covers all seven tab:sphtotals conditions on the full
L0-L2 ladder — re-run emit_appendix_tables.py to fill every dash,
and the 'runs in progress' caption clause + Sec IX 'in progress'
language can come out. GPUs 0-3 are now free.

---

## 2026-07-27 ~21:45 — chat-visibility bug diagnosed and fixed (memory committed)

The user's chat was missing my "full detail" responses because I wrote
them BETWEEN tool calls: most turns ended with trailing tool calls
(this file's mirror appends, SendMessage, and the loop protocol's
"ScheduleWakeup last"), and the harness only reliably displays a
turn's FINAL text message. Fix (committed to persistent memory):
user-facing answers now come as the last message of the turn with all
tool work — including this mirror and the loop re-arm — done first.

---

## 2026-07-27 ~22:00 — visibility bug root-caused (2nd iteration) + two-Re alpha=0 diagnosis prioritized

- VISIBILITY: the residual failure was ScheduleWakeup's result text
  ("Nothing more to do this turn") being obeyed as a stop — the final
  prose was never written after it. Memory updated: that message is
  about scheduling, not a gag order; the user-facing text comes AFTER
  the last tool result, always.
- USER DIRECTIVE: diagnose the spheroid alpha=0 problem FIRST, at both
  Re (1.5e6 campaign + 7.2e6 re72a0), before the flank skew/crossflow
  verdict. Audit agent re-scoped: two-Re kill-chain stations at
  x/L 0.2-0.7, plus the DISCRIMINATING flat-plate reference (where the
  kernel is verified vs AGS) differenced link-by-link — including the
  axisymmetric-curvature question (1/r terms in the Laplacian/omega*d
  that planar profiles lack), and the Re-scaling of the deficit
  (onset-threshold artifact vs rate deficit).
- Round-2 completion re-confirmed by its agent (stale monitor timeout;
  all three commits verified as reported).

---

## 2026-07-27 ~22:15 — user GPU check: yes, cylinders running; scaled to 4 slots

GPUs 0-1 carry two drag-crisis steady chains (~1.6GB Flow360Solver
each — cylinder-sized, not spheroid); 2-3 were idle (spheroid ladders
apparently finished); 4-7 vishal. Matrix agent instructed to use the
full 4-slot budget: independent seed-ladders/cold references in
parallel (continuation serializes only within a chain), with per-launch
occupancy checks and graceful share-back if the CFD agent's runs
return.

---

## 2026-07-27 ~22:30 — housekeeping sweep (user request) + disk re-alert

- Killed 28 stale columnarDataProcessor daemons (2+ days old; cwds in
  the deleted-pending fv1lin_* dirs and long-harvested NLF cavL2prop
  cases); fv1lin_* dirs removed; flow360_fr now FULLY deleted. Fresh
  matrix per-case daemons untouched.
- Remote (017-v100-dev): clean — no our processes or GPU allocations.
- Kept: the review-folder monitor (the loop's inbox), the drag-crisis
  matrix agent, the two-Re alpha=0 / flank audit agent. All other
  subagents are completed and idle (no processes; transcripts only).
- DISK RE-ALERT: /local_data back at 99% (24G free) — spheroid high-Re
  harvest volumes + matrix outputs consumed the earlier headroom.
  Matrix agent instructed: surface/forces-only output for remaining
  cases, purge volume/restart files of completed matrix cases after
  extraction (keep the ladder-continuation restart), pause if <10G.

---

## 2026-07-27 ~21:45 — OpenFOAM Eppler overlay ADOPTED (the held item C closes; 30/30 campaign in the paper)

Per the 2110-response decision, mirroring the NLF adoption:
- fig:epppolar + fig:eppbubble now carry steel-blue OpenFOAM circles
  (structured L1+L2 by size; L0 excluded as in the NLF figures); new
  Appendix-F table tab:data_openfoam_eppler (CL, CD, chi=1 front,
  bubble stations) behind them; fig:nlfaft caption's solver-independence
  clause extended to name the full 30-case matrix.
- EXTRACTION BUG FOUND AND FIXED before anything was printed: the
  exporter's sign(z)/global-z side split breaks on the Eppler because
  its lower surface rises ABOVE z=0 aft of x~0.58 (below-airfoil fluid
  leaked into the upper mask; the alpha=0 reattachment was truncated
  away entirely). Fixed with a geometric-midline split from the wall
  patch. NLF rows verified BIT-IDENTICAL after the fix (committed table
  untouched); only Eppler a0 changed (L2 xtr_up 0.6452 -> 0.6205; a0
  x_R now resolves: 0.775/0.806 at L1/L2 vs Flow360's 0.771/0.761,
  experiment 0.74). Also corrected the exporter's force provenance
  note: the CSV rows are the final steady sample (verified exactly),
  not a tail median (tail median agrees to <=0.5%).
- Bypass-consistency check VERIFIED numerically before printing (the
  caption claim): converged bypass magnitude from the paper's own
  fr->fv1 trees at str L2 = -5.7/-6.9/-10.1/-1.1 counts at
  alpha=0/2/5/7; OF-F360 Cd residuals = +4.8/+7.5/+1.2/-6.8 counts.
  Consistent in sign and magnitude at 0-5 deg; at 7 deg the bypass is
  only -1 count and the -6.8 residual is the near-stall case (front
  offset 0.028c, both codes' hardest condition) — scoped exactly so in
  the caption, drag agreement not claimed.
- Convention forensics worth keeping: on the OF fields chi=c_v1 sits
  0.066-0.088c aft of chi=1 (vs 0.008-0.012c on the Flow360 fields at
  0-5 deg); the codes' c_v1 crossings agree to 0.014c while the chi=1
  crossings differ by up to 0.044c — the stretched chi=1->c_v1 ramp in
  the bypass-less port is exactly the handover the bypass compresses,
  and is stated as such in the table caption.
- Ride-alongs: the OF thread's build_airfoil_case.py mods (pass-48
  minor 6); LaTeX accent fix in the F7 caption ("script's"/"F4's"
  rendered with acute accents).
- NOT touched: sec:bistability, spheroid section, drag-crisis text;
  tab_spheroid_totals.tex re-emit picked up the CFD agent's in-flight
  harvest (dash rows filled) and was REVERTED — that dash-filling
  belongs to the spheroid thread with its accompanying text work.
- Build: pdflatex x2, 0 errors / 0 undefined, 117 pages (+1).

---

## 2026-07-27 ~22:20 — drag-crisis STEADY matrix: first up-ladder complete — THE MODEL PRODUCES THE CRISIS

Milestone (campaign running on local GPUs 0-3 under the shared cap;
driver paper/repro/cfd/run_dragcrisis_matrix.py; data
/local_data/qiqi/sa-ai/dragcrisis_matrix/). Tu 0.2% up-continuation
ladder, all 12 Re converged or bounded-limit-cycle:

  Re:  6e4   1e5   1.5e5  2e5   2.5e5  3e5   3.5e5  4e5   5e5   7e5   1e6   2e6
  Cd:  0.845 0.839 0.828  0.813 0.795  0.774 0.749  0.719 0.620 0.344 0.284 0.221
  knee: 72    71    71     71    71     72    73     74    79    89    91    94
  front:137   87    81     79    78     78    79     80    86    96    97    94-98
  Cpb: -0.62 -0.62 -0.61  -0.61 -0.60  -0.60 -0.59  -0.58 -0.52 -0.40 -0.34 -0.26

- The steady branch DOES track the drag crisis: Cd collapses 0.62 ->
  0.34 between 5e5 and 7e5 (low-Tu crisis onset), reaching 0.221 at 2e6
  — the experimental supercritical-minimum class (0.20-0.25). The
  Re~3e5 gate-pair "no crisis" result was simply BELOW the model's
  low-Tu crisis onset, consistent with the physical Re_crit at
  LTPT-class Tu.
- Tu family ordering is RIGHT: at Tu 0.7% the crisis onsets earlier
  (Cd 0.677 at 3e5, 0.495 at 4e5, 0.430 at 5e5, 0.365 at 7e5) — the
  Re_crit(Tu) family the campaign targets.
- FINDING (as anticipated): in the crisis band the fSlow=0.01 polish
  stage limit-cycles at several points (Tu0.7: 3-3.5e5+; Tu0.2/0.05:
  5e5, and stage-1 cycles above 7e5 with stage-2 clean). Bounded cycles
  (Cd tail p2p recorded per case in matrix_summary.jsonl); medians
  reported; nothing tuned.
- Known caveats already logged: subcritical steady Cd sits ~0.84 (no
  shedding, -30% vs exp 1.2 — wake-dynamics, not BL); knee=separation
  marker reads the LSB onset (~90-94 deg) post-crisis, not the final
  turbulent separation — the per-case crossings field carries the
  LSB/turb-sep structure for the full record.
- Cost so far ~2 GPU-h across 4 GPUs; well inside the 10-20 GPU-h
  estimate. dn-ladders (hysteresis) + cold refs + remaining seeds in
  flight; full record + figures to agent-paper-review/ when the matrix
  lands.

---

## 2026-07-27 ~21:50 — Spheroid kernel audit: alpha=0 kill chain (two Re) + flank skew/crossflow verdict

Frozen-field audit of the live kernel on the committed L2 fields
(script `paper/repro/cfd/spheroid_flank_kernel_audit.py`, pattern of the
drag-crisis pilot check; figure+tables in `paper/repro/cfd/figs_explore/
spheroid_flank_kernel_audit.{png,json}`; full record + per-station tables
in `agent-paper-review/2026-07-27-2142-spheroid-flank-kernel-audit.md`).
NO tex edits — framing waits for your read.

**PART 1, alpha=0 first (your re-prioritization).** The deficit changes
character with Re:
- **7.2e6 a0: the onset gate is INNOCENT** (open, S=0.96-1.00, Re_Omega
  1129-2458 vs threshold 528-1580 at every mid-body station). The kill is
  the rate: P ~ 0.05 decaying to 0.02, kernel bound = 8.2 e-folds by the
  measured front (0.438) — nearly Stock's e^N N=8 — but the transported
  field realizes only 4.4, then FLAT-LINES (5.0->5.2 over x=0.55-0.85,
  ~0.05 e-folds/L, while the pointwise bound still says 4-9/L), and the
  flight-quiet seed needs 11.65. Front forms only in the tail adverse PG
  (P->1.1 at 0.94) at 0.91.
- **1.5e6 zero-skew (a10 windward meridian; NO a0 field exists at 1.5e6):
  gate CLOSED** (0.07-0.12; Re_Omega ~400 vs ~700, small-P arm),
  ~1 e-fold/L — laminar to the tail-PG gate opening at 0.90 = the
  committed windward front. Consistent with Stock's laminar-to-separation
  reading: defensible at this Re. 6.5e6 windward sits in between (gate
  0.10-0.16, will miss the workshop windward front).
- **Flat-plate difference at matched Re_theta ~1100** (plate x=3 vs a0
  x/L=0.42): gate equal (1.0/1.0); P_i 0.074 vs 0.046 (-40%, about half
  visible to the planar 2D forms too = real fuller-profile physics);
  realization N_kernel/N_actual 1.42 (plate) vs 1.86 (body). **The 1/r
  Laplacian (axisymmetric-curvature) contamination of Z is real but
  second-order**: Z_vtk vs Z_ray ±10% at matched heights (plate baseline
  1-3%), ~10-20% of P — not the driver. Conclusion: a threshold retune
  would NOT move the 7.2e6 front; the a0 miss demands a small-P
  laminar-RATE fix (plus the mid-body realization stall — sharpest new
  fact, echoes the "amplification not grid-converged" note).

**PART 2, the flank.** Your 3D-cancellation hypothesis is **refuted,
cleanly**: I_hat computed (i) solver-way, (ii) fixed-edge-direction 2D
way, (iii) local-rotating-frame way agrees to 1-8% at all six flank
stations despite 28-35 deg wall skew — the max-P height sits where local
skew is 0-1 deg. The flank kernel is the OPPOSITE of quiet: P=0.54-0.99,
gate open (Re_Omega >> threshold), bound 50-110 e-folds/L. The quiet zone
is transport: the amplifying wedge is fed by subcritical windward fluid
(gate-closed at 1.5e6) with short streamline residence (crossflow 7-36%),
so chi banks its 11.65 e-folds only along the sweep path and crosses 1 at
0.92-0.96 = the committed front. Crossflow context: the kernel functional
on the transverse profile SATURATES (P_cf 1.25-1.41, up to ~130
e-folds/L) — huge in principle, but at 1.5e6 the measured flank line is
Stock's separation line, so the crossflow gap doesn't explain THIS
condition either; it becomes live at 6.5e6.

Sec IX implications (held for your decision): the "wall-anchored kernel
misses the lifted layer" hypothesis and the "in-bubble arm suppression"
hypothesis are both eliminated for the flank; "genuinely subcritical at
this Reynolds number" survives for the windward feed, plus the new
transport/residence mechanism; the a0 paragraph's deficit should be
re-attributed rate-side (gate exonerated at 7.2e6).

---

## 2026-07-27 ~21:50 — AUDIT VERDICT (alpha=0 two-Re + flank): full record in
## agent-paper-review/2026-07-27-2142-spheroid-flank-kernel-audit.md

Headlines (chat carries the full relay):
1. alpha=0 deficit CHANGES CHARACTER with Re: 1.5e6 = onset-gated
   (defensible; matches Stock's laminar-to-separation reading);
   7.2e6 = gate fully open, kill is rate-side — and decomposed: the
   POINTWISE kernel budget (8.2 e-folds by the measured front) nearly
   matches Stock's e^N (N=8); the TRANSPORTED field realizes only 4.4
   then flat-lines (0.05 e-folds/L over x=0.55-0.85); the flight-quiet
   seed demands 11.65 e-folds vs the tunnel's ~6. 1/r Laplacian
   contamination real but second-order (~10-20% of P). Fix must act on
   delivered growth at small P; onset retuning would not move 7.2e6.
2. USER'S SKEW-CANCELLATION HYPOTHESIS: REFUTED — three I_hat
   formulations agree to 1-8% at all six flank stations despite
   28-35 deg wall skew (the amplifying height is skew-aligned; rotation
   lives in the X-dominated sublayer).
3. The flank is the MOST amplifying place on the body (P 0.54-0.99,
   gate open). The quiet zone is transport/feed: the wedge is fed by
   gate-closed (subcritical at 1.5e6) windward fluid with short
   residence; chi accumulates only along the sweep path and crosses 1
   at 0.92-0.96 = exactly the committed front. "Amplification deficit"
   = real as observed, but transport/feed, not kernel formulation.
4. Crossflow channel would SATURATE the rate ceiling on the flank
   (P_cf 1.25-1.41) — enormous in principle, but at 1.5e6 the measured
   flank line is the separation line, so its absence doesn't explain
   this condition; it goes live at 6.5e6.
5. Sharpest new fact: the 7.2e6 realization STALL (transported envelope
   frozen while the pointwise bound says 4-9 e-folds/L) — echoes the
   non-grid-converged amplification; streamline-integrated budget or an
   L1-vs-L2 audit repeat would separate sliver/diffusion physics from
   grid effects.

Sec IX reframing plan awaiting the user's read (no tex edits made).

---

## 2026-07-27 ~22:05 — user physics deep-dive request: 7.2e6 alpha=0

Q: Re_theta during the amplification; what overwhelms the amplifying
band given c_nu_ai=1/6 was tuned against laminar-viscosity overwhelm;
figures for exp vs numerical vs e^N (onset, Re, H, chi(x,walldist)).
A (immediate, from the audit tables): Re_theta = 584/736/902/1089/1363
at x/L = 0.20/0.30/0.42/0.55/0.70 — the amplification range spans
Re_theta ~ 600-1400, with the measured front at Re_theta ~ 900.
Figure-package agent launched (BL character Re_theta/H; e^N march on
our own edge conditions with N=6/8/11.65 crossings; log10-chi contour
in (x/L, walldistance) with the P>0 band, gate contour, delta99; and
the band-width/confinement-penalty quantification testing whether the
stall is the II.D eigenvalue physics at small P — band too thin for
ANY c_nu_ai — vs another leak, with a frozen-field chi transport
balance at the stalled station). Deliverables to figs_explore PNGs +
a thread md.

---

## 2026-07-28 ~00:15 — drag-crisis STEADY matrix COMPLETE: 84/84 cases, 5.8 GPU-h, crisis + Tu family + branch map delivered

Full record: agent-paper-review/2026-07-28-0010-dragcrisis-steady-matrix.md.
Figure: paper/repro/cfd/figs_explore/dragcrisis_matrix.png. Data:
/local_data/qiqi/sa-ai/dragcrisis_matrix/ (5.7 GB after the disk-order
cleanup; all GPUs released).

Headlines:
- THE MODEL PRODUCES THE DRAG CRISIS as a Tu-ordered family: crisis
  center ~4e5 at Tu 0.7%, ~5.5-6.5e5 at Tu 0.05-0.2%; supercritical
  floor 0.217-0.233 (experimental two-bubble class 0.20-0.25), rising
  slightly with Tu — the correct facility/roughness trend. Post-crisis
  wall structure is a genuine LSB (turbulent reattachment; bubble
  shrinks 10 deg -> 1.5 deg from 7e5 to 2e6) + final turbulent
  separation 107-118 deg (short of the experimental ~147 — the expected
  SA curved-wall class of error, logged).
- Branch map: up/dn/cold agree to <=0.003 Cd outside the crisis;
  inside it a reproducible ~0.05 spread with 27/84 cases limit-cycling
  (Cd tail p2p median 0.011 — weak breathers; all quantified per case).
  The dn branch lands HIGH (opposite the experimental hysteresis
  direction) — flagged as needing URANS continuation, not read as
  physics. CL = 0 in all 84: the symmetric steady protocol cannot see
  the one-bubble state (finding).
- Findings ledger in the record: crisis-band limit cycles, subcritical
  Cd wake bias (by design), dn anomaly, separation-angle gap, the
  caught-and-fixed case-name collision (clean rerun), disk measures.
- Cost 5.76 GPU-h total (est. 10-20), wall ~1.6 h on shared GPUs 0-3.
- Suggested next: Achenbach Cf/Cp overlays from the retained surface
  data; 2-4 URANS spot checks for St/one-bubble/hysteresis; cheap low-Re
  extension for the 6e4 caveat.

---

## 2026-07-28 ~00:15 — STEADY DRAG-CRISIS MATRIX COMPLETE (84/84, 5.8 GPU-h)

Record: agent-paper-review/2026-07-28-0010-dragcrisis-steady-matrix.md.
Headline: the model PRODUCES the drag crisis as a Tu-ordered family —
Re_crit decreasing with Tu (Achenbach-1971 class), supercritical floor
0.217-0.233 (experimental two-bubble class 0.20-0.25) rising correctly
with Tu, genuine LSB + turbulent-separation structure with physically
shrinking bubble length, shoulder suction deepening to the experimental
class — from chi_inf alone, no correlation, no extra equation. Honest
ledger: steady branch non-unique through the crisis band (27/84
limit-cycle flags, ~0.05 Cd spread, protocol-dependent — firm
hysteresis claims need URANS); dn-branch-higher-Cd anomaly (opposite
experimental sign — not to be read as physical); one-bubble state
invisible to the symmetric steady protocol (CL=0 all 84); known misses
= turbulent separation angle 107-118 vs ~147 [mem-unverified] and the
conceded subcritical wake bias. Under budget 30x vs the URANS design.
Suggested next steps (user-gated): Achenbach Cf/Cp digitized overlays,
2-4 URANS spot checks (St/one-bubble/hysteresis), low-Re extension.
All [mem] experimental targets remain source-unverified — nothing
enters the paper without that.

---

## 2026-07-28 ~00:25 — user directives: FS beta=+0.10 comparison + drag-crisis section

1. SPHEROID (sharpened question): the beta=+0.10 FS case is where the
   PARABOLIZED marcher verifiably delivers e^N-class growth in mild FPG
   — physics agent extended to tabulate spheroid H(x/L) and
   max(Omega_hat*I_hat) at matched (planar) convention against the
   committed FS beta=+0.10 table values. If they sit in the same class,
   the pointwise profile physics is fine and the deficit is RANS
   transport delivery (the realization stall) — marcher delivers,
   transport doesn't.
2. DRAG-CRISIS SECTION (user decision): own section placed just after
   both airfoils (new Sec VII, bistability shifts down). Macroscopic
   first: Cd vs Re figure (3 Tu levels, up/dn ladders, cold refs,
   limit-cycle cases marked with vertical oscillation bands from
   Cd_tail_p2p), then |u| + log10(chi) field panels for ~3 selected
   cases (subcritical/crisis/supercritical). Minimal text; honest
   non-uniqueness paragraph; NO unverified [mem] experimental numbers
   (qualitative comparisons only until the Achenbach digitization);
   Appendix F data table behind every plotted point. Section agent
   launched.

## 2026-07-28 ~07:30 — drag-crisis section DONE (new Sec VII, committed)

The steady drag-crisis matrix is now a paper section, "The cylinder
drag crisis" (Sec VII, after the Eppler Re-sweep; bistability shifts
to VIII, Daedalus IX, spheroid X — all cross-refs label-based, build
clean, 119 pp, 0 errors / 0 undefined).

- Fig. 18 figs/dragcrisis_cd_re.pdf (regen_dragcrisis_cd_re.py):
  Cd(Re_D) family, three seeds (CVD-validated color trio), up solid /
  dn dashed-open / cold diamonds; the 27 limit-cycle cases carry
  capped vertical bands = tail-window Cd min–max recomputed from each
  case's force history. The script also syncs
  data/dragcrisis_matrix_summary.jsonl (committed copy of the
  campaign summary) + dumps data/dragcrisis_cd_re_computed.json.
- Fig. 19 figs/dragcrisis_fields.pdf (regen_dragcrisis_fields.py):
  |u|/U_inf + log10(chi) (white chi=1 contour) at Tu 0.2% up-ladder,
  Re_D = 1e5 / 5e5 / 2e6 (subcritical dead-air wake with immediate
  shear-layer amplification -> crisis -> supercritical LSB + narrow
  wake). Rasterized fields (94 KB, no contour seams).
- Table F8 tab_dragcrisis.tex (emit_appendix_tables.dragcrisis):
  every plotted Cd point, bracketed tail p2p for flagged cases.
- Text: minimal per directive — positioning para (literature moved
  from the conclusion outlook, which is now a 4-line pointer),
  protocol para, family result (Re_crit down with Tu; floor
  0.217–0.233 rising with Tu; LSB 92/102/107 deg at 7e5 -> 1.5-deg
  bubble + 118 deg at 2e6; shoulder −0.90 -> −2.43, base −0.62 ->
  −0.26), honest para (27/84 limit cycles, ≤0.055 protocol spread in
  the band vs 0.003 on the plateaus, dn-higher = no hysteresis claim,
  CL=0 protocol blindness to one-bubble states, subcritical wake bias
  conceded, separation 106–118 deg early QUALITATIVELY — no [mem]
  experimental numbers; %TODO(dragcrisis-exp) marks the Achenbach
  digitization step).
- Every printed number re-verified from matrix_summary.jsonl +
  per-case summary.json + independent Cd/p2p recomputation from the
  force CSVs (7 spot cases exact); seeds re-verified against
  lib.calibrate_kernel.chi_inf_from_Tu_pct; chi normalization
  verified on the slice (far field reads 1.0835e-2 exactly).
