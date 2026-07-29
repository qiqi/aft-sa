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

---

## 2026-07-28 ~00:45 — spheroid a0 physics deep-dive DONE (re72a0 L2): the band is never overwhelmed; the mean flow is

Full record: agent-paper-review/2026-07-28-0040-spheroid-a0-physics.md;
script paper/repro/cfd/spheroid_a0_physics.py; five READABLE figures +
captions + JSON in paper/repro/cfd/figs_explore/spheroid_a0_physics_*.

1. c_nu_ai VERDICT: the Sec-II.D eigenvalue solved on the EXTRACTED
   profiles stays positive at every station for EVERY c_nu_ai (even 1.0;
   min 0.48/L at x/L=0.70; at 1/6: 9.2/L at the front station, 2.0/L at
   the trough). Confinement costs more than at calibration (retention
   0.50-0.61 vs FS-class 0.72-0.80; w/theta 2.8-4.4 vs 5.0-5.6) but
   NEVER wins; c_nu_ai->0 only doubles the trough growth vs an ~e^7
   deficit. No c_nu_ai retune addresses this case.
2. THE NEW ISOLATED LINK — the RANS laminar MEAN PROFILE is anomalously
   full: H=2.49 at the measured front (plate operator-control reads
   2.60; Thwaites-Mangler on the SAME u_e says 2.59-2.61; grid-stable
   L0/L1/L2 and the aft anomaly DEEPENS with refinement: P at 0.70
   0.060/0.046/0.022). Sharpened FS check: spheroid H=2.49 ~ FS
   beta=+0.10 H=2.481, and its planar max(OmegaI)=0.056 > FS 0.037 —
   the kernel reads its profile fairly (pointwise physics fine, per
   the user's marcher-delivers hypothesis); the profile itself is the
   stabler class.
3. e^N ON OUR OWN FIELD: on Thwaites shapes from OUR u_e: onset 0.059,
   N8 at 0.492 (Stock 0.425, measured 0.438 — same class), N11.65 at
   0.661. On the field's OWN H/Re_theta: onset 0.197, N8 at 0.842,
   N11.65 unreached by 0.88 — an exact Drela envelope on the RANS mean
   flow would front at ~0.84-0.9+; the model's chi=1 front 0.9244 is
   e^N-CONSISTENT with its own mean flow.
4. TRANSPORT BALANCE at x/L=0.70 (stall): production +1.00; wall-normal
   diffusion -0.79 at the chi peak (eigenmode would pay ~0.50 —
   the transported profile is sub-eigenmode-shaped, inherited from the
   upstream wider band); conv_s +0.36, conv_n -0.24 (redistribution);
   destruction/cb2 inert (<1e-8/1e-4). Realized 0.67-1.36/L vs eig 2.0
   vs sup 4.1. Realization stall = honest non-parallel transport, not
   a leak.
5. ESTIMATOR CORRECTION recorded: the flank audit's raw planar P_ii
   values were noise-inflated ~1.3-1.6x (double-np.gradient on probed
   profiles; converged smoothed values match the solver's compact
   Laplacian). Audit's P_i chain and verdicts unaffected.
6. FOLLOW-ON POINTED AT (no action): laminar mean-flow fidelity on
   curved bodies — a standalone laminar BL march / OpenFOAM laminar
   cross-solve on this geometry would discriminate solver numerics vs
   real non-equilibrium physics in hours.

---

## 2026-07-28 ~00:45 — ALPHA=0 PHYSICS VERDICT: the mean flow carries the miss

Record: agent-paper-review/2026-07-28-0040-spheroid-a0-physics.md;
figures figs_explore/spheroid_a0_physics_{blchar,eN,chifield,band,
balance}.png (+captions.md).
1. NOTHING overwhelms the amplifying band: the II.D eigenvalue stays
   positive at every station and every c_nu_ai (penalty 13-44% of
   production, never 100%); no c_nu_ai retune addresses this case.
2. DOMINANT NEW LINK: the RANS laminar mean profile is anomalously FULL
   — H = 2.49 where Thwaites on the SAME u_e (and the plate-control
   extraction, 2.60) says Blasius-class 2.59. That halves the amplifying
   coordinate (0.046 vs 0.078) and moves ANY H-based envelope's N=8
   station from 0.49 to 0.84. An exact Drela-Giles tracker on the RANS
   mean flow would put the front at 0.84-0.9+; the model's 0.924 is
   e^N-CONSISTENT GIVEN ITS MEAN FLOW. Not grid-converged: refinement
   DEEPENS the fullness anomaly.
3. FS beta=+0.10 (user's discriminator): the spheroid profile IS that
   class (H 2.49 vs 2.48) and the kernel reads it fairly (0.056 planar
   vs FS-table 0.037) — pointwise profile physics fine; the marcher
   delivers on this class.
4. Transport realization: the transported chi pays 79% of production to
   wall-normal diffusion at the stalled station vs the eigenmode's ~50%
   — honest non-parallel/history transport of a sub-eigenmode shape
   through a thinning band; not a leak, not the coefficient.
5. Correction to the audit thread: its raw planar-P numbers were
   noise-inflated 1.3-1.6x (savgol-validated estimator now canonical).
FOLLOW-ON LAUNCHED (cheap, CPU): why is the RANS laminar BL full —
standalone axisymmetric laminar BL profile march on the extracted u_e
vs the RANS profiles + a momentum-balance probe at mid-body to find the
term feeding the fullness (candidates: low-Mach/O-grid numerics,
axisymmetric metric handling, genuine non-equilibrium). Sec IX/X
reframing HELD until this discriminator + user read.

---

## 2026-07-28 ~00:50 — user directive: unstructured spheroid family

Q: what mesh does the spheroid use; shouldn't we build an unstructured
family too (triangulated skin + the in-house unstructured pipeline)?
A: the spheroid currently has ONE family — spheroid/ogrid_spheroid.py,
an analytic structured O-grid (meridian half-plane revolved, pole
prisms, 0.37/2.2/12.6M ladder, sphere far field 30L) — the only
single-family case in the paper. Agreed and launched: an unstructured
family (triangulated surface + prisms, Daedalus-cavity-style pipeline,
matched h0/clustering), running alpha=0 at 7.2e6 first as the
DISCRIMINATOR for the mean-flow fullness anomaly: unstructured
H ~ 2.49 convicts solver-core/physics; H ~ 2.59 convicts the O-grid
discretization. One GPU under the team cap; runs alongside the
Keller-box/momentum-balance discriminator.

## 2026-07-28 ~00:55 — amendment: unstructured spheroid = FULL circumference

User: revolve the full 360 (no y=0 symmetry plane) — half-body issues
have occurred before, and it is cheap. Build agent amended: wall +
farfield BCs only, ~2x nodes (fit-checked for one GPU, L0-class
fallback), and the alpha=0 run now doubles as a no-symmetry-enforced
zero-lift/side-force check (any residual asymmetry is diagnostic).
Incidence-plane convention documented for later alpha>0 parity with the
O-grid.

## 2026-07-28 ~01:00 — user challenges the band-figure physics (folded into the discriminator)

User's expected picture on a 6:1 spheroid at alpha=0: suction peak at or
slightly upstream of x/L=0.5; FPG to the peak, APG past it; hence
equilibrium H below Blasius on the front half rising past the peak, and
max(Omega_hat I_hat) rising MONOTONICALLY into the adverse region. The
band table shows the opposite trend (maxP 0.058->0.022 by 0.70, field H
2.51->2.44) and a suspiciously flat claimed equilibrium H. Discriminator
agent instructed to arbitrate first: extract/plot Cp(x) and u_e(x) (peak
location + PG-sign bands), Keller-box equilibrium H vs field H with
lambda(x)/beta_H(x) quantifying the slender-body PG, and re-verify the
aft-station extraction (thickening layer + tail curvature = likeliest
artifact zone). If the downstream maxP fall is extraction-real, a RANS
profile getting FULLER into an adverse gradient is a strong strange
signature to state plainly.

## 2026-07-28 ~01:05 — user addition: spheroid mesh cross-section figures

Build agent instructed: meridian-plane (x-r) and transverse (y-z at
x/L~0.5) cuts for BOTH families (O-grid + the new unstructured
full-circumference), with near-wall/nose zooms showing growth, pole
treatment, prism layers, and the tet transition; plus a first-cell
height/growth comparison strip vs x/L. PNGs to figs_explore with a
captions md; paths in the report.

## 2026-07-28 ~01:20 — alpha=0 mean-flow discriminator COMPLETE: solver numerics convicted (record 0105)

Full chain in agent-paper-review/2026-07-28-0105-spheroid-a0-meanflow.md;
script paper/repro/cfd/spheroid_a0_meanflow.py; six figures + JSON in
figs_explore (spheroid_a0_meanflow_*).

1. Laminar BL PROFILE march (validated to Blasius 4th decimal; full
   history + Mangler + transverse curvature) on the field's own u_e:
   H = 2.56/2.58/2.63 at x/L 0.2/0.42/0.7 vs RANS L2 2.51/2.49/2.44.
   History and axisymmetric terms are real but an order too small:
   candidates (b) and (c) eliminated; (d) eliminated by the plate
   operator control + edge/convention ladders + Cp/u_e checks.
2. Momentum-balance probe (closes 0.5% on the plate): the spheroid L2
   field carries a NEAR-WALL NUMERICAL MOMENTUM SOURCE (+33% of the
   local viscous term at 0.42, +83% at 0.70, hx-robust) — the force
   holding the profile full. Not a uniform nu_eff excess.
3. Direct staggering signature: cell-locked tangential wiggle, dominated
   by the AZIMUTHAL one-cell mode (du/u p2p 2.6% at 0.15 d99 at L2;
   3.7x bigger at L1; should be exactly zero on an axisymmetric flow);
   surface faceting sagitta ~10 first-cell heights at L2.
4. Airfoil control: NLF's low favorable-region H (2.44-2.51) matches a
   march on ITS OWN u_e to 0.02 — genuine LE history, airfoils clean;
   anomaly is spheroid-specific. Plate is the COARSEST streamwise
   (0.22 cells/d99) yet exact — cells/d99 is not the discriminator.
5. USER CHALLENGE answered point by point (record Sec 4): u_e peak at
   x/L = 0.489 (as expected; extraction right); proper laminar H does
   dip below Blasius fore and rise aft (the 0040 note's "2.59-2.61
   flat" corrected — fore gap smaller, aft gap larger); aft fall of
   field H/maxP is extraction-real (dense rays, convention ladder), and
   YES, stated plainly: the RANS layer gets FULLER into the adverse
   gradient (maxP 0.044->0.026 where marched profiles rise
   0.078->0.127) — wrong-signed response, coinciding with the largest
   measured numerical source.
6. RECOMMENDED RUN (not launched): re72a0 L1 with
   lowMachPreconditioner=true, byte-identical otherwise (~1/8 L2 cost).
   H(0.42) -> 2.58 convicts low-Mach Roe dissipation; unchanged ->
   faceting/staggering truncation (then: azimuthal x2 L1 stripe).
   Complements the build agent's unstructured full-circumference
   discriminator: preconditioner flag vs grid family, two orthogonal
   knobs on the same anomaly.

---

## 2026-07-28 ~01:10 — MEAN-FLOW DISCRIMINATOR VERDICT: solver numerics convicted

Record: agent-paper-review/2026-07-28-0105-spheroid-a0-meanflow.md.
1. Validated axisymmetric Keller-box march (Blasius to 4 digits;
   Mangler + transverse curvature included) on the field's own u_e:
   H = 2.56-2.63 across 0.2-0.7 vs RANS 2.49-2.44. History, Mangler,
   transverse curvature, extraction: all real, all captured, all an
   order too small. Airfoils exonerated properly (their low favorable-H
   is genuine LE history, march-consistent to 0.02).
2. Momentum-balance probe (plate-validated to 0.5%): a structured
   NEAR-WALL STREAMWISE MOMENTUM SOURCE of numerical origin (+0.33 and
   +0.83 of max-viscous at x/L 0.42/0.70) — the force holding the
   profile full, largest where the anomaly is deepest.
3. Discrete signature: cell-locked AZIMUTHAL staggering (du/u p2p 2.6%
   at 0.15 d99 — should be exactly zero on axisymmetric flow); the
   surface is azimuthally faceted at ~10 wall units of sagitta.
4. USER'S CHALLENGE CONFIRMED point by point: suction peak at 0.489;
   the marched laminar kernel RISES monotonically into the adverse
   region (0.078 -> 0.246) exactly as expected; the RANS layer getting
   FULLER into the APG is the wrong-signed response — extraction-robust
   and coinciding with the largest momentum source. The 0040 note's
   flat-equilibrium claim corrected (fore gap -0.02, aft -0.18).
5. VERDICT: the alpha=0 front miss is carried by a SOLVER-NUMERICS
   mean-flow artifact of the axisymmetric O-grid class — not model
   physics, not BL theory, not the pressure distribution. Model
   constants owe this case nothing.
6. PROPOSED DISCRIMINATING RUN (awaiting user go): re72a0 L1 with
   lowMachPreconditioner=true, config-only, ~minutes (1/8 L2 cost).
   H -> 2.56-2.58 convicts low-Mach Roe dissipation (L2 rerun would
   then pull the front from 0.92 toward ~0.5); H unchanged convicts
   faceting/staggering (follow-up: azimuthal x2 L1 stripe). The
   unstructured full-circumference family (building) is the independent
   topology arm.

## 2026-07-28 ~01:15 — artifact-path convention (user order) + retrofit

From now on every RESPONSES.md entry lists FULL PATHS of generated
artifacts. Retrofit for tonight's sets:
- alpha=0 physics package (ffdd99c):
  /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/spheroid_a0_physics_blchar.png
  /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/spheroid_a0_physics_eN.png
  /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/spheroid_a0_physics_chifield.png
  /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/spheroid_a0_physics_band.png
  /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/spheroid_a0_physics_balance.png
  /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/spheroid_a0_physics_captions.md
  /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/spheroid_a0_physics.json
- mean-flow discriminator:
  /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/spheroid_a0_meanflow_march.png
  /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/spheroid_a0_meanflow_profiles.png
  /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/spheroid_a0_meanflow_balance.png
  /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/spheroid_a0_meanflow_resolution.png
  /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/spheroid_a0_meanflow_wiggle.png
  /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/spheroid_a0_meanflow_cp_pg.png
  /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/spheroid_a0_meanflow.json
- alpha=0 control comparison (d736cec):
  /home/qiqi/flexcompute/sa-ai/paper/figs/spheroid_front_compare_a0.pdf (+.png)
- drag-crisis section (edb8bc5):
  /home/qiqi/flexcompute/sa-ai/paper/figs/dragcrisis_cd_re.pdf
  /home/qiqi/flexcompute/sa-ai/paper/figs/dragcrisis_fields.pdf

---

## 2026-07-28 — Drag-crisis Re-range extension + spheroid Mach answer + full-body directive

**User prompts:** (1) extend the drag-crisis section to small and large Re with exactly
the same model (mesh regen with different first-layer thickness allowed); (2) check what
Re range standard textbooks/surveys cover — the composite curve spans many orders of
magnitude, no single method covers it (no WRLES at high Re, no fully-turbulent RANS at
low Re) — this model might be the first; (3) what Mach was the spheroid run at
(density-based solver, best at Mach >= 0.1)?; (4) run the spheroid discriminator on a
FULL spheroid, not half.

**Answers/actions:**

- **Spheroid Mach:** 0.1 in every case — O-grid L0/L1/L2 and the unstructured build
  (`/local_data/qiqi/sa-ai/spheroid_fv1/case_ogrid_L{0,1,2}_saai_re72a0/Flow360.json`:
  Mach 0.1, muRef 1.389e-8, lowMachPreconditioner false). So we are at the recommended
  floor for the density-based solver; the low-Mach-Roe suspect concerns LOCAL Mach in
  the near-wall laminar layer (~0.01–0.03), not the freestream choice.
- **Half-body confirmed:** the structured O-grid family is a half-model — meridian
  half-plane revolved phi in [0, pi] with symmetry sheets at y=0
  (`/home/qiqi/flexcompute/sa-ai/spheroid/ogrid_spheroid.py`). Every published spheroid
  number rides on symmetry BCs; given the measured azimuthal one-cell staggering, the
  full-body control is well chosen.
- **LAUNCHED — full-body spheroid discriminator** (background agent): new
  `spheroid/ogrid_spheroid_full.py` (L1, phi in [0,2pi), periodic closure, 2x azimuthal
  count, no symmetry boundaries), re72a0 case byte-comparable to the half-model, two
  arms: (A) settings identical (half-vs-full discriminator), (B) lowMachPreconditioner
  true (low-Mach arm). Harvest = 0105-record instruments: H at x/L 0.20/0.42/0.70,
  chi=1 front, azimuthal staggering spectrum at 0.15 delta99. Record will land in
  agent-paper-review/ with full paths.
- **LAUNCHED — drag-crisis Re extension** (background agent): three-arm design on the
  matrix driver. Low arm Re 1e3–4e4 on the pilot mesh; creeping/steady arm Re 1–300 on
  a new R_OUT=1000D mesh (physical flow is steady below Re~47 — a true validation
  against Dennis & Chang 1970 / Tritton); high arm Re 4e6–1e7 (stretch 2e7) on a new
  y1=1e-6 D mesh. Mandatory mesh-seam overlap runs at Re 300/1e3 and 2e6/4e6; lossless
  integer Re tags; steady-only, canon env, volume off, <=2 GPUs, occupancy-checked.
  Target span: Re 1 -> 1e7 (7 decades) with one model, one set of constants.
- **LAUNCHED — literature survey** (background agent): the canonical composite Cd(Re)
  datasets with exact ranges (Wieselsberger 1921/Schlichting, Tritton 1959, Delany &
  Sorensen 1953, Achenbach 1968/71, Roshko 1961, Schewe 1983, Zdravkovich regime
  taxonomy), low-Re steady benchmarks (Dennis & Chang, Fornberg), and an adversarial
  novelty check: highest-Re published WRLES/WMLES of the crisis and every
  transition-RANS cylinder/sphere drag-crisis attempt with its Re range — to scope the
  "first single-model sweep" claim defensibly. Record will land in agent-paper-review/.
- **Disk flag (user decision needed):** /local_data is 100% full (14G free). Largest
  reclaimable: `/local_data/qiqi/sa-ai/daedalus_fv1/` = 308G, dominated by the L2
  cavity/ogrid cases (34–64G each: mesh.cgns >1G, restartOutput, volume outputs). These
  back the Daedalus field figures' repro chain, so deletion is user-gated. All three
  running campaigns are under strict lean-output discipline (volume off, log
  truncation, restart janitor, free-space checks before every launch).

**GPU status at launch:** 3 busy (unstructured-spheroid arm + vishal), 5 idle; all our
agents occupancy-check per launch under the 4-GPU own cap.

---

## 2026-07-28 — Literature ground truth landed: the composite is 7.3 decades and nobody spans it

Record: /home/qiqi/flexcompute/sa-ai/agent-paper-review/2026-07-28-0117-dragcrisis-litrange.md

- Composite Cd(Re) curve = Re 0.5 -> 1e7 (~7.3 decades), a stitch of >=4 facilities
  (Tritton 0.5-100; Relf ~1-1e3; Wieselsberger ~50-8e5; Delany-Sorensen 1.1e4-2.3e6;
  Achenbach 6e4-5e6; Roshko 1e6-1e7; Schewe 2.3e4-7.1e6; Shih to 8e6). No single
  dataset covers more than ~2.5 decades; ~2x facility scatter in Re_crit.
- Scale-resolving ceiling (verified): WRLES max = 8.5e5 (Cheng-Pullin-Samtaney 2017);
  WMLES reached 2e6 and was already wrong/Re-insensitive there (Catalano 2003); NOBODY
  has scale-resolved Roshko's transcritical range.
- Transition-RANS (verified by adversarial absence): widest pure-RANS transition sweep
  ~1 decade around the crisis, isolated spot Re, often non-convergent in the critical
  band (gamma-Re_theta sphere study "unstable in the critical and supercritical
  range"); no single-model curve across even 3 decades anywhere; no controlled Tu
  study.
- Fully-turbulent RANS: covers the span but premature-crisis at subcritical
  (Stringer 2014 Re 40-1e6 = the wrong baseline our Sec VII exploits).
- VERDICT: no published single model/method of any kind has produced the curve from
  steady-laminar through transcritical — the user's "this model might be the first"
  is supported, with scoped wording options (record Sec 4) and landmines: never claim
  "first prediction of the crisis" (LES did it); caption 47<Re<1e3 as steady-branch
  continuity, not validation (inset strategy recommended); frame the Tu family as
  spanning the experimental Re_crit band.
- Low-Re steady truth: below Re~47 steady IS physical — match Dennis-Chang/Fornberg
  to ~1% (SA-AI must stay passive; consistency check for the paper).
- Open-access digitization shortlist: Delany-Sorensen NTRS PDF, Roshko Caltech PDF,
  Henderson 1995, Cheng 2017 (Achenbach stays user-gated).

---

## 2026-07-28 — Daedalus L2 cleanup executed (user-directed)

Deleted from all six L2 cases (case_{cavity,ogrid}_L2_saai_a{4,5,6} under
/local_data/qiqi/sa-ai/daedalus_fv1/), biggest + least-likely-needed only:
- mesh.cgns_rank_*_of_*.dmp — solver partition dumps, regenerated automatically at
  launch from the retained mesh.cgns (~168G);
- restartOutput/ — continuation checkpoints; campaign complete (~33G);
- volume.pvtu + volume_proc*.vtu — raw volume output (~31G). Safe because the derived
  chi_surface.npz (verified present in all six cases) is what every paper regen script
  reads (paper/regen_daedalus_surface_maps.py, paper/repro/cfd/
  regen_daedalus_section_sheets.py); the only volume reader is
  daedalus/chi_surface_map.py, which built the npz.

KEPT per case: mesh.cgns (rerun capability), all surface_*.pvtu, chi_surface.npz, all
csv histories, Flow360.json, figures, logs.

Result: /local_data free 14G -> 274G; daedalus_fv1 308G -> 74G. The three running
campaigns (drag-crisis extension, full-body spheroid, unstructured spheroid) now have
ample headroom.

---

## 2026-07-28 — Global mesh-partition-dump cleanup (user-directed)

Swept all of /local_data/qiqi for `mesh.cgns*_rank_*_of_*.dmp` (solver partition
dumps — launch-time artifacts, regenerated automatically from the retained mesh.cgns):
deleted 242 files / 163 GB nominal (some were hardlinked across staged cases, so disk
actually freed = 114 GB). Skipped: anything written today (the active unstructured-
spheroid case) and all `restartOutput/restart_rank_*.dmp` checkpoint files — those
hold solution state, a different class, left untouched outside the already-approved
Daedalus L2 deletion. Remaining small dumps (<100 MB each, 3.6 GB total, mostly 2D
airfoil cases with active ladder relaunches) left in place as negligible.

/local_data: 273G -> 387G free (79% used). Combined with the Daedalus L2 sweep:
14G -> 387G free today.

---

## 2026-07-28 — Fig 19 restyled (line contours); Fig 18 literature overlay launched

**User prompts:** (1) Fig 19: contour not contourf like the rest of the paper; remove
per-row Re notes (caption carries them); panels at ~half-page width. (2) Fig 18:
generously overlay literature — experiments, LES, fully-turbulent RANS, transition
models. (3) Fig 18: drop the cold-start points, keep up/dn sweeps only; no vertical-
band legend entry — describe in caption.

**Done — Fig 19 (commit fc085d2):**
- /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/regen_dragcrisis_fields.py: contourf ->
  contour both columns (13 |u| levels, 14 log10-chi levels); per-row Re labels removed
  from the y-axes; chi=1 overdraw white -> BLUE (magma's low end is near-black on the
  now-white background); colorbars -> continuous gradients; figsize 10.6 -> 7.4 wide
  so each panel sits at ~0.47 textwidth with no dead space.
- /home/qiqi/flexcompute/sa-ai/paper/figs/dragcrisis_fields.pdf regenerated; preview
  /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/dragcrisis_fields.png.
- sa-ai.tex caption: "white contour" -> "blue contour" (rows' Re already in caption).

**Launched — Fig 18 overlay (background agent):** acquires + independently digitizes
(tick-calibrated, per-dataset check PNGs, provenance JSONs under
paper/repro/cfd/litdata/dragcrisis/) the canonical experiments (Wieselsberger,
Delany-Sorensen TN3038, Achenbach 1968, Roshko 1961, Schewe 1983), LES (Cheng 2017
WRLES, Rodriguez/Lehmkuhl, Catalano WMLES), fully-turbulent RANS (Stringer 2014), and
transition-RANS (gamma-Re_theta FTC 2016); extends regen_dragcrisis_cd_re.py with
class-grouped background markers; ALSO removes cold-start points and the limit-cycle-
band legend entry per (3), delivering a draft caption amendment. I integrate the tex
after reviewing its record.

---

## 2026-07-28 — Unstructured spheroid family landed: topology cleared, and the 20k budget convicted

Record: /home/qiqi/flexcompute/sa-ai/agent-paper-review/2026-07-28-0350-spheroid-unstruct-family.md

Two findings, the second bigger than the assignment:
1. TOPOLOGY CLEARED: full-circumference Flynn360 prism/tet/octree family (no symmetry
   plane) reproduces the O-grid at matched level+budget to extraction precision —
   chi=1 front 0.9176 (unstr L1) vs 0.9177 (og L1 median). The late front is NOT an
   O-grid artifact. Full-body zero-lift clean at L1 (CL ~3e-4); the L0 asymmetry
   episode reproduces in full-body form (grid-level artifact, both families).
2. THE 20k PSEUDO-STEP BUDGET IS A COLD-START ARTIFACT at this condition: the same
   unstructured L1 continued to 43k steps moves the front 0.9176 -> 0.8584, settles
   CD (0.0193 -> 0.0153), and relaxes the laminar mean flow ONTO the 0105 BL-march
   solution (H at x/L 0.2/0.42/0.7 = 2.515/2.556/2.611 vs march 2.533/2.561/2.613).
   The "anomalously full mean flow" (0040) and the 0105 solver-numerics conviction
   were largely diagnosing a TRANSIENT; the committed campaign fronts
   (0.908/0.918/0.925 L0/L1/L2) were unconverged — "never quote fronts from
   fixed-budget cold starts" applies to our own campaign.
   Residual truth: even converged, front 0.86 vs measured 0.438 — the remaining
   ~0.42 L miss is transport-realization + seed content, not mean flow.
3. Extraction hazard fixed: faceted-wall probes need facet-intersection ray re-basing
   (sag up to 11% of delta99 at unstr L0 faked H=2.18); committed O-grid station
   numbers verified safe (<=1.2e-6 L offsets).

Artifacts (full paths): mesh cross-sections
/home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/spheroid_mesh_xsec_{ogrid,unstr,transverse,spacing}.png
(+ captions .md same dir); verdict JSONs
/home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/spheroid_unstruct_a0_verdict_unstr_L{0,1}.json;
scripts spheroid/unstruct_spheroid.py, spheroid/build_unstruct_case.py,
paper/repro/cfd/spheroid_unstruct_a0_verdict.py, paper/repro/cfd/regen_spheroid_mesh_xsec.py;
cases /local_data/qiqi/sa-ai/spheroid_fv1/case_unstr_L{0,1}_saai_re72a0.

ACTIONS taken: full-body O-grid discriminator agent re-briefed (converge-by-CD-drift
instead of fixed 20k, facet-sag-corrected harvest, wiggle probe on converged field).
PROPOSED (user decision): converged-budget restart legs for ALL committed spheroid
campaign cases (~0.5-1 GPU-h per L1 leg, ~5 GPU-h L2) before fig:spheroidfront /
tab:sphtotals numbers are re-printed; Sec IX/X reframing stays HELD and now points to
transport-realization + seed (not mean flow / not solver numerics) as the carrier.

---

## 2026-07-28 — Fig 18 literature overlay LANDED and integrated (commits 6c9b82e, 28d8e93)

Overlay agent record: /home/qiqi/flexcompute/sa-ai/agent-paper-review/2026-07-28-0317-dragcrisis-overlay.md

- 12 series / ~940 points digitized (tick/gridline-detected calibrations, residual
  asserts, per-dataset check PNGs): Wieselsberger TN-84 (186), Delany-Sorensen TN 3038
  (219, incl. both hysteresis loops), Roshko 1961 (12, splitter runs excluded), Schewe
  + Achenbach-Heinecke (vector-exact via the Rodriguez 2015 replot), Achenbach 1968
  curve (via Catalano 2003), Rodriguez WRLES (Table 2), Catalano WMLES (3, asserted
  vs their Table 1), Stabnikov-Garbaruk SST/gamma-Re_theta/SST-KD sweeps, Stringer
  CFX+OpenFOAM, Henderson 1995, Tritton/Finn/Jayaweera low-Re (for the coming axis).
  QA: Schewe two-secondary agreement mean dCd 0.029; D&S primary-vs-secondary -0.012.
- User directives executed: cold-start diamonds REMOVED (up/dn sweeps only);
  limit-cycle bands kept with NO legend entry; legend minimal, description moved to
  caption.
- Figure: /home/qiqi/flexcompute/sa-ai/paper/figs/dragcrisis_cd_re.pdf
  (preview /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/dragcrisis_cd_re.png)
  — our 3-seed family threads the experimental crisis band; the missing transcritical
  rise (experiments climb to 0.5-0.7 above 1e6, we stay ~0.22) is now VISIBLE.
- Data/provenance: /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/litdata/dragcrisis/
  (9 open PDFs + README + JSONs + checks/), script
  /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/digitize_dragcrisis_lit.py.
- Guard: regen defaults to the completed-84-case window; rerun with --re-window full
  after the extension harvest (the live jsonl already carries 36 in-progress rows).
- MY integration (28d8e93): caption rewritten to carry the full legend description
  (sources named + digitization provenance line); 8 new bib entries, every field
  Crossref/NTRS-verified this session (wieselsberger_1921 TN-84, delany_sorensen_1953,
  roshko_1961, achenbach_heinecke_1981, catalano_2003, stringer_2014,
  stabnikov_garbaruk_2020). Build clean: 0 undefined refs, 120 pp.

## 2026-07-28 ~03:45 — UNSTRUCTURED spheroid family DONE (Flynn360 cavity pipeline) + a0 verdict: topology exonerated, the 20k BUDGET convicted

- Built the Daedalus cavity-family (Flynn360: STL skin -> BL prisms -> tet glue ->
  octree far field) spheroid meshes, FULL-circumference per your directive (no y=0
  symmetry plane): L0 0.74M / L1 2.81M nodes, ds_pole + h0/growth = the O-grid
  re65/re72 ladder values exactly, first-cell height node-exact 7.5e-7 L, MeshProcessor
  closedness + 0 bad LSQ nodes. Mesh cross-section figures (both families):
  paper/repro/cfd/figs_explore/spheroid_mesh_xsec_{ogrid,unstr,transverse,spacing}.png.
- alpha=0 Re 7.2e6, campaign case verbatim (ai_constants diff-identical): at MATCHED
  20k budget, unstr L1 = og L1 to extraction precision: chi=1 front 0.9176 vs 0.9177,
  H/P/drift all matched. The late front is NOT an O-grid artifact.
- THE BIG ONE: 20k is a cold-start artifact at this condition. Restarting BOTH
  families to 43k: fronts move 0.918 -> 0.8584 (unstr) / 0.8582 (og-L1), CD settles
  (0.0153 / 0.0127, drift 2e-5/1k), and the laminar mean flow lands ON the 0105 BL
  march (H_corr 2.52/2.55/2.61 at 0.2/0.42/0.7, gap <= 0.018): the "anomalously full
  mean flow" (0040/0105) and the committed a0 campaign fronts (0.908/0.918/0.925) are
  CONVERGENCE-STATE artifacts. Converged, the model still misses (0.86 vs measured
  0.438) — the residual is transport realization + seed, not the mean flow. Spheroid
  campaign numbers need converged-budget re-runs (a restart leg is ~0.5-1 GPU-h at L1).
- Also found+fixed an extraction hazard: analytic-origin rays on faceted walls read a
  spurious slip (facet sag dc^2*kappa/8) — H down to 2.18 on unstr L0 raw. Corrected
  extractor (Moller-Trumbore to the case's actual wall facets) committed; committed
  O-grid station numbers are safe (dH <= +0.01).
- Full record: agent-paper-review/2026-07-28-0350-spheroid-unstruct-family.md.
  Cases: /local_data/qiqi/sa-ai/spheroid_fv1/case_unstr_L{0,1}_saai_re72a0 (+snap20k),
  case_ogrid_L1_saai_re72a0_ext (campaign case untouched). GPUs 2/3 only, now free.

---

## 2026-07-28 — Extension harvest + full-body discriminator LANDED; Sec VII rewritten (4e5d39e)

**Drag-crisis extension complete** (record /home/qiqi/flexcompute/sa-ai/agent-paper-review/2026-07-28-0412-dragcrisis-extension.md, commit 1440422):
45 cases, 3.09 GPU-h; campaign total 129 cases / 8.85 GPU-h. Span Re 1 -> 2e7 = 7.3
decades, one model/constants/protocol. Creep arm validates (Re=10: -1.7% vs
Dennis-Chang; chi provably inert); seams 0.5-0.6% (300/1e3), 2.9% (2e6 cold-cold,
finer higher); NO transcritical rise in any seed/branch (floors 0.185/0.199/0.215;
chi front marches 101->82 deg but SA separation pinned ~120 deg, base Cp -0.23);
second supercritical steady state on the fine grid (dn 13-22% lower). Full-span
log-log figure with literature overlay:
/home/qiqi/flexcompute/sa-ai/paper/figs/dragcrisis_cd_re.pdf
(preview /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/dragcrisis_cd_re.png).

**Sec VII rewritten** (commit 4e5d39e, build clean 120pp/0 undefined/0 overfull):
scoped first-single-model-curve claim per litrange; mesh-family + seam paragraph;
low-Re validation paragraph (Dennis-Chang/Fornberg/Henderson now Crossref-verified
bib entries); concessions extended four -> six (no-transcritical-recovery with the
separation-not-transition attribution; second steady wake state); caption carries
the span + continuity band.

**Full-body spheroid discriminator complete** (record
/home/qiqi/flexcompute/sa-ai/agent-paper-review/2026-07-28-0413-spheroid-fullbody.md,
commit b5f591a): full-circumference L1 O-grid (4.35M nodes, periodic, watertight).
At 43k converged: arm A front 0.8581 / H 2.525/2.555/2.607 = half-model-ext 0.8582 =
unstructured 0.8584 = march-consistent. Arm B (lowMachPreconditioner) converged state
UNCHANGED (front 0.8555, CD +3.4e-4) — only accelerates convergence ~2.5x. The 0105
azimuthal one-cell wiggle COLLAPSES 500x when probe rings are re-based on wall facets
— it was facet-sag wall-distance modulation of the probe, not a flow mode. VERDICT
CHAIN CLOSED: symmetry sheets, topology, low-Mach dissipation, solver numerics ALL
exonerated; the alpha=0 story = 20k cold-start transient (+probe artifacts) on top of
a real remaining model miss (0.86 vs 0.438) owned by transport realization + seed.
Cases /local_data/qiqi/sa-ai/spheroid_fv1/case_ogridfull_L1_saai_re72a0{,_lowmach};
harvest /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/spheroid_fullbody_check.py.

**Standing user decisions:** (a) converged restart legs for all committed spheroid
campaign cases (bonus from arm B: the preconditioner legally accelerates the legs
~2.5x, though published numbers should keep canon settings); (b) Sec IX/X rewrite
direction — all discriminators now in, ready to draft on your word.

---

## 2026-07-28 — Transcritical questions answered; theta_tr(Re) extraction launched

**User prompts:** (1) is the missing transcritical rise consistent with other RANS
codes/models, and why? (2) shouldn't a higher-Re BL be MORE prone to separation at
the same angle? (3) was y+ low enough? (4) measure the transition angle vs Re by
radial rays from the surface, max-chi along each ray.

**(1) Consistency — verified from our own digitized overlay data
(/home/qiqi/flexcompute/sa-ai/paper/repro/cfd/litdata/dragcrisis/):**
- Experiment rises: Achenbach curve 0.45->0.68 over 1.7-4.3e6; Roshko 0.70-0.74 at
  6-8.7e6.
- Catalano WMLES: 0.347/0.317/0.327 at 0.5/1/2e6 — FLAT, Re-insensitive, no rise
  (their own conclusion).
- Stabnikov-Garbaruk gamma-Re_theta URANS: 0.319 at 1.29e6 and still FALLING where
  experiment rises; sweep stops at 1.3e6.
- SG20 fully-turbulent SST: 0.49 at 1.29e6, falling.
NO published computation of any class in our overlay reproduces the transcritical
rise; transitional URANS never even reaches the regime. Our result is consistent
with — and extends two decades beyond — the computational state of the art.

**(2) The physics:** for a turbulent BL at FIXED transition location, higher Re =>
relatively thinner, fuller profile => separates LATER, drag falls — that is what our
computation does (front creeps 101->82 deg while separation stays ~120 deg). The
experimental rise is the OTHER competitor winning: transition marches far forward
(toward the front quarter of the body), so the turbulent run lengthens faster than
Re thins it; theta at the shoulder grows, separation moves forward, base pressure
drops, Cd rebuilds. So the discriminating quantity is exactly the transition
LOCATION vs Re — the user's proposed plot — and the open question splits in two:
(a) does the model's attached-BL amplification move the front forward fast enough
(TS content in a favorable-gradient nose region at smooth-quiet seeds may genuinely
saturate — the experimental forward creep may carry bypass/roughness content); and
(b) SA's late turbulent separation (107-118 vs ~147 deg class) caps the rise even
given a forward front.

**(3) y+ — yes, measured from final fields (0412 record):** highre mesh y1=1e-6 D
gives y+max 0.26 / 0.49 / 0.71 at 4e6 / 7e6 / 1e7; the 2e7 stretch point runs at
1.42 (above the <=1 target, caveated inline in Sec VII; a y1=5e-7 re-mesh is the
fix if 2e7 is ever promoted beyond outlook).

**(4) LAUNCHED — theta_tr(Re) extraction (background agent, CPU-only):** radial rays
from the wall (r=0.5D, angles from forward stagnation, <=0.5 deg steps, reach
>=20 D), max-chi per ray, theta_tr = smallest angle with max-chi >= 1 (companion
threshold chi >= c_v1), across ALL 129 cases x 3 seeds x up/dn. Deliverables:
theta_tr vs Re plot (crisis band marked, house style), per-case JSON, script
paper/repro/cfd/dragcrisis_transition_angle.py, record in agent-paper-review/.
The radial-max convention also captures the subcritical lifted-shear-layer
transition the near-wall front convention misses — the record will quantify that
gap.

---

## 2026-07-28 — Spheroid uniformity + Cp(x) + BL-profile series LAUNCHED (user directive)

Package running (background agent, CPU-only) on the CONVERGED 43k full-body case
/local_data/qiqi/sa-ai/spheroid_fv1/case_ogridfull_L1_saai_re72a0 (arm B lowmach and
the converged unstructured case as one-line cross-checks):
1. Circumferential uniformity: u and p around full azimuth rings at x/L 0.20/0.42/
   0.70(/0.90), three wall-normal heights, facet-re-based origins (showing the raw
   convention once for contrast — the 0413 record proved raw analytic origins
   manufacture a fake 2.6-3% one-cell mode that collapses 500x under re-basing);
   rms + p2p per ring.
2. Surface Cp(x/L) nose to tail, meridian + azimuthal average, converged
   suction-peak location (20k field had u_e peak at x/L=0.489).
3. BL profile series: surface-normal rays at x/L ~0.10-0.88 straddling the converged
   front (0.858), tangential velocity u_t/u_e vs wall-normal distance, shared-axes
   panel series, thin-dashed Keller-box march overlay as the validated laminar
   reference; per-station H, delta99, Re_theta.
Deliverables: figures paper/repro/cfd/figs_explore/spheroid_{uniformity_*,cp_x,
bl_profiles}.png, script paper/repro/cfd/spheroid_uniformity_profiles.py, record in
agent-paper-review/, committed.

---

## 2026-07-28 — Drag crisis: separation/kernel probes added; FPG rate/onset audit launched

**User prompts:** (1) when probing transition location, also probe separation
location, max xi, max Re_Omega, and max(Omega_hat I_hat); (2) hypothesis: the FPG
behavior isn't carefully tuned — rate abysmally low vs Drela, onset possibly high;
the high-Re cylinder is the tuning case.

**Actions:**
- theta_tr extraction agent RE-BRIEFED: per case now also extracts separation
  angle(s) from wall Cf (laminar sep / reattachment / final sep, 0412 conventions),
  per-angle max chi, max Re_Omega, and max P = max(Omega_hat I_hat) along each
  radial ray (attached-region-restricted AND unrestricted); for the high-Re cases
  (4e6-2e7) it plots the theta-profiles of maxP and maxRe_Omega in the
  favorable-gradient nose region against the onset gate
  Re_Omega_c = softmin(1851.2, 124.6+1.424/P^2) — gate-blocked vs rate-starved is
  the discriminator. Headline figure now carries separation angle AND transition
  angle vs Re on one axes (their convergence through the crisis is the macroscopic
  story).
- LAUNCHED — FPG rate/onset audit vs Drela (background agent, CPU-only, DIAGNOSIS
  ONLY, no constant changes): (a) Falkner-Skan FPG ladder beta = 0..1: model
  frozen-profile rate vs Drela's e^N envelope dN/dRe_theta(H) (the repo's mfoil
  implementation), rate ratios + gate-opening Re_theta vs Drela's critical
  Re_theta_0; (b) cylinder nose map at Re_D 2e6/7e6/2e7: validated laminar march on
  the measured u_e(theta), then the three-way transition-location decomposition —
  where Drela's envelope puts it, where the model's frozen rate puts it, where the
  solver front actually sits (101->82 deg). Outcome classes: physics-limited (even
  e^N says no forward creep at smooth-quiet seeds -> experimental creep is
  bypass/roughness content), rate-limited (Drela forward, model rate too low), or
  gate-limited (onset blocks). Tuning implications will be SKETCHED (which
  constants the cylinder constrains) but not implemented pending user read.

---

## 2026-07-28 — Ultra-high-Re extension (to 1e10) launched; FPG epsilon-floor brainstorm

**User prompts:** (1) extend the drag-crisis sweep to Re = 1e10 (crisis lands
mid-span; different freestream turbulence should collapse; mesh shouldn't grow
much); (2) brainstorm FPG rate improvement — hunch: P = <Omega_hat (epsilon +
I_hat)> with epsilon = the rate an extreme FPG should have.

**(1) LAUNCHED — phase 3** (extension coordinator resumed): new 'ultra' mesh family
(y1 ~ 1e-9 D class, sized from measured Cf for y+ <= ~0.7 at 1e10, ~230 layers —
indeed a modest increase), Re ladder 5e7..1e10 x 3 seeds x up/dn, solver-sanity
pilot first (muRef ~ 1e-11 float-health check), 2e7 re-run on ultra to retire the
y+=1.42 caveat, seed-collapse verdict + front-angle merge with the theta_tr
convention, full-span figure axis 1 -> 1e10.

**(2) Brainstorm on P = Omega_hat (epsilon + I_hat)** — assessment (no
implementation; would be env-gated AI_* experimental constant defaulting to 0):
- COHERENT ON BOTH LEVERS: the onset gate is Re_Omega_c = softmin(1851.2,
  124.6 + 1.424/P^2); with I_hat ~ 0 in FPG, P -> 0 makes Re_Omega_c -> infinity —
  the gate NEVER opens in non-inflectional layers, independent of rate. Flooring P
  at Omega_hat*epsilon simultaneously (a) gives FPG layers a finite rate and (b)
  makes the gate threshold finite — one constant fixes both suspects.
- SAFETY BY CONSTRUCTION: the floor multiplies Omega_hat = Y/sqrt(X^2+Y^2), which
  vanishes in the freestream (Y -> 0), so no far-field contamination. In turbulent
  log layers Omega_hat ~ 0.2-0.3, so the floor adds ~0.19*0.25*epsilon of extra
  production — must be verified negligible in the Spalart-flow zero-suite before
  any adoption.
- SIZING CONSTRAINT: epsilon must sit well below the Blasius I_hat so the ZPG/TS
  calibration (and everything downstream of it) doesn't move; the FPG audit's FS
  beta-ladder (running) delivers exactly the I_hat(beta) and Drela-rate targets
  needed to size it — epsilon calibrates to Drela's envelope slope at H ~ 2.2
  (extreme FPG), which is small but nonzero above Drela's (large) critical
  Re_theta_0.
- THE 1e10 SWEEP IS THE INTEGRATION TEST: on the nose at Re_D = 1e10, Re_theta
  reaches the 1e4 class even in strong FPG — above Drela's FPG onset — so e^N
  itself predicts forward transition there. If the model's seeds fail to collapse
  at the top decades with epsilon = 0, and collapse with a Drela-sized epsilon,
  that is the tuning story; the three-way decomposition (Drela / model-rate /
  solver-front) from the running audit tells us whether epsilon is even the
  binding constraint at 2e7 vs 1e10.
- ALTERNATIVE SHAPES to keep on the table: epsilon entering the GATE only (lower
  onset, leave rate); a higher power of Omega_hat on the floor (biases toward
  BL-like shear, further protects turbulent zones); or epsilon(H)-free forms tied
  to existing local quantities. The audit record will carry the numbers to choose.

---

## 2026-07-28 — CORRECTION (user): the gate saturates, it does not blow up

My previous entry's claim "P -> 0 makes Re_Omega_c -> infinity, the gate NEVER opens
in FPG" is WRONG: the softmin caps the onset at Re_Omega_c = 1851.2, so in FPG the
gate opens once Re_Omega exceeds ~1851 — the correct value, just reached normally.
The FPG deficiency is therefore purely the RATE: with I_hat ~ 0 the growth through
an open gate is orders of magnitude too slow. The epsilon floor's primary action is
on the rate; its gate-side effect is secondary (a floored P lowers 124.6+1.424/P^2,
which can only move onset earlier, never later). The "gate-blocked vs rate-starved"
discriminator in the running theta_tr and FPG-audit briefs still resolves this —
expect it to return "rate-starved."

---

## 2026-07-28 — Whitepaper update launched (user directive)

Background agent updating /home/qiqi/flexcompute/sa-ai/paper/whitepaper.tex from the
current sa-ai.tex: (1) distilled Sec II — all descriptions (equations, constants
with values + one-line determinations, seed map, fv1-bypass/slowdown practicalities,
the branch-selection protocol recipe), no discussion or philosophizing; (2) all
results EXCEPT the spheroid (under revision) — flat plate incl. OpenFOAM
cross-solver, NLF, Eppler + Re-sweep + overlays, full-span drag crisis (as of
4e5d39e), Daedalus — matter-of-fact numbers/tables/figure refs copied verbatim, no
interpretation; (3) appendix — just plots, no text. Build-verified before commit;
whitepaper.tex only.

Also this hour: ultra-Re phase staged (759 MB template hardlinked; chains defined;
1e10 muRef=1e-11 sanity pilot queued behind vishal's jobs — no ladders until it
passes; the 2e7 ultra re-run doubles as the highre/ultra seam and retires the
y+=1.42 caveat).

---

## 2026-07-28 — GPU relocation (user directive): all GPU tasks -> 017-v100-dev GPUs 0-3

Vishal is using all local GPUs. The extension coordinator is re-briefed: the ultra
campaign (1e10 sanity pilot + ladders) moves to 017-v100-dev, GPUs 0-3 ONLY,
following the 2026-07-27-2006 pilot record's working pattern; remote solver must
produce a canon-identical ai_constants echo (stop-and-report if stale); local
sanity-pilot dequeued; results sync back into /local_data/qiqi/sa-ai/
dragcrisis_matrix/ so harvest/figures/theta_tr stay unified. The CPU-only agents
(theta_tr extraction, FPG audit, spheroid uniformity/profiles, whitepaper) are
unaffected and stay local. This 017-v100-dev/GPUs-0-3 rule applies to ALL upcoming
GPU work (incl. the spheroid converged restart legs when approved).

---

## 2026-07-28 — Spheroid uniformity + Cp(x) + BL profiles LANDED (record 1030)

Record: /home/qiqi/flexcompute/sa-ai/agent-paper-review/2026-07-28-1030-spheroid-uniformity-profiles.md

1. UNIFORMITY VERIFIED (converged full-body arm A): corrected u_s rings uniform to
   rms <= 5.8e-5 (p2p <= 2.2e-4) at all stations/heights; ring dCp at float32
   storage precision (several rings exactly 0); chi front azimuthally uniform to
   rms 3.9e-4 in x/L (6.4e-5 on the fully-settled arm B). The raw analytic-origin
   convention's 3.1e-2 "mode-160 wiggle" collapses 554x under facet re-basing —
   0413 mechanism confirmed on the converged state. Only surviving flow mode: an
   azimuthal mode-1 at 4e-5 class = arm A's residual transient (CL +4.7e-4).
2. Cp(x): converged suction peak at x/L = 0.4974 (u_e max; the 20k transient read
   0.489), wall Cp_min -0.0924 at 0.4948; meridian = azimuthal average to 1.4e-4;
   Bernoulli-consistent.
3. BL PROFILE SERIES (10 stations, 0.10-0.88): the converged RANS laminar BL IS the
   marched laminar solution at PROFILE level — |dH| <= 0.008 through x/L = 0.5,
   +0.013/+0.038 at 0.80/0.84 approaching the front (chi thickening + march nearing
   its own separation at 0.880), profiles overlay point-by-point; maxP rises
   monotonically 0.074 -> 0.21 into the adverse region (no aft collapse). At 0.88
   (just aft of chi=1 at 0.858) the profile is still laminar-shaped; Cf rise
   completes at 0.888.
Figures: /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/
spheroid_uniformity_rings.png, spheroid_uniformity_contrast.png, spheroid_cp_x.png,
spheroid_bl_profiles.png (+captions .md, +.json). Instrument bug ledger in the
record (per-azimuth Cf-front grouping/window bugs found+fixed before quoting).

---

## 2026-07-28 — theta_tr(Re) extraction LANDED (record 1033): V-shaped front, FPG verdict = rate-starved

Record: /home/qiqi/flexcompute/sa-ai/agent-paper-review/2026-07-28-1033-dragcrisis-thetatr.md
Figures: /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/dragcrisis_theta_tr.png
(+ dragcrisis_thetatr_fpg.png); data figs_explore/data/dragcrisis_theta_tr.json;
script paper/repro/cfd/dragcrisis_transition_angle.py. All 129 cases, ~56 min CPU.

1. theta_tr(Re) IS V-SHAPED: subcritical aloft front creeps forward 96.5 (1e4) ->
   79 (2e5), saturating 4-6 deg aft of laminar separation (73-75) — "immediate
   shear-layer transition" now measured as an angle; retreats through the crisis to
   ~98-101 (the Cd minimum = the front's local max retreat); then the transcritical
   march is MONOTONIC 2e6 -> 2e7: 13-17.5 deg forward per decade, Tu-ordered. At
   2e7 the front reaches the suction peak (83-87 deg) but no further (experimental
   transcritical class ~25-35 deg [mem]).
2. RADIAL-MAX vs NEAR-WALL convention: identical (+-1.5 deg) at Re >= 1.5e5; at
   4e4-6e4 the near-wall number is a base-region echo 53-65 deg AFT of the actual
   aloft transition; below 4e4 near-wall reports "no transition" while the shear
   layer aloft is turbulent. The user's radial convention is strictly better
   subcritically.
3. MACROSCOPIC STORY on one axes: separation and transition converge through the
   crisis, lock as the LSB supercritically (98.3/98.0 at 2e6 Tu0.2), then the LSB
   VANISHES (2e6->4e6 at Tu0.2): the wall sequence is ALREADY transcritical
   (attached transition ahead of a single turbulent separation) while Cd stays at
   the floor because that separation is pinned at ~120 deg — the 0412 verdict
   sharpened.
4. FPG HYPOTHESIS VERDICT (user's split confirmed, post-correction): the onset gate
   OPENS 15-36 deg ahead of the front at all high-arm Re (front is NOT
   onset-blocked where it sits); the RATE is the binder — max_bl rate 0.008-0.022
   vs a_max 0.19 (factor 10-20 low) because P ~ I_hat stays 0.01-0.05 in the
   uninflected FPG layer. The saturated gate ceiling (1851.2) additionally seals
   the deep nose (theta < 46.5 even at 2e7; gate alone would not open at 30 deg
   until Re ~ 1e8). Constant tuning cannot reach the 25-35 deg class; the epsilon
   floor (rate) + its gate side-effect, or the amplifying-zone-gate revision, is
   the required class of change.
5. Ledger highlights: solver-implemented Z (chained VTK gradients) is
   noise-dominated in the thin high-Re FPG layer — the ray-operator (exact
   wall-normal d2u/ds2) is quantitative there, both stored; stagnation-ray P~0.5 is
   a U->0 coordinate artifact (gated off by Re_Omega~0, excluded via theta>=10);
   Tu0.7 4e6 up separation row is a limit-cycle breather snapshot.

---

## 2026-07-28 — Whitepaper rewritten (commit e5f9382)

/home/qiqi/flexcompute/sa-ai/paper/whitepaper.tex (+ whitepaper.pdf, tracked): full
rewrite — the old version predated the sphere kernel entirely (retired
Gamma/lambda_p kernel, stale constants, superseded Eppler tables). New: 30 pp = 11
text + 19 plots.
- Sec 1 Model (distilled Sec II + App E): equations, sphere indicators (profile +
  solver Gram forms), every constant with one-line determination, fv1 bypass, Mack
  seed map, the branch-selection protocol statement (incl. the chi_inf*f
  pre-multiplied seed convention flagged as the AI_LAMINAR_SLOWDOWN pairing), one
  factual sentence on the spurious branch (Re_r 4.66-4.74e5). No discussion, no
  philosophy.
- Sec 2 Results, matter-of-fact, spheroid omitted (one under-revision line): flat
  plate + NLF + Eppler (all incl. OpenFOAM tables), Re-sweep, drag crisis Re 1-2e7
  (129 cases, seams, Dennis-Chang benchmarks, six concessions as a factual list,
  Cd table), Daedalus totals.
- Appendix: 23 plots, bare includegraphics, <=5-word identity captions, no text.
Verification: nine copied tables diffed numerically identical to paper/tables/ by
script; build 0 errors / 0 undefined / 0 overfull.

---

## 2026-07-28 — Angles figure in the paper (ddf72ce); "may coexist" fix; FPG audit LANDED

**User prompts:** (1) new plot just after the Cd plot: transition AND separation
angle vs Re, plus a plot of max-xi vs angle at representative Res; (2) qualify
"Every laminar solution coexists" -> "may coexist" in whitepaper AND paper.

**Done (commit ddf72ce, build 121pp clean / whitepaper 31pp clean):**
- NEW fig:dragcrisisangles directly after fig:dragcrisiscd, two panels: (a)
  theta_tr + first/final separation vs Re_D, 3 seeds, up/dn, authoritative grid
  families, crisis band shaded; (b) max_s chi along the radial ray vs angle at Re_D
  = 1e4/6e4/2e5/5e5/2e6/2e7 (middle seed), chi=1 dotted. Caption carries the full
  story (V-shape, saturation 4-6 deg aft of laminar separation, 13-17.5 deg/decade
  transcritical march, LSB vanishing near 120 deg, convention comparison, the Tu0.7
  4e6 limit-cycle outlier). Script:
  /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/regen_dragcrisis_angles.py ->
  /home/qiqi/flexcompute/sa-ai/paper/figs/dragcrisis_angles.pdf (preview
  figs_explore/dragcrisis_angles.png). Plots straight from the committed theta_tr
  JSON.
- "may coexist" applied at sa-ai.tex sec:bistability opener and the whitepaper
  protocol section.

**FPG rate audit LANDED** (record /home/qiqi/flexcompute/sa-ai/agent-paper-review/
2026-07-28-1041-fpg-rate-audit.md, commit 43dae78):
- FS ladder: kernel tracks Drela through beta<=0.1 (ratio 0.74-1.0), HALVES at 0.2,
  COLLAPSES 3 orders of magnitude between beta 0.2-0.5, zero at beta=1 — while
  Drela holds a >=5e-3 envelope floor. Killer located: the I_hat curvature term
  (-Z for Z>0), not a_max, not the gate constants. Onset NOT binding (gate opens
  before Drela's critical Rt through beta=0.2), per the user's softmin correction.
- Cylinder three-way at 2e6/7e6/2e7 (Tu 0.2, N=4.53 — the brief's 6.73 was
  misassigned, agent re-verified from eq:tumap): PHYSICS-LIMITED at all three —
  the nose is Drela-subcritical to ~79-85 deg, and at 2e7 the solver front (82) is
  10 deg AHEAD of e^N's chi=1 station (91.8) on its own mean flow. The high-Re
  cylinder at accessible Re CANNOT tune the FPG rate; the experimental 25-35 deg
  transcritical class is bypass/roughness content. Implication: the epsilon tuning
  case is the ULTRA arm (Re >= 1e8-1e10, where nose Re_theta clears Drela onset),
  exactly the sweep now running on 017-v100-dev; calibration instrument = beta
  0.2-0.5 wedges at Rt 2000-8000.
- Artifacts: paper/repro/cfd/fpg_rate_audit.py,
  figs_explore/fpg_rate_audit_{ladder,cylinder}.png, figs_explore/fpg_rate_audit.json.

---

## 2026-07-28 — New drag-crisis figures integrated (user directive; commit d6a344f)

- Paper now carries BOTH new figures: fig:dragcrisisangles (ddf72ce, after the Cd
  plot) and NEW fig:dragcrisisfpg (d6a344f, after it): left = FS-ladder envelope
  slope, model vs Drela-Giles/mfoil (the 3-decade collapse by beta=0.5, below-floor
  marker at beta=1); right = N(theta) budgets on the marched cylinder nose at
  2e6/7e6/2e7 vs N=4.53, solver fronts marked (they sit AT/AHEAD of the e^N
  crossing). A short factual paragraph after the six concessions states the
  verdict: FS-ladder deficit real but unexercised at accessible Re — the
  transcritical forward-transition gap at these Re is disturbance content beyond
  the envelope class. Build: 122 pp, 0 undefined, 0 overfull.
  Script: /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/regen_dragcrisis_fpg.py ->
  /home/qiqi/flexcompute/sa-ai/paper/figs/dragcrisis_fpg.pdf
  (preview .../figs_explore/dragcrisis_fpg.png).
- Whitepaper revision agent re-briefed to inline BOTH new figures in its
  drag-crisis block (its full revision — figures into sections, minimal Sec 2,
  appendix = main-paper appendix figures, no text — is in progress).

---

## 2026-07-28 — Why separation doesn't follow the front; FT-SA control launched

**User prompts:** transition moves upstream supercritically but separation doesn't
move forward — why? Is the missing rise the transition model's fault or SA's? Run
fully-turbulent SA on the high-Re cylinders for separation angle and drag.

**Why (mechanics):** moving transition 100 -> 82 deg adds only ~20 deg of turbulent
run ahead of the separation point, while each decade of Re thins the turbulent
layer; in SA's modeled balance the thinning wins and the turbulent layer stays
attached to ~120 deg regardless. The experimental transcritical rise rides on
transition moving to the 25-35 deg class — tripling/quadrupling the turbulent run —
which the 1041 audit shows is beyond envelope (e^N) physics at these Re and seeds.
Quasi-evidence already in hand: at 2e7 the three seeds' fronts span 77-87 deg yet
Cd spans only 0.185-0.215 — the drag barely feels the front.

**LAUNCHED — fully-turbulent SA control** (coordinator, 017-v100-dev GPUs 0-3):
standard SA (AI terms off via the canonical switch, chi_inf = 3-5 fully-turbulent
freestream), Re = 2e6/4e6/7e6/1e7/2e7 on the highre mesh (+1e10 on ultra if the
sanity gate has passed), steady, cold two-stage, harvest Cd / separation / base Cp
/ shoulder Cp vs the SA-AI values. Discriminator, stated in advance: FT-SA at ~120
deg / Cd ~0.2 / base Cp ~ -0.23 convicts baseline-SA turbulent separation + the
steady dead-air wake and exonerates the transition model (whose front is already
e^N-consistent); a materially different FT-SA separation re-opens the question.

---

## 2026-07-28 — FPG kernel recalibration study launched (user directive: Fig 4 to match Drela-Giles at stagnation H)

Target = fig:calibrate, BOTH panels at H down to 2.216: (a) rate must not collapse
below Drela in the favorable regime; (b) the N=1 onset Re_theta must track the
Drela-Giles station instead of departing above. Study agent (offline, analytic
rigs, no solver/tex changes) explores: (1) the user's epsilon floor
P = Omega_hat(epsilon + I_hat) — may also improve the low-beta mean-secant (0.84 at
Blasius); (2) the harder onset half: a naive floor + existing gate books ~15
spurious e-folds in strong FPG (gate opens Rt 1150-1280 vs Drela critical
3757-6644) — candidate fix conditions the softmin ceiling on the inflection state
so non-inflectional layers keep the rising 1/P^2 branch; (3) hard regression
constraints: Blasius anchor unchanged, attached family +-12% tracking, separated/
Stewartson branch untouched, Spalart zero-suite (parasitic production from
Omega_hat~0.25 in log layers quantified), Mack map unshifted. Deliverables:
candidate-form table, CANDIDATE Figure 4 (figs_explore only), constants +
determinations, env-gated (default-OFF) solver sketch; in-solver validation
testbeds = the running ultra-Re cylinder sweep + spheroid re72a0. Ledger must state
the DG-vs-mfoil ~60% target ambiguity at stagnation H.

**User clarification:** the recalibration is exactly the substitution
P = Omega_hat*(I_hat + epsilon) everywhere P appears (rate AND gate, the floored P
flowing into both; no separately designed gate branch). Study agent re-briefed:
headline = best single-epsilon fit of both Fig-4 panels under this exact
substitution + full regression suite at that epsilon; any residual panel-(b)
mismatch quantified honestly with the minimal closing change noted as a secondary
observation only.

---

## 2026-07-28 — Scenario walkthrough for the epsilon floor (user: does eps ~ 1e-2 make a
## visible difference? best guess now 0.03)

Study agent evaluating at eps = 0.03 (headline), 0.01, and the trade-off optimum.
My scenario-by-scenario expectation (crude slope(P) mapping from the 1041 ladder —
the study's instrument numbers supersede these):
- Blasius/flat plate: band P_floor ~ Omega_hat*eps ~ 0.02 at eps=0.03 on top of
  maxP 0.078 -> rate +25-30%; late-window secant (currently 1.00 vs Drela) -> ~1.27
  — VISIBLE regression of the ZPG anchor; all flat-plate onsets move ~20% in Re_x.
  At eps=0.01: +6-13% — borderline visible.
- Mild FPG beta 0.05-0.1 (NLF/Daedalus LE, spheroid mid-body): +40-60% rate at 0.03
  -> spheroid front from 0.858 toward ~0.7-0.75 class, airfoil fronts visibly
  upstream. At 0.01: +13-19%.
- Stagnation H (Fig 4 target): eps=0.03 -> slope ~0.003-0.0044 vs Drela 0.0074 —
  about HALF; eps=0.01 -> ~0.0011-0.0017, a fifth. Full match needs eps ~ 0.06-0.10
  which would near-double Blasius. The single-eps tension is real; the trade-off
  curve is the decision instrument.
- Onset side-effect at 0.03: floored P makes 1.424/P^2 ~ 1600-3200, comparable to
  the 1851 ceiling -> gate opens Rt ~ 1100-1500 in strong FPG vs Drela critical
  3757-6644 -> up to ~16 spurious e-folds available on LONG high-Rt FPG runs
  (only very-high-Re noses have such runs: ultra-arm cylinder — where it is the
  DESIRED effect; airfoil LEs at Re_c few-e6 have Rt < 300, gate closed, safe).
- Inflected side (Eppler LSBs, subcritical cylinder shear layers, separated
  branch): P 0.2-1 -> +2-10% at 0.03 — largely invisible; bubble physics intact.
- Transcritical cylinder 2e6-2e7: nose gate opens ~Rt 1150-1500; extra nose
  e-folds ~0.003*(Rt_shoulder - Rt_gate): ~0.2 at 2e6 (invisible), ~2-7 at 2e7
  (front pulled visibly forward); at 1e8-1e10 (ultra) the floor books the full
  budget on the nose -> forward creep + seed collapse arrive — the desired
  transcritical content.
- Frozen-Hiemenz / Sec VIII: stagnation flow is exactly the eps-active zone — the
  AI term now assists the attachment wedge; the critical bracket 4.66-4.74e5 and
  the protocol section numbers need re-verification under any adopted eps.
- Spalart zero-suite: log-layer Omega_hat ~ 0.2-0.3 -> parasitic P ~ 0.006-0.009 at
  0.03; production increment expected small vs SA's own — must be verified.
NET: eps = 0.03 is NOT a free tweak — it visibly perturbs the calibrated ZPG/mild-
FPG suite (unless jointly re-anchored) while buying ~half of Drela's stagnation
rate; eps = 0.01 is nearly free on existing results and still decisive on the ultra
arm. The study's trade-off curve + impact table (both eps values) will make this a
numbers decision; in-solver validation testbeds: ultra cylinder + spheroid re72a0.

---

## 2026-07-28 — Spheroid three-way DECOMPOSITION LANDED (record 1122): the model is
## e^N-faithful on this flow; nothing envelope-class transitions at 0.438 at this seed

Record: /home/qiqi/flexcompute/sa-ai/agent-paper-review/2026-07-28-1122-spheroid-threeway.md
Figure (paper-quality, airfoil-style dual axis, per user directive):
/home/qiqi/flexcompute/sa-ai/paper/figs/spheroid_threeway_N.pdf
(preview .../repro/cfd/figs_explore/spheroid_threeway_N.png; JSON spheroid_threeway.json;
script paper/repro/cfd/spheroid_threeway.py)

HEADLINE: on the converged mean flow the transported chi TRACKS THE DRELA-GILES
ENVELOPE ONE-FOR-ONE to x/L ~ 0.7; even a perfect envelope tracker at this case's
flight-quiet budget N = ln(1/chi_inf) = 11.645 (seed verified from the case JSON +
in-field plateau) commits only at 0.735 vs measured 0.4381. Decomposition of the
0.420 L: +0.162 envelope-method offset (Stock's own TS e^N books N=8 by 0.425
where the Drela-fit family books ~5) + 0.135 seed budget (tunnel N=8 vs flight
11.65) - 0.061 frozen-rate SURPLUS (instrument runs 0.9-1.45x Drela mid-body)
+ 0.185 transport realization (3.5 e-folds). The only deficient model physics on
this flow is the aft steepening-adverse run 0.70-0.86 (envelope demands 30->126
e-folds/L, model delivers ~20) — why the front parks at 0.858 just ahead of
laminar separation (0.880). SEED LEVER AT FULL STRENGTH on the converged field:
N=9 -> front 0.689, N=7 -> 0.537, and at the measurement-implied budget (N~5.6,
Tu~0.13%, DFVLR-tunnel class) the model's own front lands ON 0.438. No rate or
transport fix can reach 0.438 at the flight-quiet seed (6.4 e-folds forbidden by
the correlations). Supersedes the 0040 e^N section (20k-transient state). The
1041 FPG cliff is NOT implicated (mild gradients here; the aft deficit is
steep-ADVERSE, a different regime).

Figure + draft caption staged for the Sec IX/X rewrite (held per user read).

---

## 2026-07-28 — Re-seeded spheroid campaign LAUNCHED (user directive)

**User prompt:** is re72a0 purely a freestream-turbulence mis-specification? Rerun
with the correct freestream; also the lower-Re alpha=0 case and the alpha=2.5 case
if freestream+IC chi is confirmed as the main fix.

**Answer given:** operationally yes with a qualifier — 0.30 of the 0.42 L is
disturbance-content class, 0.12 L is model residual (transport + aft-adverse) that
largely cancels at tunnel-class seed on this flow; per-case verification rides
with the reruns.

**LAUNCHED (background agent, 017-v100-dev GPUs 2-3):** 5 converged-protocol L1
runs (CD-drift criterion, ~40-45k steps — the 0350 lesson applied): re72a0 with a
3-seed bracket (center from the DOCUMENTED DFVLR tunnel Tu if found in
references/, else the measurement-implied ~0.13% class, stated honestly — bracket
guards against tuned-to-fit circularity), the re65 alpha=0 case and the alpha=2.5
case (L1 build at the re72a2p5 conditions) at center seed. IC + freestream chi
both carry the new seed (physical, laminarSlowdown=1.0 convention). Harvest:
facet-re-based fronts vs the Stock/Kreplin digitized measurements and vs the 1122
seed-lever prediction (the bracket directly tests the x(N) table); alpha=2.5 gets
windward/leeward front comparison as the crossflow check. Cases tagged _reseed,
nothing clobbered. Record to agent-paper-review/.

---

## 2026-07-28 — Kernel floor redesigned: softmax_2, and the onset ceiling becomes removable

**User:** additive floor too soft (hits Blasius/mild-FPG too hard); use softmax_2.
What does it do, and does it remove the softmin constant on onset Re_Omega?

**Analysis (study agent re-briefed, headline eps=0.03):**
- P = Omega_hat*softmax_2(I_hat, eps) = Omega_hat*sqrt(I_hat^2+eps^2): quadratic
  crossover — Blasius +3.7% (vs +27% additive), mild FPG +17% (vs +60%), deep FPG
  the full floor unchanged. Sign convention to settle: clip I_hat at 0 first
  (raw form is even in I_hat and would boost strongly-stable layers; Drela
  envelope slopes never go negative — stability belongs to the onset).
- YES on constant removal: with P floored, Re_Omega_c = 124.6 + 1.424/P^2 needs NO
  softmin ceiling — the blow-up caps intrinsically at 124.6+1.424/(Omega_hat*eps)^2.
  Numerical coincidence: Omega_hat*eps ~ 0.029 reproduces the old 1851.2, i.e.
  eps ~ 0.03-0.04 at band Omega_hat — the user's guess. Net constants: drop 1851.2,
  add eps — count unchanged, cleaner formula, eps double-duty (rate floor + onset
  cap). To verify: the Omega_hat-dependence of the new cap (free-shear/handover
  onset that 1851.2 anchored), and panel-(b) FPG onset (new cap sits ABOVE 1851
  toward Drela's rising Rt0 — likely improves for free). Hiemenz/Sec VIII +
  Spalart-suite checks remain mandatory.

---

## 2026-07-28 — Floor redesign #2 (user): argument becomes eps*(-Z), preserving the
## wall-vanishing rate

**User:** any constant floor (additive OR softmax) destroys the "rate -> 0 toward
the wall" requirement that keeps the frozen-profile eigenvalue bounded at the
high-Re extreme; the correct softmax argument is eps*(-Z).

Study agent re-briefed: primary form P = Omega_hat * softmax_2(I_hat, eps*(-Z)/R)
(dimensional normalization matching I_hat, to be verified from kernel source).
Properties to verify as first-class gates: (a) wall asymptotics — eps*(-Z)/R
vanishes like d^2 (Z ~ d^2 u''(0)), eigenvalue boundedness at high Re restored;
(b) sign map across profile classes with clip convention ((-Z) floor active
precisely in non-inflectional favorable-curvature layers, inert elsewhere);
(c) eps recalibrated on Fig 4 under the new form (constant-eps 0.03 numbers
superseded); (d) onset-ceiling removal re-evaluated — the intrinsic cap is no
longer a global constant with a profile-dependent floor, so the softmin/1851.2
drop must be re-justified or retained (both variants' panel-b reported). All
regression gates stand.

---

## 2026-07-28 — FPG RECALIBRATION STUDY LANDED (record 1218): the eps*(-Z) form wins

Record: /home/qiqi/flexcompute/sa-ai/agent-paper-review/2026-07-28-1218-fpg-recalibration.md
Candidate Fig 4: /home/qiqi/flexcompute/sa-ai/paper/repro/analytic/figs_explore/model_calibrate_candidate.png
(+ trade-off curves, per-form JSONs, script paper/repro/analytic/fpg_recalibration_study.py)

RECOMMENDATION: **form zc — P = Omega_hat*sqrt(<I_hat>+^2 + <eps*(-Z)/R>+^2), gate
124.6+1.424/P^2 with the softmin ceiling DROPPED (net constants unchanged:
-1851.2 +eps), eps = 0.12 minimax** (0.08 if minimal Blasius drift preferred,
0.16 if stagnation rate prioritized; full trade-off curve recorded).
- WALL-BOUNDEDNESS (the user's requirement): zc PASSES (eigenfunction peak fixed
  at y*/theta=2.0 from Rt 1e5 through 1e7); BOTH constant floors FAIL (peak
  migrates wallward without limit) — exactly as predicted. Raw (unclipped) I_hat
  rejected separately (log-layer parasitic 0.14 of SA production).
- Fig 4 at eps=0.12: panel (b) lies ON the DG N=1 curve across the whole
  attached+separated family; worst residual confined to H<=2.29 (rate 0.60x,
  onset 1.9x early — partially canceling; inter-fit DG-vs-mfoil ambiguity is
  ~60% there anyway). Blasius: +3.2% late secant / -4.5% Rt1 / -7.1% Re_x(N9),
  absorbable by the usual k re-anchor if desired.
- OFF-TARGET FOOTPRINT (the decisive discriminator vs constant floors):
  subcritical cylinder noses CLEAN (0.03/0.44 sup e-folds at 2e6/2e7 vs ~22
  spurious for constant floors = bypass content); Sec VIII Hiemenz bracket
  UNAFFECTED (0.01 e-folds); spheroid re72a0 front shift only ~0.011 L; LSB/
  separated branch <=0.2%; Spalart zero-suite 4.4e-4 vs canon 4.0e-4.
- ULTRA ARM: at Re_D=1e9 zc books ~69 sup e-folds by 80 deg (canon 1.9) — the
  transcritical front collapse arrives, and the nose there is DG-supercritical
  (e^N-consistent forward creep, not bypass).
- CEILING REMOVAL: strictly improves panel (b); the LST graze cargo the ceiling
  protected re-verified intact (attached graze 0.98-1.24 vs canon 0.93-1.14,
  separated side unchanged).
- User's eps=0.03 was EXACTLY the optimum of the constant-softmax form (good
  guess) — but that form fails the boundedness gate.
- Solver sketch: two-line change at both kernel sites, env AI_ZFLOOR_EPS
  (default 0.0 = bit-identical canon) + AI_REOMC_CEIL=1e30; campaign JSONs
  untouched.
- In-solver validations queued on user go: ultra cylinder (expect no front
  motion <=2e7, collapse on ultra arm) + spheroid re72a0 (expect <=0.01-0.02 L)
  + flat-plate ladder for the Blasius drift (or re-anchor k first).

DECISION WITH USER: adopt zc? at which eps (0.08 / 0.12 / 0.16)? re-anchor k or
accept the -4.5% Rt1 drift? then env-gated solver implementation + validation runs.

---

## 2026-07-28 — Recalibration Part II launched: separate epsilons vs rate-only floor;
## naming question

**User:** growth and onset need separate epsilons (growth wants higher, onset then
too early) — tune separately; OR epsilon in the rate kernel only with the onset
keeping its old softmin; see which is better. Also: better name for epsilon?

Study agent resumed with both variants: A = two-epsilon (eps_r ~0.16-0.17 for the
stagnation rate; eps_o tuned to put N=1 ON the DG station; no-ceiling gate on its
own floored P_o); B = rate-only floor with the CANON softmin gate on un-floored P
(expected weakness quantified: un-floored P->0 saturates the 1851.2 ceiling ->
opens Rt~1150-1280 vs DG 3757-6779, floored rate then amplifies immediately).
Comparison vs single-eps zc@0.12 on both panels, constant bookkeeping (A and B are
net +1 constant; zc was net 0), wall-boundedness, off-target impact rows, ultra-arm
behavior. NAMING candidates to be judged in the record: c_TS (physical: the floor
IS the viscous Tollmien-Schlichting branch, active where inviscid inflectional
content vanishes — my recommendation), c_kappa/c_curv (mechanistic), c_FPG
(regime-descriptive), a_visc (rate-family pairing with a_max).

**Naming (user prefers a_visc/a_TS; a_max rename?):** recommended the TWO-BRANCH
REFACTOR — move the prefactor inside the softmax so both classical instability
branches carry named rates: rate = clip(Omega_hat*softmax_2(a_R*<I_hat>+,
a_TS*<-Z>+/R))*gate with a_R = 0.19 (Rayleigh/inflectional, the renamed a_max,
eigenvalue determination unchanged) and a_TS = a_max*eps (~0.023 at eps=0.12) as
its own constant with its own Drela-stagnation determination — epsilon disappears
as a separate symbol. Mathematically identical by softmax_2 homogeneity (agent
verifying the exact clip composition). Fallback naming if "Rayleigh" overclaims:
a_I (inflectional) / a_TS. Rename is a single mechanical commit at adoption time.

---

## 2026-07-28 — Recalibration Part II LANDED (f08a998): separate epsilons infeasible;
## single-constant two-branch form stands

Record updated in place: /home/qiqi/flexcompute/sa-ai/agent-paper-review/2026-07-28-1218-fpg-recalibration.md
(script paper/repro/analytic/fpg_recalibration_study.py, forms add/sm2raw/sm2clip/
zc/zc_ceil/zc2/zb; JSONs+figures in paper/repro/analytic/figs_explore/)

- VARIANT A (two epsilons) REJECTED — structurally infeasible as directed: a gate
  tuned to the DG station at beta=1 never saturates its ramp, so the late secant
  asymptotes ~0.5x REGARDLESS of eps_r; compensating with eps_r=0.35 destroys
  Blasius (-37% Re_x). Dominated by zc.
- VARIANT B (rate-only floor eps_r=0.1455, canon softmin gate): panel (a) PERFECT
  (0.99-1.09x family-wide) but 4.4x-early strong-FPG onset, 4 spurious nose
  e-folds at Re_D=2e7 (bypass content zc avoids), and a new ceiling strike — the
  canon gate opens in log layers at y+~760, leaking 3.9e-3 raw parasitic (10x zc).
  FALLBACK only if slope exactness is the overriding goal.
- RECOMMENDED FINAL FORM (two-branch, epsilon eliminated): Q = Omega_hat *
  softmax_2(a_R*<I_hat>+, a_TS*<-Z>+/R), rate = min(a_R, <Q>+), gate
  124.6+1.424/P^2 on P = Q/a_R, ceiling dropped — NET CONSTANT COUNT ZERO.
  a_R = 0.19 (renamed a_max, Michalke determination untouched); a_TS = 0.0228
  (minimax, eps=0.12) or 0.0304 (slope-priority, eps=0.16). Identity with the
  eps form verified to 5e-18; gotcha: the clip ceiling must be a_R, not 1.
- USER DECISION SET: adopt two-branch a_R/a_TS at 0.0228 vs 0.0304 (or fallback B);
  re-anchor k x1.066 / a_max x0.969 or accept -4.5% Rt1 drift; then env-gated
  solver implementation (AI_ZFLOOR_EPS + AI_REOMC_CEIL=1e30, default OFF) and the
  two queued validations (drag-crisis cylinder front discriminator; spheroid
  re72a0 front).

---

## 2026-07-28 — User clarifies the intended structure: variant B IS the design

**User:** the rate has two terms — softmax(a_inviscid*<Omega_hat,I_hat> +
a_visc*<Omega_hat,-Z/R>) — and the ONSET GATE never sees a_visc, keeping its own
softmax(=softmin) on Omega_hat*I_hat. Also: a_max*eps becomes the promoted a_TS/
a_visc constant.

This maps exactly onto the study's variant B (zb form) by softmax homogeneity,
with a_visc = a_max*eps_r = 0.0276 at its panel-(a)-optimal calibration. Study
agent resumed for Part III: B presented as the first-class candidate — B-focused
candidate Fig 4 at a_visc 0.0276 + a leaner 0.023; the two warts quantified for
decision ((i) strong-FPG early onset -> expected spurious front shift at 7e6/2e7
vs the 1033 theta_tr baseline; (ii) log-layer leak -> sigma_t-BLENDED parasitic
vs canon 3.0e-4 and which zero-suite flows exercise y+~760); symmetric zc-vs-B
impact table; the two-term naming block (a_inviscid short forms a_R/a_I/a_inv +
a_visc/a_TS with its own Drela-stagnation determination).

**User extension:** the 1851.2 gate ceiling is not sacred in the B structure —
retune it. Part III becomes a JOINT (a_visc, C_ceiling) optimization. Expected
double win being verified: a raised C (Drela-centering class ~6000-10000, since a
fixed C maps to Rt 3.3-5.2x below Drela's rising Rt0 across beta 0.5-1) both
delays the strong-FPG onset toward Drela AND pushes the log-layer gate opening
(y+~760) deeper or out — potentially curing both B warts. Constraints: LST graze
re-check (canon C came from the fig02 graze envelope whose favorable side the
paper already flags as 1.27-1.34 high — the retune may improve it); adverse/
separated side ceiling-independent (confirm). Deliverables: joint optimum, both
panels vs zc and canon, re-quantified warts, impact rows, C's new determination
statement in the naming block.

---

## 2026-07-28 — Cd(Re) overlay restyle + low-Re DNS series LAUNCHED (user directive)

**User:** fig 20 in the whitepaper (= the Cd(Re) literature overlay, shared PDF with
the paper's fig:dragcrisiscd): distinguish computations from experiments — symbols
without lines for experiments, lines without symbols for computations; add low-Re
unsteady DNS runs so nobody says "but there are calculations matching experiments
here" in the shedding band.

Agent launched: strict semantic restyle (experiments = symbols only, incl. the
Achenbach-1968 polyline rendered as sparse symbols; computations = lines only;
SA-AI family keeps its distinct subject styling); Henderson 1995 unsteady
spectral-element drag (already digitized) promoted into the figure as a
computation line; 1-2 additional unsteady 2D-DNS Cd(Re) sweeps in Re 47-1000
acquired + digitized with full provenance/QA (+ optional Re=3900 3D anchor);
figure regenerated (whitepaper inherits automatically via the shared PDF); draft
caption + Crossref-verified bib entries delivered for my integration.

---

## 2026-07-28 — Overlay restyle INTEGRATED (7fb7ac2); recalibration Part III LANDED
## (00fa057): the user's structure with retuned ceiling is the adopted recommendation

**Overlay (agent commit bb57d1a + my integration 7fb7ac2, build 123pp clean):**
- Strict semantic in figure + caption: symbols = experiments (Achenbach curve +
  Wieselsberger fairing now sparse symbols along their polylines — they no longer
  read as computations); lines = computations; SA-AI keeps subject styling.
- Shedding-band unsteady computations added: Henderson 1995 spectral-element
  shedding-mean (promoted) + Qu et al. 2013 DNS Table-3 transcription (8 pts,
  Re 50-200, verbatim-asserted vs the OA PDF) + Dong-Karniadakis 2005 3-D DNS
  anchor at Re=1e4 (capped tick) — unsteady computations recover the measured
  level exactly where our steady branch sits below by construction.
- Tritton/Finn/Jayaweera low-Re symbols finally attributed (Veysey-Goldenfeld
  2007). 3 new Crossref-verified bib entries. SA-AI data md5-identical (styling+
  additions only). Whitepaper inherits via the shared PDF, no wording change.
- Figure: /home/qiqi/flexcompute/sa-ai/paper/figs/dragcrisis_cd_re.pdf
  (preview .../repro/cfd/figs_explore/dragcrisis_cd_re.png); record
  /home/qiqi/flexcompute/sa-ai/agent-paper-review/2026-07-28-1321-dragcrisis-overlay2.md

**Recalibration Part III (00fa057; record 1218 final):** the user's structure WITH
the retuned ceiling is now the adopted recommendation —
rate = min(a_inv, softmax_2(a_inv*Omega_hat*<I_hat>+, a_visc*Omega_hat*<-Z>+/R)),
canon gate on un-floored Omega_hat*I_hat (gate blind to the viscous branch),
a_inv = 0.19 (renamed a_max), a_visc = 0.0276, C = 8000 (retuned; determination:
"centers the Drela-Giles critical stations of the favorable family"; the old
1851.2 came from the graze saturation the paper already flagged 1.27-1.34 high).
Net +1 constant.
- Both panels max dev 1.77x — BEATS zc@0.12 (1.90x); Blasius intact (+5.1% late /
  -2.6% Rt1 / -8.2% Re_x(N9)); candidate figure
  /home/qiqi/flexcompute/sa-ai/paper/repro/analytic/figs_explore/model_calibrate_candidate_zb.png
  — panel (b) joint-opt curve lies ON the DG station family-wide incl. stagnation.
- BOTH warts cured by C=8000: no spurious nose crossings at 2e6-2e7; ultra arm
  collapses to 35.4 deg (the experimental transcritical class; canon-C gave 12);
  log-layer gate needs y+~3300 (leak gone), blended parasitic 3.1e-4 = canon.
- Graze: adverse/separated pinned exactly; mild-favorable members move TOWARD
  their LST anchors (the retune corrects the flagged bias); ledger: beta=0.15
  member undershoots to 0.607.
- Negative findings: two-eps structurally infeasible; exact stagnation slope
  unattainable at a centered gate (rate-gate coupling; even a_visc=a_max reaches
  0.93x). The min(a_inv, .) ceiling clip is REQUIRED (free-shear overshoots ~30%
  without it — the user's shorthand formula omitted it).
- USER DECISION: adopt (a_inv 0.19 / a_visc 0.0276 / C 8000)? Then env-gated
  solver impl + the two validations (drag-crisis front discriminator at 2e7 +
  ultra arm; spheroid re72a0 front).

---

## 2026-07-28 — Part IV launched: onset-graze figure enriched to stagnation (user directive)

**User:** the zb candidate's panel (b) shows onset Re_theta too FLAT vs H at low H —
the softmin saturation (un-floored P -> 0 => threshold -> constant C) while Drela's
Re_theta0 keeps rising toward stagnation. Diagnose, starting by enriching Figure 2
(whitepaper) / fig02 onset-threshold graze with more FPG members, all the way to the
stagnation-point profile.

Study agent resumed (Part IV): FS beta {0.35, 0.5, 0.7, 1.0} added with LST neutral
points computed by the figure's own canonical method (to be identified and stated;
published OS critical Reynolds numbers as the defensible fallback near stagnation).
Deliverables: (a) enriched CANON figure (same gate, same styling, more members) as a
candidate PDF for my integration into paper + whitepaper, with the graze-ratio table
(the paper already flags beta>=0.25 at 1.27-1.34 high — the new members quantify the
growth of the miss toward stagnation); (b) DIAGNOSTIC companion (figs_explore only):
retuned C=8000 threshold + the strong-FPG members' (Re_Omega, P) loci + candidate
threshold SHAPES that would track the rising neutral points (diagnosis only, no
refit of canon); (c) the numbers table: each new member's neutral Re_Omega/Re_theta
vs canon threshold and vs C=8000.

---

## 2026-07-28 — Part IV LANDED (cce074b): the low-H onset flatness is the COORDINATE
## dying, not the ceiling

Record: Part IV appended to
/home/qiqi/flexcompute/sa-ai/agent-paper-review/2026-07-28-1218-fpg-recalibration.md
Candidate enriched figure:
/home/qiqi/flexcompute/sa-ai/paper/repro/analytic/figs_explore/onset_graze_enriched.pdf
(+png; diagnostic onset_graze_flatness.png; tables onset_graze_enriched.json;
script paper/repro/analytic/fig02_onset_graze_enriched.py; canon fig02 untouched)

- Canonical neutral-point method identified: Drela-Giles Eq. 30 Re_theta0(H)
  correlation (not per-profile OS); same method extends smoothly to H=2.216;
  Hiemenz cross-check vs published OS critical (Wazzan-Okamura-Smith,
  Re_delta*~12490) agrees ~15% [mem-flagged for citation at integration].
- Enriched graze table: legacy members 0.91-1.14 unchanged (reproduce canon to
  printed digits); NEW: beta=0.35 -> 1.27 (lands exactly on the paper's flagged
  band), beta=0.5/0.7/1.0 -> 2.33/2.94/3.70 vs the ceiling (Re_Omega* =
  6053/7630/9624; NO P>0 locus — drawn as left-edge arrows).
- DIAGNOSIS (answers the user's "softmax makes it too flat?"): beyond beta~0.4 the
  gate coordinate Omega_hat*I_hat is <= 0 across the ENTIRE profile (exact
  curvature; the 1041 audit's ~5e-4 frozen readings were estimator noise —
  erratum recorded). So EVERY threshold Re_Omega_c(P) — capped, uncapped, any
  softmin exponent — presents its constant P->0 limit precisely where the
  critical Reynolds number rises fastest (Re_Omega* ~ 1.2-1.6 x Re_theta0). The
  flatness is the death of the gate coordinate at low H, not a mis-set ceiling;
  any constant C only splits the difference.
- Shape candidates (none adopted): uncapped 1/P^2 — dead coordinate; softmin
  exponent — limit still C; rising branch in the viscous coordinate
  P_o = Omega_hat<-Z>+/R — locus exists but ill-conditioned (P_o spans only
  0.129-0.160 across strong FPG); the observed law Re_Omega* ~ 1.45 Re_theta0
  tracks trivially but requires integral-scale information — leaves the
  pointwise-local class. That structural tension is the Part-V question.
- Two earned gotchas recorded: repo FS ODE convention f''' = -[f f'' +
  beta(1-f'^2)]/(2-beta); the beta>=0.5 profiles' Omega_hat*I_hat <= 0 everywhere.

---

## 2026-07-28 — Marble sphere deepened (bb43de5); FT-SA control headline: attribution REVISED

- Fig 1 re-baked per user feedback (floor 0.66, lit side capped 0.97 — no page-white
  interior pixels; glint 0.26, rim 0.12); committed bb43de5, both documents rebuilt.
  Preview: /home/qiqi/flexcompute/sa-ai/paper/repro/analytic/figs_explore/indicator_sphere_new.png
- ULTRA/FT-SA milestone (one case left): (1) seed spread shrinks to 5-8% at 3e9-1e10
  with order scrambling — partial collapse consistent with saturation + wake-state
  multiplicity; (2) FT-SA sits SUBSTANTIALLY ABOVE SA-AI at matched Re (0.246-0.371
  vs 0.19-0.23): turbulent-from-the-nose thickens the shoulder BL and separates
  earlier — yet still ~2x under the experimental transcritical 0.55-0.7.
  ATTRIBUTION REVISION: the missing rise is BOTH (a) transition too aft (moving it
  to the nose demonstrably lifts Cd toward experiment — which the a_visc/blended-
  gate revision would supply at high Re) AND (b) a residual baseline-SA/steady-wake
  deficiency (~2x) that persists even fully turbulent. The earlier "separation
  pinned, transition exonerated" framing (0412) was half the story; knee/base-Cp
  harvest will quantify the split.

---

## 2026-07-28 — Part V LANDED (fa3e2b2): the user's blended gate WINS (softmax_2 variant);
## final recommended form supersedes Part III

Record: Parts I-V in
/home/qiqi/flexcompute/sa-ai/agent-paper-review/2026-07-28-1218-fpg-recalibration.md
Candidate Fig 4: paper/repro/analytic/figs_explore/model_calibrate_candidate_vbs.png

FINAL FORM (vbs): rate = min(a_inv, softmax_2(a_inv*Ohat<Ihat>+, a_visc*Ohat<-Z>+/R));
gate = 124.6 + 1.424/P_gate^2, P_gate = softmax_2(<Ohat*Ihat>+, c_o*Ohat<-Z>+/R),
NO ceiling. Constants: a_inv = 0.19, a_visc = 0.0276, c_o = 0.105 — NET-ZERO count
(the 1851.2 ceiling exchanged for c_o).
- The user's H-tracking intuition VERIFIED to +-10%: c_o*P_o tracks the required
  sqrt(1.424/Re_Omega*) at constant ratio through strong FPG — genuine H-tracking,
  not just boundedness.
- Linear blend (user's stated preference) not adoptable: at beta 0.05-0.15 the
  viscous term is 30-81% of the inviscid one -> graze 1.38-1.89, Blasius -15.7%.
  softmax_2 suppresses the cross-term quadratically: legacy graze 0.89-1.20 (canon
  0.93-1.14), adverse/separated <1%.
- Panel-(b) low-H flatness RESOLVED (rises continuously along the DG N=1 station to
  H=2.2; residual beta=1 earliness 0.62x = a quantified tanh-ramp-tail interaction).
  Max two-panel dev 1.63x — best of all forms (Part III 1.77x, zc 1.90x).
- Every wart closed: log-layer leak FULLY closed (stable-layer threshold ~1.7e6,
  never reached — cleaner than C=8000); gate closed at the wall (desirable,
  boundedness saturates y*/theta=2.0); noses clean 0.03/0.08/0.28; ultra arm
  29.8 deg; Hiemenz 0.013 e-folds; spheroid +1.29 (~0.011 L); Blasius -5.0% Rt1.
- Naming: trio a_inv / a_visc / c_o (c_o a dimensionless weight, not a rate).
  Curious ledgered observation: c_o/eps_r = 0.719 ~ k = 0.712.
USER DECISION: adopt vbs (a_inv 0.19 / a_visc 0.0276 / c_o 0.105)? Then env-gated
solver impl + validations (cylinder fronts at 2e7/ultra; spheroid re72a0), and the
Fig 2/Fig 4 canon updates ride with adoption.

---

## 2026-07-28 — Fig 20 restyle refinement + Part VI (a_visc grounding, c_o/eps_r=k)

**User (fig 20):** OUR computations also line-only no markers; experiments marker-only
no lines; single-Re computations = short horizontal line stub; ALL series slightly
transparent (crowded figure). Agent launched (a8e7f89c29888fd7f): SA-AI family ->
line-only (up solid / dn dashed, seed colors, markers removed; stays distinct as the
only colored lines); single-point computations -> line stubs; alpha ~0.6-0.75 on all
series; data md5-verified unchanged. Whitepaper inherits shared PDF.

**User (recal Part V rate):** at H=2.2 rate still ~3x low early / ~2x late vs Drela;
why can't a_visc be higher; does 0.0276 have physical grounding; what at 0.04/0.05;
explain c_o/eps_r=k. Study agent -> Part VI: a_visc sweep {0.0276,0.035,0.04,0.05,
late=Drela} with both panels + costs + impact rows; physical-grounding verdict
(eigenvalue vs empirical knob); rigorous DERIVATION of whether c_o = k*a_visc/a_inv
is a structural identity (would remove c_o, net -1 constant) or coincidence; verify
P_o = Omega_hat<-Z>/R flattens vs H near stagnation (bounds any c_o's onset tracking).

---

## 2026-07-28 — Fig 20 restyle INTEGRATED (e456138 + caption 540ab1c)

Refined styling committed and the one dependent caption sentence fixed (build 123pp
clean, 0 undefined/overfull):
- SA-AI family -> LINE-ONLY (up solid / dn dashed, seed colors, markers removed;
  the only colored lines = the subject); experiments marker-only (verified all
  line-free); computations line-only; Dong-Karniadakis Re=1e4 3-D DNS -> short
  horizontal STUB (half-width 0.035 decade) at the finest-grid Cd=1.143.
- Transparency: SA-AI alpha 0.75, all literature 0.62; limit-cycle bands opaque.
- Data md5-identical (styling only); regen against the committed 129-row snapshot
  (live jsonl now 173 rows w/ in-progress ultra — the HANDOVER rule-4 temp-root
  guard held).
- Caption (540ab1c): "green tick ... spans the resolution study" -> "short green
  stub ... marks the finest-grid value (N_z>=64 brackets Cd 1.110-1.143)".
- Figure /home/qiqi/flexcompute/sa-ai/paper/figs/dragcrisis_cd_re.pdf (whitepaper
  inherits); preview .../figs_explore/dragcrisis_cd_re.png; record
  /home/qiqi/flexcompute/sa-ai/agent-paper-review/2026-07-28-dragcrisis-overlay3-restyle.md

---

## 2026-07-28 — Part VI LANDED (efe0b8d): all four physics questions answered

Record: Parts I-VI in
/home/qiqi/flexcompute/sa-ai/agent-paper-review/2026-07-28-1218-fpg-recalibration.md

Q1 (why not raise a_visc): swept 0.0276-0.05. Stagnation late secant rises only
SUBLINEARLY 0.61->0.82x for x1.8 in a_visc (raising it pulls N-stations earlier,
compresses dRe_theta); the EARLY secant is gate/ramp-limited (0.34->0.40x) — the
"3x low early" cannot be closed by any rate constant. Cost = softmax_2 bleed:
viscous share at Blasius 14->35%, Blasius Re_x(N9) -10->-26%, mild-FPG beta=0.10
to 1.34x. VERDICT: no a_visc fixes stagnation under 5% Blasius drift; 0.0276 is
the largest safe value. Open candidate (not refit): sharpen the crossover
(softmax_p p>2 or an I_hat<=0 hinge) to kill the Blasius bleed while keeping the
pure-viscous stagnation gain.

Q2 (grounding): a_visc has NONE — a_inv=0.19 IS Michalke's measured KH eigenvalue
(0.1897, no free constant); a_visc=0.0276 is purely a Drela-stagnation-slope fit,
a knob. Present it like k, not like a_inv. (a_visc/a_inv=0.145 ~ viscous-TS/
inflectional 1/7, order-of-magnitude only.)

Q3 (c_o/eps_r=k): COINCIDENCE, rigorously. c_o is graze-anchored from
(Re_Omega*, P_o, Drela) alone — INDEPENDENT of a_visc (=0.1048 for every a_visc);
c_o/eps_r scales as 1/a_visc (0.72->0.40 across the sweep), hitting k at one point
only. Three constants, three unrelated anchors (beta=1 rate / beta=1 graze /
Blasius onset). c_o STAYS an independent constant (net count unchanged).

Q4 (P_o flatness, user's guess CONFIRMED): max P_o falls only -13% over beta
1->0.35 while Re_theta0 rises +141% and the required gate arg (~1/sqrt(Re_Omega*))
must fall -31% — P_o supplies only ~42% of Drela's demanded variation. Single
graze-anchored c_o leaves stagnation threshold ~0.63x low (matches Part V's 0.62x
marched onset). Required per-member weight rises 0.101->0.328 stagnation->Blasius;
no single c_o tracks the family — the low-H onset earliness is INTRINSIC to the
viscous coordinate's flatness, not a mis-set c_o. Full tracking needs a coordinate
steeper in H than P_o (the Re_theta-integral scale — outside the pointwise-local
class).

NET: the vbs form (a_inv 0.19 / a_visc 0.0276 / c_o 0.105) is the ceiling of what
the pointwise-local kernel achieves on Fig 4; the residual low-H rate (early
3x/late 2x) and onset (0.63x) misses are STRUCTURAL, not untuned. Adoption
decision unchanged; the residuals are now understood + defensible in the paper.

---

## 2026-07-28 — Part VII LANDED (9982a3b): c_nu,ai low-H susceptibility — intuition right
## at fixed Re_theta, REVERSED in realized rate (the "high Re_theta onset" clause is why)

Record: Parts I-VII in
/home/qiqi/flexcompute/sa-ai/agent-paper-review/2026-07-28-1218-fpg-recalibration.md

- FIXED-Re_theta (frozen eigenvalue, Re_theta=8000): user's mechanism CONFIRMED —
  production falls (1.1e-2->6.3e-4 over H 3.98->2.41), band thins (w 3.86->1.08),
  confinement drain grows 0.2%->23% of production, |d(ln s)/d(ln c_nu,ai)| rises
  ~10x (0.013->0.14). Peaks at the transitional band H~2.41, NOT stagnation (at
  H<=2.30 vbs production ~0 even above onset -> deficit is production, not drain).
- REALIZED (marched) rate REVERSES it: c_nu,ai 1/6->1/16 (x2.7) lifts stagnation
  late secant only +3% (0.61->0.63, never Drela) but Blasius +7%, moderate FPG +8%
  — OPPOSITE ranking. Why = the user's own "even at high Re_theta onset": low-H
  transitions at high Re_theta (marched Rt1~4200 stagnation vs ~330 Blasius) where
  the 1/Re_theta drain is already negligible; the drain the frozen map sees at a
  common Re_theta is not the drain the low-H layer actually experiences.
- vs a_visc: a_visc lifts stagnation +34% (0.61->0.82); c_nu,ai is BOTH weaker AND
  less selective (moves Blasius more than stagnation, breaks N=1 anchor -11%). No
  c_nu,ai brings H=2.216 to Drela (1/16 gives 0.63x; would need sub-molecular c).
- 1/6 was Blasius-anchored (N=9 within ~7% of Drela + resolvable-band margin ~sqrt(c)).
  Because lowering it barely helps realized low-H, 1/6 is NOT a low-H compromise —
  DIFFERENT structural story from a_visc: the low-H deficit is a PRODUCTION problem
  (a_visc / softmax_2 rate magnitude at Ihat<=0), not a diffusion-confinement one.
  Confinement is a large FRACTION at low H only because production is small.

NET understanding: c_nu,ai is not the low-H lever; the low-H rate ceiling is
production-limited and structural, confirming the vbs form is the pointwise-local
kernel's best. Data: paper/repro/analytic/fpg_recalibration_study.py --partvii,
fpg_recalibration_vbs.json, partvii_march.json.

---

## 2026-07-28 — ULTRA harvest LANDED (record 1642/commit 36676ff) + Sec VII integrated (7a96f62)

Record: /home/qiqi/flexcompute/sa-ai/agent-paper-review/2026-07-28-1642-dragcrisis-ultra.md
Figure (now Re 1->1e10): /home/qiqi/flexcompute/sa-ai/paper/figs/dragcrisis_cd_re.pdf

- SPAN: Re 1 -> 1e10, TEN DECADES, one model/constants. Front marches monotonically
  to the nose (89 deg -> 2 deg, Tu0.2 up) — transition-saturation mechanism confirmed.
- SEED COLLAPSE: PARTIAL — spread narrows 10% (2e7) -> 5% (3e9) as fronts crowd the
  nose, re-scatters to ~8% at 1e10; not a clean single curve, stated as such.
- BRANCH MULTIPLICITY (major): at 1e10 cold relaminarizes (Cd 0.108, sep ~180 deg,
  base Cp ~0) vs warm separated (0.189, sep 121 deg) — ~0.08 gap > crisis-band
  spread; ultra asymptote is a protocol band 0.11-0.21, not a level (URANS to select).
- FT-SA CONTROL (AI off, chi_inf=3, empty-echo verified): separates EARLIER
  (105-112 vs SA-AI 119-120) and drags HIGHER (0.25-0.37 vs 0.19-0.23) -> SA-AI is
  NOT just the turbulent baseline; BUT both ~2x under exp transcritical 0.55-0.7 and
  separate 105-123 vs ~147. VERDICT (refines the earlier interim "both" read): the
  missing rise is baseline-SA turbulent-separation + steady-symmetric-wake, NOT the
  transition model — marching SA-AI's front to 2 deg at 1e10 leaves Cd unchanged;
  transition closure EXONERATED for the transcritical gap.
- Logistics: 017-v100-dev GPUs 0-3, canon binary verified (commit 05d05a123b);
  results synced to unified local tree (173 cases); new ultra mesh y1=1e-9 fixes the
  2e7 y+ debt (y+ 0.44-0.47 at 1e10); 2e7 highre<->ultra seam 3.8-5.8%/seed (ultra
  authoritative). Cost +6.8 GPU-h.

INTEGRATED (7a96f62, 123pp clean): Sec VII span 1-2e7 -> 1-1e10; three-grid ->
four-grid; concession 5 rewritten to settle the attribution both ways (front-to-2deg
+ FT-SA control); concession 6 absorbs the 1e10 branch multiplicity (0.11-0.21 band).
Reseed harvest still pending (a2p5 settle leg) -> Sec IX/X rewrite next.

---

## 2026-07-28 — RESEED harvest LANDED (record 1715/commit f94de9f): alpha=0 seed-fix
## CONFIRMED; alpha=2.5 partial (leeward flank a separate effect)

Record: /home/qiqi/flexcompute/sa-ai/agent-paper-review/2026-07-28-1715-spheroid-reseed.md
Front-vs-seed figure: paper/figs/ (paper-quality, committed)

Seed determination: NO explicit DFVLR tunnel Tu% published anywhere (Stock PDF,
references/, digitization metadata all checked; Kreplin/Vollmers/Meier 1985 not in
repo). Documented anchor = Stock's limiting N_TS=8.0. Center seed = N_crit 8 via
Mack map -> chi_inf 2.382e-3, Tu 0.106%; re72a0 bracketed Tu x/div 2.

alpha=0 Re 7.2e6 fronts (chi=1 / Cf-rise vs 1122 remap):
  Tu 0.053% (N7.70): 0.628 / 0.664  (remap 0.592)
  Tu 0.106% (N6.04): 0.511 / 0.544  (remap 0.468)
  Tu 0.213% (N4.38): 0.400 / 0.427  (remap 0.355)
  measured 0.438, Stock e^N 0.425.

VERDICTS:
- alpha=0 CONFIRMED — freestream/IC chi IS the main fix. Bracket tracks the 1122
  seed-lever at full e^N strength (dx/dN~0.067), crosses measured 0.438 at budget
  N~5.0 (Tu~0.17%); documented DFVLR seed (N8/Tu0.106%) lands at 0.511 (0.07 L aft).
  No rate/transport change invoked. (Note: measured front = the model's N~5, higher-
  Tu than Stock's own N_TS=8 assignment — consistent with the 1122 finding that
  Stock's TS method books ~2.8 more e-folds than the Drela-fit family the model
  calibrates to; the disturbance-content story hangs together.)
- re65a0 (Re 6.5e6, N8): 0.556 = +0.045 aft of re72a0 same seed (correct Re
  direction); no measured DFVLR a0 front at 6.5e6, so a Reynolds-trend point only.
- alpha=2.5 PARTIAL: windward matches (model 0.444 vs meas 0.450) but a CONVERGED
  0.13 L azimuthal spread (leeward 0.576) that the near-uniform measurement (0.019 L)
  doesn't show. Freestream fixes the windward/TS meridian; the leeward flank
  lateness is a SECOND, seed-independent model-content effect (front-stationarity
  cross-leg confirmed — not a convergence artifact). Stock Fig 14b digitized fresh
  as the a2p5 reference.

READY (awaiting user go on the prose): Sec IX/X spheroid rewrite. Spine = 1122
threeway fig (seed-class miss, model e^N-faithful to x~0.7) + this reseed front-vs-
seed bracket (a0 seed-fix confirmed) + the a2p5 windward/leeward split (freestream
fixes windward, leeward flank = residual model content). Honest frame: the a0 "miss"
was freestream mis-specification (now demonstrated); residual is disturbance-content
+ the leeward-flank effect at incidence.

---

## 2026-07-28 — Spheroid alpha=0 INTEGRATED (d57b28a) + tunnel/N-factor answers + drag-crisis Qs

**Spheroid alpha=0 integration (d57b28a, 126pp clean):** new Sec X subsection
"axisymmetric pure-TS anchor" — converged flight-quiet front 0.858 vs measured
0.438; three-way N(x) fig (sphthreeway) shows model e^N-faithful to x~0.7; reseed
bracket fig (sphreseed) lands the front on 0.438 at tunnel-class Tu~0.17% with no
constant touched; the method-relative N-factor story; alpha=2.5 windward-match /
leeward-flank crossflow cost. Krimmelbein 2010 bib entry Crossref-verified.

**Q: freestream Tu <-> tunnel spec / N_crit story (answered + in the paper):**
- The model consumes a freestream seed chi_inf; the tunnel-equivalent Tu comes via
  the paper's Mack map (N_crit = -8.43 - 2.4 ln Tu, chi_inf = c_v1 e^-Ncrit).
- NO measured DFVLR tunnel Tu is published anywhere (checked). The documented
  disturbance level IS the limiting N-factor that matches the data — method-relative:
  Stock N_TS=8.0 (authoritative spheroid paper), Krimmelbein ~7.5, our Drela-
  calibrated kernel N_crit~7 (Tu~0.17%). No universal tunnel-N; each closure carries
  its own factor (Stock's pure-TS e^N books ~2.8 more factors than the Drela envelope
  by mid-body). => The authoritative spheroid paper used **N_TS = 8.0**.

**Drag-crisis Qs (coordinator c4a1a20, answers in 1642 addendum):**
- Fig 20 FT-SA now LINE-ONLY (dark-rust dash-dot, no markers) — the stray-symbol
  computation the user caught, fixed.
- Ladder spread: crisis band (3e5-1e6) = NON-CONVERGENCE (up branch limit-cycles
  hard, p2p 0.07-0.20; spread would shrink with steps/URANS) -> sharpens concession
  1; supercritical+ultra (7e6-3e9) = GENUINE BISTABILITY (both branches CD-flat,
  0.025-0.055 apart) -> sharpens concession 6. Crisis=unsteadiness, super=bistability.
- FT-SA lower at 1e10 = BRANCH SELECTION artifact: FT points are cold (attached
  branch, Cd 0.125, sep 180), SA-AI plotted line is warm (separated). Branch-for-
  branch FT-SA >= SA-AI always (attached 0.125 vs 0.108; separated 0.25-0.37 vs
  0.19-0.23). Noted in Sec VII (333d110).

**Q: any experiment above 1e7?** Per the 0117 litrange (adversarial search): the
smooth-cylinder composite tops out ~Roshko 8.4e6 / Shih 8e6 / Schewe 7.1e6 — NO
smooth-cylinder experiment above ~1e7. Our ultra arm 1e7->1e10 is entirely beyond
any measurement (a claim strength, and why the figure shows only computations there).

---

## 2026-07-28 — Spheroid Tu/N-factor literature synthesis LAUNCHED (user challenge)

User: the spheroid is a standard transition-modeling benchmark, so there should be a
plethora of model-calibrated Tu%/N values — grab and synthesize. Agent launched
(a71f9c0dd98da38b1): survey repo references (Stock, Krimmelbein, Coder, Medida, the
TransitionMPW workshop, Piotrowski-Zingg, etc.) + web for gamma-Re_theta/AFT/e^N
spheroid studies; extract each study's disturbance spec (method, calibrated N_crit
or inlet Tu%, the Re/alpha it was tuned on); CRUCIAL — distinguish a MEASURED tunnel
Tu (3x3m Goettingen) from a CALIBRATED fit value; deliver a synthesis table
(study/method/case/N or Tu/Mack-equiv Tu/measured-vs-calibrated) + where our
N_crit~7 (Tu~0.17%) sits in the spread. If the spread confirms no measured value and
a method-dependent range, it strengthens the paper's "N is method-relative" framing;
if a de-facto standard or a measured Tu exists, we adjust. Record + bib entries to
follow; I integrate into the Sec X anchor subsection.

---

## 2026-07-28 — fSlow convergence trials + two-branch native-check LAUNCHED

User: (1) try different fSlow protocols to converge crisis-band cases better; (2) why
two branches? native to SA or introduced by SA-AI? documented in literature?

Launched (coordinator a465100777339ba41): (1) fSlow trials on 2-3 crisis-band
limit-cyclers (finer stage ladder / lower final fSlow / longer holds / adaptive-CFL)
— does the CD cycle damp to a converged point, and where -> protocol-artifact vs
genuine unsteadiness (URANS); (2) definitive native-vs-SA-AI: does base SA (AI off)
show the SAME attached/separated two-branch pair -> if yes, native to base SA +
steady symmetric protocol, not SA-AI. My reasoned answer relayed to user (two
attached-flow topologies both fixed points of the steady symmetric equations;
cold->attached, warm->separated; the numerical face of the physical Schewe
bistability; FT-SA showing both = native to SA). Literature check to confirm.

---

## 2026-07-28 — Spheroid Tu/N band integrated (e46989f); roughness capability found; rough-cyl + 1e10-bistability answers

- Sec X calibrated-Tu/N paragraph CORRECTED + strengthened (e46989f, 127pp clean):
  removed the wrong Krimmelbein-N7.5-same-case (that was its tailplane); now the
  documented band Stock N_TS 8.0(DFVLR)/7.0(CEAT) + gamma-Re_theta workshop
  Tu 0.10-0.15% -> N~7-8 = Tu 0.10-0.16% (factor ~1.6), our N_crit~7/Tu~0.17%
  interior; measured primary (Meier-Michel-Kreplin 1987) exists but unused by models.
  Record 2330 / 178da0c. Meier bib entry added.
- FLOW360 HAS A ROUGHNESS MODEL (answer to user): Aupoix-Spalart (2003) SA-rough,
  input roughnessHeight = equivalent sand-grain k_s per wall patch, default 0
  (smooth). Implementation confirmed in compute/src/.../SpalartAllmaras/:
  SABoundaryConditions.h sets nonzero wall chi phiWall = C_vonKarman*u_tau*d0,
  d0 = C_roughness_d0*k_s (C_roughness_d0=0.03); SpalartAllmaras.h adds
  lengthScale += k_s*0.03 (d_eff = d + 0.03 k_s); SATurbulentViscosity.h uses it too.
  Constants C_roughness_d0=0.03, C_roughness_cr1=2.5 (standard SA-rough). => we CAN
  run rough cylinders and compare to Achenbach 1971 / Shih 1993 rough data.
- Rough-cylinder experiments: exist (Achenbach 1971 roughness family; Shih 1993
  smooth+rough) but at comparable/LOWER Re -- roughness advances the crisis to lower
  Re, does NOT raise the ~1e7 ceiling; no cylinder experiment (smooth or rough) above
  ~1e7 to my knowledge (offered to have lit verify the rough ceiling specifically).
- 1e10 bistability (user clarified: the deep-transcritical 1e10 two-branch, NOT the
  crisis band): genuine two-state (both CD-flat per the decomposition) -- attached-to-
  ~180deg low-drag (0.11) vs separated-at-~120deg (0.19), both steady fixed points;
  coordinator experiment (2) running now determines native-vs-SA-AI via the FT-SA
  cold(attached)/warm(separated) matched-Re pair.

---

## 2026-07-28 — Fig 18 stub/errorbar fix (4a698f5) + whitepaper rebuild

User: single-Re sim stubs hard to see (thicker/longer); error bars have horizontal
caps confusing with the stubs (remove caps). Done (I edited regen_dragcrisis_cd_re.py
directly; coordinator busy w/ experiments): Dong-Karniadakis single-Re DNS stub
hw 0.035->0.07, lw 1.4->2.8 (bold, longer); SA-AI limit-cycle error bars capsize
3.5->0 (capless vertical lines). Regen --include-ultra keeps full Re 1-1e10 (173
cases); data JSON untouched. Whitepaper inherits the shared PDF.
Figure /home/qiqi/flexcompute/sa-ai/paper/figs/dragcrisis_cd_re.pdf.

---

## 2026-07-28 — Fig 18 decluttered to one seed (085e922)

User: the Tu seeds barely differ except where the sim can't converge -> keep only
the cleanest-converging one. Picked Tu=0.2% by data (mean tail p2p 0.0093 vs
0.05%:0.0107, 0.7%:0.0175 -- cleanest, and the canonical middle seed w/ cold refs).
regen_dragcrisis_cd_re.py: --seeds flag (default "0.2"; "0.05,0.2,0.7" restores
family); other-seed labels guarded. Fig 18 now single orange up/dn ladder + literature
+ FT-SA to 1e10 -- far cleaner. Tu-shift-of-crisis (a real computed result) redirected
to the angle figure (fig:dragcrisisangles keeps all 3 seeds) + text; caption + traverse
sentence updated; span label 1-2e7 -> 1-1e10, "three grids" -> "four grids", filled/open
-> solid/dashed. 127pp clean, whitepaper inherits.
Figure /home/qiqi/flexcompute/sa-ai/paper/figs/dragcrisis_cd_re.pdf.

---

## 2026-07-28 — Fig 15 (Eppler Re-sweep) rework LAUNCHED (user, 5+1 requirements)

Agent aef9db94d4cd38706 reworking fig:eppresweepforces / regen_epp_resweep_forces.py:
1. Experiment (LTPT) symbols enlarged + dominant vs small-gray published-calc symbols
   (distinction was unclear).
2. Recast to 2x2: combine the separation + reattachment panels into one
   (x_sep vs x_R told apart by trend) -> {lift, drag, moment, sep+reattach}.
3. The mysterious vertical lines = alpha=4-6deg experimental/reference brackets
   straddling alpha=5 (Cole-Mueller @1e5, Ghimire @3e5; no exact alpha=5 station
   datum) -> render as capped range bars + legend/caption phrase.
4. Legend repositioned off the bottom-panel x-axis labels.
5. Investigate why LTPT oil-flow (open black circles) don't show on sep/reattach
   (OIL dict TM-4062 Table III coverage; enlarge/fill) -> data-availability finding.
+6 (mid-turn): replace LTPT force error bars with the INDIVIDUAL repeat measurements
   as separate black circles IF TM-4062 provides them; else report what needs
   digitizing, no fabrication.
Deliver reworked script + fig + draft caption (I integrate tex, incl. panel-count /
width). Record + commit to follow.

---

## 2026-07-28 — Fig 18: limit-cycle uncertainty bars removed (ca13642)

User: remove the vertical uncertainty bars (confusing, plot crowded). Done — the
SA-AI limit-cycle tail-window bars are no longer drawn (tail min/max stay in the
JSON dump + appendix table; non-uniqueness remains a text concession, Sec VII 1&6).
Caption sentence removed; script docstring/print reworded; 127pp clean; whitepaper
inherits. Fig 18 now: single Tu=0.2% up/dn ladder + literature + FT-SA, no bars.

---

## 2026-07-29 — Fig 15 (Eppler Re-sweep) reworked + integrated (b6acddd fig, 9b49282 tex)

Record: /home/qiqi/flexcompute/sa-ai/agent-paper-review/2026-07-28-fig15-rework.md
All 6 requirements done:
1. LTPT experiment now LARGE FILLED black circles (ms 8.5, top zorder) vs small open
   gray published-calc symbols -> instant experiment/CFD distinction.
2. 2x2: lift|drag(log) top, moment|combined-sep+reattach bottom (shared x/c, in-panel
   "laminar separation"/"turbulent reattachment" labels).
3. Mystery vertical lines = alpha=4-6 brackets straddling alpha=5 (no alpha=5 station
   datum) -> now capped range bars + legend entry.
4. Legend moved figure-level below axes (bbox tight) -> no x-label overlap.
5. "Missing" oil-flow circles: NOT missing data -- TM-4062 Table III tabulates oil
   sep/reattach at alpha=5 for Re=1/2/3e5 ONLY (genuine source limit); old circles
   were small/open/overlapped -> now enlarged/filled/visible.
6. Individual LTPT runs (user): replaced the assumed error bars with each individual
   Table B1 near-alpha=5 run as a separate black circle where the repo has them
   (2e5 c_l 4 runs, 1e5/2e5 c_d repeats; single elsewhere). NO fabrication -- a full
   per-run set needs re-reading TM-4062 Table B1 from mcghee_1988 PDF (flagged).
Data JSON byte-identical (styling/layout only). Integrated: caption rewritten,
includegraphics \textwidth -> 0.9\textwidth (aspect 1.39->1.20). 127pp clean;
whitepaper inherits. Figure /home/qiqi/flexcompute/sa-ai/paper/figs/eppler_resweep_forces.pdf.

OPEN (user-gated, flagged): a complete individual-run digitization of TM-4062 Table B1
(all runs x near-5deg alphas x all columns) would let every repeat show as its own
circle at every Re; currently only the partial set already in the repo is plotted.

---

## 2026-07-29 — Push complete; TM-4062 individual runs integrated; zb-rate + low-H-sphere launched

- HISTORY CLEAN + PUSHED: filter-repo purged *.cgns (219MB mesh_L1 gone); largest blob
  now 36MB; remote git@github.com:qiqi/aft-sa.git updated (955cd2c..9112b9d). *.cgns
  gitignored (955cd2c). Memory saai-no-binaries-in-git.md written.
- Fig 15 (b365942 digitization + 9112b9d caption/push): TM-4062 fetched from NTRS
  (references/mcghee1988-e387-TM4062-NTRS19890001471.pdf); every Table B1 run within
  0.1deg of alpha=5 plotted as its own black circle (1/2/4/1/1 by Re; 200k is the real
  4-run cluster) from paper/data/mcghee1988_tableB1_runs.json; 4 in-plot texts removed
  (now caption-only). 127pp clean.
- zb Fig-4 question (user, CORRECTING my error): blue/green share a_visc but green rate
  IS lower at low H. Launched investigation (ac0eaf0980dbff306): trace dN/dRe_theta +
  gate S for H=2.2/2.3 at C=1851 vs 8000 (hypothesis: higher C = higher threshold =
  gate less open at matched Re_theta = rate suppressed near onset; the [5,9] window
  overlaps the not-fully-open gate at low H); PLUS indicator-sphere viz of H=2.2-2.59
  profiles with the H=2.2-vs-2.3 sphere-separation vs their ~1.8x onset-Re_theta ratio
  (does the low-H bunching on the sphere explain the onset-resolution limit?).
- STILL RUNNING: coordinator EXP2 (FT-SA native-branch determination).

---

## 2026-07-29 — User's two-branch-gate proposal (consistency fix) launched as Part VIII

User reasoned: the onset gate is single-branch (I_hat only via P=Omega_hat*I_hat),
but at low H I_hat~0 for all profiles (they bunch on the sphere) so the gate can't
resolve them and just saturates to the constant C -- an inconsistency with the
two-branch RATE (inflectional + viscous). Proposed fix: Re_Omega^c = k*softmin(
A+B/P_I^2, A_c+B_c/P_curv^2) with P_I=Omega_hat*<I_hat>+ and P_curv=Omega_hat*<-Z>+/R
-- i.e. replace the constant ceiling with a CURVATURE-based onset branch, making the
gate two-branch like the rate. At high H the inflectional branch wins; at low H
(I_hat->0) the curvature branch (finite, H-resolving via -Z) sets the onset.
This is a symmetry/consistency argument (gate should mirror the rate's two branches)
and is a cleaner variant of Part-V vbs (which blended curvature INTO P rather than
softmin-ing two separate thresholds). Extended the running agent (ac0eaf) as Task 3:
design+calibrate the curvature onset branch to track Drela Re_theta0(H) to stagnation;
compare both panels vs canon-C and vbs; full regression (esp. curvature branch must
not fire on the separated/Stewartson side, and Spalart zero-suite); constant
bookkeeping; recommend softmin-of-two-thresholds vs vbs P-blend. Candidate Fig-4 +
Part VIII record to follow.

---

## 2026-07-29 — Part VIII landed (vg two-branch gate = vbs); Fig 2 P_curv-graze enrichment launched (Part IX)

Record: agent-paper-review/2026-07-29-0121-zb-rate-lowH-sphere.md (commit e68d31d).
- WHY green<blue rate at low H: gate-saturation confirmed. Higher C scales gate-open
  Re_theta by the C ratio (S=0.5 point 1367->5908 =x4.32); the [5,9] secant window
  lands on the still-opening gate (S 0.62-0.75 green vs 0.92-0.96 blue), throttling
  realized rate to 0.56x DG. Saturated limit is shared -> higher C can't boost rate,
  only pushes the window off saturation.
- SPHERE: H=2.216 & 2.30 both at max P=Omega_hat*I_hat ~ 1-3e-4 (on the neutral
  locus), differ 1.6e-4, while Drela onset Re_theta0 differs 1.81x (6656 vs 3680).
  YES near-degenerate -> geometric root of the onset-resolution limit. Figure
  paper/repro/analytic/figs_explore/indicator_sphere_lowH.png.
- TWO-BRANCH GATE (user's proposal, form "vg"): Re_Omega^c = softmin(A+B/P_I^2,
  A+B_c/P_curv^2), B_c=130. Cuts panel-b worst dev 4.40x->1.62x, distinct
  Drela-tracking H=2.2/2.3 onsets, clean regression (boundedness, zero-suite,
  no separated-branch firing), NET-ZERO constants. MATHEMATICALLY EQUIVALENT to
  Part-V vbs (B_c 129.74 ~ vbs B/c_o^2 130.15 to 0.3%; identical devs). vg is the
  more interpretable presentation (two-branch gate mirrors two-branch rate) -> ADOPT
  the vg framing. Candidate model_calibrate_candidate_2branchgate.png.
- LAUNCHED Part IX (user): enrich fig:onsetgraze (Fig 2) with a COMPANION panel in
  the curvature coordinate P_curv = Omega_hat*<-Z>+/R (denote <Omega_hat Z_hat>),
  plotting the low-H profiles that vanish from the P_I panel and showing the vg
  gate's SECOND branch (A+B_c/P_curv^2) GRAZES them -- the visual proof that each
  softmin branch grazes its own family (inflectional in P_I, favorable/low-H in
  P_curv). Candidate + graze table + draft caption to follow; I integrate tex.

---

## 2026-07-29 — Candidate Fig 3 (disturbance transport under vg form) launched

Clarification: paper Fig 3 = fig:nuhat = fs_nuHat_rows (disturbance transport, 3
wedges, N-contours + marched envelope vs Drela). Earlier I showed the CALIBRATION
figure (Fig 4) not Fig 3, and no candidate Fig 3 existed. Launched (ac0eaf): rebuild
fig03_fs_transport_rows under the vg two-branch candidate (a_visc floor + two-branch
gate) -- keep beta=-0.10/0/+0.10 rows (adverse+Blasius unchanged, +0.10 favorable
lifted toward Drela) and ADD a strong-FPG beta=+1 (H=2.216) row (the payoff: canon
can't ignite it, candidate envelope should rise & track Drela). Candidate PNG
figs_explore/fs_nuHat_rows_candidate.png (not overwriting canon), draft caption,
Part X record. I integrate tex after user sees it.

---

## 2026-07-29 — OVERNIGHT vg-kernel validation campaign launched (user asleep, autonomous)

Budget ~6h wall on 017-v100-dev GPUs 0-3 (free; 4/5/7 = vishal). Agent
af1d5371e6fcbc1c8: (A) implement the vg two-branch kernel in the compute solver
ENV-GATED (default OFF = bit-identical canon; a_visc=0.0276 rate floor + curvature
gate branch B_c=130 via new AI_A_VISC/AI_REOMC_BC alongside existing AI_REOMC_A/B/
CEIL), CLEAN rebuild (BUILD_CONSISTENCY.md), verify canon-identical-when-off +
solver-vs-analytic consistency + echo (STOP if any gate fails -- don't burn GPU on
a wrong kernel); (B) campaign, vg ON, cases tagged _vg (canon JSONs untouched):
PRIMARY = high-Re drag-crisis cylinder Tu=0.2 at 2e6/4e6/7e6/1e7/2e7 (+1e8/1e10 if
budget) -> theta_tr + Cd + sep vs canon vs experiment (does the nose front advance
and Cd rise toward transcritical 0.5-0.7 = the Sec VII missing rise?); REGRESSION =
flat plate ZPG + NLF design + Eppler 1-2 Re (confirm low-H change doesn't break the
anchor/rooftops). Record + commit; NO tex edits (env-gated, canon default untouched).
USER ASLEEP -- proceeding without feedback; morning summary owed. Shepherding via
hourly wakeup; agent notifies on completion.

## 2026-07-29 ~03:xx — vg campaign INTERIM (record 0320; still running)

Phase A ALL PASS + committed (compute repo 29726cb5f9, explore-lambda-v): vg kernel
env-gated (AI_A_VISC, AI_REOMC_BC; default OFF), clean rebuild; canon bit-identical
(0.0 rel err over 1921 rows; CFD Cd=0.2048=canon), solver-vs-analytic vg 8.9e-16,
echo correct. THE KERNEL IS CORRECTLY IN THE SOLVER.
Phase B PRIMARY 2e6-2e7 (Tu0.2, highre): vg front within +-2.5deg of canon, Cd
within +-0.013 (~+3-6% over canon-up) -> NO transcritical advance at practical Re
(front stays 82-99deg, not 25-35; Cd ~0.205 not 0.5-0.7) -- exactly the analytic
prediction (clean noses <=2e7), regression-clean, not the headline.
REGRESSION: flat plate ZPG -7.5%/-9.0% Re_x_tr (matches documented Blasius -8.2%,
the a_visc floor; not a break); Eppler 200k a=2 rooftop unchanged (<0.02c). Clean.
ULTRA HEADLINE (1e9, 1e10 -- where nose Re_theta clears Drela onset, the real test)
STILL RUNNING: 2 Flow360Solver active on 017, 1e9 vg case in progress; record
[TO FILL] for ultra + budget + verdict. Shepherd wakeup 03:45 will harvest on
completion. No intervention needed.

## 2026-07-29 — OVERNIGHT vg-kernel campaign COMPLETE (record 0320; sa-ai 056fc70, compute 29726cb5f9)

PHASE A (kernel in solver): ALL GATES PASS. vg two-branch rate+gate implemented
env-gated in SAAiTransition.h (__aiRateFromXYZ), AI_A_VISC + AI_REOMC_BC default 0 =
canon. Clean rebuild. Canon bit-identical (0.0 over 1921 rows; CFD Cd=0.204764=canon),
vg-vs-analytic 8.9e-16 (252 rows), echo correct. Committed compute 29726cb5f9. SOUND
+ SAFE (default-off, canon untouched).
PHASE B HEADLINE: vg does NOT recover the transcritical Cd rise.
- practical Re 2e6-2e7: vg ~= canon (front +-2.5deg, Cd +-6%), regression-clean.
- ultra Re 2e7-1e10 (protocol-matched warm): front DOES advance into the transcritical
  class (theta_tr ~33deg@3e8, ~2.5deg@1e10) BUT that advance is ALREADY in canon; vg
  adds nothing. Cd stays ~0.19-0.21 for BOTH canon+vg -- NO rise toward exp 0.5-0.7.
  (cold-start bistability hint: transient Cd~0.7@1e9 settling to limit cycle 0.225 --
  not the steady soln.) => consistent w/ the FT-SA verdict: the missing transcritical
  rise is baseline-SA turbulent separation + steady wake, NOT transition.
REGRESSION: flat plate ZPG -7.5%/-9.0% Re_x_tr (= documented Blasius -8.2%, the a_visc
floor; anchor holds); Eppler 200k a=2 rooftop unchanged. Airfoils/anchor safe.
BUDGET: ~2.5 GPU-h (well under 6h), GPUs 1/2/3 on 017; foreign/vishal GPUs untouched.
NET: vg port sound + safe to keep env-gated; low-H FPG amplification behaves as
designed on calibrated cases (Fig 3 stagnation ignition, favorable rooftops) but does
NOT fix the drag-crisis transcritical Cd (which was never a transition problem).
Outstanding: FT-branch EXP2 (coordinator native-vs-SA-AI) record not yet landed.

## 2026-07-29 — Whitepaper prose strip launched (Drela-meeting doc: numbers/eqns/figs only)

Agent ab13ddcf771eb2abd stripping paper/whitepaper.tex (ONLY; main paper untouched):
delete every sentence with no number/equation/constant-value (e.g. "a_max is an
eigenvalue not a fit"), AND delete sentences that merely repeat numbers already shown
in an adjacent figure/table/equation (single-source the number, author narrates).
Keep: equations, constants=value, non-redundant quantitative results, figure/table
envs + their quantitative captions, section headers, \ref integrity, build-clean.
No number/equation altered. The new-kernel section (already quantitative-only) left
as-is. Record + commit to follow; whitepaper.pdf rebuilt.

## 2026-07-29 — Whitepaper enrichment folded into the strip pass (add missing numbers from main paper)

Same agent ab13ddcf (one owner of whitepaper.tex): after stripping, MINE sa-ai.tex
for identifying/headline quantitative numbers absent from the whitepaper and ADD them
as per-figure/per-line caption annotations (numbers only, verbatim). Example: Fig 2
onset-graze gets beta + max(Omega_hat*I_hat) + H per line (from fig:model betas,
fig:nuhat max 0.162/0.078/0.037, low-H members beta 0.35-1.0 + graze ratios +
Re_theta0). Applied to EVERY whitepaper figure (which-curve-is-which labels: beta/H/
Re/Tu/N-crossing/front x-c/Cd/sep angle) + headline result numbers from section text.
Single-source preserved; no new prose. Same commit + record.

## 2026-07-29 14:xx — Picked up dead session (weekly-limit) leftover: whitepaper Fig 30 landed

Prior Claude session (52a71f9) hit the weekly usage limit mid-batch; picking up the
whitepaper (Drela-meeting doc). It had committed 78a4b73 (new-kernel Figs 25/26/28/29
+ Fig 21) and left three items. Done now:
- **Fig 30 (vg vs canon airfoils)**: the figure the dead session's subagent produced
  (newkernel_vg_airfoils.png) but never integrated. Stripped its suptitle to match
  house style (siblings use per-panel titles only, caption carries the story), made
  the generator plot from the cached vg_airfoils.json (reproducible w/o the CFD tree;
  --rebuild to re-read .pvtu), added the figure + caption after newkernel_vg_results.
  Upper-surface Cf(x/c), canon vs vg: NLF a0 x_tr=0.395 (both), a4 0.265->0.262,
  Eppler a2 0.580 (both) -- low-H branch does not trip the favorable rooftop.
  Whitepaper recompiles clean (63 pp, 0 undefined).
- Bisection Table t:stagbistab: previous session already confirmed unchanged (critical
  band + sustained maxchi 2.19/6.15/24.1/75.7 matched) -- no-op.
- NEXT (user-approved): rebuild the shared Cd figure (fig:dragcrisiscd,
  dragcrisis_cd_re.pdf; whitepaper + main paper) SA-AI curves as clean up/down ladders
  from the 50-row Tu=0.2% systematic campaign, keeping the exp/LES overlay.

## 2026-07-29 19:xx — Drag-crisis full migration to systematic up-ladder (both docs)

User asked to replace the paper figures with the whitepaper's updated drag-crisis
figures and update the descriptions in BOTH docs, then commit. Chosen scope: full
migration (drop the old 84-case 3-seed matrix narrative) + replace the table with a
systematic up-ladder table.
- Data: single seed Tu=0.2% (chi_inf=1.1e-2), one up-ladder, 25 matched-Re points
  Re_D=1..1e10, 4 grid families (lowre/pilot/highre/ultra). 5 of 25 fail the steady
  monitor (5.62e5,1e6 lc; 3e6 cap; 3e7 lc; 1e10 cap; tail-p2p median 0.044, max 0.171
  at 3e7). |C_L|<2e-6 (converged <6e-8). Crisis 3e5(0.773)->1.78e6(0.228); floor
  ~0.19-0.23 flat to 1e10. chi=1 front 98 deg(1.78e6)->86.5(1e7)->2.5(1e10). Down
  ladder finds a lower branch at ultra-Re (33-52% below up from 1e8; 0.108 vs 0.185 at
  1e10) -> kept as concession six (protocol-dependent band 0.11-0.19); up-ladder plotted.
- Figure: aligned the Cd open-ring criterion to verdict!=converged so rings (5) match
  the table's 5 bracketed rows exactly (dropped the two laminar Re=1,3 fSlow artifacts).
- Table: new regen_dragcrisis_table.py writes tables/tab_dragcrisis.tex (main,
  tab:data_dragcrisis, [tp]) + tables/tab_dragcrisis_wp.tex (whitepaper, t:dragcrisis,
  [H]); whitepaper switched from an inline 84-case table to \input the wp file
  (single-source). Columns Re_D/grid/Cd([p2p])/theta_tr/theta_sep.
- Prose: rewrote sa-ai.tex Sec. drag-crisis (setup para, both fig captions, body,
  six concessions, fifth-concession follow-up, TODO comment, conclusion "family"->
  "curve") and whitepaper subsection to single-seed up-ladder. All numbers taken from
  systematic_Tu0.2_summary.jsonl (see repro/cfd/sys_report.py, run_dragcrisis_systematic.py).
- Both compile clean: sa-ai 126 pp / 0 undefined; whitepaper 64 pp / 0 undefined.
