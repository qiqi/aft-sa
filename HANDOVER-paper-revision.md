# Handover: SA-AI paper revision (2026-07-27; pruned 2026-08-01)

For the agent taking over the paper revision. The previous agent keeps
ONLY the long-running CFD (see "What stays with the CFD agent") and
will deposit harvested result JSONs; everything else here is yours,
including responding to `agent-paper-review/`.

**Pruned 2026-08-01.** Three original open-work sections are DONE and
were removed: the Daedalus cavity-L2 harvest (`tab:daetotals` filled,
no dashes; the dagger/held/bypass-inactive disclosures are gone — the
two surviving `\ddagger` are the legitimate Re=6e4 XFOIL-reference
footnote), the OpenFOAM Eppler L2 decision (`tables/tab_openfoam_eppler`
is `\input` in the paper at L0-L2 with the bypass-omission scoping
stated), and the Sec-VIII bistability critical-Re re-emit (the paper now
prints the bisected `Re_r = 4.66`--`4.74e5`, not the known-wrong
`2.78`--`2.89e5`). `git log -p -- HANDOVER-paper-revision.md` has the
original text.

**Newer standing context this file does not cover** — read alongside it:

- `paper/expert_feedback.md` — Drela's 2026-07-29 meeting: the standing
  expert guidance and its open items (the handover-χ level, the high-H
  Orr--Sommerfeld basis, the lofted-χ bubble failure). Newest entries at
  the bottom.
- `paper/notes_nu_t_mapping_bubble_closure.md` (2026-07-30) — the live
  modeling frontier: physically grounded ν̃→ν_t maps for LSB closure,
  with a recommended attack order.
- `RESPONSES.md` (tail) — chronology since 2026-07-27: the drag-crisis
  migration to the systematic Tu=0.2% up-ladder (now Sec VII), the
  env-gated `vg` two-branch kernel campaign, the whitepaper strip.
- `paper/weakness-review.md` was RETIRED 2026-08-01: every catalogued
  item was closed, and its constants header still described the
  pre-sphere v2 kernel (c_ν,ai=1/12, K_λ, K_r, λ_p, gate mode 4), so it
  had become a trap. Its per-item ledger (O1-O37) survives only in the
  sphere-kernel WIP squash — recover with
  `git show 8ebed56:paper/weakness-review.md`. Next weakness catalog, if
  one is wanted, starts fresh against the sphere kernel at O38.

## What this is

`paper/sa-ai.tex` — AIAA Journal submission for the SA-AI one-equation
transition model (repo branch `sphere-kernel`, solver repo
`~/flexcompute/compute` branch `explore-lambda-v`; canon = sphere kernel
+ fv1 bypass, latest solver commit 29726cb5f9, which adds the
default-OFF `vg` two-branch kernel). ~126 pp, builds clean (pdflatex x2
+ bibtex: 0 errors, 0 undefined). Read `ONBOARDING.md` first if you
haven't. Fifty review passes have run; every number surviving in the tex
has been independently verified at least once — treat that as an asset
you must not squander.

## Hard rules (each one earned the hard way)

1. **Never paraphrase a verified number — move the sentence that
   contains it.** Every honesty disclosure keeps exactly one
   full-strength home. (Pass-46 "spirit rules", enforced by pass 47.)
2. **Write the tex via a temp file + os.replace**, never in-place
   streaming; verify each edit landed (grep) before building.
3. **Stage explicitly. NEVER `git add -A`** — the repo contains case
   trees that hang the scan. Commit messages: why/gotchas only, never
   restate the diff.
4. **Figures regenerate only when their inputs are complete** — do not
   regenerate while a background job is still building caches (this
   produced committed race artifacts once; pass 44).
5. **`AI_LAMINAR_SLOWDOWN=0.01` is load-bearing** with all campaign
   JSONs: solver-input seeds are pre-compensated (chi_BC = chi_inf *
   f = 8.76e-6); the physical seed 8.76e-4 is what the paper quotes.
   Running a committed case JSON with f=1 gives a 100x-high seed.
   Always extract+diff `ai_constants.log` when touching runs.
6. **Front conventions matter to 0.05c**: the paper's airfoil fronts
   are the solver-reported near-wall chi=1 crossing (now stated in the
   fig:nlfaft caption); chi=c_v1 sits up to 0.05c aft on slow fronts;
   Cf-rise is a third convention (spheroid, k=1.5 band 1.25-2).
   OpenFOAM cross-solver fronts in `data/openfoam_airfoil_summary.json`
   are re-extracted at chi=1 by `openfoam/scripts/
   export_airfoil_summary.py` (per-x-bin wall-adjacent band; a plain
   kd-tree near-wall band reads the Eppler bubble's lifted shear layer
   ~0.07c early — do not regress to it).
7. **Never quote fronts from fixed-budget cold starts** (negative-alpha
   lesson; the campaign converges fronts with
   `repro/driver/converge_by_xtr.py` stability batches).
8. Solver rebuilds: follow `~/flexcompute/compute/BUILD_CONSISTENCY.md`
   (clean rebuild after header edits; verify object-timestamp
   uniformity). Solver runs in-session: no mpirun; set
   OMPI_COMM_WORLD_LOCAL_RANK/RANK=0, SIZE=1 (see
   `paper/repro/driver/solve.py`).
9. Big data lives on /local_data with symlinks in-repo; `.pkl`/`.npz`
   are gitignored (regen paths documented in each script docstring).
10. Append important user-facing summaries to `RESPONSES.md` (the user
    reads it; tiny tmux scrollback).

## Review-thread protocol (now yours)

Reviewers (two threads: the main reviewer's numbered passes, and the
OpenFOAM-port thread) drop `.md` files into `agent-paper-review/`;
respond with `<same-stem>-response.md` in the same directory. Verify
before adopting: both threads are excellent but each has had refuted
items (see 0102 item 3's history, pass-45 corrections). Restart the
folder watch (the previous agent's monitor is stopped):
`ls agent-paper-review/*.md` newest-first.

Status of the folder as of 2026-08-01: reviewer-pass responses run
through `2026-07-27-1440`. Everything deposited since (all the
2026-07-28 and 2026-07-29 files) is an **implementation / campaign /
digitization record**, not an ask awaiting a response — those were
folded into `RESPONSES.md`. Don't mistake the missing `-response.md`
files for a backlog.

## Current paper state

- Structure: I intro / II model / III flat plate / IV NLF / V Eppler /
  VI Eppler Re-sweep / VII cylinder drag crisis (sec:dragcrisis, added
  after this handover was written) / VIII attachment-anchored turbulent
  branch (sec:bistability) / IX Daedalus / X spheroid (sec:spheroid) /
  conclusion (with the drag-crisis outlook paragraph) / Appendices A-H
  (H = spheroid pressure stations + supplementary maps).
- All pass-46..50 verified fixes are in; conventions single-homed
  (five-row layout + line conventions + x u_e^2: Sec IV; e^N
  substitution rule + cross-check: Sec IV; digitized-reference
  provenance: Appendix F block).

## OPEN WORK — ordered by what unblocks what

### A. Spheroid section remainder (task #37; USER DIRECTIVES verbatim)

User: digitize the experimental paper (Stock/spheroid.pdf), overlay on
the computed waterfalls, move selected conditions into the main paper.
"Any figure without an experimental comparison on top of it belongs to
the appendix. We should have as many experimental comparisons as
possible. And as many comparisons against other literature
(computation) as possible."

DONE (do not redo): Stock Figs 2-5 digitized
(`data/stock2006_fig{2,3,4,5}_digitized.json`, by
`repro/cfd/digitize_stock_waterfalls.py`) plus Figs 14a/14b/15a
(`digitize_stock_fig14a.py` / `_fig14b.py`, consumed by
`compare_stock_fronts_a0.py`); the measured chains are overlaid on our
station waterfalls by `repro/cfd/regen_spheroid_station_profiles.py`;
the main-vs-appendix split is executed per the user's rule (measured-
overlay Cf+γ stations and the front comparison in Sec X; Cp waterfalls
and the un-overlaid maps in Appendix H); the Sec-X "in progress"
language is gone; Kreplin/Vollmers/Meier is cited (`kreplin_1985`).

DONE 2026-08-01 (this session): Sec X restructured to the user's arc —
ascending incidence, regime-grouped: X.A zero incidence (pure-TS anchor
+ disturbance environment, both Reynolds numbers) / X.B α=2.5° (the
azimuthal spread appears) / X.C the mixed TS–crossflow pair at 6.5e6 /
X.D the laminar-to-separation ladder at 1.5e6. The α=0 front comparison
`spheroid_front_compare_a0.pdf` (previously committed but unused) is now
Fig. 24 in the main body; the α=0 maps went to Appendix H (no measured
overlay, per the user's rule). `tab:sphtotals` dashes are FILLED — see
below. Verified: 0 errors / 0 undefined, and every number in the old
section survives in the new one (set-diff check).

Still live:

1. **The `tab:sphtotals` dashes were a bookkeeping gap, now closed.**
   The three conditions (`7.2e6 α=2.5°`, `6.5e6 α=5°`, `6.5e6 α=10°`)
   had been run to completion on 2026-07-27 20:40 and harvested at 20:41
   (`SPHEROID-RESUME-ALL-DONE`); the table was simply never re-emitted.
   All three are in `sphere_campaign_spheroid_results.json` with
   `complete=True`, same 20k budget and same median-of-last-fifth
   convention as the four conditions the table already printed (whose
   values match it to the last digit). That JSON was missing on 019 —
   copied from 014 (md5-verified) into `flow360_fv1/` and the table
   re-emitted. **If a condition ever looks "in progress", check whether
   the harvest JSON is simply absent locally before assuming no data.**
2. The remaining spheroid gap is FIGURES, not forces: 6.5e6 α=5°/10°
   have table rows but no surface/front figures, and 1.5e6 α=5° has
   none either. Regenerating them needs the case tree, which lives only
   on **014** (see the data-location note below).
3. Stock's Figs 14c (α=5°) and 16c (α=29.7°) measured fronts remain
   undigitized — named in X.D's closing sentence as the pending item.
   Sources are on 014 at `/local_data/qiqi/sa-ai/stock_digitize`
   (14a/14b/fig2/fig3 crops present; no 14c/16c crops yet).
4. Strongest modern computational overlay: the 1st AIAA Transition
   Modeling & Prediction Workshop spheroid case (Re=6.5e6; Coder summary
   = the same document our NLF workshop overlays came from). Check
   transitionmodeling.larc.nasa.gov/workshop_i for **data files** before
   digitizing slides.
5. Known digitization gap, left as-is: fig5 `gam` at x/a=0.766, the deep
   dive over 85-110 deg, outruns the tracker's slope cap.

### B. Pass-46 length plan remainder (task #40)

Executed: D2-D6, D9-D11, L4, O3, O5, O7(i) + pass-47 corrections.
Remaining: L1-L3 language sweep (LAST, one section per pass,
move-don't-paraphrase), D9 remainder (Sec V ~l.1806 and Sec VI ~l.2090
seed restatements -> pointers), O7(iv) declined (number-dense; reviewer
accepted). USER-GATED (decided): keep one PDF for now (no supplemental,
no O1 sheet curation); Eppler mesh close-ups already dropped; ch.48
footnote dropped; epigraphs stay.

### C. Small open threads

- Conclusion K-gate sentence + drag-crisis outlook: settled and
  verified (pass 50); the K-gate model revision itself
  (1250/1315/pass-49 memos) is FOLLOW-ON work, not this paper.
- Three pre-existing bibtex empty-pages warnings (vaningen_2008,
  medida_baeder_2011, cakmakcioglu_2020) — fill when convenient.
  (zheng_lei_2016's volume is already added.)
- Pass-50 minor 4: if you tighten the verdict criterion further,
  update "steady to the solver's tolerance".

## Pending user decisions (ask before acting)

- The spheroid main/appendix split has now been executed under the
  user's stated rule; per-figure sign-off was reserved to the user (the
  rule is decided, taste isn't). Walk them through the split as it
  stands rather than re-deriving it.
- Companion-paper flag list (near-pole surrogate work) still owed to
  the user — unrelated to this paper but recorded in memory.

## Where the data actually lives (checked 2026-08-01)

The earlier "spheroid campaign resumption when GPUs free" framing was
wrong: the runs are DONE, and GPU contention was never the blocker.
What matters is which machine you are on.

- **014-v100-dev** — the spheroid/Daedalus machine. Repo root
  `~/flexcompute/sa-ai` (note: no `src/` level), `/local_data/qiqi/sa-ai`
  holds spheroid_fv1 42 G (32 cases: the full condition matrix, the α=0
  reseed bracket, full-body and unstructured controls), spheroid_meshes
  4.6 G, daedalus 54 G + daedalus_fv1 13 G, dragcrisis_matrix 25 G,
  flow360_fv1 32 G, stock_digitize. 351 G free. **The paper is built
  here** — `CJKutf8.sty` (the Daodejing epigraphs) exists only in 014's
  `~/texmf`, so pdflatex cannot complete on 019.
- **017-v100-dev** — `spheroid_fv1_reseed` 9.6 G (the five reseed cases)
  + the re65 L1 mesh. 402 G free.
- **019-v100-dev** — repo + committed figures/JSONs only. All case-tree
  symlinks are DANGLING (`/local_data/qiqi/sa-ai` down to 228 MB;
  `daedalus_fv1` dangling too) and the root filesystem is 99% full.
  Do not expect to regenerate any CFD figure here.

## What stays with the CFD agent (do NOT duplicate)

- Task #31 (solver-source rename) — compute-repo work.

## Verification ritual before every commit

pdflatex x2 (+bibtex if bib touched): 0 errors, 0 undefined; grep the
edited terms for stragglers; if figures changed, render-and-look;
stage explicitly; commit message = why only.
