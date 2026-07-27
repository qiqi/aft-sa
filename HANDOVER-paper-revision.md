# Handover: SA-AI paper revision (2026-07-27)

For the agent taking over the paper revision. The previous agent keeps
ONLY the long-running CFD (see "What stays with the CFD agent") and
will deposit harvested result JSONs; everything else here is yours,
including responding to `agent-paper-review/`.

## What this is

`paper/sa-ai.tex` — AIAA Journal submission for the SA-AI one-equation
transition model (repo branch `paper-composite-snapshot`, solver repo
`~/flexcompute/compute` branch `explore-lambda-v`, canon commit
05d05a123b = fv1 bypass). 116 pp, builds clean (pdflatex x2 + bibtex:
0 errors, 0 undefined). Read `ONBOARDING.md` first if you haven't.
Fifty review passes have run; every number surviving in the tex has
been independently verified at least once — treat that as an asset you
must not squander.

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
`ls agent-paper-review/*.md` newest-first; everything through
2026-07-27-1440 is answered.

## Current paper state (HEAD ~ the Fig-5 digitization commit)

- Structure: I intro / II model / III flat plate / IV NLF / V Eppler /
  VI Eppler Re-sweep / VII attachment-anchored turbulent branch
  (sec:bistability, promoted from appendix today, user decision) /
  VIII Daedalus / IX spheroid (sec:spheroid) / conclusion (with the
  drag-crisis outlook paragraph) / Appendices A-H (H = spheroid
  station waterfalls).
- All pass-46..50 verified fixes are in; conventions single-homed
  (five-row layout + line conventions + x u_e^2: Sec IV; e^N
  substitution rule + cross-check: Sec IV; digitized-reference
  provenance: Appendix F block).

## OPEN WORK — ordered by what unblocks what

### A. Cavity-L2 harvest (data arrives ~17:40 today; tex work = YOURS)

The Daedalus cavity-L2 fv1 recomputation chain (a4 done: CL 1.0217/
CD 0.01979; a5 done: 1.1223/0.02234; a6 running) completes today. The
CFD agent will harvest into `flow360_fv1/daedalus_fv1_results.json`
and build the chi caches, then note completion in RESPONSES.md. Then:
1. Fill the `tab:daetotals` dashes; remove daggers from daepolar /
   daesurf4/5/6 / section-sheet captions; close the Sec VII(now VIII)
   held-solutions sentences and the II.F "Status of the results"
   disclosure (the bypass/held/dagger disclosure D1 then evaporates —
   grep `dagger`, `held`, `bypass-inactive`).
2. Regenerate Daedalus figures ONLY after the chi caches exist
   (rule 4): `regen_daedalus_polar_sectional.py`,
   `regen_daedalus_surface_maps.py`, `regen_daedalus_section_sheets.py`
   (roots: SAAI_DAE_ROOT / D_STR / D_CAV env-and-dict patterns already
   point at daedalus_fv1 with cavity_L2 exceptions to flip).
3. Compute the finest-grid two-family claims at the final kernel
   (currently scoped as "awaits the held recomputation").

### B. Spheroid section overhaul (task #37; USER DIRECTIVES verbatim)

User: digitize the experimental paper (Stock/spheroid.pdf), overlay on
the computed waterfalls in Appendix H, move selected conditions into
the main paper. "Any figure without an experimental comparison on top
of it belongs to the appendix. We should have as many experimental
comparisons as possible. And as many comparisons against other
literature (computation) as possible."

State:
- DONE: Stock Figs 4 (a10) and 5 (a29.7) measured Cft+gamma_w chains
  digitized -> `data/stock2006_fig4_digitized.json`,
  `..._fig5_...json` by `repro/cfd/digitize_stock_waterfalls.py`.
  Chain identity was verified at FULL RESOLUTION with per-station
  colored overlays; unrecoverable tangle regions are truncated per
  station (`truncate` dict in the script records every cut; fronts
  and windward dives all retained). Known gap: fig5 gam x/a=0.766
  deep dive (85-110 deg) outruns the tracker's slope cap.
  Source images + check PNGs: /local_data/qiqi/sa-ai/stock_digitize.
- TODO: Figs 2-3 (Cp waterfalls, p3_img0/p3_img1 of spheroid.pdf,
  13 stations, dCp=0.42/station offsets — pass-43-verified) — same
  pipeline, add FIGS entries (tick-detect the calibrations first,
  same as fig4/fig5; measure, don't assume).
- TODO: overlay measured symbols on our waterfalls
  (`repro/cfd/regen_spheroid_station_profiles.py`, CASES dict) and
  our maps; then split figures main-vs-appendix by the user's rule.
- Literature-computation overlays: Stock's own e^N fronts are already
  in the digitized front-compare fig (fig:spheroidfront); his Figs
  14/16/17 carry measured+computed fronts at the OTHER conditions
  (6.5e6, 7.2e6) matching our re65/re72 runs — digitize when those
  runs land. Strongest modern source: the 1st AIAA Transition
  Modeling & Prediction Workshop spheroid case (Re=6.5e6; Coder
  summary = the same document our NLF workshop overlays came from;
  check transitionmodeling.larc.nasa.gov/workshop_i for data files
  before digitizing slides). The experiment's source-of-record is
  Kreplin/Vollmers/Meier (DFVLR 1985) — the user was advised to
  obtain it; it contains stations/conditions Stock never plotted.
- Sec IX text: currently "in progress" language at three places
  (pass-46 O2 flagged); rework experiment-first once overlays exist.

### C. OpenFOAM Eppler L2 (lands tomorrow morning)

When the OF thread reports the Eppler L2 ladder complete:
1. The CFD agent re-runs `export_airfoil_summary.py` (auto-picks-up
   filled CSV rows; no-VTK guard skips unsynced cases) and tells you.
2. YOUR call, per the standing pass-45/48 posture: the Eppler overlay
   is held because the port omits the fv1 bypass (model content,
   -5..-10 counts at exactly those conditions). The OF thread's L2
   data showed the NLF Cd gap collapse was grid resolution; if the
   Eppler gap collapses similarly at L2, the objection softens — then
   extend the overlays/table like the NLF ones (L1+L2, chi=1, scoped
   claims, Cd honesty clause), else keep held with the reason stated.
   Their `build_airfoil_case.py` uncommitted mods ride with that
   commit (pass-48 minor 6).

### D. Sec VII (bistability) critical-Re re-emit (CFD agent finishes)

Pass 50's addendum proved the printed critical bracket
(Re_r 2.78-2.89e5) is an iteration-cap artifact; true bracket inside
(537.5, 700] in L. A cap-proof verdict re-bisection is RUNNING
(runlogs/stag_rebisect.log -> data/stagnation_bistability_rebisect.json).
The CFD agent will merge it into `data/stagnation_bistability.json`,
regenerate `figs/stagnation_bistability.pdf`, and update the two
quoted numbers in sec:bistability (the interval sentence + the caption
band/"L ~ 530"); if that has NOT happened when you take over, it's
top priority — the printed number is known-wrong until then. The
seed-fed opening clause gets STRONGER (NLF Re_r ~ 7e4 vs the higher
threshold) — check the sentence still reads right.

### E. Pass-46 length plan remainder (task #40)

Executed: D2-D6, D9-D11, L4, O3, O5, O7(i) + pass-47 corrections.
Remaining: L1-L3 language sweep (LAST, one section per pass,
move-don't-paraphrase), D9 remainder (Sec V ~l.1806 and Sec VI ~l.2090
seed restatements -> pointers), O7(iv) declined (number-dense; reviewer
accepted). USER-GATED (decided): keep one PDF for now (no supplemental,
no O1 sheet curation); Eppler mesh close-ups already dropped; ch.48
footnote dropped; epigraphs stay.

### F. Small open threads

- Conclusion K-gate sentence + drag-crisis outlook: settled and
  verified (pass 50); the K-gate model revision itself
  (1250/1315/pass-49 memos) is FOLLOW-ON work, not this paper.
- zheng_lei_2016 bib: volume added; three pre-existing bibtex
  empty-pages warnings (vaningen_2008, medida_baeder_2011,
  cakmakcioglu_2020) — fill when convenient.
- Pass-50 minor 4: if you tighten the verdict criterion further,
  update "steady to the solver's tolerance".

## Pending user decisions (ask before acting)

- Whether the spheroid main/appendix split (B) needs sign-off per
  figure once the overlays exist (the rule is decided; taste isn't).
- Companion-paper flag list (near-pole surrogate work) still owed to
  the user — unrelated to this paper but recorded in memory.

## What stays with the CFD agent (do NOT duplicate)

- Cavity-L2 a6 completion + result harvest + chi caches (monitor
  bkjc7up4h).
- The stagnation re-bisection run + JSON/figure re-emit + the two tex
  numbers in D above (they will notify via RESPONSES.md when done —
  coordinate so you don't both edit sec:bistability).
- Spheroid campaign resumption when GPUs free (re65a10-L2 +
  re72a2p5), harvest into sphere_campaign_spheroid_results.json +
  tab:sphtotals dash-filling data.
- OpenFOAM Eppler L2 exporter re-run (C.1).
- Task #31 (solver-source rename) — compute-repo work.

## Verification ritual before every commit

pdflatex x2 (+bibtex if bib touched): 0 errors, 0 undefined; grep the
edited terms for stragglers; if figures changed, render-and-look;
stage explicitly; commit message = why only.
