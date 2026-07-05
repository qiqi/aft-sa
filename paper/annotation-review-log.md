# Annotation review log

Persistent record of review comments on `main.tex` and how each was addressed.
The source of comments is `annotated.pdf`, which is **overwritten** with each new
version — this log is the durable history. Newest rounds appended at the bottom.

---

## Round 1 — Γ-sigmoid design intent (1 comment)

**[1] §II.A, `eq:rate` discussion (highlight on "…only fuller, inflectional
(near-separation) profiles").**
> This section should clarify the design intent. For example, here the design
> intent is that Gamma=2 should fully activate the sigmoid, giving a rate at which
> disturbances would amplify in a separated shear layer. Gamma=1 should almost, but
> not completely, deactivate the sigmoid, giving a much smaller rate at which TS
> waves amplify in a Blasius BL.

*Addressed:* Added a sentence after "…reach the top." spelling out the intent — at
Γ=2 the sigmoid is fully on and `a→a_max` (separated-shear-layer growth rate); at
Γ=1 it is nearly (deliberately not fully) off, leaving the small rate of TS
amplification in a Blasius layer; the two values are fixed in Sec. `calib`.

---

## Round 2 — §III framing (5 comments)

**[1] §III section title.**
> Shouldn't this section just be "determination of model constants"?

*Addressed:* Renamed §III to **"Determination of model constants"** (label
`sec:calib` kept, so all cross-refs hold).

**[2] §III intro paragraph.**
> I'd just state, mater-of-factly, how each constant will be determined.

*Addressed:* Rewrote the opener to plainly list what fixes each constant (`a_max` ←
free-shear ceiling; `(s,g_c)` ← Blasius e^N envelope slope; `Re_Ω^f` ← onset
Reynolds number; `K_λ` ← rate Drela's critical Re rises), dropping the defensive
"rather than fit … even K_λ is derived, not fit" tone.

**[3] §III, a_max/envelope sentence.**
> across the established flat-plate transition Re_x range?

*Addressed:* Reworded to "N accumulating to ≈14 **across the established flat-plate
transition Re_x range**."

**[4] §III, the (s,g_c) paragraph.**
> Perhaps say here that with Gamma=2 fully activated, Gamma=1 barely activated, g_c
> has to be around 1.5. I would also admit that we didn't spend time tuning for
> anything other than Gamma=1 or Gamma=2. So there is room for further refinement of
> this pair of numbers.

*Addressed:* Added that `g_c≈1.5` is essentially forced (fully on at Γ=2, barely on
at Γ=1 → center midway), that only the Γ=1/Γ=2 anchors were tuned, and that
`(s,g_c)` remains open to refinement.

**[5] §III, the cliff paragraph ("The cliff needs no Γ-tilt…").**
> Do not mention any Gamma-tilt. Instead, make it clear that the lower Re_theta^c
> for higher H boundary layers is automatically handled by the higher Re_Omega /
> Re_theta of these adverse PG boundary layers. So that lambda_p only needs to be
> responsible for shifting the Re_Omega^c of the faverable PG.

*Addressed:* Removed the "no Γ-tilt" framing. Now: adverse (high-H) layers
transition earliest (Drela's `Re_θ,crit` falls with H) and carry a larger
`Re_Ω/Re_θ` than Blasius (cites `fig:indicatorplane`), so they clear the fixed
floor at a lower `Re_θ` automatically; `λ_p` is left the single job of raising
`Re_Ω^c` in favorable gradients.

---

## Round 3 — 2026-07-04 (15 comments)

**[1] §II.A, invariance sentence.**
> Single sentence too long

*Addressed:* Split the material-derivative/wall-relative invariance sentence into
four short ones.

**[2] §II.A, γ–Re_θ note.**
> Historical note doesn't belong here. Remove

*Addressed:* Removed the "This is not automatic: the γ–Re_θ model … not Galilean
invariant, due to its explicit use of the velocity vector" sentence.

**[3] Fig. `indicator_plane` (§II).**
> Remove the grayscale make all lines black. We already have each line marked with
> its beta. Also remove the text about Gamma=1 in the plot.

*Addressed:* Regenerated with all-black curves; removed the in-plot "Γ=1 (attached)"
text; caption updated (no more "grayscale/dark/light"; β labels identify curves).

**[4] Fig. `indicator_plane` caption + §II generally.**
> Remove this sentence. Remove other pure historical note sentences in Section II
> too.

*Addressed:* Removed the "To our knowledge boundary-layer profiles have not
previously been charted…" novelty sentence. Scanned §II; the remaining cited
passages (`Re_Ω` provenance, "rather than" phrases) are model formulation, not pure
historical notes, so left intact.

**[5] §III, `K_λ` derivation / Eq. (10) / Fig. 4.**
> should be explicit here that the equation in (10) is general but the numerical
> value of the denominator assumes K_lambda=10 and would inrease as K_lambda
> increases. Thus giving rise to the self-consistency condition. Maybe Figure 4 is
> too much real estate for this. Instead just give a few numbers: if K_lambda is 9,
> what Eqn (10) evaluates to, and maybe 5 and 15.

*Addressed:* Removed Figure 4 (self-consistency plot). Text now states the
denominator (worst-point slope) depends on the `K_λ` used, and gives Eq. (10)
evaluated at `K_λ`=5, 9, 10 → 12.6, 10.0, 9.5, crossing ≈9.7. (K_λ=15 omitted: the
worst-point criterion degenerates for K_λ≳12.)

**[6] §III, "This pins K_λ…" sentence.**
> Just say "The sensitivity of the worst lambda_p to beta pins K_lambda."

*Addressed:* Replaced with exactly that sentence.

**[7] §III, worst-point slope.**
> for K_lambda=10

*Addressed:* Annotated the worst-point slope `dλ_p^worst/dβ≈1.3` as "(at K_λ=10)."

**[8] §III, "The point is not the second digit…" sentence.**
> Just say we adopt the round value of 10. A more careful calibration...

*Addressed:* Trimmed to "we adopt the round value K_λ=10; its magnitude is set by
the Drela–Giles envelope and the Falkner–Skan family, not fit to the transition
cases, and a more careful calibration is left to future work."

**[9] §III, "On a Blasius profile the local rate a·ω…" paragraph.**
> This paragrapph... remove? Nothing to do with determining the model constants?

*Addressed:* Removed the paragraph.

**[10] §III, `c_ν,aft=1/12`.**
> Is this the first place in the paper we give the numerical value of 1/12? In fact
> this value of 1/12 is in fact determined from this parabolized equation solution.
> Not modifying is way too large. 1/4 is not small enough. 1/12 seems small enough.
> Again more careful tuning may make this better in future work.

*Addressed:* The value is now explicitly determined from the parabolized (Blasius
transport) solution: `c_ν,aft`=1 → envelope collapses to N≈2; 1/4 still too
diffusive; 1/12 → N≈14 (Drela range); rounded for adequacy, future tuning noted.
Fixed the premature "introduced above" → "(determined below)".

**[11] §III, handover coefficients.**
> Did we ever discuss the model coefficients in the handover? We didn't really tune
> it, just set ab initio as start handing off when xi>1 and basically finish when
> xi > cv1, or some factor of it. Again future tuning may make the hand off better
> or disturb the original SA tuning even less.

*Addressed:* Added a paragraph: the handover width `τ` is set ab initio (handoff
begins at χ>1, essentially complete by χ∼`c_v1`; `τ=4` realizes that span), not
optimized, future tuning noted.

**[12] Fig. `wall_layer` (§III).**
> Text in this figure is way too small. Redo the figure making the figure size
> smaller so that text appears bigger. Also make this figure black and white too.
> Remove all in-plot text (and if they are necessary for understanding move it
> elsewhere).

*Addressed:* Regenerated: smaller figure + larger fonts, fully B&W
(solid/dashed/dotted/dash-dot), all in-plot text/legends/boxes removed; the line
key, peak (4.6%), and ΔB (−0.33) moved to the caption.

**[13] §III, Mack correlation.**
> Don't be dramatic. State it mater of factly -- adopting Mack's isn't a big deal?

*Addressed:* Stated matter-of-factly; dropped "exactly," "adopt it wholesale rather
than fit it," and "This is legitimate because."

**[14] §III, "Assembled one-equation model" subsection.**
> This section should follow what NASA turbulence modeling resources website does:
> first state the equation, then all the terms (both old and new), then the
> constants.

*Addressed:* Restructured to NASA-TMR order: transport equation first, then all
terms (blended P/D + baseline SA closure), then all constants.

**[15] §III, constants closing sentence.**
> I'd say none of these are really fine-tuned?

*Addressed:* Changed to "None of these is finely tuned: each is fixed by a canonical
limit and, where convenient, rounded…" (rather than singling out K_λ and Re_Ω^f).

---

## Structural revisions (from discussion, not PDF annotations)

**2026-07-04 — §III subsectioned.** §III ("Determination of model constants") had
grown to ~400 source lines as a flat run of paragraphs with a single trailing
subsection, and read as tiresome. Split into five subsections (the last pre-existing):
III.A The amplification-rate kernel · III.B Favorable-gradient onset delay · III.C
Cross-stream diffusion, handover, and the turbulent layer · III.D Freestream-turbulence
receptivity · III.E Assembled one-equation model. Chose the "fewer headers" option
(merged the diffusion-determination and turbulent-footprint material into III.C). Also
deleted the redundant "The same kernel … is used unchanged" recap paragraph (its content
survives in the III.E constants closer) and trimmed the receptivity opener to avoid
repeating the new subsection title.

**2026-07-04 — airfoil mesh tables + y+ resolution table (from discussion).** Added
quantitative mesh-description tables to §V and §VI (nodes, elements, tangential Δs at
five chord stations, first-cell/outer wall-normal spacing, y+ 95/99 pct; cavity =
all-triangular, structured = all-quad). Added a coarse-mesh (y+) robustness table to
§III.C from a 1D wall-layer ODE solved at first-cell y+ = 0.25…8, showing SA-AI's grid
sensitivity matches standard SA's (regen scripts: regen_yplus_table.py; mesh metrics via
/tmp/meshtab.py-style vtk extraction).

---

## Round 4 — 2026-07-04 (18 comments)

**[1] §IV, solver preamble.** *"Remove this part of the sentence."* (numerical scheme is
otherwise irrelevant to the model) → removed.

**[2] §IV, solver preamble.** *"Remove these too, all seem unnecessary."* (the analytic-
Jacobian / "only change from baseline SA" sentence) → removed.

**[3] §IV, flat-plate setup.** *"remove"* (with the solver's own SA source terms in a
thin-layer boundary-layer march) → removed.

**[4] §IV.** *"remove. the sentence ends with 'adopted'."* (without adjustment; the flat
plate is a check, not a fit) → removed; sentence now ends "…adopted."

**[5] §IV.** *"Unnecesary. Remove"* (The standard SA model, by contrast, amplifies its own
production…) → removed.

**[6] §IV.** *"I can't see it in the figures… remove the sentence."* (digitized
Schubauer–Skramstad Fig. 7 band) → removed the figure-band sentence and the later "onset
falls within the S–S band throughout" claim.

**[7] §IV.** *"remove"* (—no constant fit on this plate—) → removed.

**[8] §IV.** *"point out instead that the AGS onset lies in between when χ crosses 1 and
crosses c_v1."* → reworded: the AGS onset lies between the χ=1 and χ=c_v1 crossings, with
χ=1 tracking it to ~7%.

**[9] §IV.** *"Remove. The same thing is said too many times… de-dupe."* → removed the
redundant Mack-B / Re_θ,crit sentence; de-dup scan across the manuscript, also trimmed a
second "not fit to the transition cases" in §III.B (the soft-floor statement now appears
once, in §III.A).

**[10] §V/§VI titles.** *"Just title it 'NLF(1)-0416 airfoil at 4M Reynolds number', and
similarly for the next."* → §V → "The NLF(1)-0416 airfoil at Re=4×10⁶"; §VI → "The Eppler
387 airfoil at Re=2×10⁵".

**[11] §V.** *"remove this part… review the entire paper for compactness (Elements of
Style)."* → removed "is taken as the primary airfoil test: the same geometry that";
targeted compactness trims in §IV/§V (a full-paper sweep noted as a separate pass).

**[12] §V.** *"remove"* (both surfaces transition aft at α≈0—the canonical low-drag-bucket
geometry) → removed.

**[13] §V.** *"'marginal' is ambiguous…"* → reworded: across a wide α range the transition
on one surface or the other is governed by slow Tollmien–Schlichting accumulation rather
than a severe adverse gradient.

**[14] §V.** *"Not just triangle everywhere, but generated with the same Delaunay-based
algorithm everywhere (confirm from code)."* → confirmed ConstrainedDelaunayMesher in
mesher/mesh2d.cpp; reworded to "the same constrained-Delaunay triangulation everywhere, the
boundary layer included, with no structured near-wall layer."

**[15] §V.** *"…not all triangles conform to the metric, per the max y+ in the mesh tables."*
→ added that the near-degenerate triangles are reflected in the elevated peak y+ of the
cavity grids in Table~\ref{tab:nlfmesh}.

**[16] Floats.** *"How can we prevent figures procrastinating into much later pages?"* →
added \usepackage[section]{placeins} so floats cannot cross section boundaries.

**[17] §V, row-2 indicator description.** *"Discuss in the Eppler section, not here."* →
removed the "for the Eppler 387 … max Γ" clause from the NLF-section paragraph (the Eppler
section already describes it).

**[18] §V.** *"Did we introduce mfoil? Citation?"* → introduced mfoil on first mention as a
coupled viscous–inviscid e^N panel solver and cited Fidkowski, AIAA J 60(4), 2022
(10.2514/1.J061341).

---

## More structural revisions (from discussion)

**2026-07-04 — mesh tables: added 0.001c wall-normal row.** Added a "cell @ 0.001c"
wall-normal-spacing row to both airfoil mesh tables (between first-cell h₀ and the
0.5c outer cell), measured by nearest-neighbour edge length at wallDistance≈0.001c
(cross-checked against the design geometric growth: cav L0 227e-6 c vs design ~216e-6).

**2026-07-04 — Elements-of-Style compactness pass.** Prose was already tight from
prior rounds (only two "in terms of" paper-wide), so this was sentence-level
tightening where genuine slack existed: intro (redundant/throat-clearing trims),
§II.A (θ-impracticality and Re_Ω sentences), and the Conclusion (fixed a "regime …
regime" repetition, de-passivised, dropped "wholesale"). The results discussions and
§III were confirmed already tight and left as-is. Not an exhaustive every-sentence
rewrite — a focused pass on the passages that carried fat.

**2026-07-04 — §V.A: L0 streamwise-resolution artifact (from user's ParaView
observation).** User asked to confirm/correct, via careful solution inspection, the
"upstream influence" seen on the structured L0 NLF mesh: coarse streamwise
discretization at the sudden laminar→turbulent BL-shape change under-resolves the
transition, producing discretization error that distorts the profile *upstream* and
plateaus χ at 0.1–1 for one–two grid points before the jump. Verified against the
volume solutions (both α=0 and α=4, both surfaces): χ climbs to ≈1 then stalls/dips
over the two cells before transition (α=4: 0.80→0.58; α=0: 1.03→0.75) then jumps
1–2 orders across one cell. Confirmed the user's refined mechanism: the effect is NOT
the σ_t handover (χ<1 there, σ_t=0); it is the under-resolved downstream jump
reaching 2–3 cells upstream → the profile-fullness indicator Γ at the ν̃-peak
collapses (≈1.1→≈0.5) → the Γ-sigmoid gate (g_c=1.572) drops the amplification rate
>20× (a≈0.013→0.0005), stalling growth. Purely numerical: L1/L2 hold Γ≈1.1 at the
same station and χ rises monotonically; only trace is a small onset shift that
converges out by L1. Added the paragraph + Fig. (l0_artifact.pdf, regen_l0_artifact.py:
3-panel χ/Γ/a vs x, L0 vs L2, α=4 upper) to §V.A.

**2026-07-04 — §V: experimental transition points + mfoil high-α double-check
(from user request).** Two linked tasks the user asked for directly (not a PDF round).
(a) *Experimental NLF transition at the 4 AoAs.* Source is Somers NASA TP-1861 Fig. 9(d)
(R=4×10⁶): transition-orifice front x_tr vs c_l (open=laminar, solid=turbulent), which
has no numerical table — hand-digitized from the scanned figure (rendered via PyMuPDF;
report p.51 = PDF p.53). Read x_tr(c_l) for both surfaces and mapped to α through the
computed section c_l. Lower surface resolved across the whole range (≈0.55→0.66 c);
upper orifices stop at the forwardmost transition orifice x/c≈0.30, so upper onset is
unresolved for c_l≳0.7 (α≥4). Direct (α,c_l) anchors from the R=2×10⁶ oil-flow photos
(Fig. 7): 0°→0.4, 4°→0.9, 8°→1.3. Added Table~\ref{tab:nlftrans} (SA-AI L2 vs mfoil
vs Exp, both surfaces) + an interpreting paragraph; added the somers_1981 bib entry;
reproducible generator regen_nlf_transition.py.
(b) *mfoil at α=9 and 15 "seems wrong."* Confirmed and made precise via the mfoil pkl:
the e^N envelope NEVER reaches N_crit=9 on the surface that separates — α=9 LOWER
(N_max=6.8, H=9.7) and α=15 UPPER (N_max=5.7, H=7.4) — so those reported "transitions"
are laminar-separation points, not e⁹ onsets; the *other* surface stays attached and
genuine at each. mfoil c_l also over-predicted (α=9: 1.67 vs RANS 1.48–1.51 and exp
~1.3 at α=8; α=15: 1.85, deep post-stall past exp c_l,max≈1.68). Flagged the two
entries with † in the table, and corrected the paper's overclaims: (i) body sentence
"tracks e⁹ within ~5% across 0≤α≤15°" → defers onset comparison to Table~\ref{tab:nlftrans};
(ii) fig:nlfcflow caption 0.03–0.05 c → 0.06 c vs e⁹ and 0.05 c vs measured;
(iii) fig:nlfcfhigh caption now states mfoil is unreliable at α=9,15 (separation before
N_crit) and defers to experiment. Compiles clean, 37 pp.

---

## Round 5 (annotated.pdf, 2026-07-04 14:04 — 3 comments; user noted some already addressed)

**[24.1] §V, fig:nlfcfhigh highlight ("α=9 ... α=15 ... lower transition remains aft of mid-chord").**
*"The mfoil results look vastly wrong. Are you sure they are run at the same AoA and Reynolds
number and same geometry? Worth double checking, and perhaps fixing, with the polar plot too."*
→ (Largely addressed the prior turn via Table 4 + caption caveats.) This round: **verified the
mfoil setup is identical to the RANS setup** — `run_mfoil_nlf.py` reads the same NLF(1)-0416
`.dat` coordinates (x:0→1), `setoper(Re=4e6, Ma=0.1)`, `ncrit=9`, per-α runs; pkl confirms
Re=4e6, ncrit=9. So it is **not** a setup bug: the "vastly wrong" high-α behavior is genuine
e^N-panel physics (laminar BL separates before N reaches N_crit — α=9 lower N_max=6.8/H=9.7,
α=15 upper N_max=5.7/H=7.4 — and c_l over-predicted). **Fixed the polar** (`regen_nlf_polar.py`):
mfoil now drawn as a connected low-drag bucket only where attached (α=0,2.5,4); α=9,15 shown as
grey × with no connecting line (they were the near-vertical shoot-up to c_l=1.67 at bucket-level
drag that looked "vastly wrong"). Updated the polar caption and the polar-discussion paragraph
to state the setup was verified identical and the breakdown is the known panel-method stall
limitation.

**[24.2] §V, "refinement" highlight.** *"Also can we extract (you worked on this before but
didn't get to the end) the experimental transition locations? Give me a .md file explaining,
with graphics if that helps, how you extracted the experimental transition locations from paper."*
→ Extraction itself was done the prior turn (Table 4 from Somers Fig. 9d). This round created the
requested explainer: **`experimental-transition-extraction.md`** with embedded graphics
(`figs/exp_transition/`: Fig. 9d full + zoom, the three Fig. 7 oil-flow photos, and a
`digitization_overlay.png` showing the digitized fronts with the α read-offs). Documents the
source, the open/solid orifice convention, the PyMuPDF render + page-offset, the digitized knots,
the matched-c_l → α mapping, oil-flow corroboration, reproducibility, and ±0.03c caveat.

**[25.1] §VI (Eppler) highlight ("where—with no constant fit to it—the model is a pure prediction").**
*"Again I feel this is emphasized too many times. Remove."* → Removed the clause; the sentence now
reads "...closes the study with the adverse-pressure-gradient regime. The McGhee..." (the
no-fit/prediction point is still made elsewhere).

Compiles clean, 38 pp.

**2026-07-04 — CORRECTION to Round 5 [24.1] mfoil α=9 (user pushed back: "How did
you respond to the mfoil question for alpha=9? The cf plot looks wrong, and the drag
polar is off").** The Round-5 framing ("mfoil separates before N_crit; the entries
are separation points") was a symptom, not the root cause. **Re-checked mfoil's
convergence flag (`m.glob.conv`)**: mfoil **does not converge at α=9** (conv=False) —
the pkl values (c_l=1.672, c_d=0.0074) are a **non-converged iterate**, which is
exactly why the Cf plot and the near-vertical polar shoot-up (c_l=1.67 at bucket-level
drag) looked wrong. Full sweep: converges through **α=7 (c_l=1.33, physical, drag
rising)**; **α=8,9 do not converge**; **α=15 DOES converge but to a stalled solution**
(upper H=7.4, c_d=0.0316, beyond measured c_l,max≈1.68). More iterations don't help
(limit-cycles near stall). Verified setup identical to RANS (same .dat, Re=4e6, Ma=0.1,
ncrit=9), so it is a genuine solver limitation.
**Fixes:**
- `run_mfoil_nlf.py`: now stores `conv` flag and adds converged α=6,7 points; re-ran pkl.
- Polar (`regen_nlf_polar.py`): mfoil plotted ONLY where converged AND upper-attached
  (α≤7); α=9 (non-conv) and α=15 (stalled) omitted entirely — no more × markers, no
  more unphysical shoot-up. Legend/caption/discussion updated.
- Cf figures (`regen_nlf_v2.py`): added `_mfoil_usable()` guard (conv AND upper H<4);
  high-α figure now omits all mfoil overlays (rows 3–5) AND drops the mfoil legend
  entry + "mfoil N" right-axis label when nothing is plotted.
- Table 4: mfoil α=9,15 entries → "n/a†"; footnote rewritten to the non-convergence /
  stalled-solution explanation. Body paragraph + fig:nlfcfhigh caption + polar
  discussion all rewritten from "separates before N_crit" to the correct
  "does not converge at α=9 / stalls at α=15; valid reference only through α=7".
- `regen_nlf_transition.py`: prints conv flag and marks mfoil n/a with the reason.
Compiles clean, 37 pp.

**2026-07-04 — Fig. 11 (l0_artifact) removed; L0 artifact explained in words
(user: "Fig 11: let's explain the cav L0 problem with words not figure. The figure
isn't very useful").** Deleted the `fig:l0artifact` figure environment (l0_artifact.pdf,
the 3-panel χ/Γ/a-vs-x plot) and its inline `Fig.~\ref{}` citation. The §V.A prose
paragraph already carried the full explanation and all quantitative values (χ stalls/dips
0.80→0.58 at α=4°, 1.03→0.75 at α=0°; Γ at the ν̃-peak collapses ≈1.1→0.5; Γ-sigmoid gate
drops a ≈0.013→0.0005; purely numerical, converges out by L1), so it now stands alone.
Note: the artifact is on the STRUCTURED L0 (kept accurate wording despite the user's "cav"
phrasing — Fig 11 and the data are the structured mesh; the cavity-L0 turbulent-Cf /
LE-peak under-resolution is a separate point already noted in the Cf-figure discussion).
Figures renumbered 17→16 (nlf polar is now Fig 11); all refs use \ref and resolve.
l0_artifact.pdf and regen_l0_artifact.py left on disk, now unreferenced. Compiles clean, 37 pp.

**2026-07-04 — Table 4 experimental transition values updated to user's re-reading
of Somers Fig. 9(d).** User re-digitized the figure and supplied authoritative values;
replaced the Exp columns:
- α=0 (c_l=0.49): upper 0.43→**0.38**, lower 0.56→**0.55**
- α=4 (c_l=0.95): upper —→**0.31** (extrapolated, just ahead of forwardmost orifice
  x/c≈0.30), lower 0.65→**0.62**
- α=9 (c_l=1.48): upper — (extrapolation not useful), lower 0.66→**0.64**
- α=15 (c_l=1.94): upper —, lower —→**0.66** (extrapolated along the flat lower branch,
  beyond measured c_l,max≈1.68)
Marked the two extrapolated entries with ‡ and added a caption note. Updated the prose
(SA-AI now matches the lower-surface front to within ~0.03c at every incidence; α=0 upper
near-exact: SA-AI 0.37 vs exp 0.38; α=4 upper SA-AI 0.26 vs extrapolated exp 0.31). Synced
`regen_nlf_transition.py` (now uses an authoritative EXP_BY_ALPHA dict with extrapolation
flags instead of interpolating my knots) and the `experimental-transition-extraction.md`
result table. Compiles clean, 37 pp.

**2026-07-04 — Table 6 (Eppler) reframed from χ-onset transition to turbulent
REATTACHMENT (user: comparing a χ-defined transition location to experiment is
apples-to-oranges for an LSB; compare reattachment, defined as Cf recovering through
+0.001; experiment = aft edge of the grey band in the Cf rows).** Reattachment computed
as the most-downstream upward crossing of signed Cf=+1e-3 in the LSB window (robust to
the marginal α=7 double-graze where Cf only just reaches zero; keying off the global
Cf-min instead picked TE separation near stall). New reproducible generator
`regen_epp_reattach.py`. Values (cav/str L2 vs exp): α=0 0.724/0.729 vs 0.74; α=2
0.677/0.680 vs 0.67; α=5 0.598/0.600 vs 0.59; α=7 0.472/0.477 vs 0.48 — SA-AI matches
oil-flow reattachment to within ~0.01–0.02c at ALL four incidences (far better than the
old χ-onset, which was 0.10–0.15c off mfoil and not directly comparable to experiment).
mfoil reattaches ~0.02–0.04c downstream at α=0,2,5 and is unreliable at α=7 (edge of
convergence). Rewrote the table + caption + discussion; dropped the old "2-efold
earlier-than-e^9 offset" paragraph (was about onset, now moot). Exp reattachment values
(0.74/0.67/0.59/0.48) now consistent with the figure's grey-band aft edge (EXP_LSB); the
old table's α=5 "0.55" was inconsistent with the 0.59 band and is corrected. Compiles
clean, 37 pp.

**2026-07-04 — xfoil cross-check of mfoil high-AoA non-convergence (user request).**
Setup: the packaged xfoil (6.99) SIGFPE-crashed on any solve (even NACA0012 inviscid α=0)
— NOT a display issue (starts fine headless; xvfb NOT needed). Cause: gfortran
`-ffpe-trap` compiled in. Fix: LD_PRELOAD a shim with no-op `_gfortran_set_fpe`/
`_gfortran_set_options` (`/tmp/nofpe.so`); xfoil then runs headless and produces polars.
FINDING (answers "is it mfoil-specific?"): YES. xfoil converges COLD at NLF α=9
(CL=1.524, ≈RANS 1.48–1.51) and α=15 (CL=1.847), and at Eppler α=7 gives a physical
CL=1.168 (monotone above α=5's 0.96) where mfoil returned an unphysical 0.928. So the
high-AoA trouble is mfoil's Newton globalization, NOT a limitation of coupled e^N panel
methods. CORRECTED the paper's false claim ("well-known non-convergence of coupled
inviscid–viscous panel methods near/beyond stall") → now states it is mfoil-specific and
cites XFOIL (added `drela_xfoil_1989` bib) converging at the same conditions. Compiles
clean, 37 pp.

**2026-07-04 — CfVec sign validation (user request), IN PROGRESS.** Confirmed `CfVec` is a
valid Flow360 surface output field (was not requested; we only asked for scalar `Cf` and
reconstructed the sign from a near-wall velocity probe · LE→TE tangent). Launched a fork
`cfvec_cavL0_eppler_a5` (L0 Eppler α=5, clear bubble) with `CfVec` added to
surfaceOutput, run on GPU1 via rans.env/solve; the sign-reconstruction code is
mesh-independent so L0 validates it. TODO next: compare sign(CfVec·t_LE→TE) vs the
velocity-probe sign node-by-node along the upper surface to confirm the signed-Cf (and
hence the reattachment table) is correct.

## Round 6 (annotated.pdf 2026-07-04 20:00) + user decisions

**[27.1] "Remove [signed-Cf reconstruction paragraph] -- directly use cfVec." → user
relaxed: "no need to re-run, just don't state it as a hack."** Validated the reconstruction
against a Flow360 CfVec fork (cfvec_cavL0_eppler_a5): |CfVec|=scalar Cf exactly, sign
agreement 100%. Reworded the §VI signed-Cf sentence: dropped the "Flow360 writes Cf as
scalar magnitude, so we reconstruct by sampling velocity..." framing; now "the signed Cf is
the wall skin friction oriented along the LE→TE tangent (equivalently Flow360's CfVec, which
we verified node-for-node)."

**[32.1] "remove the 0.10–0.15c-upstream-of-mfoil discussion" → already gone** (removed in the
Table 6 reattachment reframe).

**[34.1] tunnel wall/finite-span drag-offset attribution → removed** the speculation (now just
"shared with mfoil, common to both 2D predictions").
**[34.2/3/4/5] Conclusion trims → done**: removed "occupying a cell left empty by…",
"it replaces", "of increasing complexity", "adopted rather than fit", "not fit".

**User decision 2: add xfoil to FILL mfoil's failed cases for Cp/Cf/Cl/Cd, keep mfoil's N.**
Generated xfoil pkls (Cp via CPWR, signed Cf via DUMP, xtr/cl/cd via polar) — xfoil_nlf0416_Re4M.pkl,
xfoil_eppler387_Re200k.pkl (script /tmp/gen_xfoil.py; runs headless via the /tmp/nofpe.so shim).
Integrated as FILL-ONLY where mfoil is unusable, mfoil's N(x) kept everywhere it exists:
- NLF Cf fig (α=9,15): xfoil Cp/Cf/xtr overlays (dash-dot); mfoil N absent there (row 3 = marker only).
- NLF polar: xfoil diamonds at α=9,15 (xlim→0.034 to show both); mfoil bucket α≤7.
- NLF Table 4: α=9,15 mfoil cells → xfoil values (0.09/0.67, 0.00/0.69), † footnote rewritten to
  "from XFOIL; mfoil non-converged/stalled".
- Eppler polar: mfoil α=7 outlier replaced by xfoil diamond (cl=1.17,cd=0.0138); mfoil α≤5.
- Eppler Table 6: α=7 mfoil 0.33→xfoil 0.40; footnote + discussion updated. (Eppler Cf figures
  left on mfoil — it converged there.)
regen_nlf_v2.py/regen_nlf_polar.py/regen_eppler_v2.py updated with _xfoil_fill helpers.
Compiles clean, 37 pp.

## Round 7 (annotated.pdf 2026-07-04 20:38) — front matter

**[1.1] "profile-fullness indicator" → "And lambda_p?"** and **[1.2] "Nomenclature … seems
missing lambda_p, etc. … comprehensive review."** Did a usage-based sweep of the paper
(grep -oF counts) and expanded the Nomenclature to cover all heavily-used symbols that were
missing: added lambda_p (42×), a/a_max (13×), s/g_c, Re_Omega^f / Re_Omega^c (14/12×),
K_lambda (21×), sigma_t width tau, P_SA/P_AFT (10/6×), S̃ (SA modified vorticity, 8×),
nu_t (7×), c_v1/f_v1, N/N_crit, p/rho, |u| (wall-relative), H, alpha, c & x/c, x_tr/x_R
(6×), Re & M. Reordered into SA-baseline → indicators → rate-kernel → e^N → flow/geometry →
coefficients. (\kill width set to the widest label.)

**[2.1] intro opening → "add small aircraft (GA/sports/electric rotors/drones); transition
matters beyond drag."** Added a sentence: "…not only a transport-aircraft concern: at the
low-to-moderate chord Reynolds numbers of general-aviation and sport aircraft and of
small—increasingly electric—rotors and drones, the laminar run is a large fraction of the
chord, and where it ends drives efficiency, range and endurance, and acoustic signature,
not drag alone."

**[2.2] "occupies" → "can occupy".** Done ("The slow TS-envelope stage can occupy most of
the streamwise distance…").

Compiles clean, 38 pp.

## Round 8 (Drela/Spalart-anticipation discussion — user directives)

**Shape-factor dependence (Drela's top concern) — NEW Fig. 3 + §III.A paragraph.**
Computed the model's implied envelope slope dN/dRe_theta(H): for each Falkner-Skan
profile (shape factor H) evaluate a=a_max*S(Gamma) at the theta-scale band, convert
via FS similarity geometry [dN/dRe_theta = 2 a f''(eta_pk)/((m+1) I_theta)], normalized
ONCE to Drela-Giles at Blasius (the same anchor as the (s,g_c) calibration). Result:
tracks the Drela-Giles dn/dRe_theta(H) correlation across H~2.2-3.3 (favorable->mild
adverse) from the single Blasius anchor; overshoots near separation (H>3.5) as
Gamma_band->1 saturates the sigmoid to the free-shear ceiling a_max (the KH regime it
was set for). Script regen_shapefactor.py. Revised the "intermediate fullness not
calibrated" sentence to point to this a posteriori check.

**chi_inf (Spalart's freestream-seed concern).** Added a sentence (§III.D): the seed is
well posed as an inflow condition because production vanishes with vorticity in the
irrotational outer flow and destruction is gated off in the laminar range (sigma_t=0 for
chi<=1), so chi_inf convects essentially undisturbed inflow->LE (unlike a decaying
freestream closure).

**Analytic-Jacobian claims REMOVED (per user: continuous, not analytic).** Lines ~177
("analytically differentiable source"->"continuous source"), ~379 ("carries an analytic
Jacobian"->"adds no nu-tilde-dependence to the source linearization"), Conclusion
("analytically differentiable"->"continuous; the max blend is continuous but not
everywhere differentiable").

**"No fitted constant" claims REMOVED (ambiguous "fitted").** ->"constants are calibrated
on simple, canonical flows rather than the transition test cases" (intro); removed
"(not fit here)" (§IV) and "with no further constant"->"with the same constants" (Concl).

**Scope & limitations consolidated (F).** Made the intro one-liner a brief Scope-and-
limitations paragraph (natural TS / 2D / low-speed; bypass, crossflow, compressible out
of scope; M=0.1 validation). Removed the duplicate scope sentences in §IV and trimmed the
Conclusion future-work list to point back to it.

**1/12 (c_nu,aft):** user satisfied; no change.

**Eppler bubble-length vs Re (B) — IN PROGRESS.** Recommended alpha=5 (clean, well-
characterized bubble); launched cav L1 forks at Re=1e5 (muRef=1e-6) and 3e5
(muRef=3.333e-7) on GPU 2/3 (Re=Mach/muRef; +CfVec). TODO: compute upper-surface
reattachment/bubble length at 1e5/2e5(existing)/3e5 and report the Re-trend.

Compiles clean, 39 pp.
