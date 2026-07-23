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

## 2026-07-05 — Full repo + code review pass (from discussion, not PDF annotations)

Complete read of the paper, all repo .md, all paper/*.py generators, the Flow360
C++ SA-AI implementation + case JSONs, all flow360/*.py, the JAX src/ + tests/,
and scripts/. Full findings in `manuscript-review-suggestions.md` (new file,
with status per item). "Obvious" fixes applied this round:

1. **Bypass-scope claim corrected** (§IV): Mack seed exceeds c_v1 at Tu≳3%
   (T3A), not 1%; at Tu=1% the map gives N_crit≈2.6. (Old claim was off 3× in Tu.)
2. **Blasius Re_Ω peak location corrected** (§II.A): "located at y≈θ" →
   "at a fixed multiple of θ (y≈4.4θ≈0.6δ99, near mid-layer)"; verified by
   direct Blasius solve (max η²f''=1.453 at η=2.95; max Re_Ω=2.19 Re_θ).
3. **Eppler grid-size sentence fixed** (§VI): "3.8e5 nodes (cav L2)" was the
   element count → "1.5e4 nodes at L0 to 2.6e5 at L2" (matches Table 6).
4. **SA safeguards re-cited**: added `allmaras_2012` (ICCFD7-1902) to
   references.bib; the negative-ν̃ safeguards + non-negative-S̃ modification
   sentences (Eq. 1 preamble and §III.E) now cite it instead of the 1992 paper
   (the implementation uses the 2012 rational S̃ branch (c2,c3)=(0.7,0.9) and
   SA-neg f_n, not the 1992 forms).
5. **"Signed Cf" scoped to the Eppler**: the NLF figures plot |Cf|
   (regen_nlf_v2 has no sign reconstruction); the five-row layout paragraph now
   says signed for Eppler / unsigned for the attached NLF; NLF caption updated.
6. **τ-selection wording fixed** (§III.C): f_v1 reaches HALF saturation at
   c_v1 (σ_t=0.78 there), not "fully on" (was self-contradictory with §III/IV).
7. **Nomenclature**: c_ν,ai scales the molecular part of the (whole) SA
   diffusion term, not "cross-stream only".
8. **Flow360 descriptor corrected** (§IV): "finite-element" → "node-centered,
   second-order finite-volume" (per Flexcompute's own solver description).
9. **AGS number unified**: §IV now "within ~8% across the range" (−0.2% at
   Tu=0.04%, +7.7% at 0.60%), matching abstract/§III/conclusion.
10. **Duplicate χ∞ anchor sentence deleted** (§V had the N_crit=9 seed twice).
11. **Two implementation clauses added** (§II.A): solver-frame velocity
    coincides with wall-relative for the stationary cases; |u|² floored at a
    negligible value at stagnation (matches SpalartAllmaras.h).
12. **references.bib**: de-duplicated drela_xfoil_1989; wells_1967 volume
    3→5 (AIAA J. Vol. 5, 1967, verified via doi 10.2514/3.3931).
13. **fig:nuhat made self-verifying**: contour levels extended 11→14 in
    scripts/models/run_blasius_transport.py and figure regenerated (field max
    N=14.97 at the Re_x=4e6 outflow, so the caption's "N≈14" now has a visible
    contour). paper/figs copy updated.

Deliberately NOT addressed (author decision): the convergence/pseudo-time
procedure (fSlow, staged protocol, x_tr-stability criterion) stays out of the
paper — it is numerics, not model, and the paper is deliberately light on
numerics. See manuscript-review-suggestions.md items 14–22 for the remaining
discussion items (turbulent-SA baseline, cost sentence, landscape figure, NLF
exp-Cp overlays, seed sensitivity, Table 4 header/criterion wording, data
availability).

Compiles clean, 42 pp, 0 undefined refs/citations.

## Round 9 (annotated.pdf 2026-07-05 15:53 — 15 comments) + user decisions on the review-pass items

**[1] p1 Abstract, highlight "at Re=2×10⁵".**
> Low 60k to 460k. Also mention the wide range of AoA for both airfoils, and
> variety of mesh resolutions and types. Mention these very succinctly. Don't
> lengthen the abstract much.

*Addressed:* Eppler now "at Re=6×10⁴–4.6×10⁵"; appended "—across wide incidence
ranges, on structured and unstructured meshes at three refinement levels—"
before "with the same constants throughout" (+1 line total).

**[2] p1 Abstract, highlight "indicator—and".** *"and also lambda_p?"*
*Addressed:* rate now depends on "a vorticity Reynolds number, a
profile-fullness indicator, and a pressure-gradient parameter".

**[3] p2 Intro, highlight "not drag alone".** *"remove."* → removed.

**[4] p2 Intro, highlight "determines".** *"determine"* → "can occupy … and
largely determine" (verb agreement with "can").

**[5] p36 §VI.A opener.** *"To assess whether"* → opener rewritten "To assess
whether the same constants follow the bubble…" (also removes the
"benchmark fixes the model" calibration framing; see [12]).

**[6] p36, highlight "ithout retuning".** *"remove"* → removed (and the second
instance in the §VI λ_p paragraph reworded to "transfers unchanged").

**[7] p36, highlight "hand-read".** *"remove"* → removed.

**[8] p36, highlight "nearest-5° column".** *"column with nearest angle of
attack"* → reworded here and in the two Re-sweep figure captions and the
fig:eppcflow caption.

**[9] p36, highlight "Cd=0.040,0.021,0.011,0.0097 (unstructured L1)".**
> Let's add the str, unstr, mfoil/xfoil, and experimental CL, CD, CM as a table.

*Addressed:* new Table~\ref{tab:eppresweep} (generator
`regen_resweep_table.py`): c_l/c_d/c_m(c/4) × {str L1, cav L1, e⁹, Exp} × 5 Re.
CFD c_m transferred from the solver's LE moment reference to c/4 via
c_m = c_m,LE + (c_l cosα + c_d sinα)/4 (verified: str L1 200k → −0.0785 vs exp
−0.0785…−0.0809). mfoil re-run fresh for c_m (its cached sweep pkl lacked it;
fresh c_l/c_d match the pkl to 1e-3/1e-4); XFOIL Cm parsed from the OPER echo
at 460k. EXPERIMENTAL values hand-read from the TM-4062 Table B1 scan
(rendered pages 91–98 of the PDF; runs 27,28 / 15,16 / 9,10,13 / 3,4,5 / 20)
at the tabulated α nearest 5° (4.99°–5.06°). The old inline Cd list now defers
to the table.

**[10] p36, highlight "The two independently generated L1 meshes".**
> This sounds like the mesh is regenerated. We used the same mesh as we did in
> the alpha studies?

*Addressed:* YES — verified by md5: sweep_Re*k_a5/mesh.cgns is byte-identical
to cavL1prop_eppler387_Re200k_a5/mesh.cgns (and sweep_str ≡ strL1prop). Text
now says "reusing the L1 grids of both mesh families from the Re=2×10⁵ study
unchanged (only the reference viscosity changes)" and "The two mesh
families—on the same L1 grids as the Re=2×10⁵ study—…".

**[11] p36, highlight "—the expected behavior once the bubble fails to close…".**
> Instead of saying "expected", let's just point out that in the experiment
> different runs at the same AoA at Re 60k produced vastly different results --
> quote quantitative numbers from experiment.

*Addressed:* replaced with the Table-B1 repeat-run numbers at Re=6×10⁴ (RUNS
27,28): three consecutive points at α=4.00° give c_l=0.643/0.697/0.721
(c_d=.0431/.0386/.0400), and the measured lift collapses 0.838 (α=4.99°) →
0.639 (α=5.51°) — scatter of the same order as the cav-vs-str spread.

**[12] p36, highlight "calibration".**
> I wouldn't call the Re 200k runs "calibration". Rather they are validation --
> no constant are fitted from them. Go through the entire paper to make our
> claim self-consistent (without emphasizing): the constants are determined via
> simple flows (including parabolized march on flat plate with analytical
> Blasius profiles). After that everything are run as is.

*Addressed:* §VI.A "the Re=2×10⁵ benchmark fixes the model / the same
calibration / the single Re=2×10⁵ calibration left untouched" → all reworded to
"the same constants / the constants of Sec. III.E untouched". Full-paper sweep:
the intro roadmap claimed §V is used "to SET K_λ" — inconsistent with §III's
derivation; now "to test the favorable-gradient onset delay—the cliff with the
derived slope K_λ—against the e^N rooftop ignition". Remaining "calibration"
instances all refer to the §III canonical-flow anchoring (free-shear, Blasius
envelope, Drela, Mack) and were left.

**[13] p37 figures.** *"Remove figure suptitle (with latex). Same for the next
figure."* → suptitle removed from `regen_epp_L1compare.py`; both
eppler_L1compare_{lowRe,highRe}.pdf regenerated.

**[14] p39 Conclusion, highlight "Because the amplification rate…max…".**
*"Remove."* → sentence deleted.

**[15] p39 Conclusion, highlight "another twenty-four cases on".** *"Do mention
the 5 different Reynolds numbers."* → added "; an α=5° sweep across five
Reynolds numbers, Re=6×10⁴ to 4.6×10⁵, follows the bubble's march and
shortening."

### Same round — user decisions on the review-pass discussion items

- **Convergence/pseudo-time**: stays out of the paper (deliberately light on
  numerics). Added the ONE robustness sentence to the §IV solver preamble: the
  steady states were verified independent of the pseudo-time path (CFL,
  coupling frequency, handover width, laminar-ν̃ relaxation change only the
  route, not the converged solution).
- **Table 4 onset defined**: caption now states onset = first station where the
  near-wall max χ crosses unity; column header "mfoil" → "e⁹" with
  mfoil/XFOIL caption note.
- **Fully-turbulent SA baseline added to the polars only**: new cases
  strL2prop_{nlf0416_Re4M,eppler387_Re200k}_turb_a* (AI_SA=0, χ∞=3, same
  structured L2 grids, 60k steps; driver flow360/run_turb_baselines.py);
  grey dash-dot curve in both polar figures + one quantifying sentence in each
  polar discussion (NLF: Cd≈0.010–0.012, +60–70% over the bucket; E387:
  Cd=0.014–0.020, +40–60%).
- **Cost wording**: footnote "at essentially no added cost" → "with no
  equation added".
- **Landscape figures**: user decision — excluded (too messy).
- **NLF experimental C_p overlay**: INVESTIGATED, NOT POSSIBLE — Somers
  TP-1861 contains NO measured pressure distributions: its only tables are
  airfoil coordinates (Table I) and orifice locations (Table II), and its only
  pressure-distribution figure (Fig. 1) is the *inviscid design* distribution.
  No digitization source exists in the primary reference.
- **Seed sensitivity**: one sentence added at the end of §III.D (one e-fold of
  χ∞ = one unit of N_crit ≈ ΔRe_θ≈100 on the Blasius envelope; all airfoil
  cases at the single N_crit=9 anchor).
- **Data/code paths fixed**: all 186 scripts (flow360/, paper/, scripts/,
  tests/) re-pointed from the deleted /home/qiqi/flexcompute/aft-sa to sa-ai;
  build_paper.sh rewritten (was regenerating figures no longer in the paper and
  calling a missing script; now regenerates the six self-contained model
  figures + compiles). New flow360/README.md documents the env-var
  reproduction requirements (AI_SA gate, fSlow-compensated seed) and the
  paper-final case map.
- **Repo hygiene**: JAX 2D SA-AI positional-arg bug fixed (sa_sources.py now
  forwards by keyword; SolverConfig/PhysicsParams/schema/batch renamed to
  aft_re_omega_floor/aft_tilt_slope with committed-kernel defaults; 39 affected
  tests pass); legacy-kernel scripts (plot_falkner_skan_gamma.py,
  check_inner_layer.py) marked LEGACY with warnings; stale comments in
  ModelConstants.h (nuLamScale "0.25") and SAAiTransition.h ("convex blend
  (default)", "a_max=0.25") corrected — comment-only, no rebuild required.

## 2026-07-06 — Elements-of-Style compactness pass (user request)

Full-paper tightening, iterated over five passes until two consecutive passes
found nothing further. ~45 edits; 43 pp → 42 pp; no content, numbers, or
author-mandated sentences (design-intent, matter-of-fact statements, per-round
resolutions) removed. Two categories:

**Long-range de-duplication** (the main gains):
- The onset-delay-not-rate-suppression point was stated THREE times within one
  page of §II (the "Second—" consequence, a standalone "Favorable-gradient
  stabilization is therefore…" sentence, and the cliff-paragraph opener); the
  standalone sentence is deleted and the cliff opener compressed — the design
  decision is now stated once in the consequences list and defended once at the
  cliff equation.
- §II.C no longer duplicates §III.C: the N≈14→2 collapse ladder and the
  turbulent-calibration-intactness argument each now appear only in §III.C,
  with §II.C carrying one-line forward pointers.
- §III opener merged its twice-stated "not tuned to a transition case" into
  one sentence; §III.E's list intro dropped the third statement of it.
- §V's giant χ∞-anchor parenthetical (re-deriving the half-saturation anchor
  and re-stating f_v1) now defers to §III.C ("the N_crit=9 anchor of
  Sec. III.C") — 6 lines.
- §III.D no longer states N_crit=ln(c_v1/χ∞) twice in one paragraph; the
  fig:nuhat caption dropped its duplicated ~8%-AGS sentence and the "committed"
  repo jargon.
- The mfoil α=9/15 failure is now described once in §V (with the XFOIL
  substitution) and referenced ("as noted above") from the polar paragraph;
  Table 4's dagger note trimmed to one clause.
- ΔCd≈6×10⁻⁴ mesh-family agreement stated once (robustness paragraph), removed
  from the polar paragraph; "near-wall sublayer suppression" stated once (§II
  consequence #1), removed from §III.A's floor paragraph.
- §VI's mesh paragraph no longer re-lists the NLF recipe it declares it
  "reuses verbatim" (h_TE halving + r_TE decay live in §V only).
- Intro roadmap: dropped the K_λ parenthetical (duplicating §III.B), "with no
  further constant" (kept in §III.D), "of increasing complexity" (removed from
  the conclusion in an earlier round), and the duplicated χ=1/c_v1 body text
  that repeated the figure caption's panel description.
- §IV: removed the circular seed identity (χ∞ = c_v1 e^{−N_crit} with
  N_crit = ln(c_v1/χ∞)), the Tu-raises-seed sentence duplicating §III.D, the
  figure-plumbing sentences duplicating the caption, and the bypass-mechanism
  sentence duplicating the intro scope note.

**Local tightening** (samples): "quiet low-disturbance" → "low-disturbance";
"share a common premise" → "share a premise"; f_v1 cubic-decay stated once
instead of three ways; "for a simple reason—…—and" → "because …;"; "It is
positive" → "λ_p is positive" (antecedent had drifted); "and S the logistic
function" (shown in the equation) cut; K_λ adoption sentence reduced to the
author's Round-3 directive form ("We adopt the round value K_λ=10; a more
careful calibration is left to future work"); "Eppler 387 low-Reynolds-number
airfoil" (echoing the section title) → "Eppler 387"; assorted "then/
subsequently/precisely/at all/in any case/naturally/simply/essentially(×1)"
deletions where they carried nothing.

Compiles clean, 42 pp, 0 undefined refs/citations.

---

## 2026-07-06 — Elements-of-Style pass, Round 2 ("Do it again, making sure you don't find anything")

**User instruction (verbatim):** "Do it again, making sure you don't find anything"

**Outcome:** The re-verification pass DID find items — the previous round's
convergence claim was premature. ~30 further reductions applied across a full
line-by-line pass; a subsequent full verification pass found zero new
reducible text. 42 pp → 41 pp, ~12.7k words. Compiles clean, 0 undefined
refs/citations. One incidental correctness fix: the N_crit=9 anchor citation
pointed at Sec. sec:calib_diff (diffusion) instead of sec:calib_tu
(receptivity).

**Long-range dedups removed this round:**
- Mack constants (−8.43, 2.4) stated in Eq. (12) AND restated inline in the
  following sentence → constants now appear once (plus the §III.E table).
- σ_t gloss sentence ("zero for χ≤1 (laminar)…") restating the cases display
  of Eq. (7); equation now ends the sentence (punctuation moved inside cases).
- σ_t formula re-typed in §III.C τ paragraph → "the width τ in σ_t (Eq. 7)".
- \cite{spalart_allmaras_1992} cited twice around Eq. (1) ("in the standard
  form of Ref. [1]" dropped; "all as defined there" carries it).
- §II.A low-shoulder sentence duplicating the author's design-intent sentence
  that follows it.
- Michalke 0.19 ratio restated after both numbers given → "against a peak
  vorticity ω_peak = U_0/δ".
- aω→N integration link stated in §II and re-derived in §III.A → §III.A now
  references Sec. II ("Because aω integrates along the layer to N (Sec. 2)…").
- Table 2 announced twice (end of C_f paragraph + its own paragraph) → single
  introduction; antecedent repaired ("the computed onset locations").
- mfoil-failure story: NLF body stated it twice bracketed around the evidence
  ("This is specific to mfoil… the fault lies in mfoil's") → single claim;
  setup-identity list compressed to "identical conditions". Table 2 footnote
  cut to "mfoil fails at these incidences"; Fig. 14 caption "which converges
  at both" dropped (implied).
- Eppler α=7° mfoil-stall detail (C_l=0.928 vs 1.17) in BOTH Fig. 18 caption
  and Table 6 footnote → numbers live in the footnote only.
- "same L1 grids as the Re=2e5 study" stated 3× (sweep intro, mesh-family
  paragraph, table caption) → dropped from the middle site.
- "measured pressures track the computed distributions" repeated at ends of
  two consecutive sweep paragraphs → second occurrence cut.
- "natural-transition range" twice in flat-plate §IV → second cut.
- Receptivity-is-Mack stated twice in §IV body (once as a bare "adopted"
  sentence) → bare sentence cut; AGS "transition onset … transition onset"
  → "zero-pressure-gradient onset correlation".

**Local tightening (samples):** "not chosen a priori" (redundant with "fixed
by this solution"); "uniform, gradient-independent offset—a rigid downshift"
triple → "rigid, gradient-independent downshift"; "answers this directly" →
"answers this"; "rather than … directly"; "25,600 cells" (= 320×80);
"(1980)" beside the numeric cite; "with no structured near-wall layer at
all" (implied by "everywhere, the boundary layer included"); "zero-pressure-
gradient flat plate" echoing the section title; duplicated "L0–L2" in the
NLF ladder parenthetical; C_f<0 "where the boundary layer is separated"
(circular).

**Convergence status:** find-pass exhausted (final full sweep: zero new
items). Any further shortening would start cutting content, not redundancy.

---

## 2026-07-07 — Elements-of-Style pass on the tau_D revision (Round 3)

**User instruction (verbatim):** "Now do another round of 'elements of style'
revision, repeat until you can't find anything. And focus on global (long
range) repetitions and redundancies in the last few sweeps."

**Outcome:** 10 reductions applied, 9 of them in the tau_D material added
2026-07-06 (written for correctness, not economy); the pre-existing hardened
text yielded one item. 42 pp -> 41 pp. Compiles clean, 0 undefined refs.

**Pass structure:** Pass 1 full line-by-line read (7 finds); Pass 2 systematic
long-range audit — grep-indexed every repeated concept (log law x13, tau_D x14,
canonical x8, marginal x4, handover x18, constant-stress x4, exactly x9,
twenty-four x5, 8.76 x5) and read each site in context (1 find); Pass 3
verification of edited neighborhoods (2 finds, both echoes introduced by
Pass-1 edits); Pass 4 re-read of §IV-§VI and conclusion (0 finds).

**Long-range dedups:**
- "laminar range untouched, both gates identically zero for chi<=1" stated in
  BOTH §II.B and §III.C -> kept at the definition site (§II.B) only.
- "opens/opening faster than the production gate" in §II.B AND §III.C ->
  comparative object dropped at both ("opens faster (tau_D<tau)").
- "constant-stress wall-layer solve" named twice in §III.C for the same
  apparatus -> second mention dropped entirely ("fixes tau_D=1.36").
- §II.B forward pointer restructured ("Sec. III.C fixes tau_D by requiring the
  fully turbulent log law to match standard SA exactly") removing the zeugma
  ("...is what makes X, and fixes tau_D") and the trailing "(log law,
  freestream decay)" parenthetical that repeated the log-law mention.
- Fig. 4 caption: appositive "the residual tau_D is calibrated to cancel"
  (repeats §III.C) dropped; panel labels harmonized to "the calibrated tau_D".
- "at every resolution" twice in the y+ paragraph -> second becomes
  "throughout".
- AGS sentence: "the same convention by which the AGS markers are placed" ->
  "the convention that places the AGS markers"; duplicate "in Re_theta" after
  "converted to Re_theta" dropped.
- §V.A: "climbs toward unity but then stalls and dips" -> "stalls and dips"
  (the alpha=0 trace peaks at chi=0.41 — "toward unity" was inaccurate for the
  regenerated data).

**Audited and deliberately kept:** the log-law anchor's 13 sites (single full
treatment in §III.C; definition, table, caption, and summary sites elsewhere);
"canonical limit" as the paper's thesis echo (abstract/intro/§III
preamble/§III.E/conclusion); the conclusion's three-case marginal enumeration
(the accuracy payload); caption self-containment recaps.

**Convergence status:** Pass 4 found nothing; all Pass-3 finds were artifacts
of Pass-1 edits, not residual bloat.

---

## 2026-07-07 — Taoist touches (user-selected: B1, B2, B3, M1, E1)

**User instruction:** add a Taoist touch, mainly beginning and end, sparing in
the middle; from the suggestion menu the user chose B1+B2+B3+M1+E1, with
Chinese characters and glosses in ALL five (including M1 and E1).

Applied:
- B1: opening epigraph after the abstract — Daodejing 11
  (埏埴以為器，當其無，有器之用) with translation.
- B2: intro, "a coincidence we exploit" -> "a useful emptiness
  (無之以為用, 'usefulness comes from what is not there'; Daodejing 11) we
  exploit".
- B3: SA-AI name footnote extended — "autogenously---of itself, ziran
  (自然, 'self-so')---".
- M1: footnote on the §III calibration preamble — Zhuangzi's cook
  (庖丁解牛), blade through the joints the ox already has.
- E1: closing epigraph after the conclusion — Daodejing 40
  (天下萬物生於有，有生於無) with translation.

Build: \usepackage{CJKutf8} + bsmi (traditional); cjk/arphic installed to the
user tree via tlmgr --usermode from the TL2021 historic repo (Debian TeX Live
lacks them; journal/Overleaf builds carry them natively). 42 pp, 0 errors,
0 undefined refs; epigraph pages and both footnotes visually verified.

---

## 2026-07-07 — Daodejing 48 footnote (user-proposed, Option A)

**User instruction:** add 為學日益，為道日損 ("much of the motivation of this
paper, and echoing... Spalart, a very much Taoist work"); chose Option A from
the placement menu.

Applied as a footnote on the intro's contribution sentence (the Table 1
"otherwise empty cell" sentence): full ch. 48 line incl. 損之又損，以至於無為
with translation, reading Table 1's "extra equations" column as the field's
daily gains and SA-1992 (one equation by judgment in a two-equation era) as
the other road this paper walks one step further. Renders on the same page as
Table 1, stacked under the Yin/Yang name footnote. 42 pp, 0 errors.

---

## 2026-07-07 — Round 10 (annotated.pdf of Jul 7 12:08, 2 comments)

**[1] p.10, highlight on "Re_Omega^f set at 100":**
> "Again this number isn't carefully tuned. Future work may make it better."

Resolution: added to the cliff-floor paragraph (Sec. III.A), after the
diffusion-offset justification: "The round value is not carefully tuned;
future work may place the floor better." — the same one-line treatment its
sibling constants already carry (K_lambda: "a more careful calibration is
left to future work"; c_nu,ai: "more careful tuning is left to future work"),
phrased distinctly to avoid a third verbatim repetition.

**[2] p.11, highlight on "Shape-factor" (Fig. 3 caption):**
> "It's not clear how this is computed, the set of chain calculations
> required isn't that obvious. So it may be useful, probably as a footnote,
> to lay out the set of reproducible steps to creat this figure."

Resolution: footnote added on the Sec. III.A sentence that introduces the
conversion ("...normalized once at the Blasius point"), laying out the full
reproducible chain, taken from regen_shapefactor.py: FS similarity ODE
(f''' + (m+1)/2 ff'' + m(1-f'^2) = 0, m = beta/(2-beta)); I_theta and H from
the profile; indicators in similarity variables (Re_Omega ~ eta^2 f'',
Gamma = 2r^2/(1+r^2), r = eta f''/f'); band-mean Gamma over
Re_Omega > 1/2 max Re_Omega; post-onset rate a = a_max S(s(Gamma-g_c)) with
the cliff barrier off; dN/dRe_theta = 2 a f''(eta_peak)/[(m+1) I_theta];
single-constant normalization to Drela-Giles at Blasius (H=2.59).

Build: 42 pp, 0 errors, 0 undefined refs.

### Round 10, comment [2] — revision after user feedback

**User feedback:** "The first part of the footnote is too obvious -- if people
know about the FS boundary layer they know the ODE that solves it. But the
later part: what's the chain rule that evaluates dN/dRe_theta to the formula
and why? Why do one need to scale the curve by a constant? What is that
constant? Why can't you derive first principle without scaling by a constant?"

Resolution: footnote rewritten. FS ODE dropped (textbook). Now derives the
chain rule explicitly: dN/dx = a*omega_peak/U_e (temporal rate at the
Re_Omega peak convected at edge speed); FS similarity omega_peak =
U_e f''(eta_pk) sqrt(U_e/nu x), Re_theta = I_theta sqrt(U_e x/nu) ->
dRe_theta/dx = (m+1)/2 I_theta sqrt(U_e/nu x); Reynolds numbers cancel in the
ratio -> dN/dRe_theta = 2 a f''(eta_pk)/[(m+1) I_theta], function of H alone.
Normalization explained honestly: the closed form is a single-point proxy for
the 2-D transport of Fig. 6 — convects at U_e not local u, collapses the band
onto one Gamma (sigmoid exponentially sensitive), omits cross-stream
diffusion — suppressing the absolute scale by a common factor C ~= 40 at
Blasius (value from regen_shapefactor.py). Absolute slope is owned by the
coupled transport via the (s,g_c) anchor; a first-principles absolute value
would require solving that PDE per profile. Dividing out one constant leaves
the proxy testing only the H-dependence (never calibrated); the tracking of
Drela-Giles then shows the omitted factors are nearly H-independent.

### Round 10, comment [2] — second revision: the comparison made absolute

**User feedback:** "Are there ways to make this comparison against Drela's
dN/dRe_theta(H) more rigorous? At least reproduce the H_Blasius value without
fudge factors? If there's no way, then we have to admit that this is a very
rough comparison, since the viscous diffusion, interaction across the layer,
etc. would be different for different H."

Resolution: there is a rigorous way, and it is now Fig. 3. The single-point
similarity proxy (C~40 normalization) is replaced by the paper's own Eq.-11
disturbance transport marched on each Falkner-Skan field
(regen_shapefactor_transport.py; same numerics as the Fig.-6 Blasius solve:
implicit x-march, nu/12 diffusion, seed 1, sign-aware upwinding; kernel at
lambda_p=0 so the cliff sits at its floor). Envelope slope = secant of N over
the transition-relevant window N in [3,9] (N(Re_theta) is convex because the
diffusion-to-growth ratio decays like Re_x^{-1/2}, so a window convention is
required); domains sized to N_end ~ 13; slopes grid-converged <2% (400x300 vs
800x600).

Result, ABSOLUTE (no normalization): zero/adverse branch H=2.59-3.64 follows
Drela-Giles within -17%/+35% across its order-of-magnitude rise (ratios:
1.35 Blasius, 1.18, 1.04, 0.93, 0.86, 0.83, 0.88, 1.04 at the separation
end) — per-H viscous diffusion and layer-wide interaction included, which is
exactly what the proxy could not claim. Favorable branch: the computed rate
deliberately does NOT drop with the correlation (flat at ~0.013-0.015 vs
Drela 0.005-0.008) — favorable stabilization enters through the onset-delay
cliff (Sec. III.B), not the rate; shown as open symbols and stated in text
and caption. The old proxy's apparent favorable-side agreement was an
artifact of the band-mean-Gamma construction. Blasius consistency: [3,9]
secant 0.0140 vs Drela 0.0104 (+35%); whole-plate mean 14/1330 = 0.0105
matches the correlation almost exactly.

Sec. III.A paragraph, footnote, and Fig. 3 caption rewritten around the
absolute comparison; build_paper.sh now calls regen_shapefactor_transport.py.
The C~40 footnote is gone.

---

## 2026-07-07 — Fig. 5 equal-width purge + Fig. 3 two-measure report (user-directed)

**User instruction (1):** "remove the old tau_D=4 result from Figure 5 (and
everywhere). Previously tau_D was pretty randomly chosen. Now it's
calibrated."

Resolution: regen_wall_layer.py solves only SA and SA-AI(tau_D=1.36); the
dotted equal-width curves are gone from panels (b)/(c) (panel-c axis now
+-0.05, showing the near-zero shift); dB=-0.33, 4.6%, and "equal-width
reference" removed from Sec. III.C and the Fig. 5 caption. The calibration
argument survives via the analytic residual (1-sigma_t)(1+c_b2) kappa^2
u_tau^2/sigma; the calibrated results quoted are |dB|<1e-3 and 0.8% buffer
perturbation.

**User instruction (2):** report two numbers per H in Fig. 3 — the average
slope from N=0 (starting at Re_theta^c) to N=9, and the late/asymptotic
rate; asked for a recommendation on the latter.

Resolution: tested the principled asymptote (local slope vs 1/Re_theta
extrapolation, motivated by the 1/Re_theta diffusion deficit): rejected —
R^2 only 0.81-0.87 and no true plateau exists in the transition-relevant
decades (the cliff progressively unlocks the near-wall high-Gamma band, so
the rate keeps drifting). Adopted the user's [5,9] secant as the
late-envelope rate. Fig. 3 now shows, absolutely: onset-to-transition mean
(squares; Re_theta^c = floor / profile's peak Re_Omega/Re_theta) and late
rate (circles), band shaded. Blasius: mean 9.4e-3 vs Drela 1.04e-2 (-9%);
late 1.48e-2 (+43%). Zero/adverse branch: late rate tracks Drela within
-15/+43% (within +-10% mid-branch: ratios 1.43, 1.25, 1.09, 0.98, 0.91,
0.85, 0.92, 1.09), mean runs 0.6-0.9x; the correlation runs along the band.
Favorable branch: both measures flat (open symbols; cliff carries the
stabilization). beta=1.0 overflow guarded in the domain sizing. Sec. III.A
text, footnote, caption rewritten.

---

## 2026-07-07 — Sections II/III narrative restructure + fidelity admission (user-directed)

**User instruction:** admit that future work may increase the fidelity of the
Fig.-3 comparison (sigmoid form untuned, alternatives uninvestigated);
restructure Secs. II/III from "a bucket of information" into intent (II) then
calibration (III), with the narrative laid out partially at the start of II
and completed at the start of III; add a Blasius envelope-vs-Drela panel to
the transport figure; write for a fluid-dynamicist reader, minimizing mental
load. Asked "am I missing anything?" in the intent chain
(a_max/sigmoid/dissipation/onset/handover).

Answer to "missing anything": (i) the freestream seed/receptivity (where the
count begins) — added to the roadmap as the sixth, empirical closure,
deferred to Sec. III.D; (ii) two further handover intents — protecting the
sub-O(1) seed from SA's wall destruction, and the max-blend's
pressure-decoupling (no residue past crossover) — folded into the handover
item.

Changes:
- Sec. II preamble: five-decision intent roadmap (ceiling=free shear;
  sigmoid=Blasius TS rate; reduced diffusion=survival + undistorted Re_theta
  dependence; onset threshold=floor + FPG cliff; handover=smooth, seed
  protection, no pressure residue, log law preserved) + seed pointer.
- Sec. II.C: added the 1/Re_theta drain-scaling intent sentence;
  determination pointer -> sec:calib_rate.
- Sec. III preamble: fixing-order narrative; explicitly admits (s,g_c),
  c_nu_ai, and the floor shape ONE observable and are tuned together against
  the Blasius envelope; handover framed as an invariance requirement.
- III.A retitled "The rate ceiling and the attached-layer calibration";
  reordered: a_max -> coupling frame -> sigmoid mechanics -> floor -> joint
  Blasius fit (Eq. transport + c_nu_ai determination moved here from old
  III.C) -> two-panel Blasius figure -> FS absolute test + fidelity
  admission ("logistic form adopted for smoothness, not derived; neither it
  nor alternative kernel shapes investigated; future work on the kernel's
  form may increase the fidelity"). Conclusion future-work sentence extended
  to match.
- Blasius figure (now Fig. 3) gains panel (b): max_y nuhat vs Re_theta (log
  left / N linear right) against Drela-Giles (zero to Re_theta_crit=242,
  then slope 1.04e-2): drained below early, steeper late, N=9 within 10%.
  run_blasius_transport.py updated (two panels).
- III.C retitled "The turbulent layer and the handover"; opens with the
  invariance frame ("diffusion exactly, everywhere; gates within a thin
  buffer band, whose lone observable consequence tau_D cancels").
- Figure order is now nuhat(3), shapefactor(4), worstpoint(5), walllayer(6).
43 pp, 0 errors, 0 undefined refs.

### 2026-07-07 — reader-simulation friction pass (follow-up to the restructure)

**User instruction:** "Go through again with that reader simulation to try to
identify more friction and mental load and eliminate."

Nine fixes:
1. Sec. II roadmap reordered to match the presentation order (ceiling ->
   sigmoid -> onset threshold -> handover -> reduced diffusion); "taken up in
   turn below" is now literally true — the old order put diffusion third but
   Sec. II presents it last.
2. "The first of these, lambda_p" -> "Of the two, lambda_p" — lambda_p is
   the SECOND entry of Eq. (4); "first of these" collided with the equation.
3. "Both parts are formed to be Galilean invariant. The numerator is..." ->
   "Both descriptors..." / "The numerator of lambda_p is..." — antecedents
   made explicit.
4. "Formed in that order---Re_Omega^c first, never the reciprocal of
   lambda_p---" (dangling modifier) -> "Since lambda_p enters only through
   the exponential of Eq. (6)---nothing ever divides by it---".
5. Sec. II.B tau_D forward reference given one clause of mechanism ("equal
   widths would leave a small surplus of nuHat in the turbulent buffer
   layer, where both sources run throttled") so the reader isn't handed a
   bare log-law assertion with no shape of a reason.
6. III.A joint-frame paragraph now names the trio up front and splits the
   long em-dash clause into two sentences.
7. III preamble trio sentence slimmed to roadmap level (it duplicated the
   III.A explanation nearly verbatim — a redundancy the restructure itself
   introduced).
8. III.C opening: object-fronted relative clause ("whose lone observable
   consequence the destruction width tau_D is then chosen to cancel")
   flattened to an active construction.
9. Fig. 6 (wall-layer) caption: dropped the self-reference "(Sec. III.C)" —
   the figure sits in that section.
Sections IV-VI and conclusion re-checked with reader eyes; jargon there is
defined at first use (kernel-active band, signed C_f) — no changes. 43 pp,
0 errors, 0 undefined refs.

## Round of 2026-07-12 (annotated.pdf, 8 comments; reviewed 2026-07-13)

1. **(p5, on "pinched shut on the attached-wall locus ωd=|u|")** "The main
   goal is that it is shut close to the wall" — RESOLVED: the §II
   six-decisions summary now leads the gate with its main job ("keeps
   amplification off the wall---its main job is to be pinched shut on the
   attached-wall locus---while opening across the layer's unstable band").

2. **(p6, on Eq. indicators)** "Put Figure 1 closer to here." — RESOLVED:
   the figure environment moved to immediately after the indicators
   equation paragraph.

3. **(p6, on "purely")** "Claiming d and velocity relative to nearest wall
   as 'purely' local won't pass reviews. I'd call them something like
   'easy to work with in CFD'. Please scan the entire manuscript for
   OVER-CLAIMS and over-statements and over-emphasizes like this. We want
   to state everything as they are, don't over-claim or even
   over-emphasize things. I feel there are lots of over-emphasizes. Just
   make all the statements surgically precise." — RESOLVED: "three purely
   local descriptors" → "three local descriptors---local in the sense
   that matters in practice: each is algebraic in quantities a RANS
   solver already carries at a node"; "A clean local surrogate" → "A
   practical local surrogate"; "costs a RANS solver no new machinery" →
   "evaluating it re-uses machinery a solver already carries" (also
   O11-consistent). Full-manuscript sweep performed: remaining "purely"
   (3×) describe other models/numerics accurately; every "exactly"
   audited and each is a mathematical identity or a verified construction
   (log-law match, Q limits, cliff at floor, counting); "automatically"
   (sublayer suppression) is structural; "least obviously" is a reading
   guide, kept.

4./5. **(p6, "and Gamma_g" ×2)** — RESOLVED: "Both descriptors are formed
   to be Galilean invariant" → "All three"; "With d, ω, and ∇p already
   invariant, λ_p and Γ are frame independent" → adds ∇²u and Γ_g. The
   later duplicate ("Like the other descriptors it is frame invariant")
   removed.

6. **(p6, on "what the instability actually responds to")** "Do we have a
   citation that the instability growth actually respond to BL shape as
   opposed to pressure gradient? If so please cite here. A related
   question is whether the inception of instability respond to BL shape
   or to pressure gradient. Find citation." — ALREADY ADDRESSED (commits
   0a18228, 819f7ec): verified literature added — Mack AGARD R-709 (the
   stability problem is posed on the profile), Obremski–Morkovin–Landahl
   AGARDograph 134 (FS stability portfolio), Wazzan–Gazley–Smith AIAA J
   1981 (critical Re AND e⁹ transition Re collapse onto one H-map across
   pressure gradient, heating, and suction — covering inception
   explicitly), Serrin 1967 + Thwaites/White for the
   favorable-relaxation asymmetry that legitimizes λ_p as a shape
   surrogate.

7. **(p6, on "near-separation")** "and no matter what the boundary layer
   shape is, goes to 0 outside the BL" — RESOLVED: Γ's description now
   ends "and---whatever the layer's shape---falls to zero outside it,
   where the vorticity dies."

8. **(p6, on "without being H itself")** "while being easy to use in
   CFD." — RESOLVED: sentence now ends "without being H itself, and at
   the cost of one algebraic expression per node."

## Round of 2026-07-14 (10 comments, pages 16–17, Sec III.A/III.C)

1. **(p16, on "the system is well posed beca[use]")** "Since the valley exists, 'well posed' may not be exactly accurate. What might be a weaker phrase for it? Also, a more accurate description is that the combination of center and slope determines the upper and lower saturation point, which is decided by the separation threshold profile and Blasius growth rate respectively."
   → Replaced with "the solve converges because the conditions divide the work," and the attribution corrected: the floor sets the Blasius departure (N=1); the center g_c and slope s *together* set the sigmoid's two ends — its tail value on Blasius (pinned by the N=9 growth condition) and where it saturates up the family (pinned by the separation-limit condition).

2. **(p16, on "asymptote Γ=1")** "happens to land close to"
   → "The center g_c happens to land close to the attached asymptote Γ=1."

3. **(p16, on "secondary objective—the unfitted interior itself, or coupled-solver robustness")** "leave secondary objective open. Don't suggest anything."
   → The examples deleted; "a secondary objective could be optimized at no cost to the anchors."

4. **(p17, on "floor")** "When discussing 'floor, center, slope', readers who aren't familiar might not know what they are. Just put the symbol here whenever you mention one of the three."
   → Symbols added at the III.A determination sentence (Re_Ω^f, g_c, s) and throughout the rewritten III.C division-of-work sentence and the "center happens to land" sentence.

5. **(p17, on "Nothing remains to tune this interior away: … residual the three anchors leave")** "This emphasize seems unnecessary. Just delete?"
   → Deleted. The preceding sentence already carries the 5%-interior fact; the residual-quoted-not-absorbed point survives in the fig:nuhat caption.

6. **(p17, on "now with each wedge's own λ_p feeding the kernel (cliff and rate factor; both inert on the adverse wedge)")** "since we haven't discussed lambda_p yet, perhaps remove this sentence (I believe we will discuss this when we talk about lambda_p?)"
   → Clause removed; "The flanking rows repeat the solve on two wedges." The λ_p protocol is discussed in III.D where K_λ/K_r are determined.

7. **(p17, on "comparable late rate")** "just 'correct rate'?"
   → "then grows at the correct rate."

8. **(p17, on "A physical case seeded with χ_∞ reaches the handover where the env[elope]…")** "This discussion starting here seems out of place. Where's the best place for it? Perhaps later in the chapter?"
   → Moved to Sec III.F (receptivity), attached to the existing "fewer e-folds to the handover" sentence where N_crit and χ_∞ are the subject.

9. **(p17, on "the C_f rise begins ln c_v1 ≈ 2 e-folds earlier, at χ=1")** "Are you sure? Why? I would just avoid discussing where Cf rises if we aren't sure that it's generally so and why."
   → Clause dropped in the move (comment 8). The χ=1/C_f-rise statement survives only in Sec IV, where the flat-plate computations display it directly and the onset convention is stated.

10. **(p17, on "compresses")** "often compresses"
    → "often compresses into a short streamwise interval."

## Round of 2026-07-14b (5 comments, pages 19–20, Sec III.D)

1. **(p19, on "both anchor profiles,")** "that we use to determine the constants in Section III.C"
   → "λ_p ≤ 0 on both of the anchor profiles that determine the constants of Sec. III.C, so the conditions see neither K_λ nor K_r."

2. **(p19, on "same anchoring logic as the kernel, once one asks where on a profile growth is triggered.")** "vague statement. Perhaps remove."
   → Removed; "The onset-delay slope K_λ comes first." flows directly into the pointwise-cliff sentence.

3. **(p20, on "the worst point—the")** "'worst' is ambiguous. Rephrase (and later mentions of 'worst') as most-amplified or max-amplification point or something?"
   → Global rename. The wall-normal trigger is now "the most-amplified point" everywhere (body, both figure captions, the self-consistency sentence); the symbol λ_p^worst became λ_p^⋆ ("the star of Fig. 5", matching the star markers) in eq:klambda and the regenerated fig:klambda_sc legend/axis. The *other* sense of "worst" (the K_r fit point of the hot branch) was renamed separately to "the hot branch's largest overshoot" in all three mentions, removing the collision entirely. Whitepaper terminology matched.

4. **(p20, on "the clearance")** "give quantitative formula for 'the clearance'."
   → "colored by that clearance log₂[Re_Ω/Re_Ω^c(λ_p)]" at the Fig. 5 body mention (the caption already carried it).

5. **(p20, on "cliff is high")** "explain the 'cliff' by referring to equation number. In fact, each subsection in Section III should refer back to their respective equation numbers close to the beginning of the subsection."
   → "the cliff Re_Ω^c(λ_p) of Eq. (9) is high"; and every §III subsection now cites its governing equations in its opening sentence: III.A a_max of Eq. (5); III.B c_ν,ai of Eq. (7), c_A of Eq. (8), p of Eq. (6); III.C the floor of Eq. (9) and (g_c, s) of Eq. (6); III.D K_λ of Eq. (9) and K_r of Eq. (10); III.E the widths of Eq. (blend). III.F defines eq:tumap in place.

## Round of 2026-07-14c (2 comments + standing directive, page 23, Sec III.E)

1. **(p23, on "the one constant")** "remove this emphasis"
   → "The production width is chosen for numerics rather than physics, and it is bracketed by its failure modes."

2. **(p23, on "largely complete (σ_P=0.78")** "Just state 78% complete by the time xi reaches cv1 (fv1 half active), and another complete percentage when fv1 reaches say 90% active."
   → "τ=4 spans the natural window as a resolution compromise within that bracket: the handover begins at χ=1, where ν̃ starts to feed the eddy viscosity, is 78% complete where f_v1 is half active (χ=c_v1), and 97% complete where f_v1 reaches 90% (χ≈14.8)." (σ_P(14.77)=0.968 with τ=4.)

3. **(standing directive)** "Last scan of adjectives and adverbs seems to have missed 'largely' — can you do another scan for such. Or better, just use your LLM to go over the entire document to find such forced doings? Let the numbers speak for themselves is the wu-wei approach."
   → Full top-to-bottom read of main.tex (not just grep). 13 further removals/rewrites:
   - "largely determine where transition is observed" → "set where..." (intro)
   - "hands the production *cleanly*" → dropped (specifics follow: "no laminar residue")
   - "whether each constant *indeed* sits at a joint" → dropped (ox footnote)
   - "sensitivity splits *cleanly*" → "splits in two"
   - "interior responds *strongly*---roughly +20%/−25%" → "responds: roughly..." (quantified in the same breath)
   - "It *settles quickly*: input and output cross at 6.1" → "Input and output cross at 6.1 (Fig. 6)"
   - "Cf rise begins at χ=1 and is *essentially complete* by c_v1" → "runs between the χ=1 crossing and the χ=c_v1 crossing" (flat plate)
   - "where transition is *essentially complete*" → "the band's turbulent end" (AGS convention sentence)
   - "an *unforgiving* test" → "test" (Fig. NLF-high caption)
   - "*subtly* distorts the velocity profile" → "distorts" (L0 artifact; the Γ collapse is quantified next)
   - "noise that we *state rather than hide*" → dropped (performative honesty)
   - "*genuinely* mesh-sensitive" → "mesh-sensitive" (the bistable experiment is the evidence)
   - "converges *cleanly* there" → "converges there" (Eppler footnote)
   Kept, with reasons: restrictive only/merely ("depends only on", "digits merely solve the anchors"); hedges quantified in place ("essentially the clearance maximum", 6.10→6.29 stated; "essentially the point of largest clearance"); physical descriptors ("gentle TS waves", "mild favorable", "strongly accelerated (λ_p≳0.16)"); literature claims ("largely an artifact" — Wells's finding); user-directed phrasings ("least obvious and most convincing", "it turns out").

### Chat follow-up to round 2026-07-14c (2026-07-14)

1. "'noise that we state rather than hide' — do not say 'we don't do wrong stuff' when we are not doing wrong stuff. Scan the entire document for such '此地無銀三百兩' statements."
   → Scan (hide/honest/admit/do-not-claim/tuned-freely/never-done/...) found three survivors, all removed:
   - "anchored to the e^N critical Reynolds number *rather than tuned freely*" → clause dropped (§II, favorable-gradient paragraph)
   - "*what was never done is* adjust a constant against an airfoil observable" → stated positively: "the constants themselves were fixed against the anchors of this section alone" (§III preamble)
   - "The correspondence is not exact, *and we do not claim it is*" → "The correspondence is not exact:" (§III.F)
   Kept (scope delimitations, not denials): "a motivation, not a representation claim"; "we claim no physics for it"; "a preference, not a guarantee"; "the claim is at the level of the constants, not of the development history".
2. "'converges cleanly' — quantify or drop."
   → Already dropped in round 14c ("which converges there (C_l=1.17)"); "cleanly" carried no quantity — the contrast with mfoil's numerical stall is stated in the same footnote.
3. "'least obvious and most convincing' — any more objective way?"
   → Replaced the subjective ranking by the objective property that made the evidence binding: "The evidence that this matters comes from the unfitted interior: the adverse branch of Fig. 4(a) tracks the correlation to within 10% between the anchors, with no condition imposed there, and that agreement disappears when the gate is left open too near the wall."

## Round of 2026-07-14d (4 new comments, pages 23-24, Sec III.E; the two page-23 comments of round 14c reappeared and were already addressed)

1. **(p23, on "The destruction width is then fixed by")** "say tau_D here."
   → "The destruction width τ_D is then fixed by one more canonical anchor..."

2. **(p24, on "the one place the gates, rather than the diffusion, touch SA's calibration")** "remove"
   → Appositive removed: "Gating production and destruction by a *common* weight (σ_P=σ_D, i.e. τ_D=τ) would leave a turbulent-wall footprint."

3. **(p24, on "weight")** "what weight? use symbols to remove ambiguity." + standing directive: "scan the document for places where symbols next to wordy descriptions of constants would help."
   → "a common weight (σ_P=σ_D, i.e. τ_D=τ)"; "The weight σ_t approaches unity only for χ≳20". Document-wide scan added symbols at eight more wordy mentions: "the freestream seed χ_∞" (§II closure list and §III roadmap), "the two widths τ and τ_D" (§II blend), "the destruction width τ_D is chosen" (§II close), "the sigmoid slope s exists to settle" (III.A), "the production width τ is chosen for numerics" (III.E), "the handover width τ" (§IV convergence-protocol list), "the destruction-gate width τ_D is calibrated" (conclusion).

4. **(p24, on "disturbance branch P_AI")** "refer to equation number, and instead of saying 'extinct', quantify how small it is compared to SA production."
   → "the amplification branch P_AI of Eq. (eq:pai) is identically zero: on the wall-layer solution Re_Ω = y⁺/κ, which stays below the floor Re_Ω^f=254 out to y⁺ ≈ κRe_Ω^f ≈ 100, and the rate carries a hard cutoff, a ≡ 0 for Re_Ω ≤ Re_Ω^c (Eq. eq:sz); the band gate Q multiplies only this branch, so it too plays no part here." The honest quantification is exact zero (the cliff), not a small ratio; verified numerically on a composite wall profile (crossing at y⁺≈96; max Re_Ω=144 for y⁺<50).

## Round of 2026-07-14e (2 comments, pages 24-25)

1. **(p24, on "the band gate Q multiplies only this branch, so it too plays no part")** "unnecessary, remove"
   → Already removed (commit 1ca2869, same request made in chat): the hard cutoff a≡0 alone makes P_AI identically zero.

2. **(p25, on "The γ–Re_θ and BC/BCM models [...] instead carry the equivalent Re_θc(Tu) closure in an auxiliary equation. The flat plate then verifies rather than fits, and the adverse-gradient Eppler 387 (Sec. VI) introduces no further constant.")** "unnecessary, remove"
   → Both sentences removed from III.F; the paragraph now runs from the AGS onset-convention parenthetical directly to "Sensitivity to the seed is that of any e^N method: ...".

## Round of 2026-07-15 (3 comments, pages 26-27 of the pre-adoption build; addressed post-adoption)

1. **(p26, on "with the handover weights ... of Eq. (10) and the amplification rate ... of Eqs. (5)-(9), built from the local indicators ... of Eq. (4)")** "Would be good to just restate all these equations, after restating the original SA terms."
   → Sec VII is now self-contained: after the baseline SA closure block (eq:sa_closure), three unnumbered display groups restate the handover weights (sigma_t and the sigma_D tie), the four local indicators, and the full amplification rate (a, z, S(z), Q, cliff, f_lambda), each cross-referenced to its original equation numbers in the lead-in.

2. **(p27, on "The set has a two-tier structure. ... never feed back.")** "This sounds repetitive. Can we check if these has already been said, and if so, remove these?"
   → Confirmed repetitive: the Sec III roadmap carries the five-stage ordering and decoupling, III.D states the no-feedback property of K_lambda/K_r, III.B the anchors-re-solved bracketing, III.E the laminar decoupling. Paragraph replaced by one sentence pointing at Secs. III.A-III.F, keeping only the Sec-VII-specific claim (same set used for every computation, no per-geometry adjustment).

3. **(p27, on "The steady states reported below were verified independent of the pseudo-time path ... not the converged solution.")** "We said not discussing numerics. And some of these aren't true anyway -- can you scan the manuscript again for things that aren't certainly correct (or no longer correct after we've updated things)?"
   → Sentence removed (numerics.md is the sole numerics record). The comment's "aren't true" is confirmed: the list included the handover width tau, which is a MODEL constant -- varying it changes the converged solution, not just the route. Correctness scan re-run post-adoption: no sigma_D=0 / gates-vanish / tau_D stragglers; AGS digits fresh from the tie runs (~11%); the pre-tie "+2% rate-factor shift" kept as an approximate factor-specific claim; Eppler-sweep convergence evidence kept (it is evidence, not scheme discussion); one gap fixed -- whitepaper Sec 3.1's instrument line now carries the no-destruction/amplitude-free note matching paper III.A.

## Round of 2026-07-15b (3 comments, page 28, Sec IV flat plate)

1. **(on "...nearer its chi=1 end throughout---the basis for quoting the chi=1 convention. The eddy-viscosity rise ... is deliberately gentle...")** "Remove. This is contradictory to us setting xi_inf according to xi=cv1, and sounds like we twist definitions differently in different places however it suits us. Please scan the document to remove things like this."
   → Removed (both sentences). The receptivity map anchors N_crit at chi=c_v1 while the onset convention is chi=1; justifying the latter by AGS proximity was definitional twisting. The paragraph now stops at the two readings ("we state the convention and give both readings"). Scan: all remaining convention mentions (III.F parenthetical, Sec IV, tab:nlftrans caption, Eppler reattachment definition, conclusion, whitepaper) state their convention without fitness justification.

2. **(on "the Cf rise runs between the chi=1 crossing and the chi=c_v1 crossing")** "Are we sure about that? I wonder if we can add a right y axis on the bottom left figure plotting Cf, so that we can correlate and be sure."
   → Done: the bottom-left panel of the flat-plate figure now overlays Cf (gray, right axis) on the chi trajectories. The overlay CONFIRMS the claim: each case's Cf rise starts at its chi=1 crossing and levels off at its chi=c_v1 crossing. Text and caption updated to point at the overlay.

3. **(on the residual-acceleration caveat: "the band sensor reads lambda_p ~ +0.01 ... a ~2% aft shift ...")** "Remove. We don't know how much of this is because of the finite domain, where boundary layer causes constriction. So may be numerical issue. Not discuss."
   → Removed. Ripples: Sec V's "its one visible price is the flat plate's ~2% shift" clause removed with it; the whitepaper Fig-7 caption's residual-lambda_p sentence removed.

## Round of 2026-07-16 (6 comments, pages 35–36, Sec. IV NLF results)

**[1] Table `tab:nlftrans` caption, ‡ footnote (highlight on "‡Extrapolated
beyond the recorded orifices: the upper front at α=4° (just ahead of the
forwardmost…").**
> Obviously not upstream of x=0.3 orriface -- the experimental data just
> didn't record the transition location beyond Cl=0.68. See Figure 9(d) of
> the NLF paper. I extrapolated it.

*Addressed:* Reworded both the footnote and the companion passage in the
text. The footnote now reads "Extrapolated beyond the recorded x_tr(c_l)
data of Somers' Fig. 9d: the upper front at α=4° (orifice data recorded
only up to c_l≈0.68)…", and the text paragraph now states plainly that the
published orifice data record the upper-surface x_tr(c_l) only up to
c_l≈0.68 and the ≈0.31c datum extends that trend to the computed c_l=0.96.
Removed the incorrect "front reaches the forward edge of the instrumented
region / forwardmost orifice at x/c≈0.30" characterization entirely.

**[2] Table `tab:nlftrans` caption (highlight on "the cavity L2 grid agrees
to within 0.02c on every entry. e9: coupled viscous–inviscid panel
solvers…").**
> Don't have to say it, it's already presented in the table. Can you also
> remove other agreement statements that are too obvious from immediate
> vicinity?

*Addressed:* Deleted the cavity-agreement clause from the caption. Also
removed from the vicinity: "agreeing with the present RANS to within
~0.06c" (XFOIL sentence); the "matches the measured x_tr to within ~0.03c
at every incidence … e9 sits ~0.03–0.07c downstream" recitation (kept the
qualitative aft-ward-march + systematic-offset reading); the "(SA-AI 0.37c
on both mesh families, experiment 0.38c) … to 0.25c" recitation in the
upper-surface sentence; and the "track the e9 envelope … to within ~0.09c
… ~0.06c at L2" sentence in the Fig. `fig:nlfcflow` caption.

**[3] §IV.A paragraph "The form of the production blend (11)—a max rather
than a convex combination—matters on the NLF(1)-0416…".**
> Remove

*Addressed:* Paragraph deleted. The max-vs-convex design rationale already
lives in Sec. II at Eq. (11); the forward pointer there
("; Sec.~\ref{sec:blendnlf}") was dropped since the target no longer
exists.

**[4] §IV.A paragraph "A grid artifact on the coarsest structured mesh
exposes the streamwise-resolution demand…".**
> This discussion can be moved into a footnote on the figure

*Addressed:* Condensed the paragraph (~180 words → ~90) and moved it into
the caption of Fig. `fig:nlfcflow` (the figure whose χ row shows the L0
stall-and-dip), replacing the deleted agreement sentence there. Text
paragraph removed.

**[5] Subsection heading "A. Marginal natural transition and the production
blend" (`sec:blendnlf`).**
> THis subsection seems unnecessary.

*Addressed:* Subsection dissolved. With [3] deleted and [4] moved to the
figure caption, nothing remained; the heading and label are gone (Sec. IV
now runs without subsections through the NLF results).

**[6] Polar paragraph "The section drag summarizes the transition
prediction (Fig. 10). On the refined meshes the SA-AI polar tracks the
measured low-drag laminar bucket…".**
> The whole description recounts what's in the plot. Only point out what's
> noteworthy or easy to miss. Can you scan similar descriptions of data and
> remove / reduce?

*Addressed:* Rewrote the NLF polar paragraph to noteworthy-only: the
α=15° beyond-stall qualification, the L0 drag over-prediction, the
mfoil-vs-XFOIL Newton-globalization diagnosis, and the per-α closure of
the fully turbulent polar (the 58–78% bucket number stays only in the
figure caption). Scanned the rest of the results sections for the same
pattern and reduced: the Cf/Cp "follows the laminar branch … recovers a
turbulent level" recounting (now one grid-convergence sentence); the
Eppler reattachment paragraph (dropped the per-incidence 0.72-vs-0.74-style
recitations and the mfoil offsets, kept the L0 ±0.12c scatter, the
mechanism reading, and the α=7° miss analysis); the Eppler polar paragraph
(kept only the shared-with-e9 low-α under-prediction; mesh-bracket and
fully-turbulent numbers stay in the caption); and the Reynolds-sweep drag
paragraph (kept the 12%-at-10⁵ shared deficit and the pressure-plateau
corroboration, dropped the "tracks to within 5% / agree closely at every
Reynolds number" recitations).

## Round of 2026-07-17 (12 comments, pages 36–42, Sec. VI Eppler)

**[1] Sec-VI opener (highlight on "closes the study").**
> no longer -- we added Daedalus.

*Addressed:* Opener no longer claims to close the study.

**[2] Same sentence (highlight on "adverse-pressure-gradient regime").**
> not really -- NLF did test APG. just remove

*Addressed:* Rewrote to "The Eppler 387 tests the low-Reynolds-number,
bubble-dominated regime."

**[3] Signed-Cf definition (highlight on "the wall skin friction oriented
along the LE→TE surface tangent (equivalently Flow360's wall-shear-stress
vector output, against which we verified it node-for-node…").**
> trivial details not worth presenting in paper

*Addressed:* Deleted the orientation/verification parenthetical; kept only
that both codes show the same recirculation structure.

**[4] λp-pair sentence (highlight on "pair—the cliff … transfers unchanged
to the Eppler 387 … unchanged to roundoff with the factor on or off").**
> just say the two functions (quote equation number) are constants for
> adverse PG.

*Addressed:* Now reads "…the two λ_p functions (Eqs. (cliff) and (fpgrate))
are constants for adverse pressure gradients."

**[5] Reattachment-table discussion (highlight on "the only grid-sensitive
entry is the near-stall α=7 case").**
> Be more rigorous. Say the alpha=7 case is MORE grid sensitive than the
> other AoAs.

*Addressed:* "…is markedly more grid-sensitive than the other incidences."

**[6] Reattachment paragraph (highlight on "The reattachment—…—comes out
within two percent of chord through α=5°: the amplification onset ignites
…").**
> Again don't present a long boring recount of results reader already see
> in picture. Only point out noteworthy or easy-to-miss info, succinctly.

*Addressed:* Cut the recount; kept one sentence defining reattachment as
the quantity the test hinges on, then straight to the N-sensitivity and
α=7° analysis. Also dropped the defensive "The miss is shared by the e^N
framework itself:" lead-in (kept the factual XFOIL 0.40c comparison).

**[7] Polar paragraph (highlight on "not a defect of the present model").**
> remove

*Addressed:* Removed; the low-α drag offset is now attributed to the
under-predicted bubble length (Sec. III.C) instead.

**[8] Negative-Cf paragraph (highlight on "One feature of the low-Reynolds
Cf traces should be read as a scope limit…").**
> Make this discussion more succinct. The point is that the negative rise
> isn't completely unphysical but we don't claim modeling accuracy.

*Addressed:* Condensed ~200 words → ~80: surge qualitatively physical
(DNS refs kept), magnitude outside declared scope, no accuracy claimed;
macroscopic content is what the section tests.

**[9] Re-sweep drag deficit (highlight on "shared with the e9 reference
(0.0209 and 0.0213 vs the measured 0.0237)—again common to the
two-dimensional predictions").**
> probably partly due to our shorter bubble. Don't denigrate e^N model --
> search for other instances and delete.

*Addressed:* Re-attributed to the under-predicted bubble length
("consistent in sign with…"); swept the paper for "common to the
two-dimensional predictions" / "shared with/by the e^N" framings and
removed all of them (polar paragraph, α=7° discussion, Re-sweep).

**[10] Mesh-family sentence (highlight on "The two mesh families bracket
the same solution at the higher Reynolds numbers … diverge only at 6×10⁴").**
> Just say solution is more mesh sensitive at 6E4

*Addressed:* "The solution is markedly more mesh-sensitive at 6×10⁴ than
at the higher Reynolds numbers…"

**[11] mfoil-divergence sentence (highlight on "The mfoil e9 panel
reference converges from 6×10⁴ to 3×10⁵ but fails at 4.6×10⁵…").**
> Should be a footnote in the figure instead

*Addressed:* Deleted from the text; the divergence reason now lives in the
tab:eppresweep caption (fig:eppresweep_high already carried it).

**[12] Closer (highlight on "the full factor-of-eight Reynolds-number range
the model reproduces the bubble march with the constants … untouched").**
> Remove. The reader can see themselves

*Addressed:* Removed.


## Round of 2026-07-20 (7 comments, pages 6-8, the NEW sphere-kernel Section II)

Comments on the freshly-integrated Section II (the amplification model). All CONFIRMED; [6] extended (see below). Not yet implemented — pending the 3D-implementability / favorable-PG discussion.

**[1] p6, §II.A highlight ("...H=4.03; ...the constant-curvature, zero-wall-shear parabola (stagnation profile) is the neutral reference, and the fully inflectional free shear layer (mixing layer) is the opposite, most-unstable limit that fixes the rate ceiling (§II.F)...").**
> Let the reader "discover" this constant curvature profile from the plot, after discussing the plot, rather than forcefully introduce it here. Instead, describe the physics of each profile we aim to match here. Then integrate the current B. into this subsection too.

*Disposition:* CONFIRM. §A becomes "the profiles we must match" (physics of favorable / Blasius-TS / adverse / incipient-separation / reversed-flow-bubble / mixing-layer-as-ceiling); parabola & g=0 introduced only when the plot is read (§D/E). Merge §B (temporal-vs-spatial) into §A, keeping the "resolve WHERE it grows" correction prominent.

**[2] p6, §II.B highlight ("...rides the detached shear layer at u≈c>0...").**
> Just say about what fraction (give a cited range) of edge velocity.

*Disposition:* CONFIRM. State the critical-layer phase speed as a cited fraction of edge velocity (c_r ~ 0.3-0.4 U_e for TS; ~0.5 U_e for a free shear layer).

**[3] p6, §II.B highlight ("(u→0)").**
> u \approx 0

*Disposition:* CONFIRM (trivial). Use u≈0 (small u near the dividing streamline) rather than u→0.

**[4] p6, §II.B highlight (the paragraph "This corrects a premise of the earlier drafts, which calibrated the rate as if only its magnitude mattered...").**
> remove

*Disposition:* CONFIRM. Delete the earlier-drafts meta-paragraph.

**[5] p6, §II.C highlight (the "To second order about any wall-normal point... velocity, shear, curvature... 1/2 ... R" paragraph).**
> Well for a thin, 2D boundary layer, in which only one velocity component is non-negligible, and only wall-normal derivative of that velocity is much larger than others (and other derivatives can be changed without changing the profile), only these three are available to us as local surrogate for what the profile looks like. Strictly speaking we also have a fourth independent number, Re_Omega. But since Drela's ampflication rate is Re-independent after the cutoff, modeling rate shouldn't involve Re_Omega. So these three numbers seem natural to use for modeling the ampflication rate. The 1/2 factor on curvature is motivated by Taylor series which offers some technical advantages.

*Disposition:* CONFIRM, strongly. Fold in the justification: thin-2D-BL => one dominant velocity component, wall-normal derivative dominant => exactly three local surrogates; the 4th number Re_Omega exists but Drela's rate is Re-independent past the cutoff, so the RATE excludes Re_Omega and Re_Omega is reserved for the ONSET gate (this justifies the rate/onset split); 1/2 on curvature from Taylor.

**[6] p7, §II.C-end highlight ("...velocity, shear, and curvature indicators; the shear/velocity balance is the profile-fullness Gamma=2Y^2/(X^2+Y^2) in [0,2] used in earlier work.").**
> It is worth spending another paragraph on the specialness of the Z-poles. At any other point on the sphere (or PR3), two independent linear combination of velocity shear and curvature vanishes. However, at the Z-pole, not necessarily so because Z is higher-power d scaled than the other two, and the other two "vanishes" compared to Z could mean that we are just infinitely-far away from the wall. This is what makes the longitude meaningful even at the pole.
>
> In fact, let's get rid of all Gammas in the paper at some point. Compared to the longitude it misses a sign information (when the direction of curvature is available). Use S_hat here. Also I don't know in mathematics how people represent algebra in RP3. A single coordinate doesn't make sense to say positive or negative. But the product of two coordinates have definite sign. All our modeling terms needs to be like that, product of two coordinates that makes definite numbers.

*Disposition:* CONFIRM + EXTEND. (a) Add the Z-pole-specialness paragraph (longitude stays meaningful at the curvature pole because Z carries a d^2 scale, so the other two "vanishing" can just mean far-from-wall). (b) Replace Gamma with S_hat throughout (S_hat = Y/sqrt(X^2+Y^2) = sin(longitude) keeps the shear-direction sign that the even Gamma discards); Gamma also appears in the validation indicator plots (#68 scope). (c) The RP^2 sign-algebra principle is CORRECT and has teeth: a degree-1 coordinate (X, Y, Z, or g=Y-X-Z) flips under the antipodal map and is NOT well-defined on RP^2; only even products (two odd coords) are. The rate's S_hat*g part is even (good), but BOTH the sqrt(g) shaping (S_hat*sqrt(g)) AND the onset argument Re_Omega,crit(g) currently take g alone (odd) => not yet definite. To honor the principle, recast both on the even product P = S_hat*g. This is the hinge to 3D-implementability (3D has no global sign convention).

**[7] p8, Fig 1 highlight (caption / figure).**
> Don't plot the negative contour lines.

*Disposition:* CONFIRM (trivial). Re-render Fig 1 with only the zero + positive S_hat*g contours.

*Implemented 2026-07-21* in `main.tex` §II (compiles clean, 79 pp, verified via PyMuPDF render): §A merged profiles+temporal-vs-spatial with the earlier-drafts paragraph removed [4], phase speed cited as c≈0.5 U_e (michalke) [2], u≈0 [3]; §B added the "why exactly three / Re_Ω excluded from rate" justification [5], the Z-pole-specialness paragraph [6a], Ŝ=Y/√(X²+Y²)=sinλ with Γ removed [6b], and the even-product/3D sign-algebra paragraph [6c]; §C now discovers the parabola g=0 from the plot [1]; rate and onset recast onto the definite even product P=Ŝg (eq:P, eq:rate, eq:onset, eq:pai); Fig 1 re-rendered with positive+zero P-contours only [7]. Favorable-gradient machinery (K_λ/K_r/f_λ/cliff) dropped — CPU favorable-wedge test showed P→0 self-suppresses favorable, matching Drela (a_max=0.19 Michalke; c√ small, onset on P). §III realignment (drop favorable subsection, regenerate Figs 2–4 on the sphere kernel, nomenclature) still pending.

## Round of 2026-07-21 (14 comments, pages 6-7, merged §II.A profiles + §II.B indicator sphere)

Trims of filler + a condensation of the profile catalog + two substantive (Z-pole smoothness reframe, R-normalized wording). All ADDRESSED in main.tex §II (compiles clean).

**[1]** highlight "and---decisively---where" > *remove decisively, discussed next paragraph anyway.* — Addressed: dropped "decisively".
**[2]** highlight "strongly" > *remove -- meaningless intensifier, sweep again.* — Addressed: "strongly stable"->"stable", "strongly inflectional" dropped.
**[3]+[4]** highlight the verbose Blasius-TS / adverse-inflect / sep-limit clause chain > *remove; replace with "the instability waves amplify faster than TS waves on flat plate".* — Addressed: catalog condensed to "favorable (full, convex, stable) -> Blasius (viscous TS) -> adverse, where the instability waves amplify faster than the Tollmien-Schlichting waves of a flat plate."
**[5]** highlight "branch" > *of Falkner-Skan.* — Addressed: "reversed-flow lower branch of Falkner--Skan (Stewartson)".
**[6]** highlight "inviscidly unstable and" > *remove.* — Addressed.
**[7]** highlight "fully inflectional and" > *remove.* — Addressed.
**[8]** highlight "and, we now argue, decisive," > *but also important.* — Addressed: "easy to overlook but also important".
**[9]** highlight "over-amplify" > *unboundedly.* — Addressed: "over-amplify unboundedly".
**[10]** highlight "merely mis-" > *merely slightly mis-scale.* — Addressed: "does not merely slightly mis-scale".
**[11]** highlight "rate---the dead-air over-amplification that drives N past e^9." > *remove.* — Addressed: clause deleted ("deposits a diverging spatial rate.").
**[12]** highlight "local geometry" > *local sensors of the profile shape. The next subsection should start by echoing these "local sensors".* — Addressed: "calls for local sensors of the profile shape..."; §II.B now opens "The local sensors are few."
**[13]** highlight "The three indicators do not carry equal weight" > *Rigorously, the smoothness requirement isn't uniform; explain why the kernel should be smooth elsewhere on the sphere but not so strictly at the Z-poles.* — Addressed: paragraph reframed around the smoothness requirement (continuous profile path => no jumps), strict on the bulk but relaxed at the poles because the Z-pole is the d->inf limit a real profile approaches but never attains (measure-zero, kernel never evaluated at it); longitude carries the distinction along the approach.
**[14]** highlight "curvature-weighted" > *say metrics normalized by R in Eq. (3) instead.* — Addressed: "unlike the components normalized by $R$ in \eqref{eq:indicators}".


## Round 2026-07-21b (§II sphere-kernel merge polish, 12 remarks; pages 5-8)

Extracted via /tmp/extract_annots.py (rects only, no QuadPoints this round) + visual read of fitz-rendered pages.
Build note: this TeX Live now predefines \Bbbk, so amssymb clashed fatally; fixed with `\let\Bbbk\relax` before `\usepackage{amssymb}` (not a content change). Compiles 70 pp, 0 undefined.

**[1]** highlight "The functional forms were settled iteratively with airfoil results visible, but the constants themselves are fixed against canonical limits alone---...---so the airfoil cases of Secs. IV--V remain predictive tests." > *remove* — Addressed: sentence deleted.

**[2]** highlight "$P_\mathrm{AI}=a\,\omega\,\tilde\nu$" (inline) > *worth its equation number, so that later when we say "the rate is a temporal one" we can refer to the rate "a" in this equation* — Addressed: promoted to numbered display \eqref{eq:pai_def}; the "the rate $a$ ... is a temporal one" sentence now points to it.

**[3]** highlight "(viscous" > *remove* — Addressed: "viscous" dropped -> "the Blasius layer (Tollmien--Schlichting)".

**[4]** highlight "velocity, $u\,\partial_x\tilde\nu = a\,\omega\,\tilde\nu$" > *Refer to the previous equation, assuming steady state in a thin layer, and viscous effects negligible (any other assumptions needed to rigorously derive this simplified equation?).* — Addressed: derived from \eqref{eq:pai_def}: "with the diffusion and destruction of (2) negligible against production in the amplifying range, (3) drives the working variable along particle paths, $D\tilde\nu/Dt=a\omega\tilde\nu$; in a steady, thin shear layer the material derivative collapses to streamwise convection ($\partial_t=0$, $v\partial_y\ll u\partial_x$), giving $u\partial_x\tilde\nu=a\omega\tilde\nu$." The three assumptions (production >> diffusion+destruction; steady; thin-layer/BL $v\partial_y\ll u\partial_x$) are the complete set for this reduction.

**[5]** highlight "scales" (in "exactly three scales") > *three velocity scales* — Addressed: "exactly three velocity scales".

**[6]** highlight "divided" (in "divided by their common magnitude R") > *nondimensionalized* — Addressed: "nondimensionalized by their common magnitude $R$".

**[7]** highlight "only" (in "reached only in the far-from-wall limit $d\to\infty$") > *doesn't sound right. Theoretically you can reach the curvature pole near wall with some concocted profile. So replace "only" with typically. And instead of saying far from wall limit, which would induce the reader to still think in terms of boundary layers. Say in the shears and vortices away-from-the-wall.* — Addressed: "is typically reached in the shears and vortices standing away from the wall, where the growing $d^2$ inflates $Z$ until it dominates the velocity and shear".

**[8]** highlight "non-smooth" (in "left non-smooth at the poles") > *Continuous but non-differentiable* — Addressed: "left continuous but non-differentiable at the poles".

**[9]** highlight "We measure the shear content by the shear fraction" > *Don't say "we use" it here. Just propose this as an example of continuous, non-differentiable on the pole of the sphere, but legit to use.* — Addressed: reframed as "The shear fraction [eq] is an example: continuous everywhere on the sphere but non-differentiable at the curvature pole---where $X,Y\to0$ and the ratio depends on the direction of approach---yet legitimate on $\mathbb{RP}^2$." (dropped "we measure").

**[10]** highlight end of §II.B ("...indicator vectors survive.") > *The whole subsection started with these indicators for identifying the profiles. It should close with tracing all these profiles on the sphere, starting by identifying the wall point, walking through Fig 1 on all these profiles, all starting from the wall point and ending at the velocity pole. Each profile becomes more complex. Perhaps couple this addition to modifying next subsection's beginning, which will start by highlighting where the amplification region is -- we also need to reproducibly explain how we are marking the thickened regions in the profiles in the next subsection.* — Addressed: §II.B now CLOSES with a profile-tracing paragraph (common wall point $X=Y,Z=0,\lambda=45^\circ$ on the equator -> velocity pole $X\to1$, fanning out in order of increasing instability, walking Fig 1). The merged next subsection OPENS by marking the amplifying band with a reproducible criterion: "the contiguous interval over which the most-unstable Orr--Sommerfeld mode's production density $p=\tfrac\alpha2|\mathrm{Im}(\phi'\phi^*)||U'|$ stays above half its peak (the thick segments in Fig. 1)".

**[11]** highlight Fig 1 caption ("the bold contour is its zero locus, which contains the parabola great circle g=0") > *I don't like using a thickened special contour line for zero amplification. Perhaps instead make the contour line on some small nonzero value instead of zero, same color as the other contour lines, thus alluding to the diagonal great circle and shear=0 being something special..* — Addressed: Fig 1 regenerated (/tmp/fig12_model.py) -- removed the bold black P=0 contour; added level P=0.1 to the grey same-weight family so the lowest contour hugs the P=0 locus without a special line. Caption -> "the lowest, at small positive $P$, hugs the neutral locus $P=0$---the parabola great circle $g=0$ together with the shear-free meridian".

**[12]** highlight "D. The rate and its ceiling" (heading) > *Just merge into the last subsection, modeling where and how much the profiles are amplifying.* — Addressed: §D merged into §C; retitled "Where and how much the instabilities amplify"; labels sec:live+sec:rate+sec:calib_rate all on the one heading (all refs survive).


## Round 2026-07-22 (§II.B profile-tracing + g introduction, 6 remarks; page 7)

**[1]** highlight "while the curvature vanishes" > *The curvature doesn't vanish -- it's the d^2 weighing that makes the nondimensionalized version vanish.* — Addressed: "although the curvature $u''$ is finite there, its $d^2$ weight sends the indicator $Z\to0$".

**[2]** highlight (end of tracing, "before turning back to the pole") > *eventually settles in* — Addressed: "Each curve eventually settles at the \emph{velocity pole} ($X\to1$) as the shear dies at the edge."

**[3]** highlight "become curves on this sphere, and they share a common shape" > *Put Fig 1 close to here, and refer to the figure.* — Addressed: moved the fig:model float to immediately after the §II.B tracing paragraph (was in §II.C); the paragraph opens "become curves on this sphere (Fig.~\ref{fig:model})" and refers to it again.

**[4]** highlight "reversed-flow profiles swing progressively farther onto the shear side" > *this is wrong framing here. in terms of shear both the stable and blasius share the same range. The difference is, the stable one leaves the wall point along a particular angle defined by a parabola: X - Y + Z = 0, verifiable via Taylor series.* — Addressed: dropped the shear-range framing entirely. Every profile leaves the wall point tangent to the parabola great circle $g\equiv Y-X-Z=0$ (a Taylor identity, eq:parabola moved up into §II.B); profiles are distinguished by HOW they depart it, not by shear range.

**[5]** highlight "the neutral circle; and the adverse, incipient-separation" > *deviates from that particular angle as it leaves the wall, because it has zero curvature at the wall and thus has positive third derivative, here we can introduce the residual of the Taylor series being the integral of the third derivative, and define it as g. And allude to the physical importance of it.* — Addressed: $g$ introduced HERE as the Taylor residual $g(d)=-\int_0^d\tfrac12 s^2 u'''\,ds$ (eq:g_integral, moved up from §II.C); "a profile can leave [the parabola] only through its third derivative"; Blasius ($u''_w=0$, nonzero $u'''$) peels to small $g>0$; $g$ = accumulated curvature of the vorticity ($u'''=-\omega''$) = inflectional content Rayleigh ties to instability.

**[6]** highlight (tracing paragraph) > *First say that adverse pressure gradient one have even higher g -- the integrated third derivative is significantly larger partly because the derivative itself is larger, partly because d is larger where the third derivative is large, partly because it's integrated over a larger wall distance. And finally describe the separated profile: it goes the other direction (dotted line) but also exactly along the same angle observed by the favorable PG: a parabola but in the other direction. In this range the flow is stable. But then as it cross over the velocity=0 longitude (identified by the shear pole on the equator) g becomes positive and larger than even the separation threshold. Polish my language ... Also modify the next subsection to echo--but not repeat.* — Addressed: adverse "lifts $g$ higher still---its third derivative is larger, it is largest where the $s^2$ weight is larger, and it accumulates over a thicker layer"; separated "leaves the wall the \emph{other} way (dotted, reversed flow) along the same parabola, and along that arc it too is stable; but as it crosses the velocity-zero longitude---the shear pole on the equator---$g$ turns positive and climbs past even the separation threshold, the most unstable member". §II.C ("Where and how much") rewritten to ECHO (references eq:parabola/eq:g_integral now defined in §II.B; states only where the bands SIT and where amplification BEGINS) rather than repeat the derivation.


## 2026-07-23 agent-review round (agent-paper-review/2026-07-23-0658.md)

Full item-by-item response: `agent-paper-review/2026-07-23-0731-response.md`.

Addressed this round: **M3** (N=9 is an untuned prediction; Fig. 4 caption + Conclusion now say only N=1 is imposed), **M5** (SA wall-function $g\to g_w$; mesh growth-ratio $(g-1)$ in words), **M6** (P reserved for blended production; coordinate always written $\hat S g$), **M8** (Conclusion bubble numbers now the Sec. 5 reattachment offsets 0.04–0.06c late / 0.03c early), **M9** (footnote excludes the NLF $\alpha=0$ lower front, 0.10c mfoil–XFOIL spread), **M10** ($\lambda_p$ defined as diagnostic, nomenclature entry, distinguished from longitude $\lambda$), **M11** (tanh gate form at eq:onset; clip$\langle\cdot\rangle_0^1$ and $\langle\cdot\rangle_+$ defined at first use), **M14** (implementation/numerics paragraph added after the Flow360 solver-and-seeding paragraph at the head of Sec. 3), **M15** (abstract: three modifications named; Daedalus 3D sentence added), **M18b** (AVL, Construct2D, Falkner–Skan 1931, Stewartson 1954 cited; bib entries added), and minors 1–9, 10 (front is bold, per generator; body text fixed), 11, 12 (verified 1994 already cited), 14, 15, 16, 18, 19. Minor 13: Somers PDF absent from paper/references/, figure numbers unverifiable here.

Deferred (handled separately): M1, M2 (Daedalus Q1/Q2 / predecessor-kernel evidence), M4 (N=1 anchor operational definition), M7 (signed-vs-magnitude 3D), M12 (onset-collapse figure), M13 (Fig. 1 PDF on other machine), M16 (section reordering), M17 (Sec6/AppB numbers), M18a (literature table), M18c (flat-plate grid statement), minors 13, 17, 20.

Side effect: removed the duplicate `alam_sandham_2000` bib entry (it aborted bibtex before the new entries); kept the correct JFM 410, 1–28 copy. Build clean except the known missing `figs/indicator_sphere.pdf`.

## 2026-07-23 agent-review round 2 (agent-paper-review/2026-07-23-0807.md)
Addressed alongside the Section II restructure (II.D onset+diffusion, II.E
handover). All 6 new majors and 12 of 13 new minors resolved; item-by-item in
agent-paper-review/2026-07-23-0910-response.md. Pass-1 M1/M2/M7/M17/M18a/M18c
remain owed. NOTE: review pass 2 indirectly exposed a real defect — the
marched N=1/N=9 crossings were wall-normal-resolution sensitive; all quoted
crossings now come from the grid-converged instrument and the onset scale k
is re-anchored there.

## 2026-07-23 annotated.pdf round (10:33, p.9, 3 remarks)
1. HIGHLIGHT "p=" — "remove, together with 'p is .. not ...' comment following."
   RESOLVED: the Orr–Sommerfeld production density is no longer given a symbol
   (inline expression only) and the disambiguating clause ("lower-case p ...
   distinct from the production P") is deleted.
2. HIGHLIGHT "not the angle of attack," — "Use a subscript on alpha instead,
   remove the 'not angle of attack' part."
   RESOLVED: α → α_x with a minimal parenthetical "(α_x the streamwise
   wavenumber)"; the angle-of-attack disclaimer is gone.
3. HIGHLIGHT "shear side, toward" — "east of the velocity pole longitude,
   towards ..."
   RESOLVED: "every amplifying band lies east of the velocity-pole longitude,
   toward the shear pole +Y and away from the pure-curvature poles."

## 2026-07-23 agent-review rounds 3+4 (0915, 1023) + whole-equation k change
Addressed with the k-applied-to-all model change (k=0.712 uniform; family 6%
rms) and the annotated.pdf p.9 round; item-by-item in
agent-paper-review/2026-07-23-1050-response.md.

## 2026-07-23 annotated.pdf round 2 (15:07, pp.10-13, 6 remarks)
1. HIGHLIGHT "not error," (p.10, inviscid-limit discussion) — "remove".
   RESOLVED: "---headroom, and what consumes it is the subject of the next
   paragraphs."
2. HIGHLIGHT "kernel is the correlation to three percent ... stays bounded
   by a_max." (p.11) — "remove".
   RESOLVED: the sentence now ends at "the ratio is 0.97."; the deep-separated
   0.46 stays in Table 2 uncommented ("three facts" -> "two facts"; App. B's
   temporal-units opener no longer cites the removed aside).
3. HIGHLIGHT "the coupled solver pushes back" (p.12) — "remove".
   RESOLVED: "...would buy nothing the correlation asks for: the molecular
   floor sets..."
4. HIGHLIGHT high-Re caveat paragraph ("One caveat travels ... tune around
   it.", p.12) — "remove".
   RESOLVED: paragraph deleted; the sweep paragraph's closing now absorbs the
   one factual clause ("climbing toward the finite ceiling of Table 2 as the
   diffusion keeps vanishing") instead of pointing at the deleted text.
5. HIGHLIGHT "observed" (p.12, pivot paragraph) — "the physical -- ...one
   sentence that 'reminds' that the start of growth as a Re_theta floor
   should be useful in the beginning of the paragraph."
   RESOLVED: the paragraph opens with the reminder ("A laminar layer does not
   amplify from birth: linear theory gives each shape a floor, the critical
   Reynolds number Re_theta0(H) ... third column of Table 3"), and
   "far below any observed inception" -> "far below the physical floor".
6. HIGHLIGHT "The" (p.13, start of II.D) — "This section should start by
   describing e^N envelope theory in one or two sentences: no growth till an
   H-dependent Re_theta, then Re_theta-independent growth."
   RESOLVED: II.D now opens: "In the e^N envelope description a laminar layer
   amplifies in two regimes: nothing grows until the layer reaches a
   shape-dependent critical Reynolds number Re_theta0(H), and past it the
   envelope earns amplification at a rate that is, to the accuracy of the
   method, Reynolds-independent---a function of the shape alone [Drela-Giles].
   The algebraic rate (eq:rate) carries the second regime...; the first---the
   viscous inception---is a separate, multiplicative gate..."
