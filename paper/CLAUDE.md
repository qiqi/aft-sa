# Writing and figure conventions — SA-AI paper and whitepaper

Rules for anyone (human or agent) editing `sa-ai.tex`, `whitepaper.tex`, the shared
appendix bodies, or the figure scripts in `repro/`. These are the author's stated
preferences, not suggestions.

The two documents have different jobs since November 2026:

- `whitepaper.tex` is the arXiv reference (arXiv:2608.10303). It states.
- `sa-ai.tex` is the AIAA SciTech 2027 manuscript, in `\documentclass[conf]{new-aiaa}`
  format. It is a rewrite of the whitepaper whose only goal is to **educate**: it
  explains how the model comes about and spends no sentences arguing that the model
  is good or better than another.

## 1. Sentence length

**Aim for mostly short and very short sentences.** A long sentence is a tool for
emphasis, used sparingly — roughly once in several paragraphs, never as the default
rhythm.

- Target: median ≤ 20 words; very few sentences over 50.
- Prefer a full stop to a semicolon, and a full stop to an em-dash aside.
- A sentence carrying three nested qualifications is three sentences.

Check before and after any prose edit:

```bash
python3 repro/tools/sentence_stats.py sa-ai.tex whitepaper.tex   # distribution
python3 repro/tools/longest_sentences.py sa-ai.tex 45 20         # worst offenders
```

Status on 2026-11-19: paper median 15, whitepaper median 27. The paper's remaining
long sentences are in the results and remarks carried over from the whitepaper.

## 2. No first person — and no disguised first person

Never use *we, I, us, our, my*. All four files are currently at zero; keep them there.

More important: **do not simply convert to agentless passive.** "The shape is
represented by…" still means "by us". Give the sentence a real subject — the model,
the figure, the sweep, the campaign, the equation, the flow. Describe what happens
if anyone does this, not what the authors did.

| wrong (first person) | wrong (hidden actor) | right |
|---|---|---|
| We present a model in which… | A model is presented in which… | Natural transition can be modeled by using only… |
| We represent the shape by a soft minimum | The shape is represented by a soft minimum | The shape follows a soft minimum |
| We claim no physics for it | No physics is claimed for it | It carries no physical content |
| We therefore sweep incidence | Incidence is therefore swept | The campaign therefore sweeps incidence |

## 3. Whitepaper voice: minimize description, no discussion

The whitepaper states. It does not explain, justify, or reflect. Bare numbers,
bare statements, no motivation paragraphs, no "which is why…" clauses. If a
sentence could be deleted without losing a number or a fact, delete it.

The paper carries the reasoning, as teaching: the need first, then the quantity that
meets it, then what goes wrong with the obvious choice. It does not defend.

## 4. Structure

The model section of the paper has eight subsections and ends with the constants:

1. Transport Equation and Blended Sources
2. Local Indicators of the Boundary-Layer Profile Shape
3. Amplification Rate and Onset Gate Built from the Indicators
4. Reduced Laminar Diffusion
5. Handover Blend and Destruction Tie
6. Eddy-Viscosity Assembly
7. Freestream Seed
8. Constants

Subsections 2 and 3 carry longer titles than in the whitepaper ("Local indicators",
"Amplification rate and onset threshold") so that the titles state what each one is
for. The whitepaper has the same eight in the same order, then two more: "Relaminarization
behavior" and "Stagnation-point bistability". **Do not bring those two into the paper.**
The stagnation-point branch was later found to be held up by streamwise numerical
diffusion, and that finding, with the front-propagation analysis it rests on, is the
subject of a separate paper (`../aiaa_aviation_2027_carbuncle/`). The paper must not
contradict it or duplicate it. The residual-scaling protocol those sections ended with
now opens the paper's Results section.

Results order in both: flat plate → NLF(1)-0416 → Eppler 387 → Eppler Reynolds
sweep → two-element → cylinder drag crisis → Daedalus → spheroid, then Remarks on
the operating envelope. The paper has no table of contents (AIAA conference format);
its Introduction, Appendix and Acknowledgments sit outside the shared spine.

The `_wp` table files in `tables/` and `appendix_twosource_body_wp.tex` are `\input`
by both documents. An edit there changes both.

## 5. Notation

- $X, Y, Z$ are **dimensional** (velocity scale). $\hat X, \hat Y, \hat Z$ are
  normalized by $U_\mathrm{loc}$.
- $U_\mathrm{loc}$ is the local velocity scale. Never `R` — that reads as a radius.
- $\sigma_{Re}$ is the onset gate (was `S`, then `F`). $\sigma_{v1}$ is the
  $f_{v1}$-bypass weight (was `b`). $\sigma_P,\sigma_D$ are handover weights.
  $g_\chi, g_q$ are the bypass gates (were `s`, `G`). The $\sigma$ prefix marks a
  sigmoid-like gate.
- $S$ is shear. Do not reuse it for anything else.
- Call $I$ the **accumulated inflection** and $\hat I$ the **nondimensionalized
  accumulated inflection**. Never "Rayleigh coordinate". (Citing *Rayleigh's
  theorem* for inviscid neutrality is fine — that is the theorem, not the name of
  the coordinate.)
- Keep the nomenclature tables in both documents in the same order, and make sure
  every symbol used in the body appears in them. A symbol used in one passage only
  does not get a row: say it in words.

## 6. Wording

- **A verb that relates two quantities names the relation.** Not "dN/dRe_theta
  follows from H" but "is a function of H alone". Not "sigma_D tracks sigma_P" but
  "is a linear function of sigma_P". "Follows", "comes from", "is tied to", "carries",
  "sets", "drives" and "enters" leave the reader to guess the mathematics.
- **No verbs of position for colour.** Not "reattachment runs 0.02c late" but "the
  flow reattaches 0.02c late". Rewrite around the real verb; use "is" only for a
  true position or value.
- **"Has", "takes", "makes" and "does" are relation verbs too.** Not "a point has no
  H" but "H cannot be determined from the values and derivatives of the velocity at a
  point". Not "two terms have one ratio" but "two terms form one ratio".
- **Prefer the phrase that carries more information**: "boundary-layer profile", not
  "velocity profile", where the shape of the layer is meant; "the freestream turbulence
  intensity", not "the disturbance level of the stream"; "the boundary layer", not
  "the layer"; "two ratios", not "two numbers"; the symbol or the name, not "the field" or "the two".
- **Never refer back to a list by position** ("the first job"). Restate the thing and
  say where it was introduced. Readers keep insights and forget bookkeeping.
- **A special case or limit says so up front**: "Note that baseline SA is recovered
  exactly by setting ...".
- **Check a term for a meaning it already has in the field.** "Relaminarization" and
  "turbulence front" were both rejected for this reason. An everyday word used as a
  term goes in quotes at first use ("empty"). A coined word ("graze") is defined at
  first use by pointing at what the reader can see in a figure, or is not used.

## 7. Figures

- Every figure carries a number and a one-line description of what it is, as simple
  as possible. Figures sharing a description are grouped as one numbered figure with
  `\figcont` continuation pages ("Figure X, continued, α = Y").
- Order incidences, Reynolds numbers, and other physical parameters ascending
  across multi-page figures.
- Captions are minimal. Anything the caption states does not also belong as in-plot
  text — move it out or delete it.
- **In-plot type should render at about caption size (10pt).** The page scales the
  PDF by `\textwidth`/canvas width, so a 7.4in canvas at `\textwidth` = 6.5in shrinks
  text by 0.878. Compensate by scaling declared font sizes by the inverse
  (`_scale_text_for_page` in `regen_nlf_polar.py` and `regen_eppler_v2.py`). Verify
  by measuring the emitted PDF, not by eye.
- A later figure may repeat an earlier figure's data for comparison. The reverse —
  an early figure showing results introduced later — is backwards.

## 8. Data honesty

- **Never silently truncate a range.** A `[::3]` slice over station lists once cut
  the largest transverse-curvature value and understated a quoted range by a third.
  If coverage is bounded, say so in the text.
- **Do not reuse artifacts across model versions.** Check the solver binary date
  against the artifact date before trusting an existing `volume.pvtu`. The
  `daedalus_fv1` tree exists because of the $\sigma_{v1}$ change; anything older is
  a different model.
- Regenerate a figure after changing its script. A stale PDF on disk silently
  contradicts the source.

## 9. Tables

Prefer parallel **columns** over parallel rows when comparing codes or quantities:
rows are the physical sweep (α × grid level), column groups are the codes
(Flow360 O-grid | Flow360 cavity | OpenFOAM), each carrying the same quantities.
Merge tables that share row keys rather than stacking them.

## 10. Before finishing any edit

```bash
pdflatex -interaction=nonstopmode sa-ai.tex     # run 3x for refs
pdflatex -interaction=nonstopmode whitepaper.tex
grep -cE '^! ' sa-ai.log ; grep -c undefined sa-ai.log     # both must be 0
grep -cE '\b([Ww]e|[Oo]ur|[Oo]urs|[Mm]y)\b' sa-ai.tex      # must be 0
python3 repro/analytic/envelope_bounds.py                  # asserts quoted numbers
```

`envelope_bounds.py` asserts every number the documents quote for the operating
envelope. If a quoted value changes, that script must change with it and still pass.
