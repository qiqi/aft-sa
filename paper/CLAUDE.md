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

The model section of the paper has seven subsections and ends with the constants:

1. Transport Equation and Blended Sources
2. Local Indicators of the Boundary-Layer Profile Shape
3. Amplification Rate and Onset Gate Built from the Indicators
4. Laminar Diffusion and the Onset Factor
5. Laminar-to-Turbulence Handover Blend and Destruction Tie
6. Eddy Viscosity without Wall Damping in Separated Shear Layers
7. Constants

The freestream seed (Mack's map, $\chi_\infty=c_{v1}e^{-N_\mathrm{crit}}$) is not a
subsection of the model in the paper. It opens the flat-plate subsection of the
Results, the case that varies $Tu$, because $\chi_\infty$ is an input of each case. Table 1 lists the baseline SA constants with
the sections and equations of the 1994 journal version of the SA paper; Table 2 lists
the SA-AI constants in the order Section II determines them.

Subsections 2 and 3 carry longer titles than in the whitepaper ("Local indicators",
"Amplification rate and onset threshold") so that the titles state what each one is
for. The whitepaper has those seven and "Freestream seed" before the constants, then two more: "Relaminarization
behavior" and "Stagnation-point bistability". **Do not bring those two into the paper.**
The stagnation-point branch was later found to be held up by streamwise numerical
diffusion, and that finding, with the front-propagation analysis it rests on, is the
subject of a separate paper (`../aiaa_aviation_2027_carbuncle/`). The paper must not
contradict it or duplicate it. The residual-scaling protocol those sections ended with
now opens the paper's Results section.

Results order in both: flat plate → NLF(1)-0416 → Eppler 387 → Eppler Reynolds
sweep → two-element → cylinder drag crisis → Daedalus → spheroid, then Remarks on
the operating envelope. The paper has no table of contents (AIAA conference format);
its Introduction and Acknowledgments sit outside the shared spine. The two-source
appendix is in the whitepaper only (removed from the paper on 2026-11-25: it is not
part of the model and does not teach it).

In the paper each Results subsection opens with the one thing the case shows about the
model, and the prose walks through the figures in order. A multi-page figure set
(surface distributions, near-wall contours, grids, fields) contributes one page to the
text; the complete sets are in the unnumbered **Supplementary Figures** section, in
the order of the cases. The back matter is: Acknowledgments (with the
closing epigraph), Supplementary Figures, References. In Section III the figures and tables
float (`[!htbp]`), with a `\FloatBarrier` before each subsection, so that the prose
is continuous.

**Captions say what is plotted and nothing else**: the quantity on each axis or in
each panel, what each symbol or line style is, and the condition. Results belong in
the prose. Captions use the words of the prose ("the L2 grids", "the OpenFOAM
replication", "the largest chi near the wall crosses one", "main element"), not the
campaign's ("L2 pair", "port", "seed", "up-ladder", "sheets", "suites"). The
all-triangular grids are "the unstructured grids" in the paper; "cavity" is the name of
the mesher's operator, is not a kind of mesh, and appears once, with its citation.
The same holds inside the plots: row labels and legends say "unstructured L2" and
"L2 unstr", never "cavity" or "cav". On 2026-11-24 the labels of 86 existing figure
PDFs were replaced in place (same font, nothing else touched) and the nine scripts
that draw them were changed to match, because regenerating them needs the full
solution trees on 014. The whitepaper includes the same files.

The paper does not `\input` the shared `_wp` tables directly. It inputs
`tables/*_paper.tex`, which `repro/tools/make_paper_tables.py` writes from the `_wp`
files with the paper's captions and floating placement. Change a caption in that
script and rerun it; rerun it also after a campaign script re-emits a `_wp` table.

The `_wp` table files in `tables/` are `\input` by both documents. An edit there
changes both. `appendix_twosource_body_wp.tex` is `\input` by the whitepaper only.

## 5. Notation

- $X, Y, Z$ are **dimensional** (velocity scale). $\hat X, \hat Y, \hat Z$ are
  normalized by $U_\mathrm{loc}$.
- $U_\mathrm{loc}$ is the local velocity scale. Never `R` — that reads as a radius.
- $\sigma_{Re}$ is the onset gate (was `S`, then `F`). $\sigma_{v1}$ is the
  $f_{v1}$-bypass weight (was `b`). $\sigma_P,\sigma_D$ are handover weights.
  $\sigma_\chi, \sigma_q$ are the bypass gates (were `g_\chi`, `g_q` until
  2026-11-25, and `s`, `G` before). Every $\sigma$ with a subscript is a factor
  between zero and one; the bare $\sigma$ is SA's diffusion constant. `g` is
  SA's function inside $f_w$ and names nothing else.
- Force coefficients: $C_l, C_d, C_m$ for a section (airfoils, the two-element
  section, the cylinder); $C_L, C_D$ for the Daedalus wing only. Never $c_l$. The
  freestream speed is $U_\infty$, never $V_\infty$. Separation and reattachment
  stations are $x_{LS}, x_R$. `make_paper_tables.py` rewrites these in the shared
  tables; the whitepaper keeps its own symbols.
- `figs/daedalus_surface_a{4,5,6}.pdf` could not be redrawn on 2026-11-25: with the
  `chi_surface.npz` and surface files now on 014 the script draws corrupted $\chi$
  panels. Their row label was changed from $c_l$ to $C_l$ inside the PDFs (PyMuPDF
  redaction of three glyphs); nothing else in them changed. Find out why before the
  next redraw.
- `figs/eppler_resweep_forces.pdf` and `data/eppresweep_forces_computed.json` were
  recomputed on 2026-11-25. The stored values at $Re=2\times10^5$ were from older
  solutions and disagreed with Table `t:epp`; `check_results_prose.py` now asserts
  that the two tables and the recomputed forces agree. The script needs the case
  directories (`/local_data_2/qiqi/sa-ai-airfoil-slices` on 019, with
  `total_forces_v2.csv` copied from 014), the campaign JSON files of `flow360_fv1`,
  and `mfoil_/xfoil_eppler387_sweep_a5.pkl` (014: `flow360_fr/` and `flow360/`).
- In figures the codes are "Flexcompute Foil", "XFOIL" and "mfoil" (never "FlexFoil"
  or "xfoil"). On 2026-11-25 fourteen included figures were renamed inside their PDFs
  with `repro/tools/relabel_pdf_text.py` (it rewrites one text line and moves the
  outlined symbols with it), because their data are not all readable on 019
  (`mfoil_nlf0416_Re4M.pkl` on 014 is owner-only). The scripts carry the new labels.
  Two entries were shortened to fit their legends: "Flexcompute Foil $N$" on the
  NLF pages (was "FlexFoil $N$ ($e^9$)") and "Flexcompute Foil ($N$, $C_p$, $C_f$)"
  on the Daedalus section pages (was "FlexFoil strip (...)").
- Daedalus reference: the totals (polar, forces table) use AVL with XFOIL profile
  drag (`daedalus/avl_compare.py`, `xfoil_cd`). The strip distributions (section
  pages, third row of the surface maps, section $C_d$ in the polar's panel b) come
  from `flow360_ai/flexfoil_daedalus_strips.pkl`, computed with Flexcompute Foil by
  `regen_daedalus_strips_cache.py`. The figures are labelled accordingly.
- Section II.G is the three-dimensional form of the indicators, Eqs. (17)-(18)
  (moved from the opening of Section III on 2026-11-25); Constants is II.H.
- The end of Section II.D compares the model with Drela-Giles over the Falkner-Skan
  family. Its numbers are asserted in `repro/analytic/tab_march_stations.py`
  (attached profiles) and `repro/analytic/fig02_model_calibrate.py` (reversed-flow
  profiles). It states the unfavourable results too: the growth rate is 62-83% of
  Drela-Giles from N = 1 to 5, and the N = 9 station is 6% to 31% late.
- A drag count is $10^{-4}$; Section III.C defines it at first use.
- In-plot type (author, 2026-11-25): fix only what is excessively small on the page;
  "a bit smaller is perfectly fine". The near-wall contour figures of
  `regen_chi_sheets.py` (Eppler at Re 2e5, NLF, the Reynolds sweep at alpha 5) and
  `regen_dragcrisis_bl_sheet.py` were redrawn with 10-13 pt type, which is about
  5.6 pt on the page. The Re 1e5 / 3e5 incidence sheets and the Daedalus sheets still
  have the old 6.5 pt contour labels (3.6 pt on the page).
- The rotation table of Section IV.A was removed (author, 2026-11-25); three of its
  rows are quoted in the text. `envelope_bounds.py` still asserts all ten.
- Floats in Section III: the two polars are at 0.75 of the text width so that each
  shares a page with the short figure before it. A full-page figure delays every
  later figure of its subsection, because floats keep their order.
- Say "transition location", never "front" (author, 2026-11-25). After the first
  mention in a passage, "the location" is enough.
- Every source drawn in a figure is cited in its caption (Figs. 8, 12, 14, 21 were
  completed on 2026-11-26). Legends name authors and years, never internal tags
  ("[R15]", "IJSRP"). The "Selig" stations of the Eppler bubble figure are from
  Shahjahan et al.'s Fig. 10, which cites Selig's 2003 VKI lecture notes
  (`selig_2003`).
- `repro/tools/relabel_pdf_text.py`: use `relabel_file(src, dst, edits)`, never several
  separate passes on one page. Removing text with MuPDF can displace the line that
  follows it in the page stream; `relabel_file` detects such lines and writes them anew
  in place. On 2026-11-26 all sixteen figures edited with the helper were redone from
  their clean versions for this reason. The SA-AI entry in the legend of
  `dragcrisis_cd_re.pdf` was drawn into the PDF; the script does not produce it.
- The nomenclature is alphabetical (Latin, capitals first, then Greek). Each row
  cites the equation, table or section that gives the symbol context. Symbols
  defined and used inside one subsection of Section IV have no row.
- `figs/daedalus_polar_sectional.pdf` needs `total_forces_v2.csv` and
  `Y_slicing_forceDistribution.csv` of the twelve L1/L2 cases. On 019 they are
  copied to `/local_data_2/qiqi/sa-ai/daedalus_fv1_csv`; set `D_FV1` to that
  directory before calling `main()`. The script's default tree is complete on 014 only.
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
- **The paper names nothing the reader cannot open**: no script, data file, directory,
  case name or environment variable, and not "the campaign". Say what was done
  ("digitized from Somers' Fig. 9") and cite what is public. The file names belong in
  this file and in `repro/`.
- **Never refer back to a list by position** ("the first job"). Restate the thing and
  say where it was introduced. Readers keep insights and forget bookkeeping.
- **A special case or limit says so up front**: "Note that baseline SA is recovered
  exactly by setting ...".
- **Check a term for a meaning it already has in the field.** "Relaminarization" and
  "turbulence front" were both rejected for this reason. An everyday word used as a
  term goes in quotes at first use ("empty"). A coined word ("graze") is defined at
  first use by pointing at what the reader can see in a figure, or is not used.

- **The solver is "Flexcompute Flow"** in the paper (renamed from Flow360 in November
  2026). The first mention in Section III, in each Results subsection and in the
  Acknowledgments adds "(formerly Flow360)". File names and script names keep
  `flow360`.
- **The XFOIL reimplementation is "Flexcompute Foil"** in the paper, its official name,
  not "FlexFoil". File names and script names keep `flexfoil`.
- `figs/flat_plate_batch_openfoam.pdf` is a copy of `../openfoam/flatplate_fig_openfoam.pdf`,
  drawn by `../openfoam/scripts/regen_flatplate_compare.py`. The OpenFOAM cases are on
  host 014 (`/local_data/qiqi/openfoam-sa-ai/cases`, visible on 015 under
  `/mnt/014-local_data`). On 019 a copy of the two fields the script reads is at
  `/local_data_2/qiqi/openfoam-sa-ai/cases`; run with `SAAI_OF_CASES` set to it and
  the compute venv python.

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
python3 repro/tools/check_results_prose.py                 # Section III prose against its tables
```

After a change to the kernel constants or to Section II.D, also run
`python3 repro/analytic/tab_march_stations.py` (a few minutes on 32 cores). It asserts
the II.D growth-rate table, the $N=1$ stations that set $c_\mathrm{on}$, and the
Falkner--Skan family numbers. The marcher's grid (`NX`, `NY` in `fig04_shapefactor.py`)
is 3200 x 9600; the old 1600 x 1200 put the Blasius $N=1$ station 5% late.

The distance from the $\chi=1$ crossing to the $\chi=c_{v1}$ crossing quoted in the
opening of Section III is measured by `repro/cfd/chi_crossings_airfoil.py` on all 230
airfoil solutions and stored in `data/chi_crossings_airfoil.json`. On 019 the slices it
reads are at `/local_data_2/qiqi/sa-ai-airfoil-slices` (copied from 014); run it with
`SAAI_CFD_ROOT` set to that directory and the compute venv python.

`figs/twoelement_flap_chi_map.pdf` (the unrolled flap) is drawn by
`../twoelement/plot_flap_chi_map.py out.pdf <five L2 case directories>`, in the black
labelled-contour style of the other near-wall figures. It also writes `out.json`, kept
as `data/twoelement_flap_layer.json`: the separation and reattachment stations, the
smallest $C_f$, and the $\chi$ inside the flap's boundary layer that Section III.E
quotes. The cases are links in `../twoelement/` to `/local_data_2/qiqi/sa-ai/twoelement_cases`
on 019; use the compute venv python (about one minute).

`check_results_prose.py` recomputes every difference and ratio that the prose of
Section III derives from the tables (fronts against measurements, drag at the same
lift, distances between crossings). A sentence that quotes such a number needs a line
there.

Section IV of the paper (Remarks on the Operating Envelope) has five subsections:
Rotation, Wall Curvature, Sweep, Suction, The Wake of Another Surface. Each remark
gives the flow, what stability theory finds (with a citation), the three terms of
Eq. (7) on the exact profile, and the size of the difference, in that order: the
physics first, then the model. There is no summary table (removed 2026-11-25 at the
author's request). `figs/envelope_sphere.pdf` is drawn by
`repro/analytic/fig_envelope_sphere.py`. The viscous length there is
$\delta_\nu=\sqrt{\nu x/U_e}$ (not $\ell$, which is the curvature vector of Eq. 17) and
the frame rotation rate is $\Omega_\mathrm{f}$ (not $\Omega$). The remark on the
smoothness of the sources is at the end of Section II.G.

`envelope_bounds.py` asserts every number the documents quote for the operating
envelope. If a quoted value changes, that script must change with it and still pass.
