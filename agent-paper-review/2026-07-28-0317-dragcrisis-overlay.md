# fig:dragcrisiscd literature overlay: sources, digitization QA, figure changes, draft caption
*2026-07-28 03:17, digitization/figure agent. Companion to
2026-07-28-0117-dragcrisis-litrange.md (the source map). Everything below
was digitized INDEPENDENTLY from freshly acquired PDFs per the binding
methodology (tick/gridline-detected calibrations, per-dataset check PNGs);
scripts: `paper/repro/cfd/digitize_dragcrisis_lit.py` (new) and
`paper/repro/cfd/regen_dragcrisis_cd_re.py` (extended). Data + provenance
README: `paper/repro/cfd/litdata/dragcrisis/`. Check PNGs (not committed;
regenerate with the script): `paper/repro/cfd/litdata/dragcrisis/checks/`.*

## 1. Sources acquired (9 PDFs; URLs in litdata/dragcrisis/README.md)

OPEN/primary: Delany & Sorensen NACA TN 3038 (NTRS; the API download
carries a 436-byte junk prefix before `%PDF` -- stripped); Wieselsberger
NACA TN-84 translation (NTRS); Roshko 1961 (Caltech CODA record
m8vtc-33e74); Henderson 1995 (Caltech CODA bqdev-z6q09); Catalano/Wang/
Iaccarino CTR brief 2001 (Stanford CTR); Rodriguez et al. 2015 accepted
ms. (UPCommons); Stringer et al. 2014 author ms. (CORE mirror --
Bath's portal is Cloudflare-gated, opus.bath.ac.uk redirects into it);
Stabnikov & Garbaruk 2020 (IOP open access); Veysey & Goldenfeld
physics/0609138 (arXiv).

NOT reachable, disposition:
- **Achenbach 1968** (JFM 34, paywalled): represented twice, secondary --
  (a) the Cd(Re) curve labeled "Achenbach (1968)" in Catalano et al.
  fig. 3, vector-exact, RESTRICTED to Achenbach's measured range
  6e4<=Re<=5e6 (outside it Catalano's curve is a composite of unclear
  pedigree -- cut); (b) Achenbach & Heinecke 1981 smooth-cylinder points
  via Rodriguez fig. 4 (vector).
- **Schewe 1983** (JFM 133, paywalled): secondary via Rodriguez et al.
  2015 fig. 4 vector stream (51 pts, 3e5-7e6, incl. the critical-range
  bistable branches). Cross-replot check against the independent
  Stabnikov-Garbaruk raster replot (their ref [15]): 10 matched pts in
  the overlap, mean dCd 0.029, rms 0.057 (branch-crossing matching
  inflates the rms) -- the two secondaries agree.
- **Cheng, Pullin & Samtaney 2017 WRLES** (JFM 820): the Caltech record
  is metadata-only (files disabled); no arXiv/KAUST/citing-table copy
  found. WRLES class represented by Rodriguez et al. 2015 instead
  (primary, tabulated). Listed as the one brief-item I could not source.
- **Zheng & Lei 2016 gamma-Re_theta** (FTC, paywalled; already cited in
  the bib as zheng_lei_2016): the transition-RANS class is covered by
  Stabnikov & Garbaruk's own SST gamma-Re_theta sweep (open primary).
- Stringer 2014 was recovered via CORE after the Bath portal blocked.

## 2. Datasets digitized (counts) and QA verdicts

| dataset | method | pts | QA verdict (check PNG) |
|---|---|---|---|
| TN-84 Wieselsberger 1921 | photostat: Viterbi trace of the faired curve Re 4.2-3e3 (140 samples) + symbol blobs Re 3e3-9.5e5 (46) | 186 | PASS with caveats: white-on-black photostat; y calibrated from label-glyph rows (piecewise; mid-plot stretch +10-19 px), x from 5 strong decade lines (global log fit); a corridor (+-0.16 dec) around hand-read waypoints was REQUIRED to keep the tracker off a full-width scratch at c~0.55 and off patchy gridlines; curve samples asserted within 0.12 dec of waypoints; accuracy ~3% Re / ~2% Cd; trailing cluster Re>9.5e5 cut by the frame not recovered |
| TN 3038 Delany-Sorensen | 400 dpi; piecewise-log cal through ALL 21 x / 15 y detected grid lines (photostat stretch ~1.5%/decade; the strong row at 1213 px is the Wieselsberger dashed curve + data chain, NOT a grid line -- identified by gap-ratio analysis); ring-NCC (R15/s5) + 14 px NMS | 219 | PASS: crisis dip, both hysteresis loops, supercritical rise all covered; dense subcritical chains subsampled by NMS (by design); legend box, NACA logo, V->O inset masked after each produced ghosts; 0 points off the data bands |
| Roshko 1961 | grid-calibrated (8 rows x 28 log columns, decade identity verified via log10(2) spacing); grid stripped; EVERY detection hand-labelled at 5-6x zoom (MANUAL dict in script) | 12 plain (+6 splitter, kept separately, NOT overlaid) | PASS: matches Roshko's text (Cd 0.3->0.7 over 1e6-3.5e6, then ~0.7 plateau); splitter-plate runs and Dryden&Hill '+' excluded; the two lowest-Re 1-atm points (~1e6, Cd 0.3-0.35) are buried in the D&S dashed curves and were NOT recoverable -- noted in JSON |
| Schewe 1983 [via R15 fig4] | vector: paired H+V segments; Spitzer '*' de-aliased (6 glyphs alias as both '+' and 'x'); legend glyph column masked | 50 | PASS: exact vector coordinates; agrees with the independent SG20 replot (above) |
| Achenbach-Heinecke 1981 [via R15 fig4] | vector: single-'re' stroke squares | 31 | PASS (vector-exact) |
| Achenbach 1968 curve [via C03 fig3] | vector polyline vertices, clipped to 6e4-5e6 | 21 vertices | PASS; min Cd 0.311 at 5.4e5 |
| Rodriguez 2015 WRLES | Table 2 TRANSCRIPTION | 6 | exact (tabulated) |
| Catalano 2003 WMLES | vector: 3 blue filled circles; x-cal from detected tick columns | 3 | PASS: extracted (5.00e5,0.347)(1.00e6,0.317)(2.00e6,0.327); the 1e6 point matches their Table 1 CD=0.31 within 0.007 (asserted in-script) |
| Stabnikov-Garbaruk 2020 | 468 px raster, color classification, cross-3x3 erosion; detected inner ticks 1e5/1e6 | SST 11, gamma-Re_theta 12, SST KD 13, Schewe replot 39 | PASS: one gamma-Re_theta marker (~2e5) occluded by an Exp dot, lost; k-omega KD series + right DDES panel deliberately skipped (superseded baseline / scale-resolving hybrid) |
| Stringer 2014 | Re from the stated run matrix; Cd by NCC in +-20 px windows (scan skew +5-8 px absorbed); Re=40 transcribed from Table 3 (CD 1.55/1.55, symbols coincide) | CFX 6, OpenFOAM 5 | PASS with caveats: Re=100 pair overlaps (attribution may be swapped; both 1.2-1.45); OpenFOAM Re=1e5 is ABSENT from their fig. 5 (total Cd ~0.08 below the 0.1 axis floor) and is not invented |
| Henderson 1995 | 300 dpi scan; inward-tick cal; pressure/viscous split at Cd=0.8 (bands disjoint); fill+7x7-opening detaches the dotted fit lines; totals = paired sums | 15 totals (6 steady, 9 shedding; 41 component symbols) | PASS: shedding mean at Re=99 -> Cd 1.359 (textbook ~1.35); steady branch extends past onset to Re~55 (his bistable window), as in the paper; 2 leftmost steady pts (Re~24) + one crossover pair not separated -- noted |
| Tritton/Finn/Jayaweera [via V&G fig7] | vector; y=Cd*R/4pi converted, R<=0.05 dropped; inset+legend masked | 48+50+20 | PASS (vector-exact); Re 0.05-6 -- enters the figure only when the axis reaches it |
| D&S + Wieselsberger crosschecks [via R15 fig4] | vector | 150 / 13 | D&S: primary-vs-secondary mean dCd -0.012 (rms 0.065 incl. branch mismatch) -- validates the TN 3038 digitization. Wieselsberger: mean +0.19 -- NOT comparable (R15 cites the 1922 series, and matching across the crisis drop is ill-posed); kept for the record only |

All raster datasets have green(kept)/red(rejected) check PNGs under
`litdata/dragcrisis/checks/`; vector datasets have circled-overlay check
PNGs from the render. Nothing was accepted without eyeballing its check.

## 3. Figure changes (regen_dragcrisis_cd_re.py)

- Our family: colors/markers/linewidths untouched; **cold two-stage
  diamonds removed** (user directive; still written to the JSON dump for
  the appendix table); **limit-cycle bands kept but have NO legend
  entry** (caption text below); legend reduced to the two ladder entries
  (top right) + an 11-entry literature legend (bottom left, 8 pt).
- Axis: x window now Re 1e4-1.2e7 (auto-widens with the campaign:
  min(1e4, 0.8*Re_min) .. max(1.2e7, 1.3*Re_max)); y 0.1-1.52 linear,
  0.2 ticks; `--logy` flag ready for the Re 1-1e7 span; decade LogLocator
  replaces the old fixed x-tick list.
- **Guard against the RUNNING extension campaign** (HANDOVER rule 4):
  the live matrix_summary.jsonl already carries 36 in-progress rows
  (Re 1..2e7); the script now defaults to `--re-window 6e4:2e6` (the
  completed 84-case matrix), prints a notice, and refuses to refresh
  data/dragcrisis_matrix_summary.jsonl until run with
  `--re-window full`. Rerun with `full` when the CFD agent declares the
  extension harvested -- the low-Re literature (Henderson, Tritton/Finn/
  Jayaweera, Stringer Re<=1e3, TN-84 curve from Re 4.2) is already
  loaded and will enter the frame automatically.
- Literature styling: experiments = gray/black open symbols per source +
  thin gray lines; scale-resolving = green (WRLES filled triangles,
  WMLES open stars); transition-RANS = green thin dotted-marker line;
  fully-turbulent RANS = charcoal x / dotted line. Exactly ONE chromatic
  hue added (#3f8a4f), validated with the dataviz six-checks validator
  against the Tu trio (adjacent normal-vision dE=18; the orange-green
  protan pair is 6.0, inside the 6-8 band that is legal with secondary
  encoding -- here marker shape, line weight, alpha and the legend).
  The trio's own blue-purple all-pairs floor (9.6) is pre-existing and
  out of scope.
- Curation cuts (crowding rule from the brief): SST KD sweep (second
  transition-RANS series -- digitized, in the JSON, not plotted);
  SG20 k-omega KD + DDES panel (not digitized: superseded baseline /
  hybrid); Roshko splitter-plate points (physically different
  configuration; separate JSON series); IOP Schewe replot (duplicate of
  the vector Schewe; kept as cross-check only).

## 4. DRAFT caption amendment (for the user to integrate; I did not touch sa-ai.tex)

> Steady-RANS drag ladders for the three freestream seeds (up-ladder:
> filled, solid; dn-ladder: open, dashed), overlaid on the experimental
> and computational literature. Cases whose convergence monitor flagged
> a steady limit cycle carry a capped vertical band spanning the
> tail-window $C_d$ min--max. Gray/black open symbols are experiments,
> digitized independently from the sources: Wieselsberger's nine-
> diameter composite~\cite{...TN-84/PhysZ...}, Delany \&
> Sorensen~\cite{...TN3038...}, Roshko's pressurized-tunnel points
> (splitter-plate runs excluded)~\cite{roshko1961}, Schewe and
> Achenbach \& Heinecke as replotted by Rodr\'iguez et
> al.~\cite{...}, and the Achenbach drag curve as reproduced by
> Catalano et al.~\cite{...}. Green marks are scale-resolving
> computations (wall-resolved LES of Rodr\'iguez et al., table
> values; wall-modeled LES of Catalano et al.); the green line is the
> SST $\gamma$--$Re_\theta$ transition-model URANS sweep of Stabnikov
> \& Garbaruk; charcoal crosses and dots are fully-turbulent SST URANS
> (Stringer et al.; Stabnikov \& Garbaruk). The steady solutions
> reported here lie below the shedding-mean experimental band at
> subcritical $Re$ by construction; the comparison targets the crisis
> location, its freestream-seed shift, and the supercritical level.
> Digitization provenance: repro/cfd/litdata/dragcrisis/.

(Note for integration: the last-but-one sentence anticipates the
standing honesty framing from the 0117 record, Sec. 2b -- adjust to the
section's existing language; bib keys to be filled from references.bib,
zheng_lei_2016 is NOT cited by this figure.)

## 5. Loose ends / for the next agent

- `litdata/dragcrisis/` PDFs, `src/` rasters and `checks/` PNGs are on
  disk but NOT committed (repo hygiene; README documents re-acquisition
  and the script regenerates the rest). JSONs + README + scripts + the
  figure ARE committed.
- When the extension campaign completes: rerun
  `digitize`-nothing; just `regen_dragcrisis_cd_re.py --re-window full`
  (and consider `--logy`; Tritton/Henderson/TN-84-curve then anchor the
  laminar end; the 47<Re<1e3 steady-branch caveat of the 0117 record
  applies to OUR points there, not to the literature marks).
- Cheng et al. 2017 WRLES remains unsourced; if someone obtains the
  JFM PDF, its Cd values are tabulated and slot into the green class
  next to Rodriguez.
- The IOP SST-KD sweep is digitized and available should the text ever
  discuss algebraic transition models.
