# fig:dragcrisiscd overlay, round 2: experiment/computation semantic + low-Re unsteady DNS lines
*2026-07-28 13:21, digitization/figure agent. Companion to
2026-07-28-0317-dragcrisis-overlay.md (round-1 sources and digitization QA,
all still valid). USER DIRECTIVE: strict visual semantic -- EXPERIMENTS =
symbols WITHOUT connecting lines, COMPUTATIONS = lines WITHOUT symbols,
SA-AI family untouched (the subject) -- plus unsteady 2-D DNS lines in the
shedding band Re ~47-1000 so the figure shows computations matching the
experiments exactly where our steady branch knowingly sits below.*

## 1. Restyle (regen_dragcrisis_cd_re.py; SA-AI data byte-identical)

- **Experiments -> symbols only.** Delany-Sorensen / Roshko / Schewe [dig.
  R15] / Achenbach-Heinecke [R15] / TN-84 symbol series / Tritton-Finn-
  Jayaweera were already line-free (unchanged). The two digitized FAIRED
  CURVES are now rendered as sparse symbols resampled along their
  polylines (new `logsample()`: interp in log-log = on the chords as
  drawn, endpoints + the Cd-minimum vertex always retained, so the crisis
  dip cannot be interpolated away): Wieselsberger TN-84 curve (Re
  4.2-3e3) at 0.10 dec with the same filled-dot glyph as its symbol
  series; Achenbach-1968 curve [dig. C03] (6e4-5e6) at 0.08 dec as open
  gray circles -- it is an experiment and no longer reads as a computation
  (legend relabeled 'Achenbach 1968 [dig. C03]', was '... curve').
- **Computations -> lines only.** WRLES Rodriguez (6-pt green solid,
  lw 1.8), WMLES Catalano (3-pt green dashed, lw 1.8), SST gamma-Re_theta
  SG20 (green dash-dot; marker dots removed), fully-turbulent SST = SG20
  sweep + Stringer CFX + Stringer OpenFOAM all charcoal dotted (x markers
  removed; one class legend entry as before).
- **SA-AI family untouched**: same Tu trio colors/markers/linewidths/
  limit-cycle bands; `data/dragcrisis_cd_re_computed.json` and
  `data/dragcrisis_matrix_summary.jsonl` md5-identical before/after
  (a814dae8... / 804a3f4d...), 129 cases, 35 limit-cycle bands.
- Palette unchanged -- still exactly one added chromatic hue (#3f8a4f,
  round-1 dataviz validation carries over); identity within classes by
  marker shape (experiments) / linestyle+weight (computations) + legend.
- Within-green linestyles repeat across the two DISJOINT Re bands
  (shedding band <=1e3 vs crisis band >=5e4): solid = Henderson / WRLES,
  dashed = Qu / WMLES; unambiguous spatially, unique in the legend.

## 2. New computation series (shedding band + 3-D anchor)

| dataset | how | pts | QA |
|---|---|---|---|
| henderson1995 shedding branch (PROMOTED, digitized in round 1) | already in litdata; now plotted as green solid line, `branch=='shedding'` filter (new `where=` arg to `lit()`) | 9 (Re 64-986) | round-1 check PNG (checks/check_henderson.png); shedding mean at Re=99 -> 1.359 vs textbook ~1.35. Steady-branch points (Re 25-55) deliberately NOT plotted |
| qu2013_dns (NEW) | Qu, Norberg, Davidson, Peng & Wang, JFS 39 (2013) 347-370, author ms. via Chalmers CPL (green OA); TABLE 3 TRANSCRIPTION (tables beat digitization), every CD value asserted verbatim against the PDF text in-script (`_assert_in_pdf`) | 8 plotted (Re 50-200; 9 rows -- the Re=150 domain pair H=100/H=160 kept in JSON, H=160 plotted) | primary, tabulated; caveat recorded: their Table 2 domain study bounds H-dependence <1% at Re=100 |
| dong_karniadakis2005_dns3d (NEW, the bonus 3-D anchor) | Dong & Karniadakis, JFS 20 (2005) 519-531, open PDF from author's Purdue page; TABLE 2 TRANSCRIPTION, text-asserted (this PDF's font typesets the decimal point as a colon, '1:143' -- assert handles it) | 4 resolved cases (Nz>=64) at Re=1e4: Cd 1.110-1.143, finest DNS-B3 = 1.143 (= Wieselsberger's 1.143 row in their own table) | primary, tabulated; rendered as a capped vertical green TICK spanning the resolution study -- a line, not a symbol, honoring the semantic; coarse-spanwise Nz<=32 cases (1.155, 1.208) excluded from the tick, kept in JSON |

No raster digitization this round -> no new check PNGs; the transcription
QA is the in-script verbatim text asserts (both ran green). README.md
provenance table extended (2 PDF rows, 2 JSON rows). PDFs on disk, not
committed (repo hygiene, as round 1).

Sources tried and NOT obtainable open: Posdziech & Grundmann JFS 23
(2007) (Elsevier paywall, no TU-Dresden/Qucosa copy found) -- its Re<=250
sweep class is covered by Qu et al.; Park/Kwon/Choi KSME 12 (1998)
(Springer paywall); Kravchenko & Moin PoF 12 (2000) Re=3900 (AIP paywall)
-- the 3-D anchor class is instead Dong & Karniadakis Re=1e4 (open,
tabulated, finer provenance than digitizing a secondary Cd~1.04 quote).

## 3. Regeneration + the ultra-campaign guard

An ULTRA campaign (Re 2e7..1e10, mesh 'ultra') is RUNNING and has
appended 25 rows to the live matrix_summary.jsonl since the figure
commit (129 committed rows -> 154 live; first 129 verified identical).
Per HANDOVER rule 4 the script now DROPS mesh=='ultra' rows by default
(new `--include-ultra` flag for when that campaign is declared
harvested) and refuses to refresh data/dragcrisis_matrix_summary.jsonl
while any rows were dropped. Regenerated with the full committed window:

    python3 repro/cfd/regen_dragcrisis_cd_re.py --re-window full --logy

(the exact committed-figure invocation; both data JSONs byte-identical
afterwards, only styling + literature additions changed).

## 4. Artifacts (all absolute)

- figure: /home/qiqi/flexcompute/sa-ai/paper/figs/dragcrisis_cd_re.pdf
  (shared by sa-ai.tex fig:dragcrisiscd and whitepaper.tex
  f:dragcrisiscd -- whitepaper inherits automatically)
- preview: /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/dragcrisis_cd_re.png
- scripts: /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/regen_dragcrisis_cd_re.py,
  /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/digitize_dragcrisis_lit.py
- new data: /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/litdata/dragcrisis/qu2013_dns.json,
  /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/litdata/dragcrisis/dong_karniadakis2005_dns3d.json
- provenance: /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/litdata/dragcrisis/README.md
- acquired PDFs (on disk, uncommitted): .../litdata/dragcrisis/qu2013_jfs.pdf,
  .../litdata/dragcrisis/dong_karniadakis2005_jfs.pdf

## 5. DRAFT caption amendment (fig:dragcrisiscd; I did NOT touch any tex)

Replace the caption's styling sentences (from "Gray and black open
symbols are experiments..." through "...fully-turbulent SST
URANS~\cite{stringer_2014,stabnikov_garbaruk_2020}.") with:

> Symbols are experiments; lines are computations. Gray and black
> symbols, digitized independently from the sources: Wieselsberger's
> nine-diameter composite~\cite{wieselsberger_1921} (faired low-$Re_D$
> curve rendered as sparse symbols along its polyline), Delany \&
> Sorensen~\cite{delany_sorensen_1953}, Roshko's pressurized-tunnel
> points (splitter-plate runs excluded)~\cite{roshko_1961},
> Schewe~\cite{schewe_1983} and Achenbach \&
> Heinecke~\cite{achenbach_heinecke_1981} as replotted by Rodr\'iguez
> et al.~\cite{rodriguez_2015}, the Achenbach drag
> curve~\cite{achenbach_1968} as reproduced by Catalano et
> al.~\cite{catalano_2003} (likewise sparse symbols), and the low-$Re_D$
> laminar drag collected by Veysey \&
> Goldenfeld~\cite{veysey_goldenfeld_2007}. Green lines are
> transition-resolving computations. In the shedding band the unsteady
> two-dimensional shedding-mean drag---Henderson's spectral-element
> curve~\cite{henderson_1995} (solid) and the DNS sweep of Qu et
> al.~\cite{qu_2013} (dashed)---recovers the measured level exactly
> where the steady branch reported here sits below it by construction.
> At the crisis: wall-resolved LES~\cite{rodriguez_2015} (solid),
> wall-modeled LES~\cite{catalano_2003} (dashed), and the SST
> $\gamma$--$Re_\theta$ transition URANS sweep of Stabnikov \&
> Garbaruk~\cite{stabnikov_garbaruk_2020} (dash-dotted); the green tick
> at $Re_D=10^4$ spans the resolution study of the three-dimensional
> spectral DNS of Dong \& Karniadakis~\cite{dong_karniadakis_2005}.
> Charcoal dotted lines are fully-turbulent SST
> URANS~\cite{stringer_2014,stabnikov_garbaruk_2020}.

All other caption sentences (ladders/seams, continuity-not-validation,
limit-cycle bands, "steady solutions lie below the shedding-mean
band... comparison targets the crisis location...", table pointer,
provenance pointer) stay exactly as committed. Note: the caption's
existing source list never named the Tritton/Finn/Jayaweera-Mason
symbols that entered when the window widened to Re=1 -- the draft above
closes that gap with the single V\&G citation (the actual digitization
source; the three primaries can be named in text if preferred).

New bib entries needed (Crossref-verified fields; henderson_1995
already in references.bib):

    @article{qu_2013,
      author  = {Qu, Lixia and Norberg, Christoffer and Davidson, Lars
                 and Peng, Shia-Hui and Wang, Fujun},
      title   = {Quantitative Numerical Analysis of Flow Past a Circular
                 Cylinder at {Reynolds} Number Between 50 and 200},
      journal = {Journal of Fluids and Structures},
      volume  = {39},
      pages   = {347--370},
      year    = {2013},
      doi     = {10.1016/j.jfluidstructs.2013.02.007},
    }

    @article{dong_karniadakis_2005,
      author  = {Dong, S. and Karniadakis, G. E.},
      title   = {{DNS} of Flow Past a Stationary and Oscillating
                 Cylinder at {$Re=10\,000$}},
      journal = {Journal of Fluids and Structures},
      volume  = {20},
      number  = {4},
      pages   = {519--531},
      year    = {2005},
      doi     = {10.1016/j.jfluidstructs.2005.02.004},
    }

    @article{veysey_goldenfeld_2007,
      author  = {Veysey, John and Goldenfeld, Nigel},
      title   = {Simple Viscous Flows: From Boundary Layers to the
                 Renormalization Group},
      journal = {Reviews of Modern Physics},
      volume  = {79},
      number  = {3},
      pages   = {883--927},
      year    = {2007},
      doi     = {10.1103/RevModPhys.79.883},
    }

Whitepaper: NO wording change required -- its pointer text (traverse/
concessions block before f:dragcrisiscd) never describes the overlay
styling, and concession 3 ("the subcritical plateau sits well below
measurement, as any steady symmetric wake solution must") is only
STRENGTHENED by the new DNS lines. Optional one-liner if the user wants
the semantic stated there too: "In Figure~\ref{f:dragcrisiscd}, symbols
are experiments and lines are computations; the unsteady 2-D DNS lines
recover the shedding-mean band the steady branch sits below."

## 6. Loose ends

- When the ULTRA campaign is harvested: rerun with `--re-window full
  --logy --include-ultra` (the guard prints exactly this reminder).
- The round-1 loose ends stand (Cheng et al. 2017 WRLES unsourced; IOP
  SST-KD sweep digitized but unplotted).
