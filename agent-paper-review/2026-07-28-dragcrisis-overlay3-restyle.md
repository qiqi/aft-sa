# fig:dragcrisiscd restyle, round 3: strict lines/symbols semantic + transparency
*2026-07-28, figure agent. Companion to 2026-07-28-1321-dragcrisis-overlay2.md
(round-2 sources, digitization QA, and the first pass at the
symbols=experiments / lines=computations semantic -- all still valid). This
round applies the user's exact refinements. Styling only: the SA-AI data is
byte-identical (md5 verified).*

## What changed, per series

- **SA-AI family -> LINE-ONLY.** Removed the up-ladder filled circles and
  dn-ladder open squares. Kept the up=solid / dn=dashed distinction and the
  per-Tu seed colors (blue/orange/purple). The family now stays visually
  distinct from the literature by being the ONLY colored lines (literature
  computations are green/charcoal; experiments are gray/black symbols). The
  legend proxies for up/dn are now plain solid/dashed lines (markers dropped).
  The loop tuple was simplified (`mk_`/`mfc` fields removed) since markers are
  gone; the limit-cycle error-bar bands are unchanged.
- **Experiments -> symbols only (verified).** Wieselsberger (curve-as-sparse-
  symbols + symbol series), Delany-Sorensen, Roshko, Schewe, Achenbach-
  Heinecke, Achenbach-1968-as-symbols, and the low-Re Tritton/Finn/Jayaweera
  all carry `ls='none'`/`lw=0`. No connecting lines. No change needed.
- **Computations -> lines only (verified).** Henderson, Qu 2013, WRLES,
  WMLES, SST gamma-Re_theta, fully-turb SST all lines, no markers.
- **Single-point computation -> SHORT HORIZONTAL LINE STUB.** The Dong-
  Karniadakis Re=1e4 3-D DNS was a *vertical* capped tick spanning the
  resolution study; it is now a short green *horizontal* stub (half-width
  0.035 decade in Re, so ~7% of a decade wide, centered on Re=1e4) drawn at
  the finest-grid value Cd_finest=1.143. This makes even a single-Re point
  read as a line, honoring lines=computations. The Nz>=64 span (1.110-1.143)
  stays recorded in the JSON. (The Catalano WMLES points are already plotted
  as a connected line, so no stub was needed there.)

## Transparency

Every series -- symbols AND lines, literature AND SA-AI -- is now slightly
transparent to de-clutter the crowded figure:
- **SA-AI family: alpha = 0.75** (top of the requested 0.6-0.75 band, so it
  stays readable as the subject).
- **Literature (all symbols + all lines): alpha = 0.62.**
The limit-cycle error-bar bands keep their own (opaque) styling, untouched.
Two module constants `ALPHA_SAAI` / `ALPHA_LIT` drive it; the per-call
`alpha=0.85/0.9` literals (mk/ln dicts, the Schewe `+` series) now reference
them.

## Data integrity

`data/dragcrisis_cd_re_computed.json` md5 a814dae8... and
`data/dragcrisis_matrix_summary.jsonl` md5 804a3f4d... are byte-identical
before and after (129 cases, 35 limit-cycle bands). The live campaign jsonl
has since grown 129 -> 173 rows (38 in-progress ultra-mesh + 6 highre seam-
extension rows, incl. a Re=5e7 cold_highre limit-cycle case whose CSV does
not yet exist); per HANDOVER rule 4 these were excluded by regenerating
against the committed 129-row snapshot (temp root: committed jsonl + symlinks
to the real case trees for the limit-cycle tail min/max). The `--include-
ultra` guard already dropped the ultra rows; the 6 highre rows are new since
round 2 and are the reason a plain `--root /local_data/... --re-window full`
now crashes -- use the committed snapshot until the extension is harvested.

## Caption note (single-point-stub convention) -- for the tex integrator

YES, one sentence needs a wording tweak. The current caption
(sa-ai.tex ~l.2550) says:

> ...; the green tick at $Re_D\!=\!10^4$ spans the resolution study of the
> three-dimensional spectral DNS of Dong \& Karniadakis~\cite{...}.

Suggested replacement (stub, not tick; still names the resolution study):

> ...; the short green stub at $Re_D\!=\!10^4$ marks the finest-grid value
> of the three-dimensional spectral DNS of Dong \&
> Karniadakis~\cite{dong_karniadakis_2005} (its $N_z\!\ge\!64$ resolution
> study brackets $C_d\,1.110$--$1.143$).

No other caption sentence needs to change: experiments=symbols /
computations=lines was already stated in round 2 and still holds; the SA-AI
marker removal is not described in the caption, so it needs no edit; the
transparency is a rendering detail not called out in the caption. (I did NOT
touch any tex.)

## Artifacts (absolute)

- figure: /home/qiqi/flexcompute/sa-ai/paper/figs/dragcrisis_cd_re.pdf
- preview: /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/dragcrisis_cd_re.png
- script: /home/qiqi/flexcompute/sa-ai/paper/repro/cfd/regen_dragcrisis_cd_re.py
