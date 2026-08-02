# Stock (2006) COMPUTED transition lines, mechanism-coded — Figs. 15a, 14c, 17a

*2026-08-02, paper agent, on 019-v100-dev. The missing third leg of the
three-way comparison (our SA-AI run / Stock's computation / the experiment).
Companion to `2026-08-01-stock-fig14c-16c-digitization.md`, which covers the
measured symbols and whose section A9 sets the cross-figure accuracy floor
used here. Nothing on 014/017 touched, no `.tex` edited, nothing staged or
committed, no point fabricated.*

## Deliverables

New generator: `paper/repro/cfd/digitize_stock_computed_fronts.py`
(re-runnable, byte-idempotent, self-documenting; imports the calibration and
raster-extraction of `digitize_stock_fig14c.py` so the whole Stock family
stays one pipeline).

It writes a `stock_computed_lines` block into each existing per-figure JSON,
leaving every pre-existing key untouched (verified: Fig. 15a still carries
its legacy `stock_computed_separation_line`, `note_circles`, etc.):

| file | new block contents |
|---|---|
| `data/stock2006_fig15a_digitized.json` | `front_re6p56e6` (126 pts), `separation_line_short_dashed` (108 pts) |
| `data/stock2006_fig14c_digitized.json` | `front_re6p49e6` (112), `separation_line_short_dashed` (57), `front_re1p52e6` (41), `front_re6p49e6_piece2` (10) |
| `data/stock2006_fig17a_digitized.json` | `front_re6p56e6` (104), `front_re18p32e6` (81) |

Each block carries an ordered `points` list of `{phi_deg, xL, mechanism}`,
a collapsed `mechanism_segments` list with the azimuth boundaries, the
measured `median_dash_px` / `median_gap_px`, `known_arc_frac`,
`ambiguous_arc_frac`, and `arc_px`. Check overlays (rendered and read back):
`$DIG/check_fig{15a,14c,17a}_computed.png` (+ `_half`).

**Fig. 14a and 14b were deliberately not run** — see §6.

## 1. How the styles were discriminated (measured, not eyeballed)

Stock's key, Sec. III.C verbatim: pure TS = **continuous**, pure CF = **long
dashed**, both waves = **midsized dashed**, separation = **short dashed**;
the captions add streamlines = thin continuous. So two independent features
are needed, and both are measured.

**(a) Line width, to keep the ~20 streamlines out.** The rasters are
2x-pixel-duplicated bitmaps, so the distance transform of the ink is
quantised and cleanly separated:

```
Fig. 15a dt histogram (px):  1.0:108564   2.0:57745   2.5:4422
                             3.0:26189    3.5:3026    4.0:7414   >=5:1370
```

dt≈1 is a single streamline (2-3 px wide), dt≈2 is **two streamlines that
touch** (4 px), dt≈3-4 is a computed curve (6-8 px). Dash bodies are taken
as `dt >= 2.0` components that contain at least 3 px of `dt >= 3.0` core:
that admits genuine thick ink and rejects streamline bundles. Measured
median thickness of the accepted components is 6.3 px, i.e. exactly Stock's
heavy line weight. **No streamline segment entered any emitted curve** — the
check overlays show every traced point sitting on a heavy line, and all
thick ink that ended up in no curve is painted magenta (see §5).

**(b) Dash period and duty cycle along the traced centreline.** Each dash
component is reduced to a *medial polyline* (binned centroids along its
principal axis, so a sharply curved solid segment is followed rather than
chorded), the components are chained by endpoint proximity + tangent
continuity, and the resulting centreline is then walked at 1-px arc-length
steps to read ink occupancy. That gives the ink-run and gap arc lengths
*along the curve*. Measured on Fig. 15a, the only panel carrying all four
styles:

| style | mechanism | dash (px) | gap (px) | dash/panel width |
|---|---|---|---|---|
| short dashed | separation | 14 | 3 | 0.0084 |
| midsized dashed | TS+CF | 31 | 4 | 0.0187 |
| long dashed | CF | 64 | 11 | 0.0386 |
| continuous | TS | 239 (no gaps) | — | 0.144 |

Ratios 2.2x and 2.1x between adjacent dash modes — the four modes are
cleanly separated, which is why this is worth doing at all. Templates are
stored as fractions of panel width so they transfer between rasters of
slightly different dpi.

**Both features are needed.** Dash length alone fails, because streamlines
bridge dash gaps and merge two midsized dashes into one 66-px run that mimics
a long dash. The *gap* length is immune to bridging (bridging removes gaps,
it does not lengthen the survivors), and it is what separates CF (gap 11 px)
from short/midsized (gap 3-4 px, mutually inseparable by gap). A first
attempt that used dash length only mislabelled the whole Fig. 15a CF region;
the two-feature version reproduces Stock's text exactly (§3).

**The fit.** Rather than classify each run independently, a 4-state
piecewise-constant model is fitted over the run sequence by dynamic
programming (Viterbi): the cost of style *s* for a run of length *L* with
adjacent gap *g* is `|L - (m·d_s + (m-1)·g_s)|/d_s + 0.45·(m-1) +
1.5·|g - g_s|/g_s`, minimised over merge multiplicity *m* ≤ 8, plus a 1.3
penalty per style change. Merges are therefore explained rather than
mislabelled (the fit recovers m = 2, 3, 4 and even 6 merged short dashes on
the Fig. 15a separation line). Confidence per run is the extra **total path
cost** of forcing any other style there (forward-backward, not a local
comparison); runs with margin < 0.5 are reported `ambiguous:<label>?` and
counted in `ambiguous_arc_frac`. A curve with fewer than 3 ink runs cannot
establish a period at all and is marked wholly ambiguous by construction.

**Erased regions are UNKNOWN, not gaps.** The 5 X grid columns, 7 phi grid
rows, the legend block and the measured-symbol boxes (positions read from
this repo's own committed symbol digitizations, so they cannot drift) are
erased before tracing. Samples inside them are dropped from the dash/gap
statistics and the interrupted run is stitched; a chain jump is allowed to
be twice as long when the straight path between the two dash ends runs
through an erasure. `known_arc_frac` (0.77-0.88) reports how much of each
curve was measurable.

## 2. Calibration

Re-detected inside every panel, nothing inherited (max X-fit residual, px):
Fig. 15a `287.51/700.27/1117.71/1531.23/1947.35`, resid 1.48; Fig. 14c
`304.99/728.28/1155.57/1579.37/2003.72`, resid 1.25; Fig. 17a
`320.05/738.58/1161.00/1580.15/2001.47`, resid 1.23. phi frame rows and the
independent interior grid-row check are as in the 2026-08-01 record.
Accuracy carried forward: **±0.0025 x/L within a figure, plus ~0.005 x/L
across a Ref.49/Ref.50 figure pair**. Mechanism *boundary* azimuths are
localized to about one dash period, i.e. **~2° for short dashes and ~8° for
long dashes** — quote them no tighter than that.

## 3. Fig. 15a — α = 10°, Göttingen (priority 1)

**Re = 6.56e6 front** (`front_re6p56e6`, arc 1553 px, 30 dashes, 26 ink
runs, median dash 34 px, median gap 3 px, known 0.80, **ambiguous arc
fraction 0.126**):

| mechanism | phi (deg) | x/L |
|---|---|---|
| ambiguous:CF? | 177.80 → 173.02 | 0.3363 → 0.3214 |
| **TS** | 171.65 → 144.97 | 0.3132 → 0.2127 |
| **TS+CF** | 144.25 → 94.72 | 0.2109 → 0.2046 |
| **CF** | 92.95 → 62.74 | 0.2075 → 0.3407 |
| ambiguous:CF? | 62.29 → 56.45 | 0.3478 → 0.4294 |
| **TS+CF** | 56.07 → 1.96 | 0.4328 → 0.5698 |

Compare Stock's own words for this panel: *"Close to the symmetry planes,
pure TS waves dominate, followed in both directions by simultaneous TS and
CF waves, which produced transition. In the middle part of the prolate
spheroid, pure CF wave triggering is present."* The recovered sequence
TS → TS+CF → CF → TS+CF is exactly that, with the **CF core spanning
phi ≈ 93°→57°** and the **TS cap spanning phi ≈ 172°→145°**. The only
element of his sentence not recovered is a *second* TS cap near the windward
plane: my trace reads TS+CF all the way to phi ≈ 2°. The windward end is a
separate chain piece whose 102-px gap-free run is below the continuous-style
threshold (0.075 of panel width = 124 px), so it stays TS+CF/ambiguous rather
than being promoted to TS. **Treat "TS near the windward plane below
phi ≈ 10°" as Stock's text, not as my measurement.**

**Free vortex-layer separation line** (`separation_line_short_dashed`, arc
1514 px, 66 dashes, 70 runs, median dash 14 px, gap 3 px, **ambiguous 0.077**):
uniform short dash from phi 129.23 (x/L 0.2812) to phi 1.73 (x/L 0.9731).
For Re = 1.52e6 this *is* Stock's predicted front (the flow stays laminar to
separation). The single ambiguous stretch (phi 92.8→87.5) is one run of six
merged short dashes.

**Independent validation.** This panel already held a legacy 14-point
`stock_computed_separation_line` (digitized 2026-07-26, stated ±0.01 x/L).
My 108-point trace agrees with it point by point: mean +0.0019, **rms 0.0105,
max 0.023 x/L**, i.e. inside the legacy pass's own error bar, with the
largest deviations where the legacy 14-point sampling chords across the
curve's sharpest bend (phi ≈ 105°). Both are kept; the legacy key is
untouched.

## 4. Mechanism at each measured azimuth (what the residual study needs)

Stock's computed front vs his own measured symbols, at the symbol azimuths:

| Fig / Re | phi | measured x/L | Stock computed x/L | measured − Stock | mechanism |
|---|---|---|---|---|---|
| 15a 6.56e6 | 164.90 | 0.3095 | 0.2791 | +0.0304 | **TS** |
| 15a 6.56e6 | 59.13 | 0.3071 | 0.3908 | −0.0837 | **CF** (ambiguous-flagged) |
| 15a 6.56e6 | 40.39 | 0.3944 | 0.5467 | −0.1523 | TS+CF |
| 15a 6.56e6 | 29.04 | 0.4793 | 0.6074 | −0.1281 | TS+CF |
| 15a 6.56e6 | 20.21 | 0.5643 | 0.6026 | −0.0383 | TS+CF |
| 14c 6.49e6 | 159.89 | 0.5651 | 0.5504 | +0.0147 | TS |
| 14c 6.49e6 | 142.35 | 0.4798 | 0.4282 | +0.0516 | TS |
| 14c 6.49e6 | 36.22 | 0.4815 | 0.5267 | −0.0452 | TS |
| 14c 6.49e6 | 10.35 | 0.5669 | 0.4921 | +0.0748 | TS |
| 14c 1.52e6 | 160.78 | 0.9349 | 0.9200 | +0.0149 | TS (streamline-12 island) |
| 14c 1.52e6 | 145.36 | 0.7372 | 0.8234 | −0.0862 | TS (island) |
| 17a 6.56e6 | 161.64 | 0.3137 | 0.2991 | +0.0146 | TS |
| 17a 6.56e6 | 120.93 | 0.2263 | 0.1958 | +0.0305 | TS+CF |
| 17a 6.56e6 | 48.81 | 0.3149 | 0.4851 | −0.1702 | TS+CF |
| 17a 6.56e6 | 30.25 | 0.4017 | 0.6073 | −0.2056 | TS+CF |
| 17a 18.32e6 | 145.03 | 0.1416 | 0.1072 | +0.0344 | TS+CF |
| 17a 18.32e6 | 42.57 | 0.1428 | 0.1704 | −0.0276 | TS+CF |

**Directly answering the coordinator's question:** at phi = 164.9° (the
best-agreement azimuth, our residual +0.068) Stock's mechanism is **pure TS**
— confirmed, that azimuth sits 7° inside a 27°-long continuous segment. At
phi = 59.1° (our worst, +0.623) it is **CF** — the run there is a two-dash
merge whose confidence margin (0.48) falls just under my 0.50 ambiguity cut,
so it is reported `ambiguous:CF?`; it is bounded on the leeward side by four
unambiguous 64-px CF dashes and its own de-merged dash length is 65 px, so CF
is the reading, but it is the weakest link in the chain and is flagged as
such. So the mechanism map does support the impression — but note from the
table above that **Stock's own computation is also worst in the same region**
(his front is 0.084-0.152 x/L off his own measurements at phi 59-40, versus
0.015-0.052 in the TS regions). Whatever is happening at those azimuths
degrades the e^N prediction too, which is a more interesting statement than
"our residual tracks the mechanism".

## 5. Fig. 14c and Fig. 17a

**Fig. 14c (α = 5°).** `front_re6p49e6` (arc 1342, 19 runs, median dash 65 px,
ambiguous 0.051): **TS** phi 158.34→140.61 (x/L 0.5504→0.4212), **TS+CF**
140.27→48.03 (0.4199→0.5270), ambiguous 47.84→37.69, **TS** 37.18→2.37
(0.5267→0.4321). Stock's text: *"transition is triggered by TS waves near
the windward and leeward symmetry planes and for the remaining part of the
body surface simultaneously by TS and CF waves"* — recovered exactly,
including **both** TS caps this time.
`separation_line_short_dashed`: uniform short dash, phi 137.36→3.59,
x/L 0.9263→0.9399, ambiguous 0.000 — this is the Re = 1.52e6 predicted front.
`front_re1p52e6` (arc 500, TS phi 158.18→131.57, x/L 0.9200→0.7206) is the
**streamline-12 TS island**: Stock, *"the flow is predicted to remain laminar
up to separation for the small-Reynolds-number case, except for streamline 12,
where transition is provoked by TS waves"*. So for Re = 1.52e6 the front is
the separation line **plus** this island; the JSON name is per-Reynolds-number,
not "the whole front", and should be read with the `mechanism_segments`.
`front_re6p49e6_piece2` is the near-vertical leeward cap (phi 175.9→163.8);
with only 2 ink runs it carries no mechanism claim (ambiguous 1.000).

**Fig. 17a (α = 10°, ONERA F1).** `front_re6p56e6` (arc 1456, ambiguous
0.292): **TS** 159.13→133.91, **TS+CF** 133.58→99.97, ambiguous 99.64→55.49,
**TS+CF** 55.31→7.40. `front_re18p32e6` (arc 1009, ambiguous 0.080):
**TS+CF** 177.77→103.85, **CF** 101.89→64.09, ambiguous, **TS+CF**
51.45→21.30. No short-dashed curve is present in this panel, so there is no
separation line to report for 17a. Note 17a's higher ambiguous fraction
(0.292 on the 6.56e6 front): that curve runs at a shallow angle to the
streamlines over phi 100→55, which merges dashes heavily.

**Reynolds-number assignment** (the coordinator's "say how you told them
apart"): the short-dashed curve is named by style. Each remaining curve is
assigned to the measured symbol set it sits closest to, averaged over
azimuth-matched symbols: Fig. 15a curve0 → 6.56e6 (mean |Δx/L| 0.086, 5
symbols; the only other set is 1.52e6, whose front is the separation line);
Fig. 14c curve0 → 6.49e6 (0.046, 4), curve2 → 1.52e6 (0.045, 5); Fig. 17a
curve0 → 6.56e6 (0.086, 5), curve1 → 18.32e6 (0.030, 2). Every assignment is
logged in the JSON under `curve_naming`. The Fig. 17a and 15a assignments are
also forced by ordering (the higher Re transitions upstream everywhere), which
is an independent check.

## 6. What I could NOT establish, and what is missing

* **Un-traced thick ink**, reported per panel and painted **magenta** in the
  check overlays: Fig. 15a **6.4%** (549 of 8521 thick px, 8 small fragments,
  largest 30 px), Fig. 17a **6.2%** (2 fragments, 75 px and 52 px, at the
  leeward and windward panel edges), Fig. 14c **21.1%** (2512 of 11898 px).
  Fig. 14c's figure is dominated by **one 302-px component**: the *lower*
  branch of the streamline-12 TS wedge, which terminates **on** the
  separation line. Chaining it requires an unrestricted corner merge, and
  when I allowed that, the merge spliced the wedge into the separation line
  and produced a nonsensical polyline mixing two mechanisms. I reverted to
  the conservative tail→head cusp rule and left the branch un-traced rather
  than emit a wrong curve. **Fig. 14c's 1.52e6 TS island is therefore only
  half traced** (upper branch only) — the omission is visible in the overlay
  and quantified in `stock_computed_lines_untraced_px`.
* **Fig. 14a (α = 0) and 14b (α = 2.5) were not run.** Both are already
  digitized: their JSONs carry `computed_ts_front` from
  `digitize_stock_fig14a.py` / `_fig14b.py`, and Sec. III.C states transition
  at α = 0 and 2.5 is provoked **solely by TS waves**, so each line is one
  mechanism and there is nothing to segment. Independently, my generic
  in-panel calibrator *refuses* panel 14a: at α = 0 the streamlines are
  horizontal, so the full-width-dark-row test that locates the phi grid rows
  latches onto streamlines instead and the assertion fires
  (`phi grid row at 137.1 -> 165.55 deg, not a 30-multiple`). Forcing a
  second calibration path for zero new information was not worth the risk.
  The `skip` reason is recorded in the script's own PANELS table.
* **The windward TS cap of the Fig. 15a 6.56e6 front** (§3) is not measured;
  its gap-free run is 102 px against a 124-px threshold.
* **Absolute dash lengths are not physical lengths.** The `dt >= 2.0` erosion
  and the ±1-px occupancy probe shift measured dash/gap lengths by a couple
  of px each; only ratios matter, and the style templates were calibrated on
  the same measure. Do not read `median_dash_px` as Stock's plotted dash
  length.
* **Mechanism boundary azimuths are good to about one dash period** (~2° in
  short-dash regions, ~8° in long-dash regions). The x/L values themselves
  keep the ±0.0025 (within-figure) budget.
