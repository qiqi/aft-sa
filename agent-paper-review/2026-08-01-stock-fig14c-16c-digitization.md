# Stock (2006) Fig. 14c and Fig. 16c — measured transition fronts digitized

*2026-08-01, paper agent, on 019-v100-dev. Two new measured transition-front
datasets for Sec. X, digitized from `paper/spheroid.pdf`
(md5 `4c8254e819a3c648923934c65b1ba092`), following
`paper/tools/README.md` + `paper/ONBOARDING.md` §9/§12 and the precedent
scripts `digitize_stock_fig14a.py` / `_fig14b.py` / `_waterfalls.py`.
Nothing on 014-v100-dev or 017-v100-dev was touched; `paper/sa-ai.tex` and
`paper/whitepaper.tex` were not opened for writing; nothing was staged or
committed.*

## Deliverables

| file | contents |
|---|---|
| `paper/data/stock2006_fig14c_digitized.json` | alpha = 5.0 deg; 8 squares (Re_L = 1.52e6) + 4 circles (Re_L = 6.49e6) |
| `paper/data/stock2006_fig16c_digitized.json` | alpha = 29.7 deg; 12 squares (Re_L = 1.53e6) + 10 circles (Re_L = 6.54e6) |
| `paper/repro/cfd/digitize_stock_fig14c.py` | generator + the shared engine (docstring carries page/image/check-PNG) |
| `paper/repro/cfd/digitize_stock_fig16c.py` | generator for 16c; imports the engine so the two panels cannot drift |

Check artifacts (this machine, this session's scratchpad — the scripts default
`$DIG` to `$STOCK_DIGITIZE_DIR` else `<tmp>/stock_digitize`, deliberately NOT
to `/local_data/...`, which lives on 014):
`/tmp/claude-1006/-home-qiqi-flexcompute/3d1a461e-96df-48fb-8bfc-cdfece2123e6/scratchpad/stock_digitize/`
→ `check_fig14c.png`, `check_fig14c_glyphs.png`, `check_fig16c.png`,
`check_fig16c_glyphs.png` (+ the two extracted rasters).

## 1. Panel identification (measured, not assumed)

Both rasters were located by enumerating every image on every page with
`fitz` and reading its placement rect, then reading the page text.

* **Fig. 14 a/b/c** = page 8, image index 1, 2027x3630 px, placement rect
  x = 308-539 pt (the right column). As the brief said.
* **Fig. 16 a/b/c** = page 9, **image index 0**, 1974x3476 px, rect
  x = 305-542 pt (right column). Page 9 carries TWO figure rasters and the
  other one — index 1, 1970x3538 px, rect x = 48-273 pt (left column) — is
  **Fig. 15**. Getting this backwards would have silently produced an
  alpha = 10/15/20 dataset, so it is asserted in the script by the expected
  pixel dimensions in the filename.

Panel-within-raster identity was then confirmed from printed content, never
from assumed panel order:

| panel | printed legend | matches caption + Sec. III.C |
|---|---|---|
| 14a, 14b | "Re = 7.20 x 10^6" (single Re) | alpha = 0, 2.5 deg |
| **14c** | "Re = 1.52 x 10^6" **and** "Re = 6.49 x 10^6" (two Re) | "The results for alpha = 5.0 deg and Reynolds numbers Re = 1.52e6 and 6.49e6 is shown in Fig. 14c" |
| 16a | "Re = 6.42 x 10^6" (one Re) | alpha = 24 deg |
| 16b | 3.01 / 4.48 / 8.52e6 (three Re, squares+circles+**triangles**) | alpha = 29.5 deg |
| **16c** | "Re = 1.53 x 10^6" **and** "Re = 6.54 x 10^6" (two Re) | "The transition prediction for alpha = 29.7 deg and two Reynolds numbers (Fig. 16c)" |

The two-Re legend is decisive for panel c in both figures (a has one Re, b has
one resp. three). Panel c is the bottom panel in both.

**Reynolds number note for the brief:** the brief said "Re_L ~ 1.5e6" for both.
Fig. 14c prints **1.52e6**, Fig. 16c prints **1.53e6** — Stock quotes each
run's own Reynolds number (Fig. 15a is 1.52e6). Both datasets are keyed by the
printed value, so `measured_re1p52e6_squares` (14c) and
`measured_re1p53e6_squares` (16c).

**Regime confirmed as stated in the brief.** Sec. III.C for 14c: at 1.52e6
"the flow is predicted to remain laminar up to separation ... except for
streamline 12, where transition is provoked by TS waves ... The measured
transition line shows a fairly similar behavior." For 16c: "For the small
Reynolds number, transition is provoked completely by separation." So both
low-Re chains are the same laminar-to-separation family as Fig. 15a. Both
panels DO contain a measured transition front; nothing had to be forced.

## 2. Calibration — tick/frame detected inside panel c (the pass-43 rule)

Detected, not inherited from panels a/b (the panel-c X columns land 3-4 px off
the panel-a fit hardcoded in `digitize_stock_fig14a.py`, i.e. the panels are
independently registered on the scan and reusing panel a's numbers would have
introduced a ~0.004 X/a bias).

**Fig. 14c** (least-squares fit of the five column-coverage half-maximum
centroids to X/a = -1, -0.5, 0, +0.5, +1):

```
X px    304.99  728.28  1155.57  1579.37  2003.72
resid   +0.31   -1.25   +1.19    +0.13    -0.38   px   (max 1.25 px ~ 0.0015 X/a)
phi     row 2369.01 = 180 deg, row 3384.92 = 0 deg   (h = 1015.92 px)
```

**Fig. 16c**:

```
X px    311.50  720.01  1133.22  1541.64  1951.70
resid   +0.29   -1.40   +1.61    -0.18    -0.32   px   (max 1.61 px ~ 0.0020 X/a)
phi     row 2294.68 = 180 deg, row 3278.41 = 0 deg   (h = 983.73 px)
```

Frame rows are the two near-full-width dark rows bounding the panel (their
coverage-weighted centroids). The **independent phi check** is the interior
dashed phi grid rows, detected separately and converted with the frame-row
scale: 14c gives 120.47 and 89.73 deg (nominal 120/90); 16c gives 119.85,
89.75, 59.55, 29.54 (nominal 120/90/60/30). So the phi scale is good to
**0.5 deg**, and the script asserts every detected grid row lands within
1 deg of a 30-deg multiple. (The phi = 150 grid row is not detected in either
panel — its dashes are buried in the streamline bundle; not needed.)

The scale convention is unchanged from the rest of the family:
x/L = (X/a + 1)/2 from the nose, phi = 0 on the **windward** symmetry line
(Stock's convention; the campaign mesh convention is phi = 0 leeward, so
mirror phi -> 180 - phi when overlaying). Independent physical check that
phi = 180 is leeward: in 16c (alpha = 29.7) the measured front is at
x/L = 0.053 at phi = 143 and x/L = 0.935 at phi = 67, i.e. transition is far
upstream at high phi — the crossflow/vortex-dominated leeward side.

**The calibration's strongest verification** is external. Every recovered
X/a lands on the DFVLR hot-film ring ladder, and those stations agree with the
INDEPENDENTLY digitized Fig. 15a squares/circles (different page, different
raster, 300-dpi render, an earlier pass) to **<= 0.002 x/L** on every station
the two figures share — printed at the end of each run:

```
14c: 0.4798->0.4810  0.4815->0.4810  0.5651->0.5646  0.5669->0.5671
     0.7372->0.7392  0.8237->0.8228  0.8243->0.8228  0.8820->0.8823
     0.8831->0.8823  0.9349->0.9342  0.9361->0.9342
16c: 0.3080->0.3089 ... 0.9354->0.9342   (all |delta| <= 0.0019)
```

16c additionally resolves three stations upstream of anything in Fig. 15a
(x/L = 0.0529, 0.1373-0.1376, 0.2211-0.2214); they continue the same
~0.085 x/L (0.17 X/a) ring ladder, which is itself a consistency check.

Quoted accuracy: **+-2 px ~ +-0.0025 x/L and +-0.4 deg**, dominated by scan
bleed (the raster is a 2x-duplicated bitmap) and by the +-1 px spread of the
glyph bounding boxes.

## 3. Symbol extraction

Three stages, none of which reads a curve by eye:

1. **Detection** — local-mean zero-normalized cross-correlation (the
   `digitize_stock_fig14a.py` NCC) against two templates cut from the
   cleanest *isolated* data glyph of each type **in that same panel**
   (tight connected-component bboxes; 14c square = rows 3084-3111,
   cols 1882-1909; 14c circle = 3168-3193, 1110-1136; 16c square =
   2651-2676, 946-972; 16c circle = 3020-3047, 1509-1534), each grown by
   PAD = 3 px of surrounding white space. The padding is what separates a
   glyph from a gridline crossing, and it produces a wide empty threshold
   gap: 14c's 12 true glyphs score **0.58-1.00** against a **0.41**
   false-positive floor; 16c's 22 score **0.59-1.00** against **0.44**.
   Threshold 0.50. (A first attempt using unpadded templates lost one true
   glyph at 0.47 — recorded here because it is exactly the kind of silent
   loss the check PNG is for.)
2. **Localization** — the tight template is slid +-5 px and the
   minimum-XOR-mismatch alignment is kept; the reported centre is that
   alignment's tight-bbox centre. (An earlier draft took the NCC peak of a
   *padded* template as the centre, which was biased by up to 2.5 px because
   the padding was asymmetric. Fixed; that bug is the reason localization and
   detection now use different template crops.)
3. **Identity** (square = low Re vs circle = high Re) — three independent
   raster votes: minimum-XOR total, the ink fraction of the four 5x5 bbox
   corner blocks (a square has ink there, a circle does not), and the sign of
   the observed ink difference over the differential (square-only minus
   circle-only) mask. **No single one of the three is reliable** — a circle is
   nearly inscribed in a square, so a circle model on a square glyph produces
   almost no false ink, and a streamline or dashed grid line crossing a glyph
   can ink a circle's corners. Where all three agree the identity is taken
   automatically (29 of 34 glyphs). Where they split, the glyph **must** be
   listed in the script's `ambiguous` table with the identity read off the
   raster by hand at 5x zoom **and the pixel evidence for it**; the script
   asserts that the automatic and manual sets exactly partition the
   detections, so a re-run cannot silently drop a decision. The five
   hand-resolved glyphs, their evidence, and the votes that split are carried
   into the JSON in each point's `method` field.
4. **Stacked pair** — see §5.

## 4. Results

**Fig. 14c, alpha = 5.0 deg, Re_L = 1.52e6 (squares), 8 points**, in front
order (leeward -> windward):

| phi (deg) | x/L | | phi (deg) | x/L |
|---|---|---|---|---|
| 160.78 | 0.9349 | | 125.34 | 0.7372 |
| 155.99 | 0.8820 | | 100.54 | 0.8243 |
| 150.50 | 0.8237 | | 90.08 | 0.8831 |
| 145.36 | 0.7372 | | 50.93 | 0.9361 |

Range x/L 0.737-0.936 over phi 50.9-160.8. Four hot-film stations, two
azimuths each: the front is the "hook" Stock describes — most upstream at
phi ~ 125-145 (the streamline-12 thickening), moving downstream toward both
symmetry planes.

Re_L = 6.49e6 (circles), 4 points: x/L 0.5651 @ phi 159.89, 0.4798 @ 142.35,
0.4815 @ 36.22, 0.5669 @ 10.35 (two stations, two azimuths each, nearly
mirror-symmetric about phi = 90 as expected at alpha = 5).

**Fig. 16c, alpha = 29.7 deg, Re_L = 1.53e6 (squares), 12 points**:

| phi (deg) | x/L | | phi (deg) | x/L |
|---|---|---|---|---|
| 143.26 | 0.0529 | | 99.34 | 0.5648 |
| 122.40 | 0.1376 | | 93.12 | 0.6520 |
| 120.93 | 0.2211 | | 88.73 | 0.7373 |
| 112.52 | 0.3948 | | 81.96 | 0.8245 |
| 111.97 | 0.3083 | | 72.99 | 0.8824 |
| 105.93 | 0.4801 | | 67.32 | 0.9354 |

Range x/L 0.053-0.935 over phi 67.3-143.3, one point per station on all 12
stations, monotone in x/L except for the 112.52/111.97 near-tie (stations
0.3948 and 0.3083, i.e. the front is locally flat there). Sorted by phi
descending, matching the 14a/14b/14c convention; sort by `xL` for the
station-ordered front.

Re_L = 6.54e6 (circles), 10 points: 0.1373 @ 126.42, 0.2214 @ 91.84,
0.3080 @ 63.66, 0.3951 @ 57.99, 0.4804 @ 52.32, 0.5658 @ 49.94,
0.6523 @ 47.38, 0.7376 @ 44.81, 0.8242 @ 51.58, 0.8827 @ 52.32 — every
station from 0.222 to 0.883 with no gap.

## 5. The one genuinely hard glyph pair (Fig. 16c, station x/L = 0.137)

At the x/L ~ 0.137 station the square and the circle are drawn **overlapping**,
19 px apart in phi. Resolved by a **joint two-template overlay fit**: both
orderings (circle-above-square, square-above-circle) are scored with both
templates placed at once over a +-4 px search, and the winner is reported with
its margin. Circle-above-square wins by **82** mismatched pixels (985 vs 1067,
~8%; in a tighter window the same test gave 220 vs 330, ~33%). The raster's own
edge profiles agree independently: the upper glyph's first dark row is 13 px
wide and widens to 21 px within 3 rows — the circle template's signature
(13 -> 26 in 5 rows) — while the lower glyph has straight left/right walls over
18 consecutive rows and a flat full-width bottom edge. Both points are flagged
in the JSON with `method: joint two-glyph overlay fit`. Note the single-glyph
NCC scores favoured *square* for BOTH members of the pair (0.76 / 0.66),
i.e. the naive reading would have produced two squares and lost a circle.

The other four hand-resolved identities (all in 14c) and their evidence are in
the script table and in the JSON: the squares at (phi 160.8, x/L 0.935),
(125.3, 0.737), (100.5, 0.824) and the circle at (10.4, 0.567).

## 6. Visual verification actually performed

Both check PNGs and both glyph montages were rendered and **read back**:

* `check_fig14c.png` — 8 red squares and 4 blue circles, each drawn at the
  fitted centre, all sitting on the printed glyphs; nothing un-circled
  anywhere in the panel.
* `check_fig16c.png` — 12 red squares and 10 blue circles likewise, with the
  square chain tracking Stock's short-dashed computed separation line and the
  circle chain tracking his long-dashed CF line (an independent sanity check
  that the two Re sets were not swapped).
* `check_fig14c_glyphs.png` / `check_fig16c_glyphs.png` — every glyph tiled at
  5x with its assigned identity, phi and shape margin. All 34 were inspected
  one by one; all 34 are correctly centred and correctly typed.
* The green rectangle in each check PNG is the legend erase box, drawn so a
  reader can confirm nothing but legend text and legend glyphs is inside it.

Additional completeness audits (both clean):

* **Legend box** — re-ran the NCC with the legend NOT erased: every extra hit
  falls on the legend's own text/symbol rows (14c y ~ 2428-2433 / 2501-2507 /
  2566-2581; 16c y ~ 2348-2352 / 2413 / 2422 / 2489). No data glyph is hidden
  under a legend. (Geometrically impossible anyway: 14c's legend covers
  x/L 0.02-0.20 at phi 124-157, where the front sits at x/L 0.737.)
* **Frame lines** — the `digitize_stock_fig14b.py` on-frame-glyph scan was run
  on all four frame lines of both panels (phi = 0, phi = 180, X/a = -1,
  X/a = +1). Seven glyph-width dark clusters were flagged and all seven were
  zoomed at 4x: every one is a grid/frame crossing or a converging streamline
  bundle. **No glyph is drawn on a frame line in either panel**, so unlike
  Fig. 14a/14b there is no on-frame recovery to do and none to truncate.

## 7. Truncations

**None.** Every printed measured symbol in both panels was recovered:
12 glyphs in Fig. 14c (8 + 4) and 22 in Fig. 16c (12 + 10), matching an
independent by-eye enumeration of both panels at full resolution made before
the automatic run. There is no chain end that had to be cut, no glyph lost to
a frame stroke, and no glyph lost under the legend.

Two things that are **source coverage limits, not truncations**:

1. Fig. 14c's Re = 1.52e6 chain exists only at the four most downstream
   stations (x/L 0.737-0.936) and its Re = 6.49e6 chain only at two
   (0.480-0.567). Upstream stations carry no symbol because no transition was
   detected there at those Reynolds numbers — the same reason Fig. 15a's
   alpha = 10 chain starts at x/L 0.396. Do not read the absence of upstream
   points as missing data.
2. Neither panel has a measured point ON phi = 0 or phi = 180 (Fig. 14b does).
   The hot-film array simply has no symbol plotted there in these two tests.

## 8. What I deliberately did NOT do, and what I could not establish

* **Stock's computed curves in these two panels are not digitized.** Fig. 15a's
  `stock_computed_separation_line` has no counterpart in the new files. Each
  panel c carries four dash styles (streamlines / free vortex-layer separation
  "- - - -" / TS waves / TS+CF "– – –") for **two** Reynolds numbers, overlaid
  on ~20 streamlines, and the caption's style glyph for "TS waves" is an inline
  image that does not extract as text. I could distinguish a short-dashed curve
  and a long-dashed curve in each panel, but I could not establish
  *which Reynolds number each belongs to* with the confidence this pipeline
  demands, and for 14c I could not identify the two heavy solid polylines in
  the leeward-downstream corner at all (they are plausibly the Re = 6.49e6
  TS-wave front near the leeward plane, plausibly the streamline-12 TS
  segment — I would be guessing). This is recorded in each JSON under
  `not_digitized`. If Sec. X needs the computed separation line for
  alpha = 5 or 29.7, that is a separate, careful pass.
* **Sub-pixel refinement was not attempted.** Positions are integer-aligned
  template fits; the raster's 2x pixel duplication makes anything finer
  cosmetic.
* **The phi = 150 grid row** is undetectable in both panels (dashes buried in
  the streamline bundle), so the phi scale is checked at 2 interior rows in
  14c and 4 in 16c, not 5.
* **The residual ~0.3-0.5 deg systematic** in the interior phi grid rows is
  real and unresolved: in 16c all four detected grid rows read 0.15-0.46 deg
  LOW, which traces to the phi = 0 frame row centroid (its detected row group
  is 12 px wide — frame stroke plus axis ticks — versus 10 px at phi = 180).
  Refitting phi on the interior grid rows instead would move the phi = 0 end
  by 3 px (0.57 deg) in 16c but is *worse* in 14c, where the two available
  grid rows scatter +-2.6 px in opposite directions. I kept the frame-row
  convention of `digitize_stock_fig14a/b.py` for cross-figure consistency and
  folded the discrepancy into the quoted +-0.4-0.5 deg.

---

# ADDENDUM (2026-08-02): Stock Fig. 17a — ONERA F1, alpha = 10 deg — and the 6.56e6 cross-facility question

*Same rules and same machine (019-v100-dev). Nothing on 014/017 touched, no
`.tex` edited, nothing staged or committed, no point fabricated.*

## A1. Deliverables added

| file | contents |
|---|---|
| `paper/data/stock2006_fig17a_digitized.json` | alpha = 10 deg, ONERA F1; 5 squares (Re_L = 6.56e6) + 3 circles (Re_L = 18.32e6), plus a `cross_facility_check` block |
| `paper/repro/cfd/digitize_stock_fig17a.py` | generator; also re-reads Fig. 15a panel a and derives the verdict |

Check PNGs (same scratchpad dir): `check_fig17a.png`,
`check_fig17a_glyphs.png`, and — from the Fig. 15a re-read —
`check_fig15a_panelA.png`, `check_fig15a_panelA_glyphs.png`. All four were
rendered and read back.

Two small, documented hooks were added to the shared engine
(`digitize_stock_fig14c.py`): `cfg['extra']` (merge caller blocks into the
output JSON) and `cfg['no_write']` (a panel run that exists only to feed
another dataset's cross-check). **Side effect to be aware of:** re-running
14c/16c after that edit changed their md5s, because the `source` string
stamps the regeneration date, which had rolled to 2026-08-02. Every
numeric value in both files is unchanged (verified point by point against
the tables in the main record above).

## A2. Panel identity (verified, my panel arithmetic checked against the raster)

Fig. 17 = **page 10, image index 0, 2032x3558** (rect x = 42-279 pt), as the
brief said. Page 10's other image (index 1, 2915x1901, rect y = 554-708 pt) is
**Fig. 18**, not part of Fig. 17.

Caption, read from the PDF text layer: *"Fig. 17 Comparison of measured[50]
and computed transition locations for an angle of attack a) alpha = 10 deg,
b) alpha = 15 deg, and c) alpha = 30 deg"*. Panel a is the top third
(frame rows 54.6 / 1062.9; panel b 1202.5 / 2213.4; panel c 2337.4 / 3344.8).

Legend of panel a, read from the raster: **"Measured transition / [] Re = 6.56
x 10^6 / (o) Re = 18.32 x 10^6"** — confirms the brief exactly. Panels b and c
were also read: 17b has one Re (6.62e6), 17c has three (6.65 / 23.96 /
43.54e6) with triangles — matching the brief and making the two-Re legend
decisive for panel a.

## A3. Calibration (in-panel, nothing inherited)

**Fig. 17a**: X px `320.05 / 738.58 / 1161.00 / 1580.15 / 2001.47` ↔
-1…+1, max residual **1.23 px** (~0.0015 X/a); phi frame rows
**54.63** (180 deg) / **1062.94** (0 deg), h = 1008.31 px; independent
interior grid-row check 119.59 and 89.86 deg (nominal 120/90).

**Fig. 15a panel a** (re-read, page 9 image 1, 1970x3538): X px
`287.51 / 700.27 / 1117.71 / 1531.23 / 1947.35`, max residual **1.48 px**;
frame rows **56.72 / 1055.66**, h = 998.93; grid-row check 119.14 / 89.61 /
59.76 / 29.91.

The brief's warning was right and then some: **within the p10 raster the
five X columns shift right by ~5.4 px per panel** (a 320.05 → b 325.60 →
c 330.99) at constant panel width (1681.4 / 1680.8 / 1680.5 px) — a
progressive scan skew down the page. Inheriting any earlier fit would have
put a ~5 px (0.006 x/L) bias into whichever panel it was borrowed for.

## A4. Extraction — and the template problem specific to this raster

**Panel a contains no isolated data glyph**: all eight touch a streamline or
one of the computed fronts, and a full connected-component sweep of the
*entire* p10 raster found no isolated circle in any of its three panels
(panel b has squares only; panel c's circles are all crossed too). So:

* square template ← cleanest isolated square in **panel c of the same
  raster** (tight bbox rows 2972-2997, cols 699-725);
* circle template ← **least-contaminated panel-a circle** (rows 811-838,
  cols 546-573), which carries ~10 px of streamline ink at its upper right;
* the legend glyphs are **not** usable: as in Fig. 14a, they are drawn much
  larger than data glyphs (~44 px vs ~27 px) — measured, not assumed.

Because the circle template is defectively inked, the whole identity +
localization step was **redone with the pristine circle template of the
Fig. 15 raster** (`CI_XCHECK`, p9_img1 rows 930-957). Result, asserted in the
script: **all eight identities reproduce, and the three circle centres move
by 1.00, 0.00 and 0.00 px** (gate 1.5 px). The reported numbers are from the
in-raster template.

Detection threshold sits in the usual empty gap: 8 true glyphs score
0.55-1.00, the best false positive 0.488.

## A5. Results — Fig. 17a (ONERA F1, alpha = 10 deg)

**Re_L = 6.56e6 (squares), 5 points**, in front order:

| phi (deg) | x/L |
|---|---|
| 161.64 | 0.3137 |
| 120.93 | 0.2263 |
| 77.55 | 0.2269 |
| 48.81 | 0.3149 |
| 30.25 | 0.4017 |

**Re_L = 18.32e6 (circles), 3 points**: (145.03, 0.1416), (42.57, 0.1428),
(9.18, 0.2278).

Shape of the 6.56e6 front: downstream at the leeward plane (x/L 0.314 at
phi 162), an upstream bulge to x/L 0.226 across the mid-azimuths
(phi 78-121), then downstream again to x/L 0.402 at phi 30. The 18.32e6
front is far upstream of it everywhere (x/L 0.14-0.23), as expected.

One glyph (phi 30.25) has a tight shape margin of 7 because the phi = 30
dashed grid line runs straight through it; its votes were nevertheless
unanimous, and the ASCII edge profile settles it — the first dark row is
already 25 px wide with a flat top and straight walls, i.e. a square. No
hand-resolved identities were needed in this panel (`ambiguous=[]`).

## A6. Truncations — none

8 detected = 8 counted by eye at full resolution before the run (5 squares +
3 circles). Audits: re-running the NCC with the legend **kept** gives 17 hits,
9 of them on the legend's own three text rows (y 115-120 / 181-193 /
268-281) → 8 data glyphs, so nothing is hidden under the legend; the
four frame lines were scanned for on-frame glyphs and the single flagged
cluster (X/a = -1 at y 896-913) was zoomed at 4x and is the phi = 30 grid
line crossing the frame. The Fig. 15a re-read passes the same audits
(34 hits with legend kept, 21 on legend text rows → 13 data glyphs = 8
squares + 5 circles, matching the committed file's counts).

Coverage limit, not a truncation: Fig. 17a has no symbol anywhere downstream
of x/L 0.402 — the entire downstream half of the panel is empty.

## A7. The Fig. 15a re-read, and the standing open item RESOLVED

The engine re-read of Fig. 15a panel a vs the committed
`stock2006_fig15a_digitized.json`:

* **1.52e6 squares (8 points): the committed pass is vindicated.** All eight
  agree, |Δx/L| ≤ 0.0017, |Δphi| ≤ 0.88 deg.
* **6.56e6 circles (5 points): two of the five committed phi values are
  wrong.**

| committed (phi, x/L) | this re-read | verdict |
|---|---|---|
| 153.0, 0.2557 | **164.90, 0.3095** | committed value is **wrong** |
| 60.2, 0.3089 | 59.13, 0.3071 | agrees (−1.07 deg, −0.0018) |
| 35.6, 0.3962 | **40.39, 0.3944** | phi wrong by **+4.79 deg** |
| 27.7, 0.4810 | 29.04, 0.4793 | agrees (+1.34 deg, −0.0017) |
| 20.1, 0.5646 | 20.21, 0.5643 | agrees (+0.11 deg, −0.0003) |

**The leeward-most circle: I reproduce pass 40 (~0.31 / 165), not the
committed 0.256 / 153.** The evidence is not a judgment call. The glyph's
connected extent is rows 127-154 x cols 787-814 of p9_img1 with a rounded
top (first dark row 14 px wide) and rounded bottom, centre exactly
(140.5, 800.5) — which is precisely where the engine's overlay fit puts it.
The committed value maps to (y 206.6, x 711.2) in that raster, **66 rows and
89 columns away from any glyph**. Independent corroboration: 0.3095 lands on
the hot-film ring ladder (station 0.3083-0.3095) whereas 0.2557 is off-ladder,
and it makes the leeward point share a station with the phi = 59.13 circle,
the same two-azimuths-per-station pattern seen throughout Figs. 14c/16c.

The phi = 35.6 → 40.39 correction is the same kind of error: the committed
value maps to y 858 while the glyph occupies rows 818-845 (centre 831.5).

So `stock2006_fig15a_digitized.json`'s `note_circles` warning was justified
and can now be closed **in favour of pass 40**, and a second defect (the
phi = 35.6 point, outside that file's own ±1.5 deg claim) is newly found.
**I did not edit that committed file** — the corrected values live in
`stock2006_fig17a_digitized.json` under
`cross_facility_check.fig15a_engine_reread_re6p56e6_circles`, with the full
old-vs-new table under `fig15a_engine_vs_committed_json`. Updating the
fig15a file is your call.

## A8. VERDICT on the 6.56e6 question: two DIFFERENT measurements

The two sets, both printed as Re = 6.56e6 at alpha = 10 deg:

| | Fig. 15a (Goettingen, Ref. 49, circles) | Fig. 17a (ONERA F1, Ref. 50, squares) |
|---|---|---|
| points | 5 | 5 |
| phi sampled | 164.9, 59.1, 40.4, 29.0, 20.2 | 161.6, 120.9, 77.6, 48.8, 30.2 |
| x/L range | 0.307 – 0.564 | 0.226 – 0.402 |

Discrimination budget, set by the coarser side — except that the coarser side
is no longer fig15a's ±0.005, because I re-read it with the same engine, so
both sides carry ±0.0025 x/L / ±0.4 deg → combined 1σ **0.0035 x/L** and
**0.57 deg**. A cross-*figure* systematic is then added (see A9): **0.006
x/L**. Decision thresholds: 3σ + systematic = **0.0165 x/L** and **1.71 deg**.

Two independent tests, both failed by a wide margin:

1. **Azimuth coverage.** 4 of the 5 Fig. 15a azimuths (20.21, 40.39, 59.13,
   164.90) have **no** Fig. 17a counterpart within 1.71 deg. Conversely
   Fig. 17a has two points at phi 120.9 and 77.6 where Fig. 15a samples
   nothing at all between phi 59 and 165. Replotted points cannot appear and
   disappear.
2. **x/L at the one well-matched azimuth.** At phi ≈ 30 (Δphi 1.21 deg):
   Fig. 17a x/L 0.4017 vs Fig. 15a 0.4793, **Δ = −0.0776**, i.e. 4.7x the
   decision threshold and 22σ. The only other near-pair, phi ≈ 163
   (Δphi 3.26 deg, itself 5.7σ), agrees in x/L (+0.0042, within threshold) but
   is at a demonstrably different azimuth.

The front *shapes* differ too: the Goettingen 6.56e6 front is flat at
x/L ≈ 0.31 from phi 165 down to phi 59 and then sweeps downstream to 0.56 at
phi 20; the F1 front has an upstream bulge to x/L 0.226 at mid-azimuth. So
this is **a genuine second measurement at matched nominal Reynolds number in a
second facility — not Goettingen data replotted.** The test discriminates
decisively; nothing had to be forced.

Stock's own text supports the reading and explains *why* the overlap exists:
Sec. II gives Goettingen alpha 0-29.7 deg at Re 1.5e6-8.5e6 and ONERA F1
alpha 10-30 deg at Re 6.5e6-43.5e6 — 6.56e6 is the bottom of the F1 range and
inside the Goettingen range — and Sec. III.C says *"The high Reynolds number
possibilities in the pressurized facility were the main argument for this
wind-tunnel campaign **aside from comparison purposes**."* The F1 campaign
deliberately repeated the Goettingen Reynolds number.

## A9. Two findings that came out of the test and matter beyond it

**(i) There is a ~0.005 x/L systematic between the Ref. 49 and Ref. 50
figures' symbol layers.** The hot-film ring ladder recovered from all three
Fig. 17 panels sits +0.0045 to +0.0056 x/L from the ladder recovered from
Figs. 14c/15a/16c (measured: 17a mean +0.0056 over 7 glyphs, 17b +0.0042 over
7, 17c +0.0045 over 13; the ~0.086 x/L ring pitch is identical). This is
**not** a different instrumentation layout, because Stock Sec. II states *"The
prolate spheroid of 2.4-m total length was tested in the DFVLR 3 x 3 Meter Low
Speed Wind Tunnel Goettingen[49] ... and in the (CERT)/ONERA F1 Wind Tunnel Le
Fauga-Mauzac Center Toulouse[39,50]"* — **the same 2.4-m model in both
tunnels**, so the rings are physically the same and the offset has to be a
plotting/registration difference between the two figures. Consequence for us:
**any cross-figure x/L comparison in this campaign has a ~0.005 x/L floor**,
which is why it is folded into the threshold in A8. Within one figure
(and hence within one facility) the accuracy is the full ±0.0025.

**(ii) Stock has already quantified the disturbance-environment difference
between the two tunnels — and it runs opposite to the direction the brief
assumed.** From these very datasets he derives facility-specific limiting N
factors, stating *"the limiting N factors for wind tunnels are specific
quantities depending on the flow quality of the considered facility"*:

| facility | N_TS | N_CF |
|---|---|---|
| DFVLR 3x3 m Goettingen (Ref. 49, Fig. 11) | **8.0** | **5.5** |
| CERT/ONERA F1 (Ref. 50, Fig. 13) | **7.0** | **6.0** |

A *lower* limiting N_TS means transition at *less* amplification, i.e. F1 is
the **less TS-stable** environment in Stock's own calibration (ΔN_TS = −1.0),
while being slightly **more CF-stable** (ΔN_CF = +0.5). My digitized fronts
are consistent with that: at phi ≈ 30, close to the windward symmetry plane
where Stock says pure TS waves dominate, the F1 front is 0.078 x/L
**upstream** of the Goettingen front. If instead F1's freestream turbulence
really is <0.1% against Goettingen's 0.33-0.4%, then that Tu ratio does *not*
carry through to Stock's N_TS limit in the naive direction, and the difference
must be dominated by something else (acoustic/vibration environment, unit
Reynolds number — F1 is pressurized, so the same Re comes at much lower speed
— or surface condition of the shared model).

Caveat I must flag: **this paper quotes no freestream turbulence level for
either tunnel.** I verified that by full-text search of `spheroid.pdf`. The
0.33-0.4% / <0.1% figures in the brief must be sourced from the Kreplin,
Vollmers & Meier reports (Refs. 49 = IB 222-84 A 33 and 50 = IB 222-84 A 34)
or elsewhere, and should be cited to those, not to Stock. Also note the
cross-facility front comparison rests on **one** well-matched azimuth at
**one** Reynolds number, so it supports "consistent with" and not more.

## A10. Still not done / not established

* Stock's computed curves in Fig. 17a are again **not** digitized, for the
  same reason as in 14c/16c (four dash styles, two Reynolds numbers, ~20
  streamlines, and the caption's TS-wave style glyph is an inline image that
  does not extract as text).
* Whether the +0.005 x/L ladder offset is a Fig.-17-wide symbol-layer shift
  or something in Stock's Fig. 15 axis drawing **cannot be decided from the
  raster**; both figures are internally self-consistent. I treat it as an
  irreducible cross-figure systematic.
* Fig. 17b (alpha = 15, Re 6.62e6) and 17c (alpha = 30, three Re) were
  detected only for the station-ladder audit in A9 — those numbers are *not*
  a validated dataset (no identity votes, no glyph-by-glyph visual pass, and
  17c's triangles were not modelled at all, so the 13 "square-template hits"
  there certainly mix triangles in). Do not use them as data.
