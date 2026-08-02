"""Digitize Stock (2006) Fig. 16c: alpha = 29.7 deg -- MEASURED transition
locations (DFVLR 3x3 m tunnel hot films, Kreplin et al., Stock Ref. 49) at
BOTH Reynolds numbers printed in the panel legend: open squares
Re_L = 1.53e6 and open circles Re_L = 6.54e6.  The 1.53e6 square chain is
the high-incidence laminar-up-to-separation reference (Stock Sec. III.C:
"The transition prediction for alpha = 29.7 deg and two Reynolds numbers
(Fig. 16c) is of similar quality. For the small Reynolds number,
transition is provoked completely by separation."), i.e. the same regime as
Fig. 15a (alpha = 10, Re = 1.52e6) and Fig. 14c (alpha = 5, Re = 1.52e6).

Source raster: spheroid.pdf page 9, image index 0 (the RIGHT column of the
page, placement rect x = 305-542 pt, Fig. 16 a/b/c stacked, 1974x3476 px)
-> $DIG/p9_img0_1974x3476.png; re-extract with --extract.  Page 9 carries
TWO figure rasters: image index 1 (x = 48-273 pt, 1970x3538 px) is the
LEFT column and is Fig. 15 -- do not confuse them.  Panel c is the BOTTOM
panel of Fig. 16; identity verified from its printed legend ("Re = 1.53 x
10^6" / "Re = 6.54 x 10^6") against the Fig. 16 caption and the Sec. III.C
text: panel a is alpha = 24 deg (single Re = 6.42e6), panel b is
alpha = 29.5 deg (three Reynolds numbers 3.01/4.48/8.52e6, squares +
circles + triangles), panel c is alpha = 29.7 deg (two Reynolds numbers).
The two-Re legend is therefore decisive for panel c.

NOTE the Reynolds number: the task brief said "Re_L ~ 1.5e6", and the
panel legend indeed prints Re = 1.53 x 10^6 (not 1.52e6 as in Figs 14c and
15a -- Stock quotes the individual run Reynolds number of each test).

Method, calibration, accuracy and the deliberate exclusion of Stock's
computed curves: identical to digitize_stock_fig14c.py, whose engine this
script imports so the two panels cannot drift apart.  Read that docstring.
The one Fig.-16c-specific complication: at the x/L = 0.1382 hot-film
station the square and the circle are drawn OVERLAPPING (the circle 19 px
above the square), so that pair is resolved by a JOINT two-template
overlay fit -- both hypotheses are scored and the winning one is reported
with its margin (circle-above-square wins by ~33% of the mismatch count;
the raster's own edge profiles agree: the upper glyph's first dark row is
13 px wide and widens to 21 px in 3 rows, the signature of the circle
template, while the lower glyph has straight walls and a flat bottom).

Run from paper/:  python3 repro/cfd/digitize_stock_fig16c.py
-> data/stock2006_fig16c_digitized.json
   + check overlay  $DIG/check_fig16c.png
   + glyph montage  $DIG/check_fig16c_glyphs.png   (LOOK AT BOTH)
$DIG defaults to $STOCK_DIGITIZE_DIR, else <tmp>/stock_digitize.  For the
2026-08-01 pass on 019-v100-dev that was
/tmp/claude-1006/-home-qiqi-flexcompute/3d1a461e-96df-48fb-8bfc-cdfece2123e6/scratchpad/stock_digitize
(do NOT default this to /local_data/... -- that path lives on 014).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from digitize_stock_fig14c import run                       # noqa: E402

FIG16C = dict(
    key='fig16c', page=8, image=0, img='p9_img0_1974x3476.png',
    zone=(2270, 3300),               # panel-c search band, full-image px
    legend=(2318, 2512, 1350, 1968),  # y0,y1,x0,x1 erase (legend block)
    tsq=(2651, 2676, 946, 972),      # tight bbox, isolated square glyph
    tci=(3020, 3047, 1509, 1534),    # tight bbox, isolated circle glyph
    alpha=29.7, re_sq=1.53e6, re_ci=6.54e6,
    fig='16c', panel='c (bottom)',
    # glyphs whose three automatic identity votes disagree (see the engine
    # docstring): identity read off the raster by hand at 5x zoom.
    ambiguous=[
        (2959, 960, 'ci',
         'the bottom edge tapers 26->23->17->15->11->10 px over rows '
         '2969-2976, the rounded-bottom circle signature (a square bottom '
         'edge keeps full width for ~5 rows and then stops); the inked top '
         'corners come from the phi=60 dashed grid line, a full-width dark '
         'band at rows 2949-2955 that crosses the glyph and is what flips '
         'the corner-ink and XOR votes; reading it as a circle also '
         'completes the Re=6.54e6 chain, which then occupies every hot-film '
         'station from x/L=0.222 to x/L=0.883 with no gap'),
    ],
    caption_note='Stock Fig. 16 caption: "Comparison of measured[49] and '
                 'computed transition locations for angle of attack a) '
                 'alpha = 24 deg, b) alpha = 29.5 deg, and c) alpha = 29.7 '
                 'deg"; panel-c legend reads "Measured transition / [] Re = '
                 '1.53 x 10^6 / (o) Re = 6.54 x 10^6"',
    regime='Sec. III.C: "The transition prediction for alpha = 29.7 deg and '
           'two Reynolds numbers (Fig. 16c) is of similar quality. For the '
           'small Reynolds number, transition is provoked completely by '
           'separation." (i.e. the Re = 1.53e6 measured chain is the '
           'laminar-to-separation front); at the high Re transition is CF-'
           'wave triggered except close to the symmetry planes',
)

if __name__ == '__main__':
    run(FIG16C)
