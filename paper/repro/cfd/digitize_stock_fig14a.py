"""Digitize Stock (2006) Fig. 14a: the alpha=0 deg transition control
case -- MEASURED transition locations (DFVLR hot films, Kreplin et al.,
Ref. 49 of Stock; open squares, Re = 7.20e6, exactly our re72a0 ladder
condition, no Re mismatch) and Stock's COMPUTED pure-TS-wave e^N front
(the continuous near-vertical line; the caption differentiates TS-wave
transition as the continuous style, and the Sec. III.C text states
transition at alpha=0 is provoked solely by TS waves).

Source raster: spheroid.pdf p.8, image 1 (right column, Fig. 14 a/b/c
stacked) -> /local_data/qiqi/sa-ai/stock_digitize/p8_img1_2027x3630.png
(re-extract with --extract). Panel a is the top panel.

Calibration MEASURED by tick detection on the full-resolution raster
(same ritual as digitize_stock_waterfalls.py):
  X/a: five full-height grid/frame columns detected at px
       302.0/725.0/1152.5/1576.0/2000.5 <-> -1.0/-0.5/0.0/+0.5/+1.0
       (least-squares fit below; residual < 1.5 px ~ 0.002 X/a);
  phi: panel-a frame rows 56.0 (phi=180) / 1076.5 (phi=0), midline
       check: detected phi=90 dashed grid at row 567 vs 566.25
       predicted. Note x/L = (X/a + 1)/2.

Extraction:
  measured squares -- NCC template match (legend glyph at ~(1408,207)
       as template, the digitize_stock_waterfalls.py --symbols method),
       verified in the check overlay PNG;
  computed TS front -- the only non-calibration column cluster with
       >90% panel-height dark coverage (px ~1020-1026); per-row dark
       centroid inside +-8 px, reported per phi.

Run from paper/:  python3 repro/cfd/digitize_stock_fig14a.py
-> data/stock2006_fig14a_digitized.json
   + check overlay /local_data/qiqi/sa-ai/stock_digitize/check_fig14a.png
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

DIG = '/local_data/qiqi/sa-ai/stock_digitize'
_H = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(_H, '..', '..'))
IMG = 'p8_img1_2027x3630.png'

# panel a geometry (full-image px), tick-detected -- see docstring
XCOLS = np.array([302.0, 725.0, 1152.5, 1576.0, 2000.5])
XVALS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
Y_PHI180, Y_PHI0 = 56.0, 1076.5
PANEL = (62, 1072, 305, 1997)            # y0, y1, x0, x1 interior
# full legend erase: glyph (~1408,207) AND the two text lines -- the
# letter glyphs of "Measured transition" otherwise NCC-match as squares
LEGEND_BOX = (90, 270, 1360, 1997)       # y0, y1, x0, x1


def extract():
    import fitz
    doc = fitz.open(os.path.join(PAPER, 'spheroid.pdf'))
    xref = doc[7].get_images(full=True)[1][0]
    pix = fitz.Pixmap(doc, xref)
    pix.save(os.path.join(DIG, IMG))
    print('extracted', IMG, pix.width, 'x', pix.height)


def x_to_Xa(px):
    A = np.polyfit(XCOLS, XVALS, 1)
    return A[0] * px + A[1]


def y_to_phi(py):
    return 180.0 * (Y_PHI0 - py) / (Y_PHI0 - Y_PHI180)


def main():
    if '--extract' in sys.argv or not os.path.exists(f'{DIG}/{IMG}'):
        extract()
    dark = np.array(Image.open(f'{DIG}/{IMG}').convert('L')) < 128
    y0, y1, x0, x1 = PANEL
    panel = dark[y0:y1, x0:x1].copy()
    ly0, ly1, lx0, lx1 = LEGEND_BOX
    panel[ly0-y0:ly1-y0, lx0-x0:lx1-x0] = False

    # ---- computed TS front: full-height column cluster ----------------------
    h = panel.shape[0]
    cov = panel.sum(0) / h
    grid_px = np.concatenate([XCOLS - x0, ])
    cand = [c for c in np.where(cov > 0.9)[0]
            if np.abs(c - grid_px).min() > 12]
    assert cand, 'no computed-front column found'
    assert max(cand) - min(cand) < 15, f'front cluster not unique: {cand}'
    cline = float(np.mean(cand))
    ts = []
    for r in range(h):
        w = np.where(panel[r, int(cline)-8:int(cline)+9])[0]
        if len(w):
            ts.append((r, float(w.mean()) + cline - 8))
    ts_phi = [round(y_to_phi(r + y0), 2) for r, _ in ts]
    ts_Xa = [round(float(x_to_Xa(c + x0)), 4) for _, c in ts]
    print(f'computed TS front: {len(ts)} rows, X/a = '
          f'{np.mean(ts_Xa):.4f} +- {np.std(ts_Xa):.4f} '
          f'(x/L {np.mean([(v+1)/2 for v in ts_Xa]):.4f})')

    # ---- measured squares: NCC template match ------------------------------
    # The legend glyph (50x48 px at ~(1408,207)) is drawn LARGER than the
    # in-panel squares (~27x30 px) -- Stock's plotting package scales
    # legend symbols up -- so the template is cut from a clean, isolated
    # DATA square instead (rows 296-323 full-image, phi ~ 135; verified
    # visually).  Identity is unambiguous (one symbol type in the panel)
    # and every match is verified in the check overlay.
    from scipy.signal import fftconvolve
    t = dark[293:327, 1030:1062].astype(float)
    print('template', t.shape)
    tm = t - t.mean()
    img = panel.astype(float)
    # zero-normalized cross-correlation with the LOCAL window mean (a
    # global-mean version scores the true glyph only ~0.41; this one
    # scores it 1.0 with the false-positive floor at ~0.6)
    ones = np.ones_like(t)
    s1 = fftconvolve(img, ones, mode='same')
    s2 = fftconvolve(img**2, ones, mode='same')
    num = fftconvolve(img, tm[::-1, ::-1], mode='same')  # tm zero-mean
    var = np.maximum(s2 - s1**2 / t.size, 1e-6)
    score = num / np.sqrt(var * (tm**2).sum())
    sq = []
    s = score.copy()
    while True:
        i = np.argmax(s)
        r, c = np.unravel_index(i, s.shape)
        if s[r, c] < 0.55:
            break
        sq.append((int(r), int(c), float(s[r, c])))
        s[max(0, r-14):r+15, max(0, c-14):c+15] = -1
    sq.sort()
    print(f'{len(sq)} measured squares (scores '
          f'{min(q[2] for q in sq):.2f}-{max(q[2] for q in sq):.2f})')

    # ---- check overlay -------------------------------------------------------
    im = Image.open(f'{DIG}/{IMG}').convert('RGB').crop((0, 0, 2027, 1150))
    dr = ImageDraw.Draw(im)
    for r, c, _ in sq:
        dr.ellipse([c+x0-16, r+y0-16, c+x0+16, r+y0+16],
                   outline=(255, 0, 0), width=3)
    for r, c in ts[::10]:
        dr.ellipse([c+x0-3, r+y0-3, c+x0+3, r+y0+3], outline=(0, 120, 255))
    im.save(f'{DIG}/check_fig14a.png')

    sq_out = [dict(xL=round((float(x_to_Xa(c + x0)) + 1) / 2, 4),
                   Xa=round(float(x_to_Xa(c + x0)), 4),
                   phi_deg=round(y_to_phi(r + y0), 2), ncc=round(sc, 3))
              for r, c, sc in sq]
    out = dict(
        source='Stock, AIAA J 44(1) 2006, Fig. 14a (DOI 10.2514/1.16026); '
               'alpha=0 deg, Re_L=7.20e6 (exact match to our re72a0 '
               'ladder); measured transition = DFVLR hot films (Kreplin '
               'et al., Stock Ref. 49); computed = Stock pure-TS-wave e^N '
               '(N_TS=8.0, his DFVLR-tunnel limit, Fig. 11a), his '
               'continuous-line style; digitized '
               f'{__import__("datetime").date.today()} from spheroid.pdf '
               'p.8 raster (p8_img1) by NCC template match (squares) + '
               'full-height column trace (TS line); tick-detected '
               'calibration, see script docstring',
        convention='phi=0 windward symmetry line; x/L=(X/a+1)/2 from nose',
        calibration=dict(x_pix=XCOLS.tolist(), x_val=XVALS.tolist(),
                         y_pix=[Y_PHI180, Y_PHI0], y_val=[180.0, 0.0]),
        measured_squares=sq_out,
        computed_ts_front=dict(
            phi_deg=ts_phi[::5], Xa=ts_Xa[::5],
            xL=[round((v + 1) / 2, 4) for v in ts_Xa[::5]],
            mean_xL=round(float(np.mean([(v+1)/2 for v in ts_Xa])), 4),
            std_xL=round(float(np.std([(v+1)/2 for v in ts_Xa])), 4)),
        note='chain recovered at 17 azimuths, phi 3-177 deg; symbols '
             'drawn exactly ON the phi=0/180 frame lines (the raster '
             'shows merged dark blobs there) are unrecoverable; every '
             'kept match is circled in check_fig14a.png',
    )
    path = f'{PAPER}/data/stock2006_fig14a_digitized.json'
    json.dump(out, open(path, 'w'), indent=1)
    ml = [q['xL'] for q in sq_out]
    print(f'measured squares x/L: {np.mean(ml):.4f} +- {np.std(ml):.4f} '
          f'(range {min(ml):.4f}-{max(ml):.4f})')
    print('wrote', path, 'and', f'{DIG}/check_fig14a.png')


if __name__ == '__main__':
    main()
