"""Digitize/extract literature Cd(Re) datasets for the drag-crisis figure
(fig:dragcrisiscd overlay). Sources live in paper/repro/cfd/litdata/
dragcrisis/ (acquired PDFs; see README.md there for provenance URLs).

Methodology (binding, per digitize_stock_waterfalls.py precedent):
independent digitization from the acquired source PDFs -- never trust
committed JSONs; every dataset gets an axis calibration MEASURED from
the source (vector figures: exact tick/frame coordinates from the PDF
drawing stream; raster figures: tick/gridline detection on the rendered
page) plus a check PNG overlaying every digitized point on the source
image (litdata/dragcrisis/checks/). Residual asserts guard each
calibration. Tabulated values (Rodriguez Table 2, Catalano Table 1
cross-check) are transcribed, not digitized.

Vector figures are extracted from the PDF drawing stream itself
(get_drawings): symbol centers are exact to the plotting program's
output, so the only error is the original authors' plotting accuracy.
Raster figures (scans) are digitized by thresholded blob detection with
shape/size classification; every kept blob is circled in the check PNG.

Datasets (one JSON each, litdata/dragcrisis/<name>.json):
  vector  catalano2001_wmles      WMLES points (+ Achenbach-1968 curve as
                                  reproduced by Catalano et al. [secondary])
  vector  rodriguez2015_fig4_exp  Schewe 1983 '+' and Achenbach&Heinecke
                                  1981 open squares [secondary via fig 4]
                                  (+ Delany-Sorensen 'x' as cross-check)
  vector  veysey_fig7_lowre       Tritton/Finn/Jayaweera low-Re drag
                                  [secondary replot, arXiv physics/0609138]
  table   rodriguez2015_les       WRLES Table 2 transcription (primary)
  table   qu2013_dns              2-D unsteady DNS-class sweep Re 50-200,
                                  Qu et al. 2013 Table 3 (primary)
  table   dong_karniadakis2005_dns3d  3-D spectral DNS, Re=1e4 resolution
                                  study, Dong & Karniadakis 2005 Table 2
  raster  iop2020_models          Stabnikov-Garbaruk fig 3 left: SST
                                  (fully turbulent), SST gamma-Re_theta,
                                  SST KD sweeps + their Schewe replot
  raster  stringer2014_urans      fig 5: CFX + OpenFOAM SST URANS sweep
  raster  roshko1961              fig 2: Roshko's own points 1e6-1e7
  raster  henderson1995           fig 1: 2-D spectral pressure+viscous
                                  drag, steady + shedding branches
  raster  tn3038_delany_sorensen  fig 5: circular-cylinder Cd circles
  raster  tn84_wieselsberger      fig 1: composite 9-diameter Cd(Re)

Run from anywhere:  python3 repro/cfd/digitize_dragcrisis_lit.py [name...]
"""
import json
import os
import sys

import numpy as np
import fitz
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
LIT = os.path.join(HERE, 'litdata', 'dragcrisis')
CHECKS = os.path.join(LIT, 'checks')
os.makedirs(CHECKS, exist_ok=True)
TODAY = str(__import__('datetime').date.today())


# ---------------------------------------------------------------- helpers
class Cal:
    """1-D axis calibration from (pix, val) anchor pairs; log or linear.
    Fits val (or log10 val) linear in pix; residual asserted."""

    def __init__(self, pix, val, log=False, tol=None, name=''):
        self.log, self.name = log, name
        pix = np.asarray(pix, float)
        v = np.log10(val) if log else np.asarray(val, float)
        self.a, self.b = np.polyfit(pix, v, 1)
        res = np.abs(self.a * pix + self.b - v).max()
        self.res = res
        if tol is not None:
            assert res < tol, f'{name}: calibration residual {res} > {tol}'

    def __call__(self, px):
        v = self.a * np.asarray(px, float) + self.b
        return 10.0 ** v if self.log else v

    def inv(self, val):
        v = np.log10(val) if self.log else np.asarray(val, float)
        return (v - self.b) / self.a

    def report(self):
        return dict(kind='log' if self.log else 'linear',
                    slope=self.a, intercept=self.b, max_residual=self.res)


def dump(name, payload):
    path = os.path.join(LIT, name + '.json')
    json.dump(payload, open(path, 'w'), indent=1)
    n = sum(len(v.get('points', v.get('Re', [])))
            if isinstance(v, dict) else 0 for v in payload.values()
            if isinstance(v, dict))
    print(f'wrote {path}')
    return path


def vec_check_png(pdfpath, pageno, marks, out, zoom=3.0, clip=None):
    """Render a PDF page and circle every extracted point.
    marks: list of (x_pt, y_pt, rgb) in PDF points."""
    doc = fitz.open(pdfpath)
    pix = doc[pageno].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    im = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
    dr = ImageDraw.Draw(im)
    for x, y, col in marks:
        X, Y = x * zoom, y * zoom
        dr.ellipse([X - 6, Y - 6, X + 6, Y + 6], outline=col, width=2)
    if clip:
        im = im.crop([int(c * zoom) for c in clip])
    im.save(os.path.join(CHECKS, out))
    print('  check:', os.path.join(CHECKS, out))


def segments(page):
    """All 1-item line segments from the drawing stream:
    (x0,y0,x1,y1,color)."""
    out = []
    for d in page.get_drawings():
        if len(d['items']) == 1 and d['items'][0][0] == 'l':
            p, q = d['items'][0][1], d['items'][0][2]
            out.append((p.x, p.y, q.x, q.y, tuple(d.get('color') or ())))
    return out


def all_segments(page):
    """Every 'l' item of every drawing (multi-item paths included)."""
    out = []
    for d in page.get_drawings():
        col = tuple(d.get('color') or ())
        for it in d['items']:
            if it[0] == 'l':
                p, q = it[1], it[2]
                out.append((p.x, p.y, q.x, q.y, col))
    return out


def raster_check(img, keep, out, rej=(), r=9, scale=1.0):
    """Circle kept (green) and rejected (red) blob centers on a raster."""
    im = img.convert('RGB') if img.mode != 'RGB' else img.copy()
    dr = ImageDraw.Draw(im)
    for x, y in rej:
        dr.ellipse([x - r, y - r, x + r, y + r], outline=(255, 60, 60),
                   width=2)
    for x, y in keep:
        dr.ellipse([x - r, y - r, x + r, y + r], outline=(0, 200, 60),
                   width=2)
    if scale != 1.0:
        im = im.resize((int(im.width * scale), int(im.height * scale)))
    im.save(os.path.join(CHECKS, out))
    print('  check:', os.path.join(CHECKS, out))


def blobs(mask, amin, amax):
    """Connected components of a boolean mask within an area window.
    Returns (cx, cy, area, h, w, slice) per blob."""
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    out = []
    for sl in ndimage.find_objects(lab):
        if sl is None:
            continue
        sub = lab[sl] > 0
        a = int(sub.sum())
        if not amin <= a <= amax:
            continue
        ys, xs = np.nonzero(sub)
        out.append((sl[1].start + xs.mean(), sl[0].start + ys.mean(), a,
                    sl[0].stop - sl[0].start, sl[1].stop - sl[1].start, sl))
    return out


# ------------------------------------------------------- vector datasets
def d_catalano2001_wmles():
    """CTR brief fig 3 (p.5): 3 blue filled circles = WMLES CD at
    Re = 5e5 / 1e6 / 2e6; red polyline = the experimental composite that
    Catalano, Wang & Iaccarino label 'Achenbach (1968)' -- kept only over
    Achenbach's measured range 6e4-5e6 and marked [secondary].
    Calibration from the x decade-label tick columns (10^1..10^7 evenly
    spaced -- verified) and the y labels 0..2; both cross-checked against
    the axis frame ticks in the drawing stream."""
    f = os.path.join(LIT, 'catalano_wang_ctr2001.pdf')
    page = fitz.open(f)[4]
    words = page.get_text('words')
    # x calibration from the DETECTED major tick columns of fig 3's
    # bottom axis (vertical 1-item segments ending at y 255-270); they
    # sit at 185.2..408.1 pt, one per decade label 10^1..10^7.
    tens = sorted(set(round(s[0], 1) for s in segments(page)
                      if abs(s[0] - s[2]) < 0.01 and
                      255 < max(s[1], s[3]) < 270))
    assert len(tens) == 7, tens
    gaps = np.diff(tens)
    assert gaps.std() < 0.15, f'uneven decade spacing {gaps}'
    calx = Cal(tens, [10.0 ** k for k in range(1, 8)], log=True, tol=0.004,
               name='catalano x')
    ylab = [(266.6, 0.0), (241.0, 0.4), (215.4, 0.8), (189.8, 1.2),
            (164.2, 1.6), (138.6, 2.0)]
    got = sorted(((w[1] + w[3]) / 2, w[4]) for w in words
                 if 130 < w[1] < 272 and w[0] < 180 and
                 w[4] in ('0', '0.4', '0.8', '1.2', '1.6', '2'))
    caly = Cal([g[0] for g in got], [float(g[1]) for g in got][::1],
               log=False, tol=0.01, name='catalano y') \
        if [g[1] for g in got] == ['2', '1.6', '1.2', '0.8', '0.4', '0'] \
        else Cal([p for p, _ in ylab], [v for _, v in ylab], tol=0.01,
                 name='catalano y fallback')
    pts, red, marks = [], [], []
    for d in page.get_drawings():
        fill = tuple(d.get('fill') or ())
        col = tuple(d.get('color') or ())
        r = d['rect']
        if fill and abs(fill[2] - 1.0) < 1e-6 and fill[0] < 0.5 and \
                len(d['items']) == 12:                       # blue circles
            cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
            pts.append((float(calx(cx)), float(caly(cy))))
            marks.append((cx, cy, (0, 200, 60)))
        if col and col[0] > 0.7 and (not fill) and len(d['items']) > 20 \
                and r.y1 < 280:                              # red polyline
            for it in d['items']:
                if it[0] == 'l':
                    for q in (it[1], it[2]):
                        red.append((q.x, q.y))
    pts.sort()
    assert len(pts) == 3, pts
    # thin + convert the red curve, restrict to Achenbach's Re range
    red = sorted(set(red))
    curve = [(float(calx(x)), float(caly(y))) for x, y in red]
    curve = [(re, cd) for re, cd in curve if 6e4 <= re <= 5e6]
    for x, y in red:
        if 6e4 <= calx(x) <= 5e6:
            marks.append((x, y, (255, 120, 0)))
    vec_check_png(f, 4, marks, 'check_catalano2001.png', zoom=3.0,
                  clip=(150, 120, 470, 290))
    dump('catalano2001_wmles', dict(
        source='Catalano, Wang & Iaccarino, CTR Annual Research Briefs '
               '2001, pp. 45-50 (open PDF, see README); journal version '
               'Catalano et al., Int. J. Heat Fluid Flow 24 (2003) '
               '463-469. Fig. 3, p. 49 (PDF p. 5); vector extraction '
               f'from the drawing stream, {TODAY}.',
        method='vector: blue 12-segment filled circles = WMLES points; '
               'calibration from the seven evenly-spaced decade labels '
               '(std of gaps < 0.15 pt) and the 0..2 y labels; check '
               'PNG checks/check_catalano2001.png',
        calibration=dict(x=calx.report(), y=caly.report()),
        wmles=dict(cls='WMLES', access='primary',
                   points=[dict(Re=re, Cd=cd) for re, cd in pts],
                   note='Table 1 (p. 48) states CD=0.31 at Re=1e6; the '
                        'extracted middle point must match to ~0.005'),
        achenbach1968_curve=dict(
            cls='experiment', access='secondary via Catalano fig. 3',
            points=[dict(Re=re, Cd=cd) for re, cd in curve],
            note='the red curve is labeled "Achenbach (1968)" by '
                 'Catalano et al.; kept only inside Achenbach\'s '
                 'measured range 6e4 <= Re <= 5e6 (JFM 34:625-639); '
                 'plot as a faint line, not symbols'),
    ))
    mid = [p for p in pts if 8e5 < p[0] < 1.3e6][0]
    assert abs(mid[1] - 0.31) < 0.01, f'Table-1 cross-check failed: {mid}'
    print(f'  catalano WMLES pts: {pts}')


def d_rodriguez2015_fig4():
    """Rodriguez et al. 2015 preprint fig 4 (p. 12, vector): Schewe 1983
    '+' (H+V segment pairs) and Achenbach & Heinecke 1981 open squares;
    Delany & Sorensen 'x' kept as a cross-check series only.
    Calibration from the decade labels 10^4..10^7 and y labels 0..1.4."""
    f = os.path.join(LIT, 'rodriguez2015_ijhff.pdf')
    page = fitz.open(f)[11]
    words = page.get_text('words')
    # x calibration from DETECTED bottom-axis tick columns (gnuplot
    # draws them as short vertical segments ending at the frame line
    # y=432.5); the four decade ticks are the ones under the labels
    # '104'..'107' (label x-centers shifted <2 pt).
    labx = sorted((w[0] + w[2]) / 2 for w in words
                  if 434 < w[1] < 435 and w[4] in ('104', '105', '106',
                                                   '107'))
    assert len(labx) == 4, labx
    allt = sorted(set(round(s[0], 1) for s in all_segments(page)
                      if abs(s[0] - s[2]) < 0.01 and
                      abs(s[1] - s[3]) < 6 and
                      abs(max(s[1], s[3]) - 432.5) < 1.0))
    tick = [min(allt, key=lambda t: abs(t - lx)) for lx in labx]
    assert all(abs(t - lx) < 3 for t, lx in zip(tick, labx)), (tick, labx)
    calx = Cal(tick, [1e4, 1e5, 1e6, 1e7], log=True, tol=0.004,
               name='rodriguez x')
    yl = sorted((float((w[1] + w[3]) / 2), float(w[4])) for w in words
                if 160 < w[0] < 175 and 255 < w[1] < 440 and
                w[4] in ('0', '0.2', '0.4', '0.6', '0.8', '1', '1.2'))
    assert len(yl) == 7, yl
    caly = Cal([p for p, _ in yl], [v for _, v in yl], tol=0.02,
               name='rodriguez y')
    # plot frame interior (from the axis label extremes)
    x0, x1 = calx.inv(1e4), calx.inv(2e7)
    y1, y0 = caly.inv(0.0), caly.inv(1.4)
    LEG = (300, 248, 446.5, 306)  # in-plot legend box incl. the
    # sample glyph column at x ~ 420-445 (pt), measured
    segs = [s for s in segments(page)
            if x0 + 1 < min(s[0], s[2]) and max(s[0], s[2]) < x1 - 1 and
            y0 + 1 < min(s[1], s[3]) and max(s[1], s[3]) < y1 - 1 and
            not (LEG[0] < (s[0] + s[2]) / 2 < LEG[2] and
                 LEG[1] < (s[1] + s[3]) / 2 < LEG[3])]
    H = [s for s in segs if abs(s[1] - s[3]) < .01 and
         1 < abs(s[0] - s[2]) < 6]
    V = [s for s in segs if abs(s[0] - s[2]) < .01 and
         1 < abs(s[1] - s[3]) < 6]
    D = [s for s in segs if abs(abs(s[0] - s[2]) - abs(s[1] - s[3])) < .3
         and 1 < abs(s[0] - s[2]) < 6]
    def centers(A, B, tol=0.35):
        out = []
        for a in A:
            ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
            for b in B:
                cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                if abs(ca[0] - cb[0]) < tol and abs(ca[1] - cb[1]) < tol:
                    out.append(((ca[0] + cb[0]) / 2, (ca[1] + cb[1]) / 2))
        return out
    plus = centers(H, V)
    cross = centers([d for d in D if (d[2]-d[0])*(d[3]-d[1]) > 0],
                    [d for d in D if (d[2]-d[0])*(d[3]-d[1]) < 0])
    # Spitzer 1965 '*' glyphs = H+V+diagonals through one center; they
    # alias as '+' AND as 'x' -- drop any center that has both
    star = [pp for pp in plus if any(abs(pp[0] - c[0]) < 0.35 and
                                     abs(pp[1] - c[1]) < 0.35
                                     for c in cross)]
    plus = [pp for pp in plus if pp not in star]
    cross = [c for c in cross if not any(abs(c[0] - s[0]) < 0.35 and
                                         abs(c[1] - s[1]) < 0.35
                                         for s in star)]
    # open squares: single-'re' stroke drawings, small (Achenbach &
    # Heinecke); Wieselsberger 1922 open triangles ('l','l','l' strokes)
    # kept as a cross-check of our own TN-84 digitization
    sq, wtri = [], []
    for d in page.get_drawings():
        r = d['rect']
        inplot = (x0 < r.x0 and r.x1 < x1 and y0 < r.y0 and r.y1 < y1 and
                  not (LEG[0] < r.x0 < LEG[2] and LEG[1] < r.y0 < LEG[3]))
        if not (inplot and 2 < r.width < 6 and 2 < r.height < 6):
            continue
        kinds = tuple(it[0] for it in d['items'])
        if d['type'] == 's' and kinds == ('re',):
            sq.append(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2))
        elif d['type'] == 's' and kinds == ('l', 'l', 'l'):
            wtri.append(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2))
    conv = lambda c: (float(calx(c[0])), float(caly(c[1])))
    schewe = sorted(conv(c) for c in plus)
    ah81 = sorted(conv(c) for c in sq)
    dscheck = sorted(conv(c) for c in cross)
    wiescheck = sorted(conv(c) for c in wtri)
    marks = [(x, y, (0, 200, 60)) for x, y in plus]
    marks += [(x, y, (255, 120, 0)) for x, y in sq]
    marks += [(x, y, (60, 60, 255)) for x, y in cross]
    marks += [(x, y, (200, 0, 200)) for x, y in wtri]
    vec_check_png(f, 11, marks, 'check_rodriguez_fig4.png', zoom=3.0,
                  clip=(100, 250, 620, 500))
    dump('rodriguez2015_fig4_exp', dict(
        source='Rodriguez, Lehmkuhl, Chiva, Borrell & Oliva, Int. J. '
               'Heat Fluid Flow 55 (2015) 91-103, accepted-manuscript '
               'PDF (UPCommons, see README), Fig. 4 (p. 12); vector '
               f'extraction {TODAY}. Experimental points as digitized '
               'by Rodriguez et al. -- SECONDARY provenance.',
        method='vector: "+" = paired H+V 1-item segments (Schewe 1983, '
               'piezobalance, JFM 133:265-285); open small 4-segment '
               'squares = Achenbach & Heinecke 1981 (JFM 109); "x" '
               'diagonal pairs = Delany & Sorensen kept ONLY as a '
               'cross-check of our own TN-3038 digitization; legend '
               'box masked; check checks/check_rodriguez_fig4.png',
        calibration=dict(x=calx.report(), y=caly.report()),
        schewe1983=dict(cls='experiment',
                        access='secondary via Rodriguez 2015 fig. 4',
                        points=[dict(Re=r, Cd=c) for r, c in schewe]),
        achenbach_heinecke1981=dict(
            cls='experiment', access='secondary via Rodriguez 2015 fig. 4',
            points=[dict(Re=r, Cd=c) for r, c in ah81]),
        delany_sorensen_crosscheck=dict(
            cls='experiment', access='secondary; cross-check only',
            points=[dict(Re=r, Cd=c) for r, c in dscheck]),
        wieselsberger_crosscheck=dict(
            cls='experiment', access='secondary; cross-check only',
            points=[dict(Re=r, Cd=c) for r, c in wiescheck]),
    ))
    print(f'  schewe+: {len(schewe)}  A&H sq: {len(ah81)} '
          f' D&S x: {len(dscheck)}  Wies tri: {len(wiescheck)}')


def d_veysey_fig7():
    """Veysey & Goldenfeld (arXiv physics/0609138) Fig. 7 (p. 25,
    vector): low-Re cylinder drag, y = Cd*R/(4pi) vs R (linear axes).
    Red 4-bezier open circles = Finn 1953; black filled 4-bezier =
    Tritton 1959; blue 'x' segment pairs = Jayaweera & Mason 1965.
    Main panel only (inset masked); Cd = 4*pi*y/R."""
    f = os.path.join(LIT, 'wieselsberger_replot_physics0609138.pdf')
    page = fitz.open(f)[24]
    words = page.get_text('words')
    xl = sorted((float((w[0] + w[2]) / 2), float(w[4])) for w in words
                if w[4] in tuple('0123456') and 323 < w[1] < 337)
    assert len(xl) == 7, xl
    calx = Cal([p for p, _ in xl], [v for _, v in xl], tol=0.02,
               name='veysey x')
    yl = sorted((float((w[1] + w[3]) / 2), float(w[4])) for w in words
                if w[2] < 168 and 85 < w[1] < 322 and
                w[4] in ('0', '0.2', '0.4', '0.6', '0.8', '1', '1.2', '1.4'))
    assert len(yl) == 8, yl
    caly = Cal([p for p, _ in yl], [v for _, v in yl], tol=0.02,
               name='veysey y')
    x0, x1 = calx.inv(0) - 2, calx.inv(6) + 4
    y1, y0 = caly.inv(0), caly.inv(1.4) - 3
    INSET = (280, 200, 465, 310)   # measured inset box (pt)
    LEG = (166, 92, 278, 162)
    def inside(cx, cy):
        return (x0 < cx < x1 and y0 < cy < y1 and
                not (INSET[0] < cx < INSET[2] and INSET[1] < cy < INSET[3])
                and not (LEG[0] < cx < LEG[2] and LEG[1] < cy < LEG[3]))
    finn, tritton, marks = [], [], []
    for d in page.get_drawings():
        r = d['rect']
        cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
        if not (len(d['items']) == 4 and r.width < 6 and inside(cx, cy)):
            continue
        col = tuple(d.get('color') or ())
        fill = tuple(d.get('fill') or ())
        if d['type'] == 's' and col and col[0] > 0.7:
            finn.append((cx, cy)); marks.append((cx, cy, (255, 120, 0)))
        elif d['type'] == 'f' and fill == (0.0, 0.0, 0.0):
            tritton.append((cx, cy)); marks.append((cx, cy, (0, 200, 60)))
    segs = [s for s in segments(page)
            if s[4] == (0.0, 0.0, 1.0) and inside((s[0]+s[2])/2,
                                                  (s[1]+s[3])/2)]
    D1 = [s for s in segs if (s[2]-s[0])*(s[3]-s[1]) > 0]
    D2 = [s for s in segs if (s[2]-s[0])*(s[3]-s[1]) < 0]
    jaya = []
    for a in D1:
        ca = ((a[0]+a[2])/2, (a[1]+a[3])/2)
        for b in D2:
            cb = ((b[0]+b[2])/2, (b[1]+b[3])/2)
            if abs(ca[0]-cb[0]) < .4 and abs(ca[1]-cb[1]) < .4:
                jaya.append(ca); marks.append((ca[0], ca[1], (60, 60, 255)))
    def conv(cs):
        out = []
        for cx, cy in cs:
            R, Y = float(calx(cx)), float(caly(cy))
            if R > 0.05:
                out.append((R, 4 * np.pi * Y / R))
        return sorted(out)
    vec_check_png(f, 24, marks, 'check_veysey_fig7.png', zoom=3.0,
                  clip=(80, 80, 520, 380))
    dump('veysey_fig7_lowre', dict(
        source='Veysey & Goldenfeld, Rev. Mod. Phys. 79 (2007) 883, '
               'arXiv:physics/0609138v2 Fig. 7 (p. 25); their replot of '
               'Finn 1953 (J. Appl. Phys. 24), Tritton 1959 (JFM 6:547) '
               'and Jayaweera & Mason 1965 (JFM 22) -- SECONDARY '
               f'provenance; vector extraction {TODAY}.',
        method='vector; y axis is Cd*R/(4pi), converted to Cd by '
               'Cd=4*pi*y/R (linear axes; points with R<=0.05 dropped '
               'as division-noise); inset and legend masked; check '
               'checks/check_veysey_fig7.png',
        calibration=dict(x=calx.report(), y=caly.report()),
        finn1953=dict(cls='experiment', access='secondary via V&G fig. 7',
                      points=[dict(Re=r, Cd=c) for r, c in conv(finn)]),
        tritton1959=dict(cls='experiment',
                         access='secondary via V&G fig. 7',
                         points=[dict(Re=r, Cd=c) for r, c in conv(tritton)]),
        jayaweera_mason1965=dict(
            cls='experiment', access='secondary via V&G fig. 7',
            points=[dict(Re=r, Cd=c) for r, c in conv(jaya)]),
    ))
    print(f'  finn {len(finn)}  tritton {len(tritton)}  jaya {len(jaya)}')


# -------------------------------------------------------- table datasets
def d_rodriguez2015_les():
    """Rodriguez et al. 2015 Table 2 (p. 16) -- direct transcription."""
    rows = [(2.5e5, 0.833), (3.8e5, 0.481), (5.3e5, 0.296),
            (6.5e5, 0.232), (7.2e5, 0.213), (8.5e5, 0.218)]
    dump('rodriguez2015_les', dict(
        source='Rodriguez, Lehmkuhl, Chiva, Borrell & Oliva, "On the '
               'flow past a circular cylinder from critical to super-'
               'critical Reynolds numbers: wake topology and vortex '
               'shedding", Int. J. Heat Fluid Flow 55 (2015) 91-103; '
               'TABLE 2 (accepted-manuscript p. 16), transcribed '
               f'{TODAY} (no digitization -- tabulated values).',
        method='transcription of Table 2 column CD',
        les=dict(cls='LES', access='primary',
                 points=[dict(Re=r, Cd=c) for r, c in rows]),
    ))
    print('  rodriguez table2: 6 pts')


def _assert_in_pdf(pdf, pageno, needles):
    """Transcription QA: every needle string must appear verbatim in the
    PDF page text (guards against transcription typos)."""
    page = fitz.open(os.path.join(LIT, pdf))[pageno]
    txt = page.get_text()
    missing = [n for n in needles if n not in txt]
    assert not missing, f'{pdf} p{pageno}: not found in text: {missing}'


def d_qu2013_dns():
    """Qu, Norberg, Davidson, Peng & Wang 2013 Table 3 (author ms.
    p. 19) -- direct transcription of the 2-D unsteady DNS-class sweep
    (finite volume, mesh 386x322, dt=0.01), mean drag CD vs Re. Every
    value is asserted verbatim against the PDF text. Re=150 appears
    twice (domain H=100 and H=160); the larger-domain row is the
    plotted point, both are kept."""
    rows = [  # (Re, H, CD)
        (50, 200, 1.397), (60, 160, 1.377), (80, 160, 1.336),
        (100, 120, 1.317), (120, 120, 1.306), (150, 100, 1.305),
        (150, 160, 1.301), (180, 100, 1.310), (200, 100, 1.316)]
    _assert_in_pdf('qu2013_jfs.pdf', 18,
                   [f'{cd:.3f}' for _, _, cd in rows] +
                   ['Table 3: Global results'])
    plotted = [(re, cd) for re, h, cd in rows if (re, h) != (150, 100)]
    dump('qu2013_dns', dict(
        source='Qu, Norberg, Davidson, Peng & Wang, "Quantitative '
               'numerical analysis of flow past a circular cylinder at '
               'Reynolds number between 50 and 200", J. Fluids Struct. '
               '39 (2013) 347-370, doi 10.1016/j.jfluidstructs.2013.'
               '02.007; TABLE 3 (author-ms. p. 19), transcribed '
               f'{TODAY} (no digitization -- tabulated values, each '
               'asserted verbatim against the PDF text).',
        method='transcription of Table 3 column CD (mesh 386x322, '
               'dt=0.01, domain H per row)',
        caveats='their own Table 2 domain study at Re=100 puts the '
                'H-dependence below 1% (H=200: 1.310 vs H=120: 1.317); '
                'at Re=150 the H=160 row (1.301) is the plotted point, '
                'the H=100 row (1.305) kept in all_rows',
        dns2d=dict(cls='2-D unsteady DNS-class (laminar shedding)',
                   access='primary',
                   points=[dict(Re=r, Cd=c) for r, c in plotted]),
        all_rows=[dict(Re=r, H=h, Cd=c) for r, h, c in rows],
    ))
    print(f'  qu2013 table3: {len(plotted)} plotted pts '
          f'({len(rows)} rows incl. Re=150 domain pair)')


def d_dong2005_dns3d():
    """Dong & Karniadakis 2005 Table 2 (p. 524 = pdf page 5) -- direct
    transcription: 3-D spectral DNS of the stationary cylinder at
    Re=10,000, mean drag Cd across their resolution study. Plotted as
    a vertical tick spanning the spanwise-resolved cases (Nz>=64);
    coarse-spanwise cases (Nz<=32) kept separately."""
    cases = [  # (case, P, Nz, K, Cd)
        ('DNS-A1', 5, 16, 6272, 1.155), ('DNS-A2', 5, 64, 6272, 1.110),
        ('DNS-A3', 5, 128, 6272, 1.128), ('DNS-B1', 5, 32, 9272, 1.208),
        ('DNS-B2', 4, 64, 9272, 1.120), ('DNS-B3', 5, 128, 9272, 1.143)]
    _assert_in_pdf('dong_karniadakis2005_jfs.pdf', 5,
                   [c for c, *_ in cases] + ['Table 2'])
    # values are typeset as "1:155" (colon decimal sep in this PDF font)
    _assert_in_pdf('dong_karniadakis2005_jfs.pdf', 5,
                   [f'{cd:.3f}'.replace('.', ':') for *_, cd in cases])
    fine = [cd for _, _, nz, _, cd in cases if nz >= 64]
    dump('dong_karniadakis2005_dns3d', dict(
        source='Dong & Karniadakis, "DNS of flow past a stationary and '
               'oscillating cylinder at Re=10000", J. Fluids Struct. 20 '
               '(2005) 519-531, doi 10.1016/j.jfluidstructs.2005.02.004; '
               f'TABLE 2 (p. 524), transcribed {TODAY} (no digitization '
               '-- tabulated values, asserted against the PDF text, '
               'whose font renders the decimal point as a colon).',
        method='transcription of Table 2 column Cd (stationary '
               'cylinder, Re=10,000, spectral/Fourier 3-D DNS)',
        caveats='plotted as a vertical tick spanning the Nz>=64 cases '
                '(1.110-1.143, finest DNS-B3 = 1.143); coarse-spanwise '
                'Nz<=32 cases (1.155, 1.208) excluded from the tick',
        dns3d=dict(cls='3-D DNS', access='primary',
                   points=[dict(Re=1.0e4, Cd=c, case=n)
                           for n, _, nz, _, c in cases if nz >= 64],
                   Cd_min=min(fine), Cd_max=max(fine), Cd_finest=1.143),
        all_rows=[dict(case=n, P=p, Nz=nz, K=k, Cd=c)
                  for n, p, nz, k, c in cases],
    ))
    print(f'  dong2005 table2: {len(fine)} resolved cases, '
          f'tick {min(fine)}-{max(fine)}')


# -------------------------------------------------------- raster datasets
def _src_image(pdf, pageno, imgidx, out, dpi=None):
    """Extract an embedded raster (or render the page at dpi) into
    litdata/dragcrisis/src/; reused if present."""
    src = os.path.join(LIT, 'src')
    os.makedirs(src, exist_ok=True)
    path = os.path.join(src, out)
    if not os.path.exists(path):
        doc = fitz.open(os.path.join(LIT, pdf))
        if dpi is None:
            xref = doc[pageno].get_images(full=True)[imgidx][0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n > 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
        else:
            pix = doc[pageno].get_pixmap(dpi=dpi)
        pix.save(path)
        print(f'  extracted {out} {pix.width}x{pix.height}')
    return path


def _runs(flags, gap=1):
    """Group near-consecutive True indices; return centers."""
    out = []
    for i, f in enumerate(flags):
        if f:
            if out and i - out[-1][-1] <= gap:
                out[-1].append(i)
            else:
                out.append([i])
    return [float(np.mean(r)) for r in out]


def d_iop2020_models():
    """Stabnikov & Garbaruk 2020 (J.Phys.Conf.Ser. 1697:012224, open
    access) fig. 3 LEFT panel, embedded raster 468x308: 3D URANS Cd(Re)
    sweeps -- SST (fully turbulent, orange), SST gamma-Re_theta
    (green), SST KD algebraic transition (blue) -- plus their replot of
    Schewe 1983 (black dots, their ref [15]).  k-omega KD (red) skipped
    (superseded baseline; noted).  Right panel (DDES hybrids) skipped:
    scale-resolving, not RANS.
    Calibration: DETECTED inner major ticks x=137px (1e5), 341px (1e6)
    -> 204 px/decade (log); left ticks 277..93px = 0.2..1.2 (linear).
    Legend box masked (measured px box)."""
    from scipy import ndimage
    p = _src_image('iop2020_transition_ddes.pdf', 5, 0, 'iop_fig3_left.png')
    img = Image.open(p).convert('RGB')
    im = np.array(img).astype(int)
    h, w, _ = im.shape
    dark = im.sum(2) < 250
    # tick detection (ritual: measure, don't assume)
    bt = [c for c in _runs(dark[281:287, :].sum(0) >= 4)
          if 40 < c < 430]
    assert len(bt) == 2 and abs(bt[0] - 137) < 2 and abs(bt[1] - 341) < 2, bt
    calx = Cal(bt, [1e5, 1e6], log=True, name='iop x')
    lt = [r for r in _runs(dark[:, 31:36].sum(1) >= 4) if 40 < r < 285]
    assert len(lt) == 6, lt
    caly = Cal(lt, [1.2, 1.0, 0.8, 0.6, 0.4, 0.2], name='iop y', tol=0.01)
    FRAME = (30, 30, 438, 287)          # x0,y0,x1,y1
    LEG = (248, 38, 437, 135)           # measured legend box (incl Exp row)
    def inplot(x, y):
        return (FRAME[0] + 2 < x < FRAME[2] - 2 and
                FRAME[1] + 2 < y < FRAME[3] - 2 and
                not (LEG[0] < x < LEG[2] and LEG[1] < y < LEG[3]))
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    series = {
        'sst': (r > 180) & (g > 90) & (g < 200) & (b < 110),      # orange
        'sst_gamma_retheta': (g > 140) & (r < 160) & (b < 170),   # green
        'sst_kd': (b > 180) & (r < 130),                          # blue
    }
    out = {}
    marks, rej = [], []
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    for name, mask in series.items():
        # marker cores: a 3x3 cross fits inside the 5-6 px markers but
        # not inside the ~2 px connecting lines (which fail the 3-tall
        # vertical arm for near-horizontal runs)
        er = ndimage.binary_erosion(mask, cross)
        pts = []
        for cx, cy, a, hh, ww, sl in blobs(er, 2, 200):
            if inplot(cx, cy):
                pts.append((float(calx(cx)), float(caly(cy))))
                marks.append((cx, cy))
            else:
                rej.append((cx, cy))
        out[name] = sorted(pts)
    # Schewe replot: black dots (unsaturated, dark), frame/ticks removed
    sat = (np.abs(r - g) + np.abs(r - b) + np.abs(g - b))
    blk = dark & (sat < 90)
    blk[FRAME[1] - 2:FRAME[1] + 3, :] = False
    blk[FRAME[3] - 8:, :] = False
    blk[:, FRAME[0] - 2:FRAME[0] + 8] = False
    blk[:, FRAME[2] - 8:] = False
    blk = ndimage.binary_erosion(blk, np.ones((3, 3)))
    pts = []
    for cx, cy, a, hh, ww, sl in blobs(blk, 1, 30):
        if inplot(cx, cy) and hh <= 8 and ww <= 8:
            pts.append((float(calx(cx)), float(caly(cy))))
            marks.append((cx, cy))
        else:
            rej.append((cx, cy))
    out['schewe1983_replot'] = sorted(pts)
    raster_check(img, marks, 'check_iop2020.png', rej=rej, r=5, scale=2.0)
    dump('iop2020_models', dict(
        source='Stabnikov & Garbaruk, "Prediction of drag crisis on a '
               'circular cylinder using a new algebraic transition '
               'model coupled with SST DDES", J. Phys.: Conf. Ser. 1697 '
               '(2020) 012224 (OPEN ACCESS), Fig. 3 left panel (PDF '
               f'p. 6, embedded 468x308 raster); digitized {TODAY}.',
        method='raster color classification, 3x3 erosion isolates the '
               'markers from the connecting 1-2 px lines; NOTE marker '
               'centers only -- the connecting polylines are not '
               'sampled; black Exp dots = the authors\' replot of '
               'Schewe 1983 [their ref 15] -- SECONDARY, kept for '
               'cross-checking the vector Schewe set; k-omega KD '
               'series and the right (DDES) panel deliberately '
               'skipped; check checks/check_iop2020.png',
        calibration=dict(x=calx.report(), y=caly.report(),
                         ticks=dict(bottom_px=bt, left_px=lt)),
        sst_fully_turbulent=dict(
            cls='RANS fully turbulent', access='primary',
            points=[dict(Re=re, Cd=cd) for re, cd in out['sst']]),
        sst_gamma_retheta=dict(
            cls='RANS transition model', access='primary',
            points=[dict(Re=re, Cd=cd)
                    for re, cd in out['sst_gamma_retheta']]),
        sst_kd=dict(
            cls='RANS transition model', access='primary',
            points=[dict(Re=re, Cd=cd) for re, cd in out['sst_kd']]),
        schewe1983_replot=dict(
            cls='experiment', access='secondary via Stabnikov-Garbaruk',
            points=[dict(Re=re, Cd=cd)
                    for re, cd in out['schewe1983_replot']]),
    ))
    print({k: len(v) for k, v in out.items()})


def d_stringer2014():
    """Stringer, Zang & Hillis 2014 (Ocean Eng. 87:1-9, green-OA author
    manuscript via CORE) fig. 5 (scan p. 15, embedded 1240x1753 plot
    layer without text): 2D URANS k-omega SST sweep, Re = 40, 100, 1e3,
    1e4, 1e5, 1e6 -- CFX open squares, OpenFOAM filled triangles.
    Log-log axes; calibration from DETECTED decade gridlines (x: six
    interior decade lines 10..1e6; y: rows Cd=10/1/0.1); symbols
    classified solid-triangle vs open-square by interior fill."""
    from scipy import ndimage
    p = _src_image('stringer2014_oceaneng.pdf', 14, 0, 'stringer_fig5.png')
    img = Image.open(p).convert('L')
    im = np.array(img)
    grid = im < 200
    sub = grid[185:676]
    colfrac = sub.sum(0) / sub.shape[0]
    cand = _runs(colfrac > 0.5)
    # decade columns: the six pairs (9-line, decade-line) end at the
    # decade line; select candidates nearest to an even 7-decade comb
    # anchored on the frame edges
    rowfrac = grid[:, 200:950].sum(1) / 750.0
    rows = _runs(rowfrac > 0.5)
    assert len(rows) == 3, rows          # Cd = 10, 1, 0.1
    caly = Cal(rows, [10.0, 1.0, 0.1], log=True, tol=0.01,
               name='stringer y')
    dec = []
    for k in range(1, 7):                # 10^1 .. 10^6 interior lines
        guess = 193.4 + k * 108.65       # frame Re=1 at ~193, 1e7 ~954
        near = [c for c in cand if abs(c - guess) < 8]
        assert near, (k, guess)
        dec.append(min(near, key=lambda c: abs(c - guess)))
    calx = Cal(dec, [10.0 ** k for k in range(1, 7)], log=True, tol=0.01,
               name='stringer x')
    # The run Re are STATED in the paper (40/100/1e3/1e4/1e5/1e6), so
    # only Cd is digitized: NCC template match (open 13x13 square /
    # solid 11x10 triangle, the fig14a ritual) inside a +-20 px window
    # around each expected Re column.  The scan carries a small local
    # skew (+5..8 px between the averaged gridline columns and the
    # symbols); windows absorb it, Cd uncertainty ~2%% noted in JSON.
    from scipy.signal import fftconvolve
    def ncc(imgf, t):
        tm = t - t.mean()
        ones = np.ones_like(t)
        s1 = fftconvolve(imgf, ones, mode='same')
        s2 = fftconvolve(imgf ** 2, ones, mode='same')
        num = fftconvolve(imgf, tm[::-1, ::-1], mode='same')
        var = np.maximum(s2 - s1 ** 2 / t.size, 1e-6)
        return num / np.sqrt(var * (tm ** 2).sum())
    darkf = (im < 160).astype(float)
    sqt = np.zeros((13, 13))
    sqt[:2, :] = sqt[-2:, :] = sqt[:, :2] = sqt[:, -2:] = 1
    trit = np.zeros((10, 11))
    for rr in range(10):
        wdt = int(1 + rr / 9 * 10)
        trit[rr, (11 - wdt) // 2:(11 - wdt) // 2 + wdt] = 1
    smaps = {'sq': ncc(darkf, sqt), 'tri': ncc(darkf, trit)}
    y0, y1 = int(caly.inv(10.0)) + 8, int(caly.inv(0.1)) - 8
    # Re=40 is NOT digitized: both symbols coincide there and Table 3
    # (p. 13) tabulates CD=1.55 for both solvers -- transcribed.
    # OpenFOAM at Re=1e5 is NOT in fig. 5 at all (its total Cd ~0.08
    # from the fig. 6 components is below the 0.1 axis floor) -- the
    # naive window match latches onto a Massey-curve dash; skipped.
    RES = [100.0, 1e3, 1e4, 1e5, 1e6]
    SKIP = {('tri', 1e5)}
    found = {'sq': [], 'tri': []}
    marks, rej = [], []
    for re_v in RES:
        xc = int(calx.inv(re_v)) + 6
        for tag in ('sq', 'tri'):
            win = smaps[tag][y0:y1, xc - 20:xc + 21]
            i = int(np.argmax(win))
            r, c = np.unravel_index(i, win.shape)
            v = float(win[r, c])
            if (tag, re_v) in SKIP:
                rej.append((c + xc - 20, r + y0))
                continue
            if v >= 0.5:
                found[tag].append((re_v, float(caly(r + y0)), v))
                marks.append((c + xc - 20, r + y0))
            else:
                rej.append((xc, (y0 + y1) / 2))
                print(f'  MISS {tag} at Re={re_v:g} (ncc {v:.2f})')
    cfx = [(40.0, 1.55)] + [(re_v, cd) for re_v, cd, v in found['sq']]
    of = [(40.0, 1.55)] + [(re_v, cd) for re_v, cd, v in found['tri']]
    raster_check(Image.open(p).convert('RGB'), marks,
                 'check_stringer.png', rej=rej, r=14)
    dump('stringer2014_urans', dict(
        source='Stringer, Zang & Hillis, "Unsteady RANS computations '
               'of flow around a circular cylinder for a wide range of '
               'Reynolds numbers", Ocean Eng. 87 (2014) 1-9; author-'
               'accepted manuscript (Bath/CORE, see README), Fig. 5 '
               f'(p. 15 scan, embedded plot layer); digitized {TODAY}. '
               'Computed points at Re = 40/100/1e3/1e4/1e5/1e6; Re=40 '
               'laminar, Re=100 laminar shedding, SST above.',
        method='raster: decade-gridline calibration (detected, log-log)'
               '; Re taken from the stated run matrix, Cd digitized by '
               'NCC template match (open square = CFX, solid triangle '
               '= OpenFOAM) in +-20 px windows around each Re column; '
               'scan skew +5-8 px absorbed by the windows, residual Cd '
               'uncertainty ~2%; symbols at Re=40/100 overlap each '
               'other and the reference curves -- verify in '
               'checks/check_stringer.png',
        calibration=dict(x=calx.report(), y=caly.report(),
                         x_decades_px=dec, y_decades_px=rows),
        cfx=dict(cls='RANS fully turbulent (2D URANS, CFX SST)',
                 access='primary',
                 points=[dict(Re=re, Cd=cd) for re, cd in cfx]),
        openfoam=dict(cls='RANS fully turbulent (2D URANS, OF SST)',
                      access='primary',
                      points=[dict(Re=re, Cd=cd) for re, cd in of]),
        caveats='Re=40 for BOTH solvers transcribed from Table 3 '
                '(CD=1.55/1.55), not digitized (symbols coincide); '
                'Re=100 symbols overlap each other and the reference '
                'curves -- solver attribution there may be swapped '
                '(both lie in 1.2-1.45); OpenFOAM Re=1e5 is absent '
                'from fig. 5 (total Cd ~0.08 is below the 0.1 axis '
                'floor; see their fig. 6 components) and is NOT '
                'included; Re<=100 runs are laminar (no SST).',
    ))
    print(f'  CFX sq: {len(cfx)}  OF tri: {len(of)}  rej: {len(rej)}')
    print('  cfx:', [(round(r), round(c, 3)) for r, c in cfx])
    print('  of :', [(round(r), round(c, 3)) for r, c in of])


def d_henderson1995():
    """Henderson 1995 (Phys. Fluids 7:2102, open Caltech copy) Fig. 1
    (p. 3 scan, rendered at 300 dpi): 2-D spectral-element pressure and
    viscous drag coefficients, steady branch (filled symbols, Re<~47)
    and shedding-mean branch (open symbols).  Pressure (triangles) and
    viscous (circles) bands are separated by Cd=0.8, so class = band;
    filled/open = center fill.  Totals = pressure + viscous paired at
    equal Re (3% log tolerance).  Frame + tick calibration detected;
    axis-label text boxes masked (measured)."""
    from scipy import ndimage
    p = _src_image('henderson1995_pof.pdf', 2, None,
                   'henderson_fig1.png', dpi=300)
    img = Image.open(p).convert('L')
    im = np.array(img)
    dark = im < 150
    # frame detection: strongest line row/col inside search windows
    # (the box strokes are thin and broken in the scan -- peak dark
    # fraction ~0.45-0.5, so argmax-in-window instead of threshold)
    rf = dark[:, 600:2000].sum(1) / 1400.0
    cf = dark[150:1250, :].sum(0) / 1100.0
    def peak(frac, a, b):
        i = int(np.argmax(frac[a:b])) + a
        assert frac[i] > 0.28, (a, b, frac[i])
        return i
    top, bot = peak(rf, 140, 400), peak(rf, 1100, 1400)
    lef, rig = peak(cf, 500, 700), peak(cf, 1900, 2100)
    # ticks point INWARD in this figure: major x ticks = columns dark
    # over >=13 of the 18 rows just above the bottom frame
    above = dark[bot - 20:bot - 2, :].sum(0)
    cand = [c for c in _runs(above >= 13) if lef + 10 < c < rig - 10]
    # the candidate list also contains minor log ticks and the two
    # dashed Re_1/Re_2 lines; the decade ticks (100, 1000) are the
    # pair forming an even comb with the left frame (= Re 10)
    xt = None
    for i, c1 in enumerate(cand):
        for c2 in cand[i + 1:]:
            if abs((c1 - lef) - (c2 - c1)) < 4:
                xt = [float(lef), c1, c2]
    assert xt, cand
    calx = Cal(xt, [10.0, 100.0, 1000.0], log=True, tol=0.01,
               name='henderson x')
    # major y ticks: rows dark over >=13 of the 18 cols right of the
    # left frame (interior ticks); 0 and 2 coincide with the frame
    right_of = dark[:, lef + 2:lef + 20].sum(1)
    ycand = [r for r in _runs(right_of >= 13) if top - 6 < r < bot + 6]
    # candidates include 0.1-step minor ticks below 0.5; the majors
    # are the even 5-comb between the extreme candidates
    exp = np.linspace(min(ycand), max(ycand), 5)
    yt = [min(ycand, key=lambda c: abs(c - e)) for e in exp]
    assert all(abs(t - e) < 5 for t, e in zip(yt, exp)), (yt, exp)
    caly = Cal(yt, [2.0, 1.5, 1.0, 0.5, 0.0], tol=0.02,
               name='henderson y')
    # symbol mask: interior minus frame margins and label-text boxes
    blk = dark.copy()
    blk[:top + 4, :] = False
    blk[bot - 3:, :] = False
    blk[:, :lef + 4] = False
    blk[:, rig - 3:] = False
    for x0, y0, x1, y1 in ((620, 240, 810, 310),    # 'Total'
                           (590, 420, 890, 495),    # 'Pressure'
                           (630, 860, 870, 935)):   # 'Viscous'
        blk[y0:y1, x0:x1] = False
    # the dotted fit lines chain into the symbols: close the symbol
    # rings (fill_holes), then a 7x7 opening removes the ~4 px dots
    # and the chain links while the ~15 px symbol discs survive
    filled_m = ndimage.binary_fill_holes(blk)
    opened = ndimage.binary_opening(filled_m, np.ones((7, 7)))
    pres, visc, rej = [], [], []
    for cx, cy, a, hh, ww, sl in blobs(opened, 60, 900):
        if not (9 <= hh <= 30 and 9 <= ww <= 30):
            rej.append((cx, cy))
            continue
        iy, ix = int(round(cy)), int(round(cx))
        core = blk[iy - 2:iy + 3, ix - 2:ix + 3]
        filled = core.mean() > 0.55
        re_v, cd = float(calx(cx)), float(caly(cy))
        (pres if cd > 0.8 else visc).append((re_v, cd, filled, cx, cy))
    # pair pressure + viscous at equal Re
    totals, comps = [], []
    used = set()
    for re_p, cd_p, fil_p, *_ in sorted(pres):
        best = None
        for j, (re_v, cd_v, fil_v, *_) in enumerate(sorted(visc)):
            if j in used or fil_p != fil_v:
                continue
            derr = abs(np.log10(re_p / re_v))
            if derr < 0.015 and (best is None or derr < best[0]):
                best = (derr, j, re_v, cd_v)
        if best:
            used.add(best[1])
            re_m = float(np.sqrt(re_p * best[2]))
            totals.append((re_m, cd_p + best[3], fil_p))
            comps.append(dict(Re=re_m, Cd_pressure=cd_p,
                              Cd_viscous=best[3],
                              branch='steady' if fil_p else 'shedding'))
    keep = [(x, y) for _, _, _, x, y in pres + visc]
    raster_check(Image.open(p).convert('RGB').crop((400, 100, 2200,
                                                    1700)),
                 [(x - 400, y - 100) for x, y in keep],
                 'check_henderson.png',
                 rej=[(x - 400, y - 100) for x, y in rej], r=13)
    totals.sort()
    dump('henderson1995', dict(
        source='Henderson, "Details of the drag curve near the onset '
               'of vortex shedding", Phys. Fluids 7(9) (1995) '
               '2102-2104 (open copy via Caltech CODA, see README), '
               f'Fig. 1 (scan p. 3 at 300 dpi); digitized {TODAY}.',
        method='raster: frame+tick detection (x log 10/100/1000, y '
               'linear 0..2); pressure/viscous split at Cd=0.8 (bands '
               'are disjoint in Fig. 1); filled(steady, Re<47) vs '
               'open(shedding mean) by symbol-core fill; totals = '
               'pressure+viscous paired at equal Re (1.5% log-Re '
               'tolerance); the drawn Total curve is NOT traced -- '
               'the summed points must lie on it in the check PNG '
               '(checks/check_henderson.png); dotted fit-line dots '
               'are below the blob size window',
        calibration=dict(x=calx.report(), y=caly.report(),
                         frame=dict(top=top, bottom=bot, left=lef,
                                    right=rig), xticks=xt, yticks=yt),
        totals=dict(cls='2-D spectral computation', access='primary',
                    points=[dict(Re=r, Cd=c,
                                 branch='steady' if f else 'shedding')
                            for r, c, f in totals]),
        components=comps,
        note='steady-branch totals (filled) are the exact steady '
             'solution; shedding-branch totals (open) are cycle '
             'means -- match our steady solver only below Re~47',
    ))
    ns = sum(1 for *_, f in totals if f)
    print(f'  pressure {len(pres)}  viscous {len(visc)}  totals '
          f'{len(totals)} ({ns} steady)  rej {len(rej)}')


def _strip_lines(mask, run=30):
    """Clear vertical and horizontal dark runs longer than `run` px
    (grid lines), keeping symbol-sized features."""
    out = mask.copy()
    for axis in (0, 1):
        m = mask if axis == 0 else mask.T
        keep = np.zeros_like(m)
        for c in range(m.shape[1]):
            col = m[:, c]
            d = np.diff(np.concatenate([[0], col.view(np.int8), [0]]))
            starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
            for s, e in zip(starts, ends):
                if e - s >= run:
                    keep[s:e, c] = True
        out &= ~(keep if axis == 0 else keep.T)
    return out


def d_roshko1961():
    """Roshko 1961 (JFM 10:345-356, open Caltech copy) Figure 2 (p. 4
    scan, embedded 1950x2924 raster): his CWT measurements Re ~1e6-1e7.
    Kept: plain circles/dots (p = 1/2/4 atm -- NOT distinguished, all
    'Roshko 1961').  Splitter-plate variants (flagged symbols, taller
    bbox from the tail) separated into their own series and NOT meant
    for the overlay; Dryden & Hill '+' excluded (size).  The Delany &
    Sorensen dashed curves reject as thin dashes.  Calibration from
    the full grid: 8 horizontal lines Cd = 0..1.4 and the four decade
    columns 1e4..1e7 (all detected)."""
    from scipy import ndimage
    p = _src_image('roshko1961_jfm.pdf', 3, 0, 'roshko_fig2.png')
    img = Image.open(p).convert('L')
    im = np.array(img)
    dark = im < 128
    rf = dark[:, 400:1600].sum(1) / 1200.0
    flags = np.zeros(im.shape[0], bool)
    flags[850:1900] = rf[850:1900] > 0.55
    hrows = _runs(flags)
    assert len(hrows) == 8, hrows
    caly = Cal(hrows, np.linspace(1.4, 0.0, 8), tol=0.01, name='roshko y')
    y0, y1 = int(hrows[0]), int(hrows[-1])
    cf = dark[y0 + 5:y1 - 5, :].sum(0) / float(y1 - y0 - 10)
    vcols = _runs(cf > 0.55)
    assert len(vcols) == 28, vcols
    dec = [vcols[0], vcols[9], vcols[18], vcols[27]]
    # decade identity check: the 2-line sits log10(2) into each decade
    W = (dec[3] - dec[0]) / 3.0
    assert abs((vcols[1] - dec[0]) / W - np.log10(2)) < 0.01
    calx = Cal(dec, [1e4, 1e5, 1e6, 1e7], log=True, tol=0.005,
               name='roshko x')
    # symbol scan region: Re >= 7e5, 0.15 <= Cd <= 1.0, legend and the
    # 'Roshko 1955 splitter' annotation masked
    blk = dark.copy()
    blk[:int(caly.inv(1.0)), :] = False
    blk[int(caly.inv(0.15)):, :] = False
    blk[:, :int(calx.inv(7e5))] = False
    blk[:, int(calx.inv(1.05e7)):] = False
    blk[950:1258, 1215:1645] = False        # legend box
    blk[1655:1715, 1200:1510] = False       # 'Delany & Sorensen' caption
    blk = _strip_lines(blk, run=25)
    # symbol CORES: close the rings, then double 3x3 erosion kills the
    # D&S dashes (4-5 px), grid-junction specks and the thin arms of
    # the Dryden&Hill '+', leaving only the >=10 px symbol discs
    filled_m = ndimage.binary_fill_holes(
        ndimage.binary_closing(blk, np.ones((3, 3))))
    core = ndimage.binary_erosion(filled_m, np.ones((3, 3)), iterations=2)
    # Every auto-detected blob was inspected by eye at 5-6x zoom
    # (2026-07-28; tile sheet + per-cluster crops) and hand-labelled:
    # the printed splitter-plate flags are 2-3 px tails no shape
    # heuristic separated reliably.  Keys = (Re/1e6, Cd) rounded to
    # (2,2); any detection without a label aborts (re-inspect!).
    MANUAL = {
        (0.82, 0.35): 'drop',      # dash-crossing knot, no symbol
        (1.81, 0.45): 'splitter',  # flagged (tail), z4
        (1.81, 0.43): 'plain',     # filled, solid fair-curve tangent
        (1.88, 0.46): 'splitter',  # flagged (tail), z4
        (2.22, 0.52): 'plain',     # open circle ON the D&S dash
        (2.29, 0.56): 'splitter',  # flagged, z3
        (2.67, 0.57): 'splitter',  # filled-flagged, z3
        (2.73, 0.59): 'splitter',  # flagged, z3
        (3.51, 0.69): 'plain',     # clean open circle left of '+'
        (3.64, 0.67): 'plain',     # open circle, '+' arm through
        (3.67, 0.73): 'plain',     # filled
        (4.19, 0.62): 'splitter',  # flagged 'b', z1
        (4.44, 0.74): 'plain',     # filled
        (5.03, 0.74): 'plain',     # filled (merged with adjacent open)
        (5.2, 0.68): 'plain',      # clean open circle, z1
        (5.21, 0.71): 'ambiguous', # merged open+flagged pair, z1
        (6.17, 0.7): 'plain',      # open
        (6.91, 0.68): 'plain',     # open
        (8.38, 0.71): 'plain',     # open, z2
        (8.66, 0.74): 'plain',     # open
    }
    pts, splitter, rej = [], [], []
    for cx, cy, a, hh, ww, sl in blobs(core, 4, 250):
        if not (3 <= ww <= 15 and 3 <= hh <= 15):
            rej.append((cx, cy))
            continue
        re_v, cd = float(calx(cx)), float(caly(cy))
        key = (round(re_v / 1e6, 2), round(cd, 2))
        lab = MANUAL.get(key)
        assert lab, f'unlabelled detection {key} -- re-inspect by eye'
        if lab == 'plain':
            pts.append((re_v, cd, cx, cy))
        elif lab == 'splitter':
            splitter.append((re_v, cd, cx, cy))
        else:
            rej.append((cx, cy))
    raster_check(Image.open(p).convert('RGB').crop((350, 900, 1700,
                                                    1900)),
                 [(x - 350, y - 900) for *_, x, y in pts],
                 'check_roshko.png',
                 rej=[(x - 350, y - 900) for x, y in rej] +
                     [(x - 350, y - 900) for *_, x, y in splitter], r=13)
    pts.sort()
    splitter.sort()
    dump('roshko1961', dict(
        source='Roshko, "Experiments on the flow past a circular '
               'cylinder at very high Reynolds number", J. Fluid Mech. '
               '10 (1961) 345-356 (open copy via Caltech CODA, see '
               f'README), Figure 2 (p. 348, scan p. 4); digitized '
               f'{TODAY}.',
        method='raster: calibration from the printed grid (8 rows '
               'Cd 0..1.4; 28 log-spaced columns, decade identity '
               'verified via the log10(2) spacing); grid lines '
               'stripped by >=25 px run removal; every detection '
               'hand-labelled at 5-6x zoom (MANUAL dict in the '
               'script: plain p=1/2/4 atm kept, splitter-plate '
               'flagged symbols separated, one merged pair dropped '
               'as ambiguous); Dryden & Hill + excluded; the two '
               'lowest-Re 1-atm points (Cd~0.3-0.35 near 1e6) were '
               'NOT recoverable (buried in the D&S dashed curves); '
               'check checks/check_roshko.png (green = kept)',
        calibration=dict(x=calx.report(), y=caly.report(),
                         h_rows=hrows, decade_cols=dec),
        roshko=dict(cls='experiment', access='primary',
                    points=[dict(Re=r, Cd=c) for r, c, *_ in pts]),
        splitter_plate=dict(
            cls='experiment (splitter plate -- do not overlay)',
            access='primary',
            points=[dict(Re=r, Cd=c) for r, c, *_ in splitter]),
        note='plain/splitter classes are HAND-verified per symbol '
             '(MANUAL dict); Cd accuracy limited by the ~13 px symbol '
             'size ~ +-0.01',
    ))
    print(f'  roshko plain {len(pts)}  splitter {len(splitter)} '
          f' rej {len(rej)}')


def _ncc_map(imgf, t):
    from scipy.signal import fftconvolve
    tm = t - t.mean()
    ones = np.ones_like(t)
    s1 = fftconvolve(imgf, ones, mode='same')
    s2 = fftconvolve(imgf ** 2, ones, mode='same')
    num = fftconvolve(imgf, tm[::-1, ::-1], mode='same')
    var = np.maximum(s2 - s1 ** 2 / t.size, 1e-6)
    return num / np.sqrt(var * (tm ** 2).sum())


class PwCal:
    """Piecewise-linear axis calibration through MANY detected grid
    lines (for photostat scans with non-uniform stretch); works in
    log10(value) space."""

    def __init__(self, pix, val, name=''):
        self.pix = np.asarray(pix, float)
        self.lv = np.log10(val)
        assert np.all(np.diff(self.pix) > 0) and \
            (np.all(np.diff(self.lv) > 0) or np.all(np.diff(self.lv) < 0))
        self.name = name

    def __call__(self, px):
        return 10.0 ** np.interp(px, self.pix, self.lv)

    def report(self):
        return dict(kind='piecewise-log', pix=self.pix.tolist(),
                    log10_val=self.lv.tolist())


def d_tn3038():
    """Delany & Sorensen, NACA TN 3038 (1953), Figure 5 (PDF p. 14,
    rendered 400 dpi): circular-cylinder Cd(Re), 1.1e4-2.3e6.  All
    three model sizes (12/4/1 inch, open / ticked / half-filled
    circles) are ONE dataset here.  The photostat grid is non-uniform
    (decade width drifts ~1.5%), so both axes use piecewise-linear
    calibration through EVERY detected grid line; the y-line identity
    was resolved by gap-ratio analysis (see comments) -- the row at
    ~1213 px is the Wieselsberger dashed curve, not a grid line.
    Symbols found by ring-template NCC + non-max suppression (they
    chain/overlap); Strouhal band (upper) excluded by Cd < 1.6 cut."""
    p = _src_image('delany_sorensen_tn3038.pdf', 13, None,
                   'tn3038_fig5.png', dpi=400)
    img = Image.open(p).convert('L')
    im = np.array(img)
    dark = im < 150
    h, w = dark.shape
    rowfrac = dark.sum(1) / w
    colfrac = dark.sum(0) / h
    rows = _runs(rowfrac > 0.22, gap=6)
    cols = _runs(colfrac > 0.22, gap=6)
    # 15 printed Cd lines incl. the unlabeled 0.9/0.7/0.5; the strong
    # row at ~1213 px is NOT a grid line (subcritical data chain +
    # Wieselsberger dashed curve at Cd~1.2) -- excluded; no 1.5 line
    # is printed (log-interp bridges 2..1)
    YV = [6, 5, 4, 3, 2, 1, .9, .8, .7, .6, .5, .4, .3, .2, .1]
    yrows = [r for r in rows if 400 < r < 2400 and abs(r - 1213) > 20]
    assert len(yrows) == 15, yrows
    caly = lambda r0: float(10.0 ** np.interp(
        r0, np.asarray(yrows, float), np.log10(np.asarray(YV, float))))
    XV = ([k * 1e4 for k in range(1, 10)] + [k * 1e5 for k in range(1, 10)]
          + [1e6, 2e6, 3e6])
    xcols = [c for c in cols if 700 < c < 3500]
    assert len(xcols) == 21, xcols
    calx = PwCal(xcols, XV, name='tn3038 x')
    # gap-ratio sanity: decade width from (1e4,1e5) vs (1e5,1e6)
    d1, d2 = xcols[9] - xcols[0], xcols[18] - xcols[9]
    assert abs(d1 - d2) < 0.02 * d1, (d1, d2)
    # ring-template NCC over the line-stripped mask
    strip = _strip_lines(dark, run=60)
    R, s = 15, 5
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    rr = np.sqrt(yy ** 2 + xx ** 2)
    ring = ((rr <= R) & (rr >= R - s)).astype(float)
    score = _ncc_map(strip.astype(float), ring)
    # valid region: inside frame, Cd < 1.6 (above = Strouhal band),
    # legend box and NACA logo masked
    y16 = int(np.interp(np.log10(1.45),
                        np.log10(np.asarray(YV[::-1], float)),
                        np.asarray(yrows[::-1], float)))
    valid = np.zeros_like(score, bool)
    valid[y16:int(yrows[-1]) - 5, int(xcols[0]) + 5:int(xcols[-1]) - 5] = True
    valid[1280:2060, 1290:2330] = False       # legend box (+Ref.7 row)
    valid[2150:2400, 2850:3400] = False       # NACA logo
    valid[1060:1350, 3030:3370] = False       # V->O flow inset
    sc = np.where(valid, score, -1.0)
    pts, marks = [], []
    while True:
        i = int(np.argmax(sc))
        r0, c0 = np.unravel_index(i, sc.shape)
        v = float(sc[r0, c0])
        if v < 0.27:
            break
        sc[max(0, r0 - 14):r0 + 15, max(0, c0 - 14):c0 + 15] = -1
        pts.append((float(calx(c0)), caly(r0), int(c0), int(r0),
                    round(v, 2)))
        marks.append((c0, r0))
    pts.sort()
    raster_check(Image.open(p).convert('RGB'), marks, 'check_tn3038.png',
                 r=16, scale=0.5)
    dump('tn3038_delany_sorensen', dict(
        source='Delany & Sorensen, "Low-speed drag of cylinders of '
               'various shapes", NACA TN 3038 (1953), Figure 5 (PDF '
               'p. 14, NTRS open PDF, see README); circular-cylinder '
               f'Cd only (circles; Strouhal band excluded); digitized '
               f'{TODAY}.',
        method='raster 400 dpi: piecewise-log calibration through all '
               '21 x / 13 y detected grid lines (photostat stretch '
               '~1.5% per decade); the Cd row identity was fixed by '
               'gap-ratio analysis (row 1213 px = Wieselsberger '
               'dashed curve, excluded); ring NCC (R=15, stroke 5) + '
               '14 px NMS, threshold 0.27; all three cylinder sizes '
               'merged into one series; check checks/check_tn3038.png',
        calibration=dict(x=calx.report(), y=dict(
            kind='piecewise-log', pix=yrows,
            log10_val=[float(np.log10(v)) for v in YV])),
        delany_sorensen=dict(
            cls='experiment', access='primary',
            points=[dict(Re=re, Cd=cd, ncc=v)
                    for re, cd, _, _, v in pts]),
        note='overlapping symbol chains: NMS keeps one center per '
             '14 px, so the densest chains are subsampled; hysteresis '
             'loops in the crisis region retained',
    ))
    print(f'  D&S circles: {len(pts)}')


def caly_inv_row(lab, YVALS, cd):
    """row for a given Cd on the TN-84 label-anchored y axis."""
    lv = np.log10(np.asarray(YVALS, float))[::-1]
    rows = np.asarray(lab, float)[::-1]
    return float(np.interp(np.log10(cd), lv, rows))


def d_tn84():
    """Wieselsberger 1921 via NACA TN-84 (1922 translation), Fig. 1
    (PDF p. 16, white-on-black photostat, rendered 400 dpi and rotated
    -90 deg): the classic 9-diameter composite c(Re), Re ~4-8e5.  All
    diameters merged into one series (the symbols are too degraded to
    classify).  Grid lines are patchy: calibration lines are found by
    LOCAL peak search around label-anchored estimates and fit
    globally (log); residuals reported -- expect ~2-3% Re / ~1-2% Cd,
    on top of the printing/photostat scatter of the 1921 original."""
    from scipy import ndimage
    p = _src_image('wieselsberger_tn84.pdf', 15, None, 'tn84_fig1.png',
                   dpi=400)
    img = Image.open(p).convert('L').rotate(-90, expand=True)
    a = np.array(img).astype(float)
    bg = ndimage.gaussian_filter(a, 25)
    bright = (a - bg) > 22
    h, w = bright.shape
    # local line search around label-anchored estimates (measured on
    # the rotated overview; the photostat grid is too patchy for
    # global fraction thresholds)
    def peak_col(est, half=30, band=(300, 2900)):
        fr = bright[band[0]:band[1], :].sum(0) / (band[1] - band[0])
        i = int(np.argmax(fr[est - half:est + half])) + est - half
        return i, float(fr[i])
    def peak_row(est, half=30, band=(400, 3900)):
        fr = bright[:, band[0]:band[1]].sum(1) / (band[1] - band[0])
        i = int(np.argmax(fr[est - half:est + half])) + est - half
        return i, float(fr[i])
    xest = {1.0: 786, 1e3: 2370, 1e4: 2915, 1e5: 3493, 1e6: 4010}
    xf = {v: peak_col(e) for v, e in xest.items()}
    xuse = {v: px for v, (px, fr) in xf.items() if fr > 0.15}
    assert len(xuse) == 5, xf
    calx = Cal(list(xuse.values()), list(xuse.keys()), log=True,
               tol=0.025, name='tn84 x')
    # y calibration from the LABEL GLYPH row-centers (the gridlines
    # themselves are too patchy); the label column is x 30-215; groups
    # of bright rows >= 12 tall are label lines, matched IN ORDER to
    # the printed values.  Cross-check: 20 must sit log10(2) below 40
    # (residual asserted); mid-plot photostat stretch (+10..19 px) is
    # absorbed by piecewise interpolation.
    strip = bright[:, 30:215].sum(1)
    grp = []
    for y in range(len(strip)):
        if strip[y] > 6:
            if grp and y - grp[-1][-1] <= 25:
                grp[-1].append(y)
            else:
                grp.append([y])
    lab = [float(np.mean([g[0], g[-1]])) for g in grp if len(g) >= 12
           and g[0] > 300]
    YVALS = [40, 20, 10, 8, 6, 4, 3, 2, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4,
             0.3]
    assert len(lab) == len(YVALS), (lab, YVALS)
    dec = (lab[2] - lab[0]) / np.log10(4.0)
    assert abs(lab[0] + dec * np.log10(2) - lab[1]) < 6, (lab[:3], dec)
    caly = lambda r0: float(10.0 ** np.interp(
        r0, np.asarray(lab, float), np.log10(np.asarray(YVALS, float))))
    # The photostat grid is broken into knots that defeat blob
    # filtering (hundreds of intersection ghosts).  Instead the FAIRED
    # CURVE -- the object Roshko 1961 and the textbooks quote as
    # "Wieselsberger's curve", drawn through all nine diameter series
    # -- is traced by a Viterbi path over the brightness map (one row
    # state per column, |drow| <= 3 per column, emission = 5-row
    # brightness sum), then symbol-candidate blobs are kept only
    # within 22 px of the path.
    blk = bright.copy()
    blk[:120, :] = blk[3010:, :] = False          # frame margins
    blk[:, :230] = blk[:, 4030:] = False
    blk[440:1130, 3040:3760] = False              # legend
    blk[960:1060, 850:1450] = False               # "Lamb's formula"
    blk[2850:2990, 1980:2210] = False             # "Fig. 1."
    blk[2830:3010, 3560:3710] = False             # "Vd/nu"
    em = ndimage.uniform_filter((a - bg).clip(0, 60), (5, 3))
    em[:120] = em[3005:] = 0.0
    # suppress full-width horizontal features (grid rows, the dotted
    # dirt row at c~0.55): subtract each row's mean -- the data
    # curve occupies any row only over a fraction of the width
    em = np.clip(em - 1.0 * em.mean(axis=1, keepdims=True), 0, None)
    # hand-read corridor waypoints (Re, c) from the rotated overview:
    # a continuous photostat SCRATCH at c~0.55 spanning Re 1e4..1e6
    # otherwise hijacks the tracker.  The corridor is +-0.16 decade
    # around the log-log interpolant -- wide enough that the path
    # position inside it comes from the emission, not the corridor.
    WAY = [(0.35, 50.0), (1.0, 23.0), (4.0, 4.5), (10.0, 3.0),
           (30.0, 2.0), (100.0, 1.45), (400.0, 1.05), (1e3, 0.90),
           (3e3, 0.85), (1e4, 1.04), (3e4, 1.10), (1e5, 1.10),
           (2e5, 1.08), (3e5, 0.95), (4.5e5, 0.55), (6e5, 0.42),
           (9e5, 0.35)]
    wlre = np.log10([wp[0] for wp in WAY])
    wlcd = np.log10([wp[1] for wp in WAY])
    C0, C1 = 240, 4020
    Y0, Y1 = 122, 3004
    for c in range(C0, C1):
        lcd = np.interp(np.log10(calx(c)), wlre, wlcd)
        rlo = caly_inv_row(lab, YVALS, 10.0 ** (lcd + 0.16))
        rhi = caly_inv_row(lab, YVALS, 10.0 ** (lcd - 0.16))
        em[:max(0, int(rlo)), c] = -1e5
        em[min(3128, int(rhi)):, c] = -1e5
    sub = em[Y0:Y1, C0:C1]
    ny, nc = sub.shape
    JUMP = 4
    cost = np.full(ny, -1e9)
    start = int(np.argmax(sub[:400, 0]))          # curve enters top-left
    cost[max(0, start - 5):start + 6] = 0.0
    back = np.zeros((nc, ny), np.int16)
    for c in range(1, nc):
        best = cost.copy()
        arg = np.zeros(ny, np.int16)
        for d in range(-JUMP, JUMP + 1):
            if d == 0:
                continue
            sh = np.roll(cost, d)
            sh[:d] = -1e9
            if d < 0:
                sh[d:] = -1e9
            pen = 0.35 * abs(d)
            better = sh - pen > best
            best[better] = sh[better] - pen
            arg[better] = d
        cost = best + sub[:, c]
        back[c] = arg
    path = np.zeros(nc, np.int32)
    # terminal anchor: the curve ends bottom-right (c ~ 0.35)
    endw = np.full(ny, -1e9)
    e0 = int(caly_inv_row(lab, YVALS, 0.38) - Y0)
    endw[e0:] = 0.0
    path[-1] = int(np.argmax(cost + endw))
    for c in range(nc - 1, 0, -1):
        path[c - 1] = path[c] - back[c, path[c]]
    prow = path + Y0
    pcol = np.arange(C0, C1)
    # The Viterbi path is reliable (visually verified) only from the
    # start of the DATA (Re = 4.2 -- below that the drawn line is
    # Lamb's formula, not measurement) to Re ~ 3e3; beyond, patchy
    # gridline knots rival the thin curve.  Curve samples are emitted
    # only there, and asserted to stay within 0.12 decade of the
    # hand-read waypoint interpolant.
    step = max(1, int((C1 - C0) / ((np.log10(1e6) - np.log10(0.35)) /
                                   0.02)))
    curve = []
    for c, r in zip(pcol[::step], prow[::step]):
        re_v, cd = float(calx(c)), float(caly(r))
        if 4.2 <= re_v <= 3e3:
            wcd = 10.0 ** np.interp(np.log10(re_v), wlre, wlcd)
            assert abs(np.log10(cd / wcd)) < 0.12, (re_v, cd, wcd)
            curve.append((re_v, cd))
    # Above Re 3e3 the symbols (the large 7.9/42/80/300 mm glyphs) are
    # the cleanest feature: keep blobs within +-0.12 decade of the
    # waypoint interpolant, then reject local-median outliers (>0.05
    # decade from the running median of 7 in log Cd) -- this removes
    # the few gridline knots that fall inside the band.
    blk = _strip_lines(blk, run=45)
    filled_m = ndimage.binary_fill_holes(
        ndimage.binary_closing(blk, np.ones((3, 3))))
    core = ndimage.binary_erosion(filled_m, np.ones((3, 3)),
                                  iterations=2)
    cand, rej = [], []
    for cx, cy, aa, hh, ww, sl in blobs(core, 5, 3000):
        subpts = [(cx, cy)]
        if 3 <= hh <= 38 and ww > 38:
            # chained symbols merge into a wide core: slice into
            # ~26 px segments, one centroid each
            sub = np.asarray(core[sl], bool)
            nseg = int(np.ceil(sub.shape[1] / 30.0))
            subpts = []
            for k in range(nseg):
                seg = sub[:, k * 30:(k + 1) * 30]
                if seg.sum() < 5:
                    continue
                ys, xs = np.nonzero(seg)
                subpts.append((sl[1].start + k * 30 + xs.mean(),
                               sl[0].start + ys.mean()))
        elif not (3 <= hh <= 38 and 3 <= ww <= 38):
            rej.append((cx, cy))
            continue
        for scx, scy in subpts:
            re_v, cd = float(calx(scx)), float(caly(scy))
            ok = 3e3 <= re_v <= 9.5e5
            if ok:
                wcd = 10.0 ** np.interp(np.log10(re_v), wlre, wlcd)
                ok = abs(np.log10(cd / wcd)) < 0.12
            if ok:
                cand.append((re_v, cd, scx, scy))
            else:
                rej.append((scx, scy))
    cand.sort()
    pts, marks = [], []
    lcd_arr = np.log10([c[1] for c in cand]) if cand else np.array([])
    for i, (re_v, cd, cx, cy) in enumerate(cand):
        lo, hi = max(0, i - 3), min(len(cand), i + 4)
        med = np.median(lcd_arr[lo:hi])
        if abs(np.log10(cd) - med) < 0.05:
            pts.append((re_v, cd, cx, cy))
            marks.append((cx, cy))
        else:
            rej.append((cx, cy))
    pts.sort()
    chk = img.convert('RGB')
    dr = ImageDraw.Draw(chk)
    dr.line([(int(c), int(r)) for c, r in zip(pcol[::6], prow[::6])],
            fill=(60, 140, 255), width=3)
    raster_check(chk, marks, 'check_tn84.png', rej=rej, r=16, scale=0.5)
    dump('tn84_wieselsberger', dict(
        source='Wieselsberger, "New data on the laws of fluid '
               'resistance" (Phys. Zeit. 22:321-328, 1921), English '
               'translation NACA TN-84 (1922), Fig. 1 (PDF p. 16 '
               'photostat, NTRS open PDF, see README); digitized '
               f'{TODAY}; all nine cylinder diameters merged.',
        method='raster 400 dpi, rotated -90; white-on-black photostat:'
               ' background-subtracted (sigma=25 gaussian) threshold '
               '+22; grid stripped by >=45 px run removal; symbol '
               'cores via closing+fill+double-erosion; calibration '
               'lines found by local peak search around label anchors '
               'and fit log-globally (residuals in calibration block);'
               ' check checks/check_tn84.png',
        calibration=dict(x=calx.report(),
                         y=dict(kind='piecewise-log-labels',
                                pix=list(lab), val=YVALS),
                         x_lines={str(k): v for k, v in xuse.items()}),
        wieselsberger_curve=dict(
            cls='experiment (faired curve)', access='primary',
            points=[dict(Re=re, Cd=cd) for re, cd in curve]),
        wieselsberger_symbols=dict(
            cls='experiment', access='primary',
            points=[dict(Re=re, Cd=cd) for re, cd, *_ in pts]),
        note='photostat quality limits accuracy to ~3% in Re and '
             '~2% in Cd; symbols merged into the faired curve may be '
             'subsampled; blotches rejected by the size window are '
             'red-circled in the check PNG',
    ))
    print(f'  curve samples: {len(curve)}  symbol pts: '
          f'{len(pts)}  rej {len(rej)}')
    print('  x lines used:', xuse)
    print('  y label rows:', lab)


REG = {
    'catalano2001_wmles': d_catalano2001_wmles,
    'rodriguez2015_fig4_exp': d_rodriguez2015_fig4,
    'veysey_fig7_lowre': d_veysey_fig7,
    'rodriguez2015_les': d_rodriguez2015_les,
    'qu2013_dns': d_qu2013_dns,
    'dong_karniadakis2005_dns3d': d_dong2005_dns3d,
    'iop2020_models': d_iop2020_models,
    'stringer2014_urans': d_stringer2014,
    'henderson1995': d_henderson1995,
    'roshko1961': d_roshko1961,
    'tn3038_delany_sorensen': d_tn3038,
    'tn84_wieselsberger': d_tn84,
}


if __name__ == '__main__':
    names = sys.argv[1:] or list(REG)
    for n in names:
        print('==', n)
        REG[n]()
