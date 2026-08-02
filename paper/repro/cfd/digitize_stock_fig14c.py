"""Digitize Stock (2006) Fig. 14c: alpha = 5.0 deg -- MEASURED transition
locations (DFVLR 3x3 m tunnel hot films, Kreplin et al., Stock Ref. 49) at
BOTH Reynolds numbers printed in the panel legend: open squares
Re_L = 1.52e6 and open circles Re_L = 6.49e6.  The 1.52e6 square chain is
the laminar-up-to-separation reference for our alpha = 5 deg / Re ~ 1.5e6
condition (Stock Sec. III.C: "The flow is predicted to remain laminar up
to separation for the small-Reynolds-number case, except for streamline
12, where transition is provoked by TS waves ... The measured transition
line shows a fairly similar behavior") -- the same regime as Fig. 15a
(alpha = 10, Re = 1.52e6), already in data/stock2006_fig15a_digitized.json.

Source raster: spheroid.pdf page 8, image index 1 (right column, Fig. 14
a/b/c stacked, 2027x3630 px) -> $DIG/p8_img1_2027x3630.png; re-extract
with --extract.  Panel c is the BOTTOM panel; identity verified from its
printed legend ("Re = 1.52 x 10^6" / "Re = 6.49 x 10^6"), which is the
only panel-c-specific text and which matches the Fig. 14 caption and the
Sec. III.C sentence quoted above (panels a/b both read "Re = 7.20e6").

This file also holds the shared engine used by digitize_stock_fig16c.py
(same figure family, same plotting package, same glyph set), so that the
two panels cannot drift apart.  Only measured symbols are digitized; the
computed lines (streamlines / free vortex-layer separation / TS / TS+CF)
are NOT extracted -- with two Reynolds numbers and four line styles in
one panel, style identity per curve could not be established from the
raster alone with the confidence the house policy demands.

Method (all calibration MEASURED from the raster, never assumed):
  X/a  -- the five vertical grid/frame columns of panel c are found as
          the column-coverage peaks of the panel interior and the line
          strokes are reduced to half-maximum-weighted centroids; a
          least-squares fit to (-1,-0.5,0,+0.5,+1) is asserted to hold
          to < 2.5 px (~0.003 X/a).  x/L = (X/a + 1)/2.
  phi  -- the two near-full-width dark rows bounding panel c are
          phi = 180 (top) and phi = 0 (bottom); the interior dashed
          phi grid rows are detected independently and asserted to land
          within 1 deg of their nominal 30-deg multiples.
  symbols -- (1) DETECTION by zero-normalized cross-correlation (local-
          window mean, as in digitize_stock_fig14a.py) against two
          templates cut from the cleanest ISOLATED data glyph of each
          type in this very panel (tight connected-component bboxes,
          see TSQ/TCI, grown by PAD px of the surrounding white space --
          the padding is what separates a glyph from a gridline
          crossing).  The threshold 0.50 sits in a wide empty gap: in
          Fig. 14c the 12 true glyphs score 0.58-1.00 and the best
          streamline/gridline false positive 0.41; in Fig. 16c 22 true
          glyphs score 0.59-1.00 against a 0.44 false-positive floor.
          (2) LOCALIZATION by binary template overlay -- the tight
          template is slid over +-5 px and the minimum-XOR-mismatch
          alignment is taken; the reported centre is that alignment's
          tight-bbox centre, so the number never comes from eyeballing
          a curve or a marker.
          (3) IDENTITY (square = low Re, circle = high Re) by three
          independent raster votes: minimum-XOR total, the ink fraction
          of the four 5x5 bbox corner blocks (a square has ink there, a
          circle does not), and the sign of the observed ink difference
          over the differential mask (square-only minus circle-only
          pixels).  Where all three agree the identity is taken as
          automatic.  Where they do not -- a streamline or a dashed grid
          line crossing the glyph can ink a circle's corners or erode a
          square's -- the glyph MUST appear in the AMBIGUOUS table
          below, which carries the identity read off the raster by hand
          at 5x zoom together with the pixel evidence for it; the script
          asserts that the automatic and manual sets exactly partition
          the detections, so a re-run cannot silently drop a decision.
          (4) glyph pairs closer than PAIR_DY px in phi on one station
          are fitted JOINTLY (both square/circle orderings, two
          templates at once) -- needed for Fig. 16c, unused here.
Accuracy: +-2 px ~ +-0.0025 x/L and +-0.4 deg, dominated by scan bleed
(the raster is a 2x-duplicated bitmap) and by the +-1 px spread of the
glyph bboxes themselves.

Independent check that the calibration is right: the recovered X/a values
cluster on the hot-film ring stations, and those stations agree with the
INDEPENDENTLY digitized Fig. 15a squares (different page, different
raster, 300-dpi render, pass ~26) to <= 0.002 x/L -- printed at the end
of the run.

Run from paper/:  python3 repro/cfd/digitize_stock_fig14c.py
-> data/stock2006_fig14c_digitized.json
   + check overlay  $DIG/check_fig14c.png
   + glyph montage  $DIG/check_fig14c_glyphs.png   (LOOK AT BOTH)
$DIG defaults to $STOCK_DIGITIZE_DIR, else <tmp>/stock_digitize.  For the
2026-08-01 pass on 019-v100-dev that was
/tmp/claude-1006/-home-qiqi-flexcompute/3d1a461e-96df-48fb-8bfc-cdfece2123e6/scratchpad/stock_digitize
(do NOT default this to /local_data/... -- that path lives on 014).
"""
import datetime
import json
import os
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import fftconvolve

DIG = os.environ.get('STOCK_DIGITIZE_DIR',
                     os.path.join(tempfile.gettempdir(), 'stock_digitize'))
_H = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(_H, '..', '..'))

XVALS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
NCC_THR = 0.50
PAD = 3                  # white margin added to the tight template for NCC
SUPPRESS = 13            # non-max radius, px (glyphs are ~27 px across)
PAIR_DY, PAIR_DX = 34, 8  # joint-fit trigger for stacked glyph pairs
CORNER = 5               # corner block size, px, for the corner-ink vote

FIG14C = dict(
    key='fig14c', page=7, image=1, img='p8_img1_2027x3630.png',
    zone=(2340, 3400),               # panel-c search band, full-image px
    legend=(2395, 2612, 360, 995),   # y0,y1,x0,x1 erase (legend block)
    tsq=(3084, 3111, 1882, 1909),    # tight bbox, isolated square glyph
    tci=(3168, 3193, 1110, 1136),    # tight bbox, isolated circle glyph
    alpha=5.0, re_sq=1.52e6, re_ci=6.49e6,
    fig='14c', panel='c (bottom)',
    # glyphs whose three automatic identity votes disagree: identity read
    # off the raster by hand at 5x zoom, with the pixel evidence.  Every
    # such glyph is circled in the check overlay and tiled in the montage.
    ambiguous=[
        (2478, 1894, 'sq',
         'all four bbox corner blocks inked (0.88/0.60/1.00/0.60) and sharp '
         'corners at 5x zoom; the differential-mask vote is spoiled by the '
         'two streamlines grazing the glyph top-left; its station x/L=0.935 '
         'is shared with the square at phi=50.9, while every Re=6.49e6 '
         'circle in this panel sits at x/L 0.48-0.57'),
        (2678, 1558, 'sq',
         'straight left/right walls, flat top and bottom edges, four inked '
         'corners at 5x zoom; the XOR totals nearly tie (sq 175 / ci 171) '
         'and the differential-mask vote ties at 0.00 because the X/a=+0.5 '
         'dashed grid column runs down the glyph; same hot-film station '
         '(x/L=0.737) as the square at phi=145.4'),
        (2818, 1706, 'sq',
         'four inked corners and the clearly lowest square XOR total '
         '(126 vs 165); only the differential-mask vote dissents, spoiled '
         'by the streamline crossing the glyph interior'),
        (3327, 1268, 'ci',
         'rounded top AND bottom: the first dark row is 13 px wide and '
         'widens to 27 px within 5 rows (rows 3312-3319), the last dark '
         'rows taper 26->12 px (rows 3334-3341) -- the circle signature; '
         'the two "inked corners" are the phi=0 frame line at rows '
         '3338-3339, not glyph ink; station x/L=0.567 is the Re=6.49e6 '
         'station also occupied by the circle at phi=159.9'),
    ],
    caption_note='Stock Fig. 14 caption: "Comparison of measured[49] and '
                 'computed transition locations for angle of attack: a) '
                 'alpha = 0 deg, b) alpha = 2.5 deg, and c) alpha = 5 deg"; '
                 'panel-c legend reads "Measured transition / [] Re = 1.52 x '
                 '10^6 / (o) Re = 6.49 x 10^6"',
    regime='Sec. III.C: at Re = 1.52e6 the flow is predicted to remain '
           'laminar up to separation except near streamline 12 (TS waves), '
           'and "the measured transition line shows a fairly similar '
           'behavior"; at Re = 6.49e6 transition is TS near both symmetry '
           'planes and TS+CF in between',
)


# --------------------------------------------------------------------------
# raster
# --------------------------------------------------------------------------
def extract(cfg):
    import fitz
    os.makedirs(DIG, exist_ok=True)
    doc = fitz.open(os.path.join(PAPER, 'spheroid.pdf'))
    xref = doc[cfg['page']].get_images(full=True)[cfg['image']][0]
    pix = fitz.Pixmap(doc, xref)
    assert f'{pix.width}x{pix.height}' in cfg['img'], (
        f'page {cfg["page"]+1} image {cfg["image"]} is {pix.width}x'
        f'{pix.height}, expected {cfg["img"]}')
    pix.save(os.path.join(DIG, cfg['img']))
    print('extracted', cfg['img'], pix.width, 'x', pix.height)


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
def calibrate(dark, zone):
    """Tick/frame-detected panel calibration.  Returns (y180, y0, xcols)."""
    zy0, zy1 = zone
    W = dark.shape[1]
    rowcov = dark[zy0:zy1].sum(1) / W
    wide = np.where(rowcov > 0.5)[0]
    groups = np.split(wide, np.where(np.diff(wide) > 3)[0] + 1)
    frames = [g for g in groups if len(g) >= 2]
    assert len(frames) >= 2, 'panel frame rows not found'

    def cen(g):
        return float((rowcov[g] * g).sum() / rowcov[g].sum()) + zy0
    y180, y0f = cen(frames[0]), cen(frames[-1])
    assert 900 < y0f - y180 < 1100, (y180, y0f)

    # independent check: interior dashed phi grid rows on 30-deg multiples
    checks = []
    for g in groups:
        if g[0] > frames[0][-1] + 20 and g[-1] < frames[-1][0] - 20:
            phi = 180.0 * (y0f - cen(g)) / (y0f - y180)
            checks.append((round(cen(g), 2), round(phi, 2)))
            assert abs(phi - 30 * round(phi / 30)) < 1.0, \
                f'phi grid row at {cen(g):.1f} -> {phi:.2f} deg, not a 30-multiple'
    assert checks, 'no interior phi grid row detected -- cannot check phi scale'

    yi0, yi1 = int(y180) + 7, int(y0f) - 6
    colcov = dark[yi0:yi1].sum(0) / (yi1 - yi0)
    on = np.where(colcov > 0.30)[0]
    xcols = []
    for g in np.split(on, np.where(np.diff(on) > 4)[0] + 1):
        pk = colcov[g].max()
        if pk < 0.55:
            continue
        s = g[colcov[g] >= 0.5 * pk]          # half-maximum stroke only
        xcols.append(float((colcov[s] * s).sum() / colcov[s].sum()))
    assert len(xcols) == 5, f'expected 5 X grid columns, got {xcols}'
    xcols = np.array(xcols)
    A = np.polyfit(xcols, XVALS, 1)
    res_px = (np.polyval(A, xcols) - XVALS) / A[0]
    assert np.abs(res_px).max() < 2.5, f'X calibration residual {res_px}'
    print(f'  frame rows phi=180/0: {y180:.2f} / {y0f:.2f}  (h {y0f-y180:.2f})')
    print(f'  phi grid-row checks (row, deg): {checks}')
    print(f'  X columns: {np.round(xcols, 2).tolist()}')
    print(f'  X fit residual (px): {np.round(res_px, 2).tolist()}')
    return y180, y0f, xcols, A


# --------------------------------------------------------------------------
# symbols
# --------------------------------------------------------------------------
def _ncc(img, t):
    tm = t - t.mean()
    ones = np.ones_like(t)
    s1 = fftconvolve(img, ones, mode='same')
    s2 = fftconvolve(img ** 2, ones, mode='same')
    num = fftconvolve(img, tm[::-1, ::-1], mode='same')
    var = np.maximum(s2 - s1 ** 2 / t.size, 1e-6)
    return num / np.sqrt(var * (tm ** 2).sum())


def _fit1(dark, t, cy, cx, rad=5):
    """Best single-template overlay near (cy, cx): (mismatch, fp, y0, x0)
    with (y0, x0) the fitted template top-left corner."""
    h, w = t.shape
    best = None
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            y0, x0 = cy + dy - h // 2, cx + dx - w // 2
            obs = dark[y0 - 3:y0 + h + 3, x0 - 3:x0 + w + 3]
            m = np.zeros_like(obs)
            m[3:3 + h, 3:3 + w] = t
            fp = int((m & ~obs).sum())
            fn = int((~m & obs).sum())
            if best is None or fp + fn < best[0]:
                best = (fp + fn, fp, y0, x0)
    return best


def _votes(dark, T, hy, hx):
    """Three independent raster votes on glyph identity.  Returns
    (fits, votes, diag)."""
    f = {k: _fit1(dark, T[k], hy, hx) for k in T}
    v_xor = 'sq' if f['sq'][0] <= f['ci'][0] else 'ci'

    h, w = T['sq'].shape
    y0, x0 = f['sq'][2], f['sq'][3]
    B = CORNER
    corners = [float(dark[a:a + B, b:b + B].mean())
               for a in (y0, y0 + h - B) for b in (x0, x0 + w - B)]
    n_ink = sum(1 for v in corners if v > 0.5)
    v_cor = 'sq' if n_ink >= 3 else 'ci'

    msq = np.zeros((h + 8, w + 8), bool)
    msq[4:4 + h, 4:4 + w] = T['sq']
    h2, w2 = T['ci'].shape
    mci = np.zeros_like(msq)
    yy, xx = f['ci'][2] - (y0 - 4), f['ci'][3] - (x0 - 4)
    mci[yy:yy + h2, xx:xx + w2] = T['ci']
    obs = dark[y0 - 4:y0 + h + 4, x0 - 4:x0 + w + 4]
    only_sq, only_ci = msq & ~mci, mci & ~msq
    d_sq = float(obs[only_sq].mean()) if only_sq.sum() else 0.0
    d_ci = float(obs[only_ci].mean()) if only_ci.sum() else 0.0
    v_diff = 'sq' if d_sq - d_ci > 0 else 'ci'

    diag = dict(xor_sq=f['sq'][0], xor_ci=f['ci'][0],
                corners=[round(v, 2) for v in corners],
                diff_sq=round(d_sq, 2), diff_ci=round(d_ci, 2))
    return f, (v_xor, v_cor, v_diff), diag


def _centre(t, y0, x0):
    return y0 + (t.shape[0] - 1) / 2.0, x0 + (t.shape[1] - 1) / 2.0


def _fit2(dark, tA, tB, pA, pB, rad=4):
    """Joint two-template overlay for a stacked pair; returns (mismatch,
    fp, centreA, centreB)."""
    hA, wA = tA.shape
    hB, wB = tB.shape
    ys = [min(pA[0], pB[0]) - hA - 6, max(pA[0], pB[0]) + hB + 6]
    xs = [min(pA[1], pB[1]) - wA - 6, max(pA[1], pB[1]) + wB + 6]
    obs = dark[ys[0]:ys[1], xs[0]:xs[1]]
    best = None
    for ay in range(-rad, rad + 1):
        for ax in range(-rad, rad + 1):
            mA = np.zeros_like(obs)
            y0 = pA[0] + ay - hA // 2 - ys[0]
            x0 = pA[1] + ax - wA // 2 - xs[0]
            mA[y0:y0 + hA, x0:x0 + wA] = tA
            cA = (y0 + ys[0] + (hA - 1) / 2.0, x0 + xs[0] + (wA - 1) / 2.0)
            for by in range(-rad, rad + 1):
                for bx in range(-rad, rad + 1):
                    m = mA.copy()
                    y1 = pB[0] + by - hB // 2 - ys[0]
                    x1 = pB[1] + bx - wB // 2 - xs[0]
                    m[y1:y1 + hB, x1:x1 + wB] |= tB
                    fp = int((m & ~obs).sum())
                    fn = int((~m & obs).sum())
                    if best is None or fp + fn < best[0]:
                        best = (fp + fn, fp, cA,
                                (y1 + ys[0] + (hB - 1) / 2.0,
                                 x1 + xs[0] + (wB - 1) / 2.0))
    return best


def run(cfg):
    os.makedirs(DIG, exist_ok=True)
    path = f'{DIG}/{cfg["img"]}'
    if '--extract' in sys.argv or not os.path.exists(path):
        extract(cfg)
    full = np.array(Image.open(path).convert('L'), copy=True)
    dark = full < 128
    print(f'Fig. {cfg["fig"]} panel {cfg["panel"]}, raster {cfg["img"]} '
          f'({full.shape[1]}x{full.shape[0]})')

    y180, y0f, xcols, A = calibrate(dark, cfg['zone'])

    def x2X(px):
        return float(A[0] * px + A[1])

    def y2phi(py):
        return 180.0 * (y0f - py) / (y0f - y180)

    y0, y1 = int(y180) + 4, int(np.ceil(y0f)) - 3
    panel = dark[y0:y1].copy()
    panel[:, :int(xcols[0]) + 3] = False        # frame strokes
    panel[:, int(xcols[-1]) - 2:] = False
    ly0, ly1, lx0, lx1 = cfg['legend']
    panel[ly0 - y0:ly1 - y0, lx0:lx1] = False   # legend block
    img = panel.astype(float)

    T, D = {}, {}
    for k, box in (('sq', cfg['tsq']), ('ci', cfg['tci'])):
        ty0, ty1, tx0, tx1 = box                # INCLUSIVE tight bbox
        T[k] = dark[ty0:ty1 + 1, tx0:tx1 + 1]
        D[k] = dark[ty0 - PAD:ty1 + 1 + PAD, tx0 - PAD:tx1 + 1 + PAD]
        print(f'  template {k}: {T[k].shape} from tight bbox {box} '
              f'(centre {(ty0+ty1)/2:.1f},{(tx0+tx1)/2:.1f}), '
              f'NCC template {D[k].shape}')

    maps = {k: _ncc(img, t.astype(float)) for k, t in D.items()}
    comb = np.maximum(maps['sq'], maps['ci'])
    hits, s = [], comb.copy()
    while True:
        i = int(np.argmax(s))
        r, c = np.unravel_index(i, s.shape)
        if s[r, c] < NCC_THR:
            break
        hits.append((int(r) + y0, int(c), float(maps['sq'][r, c]),
                     float(maps['ci'][r, c])))
        s[max(0, r - SUPPRESS):r + SUPPRESS + 1,
          max(0, c - SUPPRESS):c + SUPPRESS + 1] = -1
    hits.sort()
    print(f'  {len(hits)} NCC hits above {NCC_THR} '
          f'(next candidate {s.max():.3f})')

    # ---- identity + localization -----------------------------------------
    amb = list(cfg.get('ambiguous', []))
    amb_used = [False] * len(amb)
    used, out = set(), []
    for i, (hy, hx, ns, nc) in enumerate(hits):
        if i in used:
            continue
        j = next((k for k in range(i + 1, len(hits))
                  if abs(hits[k][0] - hy) <= PAIR_DY
                  and abs(hits[k][1] - hx) <= PAIR_DX), None)
        if j is not None:                          # stacked pair: joint fit
            used.update({i, j})
            gy, gx = hits[j][0], hits[j][1]
            fits = {hyp: _fit2(dark, T[hyp[0]], T[hyp[1]], (hy, hx), (gy, gx))
                    for hyp in (('ci', 'sq'), ('sq', 'ci'))}
            hyp = min(fits, key=lambda h: fits[h][0])
            other = [h for h in fits if h != hyp][0]
            m, _fp, cA, cB = fits[hyp]
            marg = fits[other][0] - m
            print(f'  stacked pair ({hy},{hx})/({gy},{gx}): {hyp[0]}/{hyp[1]}'
                  f' mismatch {m} vs {other[0]}/{other[1]} '
                  f'{fits[other][0]} -> margin {marg}')
            assert marg > 0, 'stacked pair ordering is a tie'
            meth = ('joint two-glyph overlay fit (square and circle drawn '
                    'overlapping on one hot-film station)')
            for kind, cc, sc in ((hyp[0], cA, ns if hyp[0] == 'sq' else nc),
                                 (hyp[1], cB, hits[j][2] if hyp[1] == 'sq'
                                  else hits[j][3])):
                out.append(dict(kind=kind, yc=cc[0], xc=cc[1], ncc=sc,
                                mismatch=m, margin=marg, method=meth,
                                diag=None))
            continue
        used.add(i)
        f, votes, diag = _votes(dark, T, hy, hx)
        man = next((k for k, a in enumerate(amb)
                    if abs(a[0] - hy) <= 8 and abs(a[1] - hx) <= 8), None)
        if len(set(votes)) == 1 and man is None:
            kind = votes[0]
            meth = ('NCC detection + overlay refit; identity unanimous on '
                    'all three raster votes')
        else:
            assert man is not None, (
                f'glyph at ({hy},{hx}) has split identity votes {votes} '
                f'{diag} but no entry in the AMBIGUOUS table -- resolve it '
                f'by hand from the raster before trusting this run')
            amb_used[man] = True
            kind = amb[man][2]
            meth = (f'NCC detection + overlay refit; identity votes {votes} '
                    f'split, resolved by hand: {amb[man][3]}')
        oth = 'ci' if kind == 'sq' else 'sq'
        out.append(dict(kind=kind, ncc=ns if kind == 'sq' else nc,
                        mismatch=f[kind][0], margin=f[oth][0] - f[kind][0],
                        method=meth, diag=diag,
                        **dict(zip(('yc', 'xc'),
                                   _centre(T[kind], f[kind][2], f[kind][3])))))
    for k, a in enumerate(amb):
        assert amb_used[k], (f'AMBIGUOUS entry {a[:3]} matched no detection '
                             '-- the table is stale')
    out.sort(key=lambda d: d['yc'])

    for d in out:
        d['phi_deg'] = round(y2phi(d['yc']), 2)
        d['Xa'] = round(x2X(d['xc']), 4)
        d['xL'] = round((x2X(d['xc']) + 1) / 2, 4)
        tag = '  <-- identity by hand' if 'by hand' in d['method'] else ''
        print(f'  {d["kind"]}  phi {d["phi_deg"]:7.2f}  X/a {d["Xa"]:+.4f}  '
              f'x/L {d["xL"]:.4f}  ncc {d["ncc"]:.2f}  '
              f'shape-margin {d["margin"]}{tag}')

    # ---- check overlay + glyph montage -----------------------------------
    im = Image.open(path).convert('RGB').crop(
        (0, y0 - 45, full.shape[1], y1 + 45))
    dr = ImageDraw.Draw(im)
    oy = y0 - 45
    dr.rectangle([lx0, ly0 - oy, lx1, ly1 - oy], outline=(0, 200, 0), width=2)
    for d in out:
        x, y = d['xc'], d['yc'] - oy
        if d['kind'] == 'sq':
            dr.rectangle([x - 22, y - 22, x + 22, y + 22],
                         outline=(255, 0, 0), width=3)
        else:
            dr.ellipse([x - 22, y - 22, x + 22, y + 22],
                       outline=(0, 130, 255), width=3)
    chk = f'{DIG}/check_{cfg["key"]}.png'
    im.save(chk)

    src = Image.open(path).convert('RGB')
    Z, R = 5, 24
    w, cols = 2 * R * Z, 6
    rows = (len(out) + cols - 1) // cols
    mon = Image.new('RGB', (cols * (w + 8), rows * (w + 24)), (255, 255, 255))
    md = ImageDraw.Draw(mon)
    for i, d in enumerate(out):
        cx, cy = int(round(d['xc'])), int(round(d['yc']))
        t = src.crop((cx - R, cy - R, cx + R, cy + R)).resize(
            (w, w), Image.NEAREST)
        px, py = (i % cols) * (w + 8), (i // cols) * (w + 24) + 20
        mon.paste(t, (px, py))
        md.text((px + 2, py - 18),
                f'{i} {d["kind"]} phi{d["phi_deg"]:.0f} m{d["margin"]}',
                fill=(255, 0, 0))
    mgl = f'{DIG}/check_{cfg["key"]}_glyphs.png'
    mon.save(mgl)

    # ---- station cross-check against the Fig. 15a digitization -----------
    ref = os.path.join(PAPER, 'data', 'stock2006_fig15a_digitized.json')
    stations = sorted({d['xL'] for d in out})
    if os.path.exists(ref):
        r = json.load(open(ref))
        rst = sorted({p['xL'] for k, v in r.items()
                      if k.startswith('re_') for p in v})
        print('  hot-film station cross-check vs Fig. 15a '
              '(this x/L -> nearest 15a x/L, delta):')
        for st in stations:
            n = min(rst, key=lambda v: abs(v - st))
            print(f'    {st:.4f} -> {n:.4f}  ({st-n:+.4f})')

    # ---- JSON -------------------------------------------------------------
    def block(kind):
        return [dict(xL=d['xL'], Xa=d['Xa'], phi_deg=d['phi_deg'],
                     ncc=round(d['ncc'], 3), shape_margin=d['margin'],
                     **({'method': d['method']}
                        if 'unanimous' not in d['method'] else {}))
                for d in out if d['kind'] == kind]
    sq, ci = block('sq'), block('ci')
    today = datetime.date.today()
    res = dict(
        source=f'Stock, AIAA J 44(1) 2006, Fig. {cfg["fig"]} '
               f'(DOI 10.2514/1.16026); alpha={cfg["alpha"]} deg; MEASURED '
               f'transition locations only (DFVLR 3x3 m tunnel hot films, '
               f'Kreplin et al., Stock Ref. 49): open squares '
               f'Re_L={cfg["re_sq"]:.3g}, open circles Re_L={cfg["re_ci"]:.3g} '
               f'(both printed in the panel legend); digitized {today} from '
               f'spheroid.pdf page {cfg["page"]+1} image {cfg["image"]} '
               f'({cfg["img"]}), panel {cfg["panel"]}, by '
               f'repro/cfd/digitize_stock_{cfg["key"]}.py -- NCC template '
               f'match + binary glyph overlay refit, tick-detected '
               f'calibration; see script docstring',
        caption=cfg['caption_note'],
        regime=cfg['regime'],
        convention='phi=0 windward symmetry line (STOCK convention; the '
                   'campaign mesh convention is phi=0 leeward -- mirror '
                   'phi -> 180-phi when overlaying); x/L=(X/a+1)/2 from nose',
        accuracy='+-2 px ~ +-0.0025 x/L and +-0.4 deg; glyph centres are '
                 'template-overlay fits, not eyeballed; every point is '
                 f'circled in check_{cfg["key"]}.png and shown zoomed in '
                 f'check_{cfg["key"]}_glyphs.png',
        calibration=dict(x_pix=np.round(xcols, 2).tolist(),
                         x_val=XVALS.tolist(),
                         y_pix=[round(y180, 2), round(y0f, 2)],
                         y_val=[180.0, 0.0],
                         detection='column-coverage half-maximum centroids '
                                   '(X) and the two near-full-width frame '
                                   'rows (phi), both measured inside panel '
                                   f'{cfg["panel"]}; interior dashed phi '
                                   'grid rows reproduce their nominal '
                                   '30-deg multiples to <1 deg'),
        n_squares=len(sq), n_circles=len(ci),
        hotfilm_stations_xL=stations,
    )
    res[f'measured_re{("%.2f" % (cfg["re_sq"]/1e6)).replace(".", "p")}e6'
        '_squares'] = sq
    res[f'measured_re{("%.2f" % (cfg["re_ci"]/1e6)).replace(".", "p")}e6'
        '_circles'] = ci
    res['not_digitized'] = (
        'Stock\'s COMPUTED curves in this panel (streamlines; free '
        'vortex-layer separation, short dashes; TS waves; TS+CF waves, '
        'mid dashes) are deliberately NOT extracted: with two Reynolds '
        'numbers and four dash styles overlaid on ~20 streamlines, the '
        'style->curve assignment could not be established from the raster '
        'with the confidence this pipeline requires. Fig. 15a\'s '
        'stock_computed_separation_line has no counterpart here.')
    dst = os.path.join(PAPER, 'data', f'stock2006_{cfg["key"]}_digitized.json')
    json.dump(res, open(dst, 'w'), indent=1)
    print(f'  squares: {len(sq)}, circles: {len(ci)}')
    if sq:
        print(f'  square x/L range {min(p["xL"] for p in sq):.4f}-'
              f'{max(p["xL"] for p in sq):.4f}, phi '
              f'{min(p["phi_deg"] for p in sq):.1f}-'
              f'{max(p["phi_deg"] for p in sq):.1f}')
    print('wrote', dst)
    print('     ', chk)
    print('     ', mgl)
    return res


if __name__ == '__main__':
    run(FIG14C)
