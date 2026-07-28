"""Digitize Stock (2006) Fig. 14b: alpha = 2.5 deg, Re = 7.20e6 --
MEASURED transition (DFVLR hot films, Kreplin et al., Stock Ref. 49; open
squares) and Stock's COMPUTED pure-TS-wave e^N front (continuous line;
Sec. III.C: transition at alpha = 0 AND 2.5 deg is provoked solely by TS
waves at this Re).  This is the measured reference for the campaign's
re72a2p5 condition (the reseed record 2026-07-28).

Method = digitize_stock_fig14a.py transplanted to panel b of the same
raster (spheroid.pdf p.8, p8_img1_2027x3630.png), with two differences:
  - the computed front at alpha = 2.5 is SLIGHTLY slanted, so instead of
    the full-height-column detector the line is traced by a windowed
    per-row walk (seeded at the max-coverage column of the panel's middle
    third, then following the nearest dark run of plausible width row by
    row; dashed phi-gridline rows are skipped by the run-width cap);
  - the measured squares are sparse (the alpha = 2.5 test has few hot-film
    azimuths) and two of them sit ON the phi = 0/180 frame lines; NCC with
    the panel-a data-square template finds them anyway because the glyph
    sticks out of the 4-px frame stroke -- every match is verified in the
    check overlay (check_fig14b.png).

Calibration: X columns re-detected inside panel b (same tick ritual as
14a; they land within 1 px of the panel-a fit), frame rows by the two
full-width dark rows bounding the panel.

Run from paper/:  python3 repro/cfd/digitize_stock_fig14b.py
-> data/stock2006_fig14b_digitized.json
   + check overlay /local_data/qiqi/sa-ai/stock_digitize/check_fig14b.png
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw

DIG = '/local_data/qiqi/sa-ai/stock_digitize'
_H = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(_H, '..', '..'))
IMG = 'p8_img1_2027x3630.png'

# panel-b bounding zone (full-image px; frame rows re-detected inside)
ZONE_Y = (1150, 2300)
XVALS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
XCOLS_A = np.array([302.0, 725.0, 1152.5, 1576.0, 2000.5])  # panel-a fit
# legend erase ("Measured transition / Re = 7.20e6" block, panel b)
LEGEND_BOX = (1240, 1420, 1330, 2010)    # y0, y1, x0, x1 full-image px
TEMPLATE = (293, 327, 1030, 1062)        # the panel-a clean data square


def _runs(panel, row, lo, hi):
    """(center, width) of contiguous dark runs of panel[row, lo:hi]."""
    w = np.where(panel[row, lo:hi])[0]
    if not len(w):
        return []
    gaps = np.where(np.diff(w) > 1)[0]
    return [(s.mean() + lo, len(s)) for s in np.split(w, gaps + 1)]


def main():
    dark = np.array(Image.open(f'{DIG}/{IMG}').convert('L')) < 128

    # ---- frame rows: the two widest dark rows in the zone ------------------
    zy0, zy1 = ZONE_Y
    rowcov = dark[zy0:zy1].sum(1)
    wide = np.where(rowcov > 0.8 * dark.shape[1])[0] + zy0
    assert len(wide) >= 2, 'frame rows not found'
    y_phi180, y_phi0 = float(wide.min()), float(wide.max())
    assert y_phi0 - y_phi180 > 900, (y_phi180, y_phi0)

    # ---- X calibration columns re-detected inside the panel ----------------
    y0, y1 = int(y_phi180) + 6, int(y_phi0) - 5
    colcov = dark[y0:y1].sum(0) / (y1 - y0)
    xcols = []
    for ca in XCOLS_A:
        w = np.arange(int(ca) - 8, int(ca) + 9)
        c = w[np.argmax(colcov[w])]
        run = [c]
        while colcov[run[0] - 1] > 0.85: run.insert(0, run[0] - 1)   # noqa: E701
        while colcov[run[-1] + 1] > 0.85: run.append(run[-1] + 1)    # noqa: E701
        xcols.append(float(np.mean(run)))
    xcols = np.array(xcols)
    A = np.polyfit(xcols, XVALS, 1)
    x_to_Xa = lambda px: A[0] * px + A[1]                            # noqa: E731
    y_to_phi = lambda py: 180.0 * (y_phi0 - py) / (y_phi0 - y_phi180)  # noqa: E731

    panel = dark[y0:y1].copy()
    panel[:, :int(xcols[0]) + 4] = False
    panel[:, int(xcols[-1]) - 3:] = False
    ly0, ly1, lx0, lx1 = LEGEND_BOX
    panel[max(0, ly0 - y0):ly1 - y0, lx0:lx1] = False

    # ---- computed TS front: seeded windowed walk ----------------------------
    h = panel.shape[0]
    # the alpha=2.5 front drifts ~50 px across the panel (x/L 0.529 at
    # phi=180 to 0.500 at phi=0 class), so neither a single column nor a
    # narrow band carries full coverage.  Seed by VOTE: per-row centers
    # of plausible-width dark runs (the near-vertical front contributes
    # ~1 vote/row inside a 50-px band; near-horizontal streamlines smear
    # their crossings over all columns; grid columns are excluded by the
    # width cap since they are continuous verticals -> full-height runs
    # are not row-runs anyway, and dashes are wider than 14 px)
    hist = np.zeros(panel.shape[1])
    for r in range(h):
        for c, n in _runs(panel, r, 0, panel.shape[1]):
            if n <= 14 and np.abs(c - xcols).min() > 16:
                hist[int(c) - 5:int(c) + 6] += 1
    seed = float(np.argmax(hist))
    assert hist[int(seed)] > 0.3 * h, f'front vote too weak: {hist.max()}'

    def runs_in(row, lo, hi):
        return _runs(panel, row, max(0, lo), min(panel.shape[1], hi))

    def walk(r0, r1, step):
        pts, ctr = [], seed
        for r in range(r0, r1, step):
            rs = [(c, n) for c, n in runs_in(r, int(ctr) - 22, int(ctr) + 23)
                  if n <= 14]                       # skip dashes/blobs
            if not rs:
                continue
            c = min(rs, key=lambda t: abs(t[0] - ctr))[0]
            if abs(c - ctr) > 12:
                continue
            ctr = 0.5 * (ctr + c)
            pts.append((r, c))
        return pts

    ts = sorted(walk(h // 2, h, 1) + walk(h // 2 - 1, -1, -1))
    ts_phi = [round(y_to_phi(r + y0), 2) for r, _ in ts]
    ts_Xa = [round(float(x_to_Xa(c)), 4) for _, c in ts]
    print(f'computed TS front: {len(ts)} rows, X/a '
          f'{np.mean(ts_Xa):.4f} +- {np.std(ts_Xa):.4f} '
          f'(x/L {np.mean([(v + 1) / 2 for v in ts_Xa]):.4f}), '
          f'ends x/L {(ts_Xa[0]+1)/2:.4f} (phi {ts_phi[0]}) / '
          f'{(ts_Xa[-1]+1)/2:.4f} (phi {ts_phi[-1]})')

    # ---- measured squares: NCC (panel-a data-square template) --------------
    from scipy.signal import fftconvolve
    ty0, ty1, tx0, tx1 = TEMPLATE
    t = dark[ty0:ty1, tx0:tx1].astype(float)
    tm = t - t.mean()
    img = panel.astype(float)
    ones = np.ones_like(t)
    s1 = fftconvolve(img, ones, mode='same')
    s2 = fftconvolve(img ** 2, ones, mode='same')
    num = fftconvolve(img, tm[::-1, ::-1], mode='same')
    var = np.maximum(s2 - s1 ** 2 / t.size, 1e-6)
    score = num / np.sqrt(var * (tm ** 2).sum())
    sq, s = [], score.copy()
    while True:
        i = np.argmax(s)
        r, c = np.unravel_index(i, s.shape)
        if s[r, c] < 0.5:
            break
        sq.append((int(r), int(c), float(s[r, c])))
        s[max(0, r - 14):r + 15, max(0, c - 14):c + 15] = -1
    sq.sort()
    print(f'{len(sq)} measured squares (scores '
          f'{min(q[2] for q in sq):.2f}-{max(q[2] for q in sq):.2f})')

    # ---- frame-row squares (phi = 0/180): targeted cluster centroid --------
    # Two of the alpha=2.5 squares are drawn exactly ON the frame lines
    # (fig14a's "merged dark blobs"); blind NCC misses them, but in this
    # panel the glyph sticks cleanly out of the ~6-px frame stroke: in a
    # +-22-row window around the frame row, after dropping near-full-width
    # rows (the stroke itself), the square is the ONLY dark-column cluster
    # of glyph width (20-35 px; dashes <= 8, grid columns excluded).
    frame_sq = []
    for phi_val, yc in ((180.0, y_phi180), (0.0, y_phi0)):
        win = dark[int(yc) - 22:int(yc) + 23,
                   int(xcols[0]) + 6:int(xcols[-1]) - 5].copy()
        keep = win.sum(1) < 0.5 * win.shape[1]
        prof = win[keep].sum(0)
        on = np.where(prof > 3)[0]
        gaps = np.where(np.diff(on) > 2)[0]
        clus = [s for s in np.split(on, gaps + 1)
                if 20 <= s[-1] - s[0] + 1 <= 35
                and prof[s].mean() > 10]     # dense glyph, not a thin
                                             # streamline/dash cluster
        assert len(clus) == 1, f'phi={phi_val}: {len(clus)} glyph clusters'
        c = float((prof[clus[0]] * clus[0]).sum() / prof[clus[0]].sum()) \
            + xcols[0] + 6
        frame_sq.append((phi_val, c))
        print(f'  frame-row square phi={phi_val:5.1f}: col {c:.1f} '
              f'(x/L {(float(x_to_Xa(c)) + 1) / 2:.4f})')

    # ---- check overlay ------------------------------------------------------
    im = Image.open(f'{DIG}/{IMG}').convert('RGB').crop(
        (0, zy0 - 30, 2027, zy1 + 30))
    dr = ImageDraw.Draw(im)
    oy = zy0 - 30
    for r, c, _ in sq:
        dr.ellipse([c - 16, r + y0 - oy - 16, c + 16, r + y0 - oy + 16],
                   outline=(255, 0, 0), width=3)
    for phi_val, c in frame_sq:
        yc = y_phi180 if phi_val == 180.0 else y_phi0
        dr.ellipse([c - 16, yc - oy - 16, c + 16, yc - oy + 16],
                   outline=(0, 180, 0), width=3)
    for r, c in ts[::10]:
        dr.ellipse([c - 3, r + y0 - oy - 3, c + 3, r + y0 - oy + 3],
                   outline=(0, 120, 255))
    im.save(f'{DIG}/check_fig14b.png')

    sq_out = [dict(xL=round((float(x_to_Xa(c)) + 1) / 2, 4),
                   Xa=round(float(x_to_Xa(c)), 4),
                   phi_deg=round(y_to_phi(r + y0), 2), ncc=round(sc, 3))
              for r, c, sc in sq]
    sq_out += [dict(xL=round((float(x_to_Xa(c)) + 1) / 2, 4),
                    Xa=round(float(x_to_Xa(c)), 4), phi_deg=phi_val,
                    method='frame-row cluster centroid (on-frame glyph)')
               for phi_val, c in frame_sq]
    out = dict(
        source='Stock, AIAA J 44(1) 2006, Fig. 14b (DOI 10.2514/1.16026); '
               'alpha=2.5 deg, Re_L=7.20e6 (exact match to our re72a2p5 '
               'ladder); measured transition = DFVLR hot films (Kreplin et '
               'al., Stock Ref. 49); computed = Stock pure-TS-wave e^N '
               '(N_TS=8.0, his DFVLR-tunnel limit, Fig. 11a; Sec III.C: '
               'TS-only at alpha=0 and 2.5), continuous-line style; '
               f'digitized {__import__("datetime").date.today()} from '
               'spheroid.pdf p.8 raster (p8_img1 panel b) by NCC template '
               'match (squares) + windowed per-row walk (TS line); '
               'calibration re-detected inside panel b, see docstring',
        convention='phi=0 windward symmetry line (STOCK convention; the '
                   'campaign mesh convention is phi=0 leeward -- mirror '
                   'phi -> 180-phi when overlaying); x/L=(X/a+1)/2',
        calibration=dict(x_pix=xcols.tolist(), x_val=XVALS.tolist(),
                         y_pix=[y_phi180, y_phi0], y_val=[180.0, 0.0]),
        measured_squares=sq_out,
        computed_ts_front=dict(
            phi_deg=ts_phi[::5], Xa=ts_Xa[::5],
            xL=[round((v + 1) / 2, 4) for v in ts_Xa[::5]],
            mean_xL=round(float(np.mean([(v + 1) / 2 for v in ts_Xa])), 4),
            std_xL=round(float(np.std([(v + 1) / 2 for v in ts_Xa])), 4)),
        note='alpha=2.5 hot-film coverage is sparse; squares on the '
             'phi=0/180 frame lines are matched THROUGH the frame stroke '
             '-- verify every red circle in check_fig14b.png before use',
    )
    path = f'{PAPER}/data/stock2006_fig14b_digitized.json'
    json.dump(out, open(path, 'w'), indent=1)
    for q in sq_out:
        print(f"  square x/L {q['xL']:.4f}  phi {q['phi_deg']:7.2f}  "
              + (f"ncc {q['ncc']:.2f}" if 'ncc' in q else q['method']))
    print('wrote', path, 'and', f'{DIG}/check_fig14b.png')


if __name__ == '__main__':
    main()
