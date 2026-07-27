"""Digitize the MEASURED symbol chains from Stock (2006) Figs. 2-3
(Cp vs phi, thirteen pressure stations) and Figs. 4-5 (total skin
friction Cft and wall-shear direction gamma_w vs phi, seven hot-film
stations) -- raster scans inside spheroid.pdf; the extracted images
live in /local_data/qiqi/sa-ai/stock_digitize (regenerate with
--extract).

Method per panel: binary dark mask of the plot interior; erase grid
bands and in-panel legend boxes; separate the measured chains from
Stock's computed strokes (figs 4-5: morphological opening removes the
THICK computation strokes; figs 2-3: a per-column blob vertical-extent
floor removes the THIN potential-theory strokes and grid dashes,
keeping the tall symbol glyphs); run one exclusive continuity tracker
per station seeded at the left edge, nearest-blob with a slope cap and
mutual exclusion. Output: per-figure JSON with, per station, arrays of
(phi_deg, value) in PHYSICAL units (waterfall offset REMOVED using the
printed per-station offsets; figs 2-3 additionally shed a +0.14 base
displacement, derivation at FIGS below), plus full-resolution colored
check overlays in stock_digitize/ -- chain identity was verified there
per station, and every unrecoverable tangle is truncated via the
per-figure truncate dicts.

Calibrations measured from tick detection on the extracted images
(fig4: Cft 5.0@y26 to -10.0@y1222, gamma 30@y1371 to -120@y2615,
phi 0@x275 to 180@x2239; figs 2-3: per-entry comments below).

Run from paper/: python3 repro/cfd/digitize_stock_waterfalls.py fig2
-> data/stock2006_fig2_digitized.json + check PNGs in stock_digitize/.
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

DIG = '/local_data/qiqi/sa-ai/stock_digitize'
_H = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(_H, '..', '..'))

# per-figure geometry: image file, panels, calibration, stations, offsets
# Figs 2-3 (Cp): the plotted quantity is -Cp; the per-station step is
# 0.42 (pass-43, printed axes) and the WHOLE family additionally sits
# +0.14 above physical: off_i = 0.14 + 0.42*i bottom-up.  The base
# displacement is Stock's stated "0.14" (not a typo of the step) and
# was derived, not assumed: exact potential theory on the 6:1 spheroid
# (v = (1+k1)Ucos(a)(x_hat - n_x n) + (1+k2)Usin(a)(z_hat - n_z n),
# which matches our L2 RANS station Cp to ~0.005 away from nose/tail)
# sits a uniform +0.12..0.15 below the as-printed chains at BOTH
# incidences and below Stock's own printed potential-theory curves by
# +0.15; subtracting 0.14 closes both to ~0.01.  Stations are listed
# top-to-bottom (tracker order): off_i = off0 + i*doff, off0 =
# 12*0.42 + 0.14 = 5.18, doff = -0.42.
# The measured chains are ~20-px symbol glyphs while Stock's computed
# potential-theory curves are 3-5-px thin strokes: min_extent keeps
# only blobs tall enough to be symbols (mode='all', no thick/thin
# opening -- the roles are inverted relative to figs 4-5).
FIGS = {
    'fig2': dict(
        img='p3_img0_2219x2679.png',
        # tick-detected: phi grid cols 250.5/483.5/719.5/953/1186/1419/
        # 1655; -Cp rows 28.5 (6.0) .. 2402.5 (-1.0), 339.1 px/unit
        x_cal=(250.5, 0.0, 1655.0, 180.0),
        panels=dict(
            cp=dict(y_cal=(28.5, 6.0, 2402.5, -1.0),
                    yspan=(28, 2404), doff=-0.42, off0=5.18,
                    gridstep=1.0, min_extent=8),
        ),
        # in-panel legend text ("Potential theory"/"Free vortex layer
        # separation", top center) -- erase boxes in full-image px --
        # plus a stepped band along the thick dashed free-vortex-
        # separation curve where it crosses stations 0.212/-0.038/
        # -0.288 (it otherwise hijacks the 0.212 tracker at phi~106)
        erase=[(40, 230, 755, 1656),
               (620, 680, 1000, 1060), (680, 740, 1035, 1095),
               (740, 800, 1070, 1125), (800, 860, 1100, 1155),
               (860, 920, 1130, 1185), (920, 980, 1155, 1210)],
        stations=[0.79240, 0.66920, 0.46180, 0.21220, -0.03780,
                  -0.28780, -0.53700, -0.74320, -0.86520, -0.92840,
                  -0.96560, -0.98520, -0.99880],
        truncate={},
    ),
    'fig3': dict(
        img='p3_img1_1990x2414.png',
        # tick-detected: phi grid cols 236/445.5/656/865.5/1073/1283/
        # 1495.5; -Cp rows 25 (6.0) .. 2153 (-1.0), 304.0 px/unit
        x_cal=(236.0, 0.0, 1495.5, 180.0),
        panels=dict(
            cp=dict(y_cal=(25.0, 6.0, 2153.0, -1.0),
                    yspan=(25, 2155), doff=-0.42, off0=5.18,
                    gridstep=1.0, min_extent=8),
        ),
        erase=[(1915, 2085, 700, 1492)],
        stations=[0.79240, 0.66920, 0.46160, 0.21240, -0.03760,
                  -0.28780, -0.53700, -0.74320, -0.86520, -0.92840,
                  -0.96560, -0.98520, -0.99880],
        # manual seeds (panel rows, top-to-bottom): the -0.9852 and
        # -0.9988 symbol chains overlap into one band at the left edge
        # (their printed symbols coincide below phi~25), so automatic
        # left-edge clustering finds 12 rows for 13 chains; the pair is
        # split here and identity is checked where they separate
        seed_rows=[218, 302, 522, 671, 820, 968, 1127, 1292, 1459,
                   1638, 1809, 1948, 1962],
        # alpha=29.7 leeward dive tangle (phi~90-125): the aft-station
        # chains plunge together with the thick dashed free-vortex
        # curve and each other; identity verified per station at full
        # resolution up to these azimuths (windward rise, suction peak
        # and dive onset retained; post-tangle plateaus dropped where
        # the tracker demonstrably switched carriers).  Stations
        # -0.537/-0.743/-0.928/-0.966/-0.985/-0.999 verified clean over
        # the full 0-180 range.
        truncate={('cp', 0.79240): 92.0, ('cp', 0.66920): 94.0,
                  ('cp', 0.46160): 101.0, ('cp', 0.21240): 100.0,
                  ('cp', -0.03760): 108.0, ('cp', -0.28780): 108.0,
                  ('cp', -0.86520): 122.0},
    ),
    'fig4': dict(
        img='p3_img2_3092x2936.png',
        x_cal=(275.0, 0.0, 2239.0, 180.0),      # px0, phi0, px1, phi1
        panels=dict(
            cft=dict(y_cal=(26.0, 5.0, 1222.0, -10.0),
                     yspan=(26, 1250), doff=-1.5),
            gam=dict(y_cal=(1371.0, 30.0, 2615.0, -120.0),
                     yspan=(1360, 2680), doff=-15.0),
        ),
        stations=[-0.894, -0.722, -0.382, -0.040, 0.304, 0.650, 0.766],
        # chain identity in the post-transition tangle is not resolvable
        # by continuity tracking; truncate these (panel, station) chains
        # at phi_max deg and record it (the pre-transition rise and front
        # region are complete)
        truncate={('cft', -0.382): 125.0, ('cft', -0.040): 125.0,
                  ('cft', 0.304): 125.0, ('cft', 0.650): 125.0,
                  ('gam', -0.382): 135.0, ('gam', -0.040): 135.0,
                  ('gam', 0.304): 140.0, ('gam', 0.650): 92.0},
    ),
    'fig5': dict(
        img='p4_img1_2600x2351.png',
        x_cal=(230.0, 0.0, 1848.0, 180.0),
        panels=dict(
            cft=dict(y_cal=(28.0, 10.0, 1012.0, -20.0),
                     yspan=(28, 1040), doff=-3.0, gridstep=10.0),
            gam=dict(y_cal=(1130.0, 80.0, 2115.0, -160.0),
                     yspan=(1120, 2160), doff=-20.0, gridstep=40.0),
        ),
        stations=[-0.894, -0.722, -0.382, -0.040, 0.130, 0.650, 0.766],
        # alpha=29.7: identity is unrecoverable in the leeward tangle;
        # verified at full resolution up to these azimuths (fronts and
        # windward dives all retained)
        truncate={('cft', -0.894): 138.0, ('cft', -0.722): 115.0,
                  ('cft', -0.382): 112.0, ('cft', -0.040): 110.0,
                  ('cft', 0.130): 118.0, ('cft', 0.650): 95.0,
                  ('cft', 0.766): 95.0,
                  ('gam', -0.894): 133.0, ('gam', -0.722): 110.0,
                  ('gam', -0.382): 105.0, ('gam', -0.040): 110.0,
                  ('gam', 0.130): 105.0, ('gam', 0.650): 100.0,
                  ('gam', 0.766): 100.0},
    ),
}


def load_dark(img):
    im = np.array(Image.open(f'{DIG}/{img}').convert('L'))
    return im < 128


def clean_panel(dark, fig, panel):
    """Interior mask with grid bands erased and thick strokes separated.

    Panels with `min_extent` (the Cp figures) skip both the horizontal
    grid erase and the thick/thin opening: there the measured chains
    are tall symbol glyphs and everything thin (grid dashes, frame
    rows, potential-theory strokes) is dropped later by the blob
    vertical-extent filter, which avoids splitting symbols that sit on
    a grid line.
    """
    g = FIGS[fig]
    x0px = int(g['x_cal'][0]) + 4
    x1px = int(g['x_cal'][2]) - 4
    y0, y1 = g['panels'][panel]['yspan']
    m = dark[y0:y1, x0px:x1px].copy()
    # erase in-panel legend boxes (full-image px)
    for (ey0, ey1, ex0, ex1) in g.get('erase', []):
        m[max(0, ey0-y0):max(0, ey1-y0),
          max(0, ex0-x0px):max(0, ex1-x0px)] = False
    # erase dashed horizontal grid bands (at tick values) and vertical
    # 30-deg bands
    ycal = g['panels'][panel]['y_cal']
    py0, v0, py1, v1 = ycal
    px_per = (py1-py0)/(v1-v0)
    extent_mode = 'min_extent' in g['panels'][panel]
    if not extent_mode:
        # grid at each labeled tick value
        step = g['panels'][panel].get('gridstep',
                                      2.5 if panel == 'cft' else 30.0)
        vals = np.arange(v0, v1 + np.sign(v1-v0)*0.1,
                         (v1-v0)/abs(v1-v0) * step)
        for v in vals:
            r = int(py0 + (v-v0)*px_per) - y0
            if -6 < r < m.shape[0]+6:
                m[max(0, r-5):min(m.shape[0], r+6), :] = False
        # frame rows (thick) get a wider erase
        for v in (v0, v1):
            r = int(py0 + (v-v0)*px_per) - y0
            m[max(0, r-8):min(m.shape[0], r+9), :] = False
    xc = g['x_cal']
    pxdeg = (xc[2]-xc[0])/(xc[3]-xc[1])
    for phi in range(0, 181, 30):
        c = int(xc[0] + phi*pxdeg) - x0px
        if 6 < c < m.shape[1]-6:
            m[:, c-5:c+6] = False
    if extent_mode:
        return m, (y0, x0px)
    # thick computation strokes: survive a 2-iteration erosion
    thick = ndimage.binary_erosion(m, iterations=2)
    thick = ndimage.binary_dilation(thick, iterations=4)
    thin = m & ~thick
    return thin, (y0, x0px)


def _blobs(col_rows, min_extent=0):
    """Cluster a column's dark rows into blob centroids; blobs whose
    vertical extent is below min_extent px are dropped (thin strokes)."""
    if not len(col_rows):
        return []
    out = []
    start = prev = col_rows[0]
    for r in col_rows[1:]:
        if r - prev > 6:
            if prev - start >= min_extent:
                out.append((start+prev)/2.0)
            start = r
        prev = r
    if prev - start >= min_extent:
        out.append((start+prev)/2.0)
    return out


def track(thin, seeds, slope_cap=14, win0=26, hist=15, min_extent=0):
    """Simultaneous exclusive tracking: all chains advance column by
    column; blobs are assigned greedily by |prediction - blob| with
    mutual exclusion, so crossing chains cannot collapse onto one
    another. Prediction extrapolates the recent slope."""
    h, w = thin.shape
    cols = [np.where(thin[:, c])[0] for c in range(w)]
    n = len(seeds)
    ys = [float(s) for s in seeds]
    slopes = [0.0]*n
    miss = [0]*n
    recent = [[(0, float(s))] for s in seeds]
    tracks = [[] for _ in range(n)]
    for c in range(w):
        blobs = _blobs(cols[c], min_extent)
        if blobs:
            preds = [ys[i] + slopes[i]*min(miss[i]+1, 20) for i in range(n)]
            pairs = sorted((abs(preds[i]-b), i, bi)
                           for i in range(n) for bi, b in enumerate(blobs))
            used_t, used_b = set(), set()
            for dcost, i, bi in pairs:
                if i in used_t or bi in used_b:
                    continue
                win = win0 + min(miss[i], 30)*2
                if dcost > win:
                    continue
                used_t.add(i); used_b.add(bi)
                ynew = blobs[bi]
                ys[i] = ynew
                tracks[i].append((c, ynew))
                recent[i].append((c, ynew))
                if len(recent[i]) > hist:
                    recent[i].pop(0)
                if len(recent[i]) >= 4:
                    xs = np.array([q[0] for q in recent[i]])
                    yy = np.array([q[1] for q in recent[i]])
                    sl = np.polyfit(xs, yy, 1)[0]
                    slopes[i] = float(np.clip(sl, -slope_cap/4, slope_cap/4))
                miss[i] = 0
            for i in range(n):
                if i not in used_t:
                    miss[i] += 1
        else:
            for i in range(n):
                miss[i] += 1
    return tracks


def seeds_at_left(thin, n, look=60):
    """Cluster dark rows in the first `look` columns into seed rows;
    the merge gap adapts downward until at least n clusters emerge."""
    rows = np.where(thin[:, :look].sum(1) > 2)[0]
    for gap in (28, 18, 12, 8):
        groups = []
        if len(rows):
            start = prev = rows[0]
            for r in rows[1:]:
                if r-prev > gap:
                    groups.append((start+prev)/2); start = r
                prev = r
            groups.append((start+prev)/2)
        if len(groups) >= n:
            return groups
    return groups


def bidi_confirm(thin, fwd, n, tol=7.0):
    """Track backward from the right edge and keep only points both
    directions agree on (chain-identity swaps in the tangle then drop
    out instead of contaminating the overlay)."""
    flip = thin[:, ::-1]
    seeds_r = seeds_at_left(flip, n)
    back = track(flip, seeds_r)
    w = thin.shape[1]
    back = [[(w-1-c, y) for c, y in tr] for tr in back]
    # match backward tracks to forward tracks by median agreement
    confirmed = []
    bdicts = [dict(tr) for tr in back]
    for tr in fwd:
        d = dict(tr)
        best, bestscore = None, 1e9
        for bd in bdicts:
            common = set(d) & set(bd)
            if len(common) < 40:
                continue
            sc = np.median([abs(d[c]-bd[c]) for c in common])
            if sc < bestscore:
                bestscore, best = sc, bd
        if best is None:
            confirmed.append(tr)
            continue
        conf = [(c, y) for c, y in tr if c in best and abs(best[c]-y) <= tol]
        frac = len(conf)/max(len(tr), 1)
        if frac < 0.6:
            print(f'  WARNING: only {frac*100:.0f}% of a track '
                  f'bidirectionally confirmed')
        confirmed.append(conf)
    return confirmed


LEGEND_ROWS_FIG4 = [988, 1065, 1145, 1222, 1302, 1377, 1454]  # station order


def legend_templates(dark, rows, x0=2255, x1=2470, half=15):
    """Cut one template per legend symbol: the leftmost dark cluster in
    the legend row is the glyph (drawn on its line segment)."""
    tmpls = []
    for y in rows:
        strip = dark[y-half:y+half+1, x0:x1]
        cols = np.where(strip.sum(0) > 0)[0]
        # symbol center: centroid of the densest 30-px window
        dens = np.convolve(strip.sum(0), np.ones(30), 'same')
        cx = int(np.argmax(dens))
        t = dark[y-half:y+half+1, x0+cx-half:x0+cx+half+1].astype(float)
        tmpls.append(t - t.mean())
    return tmpls


def match_symbols(dark, fig, panel, tmpls, thr=0.34, nms=13):
    """NCC template matching inside the panel; returns per-station
    (col,row) marker centers in panel coordinates."""
    from scipy.signal import fftconvolve
    thin, (y0, x0px) = clean_panel(dark, fig, panel)
    img = thin.astype(float)
    imgm = img - img.mean()
    # score map per template (energy floor guards FFT ringing in empty
    # regions -- a real symbol window has >=50 dark px)
    e_img = fftconvolve(img**2, np.ones_like(tmpls[0]), mode='same')
    denom = np.sqrt(np.maximum(e_img, 30.0))
    valid = e_img > 30.0
    scores = []
    for t in tmpls:
        num = fftconvolve(imgm, t[::-1, ::-1], mode='same')
        scores.append(np.where(valid, num/(denom*np.sqrt((t**2).sum())),
                               -1.0))
    scores = np.stack(scores)                      # (n_tmpl, H, W)
    # joint: peaks of the max-over-templates map, label = argmax template
    comb = scores.max(0)
    lab = scores.argmax(0)
    out = [[] for _ in tmpls]
    s = comb.copy()
    for _ in range(3000):
        i = np.argmax(s)
        r, c = np.unravel_index(i, s.shape)
        if s[r, c] < thr:
            break
        out[lab[r, c]].append((int(c), int(r), float(s[r, c])))
        s[max(0, r-nms):r+nms+1, max(0, c-nms):c+nms+1] = -1
    return out, (y0, x0px)


def run_fig_symbols(fig):
    """Symbol-template digitization (chain identity from glyph shape)."""
    g = FIGS[fig]
    dark = load_dark(g['img'])
    tmpls = legend_templates(dark, LEGEND_ROWS_FIG4)
    out = {'source': f"Stock (2006) {fig}, raster digitization by "
                     "legend-template symbol matching; waterfall offsets "
                     "removed", 'stations': {}}
    from PIL import ImageDraw
    im = Image.open(f"{DIG}/{g['img']}").convert('RGB')
    dr = ImageDraw.Draw(im)
    colors = [(255, 0, 0), (0, 150, 0), (0, 0, 255), (200, 120, 0),
              (160, 0, 200), (0, 160, 160), (220, 0, 120)]
    for pname, p in g['panels'].items():
        pts_by_st, (yoff, xoff) = match_symbols(dark, fig, pname, tmpls)
        py0, v0, py1, v1 = p['y_cal']
        xc = g['x_cal']
        for i, (st, pts) in enumerate(zip(g['stations'], pts_by_st)):
            pts = sorted(pts)
            phi = [(c+xoff-xc[0])/((xc[2]-xc[0])/(xc[3]-xc[1]))
                   for c, r, s in pts]
            val = [v0 + ((r+yoff)-py0)*(v1-v0)/(py1-py0) - i*p['doff']
                   for c, r, s in pts]
            key = f'{pname}_x{st:+.3f}'
            out['stations'][key] = dict(
                station=st, offset=i*p['doff'],
                phi=[round(q, 2) for q in phi],
                value=[round(q, 4) for q in val])
            for c, r, s in pts:
                dr.ellipse([c+xoff-5, r+yoff-5, c+xoff+5, r+yoff+5],
                           outline=colors[i % 7], width=2)
            print(f'{fig}/{pname} station {st:+.3f}: {len(pts)} markers')
    im.thumbnail((1500, 1500))
    im.save(f'{DIG}/check_{fig}_symbols.png')
    path = f'{PAPER}/data/stock2006_{fig}_digitized.json'
    json.dump(out, open(path, 'w'))
    print('wrote', path, 'and', f'{DIG}/check_{fig}_symbols.png')


def run_fig(fig):
    g = FIGS[fig]
    dark = load_dark(g['img'])
    note = ('; value is -Cp (as plotted)' if 'cp' in g['panels'] else '')
    out = {'source': f"Stock (2006) {fig}, raster digitization; "
                     f"waterfall offsets removed{note}", 'stations': {}}
    from PIL import ImageDraw
    im = Image.open(f"{DIG}/{g['img']}").convert('RGB')
    dr = ImageDraw.Draw(im)
    colors = [(255, 0, 0), (0, 150, 0), (0, 0, 255), (200, 120, 0),
              (160, 0, 200), (0, 160, 160), (220, 0, 120), (120, 80, 0),
              (255, 120, 120), (0, 220, 80), (100, 100, 255),
              (240, 180, 0), (255, 0, 255)]
    for pname, p in g['panels'].items():
        thin, (yoff, xoff) = clean_panel(dark, fig, pname)
        seeds = g.get('seed_rows') or seeds_at_left(thin, len(g['stations']))
        me = p.get('min_extent', 0)
        if me and 'seed_rows' not in g:
            # a valid seed must sit on a symbol blob (extent >= me) in
            # the first columns -- grid-dash rows otherwise seed
            # wandering trackers that steal blobs from real chains
            good = []
            for s in seeds:
                for c in range(80):
                    if any(abs(b - s) <= 15
                           for b in _blobs(np.where(thin[:, c])[0], me)):
                        good.append(s)
                        break
            print(f'{fig}/{pname}: seed extent filter {len(seeds)} -> '
                  f'{len(good)}')
            seeds = good
        tracks = track(thin, seeds, min_extent=me)
        # filter: drop frame/spurious seeds -- tracks that are too short or
        # ride the panel edge with near-zero variation
        keep = []
        for s, tr in zip(seeds, tracks):
            if len(tr) < 0.22*thin.shape[1]:
                continue
            ys = np.array([y for _, y in tr])
            if ys.std() < 4.0 and (ys.mean() < 15 or
                                   ys.mean() > thin.shape[0]-15):
                continue
            keep.append((s, tr))
        # dedupe: two seeds that landed on the same chain
        dedup = []
        for s, tr in keep:
            dup = False
            d = dict(tr)
            for s2, tr2 in dedup:
                d2 = dict(tr2)
                common = set(d) & set(d2)
                if len(common) > 50:
                    diff = np.median([abs(d[c]-d2[c]) for c in common])
                    if diff < 8.0:
                        dup = True
                        break
            if not dup:
                dedup.append((s, tr))
        keep = dedup
        print(f'{fig}/{pname}: {len(seeds)} seeds -> {len(keep)} kept '
              f'(want {len(g["stations"])})')
        assert len(keep) == len(g['stations']), \
            f'station count mismatch in {fig}/{pname}'
        keep.sort(key=lambda st: st[0])          # top-to-bottom = station order
        tracks = [tr for _, tr in keep]
        py0, v0, py1, v1 = p['y_cal']
        xc = g['x_cal']
        for i, (st, tr) in enumerate(zip(g['stations'], tracks)):
            pmax = g.get('truncate', {}).get((pname, st), 999.0)
            pxdeg = (xc[2]-xc[0])/(xc[3]-xc[1])
            tr = [(c, y) for c, y in tr if (c+xoff-xc[0])/pxdeg <= pmax]
            off = p.get('off0', 0.0) + i*p['doff']
            phi = [ (c+xoff-xc[0])/((xc[2]-xc[0])/(xc[3]-xc[1])) for c, _ in tr ]
            val = [ v0 + ((y+yoff)-py0)*(v1-v0)/(py1-py0) - off
                    for _, y in tr ]
            key = f'{pname}_x{st:+.3f}'
            out['stations'][key] = dict(station=st, offset=off,
                                        phi=[round(q, 2) for q in phi],
                                        value=[round(q, 4) for q in val])
            col = colors[i % len(colors)]
            for c, y in tr[::4]:
                dr.ellipse([c+xoff-3, y+yoff-3, c+xoff+3, y+yoff+3],
                           outline=col)
    im.save(f'{DIG}/check_{fig}_full.png')
    im.thumbnail((1400, 1400))
    im.save(f'{DIG}/check_{fig}.png')
    path = f'{PAPER}/data/stock2006_{fig}_digitized.json'
    json.dump(out, open(path, 'w'))
    print('wrote', path, 'and', f'{DIG}/check_{fig}.png')


if __name__ == '__main__':
    if '--symbols' in sys.argv:
        run_fig_symbols([a for a in sys.argv[1:] if not a.startswith('-')][0] if [a for a in sys.argv[1:] if not a.startswith('-')] else 'fig4')
    else:
        run_fig(sys.argv[1] if len(sys.argv) > 1 else 'fig4')
