"""[WIP -- tracker validated visually except the cft phi>125 tangle; next:
legend-template symbol matching for chain identity]

Digitize the MEASURED symbol chains from Stock (2006) Figs. 4-5
(total skin friction Cft and wall-shear direction gamma_w vs phi, seven /
eight stations, waterfall offsets) and Figs. 2-3 (Cp) -- raster scans
inside spheroid.pdf; the extracted images live in
/local_data/qiqi/sa-ai/stock_digitize (regenerate with --extract).

Method per panel: binary dark mask of the plot interior; erase dashed
grid bands (known tick rows/cols); split thick COMPUTATION strokes from
thin measured symbol chains by morphological opening; run one
continuity tracker per station seeded at the left edge, nearest-blob
with a slope cap. Output: per-figure JSON with, per station, arrays of
(phi_deg, value) in PHYSICAL units (waterfall offset REMOVED using the
printed per-station offsets), plus the tracker's pixel trace for the
overlay-check PNG.

Calibrations measured from tick detection on the extracted images
(fig4: Cft 5.0@y26 to -10.0@y1222, gamma 30@y1371 to -120@y2615,
phi 0@x275 to 180@x2239).

Run from paper/: python3 repro/cfd/digitize_stock_waterfalls.py fig4
-> data/stock2006_fig4_digitized.json + check PNG in stock_digitize/.
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
FIGS = {
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
    ),
}


def load_dark(img):
    im = np.array(Image.open(f'{DIG}/{img}').convert('L'))
    return im < 128


def clean_panel(dark, fig, panel):
    """Interior mask with grid bands erased and thick strokes separated."""
    g = FIGS[fig]
    x0px = int(g['x_cal'][0]) + 4
    x1px = int(g['x_cal'][2]) - 4
    y0, y1 = g['panels'][panel]['yspan']
    m = dark[y0:y1, x0px:x1px].copy()
    # erase dashed horizontal grid bands (at tick values) and vertical
    # 30-deg bands
    ycal = g['panels'][panel]['y_cal']
    py0, v0, py1, v1 = ycal
    px_per = (py1-py0)/(v1-v0)
    # grid at each labeled tick value
    vals = np.arange(v0, v1 + np.sign(v1-v0)*0.1,
                     (v1-v0)/abs(v1-v0) * (2.5 if panel == 'cft' else 30.0))
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
    # thick computation strokes: survive a 2-iteration erosion
    thick = ndimage.binary_erosion(m, iterations=2)
    thick = ndimage.binary_dilation(thick, iterations=4)
    thin = m & ~thick
    return thin, (y0, x0px)


def _blobs(col_rows):
    """Cluster a column's dark rows into blob centroids."""
    if not len(col_rows):
        return []
    out = []
    start = prev = col_rows[0]
    for r in col_rows[1:]:
        if r - prev > 6:
            out.append((start+prev)/2.0); start = r
        prev = r
    out.append((start+prev)/2.0)
    return out


def track(thin, seeds, slope_cap=14, win0=26, hist=15):
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
        blobs = _blobs(cols[c])
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
    """Cluster dark rows in the first `look` columns into n seed rows."""
    rows = np.where(thin[:, :look].sum(1) > 2)[0]
    groups = []
    if len(rows):
        start = prev = rows[0]
        for r in rows[1:]:
            if r-prev > 28:
                groups.append((start+prev)/2); start = r
            prev = r
        groups.append((start+prev)/2)
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


def run_fig(fig):
    g = FIGS[fig]
    dark = load_dark(g['img'])
    out = {'source': f"Stock (2006) {fig}, raster digitization; "
                     "waterfall offsets removed", 'stations': {}}
    from PIL import ImageDraw
    im = Image.open(f"{DIG}/{g['img']}").convert('RGB')
    dr = ImageDraw.Draw(im)
    for pname, p in g['panels'].items():
        thin, (yoff, xoff) = clean_panel(dark, fig, pname)
        seeds = seeds_at_left(thin, len(g['stations']))
        tracks = track(thin, seeds)
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
        tracks = bidi_confirm(thin, [tr for _, tr in keep],
                              len(g['stations']))
        py0, v0, py1, v1 = p['y_cal']
        xc = g['x_cal']
        for i, (st, tr) in enumerate(zip(g['stations'], tracks)):
            phi = [ (c+xoff-xc[0])/((xc[2]-xc[0])/(xc[3]-xc[1])) for c, _ in tr ]
            val = [ v0 + ((y+yoff)-py0)*(v1-v0)/(py1-py0) - i*p['doff']
                    for _, y in tr ]
            key = f'{pname}_x{st:+.3f}'
            out['stations'][key] = dict(station=st, offset=i*p['doff'],
                                        phi=[round(q, 2) for q in phi],
                                        value=[round(q, 4) for q in val])
            for c, y in tr[::4]:
                dr.ellipse([c+xoff-3, y+yoff-3, c+xoff+3, y+yoff+3],
                           outline=(255, 0, 0))
    im.thumbnail((1400, 1400))
    im.save(f'{DIG}/check_{fig}.png')
    path = f'{PAPER}/data/stock2006_{fig}_digitized.json'
    json.dump(out, open(path, 'w'))
    print('wrote', path, 'and', f'{DIG}/check_{fig}.png')


if __name__ == '__main__':
    run_fig(sys.argv[1] if len(sys.argv) > 1 else 'fig4')
