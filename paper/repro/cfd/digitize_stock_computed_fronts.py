"""Digitize Stock (2006) COMPUTED transition lines -- the mechanism-coded
curves that Figs. 14-17 overlay on the measured symbols -- so that the
paper can do a three-way comparison (our SA-AI run / Stock's computation /
the experiment) at each tunnel condition.

Stock's own key, Sec. III.C verbatim: "The transition prediction process
differentiates in Figs. 14-17 between pure TS waves produced transition
(CONTINUOUS line), pure CF waves triggered transition (LONG DASHED lines),
transition provoked by both types of waves (MIDSIZED DASHED lines), and
separation (SHORT DASHED lines)."  The figure captions add "----,
streamlines" -- also continuous, but THIN.  So the discrimination is
two-dimensional and both dimensions are measured here, never eyeballed:

  1. LINE WIDTH separates the computed curves from the ~20 streamlines.
     The rasters are 2x-pixel-duplicated bitmaps: a streamline is 2-3 px
     wide (distance transform dt ~ 1), two streamlines that touch make
     4 px (dt ~ 2), and a computed curve is 6-8 px (dt ~ 3-4).  The dt
     histogram of every panel processed here is cleanly bimodal at those
     values (printed by --stats).  Dash bodies are therefore taken as
     dt >= 2.0 components that contain at least 3 px of dt >= 3.0 core,
     which admits genuine thick ink and rejects streamline bundles.
  2. DASH PERIOD AND DUTY CYCLE along the traced curve separate the four
     styles.  Each chain's centreline is traced first; then the walk
     samples ink occupancy in arc length ALONG that centreline, giving
     the on/off sequence from which per-dash arc lengths are measured.
     Style is assigned per dash from the panel's own measured dash
     population (see classify()), and any dash whose length falls in a
     gap-overlap zone between two styles is tagged `ambiguous` rather
     than guessed.

Regions where the answer is unknowable are tracked as UNKNOWN, not as
"gap": the 5 X grid columns, the 7 phi grid rows, the legend block and the
measured-symbol boxes are all erased before tracing, so a curve crossing
one of them would otherwise fake a dash gap there.  Occupancy samples
inside an erased region are dropped from the dash/gap statistics and the
run they interrupt is stitched.

Measured-symbol positions are read from the committed per-figure JSONs
(this repo's own digitizations) so the symbol erasure cannot drift from
the datasets the computed lines will be compared against.

Calibration is re-detected inside every panel by
digitize_stock_fig14c.calibrate() -- nothing is inherited from a
neighbouring panel (on page 10 the five X columns drift +5.4 px per panel
down the page).  Accuracy carried forward: +-0.0025 x/L within a figure,
and a further ~0.005 x/L when comparing ACROSS a Ref.49 and a Ref.50
figure (see agent-paper-review/2026-08-01-stock-fig14c-16c-digitization.md
section A9).  The polyline itself is a per-row centroid of thick ink, so
its own noise is well under a pixel; the dominant error is the same
calibration budget as the symbols.

Run from paper/:
  python3 repro/cfd/digitize_stock_computed_fronts.py            # all panels
  python3 repro/cfd/digitize_stock_computed_fronts.py fig15a      # one panel
  python3 repro/cfd/digitize_stock_computed_fronts.py --stats     # + geometry
Writes the `stock_computed_*` blocks into the existing per-figure JSONs
(data/stock2006_fig15a_digitized.json etc.), preserving every other key,
plus per-panel check overlays $DIG/check_<key>_computed.png and dash
histograms $DIG/check_<key>_dashes.png.  LOOK AT BOTH.
$DIG: $STOCK_DIGITIZE_DIR, else <tmp>/stock_digitize.
"""
import json
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from digitize_stock_fig14c import (DIG, PAPER, XVALS, calibrate,  # noqa: E402
                                   extract)

# ---- tuned once, from the measured geometry (see module docstring) --------
DT_BODY, DT_CORE = 2.0, 3.0   # px, half-width of thick ink / of its core
MIN_CORE_PX, MIN_DASH_PX = 3, 22
GRID_HALF, SYM_HALF = 8, 21   # erase half-widths, px
CHAIN_GAP_PX = 52             # max endpoint-to-endpoint jump inside a chain
CHAIN_ANGLE = 42.0            # deg, max turn between tangent and the jump
MIN_CHAIN_ARC = 120           # px, shorter chains are dropped as debris
STEP = 1.0                    # arc-length sampling step, px
OCC_HALF = 1                  # occupancy probe half-window, px (3x3)
MIN_INK_SAMPLES = 3           # shorter ink blips are streamline crossings
SOLID_ARC = 110               # px, a run this long with no gap is CONTINUOUS


PANELS = {
    'fig15a': dict(
        key='fig15a', page=8, image=1, img='p9_img1_1970x3538.png',
        zone=(30, 1100), legend=(80, 295, 1320, 1965),
        json='stock2006_fig15a_digitized.json',
        sym_keys=('re_1p52e6_alpha10_squares', 're_6p56e6_alpha10_circles'),
        alpha=10.0, ref=49, facility='DFVLR 3x3 m Goettingen',
        expect='two computed curves: the free vortex-layer separation line '
               '(short dashes) -- which for Re=1.52e6 IS the predicted front, '
               'the flow staying laminar to separation -- and the Re=6.56e6 '
               'transition front, whose style changes along it (Sec. III.C: '
               '"Close to the symmetry planes, pure TS waves dominate, '
               'followed in both directions by simultaneous TS and CF waves '
               '... In the middle part of the prolate spheroid, pure CF wave '
               'triggering is present")',
    ),
    'fig14c': dict(
        key='fig14c', page=7, image=1, img='p8_img1_2027x3630.png',
        zone=(2340, 3400), legend=(2395, 2612, 360, 995),
        json='stock2006_fig14c_digitized.json',
        sym_keys=('measured_re1p52e6_squares', 'measured_re6p49e6_circles'),
        alpha=5.0, ref=49, facility='DFVLR 3x3 m Goettingen',
        expect='Sec. III.C: at Re=1.52e6 laminar to separation except near '
               'streamline 12 (TS); at Re=6.49e6 TS near both symmetry '
               'planes and TS+CF over the rest',
    ),
    'fig17a': dict(
        key='fig17a', page=9, image=0, img='p10_img0_2032x3558.png',
        zone=(30, 1100), legend=(80, 310, 1360, 2020),
        json='stock2006_fig17a_digitized.json',
        sym_keys=('measured_re6p56e6_squares', 'measured_re18p32e6_circles'),
        alpha=10.0, ref=50, facility='CERT/ONERA F1 Le Fauga-Mauzac',
        expect='Sec. III.C: 17a/17b "comparable to those of Ref. 49 (Fig. 15), '
               'except for the high-Reynolds-number case in Fig. 17a"',
    ),
    'fig14a': dict(
        key='fig14a', page=7, image=1, img='p8_img1_2027x3630.png',
        zone=(40, 1120), legend=(90, 270, 1360, 1997),
        json='stock2006_fig14a_digitized.json',
        sym_keys=('measured_squares',),
        alpha=0.0, ref=49, facility='DFVLR 3x3 m Goettingen',
        expect='a single near-vertical pure-TS (continuous) front',
        skip='ALREADY DIGITIZED, and this pipeline must not be used here. '
             'data/stock2006_fig14a_digitized.json already carries the '
             'computed front as `computed_ts_front` (17 azimuths) from '
             'digitize_stock_fig14a.py, and Sec. III.C states transition at '
             'alpha=0 is provoked solely by TS waves, so the whole line is '
             'one mechanism and there is nothing to segment. Independently, '
             'the generic in-panel calibrator REFUSES this panel: at '
             'alpha = 0 the streamlines are horizontal, so the full-width '
             'dark-row test that finds the phi grid rows hits streamlines '
             'instead (it aborts on a row at 165.55 deg, not a 30-multiple). '
             'Forcing it would need a different calibration path for no new '
             'information.',
    ),
    'fig14b': dict(
        key='fig14b', page=7, image=1, img='p8_img1_2027x3630.png',
        zone=(1150, 2300), legend=(1240, 1420, 1330, 2010),
        json='stock2006_fig14b_digitized.json',
        sym_keys=('measured_squares',),
        alpha=2.5, ref=49, facility='DFVLR 3x3 m Goettingen',
        expect='a single slightly slanted pure-TS (continuous) front',
        skip='ALREADY DIGITIZED: data/stock2006_fig14b_digitized.json carries '
             'the computed front as `computed_ts_front` from '
             'digitize_stock_fig14b.py, and Sec. III.C states transition at '
             'alpha = 2.5 is provoked solely by TS waves, so there is one '
             'mechanism and nothing to segment. Same near-horizontal '
             'streamline problem as 14a at the leeward/windward ends.',
    ),
}


# --------------------------------------------------------------------------
def _masks(cfg, dark, y180, y0f, xcols):
    """(body, core, known) inside the panel interior, everything that cannot
    carry curve information erased and recorded in `known`."""
    y0, y1 = int(y180) + 5, int(np.ceil(y0f)) - 4
    sub = dark[y0:y1].copy()
    known = np.ones(sub.shape, bool)
    known[:, :int(xcols[0]) + 4] = False
    known[:, int(xcols[-1]) - 3:] = False
    ly0, ly1, lx0, lx1 = cfg['legend']
    known[max(0, ly0 - y0):max(0, ly1 - y0), lx0:lx1] = False
    for c in xcols:                                    # X grid columns
        known[:, max(0, int(c) - GRID_HALF):int(c) + GRID_HALF + 1] = False
    for phi in (180, 150, 120, 90, 60, 30, 0):         # phi grid rows
        r = int(round(y0f - phi / 180.0 * (y0f - y180))) - y0
        known[max(0, r - GRID_HALF):r + GRID_HALF + 1, :] = False

    A = np.polyfit(xcols, XVALS, 1)
    syms = []
    j = json.load(open(os.path.join(PAPER, 'data', cfg['json'])))
    for k in cfg['sym_keys']:
        for p in j.get(k, []):
            syms.append((p['phi_deg'], p['xL']))
    assert syms, f'no measured symbols found in {cfg["json"]} for erasure'
    for phi, xL in syms:
        r = int(round(y0f - phi / 180.0 * (y0f - y180))) - y0
        c = int(round((2 * xL - 1 - A[1]) / A[0]))
        known[max(0, r - SYM_HALF):r + SYM_HALF + 1,
              max(0, c - SYM_HALF):c + SYM_HALF + 1] = False

    dt = ndimage.distance_transform_edt(sub)
    body = (dt >= DT_BODY) & known
    core = (dt >= DT_CORE) & known
    return sub, body, core, known, dt, y0, y1, len(syms)


MIN_AREA_OVER_L = 2.5         # px, rejects hairline debris at the frame edge


def _dashes(body, core, known):
    """Thick-ink components, each reduced to its principal-axis segment."""
    lab, n = ndimage.label(body)
    ncore = ndimage.sum(core, lab, range(1, n + 1))
    size = ndimage.sum(body, lab, range(1, n + 1))
    objs = ndimage.find_objects(lab)
    out = []
    for i in range(1, n + 1):
        if ncore[i - 1] < MIN_CORE_PX or size[i - 1] < MIN_DASH_PX:
            continue
        sl = objs[i - 1]
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        if h <= 4 and w >= 12:
            continue          # merged near-horizontal streamline bundle
        ys, xs = np.where(lab[sl] == i)
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        P = np.stack([ys, xs]).astype(float)
        mu = P.mean(1, keepdims=True)
        Q = P - mu
        ev, V = np.linalg.eigh(Q @ Q.T / Q.shape[1])
        ax = V[:, -1]
        if ax[0] < 0:
            ax = -ax
        pr = ax @ Q
        L = float(pr.max() - pr.min() + 1)
        if size[i - 1] / L < MIN_AREA_OVER_L:
            continue
        # medial polyline: bin the pixels along the principal axis and take
        # the centroid of each bin, so a CURVED dash (the continuous TS
        # segments hook sharply) is followed instead of being chorded
        nb = max(2, int(round(L / 5.0)))
        edges = np.linspace(pr.min(), pr.max() + 1e-6, nb + 1)
        idx = np.clip(np.digitize(pr, edges) - 1, 0, nb - 1)
        med = []
        for b in range(nb):
            sel = idx == b
            if sel.sum() >= 2:
                med.append((float(ys[sel].mean()), float(xs[sel].mean())))
        if len(med) < 2:
            med = [(float(mu[0, 0] + ax[0] * pr.min()),
                    float(mu[1, 0] + ax[1] * pr.min())),
                   (float(mu[0, 0] + ax[0] * pr.max()),
                    float(mu[1, 0] + ax[1] * pr.max()))]
        if med[0][0] > med[-1][0]:
            med = med[::-1]
        pad = known[max(0, sl[0].start - 2):sl[0].stop + 2,
                    max(0, sl[1].start - 2):sl[1].stop + 2]
        out.append(dict(id=i, size=int(size[i - 1]), L=L, med=med,
                        r0=med[0][0], c0=med[0][1],
                        r1=med[-1][0], c1=med[-1][1],
                        clipped=bool(not pad.all())))
    out.sort(key=lambda d: d['r0'])
    return out, lab


def _tangent(d):
    v = np.array([d['r1'] - d['r0'], d['c1'] - d['c0']])
    n = np.linalg.norm(v)
    return v / n if n else np.array([1.0, 0.0])


def _unknown_frac(known, p, q, n=24):
    """Fraction of the straight path p->q that lies in an erased region."""
    H, W = known.shape
    bad = 0
    for r, c in zip(np.linspace(p[0], q[0], n), np.linspace(p[1], q[1], n)):
        ri, ci = int(round(r)), int(round(c))
        if 0 <= ri < H and 0 <= ci < W and not known[ri, ci]:
            bad += 1
    return bad / n


def _chain(dashes, known):
    """Chain dashes by endpoint proximity plus tangent continuity.  Both
    curve families in these panels are single-valued in phi (verified in the
    check overlay), which is what makes a top-down sweep safe; the tangent
    test is what stops a chain hopping onto a neighbouring curve.  A jump may
    be twice as long when the straight path between the two endpoints runs
    through an erased region (grid line / symbol box), because the gap is
    then explained by the erasure rather than by the drawing."""
    used, chains = set(), []
    for seed in dashes:
        if seed['id'] in used:
            continue
        ch = [seed]
        used.add(seed['id'])
        while True:
            last = ch[-1]
            t = _tangent(last)
            best = None
            for cand in dashes:
                if cand['id'] in used or cand['r0'] <= last['r1'] - 3:
                    continue
                v = np.array([cand['r0'] - last['r1'],
                              cand['c0'] - last['c1']])
                dist = float(np.linalg.norm(v))
                if dist == 0:
                    continue
                lim = CHAIN_GAP_PX
                if _unknown_frac(known, (last['r1'], last['c1']),
                                 (cand['r0'], cand['c0'])) > 0.4:
                    lim = 2 * CHAIN_GAP_PX
                if dist > lim:
                    continue
                cosang = float(v @ t) / dist
                if cosang < np.cos(np.deg2rad(CHAIN_ANGLE)):
                    continue
                if float(v @ _tangent(cand)) / dist < np.cos(
                        np.deg2rad(CHAIN_ANGLE + 12)):
                    continue
                score = dist * (2.0 - cosang)
                if best is None or score < best[0]:
                    best = (score, cand)
            if best is None:
                break
            ch.append(best[1])
            used.add(best[1]['id'])
        chains.append(ch)
    return _merge_cusps(chains)


CUSP_GAP = 26.0    # px


def _revd(d):
    """A dash with its medial polyline reversed."""
    e = dict(d)
    e['med'] = d['med'][::-1]
    e['r0'], e['c0'], e['r1'], e['c1'] = d['r1'], d['c1'], d['r0'], d['c0']
    return e


def _rev(ch):
    return [_revd(d) for d in ch[::-1]]


def _ends(ch):
    return (ch[0]['r0'], ch[0]['c0']), (ch[-1]['r1'], ch[-1]['c1'])


def _merge_cusps(chains):
    """Each computed front has sharp apexes -- its most upstream point, and in
    Fig. 14c a whole out-and-back wedge around streamline 12 -- where the
    tangent reverses and the angle test above deliberately refuses to cross.
    Re-join two chains across such a corner, and only there: any pair of
    endpoints within CUSP_GAP px, with either chain reversed as needed.  No
    angle condition, which is the whole point; every join is visible in the
    check overlay."""
    chains = [list(c) for c in chains]
    changed = True
    while changed:
        changed = False
        for i in range(len(chains)):
            if not chains[i]:
                continue
            for j in range(len(chains)):
                if i == j or not chains[j]:
                    continue
                a, b = chains[i], chains[j]
                ah, at = _ends(a)
                bh, bt = _ends(b)
                # tail->head only.  Head-head / tail-tail joins were tried
                # and REJECTED: in Fig. 14c the streamline-12 TS wedge
                # terminates ON the separation line, and an unrestricted
                # corner merge splices those two different curves (and their
                # different mechanisms) into one nonsensical polyline.  What
                # this conservative rule cannot bridge is left un-traced and
                # painted magenta in the check overlay.
                opts = ((at, bh, a, b),)
                for p, q, aa, bb in opts:
                    if np.hypot(p[0] - q[0], p[1] - q[1]) <= CUSP_GAP:
                        chains[i] = aa + bb
                        chains[j] = []
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
        chains = [c for c in chains if c]
    return chains


def _polyline(chain):
    """Concatenated medial polylines of the chain's dashes; straight
    interpolation across the gaps is exactly what the walk needs."""
    pr, pc = [], []
    for d in chain:
        for r, c in d['med']:
            pr.append(r)
            pc.append(c)
    return np.array(pr), np.array(pc)


def _walk(rows, cols, body, known):
    """Resample the polyline at STEP px of arc length and read occupancy.
    Returns (s, r, c, occ, kno) with occ/kno boolean arrays."""
    ds = np.hypot(np.diff(cols), np.diff(rows))
    s = np.concatenate([[0], np.cumsum(ds)])
    su = np.arange(0, s[-1], STEP)
    ru = np.interp(su, s, rows)
    cu = np.interp(su, s, cols)
    occ = np.zeros(len(su), bool)
    kno = np.zeros(len(su), bool)
    H, W = body.shape
    for i, (r, c) in enumerate(zip(ru, cu)):
        ri, ci = int(round(r)), int(round(c))
        r0, r1 = max(0, ri - OCC_HALF), min(H, ri + OCC_HALF + 1)
        c0, c1 = max(0, ci - OCC_HALF), min(W, ci + OCC_HALF + 1)
        occ[i] = body[r0:r1, c0:c1].any()
        kno[i] = known[max(0, ri - 3):ri + 4, max(0, ci - 3):ci + 4].all()
    return su, ru, cu, occ, kno


def _runs(occ, kno):
    """Ink/gap runs in arc length, stitching across UNKNOWN stretches."""
    state, start, out = None, 0, []
    for i in range(len(occ)):
        if not kno[i]:
            continue
        v = bool(occ[i])
        if state is None:
            state, start = v, i
        elif v != state:
            out.append((state, start, i - 1))
            state, start = v, i
    if state is not None:
        out.append((state, start, len(occ) - 1))
    # An ink run shorter than MIN_INK_SAMPLES is a streamline-pair crossing
    # (two thin lines that touch reach DT_BODY), not a dash: absorb it into
    # the surrounding gap.  Then coalesce equal neighbours.
    clean = []
    for v, s, e in out:
        if v and (e - s + 1) < MIN_INK_SAMPLES:
            v = False
        clean.append((v, s, e))
    merged = []
    for r in clean:
        if merged and merged[-1][0] == r[0]:
            merged[-1] = (merged[-1][0], merged[-1][1], r[2])
        else:
            merged.append(r)
    return merged


# --------------------------------------------------------------------------
# STYLE MODEL.  Stock's four styles are (dash, gap) pairs.  Measured on
# Fig. 15a -- the one panel that carries all four -- as ink-run/gap arc
# lengths along the traced centreline, and normalized by the panel WIDTH
# (= 2 X/a) so they transfer between rasters of slightly different dpi:
#
#   style        mechanism   dash px   gap px   dash/width   gap/width
#   short dash   separation     14       3       0.00843      0.00181
#   midsized     TS+CF          31       4       0.01867      0.00241
#   long dash    CF             64      11       0.03856      0.00663
#   continuous   TS            (no gaps at all)
#
# Two features are used, not one: the DASH length separates short from
# midsized from long (ratios 2.2x and 2.1x), and the GAP length separates
# long (11 px) from both short and midsized (3-4 px, which are NOT
# separable from each other by gap alone).  Every panel prints its own
# measured run/gap histogram so these templates can be checked (--stats).
STYLE = (
    ('separation', 'short dashed', 0.00843, 0.00181),
    ('TS+CF', 'midsized dashed', 0.01867, 0.00241),
    ('CF', 'long dashed', 0.03856, 0.00663),
)
TS_MIN = 0.075          # a gap-free run longer than this fraction of the
#                         panel width can only be the continuous (TS) style:
#                         it exceeds 1.9x the longest single CF dash.
MERGE_PEN = 0.45        # cost per extra dash absorbed by a streamline bridge
SWITCH_PEN = 1.30       # cost of a style change between adjacent runs
MAX_MERGE = 8           # a run is never read as more than this many dashes
AMB_MARGIN = 0.50       # if best and runner-up cost differ by less than
#                         this, the run's mechanism is reported ambiguous


W_GAP = 1.5      # weight of the gap-length term relative to the dash term


def _run_costs(L, gobs, width):
    """Cost of explaining an ink run of arc length L px, bounded by the
    shorter adjacent gap `gobs` px (None if neither side is usable), by each
    of Stock's four styles.  BOTH features matter: the dash length separates
    short from midsized from long, and the gap length is what separates the
    long dash (gap ~11 px on Fig. 15a) from the short and midsized dashes
    (gap 3-4 px), which the dash length alone confuses whenever streamline
    bridges merge two midsized dashes into one 66-px run."""
    out = {}
    for name, sty, df, gf in STYLE:
        d, g = df * width, gf * width
        best = None
        for m in range(1, MAX_MERGE + 1):
            pred = m * d + (m - 1) * g
            c = abs(L - pred) / d + MERGE_PEN * (m - 1)
            if gobs is not None:
                c += W_GAP * abs(gobs - g) / max(g, 0.0015 * width)
            if best is None or c < best[0]:
                best = (c, m)
        out[name] = best
    ts = 0.0 if L >= TS_MIN * width else (TS_MIN * width - L) / (0.02 * width)
    out['TS'] = (ts, 1)
    return out


def _bounding_gaps(runs, width):
    """For each ink run, the shorter adjacent gap, ignoring gaps long enough
    to be a chain jump across an erased region rather than a drawn gap."""
    seq = [(v, (e - s + 1) * STEP) for v, s, e in runs]
    lim = 0.02 * width          # a drawn gap is never this long
    out = []
    for i, (v, L) in enumerate(seq):
        if not v:
            continue
        cand = []
        for j in (i - 1, i + 1):
            if 0 <= j < len(seq) and not seq[j][0] and seq[j][1] <= lim:
                cand.append(seq[j][1])
        out.append(min(cand) if cand else None)
    return out


def _viterbi(arcs, gobs, width):
    """Piecewise-constant style over the run sequence (merge tolerant)."""
    names = [s[0] for s in STYLE] + ['TS']
    costs = [_run_costs(L, g, width) for L, g in zip(arcs, gobs)]
    n, k = len(arcs), len(names)
    D = np.full((n, k), np.inf)
    B = np.zeros((n, k), int)
    for j, nm in enumerate(names):
        D[0, j] = costs[0][nm][0]
    for i in range(1, n):
        for j, nm in enumerate(names):
            c = costs[i][nm][0]
            prev = D[i - 1] + SWITCH_PEN * (np.arange(k) != j)
            B[i, j] = int(np.argmin(prev))
            D[i, j] = c + prev[B[i, j]]
    path = [int(np.argmin(D[-1]))]
    for i in range(n - 1, 0, -1):
        path.append(int(B[i, path[-1]]))
    path = path[::-1]
    # backward table, so that the confidence of run i is the extra TOTAL path
    # cost incurred by forcing any other style on that run -- not merely the
    # local cost difference, which ignores how the neighbours constrain it
    E = np.full((n, k), np.inf)
    E[-1] = 0.0
    for i in range(n - 2, -1, -1):
        for j in range(k):
            nxt = [costs[i + 1][names[j2]][0] + E[i + 1, j2]
                   + SWITCH_PEN * (j2 != j) for j2 in range(k)]
            E[i, j] = min(nxt)
    lab, marg, mult = [], [], []
    for i, j in enumerate(path):
        tot = D[i] + E[i]
        best = tot[j]
        second = min(v for j2, v in enumerate(tot) if j2 != j)
        lab.append(names[j])
        marg.append(float(second - best))
        mult.append(int(costs[i][names[j]][1]))
    return lab, marg, mult


MECH_NOTE = {'separation': 'free vortex-layer separation (short dashed)',
             'TS+CF': 'TS and CF waves (midsized dashed)',
             'CF': 'CF waves (long dashed)',
             'TS': 'TS waves (continuous)'}


# --------------------------------------------------------------------------
def _segments(pts):
    out = []
    for p in pts:
        if out and out[-1]['mechanism'] == p['mechanism']:
            out[-1]['phi_to'] = p['phi_deg']
            out[-1]['xL_to'] = p['xL']
            out[-1]['n'] += 1
        else:
            out.append(dict(mechanism=p['mechanism'], phi_from=p['phi_deg'],
                            phi_to=p['phi_deg'], xL_from=p['xL'],
                            xL_to=p['xL'], n=1))
    return out


def run(cfg, stats=False):
    os.makedirs(DIG, exist_ok=True)
    path = f'{DIG}/{cfg["img"]}'
    if not os.path.exists(path):
        extract(cfg)
    dark = np.array(Image.open(path).convert('L'), copy=True) < 128
    print(f'=== Fig. {cfg["key"][3:]}  {cfg["img"]} '
          f'(page {cfg["page"]+1} image {cfg["image"]})')
    y180, y0f, xcols, A = calibrate(dark, cfg['zone'])
    width = float(xcols[-1] - xcols[0])
    sub, body, core, known, dt, y0, y1, nsym = _masks(
        cfg, dark, y180, y0f, xcols)
    print(f'  panel width {width:.1f} px; {nsym} measured symbols erased; '
          f'thick ink (dt>={DT_BODY}) {body.sum()} of {sub.sum()} dark px')
    if stats:
        hh, ee = np.histogram(dt[sub], bins=np.arange(0.5, 6.6, 0.5))
        print('  dt histogram:', ' '.join(f'{e:.1f}:{c}'
                                          for c, e in zip(hh, ee) if c))
    dashes, lab = _dashes(body, core, known)
    chains = _chain(dashes, known)

    def x2X(px):
        return float(A[0] * px + A[1])

    def y2phi(py):
        return 180.0 * (y0f - (py + y0)) / (y0f - y180)

    curves = []
    for ch in chains:
        pr, pc = _polyline(ch)
        arc = float(np.hypot(np.diff(pc), np.diff(pr)).sum())
        if arc < MIN_CHAIN_ARC:
            continue
        su, ru, cu, occ, kno = _walk(pr, pc, body, known)
        runs = _runs(occ, kno)
        ink = [(s, e) for v, s, e in runs if v]
        if len(ink) < 2:
            continue
        arcs = [(e - s + 1) * STEP for s, e in ink]
        gaps = [(e - s + 1) * STEP for v, s, e in runs if not v]
        gobs = _bounding_gaps(runs, width)
        labs, margs, mults = _viterbi(arcs, gobs, width)
        if len(arcs) < 3:
            # too few dashes to establish a period: no mechanism claim
            margs = [0.0] * len(arcs)
        pts = []
        for i, (s, e) in enumerate(ink):
            m = labs[i] if margs[i] >= AMB_MARGIN else f'ambiguous:{labs[i]}?'
            for kk in range(s, e + 1):
                pts.append(dict(phi_deg=round(y2phi(ru[kk]), 2),
                                xL=round((x2X(cu[kk]) + 1) / 2, 4),
                                mechanism=m))
        curves.append(dict(chain=ch, arc=arc, ru=ru, cu=cu, ink=ink,
                           arcs=arcs, gaps=gaps, labs=labs, margs=margs,
                           mults=mults, pts=pts,
                           known_frac=float(kno.mean())))
    curves.sort(key=lambda c: -c['arc'])
    for i, c in enumerate(curves):
        amb = sum(len(range(s, e + 1)) for (s, e), m in zip(c['ink'], c['margs'])
                  if m < AMB_MARGIN)
        tot = sum(len(range(s, e + 1)) for s, e in c['ink'])
        c['ambiguous_arc_frac'] = round(amb / tot, 3)
        print(f'  curve {i}: arc {c["arc"]:6.0f} px, {len(c["chain"])} dashes, '
              f'{len(c["arcs"])} runs (median {np.median(c["arcs"]):.0f} px, '
              f'gap median {np.median(c["gaps"]) if c["gaps"] else 0:.0f}), '
              f'known {c["known_frac"]:.2f}, ambiguous arc frac '
              f'{c["ambiguous_arc_frac"]:.3f}')
        for s in _segments(c['pts']):
            if s['n'] < 4:
                continue
            print(f'      {s["mechanism"]:22s} phi {s["phi_from"]:7.2f}->'
                  f'{s["phi_to"]:7.2f}   x/L {s["xL_from"]:.4f}->'
                  f'{s["xL_to"]:.4f}   ({s["n"]} px)')
        if stats:
            print('      runs px:', [round(v) for v in c['arcs']])
            print('      m      :', c['mults'])
            print('      margin :', [round(v, 2) for v in c['margs']])
            print('      gaps px:', [round(v) for v in c['gaps']])
    return dict(cfg=cfg, curves=curves, width=width, y180=y180, y0f=y0f,
                xcols=xcols, y0=y0, sub=sub, body=body, known=known,
                dashes=dashes, A=A, lab=lab)


def emit(res, stats=False):
    """Write the curves into the panel's JSON and draw the check images."""
    cfg = res['cfg']
    curves = res['curves']
    y0, A, y180, y0f = res['y0'], res['A'], res['y180'], res['y0f']
    dst = os.path.join(PAPER, 'data', cfg['json'])
    j = json.load(open(dst))
    blocks = {}
    for i, c in enumerate(curves):
        pts, keep = c['pts'], []
        # decimate to ~1 point per 12 px of arc but never drop a mechanism
        # change, so the segment boundaries stay exact
        for k, p in enumerate(pts):
            if (k == 0 or k == len(pts) - 1 or k % 12 == 0
                    or p['mechanism'] != pts[k - 1]['mechanism']):
                keep.append(p)
        blocks[f'curve{i}'] = dict(
            n_points=len(keep),
            arc_px=round(c['arc'], 1),
            n_dashes=len(c['chain']),
            n_ink_runs=len(c['arcs']),
            median_dash_px=round(float(np.median(c['arcs'])), 1),
            median_gap_px=round(float(np.median(c['gaps'])), 1)
            if c['gaps'] else None,
            known_arc_frac=round(c['known_frac'], 3),
            ambiguous_arc_frac=c['ambiguous_arc_frac'],
            mechanism_segments=[
                {k2: v2 for k2, v2 in s2.items()}
                for s2 in _segments(c['pts']) if s2['n'] >= 4],
            points=keep)
    # ---- name each curve: separation by style, fronts by which measured
    # symbol set they sit closest to at matched azimuth ------------------
    sym = {}
    for k in cfg['sym_keys']:
        pts = json.load(open(dst)).get(k, [])
        if pts:
            sym[k] = pts
    names, assign_log = {}, []
    for i, c in enumerate(curves):
        segs = [s2 for s2 in _segments(c['pts']) if s2['n'] >= 4]
        dom = max(segs, key=lambda s2: s2['n'])['mechanism'] if segs else '?'
        if dom == 'separation':
            names[f'curve{i}'] = 'separation_line_short_dashed'
            assign_log.append(f'curve{i}: dominant style is the short dash '
                              '-> free vortex-layer separation line')
            continue
        best = None
        for k, pts in sym.items():
            ds = []
            for q in pts:
                near = min(c['pts'], key=lambda p:
                           abs(p['phi_deg'] - q['phi_deg']))
                if abs(near['phi_deg'] - q['phi_deg']) <= 6.0:
                    ds.append(abs(near['xL'] - q['xL']))
            if ds:
                m = float(np.mean(ds))
                if best is None or m < best[0]:
                    best = (m, k, len(ds))
        if best is None:
            names[f'curve{i}'] = f'curve{i}_unassigned'
            assign_log.append(f'curve{i}: no measured symbol within 6 deg of '
                              'it -- Reynolds number NOT assigned')
        else:
            mm = re.search(r'(\d+p\d+e\d+)', best[1])
            tag = mm.group(1) if mm else f'set{i}'
            nm = f'front_re{tag}'
            if nm in names.values():          # a second, shorter piece
                k2 = 2
                while f'{nm}_piece{k2}' in names.values():
                    k2 += 1
                nm = f'{nm}_piece{k2}'
            names[f'curve{i}'] = nm
            assign_log.append(
                f'curve{i} -> {best[1]}: mean |dx/L| {best[0]:.3f} over '
                f'{best[2]} azimuth-matched symbols (closest of '
                f'{len(sym)} sets)')
    blocks = {names.get(k, k): v for k, v in blocks.items()}
    j['stock_computed_lines'] = dict(
        curve_naming=assign_log,
        source=f'Stock, AIAA J 44(1) 2006, Fig. {cfg["key"][3:]} '
               f'(DOI 10.2514/1.16026), alpha={cfg["alpha"]} deg, '
               f'Ref. {cfg["ref"]} ({cfg["facility"]}); COMPUTED transition '
               f'lines, mechanism-coded by line style; digitized '
               f'{__import__("datetime").date.today()} by '
               'repro/cfd/digitize_stock_computed_fronts.py',
        mechanism_key='Stock Sec. III.C: pure TS = continuous, pure CF = long '
                      'dashed, TS and CF together = midsized dashed, '
                      'separation = short dashed; streamlines are thin '
                      'continuous and are excluded by line width',
        method='thick-ink (distance-transform) separation from the '
               'streamlines, dash chaining, then a 4-state piecewise-constant '
               'fit of Stock style templates to the measured dash AND gap '
               'arc lengths along each traced centreline; runs whose '
               'best/runner-up total path cost differ by less than '
               f'{AMB_MARGIN} are reported as ambiguous',
        convention='phi=0 windward symmetry line (Stock convention); '
                   'x/L=(X/a+1)/2 from nose',
        accuracy='+-0.0025 x/L within this figure; add ~0.005 x/L when '
                 'comparing across a Ref.49 and a Ref.50 figure. Mechanism '
                 'boundary azimuths are localized to about one dash period, '
                 '~2-8 deg depending on style',
        expectation_from_text=cfg['expect'],
        **blocks)
    # ---- check overlay ----------------------------------------------------
    im = Image.open(f'{DIG}/{cfg["img"]}').convert('RGB')
    # thick ink that ended up in NO emitted curve, so an omission is visible
    used = {d['id'] for c in curves for d in c['chain']}
    lost = [d for d in res['dashes'] if d['id'] not in used]
    miss = np.isin(res['lab'], [d['id'] for d in lost])
    tot_px = sum(d['size'] for d in res['dashes'])
    lost_px = sum(d['size'] for d in lost)
    print(f'  un-traced thick ink: {lost_px} of {tot_px} px '
          f'({lost_px/max(tot_px,1):.1%}) in {len(lost)} components'
          + (', largest at ' + ', '.join(
              f'(phi {180.0*(y0f-(d["r0"]+y0))/(y0f-y180):.0f}, '
              f'x/L {(A[0]*d["c0"]+A[1]+1)/2:.2f}, L{d["L"]:.0f})'
              for d in sorted(lost, key=lambda z: -z['L'])[:4]) if lost else ''))
    j['stock_computed_lines_untraced_px'] = [lost_px, tot_px]
    arr = np.array(im)
    ys, xs = np.where(miss)
    arr[ys + y0, xs] = (255, 0, 255)
    im = Image.fromarray(arr)
    band = im.crop((0, y0 - 30, im.width, y0 + res['sub'].shape[0] + 30))
    dr = ImageDraw.Draw(band)
    COL = {'TS': (220, 0, 0), 'TS+CF': (230, 130, 0), 'CF': (0, 90, 220),
           'separation': (0, 150, 60)}
    for c in curves:
        for p in c['pts']:
            r = y0f - p['phi_deg'] / 180.0 * (y0f - y180) - (y0 - 30)
            x = (2 * p['xL'] - 1 - A[1]) / A[0]
            col = COL.get(p['mechanism'].replace('ambiguous:', '')
                          .rstrip('?'), (150, 0, 150))
            dr.ellipse([x - 2, r - 2, x + 2, r + 2], fill=col)
    chk = f'{DIG}/check_{cfg["key"]}_computed.png'
    band.save(chk)
    band.resize((band.width // 2, band.height // 2)).save(
        chk.replace('.png', '_half.png'))
    # the symbol-only pass left a `not_digitized` note saying the computed
    # curves were deliberately skipped; that is now stale for this panel
    if 'not_digitized' in j and 'COMPUTED curves' in str(j['not_digitized']):
        j['not_digitized_superseded'] = j.pop('not_digitized')
        j['not_digitized'] = (
            'Superseded on 2026-08-02: the computed curves ARE now digitized, '
            'see `stock_computed_lines` (and the previous note, kept as '
            '`not_digitized_superseded`, for what the symbol-only pass '
            'declined to do). Still not extracted: the thin-solid streamlines, '
            'and the fraction of thick ink reported in '
            '`stock_computed_lines_untraced_px`.')
    json.dump(j, open(dst, 'w'), indent=1)
    print(f'  wrote {dst}  [stock_computed_lines: {len(blocks)} curves]')
    print(f'        {chk}')
    return blocks


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    stats = '--stats' in sys.argv
    keys = args if args else list(PANELS)
    for k in keys:
        if PANELS[k].get('skip'):
            print(f'=== Fig. {k[3:]}: SKIPPED -- {PANELS[k]["skip"]}')
            continue
        emit(run(PANELS[k], stats=stats), stats=stats)


if __name__ == '__main__':
    main()
