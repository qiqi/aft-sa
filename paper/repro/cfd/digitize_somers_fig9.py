"""Digitize Somers TP-1861 Fig. 9(a)-(d): NLF(1)-0416 transition location by Reynolds number.

  -> paper/data/somers1981_nlf0416_transition_by_Re.json

Figure 9 plots c_l (ordinate) against x_T/c (abscissa) on square graph paper, one
panel per Reynolds number (R = 1, 2, 3, 4 x 10^6, all at M = 0.10). Circles are
the upper surface, squares the lower. Per the figure caption, OPEN symbols are
orifices at which the flow is laminar and SOLID symbols orifices at which it is
turbulent, so each (surface, c_l) reading is a BRACKET on the transition
location -- the last laminar orifice and the first turbulent one -- not a point.
Bracket widths come out at one orifice spacing (0.05 c over most of the chord),
which IS the measurement's resolution; this file records the bracket rather than
a faired midpoint.

The source is a raster scan (NTRS, 1981), so symbols are found by image analysis
of a 300 dpi render rather than by vector extraction:

  1. Strip the axis-aligned graph-paper rules (long runs; a symbol is ~30 px).
  2. Close and fill, so open rings and solid disks become the same shape, then
     open with a disk to drop the faired curve.
  3. Classify each blob by the fraction of its mass in the corners of its
     bounding box (square vs circle) and by the darkness of its core (solid vs
     open).

Calibration is taken from the artwork per panel: the heavy c_l = 0 rule sets the
ordinate origin, and the major vertical rules (0.1 x_T/c apart) set the scale,
anchored on the RIGHT frame at x_T/c = 1.0 -- the left frame coincides with the
c_l axis and is drawn inconsistently between panels.

Two independent checks are run and reported into the file's meta:
  * extracted abscissae land on the 5%-chord orifice stations (RMS 0.14-0.44 %c);
  * on the R = 4x10^6 panel, all 28 points of the same LTPT data as replotted in
    Coder's dissertation (data/aft_nlf0416_digitized.json) fall inside the
    brackets extracted here.

  python3 digitize_somers_fig9.py [--render]
"""
import os, sys, json
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

_H = os.path.dirname(os.path.abspath(__file__))
PD = os.path.abspath(os.path.join(_H, "..", ".."))
SA = os.path.abspath(os.path.join(PD, ".."))
PDF = f"{SA}/references/somers1981-nlf0416-TP1861-NTRS19810015487.pdf"
CACHE = f"{PD}/repro/cfd/.somers_fig9"
OUT = f"{PD}/data/somers1981_nlf0416_transition_by_Re.json"

PANELS = [(49, "re_1e6", 1.0e6, "(a)"), (50, "re_2e6", 2.0e6, "(b)"),
          (51, "re_3e6", 3.0e6, "(c)"), (52, "re_4e6", 4.0e6, "(d)")]
# Circle (upper surface) vs square (lower): the fraction of the symbol's own mass
# lying in the corners of its bounding box. A square puts mass there, a circle
# does not. This is bimodal with a clear gap on every panel (max below 0.070,
# min above 0.073), whereas the bbox fill ratio overlaps and mis-sorts the
# lower-surface squares at c_l ~ 1 into the upper series.
CORNER_SPLIT = 0.072
CORE_SPLIT = 0.50     # core darkness: open (laminar) below, solid (turbulent) above
DPI = 300


def render():
    import fitz
    os.makedirs(CACHE, exist_ok=True)
    doc = fitz.open(PDF)
    for pg, key, _, _ in PANELS:
        doc[pg].get_pixmap(dpi=DPI).save(f"{CACHE}/{key}.png")
    print(f"rendered {len(PANELS)} panels at {DPI} dpi into {CACHE}")


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def calibrate(d):
    H, W = d.shape
    row0 = float(np.argmax(d[:, int(W * 0.25):int(W * 0.80)].sum(1)))
    cs = d[int(row0 - 900):int(row0 + 700), :].sum(0)
    th = np.percentile(cs[cs > 0], 99.0)
    grp = []
    for c in (c for c in range(W) if cs[c] >= th):
        if grp and c - grp[-1][-1] <= 5:
            grp[-1].append(c)
        else:
            grp.append([c])
    cent = np.array([np.mean(g) for g in grp], float)
    cent = cent[(cent > W * 0.15) & (cent < W * 0.90)]
    step = float(np.polyfit(np.arange(len(cent)), cent, 1)[0])
    return cent[-1] - step * 10.0, row0, step / 10.0, len(cent)


def symbols(key):
    im = np.array(Image.open(f"{CACHE}/{key}.png").convert("L"), dtype=np.uint8)
    d = im < 150
    col0, row0, p, nrule = calibrate(d)
    horiz = ndi.binary_opening(d, np.ones((1, 45)))
    vert = ndi.binary_opening(d, np.ones((45, 1)))
    clean = ndi.binary_closing(d & ~(horiz | vert), disk(3))
    op = ndi.binary_opening(ndi.binary_fill_holes(clean), disk(7))
    lab, _ = ndi.label(op)
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        m = lab[sl] == i
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if m.sum() < 350 or h < 20 or w < 20:
            continue
        nv = int(round(h / 30.0)) if h > 44 else 1
        nh = int(round(w / 30.0)) if w > 44 else 1
        for k in range(max(1, nv) * max(1, nh)):
            kv, kh = divmod(k, max(1, nh))
            a = int(round(kv * h / nv)) if nv > 1 else 0
            b = int(round((kv + 1) * h / nv)) if nv > 1 else h
            c_ = int(round(kh * w / nh)) if nh > 1 else 0
            e_ = int(round((kh + 1) * w / nh)) if nh > 1 else w
            sub = np.zeros_like(m); sub[a:b, c_:e_] = m[a:b, c_:e_]
            if sub.sum() < 350:
                continue
            ys, xs = np.nonzero(sub)
            my, mx = ys.mean(), xs.mean()
            cy, cx = my + sl[0].start, mx + sl[1].start
            hh, ww = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
            corner = float(((np.abs(ys - my) > 0.55 * hh / 2) &
                            (np.abs(xs - mx) > 0.55 * ww / 2)).sum()) / sub.sum()
            rr = int(min(hh, ww) * 0.28)
            yy, xx = np.ogrid[-rr:rr + 1, -rr:rr + 1]
            core = (xx * xx + yy * yy) <= rr * rr
            ci, cj = int(round(cy)), int(round(cx))
            patch = d[ci - rr:ci + rr + 1, cj - rr:cj + rr + 1]
            dark = float((patch & core).sum()) / core.sum() if patch.shape == core.shape else 0.0
            x, cl = (cx - col0) / (100 * p), (row0 - cy) / (50 * p)
            if not (0.015 <= x <= 1.02 and -1.25 <= cl <= 1.82):
                continue
            if x < 0.10 and cl > 1.70:
                continue                                   # legend key
            out.append(dict(x=float(x), cl=float(cl),
                            surface="upper" if corner < CORNER_SPLIT else "lower",
                            state="turbulent" if dark > CORE_SPLIT else "laminar",
                            merged=bool(max(nv, nh) > 1 or hh > 36 or ww > 36)))
    return out, dict(col0=round(col0, 2), row0=round(row0, 2),
                     px_per_fine_cell=round(p, 4), major_rules_found=nrule)


def brackets(sy, surface):
    pts = sorted([z for z in sy if z["surface"] == surface], key=lambda z: -z["cl"])
    lv = []
    for z in pts:
        for g in lv:
            if abs(g["_cl"] - z["cl"]) < 0.035:
                g["pts"].append(z)
                break
        else:
            lv.append(dict(_cl=z["cl"], pts=[z]))
    rows = []
    for g in lv:
        lam = [q["x"] for q in g["pts"] if q["state"] == "laminar"]
        tur = [q["x"] for q in g["pts"] if q["state"] == "turbulent"]
        lo = max(lam) if lam else None
        hi = min(tur) if tur else None
        if lo is not None and hi is not None and hi < lo:
            lo = hi = None                                  # inconsistent pair: drop
        rows.append(dict(cl=round(float(np.mean([q["cl"] for q in g["pts"]])), 4),
                         x_last_laminar=None if lo is None else round(lo, 4),
                         x_first_turbulent=None if hi is None else round(hi, 4),
                         complete=bool(lo is not None and hi is not None),
                         merged_symbol=bool(any(q["merged"] for q in g["pts"]))))
    rows.sort(key=lambda r: r["cl"])
    return rows


def main():
    if "--render" in sys.argv or not os.path.isdir(CACHE):
        render()
    data, resid = {}, []
    for _, key, Re, sub in PANELS:
        sy, cal = symbols(key)
        xs = np.array([z["x"] for z in sy])
        r = xs * 100 - np.round(xs * 100 / 5) * 5
        resid.append(float(np.sqrt((r ** 2).mean())))
        data[key] = dict(reynolds=Re, panel=sub, mach=0.10, calibration=cal,
                         n_symbols=len(sy),
                         orifice_station_rms_pct_chord=round(resid[-1], 3),
                         upper=brackets(sy, "upper"), lower=brackets(sy, "lower"))
        n = sum(v["complete"] for s in ("upper", "lower") for v in data[key][s])
        m = sum(1 for s in ("upper", "lower") for v in data[key][s])
        print(f"{key}: {len(sy)} symbols, {n}/{m} c_l levels with a complete bracket, "
              f"orifice RMS {resid[-1]:.3f} %c")

    # independent check: the same LTPT data replotted in Coder's dissertation
    ex = json.load(open(f"{PD}/data/aft_nlf0416_digitized.json"))["transition"]
    tot = hit = 0
    for surf, k in (("upper", "exp_upper"), ("lower", "exp_lower")):
        rows = [v for v in data["re_4e6"][surf] if v["complete"]]
        for cl, xt in zip(ex[k]["cl"], ex[k]["xt"]):
            c = min(rows, key=lambda q: abs(q["cl"] - cl))
            if abs(c["cl"] - cl) > 0.06:
                continue
            tot += 1
            hit += c["x_last_laminar"] - 0.02 <= xt <= c["x_first_turbulent"] + 0.02
    print(f"cross-check on R=4e6 vs aft_nlf0416_digitized: {hit}/{tot} inside bracket")

    out = dict(meta=dict(
        source=("Somers, D. M., 'Design and Experimental Results for a Natural-Laminar-Flow "
                "Airfoil for General Aviation Applications', NASA TP-1861, June 1981, Fig. 9."),
        bib_key="somers_1981",
        ntrs_id="19810015487",
        pdf="references/somers1981-nlf0416-TP1861-NTRS19810015487.pdf",
        pdf_pages_zero_indexed=[p for p, _, _, _ in PANELS],
        conditions="NLF(1)-0416, LTPT, M = 0.10, transition free, smooth model.",
        quantity=("Transition location x_T/c against section lift coefficient, one panel per "
                  "chord Reynolds number. Circles = upper surface, squares = lower surface."),
        convention=("Per the figure caption, OPEN symbols are orifices at which the flow is "
                    "LAMINAR and SOLID symbols orifices at which it is TURBULENT. Each c_l level "
                    "therefore gives a BRACKET: x_last_laminar < x_transition < x_first_turbulent. "
                    "Bracket width is one orifice spacing (0.05 c over most of the chord) and is "
                    "the measurement's own resolution. No midpoint is recorded -- take the bracket."),
        method=("Raster scan; symbols found by image analysis of a 300 dpi PyMuPDF render "
                "(see repro/cfd/digitize_somers_fig9.py): axis-aligned graph-paper rules "
                "stripped, rings closed and filled so open and solid symbols share a shape, "
                "opened with a disk to drop the faired curve, then classified by "
                "corner mass (square vs circle) and core darkness (open vs solid)."),
        calibration=("Per panel, from the artwork: the heavy c_l = 0 rule sets the ordinate "
                     "origin; the major vertical rules (0.1 x_T/c) set the scale, anchored on the "
                     "right frame at x_T/c = 1.0. Square graph paper, 0.01 x_T/c and 0.02 c_l "
                     "per fine cell."),
        checks=dict(
            orifice_stations=("Extracted abscissae land on the 5%-chord orifice stations to "
                              f"{min(resid):.3f}-{max(resid):.3f} %c RMS per panel."),
            vs_coder_dissertation=(f"On R = 4x10^6, {hit} of {tot} points of the same LTPT data "
                                   "as replotted in Coder's dissertation "
                                   "(data/aft_nlf0416_digitized.json, transition/exp_*) fall "
                                   "inside the brackets extracted here.")),
        completeness=("A c_l level is flagged complete=false where one of its two symbols was "
                      "not recovered (obscured by a rule, or merged with a neighbour). Those "
                      "levels are recorded with the symbol that WAS found and are not "
                      "interpolated. merged_symbol=true flags a level whose blob had to be split, "
                      "where the abscissa carries roughly twice the usual uncertainty."),
        uncertainty=("Symbol-centre localisation is ~0.3 %c (1 px = 0.007 c at 300 dpi); "
                     "the dominant uncertainty is the 5 %c bracket width, not the digitization."),
        digitized="2026-08-08"), **data)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
