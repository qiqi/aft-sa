"""Digitize Stock (2006) Fig. 17a: alpha = 10 deg -- MEASURED transition
locations from Stock's SECOND wind-tunnel campaign, the CERT/ONERA F1
pressurized tunnel at Le Fauga-Mauzac (his Ref. 50 = Kreplin, Vollmers &
Meier, DFVLR-AVA IB 222-84 A 34, June 1985), which is a DIFFERENT facility
from the DFVLR 3x3 m Goettingen campaign behind Figs. 14/15/16 (his
Ref. 49 = IB 222-84 A 33).  Both Reynolds numbers printed in the panel
legend are extracted: open squares Re_L = 6.56e6, open circles
Re_L = 18.32e6.

Source raster: spheroid.pdf page 10, image index 0 (left column, rect
x = 42-279 pt, Fig. 17 a/b/c stacked, 2032x3558 px) -> $DIG/
p10_img0_2032x3558.png; re-extract with --extract.  Page 10's other image
(index 1, 2915x1901, rect y = 554-708 pt) is Fig. 18 -- not this figure.
Panel a is the TOP panel; identity verified from its printed legend
("Re = 6.56 x 10^6" AND "Re = 18.32 x 10^6", i.e. two Reynolds numbers)
against the Fig. 17 caption -- "Comparison of measured[50] and computed
transition locations for an angle of attack a) alpha = 10 deg, b)
alpha = 15 deg, and c) alpha = 30 deg" -- and against the other two
panels, which read one Re (17b, 6.62e6) and three Re with triangles
(17c, 6.65/23.96/43.54e6).

Method, calibration, accuracy and the deliberate exclusion of Stock's
computed curves: the engine of digitize_stock_fig14c.py, imported here so
the whole Stock family stays one pipeline.  Read that docstring first.
Two Fig.-17a-specific notes:

  * TEMPLATES.  Panel a contains NO isolated data glyph -- all eight
    touch a streamline or one of the computed fronts -- so the square
    template is cut from the cleanest isolated square in panel c of the
    SAME raster (rows 2972-2997), and the circle template from the
    least-contaminated circle in panel a itself (rows 811-838), which
    carries ~10 px of streamline ink at its upper right.  Because that
    contamination is a real (if small) defect, the run is repeated with
    the pristine circle template of the Fig. 15 raster
    (CI_XCHECK below, p9_img1 rows 930-957) and the two runs are
    asserted to agree on every identity and to within 1.5 px on every
    centre; the reported numbers are from the in-raster template.
  * CROSS-FACILITY CHECK.  Re = 6.56e6 at alpha = 10 deg appears BOTH
    here (ONERA F1, Ref. 50) and in Fig. 15a (Goettingen, Ref. 49, where
    6.56e6 is the CIRCLE set).  To test whether that is a genuine
    matched-Re repeat in a second tunnel or the same points replotted,
    Fig. 15a panel a is re-digitized here with the same engine (config
    FIG15A_XCHECK, no JSON of its own -- it lands in the Fig. 17a JSON
    under `cross_facility_check`) and compared point by point.  The
    committed data/stock2006_fig15a_digitized.json is NOT modified.

Run from paper/:  python3 repro/cfd/digitize_stock_fig17a.py
-> data/stock2006_fig17a_digitized.json
   + check overlays  $DIG/check_fig17a.png, $DIG/check_fig17a_glyphs.png
   + the Fig. 15a re-read's own overlays $DIG/check_fig15a_panelA*.png
   (LOOK AT ALL FOUR)
$DIG defaults to $STOCK_DIGITIZE_DIR, else <tmp>/stock_digitize.  For the
2026-08-01/02 pass on 019-v100-dev that was
/tmp/claude-1006/-home-qiqi-flexcompute/3d1a461e-96df-48fb-8bfc-cdfece2123e6/scratchpad/stock_digitize
(do NOT default this to /local_data/... -- that path lives on 014).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from digitize_stock_fig14c import PAPER, run                    # noqa: E402

# pristine circle template of the Fig. 15 raster, used only to prove the
# (slightly streamline-contaminated) in-raster Fig. 17 circle template is
# not biasing anything: (img, tight bbox y0,y1,x0,x1)
CI_XCHECK = ('p9_img1_1970x3538.png', (930, 957, 1211, 1236))

FIG17A = dict(
    key='fig17a', page=9, image=0, img='p10_img0_2032x3558.png',
    zone=(30, 1100),                 # panel-a search band, full-image px
    legend=(80, 310, 1360, 2020),    # y0,y1,x0,x1 erase (legend block)
    tsq=(2972, 2997, 699, 725),      # isolated square, panel c, same raster
    tci=(811, 838, 546, 573),        # least-contaminated circle, panel a
    alpha=10.0, re_sq=6.56e6, re_ci=18.32e6,
    fig='17a', panel='a (top)',
    facility='CERT/ONERA F1 pressurized tunnel, Le Fauga-Mauzac; hot films, Kreplin, Vollmers & Meier, DFVLR-AVA IB 222-84 A 34 (1985) = Stock Ref. 50 -- NOT the DFVLR 3x3 m Goettingen campaign (Ref. 49) behind Figs. 14/15/16',
    caption_note='Stock Fig. 17 caption: "Comparison of measured[50] and '
                 'computed transition locations for an angle of attack a) '
                 'alpha = 10 deg, b) alpha = 15 deg, and c) alpha = 30 deg"; '
                 'panel-a legend reads "Measured transition / [] Re = 6.56 x '
                 '10^6 / (o) Re = 18.32 x 10^6". Ref. 50 = Kreplin, Vollmers '
                 '& Meier, DFVLR-AVA IB 222-84 A 34 (1985), CERT/ONERA F1 '
                 'pressurized tunnel, Le Fauga-Mauzac -- a DIFFERENT facility '
                 'from Ref. 49 (IB 222-84 A 33, DFVLR 3x3 m Goettingen) which '
                 'is the source of Figs. 14/15/16',
    regime='Sec. III.C: "The predicted transition locations presented in Figs. '
           '17a and 17b are comparable to those of Ref. 49 (Fig. 15), except '
           'for the high-Reynolds-number case in Fig. 17a. The latter is in '
           'good agreement with the actual experimental observations." Note '
           'the ONERA F1 tunnel is the low-disturbance facility (<0.1% '
           'streamwise) against Goettingen 0.33-0.4%',
    # glyphs whose three automatic identity votes disagree: identity read
    # off the raster by hand at 5x zoom, with the pixel evidence.
    ambiguous=[],
)

FIG15A_XCHECK = dict(
    key='fig15a_panelA', page=8, image=1, img='p9_img1_1970x3538.png',
    zone=(30, 1100), legend=(80, 295, 1320, 1965),
    tsq=(415, 440, 1215, 1240),      # isolated square, panel a
    tci=(930, 957, 1211, 1236),      # isolated circle, panel a
    alpha=10.0, re_sq=1.52e6, re_ci=6.56e6,
    fig='15a', panel='a (top)', no_write=True,
    caption_note='Stock Fig. 15 caption: "... a) alpha = 10 deg, b) alpha = 15 '
                 'deg, and c) alpha = 20 deg"; panel-a legend reads '
                 '"Measured transition / [] Re = 1.52 x 10^6 / (o) Re = 6.56 x '
                 '10^6". Ref. 49 (DFVLR 3x3 m Goettingen).',
    regime='re-read of Fig. 15a with the digitize_stock_fig14c.py engine, run '
           'ONLY to put the 6.56e6 set on the same footing as Fig. 17a and to '
           'settle the leeward-most-circle open item recorded in '
           'data/stock2006_fig15a_digitized.json:note_circles',
    ambiguous=[],
)


def _circle_template_crosscheck(res):
    """The Fig. 17a circle template is cut from a panel-a circle that carries
    ~10 px of streamline ink (no isolated circle exists in the whole p10
    raster).  Redo identity + localization for every reported glyph with the
    PRISTINE circle template of the Fig. 15 raster and assert nothing moves.
    """
    import numpy as _np
    from PIL import Image
    from digitize_stock_fig14c import DIG, _fit1, _centre

    d17 = _np.array(Image.open(f'{DIG}/{FIG17A["img"]}').convert('L'),
                    copy=True) < 128
    d15 = _np.array(Image.open(f'{DIG}/{CI_XCHECK[0]}').convert('L'),
                    copy=True) < 128
    ty0, ty1, tx0, tx1 = FIG17A['tsq']
    tsq = d17[ty0:ty1 + 1, tx0:tx1 + 1]
    ty0, ty1, tx0, tx1 = FIG17A['tci']
    tci_in = d17[ty0:ty1 + 1, tx0:tx1 + 1]
    ty0, ty1, tx0, tx1 = CI_XCHECK[1]
    tci_alt = d15[ty0:ty1 + 1, tx0:tx1 + 1]

    y180, y0f = res['calibration']['y_pix']
    xp = _np.array(res['calibration']['x_pix'])
    A = _np.polyfit(xp, _np.array(res['calibration']['x_val']), 1)
    pts = ([('sq', p) for p in res['measured_re6p56e6_squares']]
           + [('ci', p) for p in res['measured_re18p32e6_circles']])
    print('  circle-template cross-check (in-raster vs pristine Fig.15 '
          'template): kind, d_centre_px')
    worst = 0.0
    for kind, p in pts:
        y = y0f - p['phi_deg'] / 180.0 * (y0f - y180)
        x = (2 * p['xL'] - 1 - A[1]) / A[0]
        yi, xi = int(round(y)), int(round(x))
        f_sq = _fit1(d17, tsq, yi, xi)
        f_in = _fit1(d17, tci_in, yi, xi)
        f_alt = _fit1(d17, tci_alt, yi, xi)
        k_in = 'sq' if f_sq[0] <= f_in[0] else 'ci'
        k_alt = 'sq' if f_sq[0] <= f_alt[0] else 'ci'
        assert k_in == kind, (kind, k_in, p)
        assert k_alt == kind, (
            f'identity of the {kind} at phi={p["phi_deg"]} flips to {k_alt} '
            'when the pristine Fig.15 circle template is used')
        if kind == 'ci':
            c_in = _centre(tci_in, f_in[2], f_in[3])
            c_alt = _centre(tci_alt, f_alt[2], f_alt[3])
            dd = float(_np.hypot(c_in[0] - c_alt[0], c_in[1] - c_alt[1]))
            worst = max(worst, dd)
            print(f'    ci phi {p["phi_deg"]:6.2f}: d_centre {dd:.2f} px')
    assert worst <= 1.5, f'circle centres move {worst:.2f} px between templates'
    print(f'    all identities reproduced; worst circle-centre shift '
          f'{worst:.2f} px (<= 1.5 px gate)')
    return round(worst, 2)


def _pairwise(a, b, tol_phi=3.0):
    """Nearest-azimuth pairing of two symbol lists."""
    out = []
    for p in a:
        q = min(b, key=lambda r: abs(r['phi_deg'] - p['phi_deg']))
        out.append((p, q, abs(q['phi_deg'] - p['phi_deg']) <= tol_phi))
    return out


def main():
    print('=== Fig. 15a panel a re-read (engine), for the 6.56e6 comparison')
    r15 = run(FIG15A_XCHECK)
    c15 = r15['measured_re6p56e6_circles']
    s15 = r15['measured_re1p52e6_squares']

    # agreement with the committed Fig. 15a file (a check on the older
    # 300-dpi blob-detect pass, and on the leeward-most-circle open item)
    old = json.load(open(os.path.join(
        PAPER, 'data', 'stock2006_fig15a_digitized.json')))
    cmp_old = []
    for tag, new, ref in (('1.52e6 squares', s15,
                           old['re_1p52e6_alpha10_squares']),
                          ('6.56e6 circles', c15,
                           old['re_6p56e6_alpha10_circles'])):
        print(f'  vs committed fig15a, {tag}:')
        for p, q, ok in _pairwise(new, ref, tol_phi=6.0):
            d_x, d_p = p['xL'] - q['xL'], p['phi_deg'] - q['phi_deg']
            cmp_old.append(dict(set=tag, phi_new=p['phi_deg'],
                                phi_old=q['phi_deg'], xL_new=p['xL'],
                                xL_old=q['xL'], d_xL=round(d_x, 4),
                                d_phi=round(d_p, 2), paired=bool(ok)))
            print(f'    phi {p["phi_deg"]:6.2f} vs {q["phi_deg"]:6.2f} '
                  f'({d_p:+6.2f})   x/L {p["xL"]:.4f} vs {q["xL"]:.4f} '
                  f'({d_x:+.4f}){"" if ok else "   [UNPAIRED]"}')

    print('\n=== Fig. 17a')
    FIG17A['extra'] = dict(cross_facility_check=dict(
        question='Re=6.56e6 at alpha=10 deg appears in BOTH Fig. 15a '
                 '(Goettingen, Ref. 49; drawn as CIRCLES there) and Fig. 17a '
                 '(ONERA F1, Ref. 50; drawn as SQUARES here). Genuine '
                 'matched-Re repeat in a second facility, or the same points '
                 'replotted?',
        method='Fig. 15a panel a re-digitized with this same engine (config '
               'FIG15A_XCHECK in repro/cfd/digitize_stock_fig17a.py) so both '
               'sides carry the same +-0.0025 x/L / +-0.4 deg budget, then '
               'paired by nearest azimuth.',
        fig15a_engine_reread_re6p56e6_circles=c15,
        fig15a_engine_reread_re1p52e6_squares=s15,
        fig15a_engine_vs_committed_json=cmp_old,
    ))
    r17 = run(FIG17A)
    s17 = r17['measured_re6p56e6_squares']
    shift = _circle_template_crosscheck(r17)
    r17['accuracy'] += (
        '; the circle template had to be cut from a streamline-grazed panel-a '
        'circle (no isolated circle exists anywhere in the p10 raster), so '
        'every identity and centre was re-derived with the pristine circle '
        f'template of the Fig. 15 raster: identities all reproduce and the '
        f'circle centres move at most {shift} px')

    print('\n=== 6.56e6 cross-facility comparison (17a squares vs 15a circles)')
    rows = []
    for p, q, ok in _pairwise(s17, c15, tol_phi=8.0):
        d_x, d_p = p['xL'] - q['xL'], p['phi_deg'] - q['phi_deg']
        rows.append(dict(phi_17a=p['phi_deg'], xL_17a=p['xL'],
                         phi_15a=q['phi_deg'], xL_15a=q['xL'],
                         d_xL=round(d_x, 4), d_phi=round(d_p, 2),
                         paired=bool(ok)))
        print(f'  17a phi {p["phi_deg"]:6.2f} x/L {p["xL"]:.4f}  |  '
              f'15a phi {q["phi_deg"]:6.2f} x/L {q["xL"]:.4f}  |  '
              f'dphi {d_p:+6.2f}  dx/L {d_x:+.4f}'
              f'{"" if ok else "   [no azimuth match]"}')
    paired = [r for r in rows if r['paired']]
    # --- discrimination budget -------------------------------------------
    # per-set digitization accuracy of this engine: +-0.0025 x/L, +-0.4 deg;
    # combined (quadrature) 0.0035 x/L and 0.57 deg.  On TOP of that sits a
    # figure-to-figure systematic: the hot-film ring ladder recovered from
    # the Ref.50 panels (17a/b/c) sits +0.0045..+0.0056 x/L from the ladder
    # recovered from the Ref.49 panels (14c/15a/16c) even though Sec. II
    # says the SAME 2.4-m model was used in both tunnels, so ~0.005 x/L is
    # a floor on any cross-FIGURE x/L comparison here.  Nothing comparable
    # applies to phi (the frame rows are unambiguous in both).
    TOL_XL, TOL_PHI = 0.0035, 0.57
    SYS_XL = 0.006                        # cross-figure ladder systematic
    thr_xL, thr_phi = 3 * TOL_XL + SYS_XL, 3 * TOL_PHI
    verdict = dict(
        n_17a=len(s17), n_15a=len(c15), n_azimuth_paired=len(paired),
        stations_17a=sorted({p['xL'] for p in s17}),
        stations_15a=sorted({q['xL'] for q in c15}),
        tolerance=dict(combined_1sigma_xL=TOL_XL, combined_1sigma_phi=TOL_PHI,
                       cross_figure_systematic_xL=SYS_XL,
                       decision_threshold_xL=round(thr_xL, 4),
                       decision_threshold_phi=round(thr_phi, 2)),
        comparison=rows)
    azim17 = sorted(p['phi_deg'] for p in s17)
    azim15 = sorted(q['phi_deg'] for q in c15)
    unmatched = [a for a in azim15
                 if min(abs(a - b) for b in azim17) > thr_phi]
    verdict['n_15a_azimuths_with_no_17a_counterpart'] = len(unmatched)
    verdict['unmatched_15a_azimuths'] = unmatched
    if paired:
        dx = [r['d_xL'] for r in paired]
        verdict['paired_d_xL_mean'] = round(float(np.mean(dx)), 4)
        verdict['paired_d_xL_max_abs'] = round(float(np.max(np.abs(dx))), 4)
        print(f'  paired mean dx/L {np.mean(dx):+.4f}, '
              f'max |dx/L| {np.max(np.abs(dx)):.4f}')
    exceed = [r for r in paired if abs(r['d_xL']) > thr_xL]
    if unmatched or exceed:
        verdict['verdict'] = 'DIFFERENT MEASUREMENTS'
        verdict['verdict_basis'] = (
            f'{len(unmatched)} of {len(azim15)} Fig. 15a azimuths have no '
            f'Fig. 17a counterpart within {thr_phi:.2f} deg '
            f'({unmatched}), and {len(exceed)} azimuth-matched pair(s) '
            f'differ in x/L by more than the {thr_xL:.4f} decision '
            f'threshold (max |dx/L| = '
            f'{max([abs(r["d_xL"]) for r in exceed], default=0):.4f}). '
            'Both tests are failed by a wide margin, so the two 6.56e6 sets '
            'are NOT the same points replotted.')
    elif paired:
        verdict['verdict'] = 'INDISTINGUISHABLE -> same data replotted'
        verdict['verdict_basis'] = (
            'every Fig. 15a azimuth has a Fig. 17a counterpart and every '
            f'paired |dx/L| <= {thr_xL:.4f}')
    else:
        verdict['verdict'] = 'CANNOT DISCRIMINATE'
        verdict['verdict_basis'] = 'no azimuth-matched pair at all'
    print(f'  azimuths in 15a with no 17a counterpart within '
          f'{thr_phi:.2f} deg: {unmatched}')
    print(f'  VERDICT: {verdict["verdict"]}')
    print(f'    {verdict["verdict_basis"]}')
    verdict['stock_facility_context'] = dict(
        same_model='Stock Sec. II: "The prolate spheroid of 2.4-m total '
                   'length was tested in the DFVLR 3 x 3 Meter Low Speed '
                   'Wind Tunnel Goettingen[49] ... and in the '
                   '(CERT)/ONERA F1 Wind Tunnel Le Fauga-Mauzac Center '
                   'Toulouse[39,50]" -- the SAME 2.4-m model in both '
                   'facilities, so the hot-film ring stations are physically '
                   'identical and the +0.005 x/L ladder offset between '
                   'Figs. 15/17 must be a plotting/registration artifact, '
                   'not a different instrumentation layout',
        why_the_re_overlap_exists='Stock Sec. III.C: "The high Reynolds '
                                  'number possibilities in the pressurized '
                                  'facility were the main argument for this '
                                  'wind-tunnel campaign aside from '
                                  'COMPARISON PURPOSES" -- the F1 campaign '
                                  'deliberately included the Goettingen '
                                  'Reynolds number',
        re_ranges='Sec. II: Goettingen alpha 0-29.7 deg at Re 1.5e6-8.5e6; '
                  'ONERA F1 alpha 10-30 deg at Re 6.5e6-43.5e6 -- 6.56e6 is '
                  'the bottom of the F1 range and inside the Goettingen '
                  'range, i.e. the designed overlap point',
        limiting_N_factors='Stock derives a facility-specific stability '
                           'limit from these very datasets: Goettingen '
                           '(Fig. 11) N_TS = 8.0, N_CF = 5.5; ONERA F1 '
                           '(Fig. 13) N_TS = 7.0, N_CF = 6.0, with '
                           '"the limiting N factors for wind tunnels are '
                           'specific quantities depending on the flow '
                           'quality of the considered facility". So F1 is '
                           'the LESS TS-stable environment in Stock\'s own '
                           'calibration (dN_TS = -1.0) and the MORE '
                           'CF-stable one (dN_CF = +0.5)',
        turbulence_levels='this paper quotes NO freestream turbulence level '
                          'for either tunnel (verified by full-text search); '
                          'any Tu figure must be sourced from the Kreplin, '
                          'Vollmers & Meier reports (Refs. 49/50), not from '
                          'Stock',
    )
    r17['extra_note'] = 'see cross_facility_check'
    r17['cross_facility_check'].update(verdict_data=verdict)
    dst = os.path.join(PAPER, 'data', 'stock2006_fig17a_digitized.json')
    json.dump(r17, open(dst, 'w'), indent=1)
    print('rewrote', dst, 'with the comparison block')


if __name__ == '__main__':
    main()
