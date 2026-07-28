"""fig:dragcrisiscd -> paper/figs/dragcrisis_cd_re.pdf.

The drag-crisis steady matrix, macroscopic view: Cd(Re_D) for the three
freestream seeds chi_inf(Tu) (color per Tu, CVD-validated trio), up-ladder
solid/filled vs dn-ladder dashed/open. Cases whose convergence monitor
flagged a steady limit cycle carry a capped vertical band spanning the
tail-window Cd min--max (the band, not a legend entry, is the limit-cycle
marking; described in the caption). Cold two-stage reference cases are
loaded into the JSON dump but NOT plotted (user directive 2026-07-28).

Literature overlay (light, small, background): independently digitized
Cd(Re) datasets from repro/cfd/litdata/dragcrisis/ (see README.md there
and digitize_dragcrisis_lit.py; every dataset has a calibration +
check-PNG audit trail, tables are transcribed + text-asserted). STRICT
SEMANTIC (user directive 2026-07-28):
  EXPERIMENTS   = gray/black symbols WITHOUT connecting lines (digitized
                  faired curves -- Wieselsberger TN-84, Achenbach 1968 --
                  render as sparse symbols resampled along the polyline)
  COMPUTATIONS  = lines WITHOUT symbols; green = transition-resolving/
                  scale-resolving (2-D unsteady Henderson 1995 + Qu 2013
                  in the shedding band, WRLES/WMLES + SST gamma-Re_theta
                  at the crisis; the single-Re 3-D DNS of Dong &
                  Karniadakis 2005 is a SHORT HORIZONTAL LINE STUB at
                  Re 1e4 -- a few % of a decade wide, at the finest-grid
                  value -- so even a single point reads as a line, not a
                  symbol); charcoal dotted = fully-turbulent SST
  SA-AI family  = the subject: LINE-ONLY (no markers), Tu-colored
                  up=solid / dn=dashed ladders -- the only COLORED lines,
                  distinct from the green/charcoal literature lines and
                  the gray/black experiment symbols
Every series (symbols AND lines, literature AND SA-AI) is drawn slightly
transparent (crowded figure): SA-AI at ALPHA_SAAI (top of the band, stays
readable as the subject), literature at ALPHA_LIT. The limit-cycle bands
keep their own (opaque) styling.
Only ONE chromatic hue is added (#3f8a4f, validated against the Tu trio:
adjacent normal-vision dE 18; the orange<->green protan pair sits in the
6-8 secondary-encoding band, carried by line weight + style).
The 2-D unsteady lines cover Re ~50-1000, where the experiments are
shedding means and OUR steady branch knowingly sits below (caption's
continuity-not-validation statement).

Data: matrix_summary.jsonl of the 84-case campaign
(repro/cfd/run_dragcrisis_matrix.py); the band extents are recomputed here
from each flagged case's final-stage force history over the same tail
window (min(3000, len) rows) the campaign medians use, and dumped with the
plotted medians into data/dragcrisis_cd_re_computed.json (the appendix
table reads that dump + the jsonl copy in data/).

Run from anywhere: python3 repro/cfd/regen_dragcrisis_cd_re.py
  [--root /local_data/qiqi/sa-ai/dragcrisis_matrix] [--no-lit]
  [--xmin RE] [--xmax RE] [--logy]
"""
import argparse
import csv
import json
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, '..', '..', 'figs'))
DATA = os.path.abspath(os.path.join(HERE, '..', '..', 'data'))
PREV = os.path.join(HERE, 'figs_explore')
LIT = os.path.join(HERE, 'litdata', 'dragcrisis')

# ordered identity colors per Tu level (validated: dataviz six-checks trio)
TU_COLOR = {"0.05": "#2f6fa8", "0.2": "#c95f2b", "0.7": "#6d55a3"}
TU_ORDER = ("0.05", "0.2", "0.7")
GREEN = '#3f8a4f'      # the single added literature-CFD hue (validated)
CHARCOAL = '#3f3f3f'   # fully-turbulent RANS (neutral by design)
GRAY = '0.42'          # experiments
# transparency (user directive 2026-07-28): the figure is crowded, so every
# series is slightly transparent. SA-AI (the subject) sits at the top of the
# requested band so it stays readable; the literature sits just below it.
ALPHA_SAAI = 0.75
ALPHA_LIT = 0.62


def load_rows(root, exclude_highre=False, mesh_key=False,
              include_ultra=False):
    """later rows win per (Tu,dir,Re) key; with exclude_highre the
    extension campaign's '_highre' warm-start anchor re-runs are NOT
    allowed to displace the published matrix rows (they re-use the
    same keys, e.g. cyl_Re2000000_Tu0.2_dn_highre). With mesh_key
    (--re-window full) the mesh family joins the key, so the seam-overlap
    duplicates (same Re+Tu+dir on two meshes: Re 300/1e3 lowre+pilot,
    2e6/4e6 pilot+highre) COEXIST instead of displacing each other; they
    plot as overlapping markers whose offset is the measured seam delta.

    'ultra'-mesh rows (the RUNNING Re>2e7 ultra campaign, appended to
    the live jsonl from 2026-07-28 on) are DROPPED unless include_ultra:
    regenerating from an in-progress campaign is forbidden (HANDOVER
    rule 4). Returns (rows, n_ultra_dropped)."""
    rows, n_ultra = {}, 0
    with open(os.path.join(root, "matrix_summary.jsonl")) as f:
        for ln in f:
            r = json.loads(ln)
            r.setdefault("mesh", "pilot")
            if r["mesh"] == "ultra" and not include_ultra:
                n_ultra += 1
                continue
            k = (r["Tu"], r["dir"], r["re"]) + \
                ((r["mesh"],) if mesh_key else ())
            if exclude_highre and "_highre" in r.get("case", "") and \
                    k in rows and "_highre" not in rows[k]["case"]:
                continue
            rows[k] = r
    return rows, n_ultra


def tail_minmax(case_dir):
    """Final-stage Cd tail min/max, same windowing as the campaign
    extractor (run_dragcrisis_matrix.extract)."""
    ps, cd = [], []
    with open(os.path.join(case_dir, "total_forces_v2.csv")) as f:
        for row in csv.reader(f):
            try:
                ps.append(int(float(row[1])))
                cd.append(float(row[3]))
            except (ValueError, IndexError):
                continue
    k = len(ps) - 1
    while k > 0 and ps[k - 1] <= ps[k]:
        k -= 1
    w = np.array(cd[k:])
    w = w[-min(len(w), 3000):]
    return float(w.min()), float(w.max())


def lit(name, *keys, where=None):
    """(Re[], Cd[]) arrays from a litdata JSON series; [] if absent.
    where: optional point-dict predicate (e.g. branch selection)."""
    path = os.path.join(LIT, name + '.json')
    if not os.path.exists(path):
        print(f'  [lit] MISSING {path} -- run digitize_dragcrisis_lit.py')
        return np.array([]), np.array([])
    d = json.load(open(path))
    for k in keys:
        d = d[k]
    pts = d['points']
    if where is not None:
        pts = [p for p in pts if where(p)]
    return (np.array([p['Re'] for p in pts]),
            np.array([p['Cd'] for p in pts]))


def logsample(re_, cd, step):
    """Sparse symbols along a digitized experimental faired curve:
    resample the polyline at ~step decades in Re (interp in log-log =
    on the chords as drawn on the log-log axes), keeping both endpoints
    and the polyline's own Cd-minimum vertex (the crisis dip must not
    be interpolated away)."""
    o = np.argsort(re_)
    lr, lc = np.log10(re_[o]), np.log10(cd[o])
    s = np.arange(lr[0], lr[-1], step)
    s = np.unique(np.concatenate([s, [lr[-1], lr[np.argmin(lc)]]]))
    return 10.0 ** s, 10.0 ** np.interp(s, lr, lc)


def overlay_literature(ax):
    """Background literature marks; returns legend handles (compact,
    one per source/class). STRICT SEMANTIC (user directive 2026-07-28):
    experiments = symbols WITHOUT connecting lines (digitized faired
    curves render as sparse symbols along the polyline); computations =
    lines WITHOUT symbols (the single-Re 3-D DNS renders as a capped
    vertical tick spanning its resolution study -- a line, not a
    symbol). The SA-AI family keeps its own distinct styling."""
    z = 1.5   # everything behind our family (zorder 3/4)
    hs = []
    mk = dict(lw=0, mew=0.8, alpha=ALPHA_LIT, zorder=z, ls='none')
    ln = dict(alpha=ALPHA_LIT, zorder=z)

    # --- experiments: gray/black SYMBOLS ONLY -------------------------
    # TN-84 faired curve (Re 4.2-3e3): sparse symbols along the
    # digitized polyline, same glyph as the TN-84 symbol series
    re_, cd = lit('tn84_wieselsberger', 'wieselsberger_curve')
    rs, cs = logsample(re_, cd, 0.10)
    ax.plot(rs, cs, marker='o', ms=2.6, mfc='0.55', mec='0.55', **mk)
    re2, cd2 = lit('tn84_wieselsberger', 'wieselsberger_symbols')
    ax.plot(re2, cd2, marker='o', ms=2.6, mfc='0.55', mec='0.55', **mk)
    hs.append(Line2D([], [], ls='none', marker='o', ms=2.6,
                     mfc='0.55', mec='0.55',
                     label='Wieselsberger 1921'))

    re_, cd = lit('tn3038_delany_sorensen', 'delany_sorensen')
    ax.plot(re_, cd, marker='^', ms=3.2, mfc='none', mec=GRAY, **mk)
    hs.append(Line2D([], [], ls='none', marker='^', ms=3.6, mfc='none',
                     mec=GRAY, mew=0.8, label='Delany–Sorensen 1953'))

    re_, cd = lit('roshko1961', 'roshko')
    ax.plot(re_, cd, marker='s', ms=3.2, mfc='none', mec=GRAY, **mk)
    hs.append(Line2D([], [], ls='none', marker='s', ms=3.6, mfc='none',
                     mec=GRAY, mew=0.8, label='Roshko 1961'))

    re_, cd = lit('rodriguez2015_fig4_exp', 'schewe1983')
    ax.plot(re_, cd, marker='+', ms=4.2, mec='0.25', mew=0.9,
            lw=0, alpha=ALPHA_LIT, zorder=z, ls='none')
    hs.append(Line2D([], [], ls='none', marker='+', ms=4.6, mec='0.25',
                     mew=0.9, label='Schewe 1983 [dig. R15]'))

    re_, cd = lit('rodriguez2015_fig4_exp', 'achenbach_heinecke1981')
    ax.plot(re_, cd, marker='d', ms=3.0, mfc='none', mec=GRAY, **mk)
    hs.append(Line2D([], [], ls='none', marker='d', ms=3.4, mfc='none',
                     mec=GRAY, mew=0.8,
                     label='Achenbach–Heinecke 1981 [R15]'))

    # Achenbach-1968 faired curve (an EXPERIMENT: sparse symbols along
    # the digitized polyline, not a line that would read as computation)
    re_, cd = lit('catalano2001_wmles', 'achenbach1968_curve')
    rs, cs = logsample(re_, cd, 0.08)
    ax.plot(rs, cs, marker='o', ms=3.0, mfc='none', mec=GRAY, **mk)
    hs.append(Line2D([], [], ls='none', marker='o', ms=3.4, mfc='none',
                     mec=GRAY, mew=0.8,
                     label='Achenbach 1968 [dig. C03]'))

    # low-Re experiments (Re 0.05-6), symbols as always
    for nm in ('tritton1959', 'finn1953', 'jayaweera_mason1965'):
        re_, cd = lit('veysey_fig7_lowre', nm)
        ax.plot(re_, cd, marker='o', ms=2.4, mfc='0.55', mec='0.55', **mk)

    # --- computations: LINES ONLY --------------------------------------
    # 2-D unsteady band (Re ~50-1000): shedding-mean drag, where the
    # experiments are shedding means and OUR steady branch sits below
    re_, cd = lit('henderson1995', 'totals',
                  where=lambda p: p['branch'] == 'shedding')
    ax.plot(re_, cd, '-', color=GREEN, lw=1.1, **ln)
    hs.append(Line2D([], [], color=GREEN, lw=1.1,
                     label='2-D spectral, shedding mean (Henderson 1995)'))

    re_, cd = lit('qu2013_dns', 'dns2d')
    ax.plot(re_, cd, '--', color=GREEN, lw=1.1, **ln)
    hs.append(Line2D([], [], color=GREEN, lw=1.1, ls='--',
                     label='2-D DNS (Qu et al. 2013)'))

    # 3-D DNS anchor at Re=1e4: a SINGLE-Re computation. Per the strict
    # lines=computations rule (user directive 2026-07-28) it is drawn as a
    # SHORT HORIZONTAL LINE STUB (a few % of a decade wide, centered on
    # Re=1e4) rather than a marker/tick -- so even a single point reads as
    # a line. The stub sits at the finest-grid DNS value (Cd_finest=1.143;
    # the Nz>=64 resolution span 1.110-1.143 is recorded in the JSON).
    path = os.path.join(LIT, 'dong_karniadakis2005_dns3d.json')
    if os.path.exists(path):
        d = json.load(open(path))['dns3d']
        cd_fin = d.get('Cd_finest', 0.5 * (d['Cd_min'] + d['Cd_max']))
        hw = 0.035   # stub half-width in decades of Re
        ax.plot([10.0 ** (4 - hw), 10.0 ** (4 + hw)], [cd_fin, cd_fin],
                '-', color=GREEN, lw=1.4, alpha=ALPHA_LIT, zorder=z)
        hs.append(Line2D([], [], color=GREEN, lw=1.4,
                         label='3-D DNS (Dong–Karniadakis 2005)'))
    else:
        print(f'  [lit] MISSING {path} -- run digitize_dragcrisis_lit.py')

    # crisis-range scale-resolving
    re_, cd = lit('rodriguez2015_les', 'les')
    ax.plot(re_, cd, '-', color=GREEN, lw=1.8, **ln)
    hs.append(Line2D([], [], color=GREEN, lw=1.8,
                     label='WRLES (Rodríguez 2015)'))

    re_, cd = lit('catalano2001_wmles', 'wmles')
    ax.plot(re_, cd, '--', color=GREEN, lw=1.8, **ln)
    hs.append(Line2D([], [], color=GREEN, lw=1.8, ls='--',
                     label='WMLES (Catalano 2003)'))

    # transition-model RANS
    re_, cd = lit('iop2020_models', 'sst_gamma_retheta')
    ax.plot(re_, cd, '-.', color=GREEN, lw=1.1, **ln)
    hs.append(Line2D([], [], color=GREEN, lw=1.1, ls='-.',
                     label='SST $\\gamma$–$Re_\\theta$ URANS [SG20]'))

    # fully-turbulent RANS: charcoal dotted (3 series, one class entry)
    re_, cd = lit('iop2020_models', 'sst_fully_turbulent')
    ax.plot(re_, cd, ':', color=CHARCOAL, lw=1.1, **ln)
    for series in ('cfx', 'openfoam'):
        re_, cd = lit('stringer2014_urans', series)
        ax.plot(re_, cd, ':', color=CHARCOAL, lw=1.1, **ln)
    hs.append(Line2D([], [], color=CHARCOAL, lw=1.1, ls=':',
                     label='fully-turb. SST [SG20; S14]'))
    return hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/local_data/qiqi/sa-ai/dragcrisis_matrix")
    ap.add_argument("--no-lit", action="store_true",
                    help="suppress the literature overlay")
    ap.add_argument("--xmin", type=float, default=None)
    ap.add_argument("--xmax", type=float, default=None)
    ap.add_argument("--logy", action="store_true",
                    help="log Cd axis (for the future Re 1-1e7 span)")
    ap.add_argument("--re-window", default="6e4:2e6",
                    help="Re window of COMPLETED campaign rows to plot "
                         "('full' once the Re 1-1e7 extension campaign "
                         "has finished; default = the published 84-case "
                         "matrix -- the live jsonl already carries "
                         "in-progress extension rows, and regenerating "
                         "from a running campaign is forbidden, "
                         "HANDOVER rule 4)")
    ap.add_argument("--include-ultra", action="store_true",
                    help="admit 'ultra'-mesh rows (the Re>2e7 ultra "
                         "campaign). FORBIDDEN while that campaign is "
                         "still running (HANDOVER rule 4); default "
                         "drops them and leaves the committed jsonl "
                         "copy untouched")
    args = ap.parse_args()
    rows, n_ultra = load_rows(args.root,
                              exclude_highre=args.re_window != 'full',
                              mesh_key=args.re_window == 'full',
                              include_ultra=args.include_ultra)
    if n_ultra:
        print(f'  NOTE: {n_ultra} ultra-campaign rows dropped '
              f'(in progress; rerun with --include-ultra when the '
              f'ultra campaign is declared harvested)')
    if args.re_window != 'full':
        lo, hi = (float(v) for v in args.re_window.split(':'))
        n0 = len(rows)
        rows = {k: v for k, v in rows.items() if lo <= k[2] <= hi}
        if len(rows) != n0:
            print(f'  NOTE: {n0 - len(rows)} campaign rows outside '
                  f'Re [{lo:g},{hi:g}] excluded (extension in progress; '
                  f'rerun with --re-window full when it completes)')
    # keep the committed copy of the campaign summary in sync (small file;
    # the case trees themselves stay on /local_data) -- but NOT while the
    # live file carries in-progress rows we are excluding
    if args.re_window == 'full' and n_ultra == 0:
        shutil.copy2(os.path.join(args.root, "matrix_summary.jsonl"),
                     os.path.join(DATA, "dragcrisis_matrix_summary.jsonl"))
    else:
        print('  NOTE: data/dragcrisis_matrix_summary.jsonl left '
              'untouched (live campaign file has extra rows)')

    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13})
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    dump = {}
    remin, remax = np.inf, 0.0
    for tu in TU_ORDER:
        c = TU_COLOR[tu]
        # SA-AI family is LINE-ONLY (user directive 2026-07-28): no markers,
        # up=solid / dn=dashed, seed color per Tu. It stays visually distinct
        # from the literature by being the only COLORED lines (literature
        # computations are green/charcoal). Cold two-stage reference cases:
        # kept in the JSON dump for the appendix table, NOT plotted.
        for d, ls in (("up", "-"), ("dn", "--"), ("cold", None)):
            pts = sorted(((r["re"], r) for r in rows.values()
                          if r["Tu"] == tu and r["dir"] == d),
                         key=lambda p: (p[0], p[1]["Cd"]))
            if not pts:
                continue
            re = np.array([p[0] for p in pts])
            cd = np.array([p[1]["Cd"] for p in pts])
            if ls is not None:
                remin, remax = min(remin, re.min()), max(remax, re.max())
                ax.plot(re, cd, ls=ls, color=c, lw=1.6,
                        alpha=ALPHA_SAAI, zorder=4)
            for reval, r in pts:
                lc = any("limit_cycle" in v for v in r["verdicts"])
                ent = {"Cd": r["Cd"], "Cd_tail_p2p": r["Cd_tail_p2p"],
                       "limit_cycle": lc, "verdicts": r["verdicts"]}
                if lc:
                    lo, hi = tail_minmax(os.path.join(args.root, r["case"]))
                    ent["Cd_tail_min"], ent["Cd_tail_max"] = lo, hi
                    if ls is not None:
                        ax.errorbar([reval], [r["Cd"]],
                                    yerr=[[r["Cd"] - lo], [hi - r["Cd"]]],
                                    fmt='none', ecolor=c, elinewidth=1.4,
                                    capsize=3.5, capthick=1.4, zorder=3)
                dump[r["case"]] = ent

    # FT-SA control (this work, AI_SA=0, chi_inf=3): a fully-turbulent SA
    # computation LINE (fault-attribution for the missing transcritical
    # rise, record 2026-07-28 ultra). Charcoal + dot markers to sit with the
    # other fully-turbulent computations without colliding with the
    # digitized Stringer/SG20 charcoal-dotted line.
    ft = sorted(((r["re"], r["Cd"]) for r in rows.values()
                 if r.get("Tu") == "ft"), key=lambda p: p[0])
    if ft:
        fre = np.array([p[0] for p in ft])
        fcd = np.array([p[1] for p in ft])
        ax.plot(fre, fcd, ls='-.', color=CHARCOAL, lw=1.5, marker='D',
                ms=4.0, mfc=CHARCOAL, alpha=0.9, zorder=4)
        for reval, r in ((r["re"], r) for r in rows.values()
                         if r.get("Tu") == "ft"):
            dump[r["case"]] = {"Cd": r["Cd"],
                               "Cd_tail_p2p": r["Cd_tail_p2p"],
                               "limit_cycle": any("limit_cycle" in v
                                                  for v in r["verdicts"]),
                               "verdicts": r["verdicts"]}

    lit_handles = [] if args.no_lit else overlay_literature(ax)

    ax.set_xscale("log")
    # window: literature context around OUR data; grows with the campaign
    xmin = args.xmin if args.xmin else min(1.0e4, 0.8 * remin)
    xmax = args.xmax if args.xmax else max(1.2e7, 1.3 * remax)
    ax.set_xlim(xmin, xmax)
    if args.logy:
        ax.set_yscale('log')
        cds = [r["Cd"] for r in rows.values()]
        ax.set_ylim(0.75 * min(cds), 1.35 * max(cds))
    else:
        ax.set_ylim(0.1, 1.52)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.xaxis.set_major_locator(mticker.LogLocator(numticks=12))
    ax.xaxis.set_minor_locator(
        mticker.LogLocator(subs=(2, 3, 5), numticks=12))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel(r'$Re_D$')
    ax.set_ylabel(r'$C_d$')
    ax.grid(alpha=0.3, which='major')

    # direct family labels where the three seeds separate
    ax.text(2.3e5, 0.50, r'$Tu\,0.7\%$', color=TU_COLOR["0.7"],
            fontsize=11, ha='right')
    ax.text(6.6e5, 0.60, r'$Tu\,0.2\%$', color=TU_COLOR["0.2"],
            fontsize=11, ha='left')
    ax.text(4.4e5, 0.78, r'$Tu\,0.05\%$', color=TU_COLOR["0.05"],
            fontsize=11, ha='left')

    ours = [
        Line2D([], [], color='0.25', ls='-', lw=1.6,
               label='SA-AI up-ladder'),
        Line2D([], [], color='0.25', ls='--', lw=1.6,
               label='SA-AI dn-ladder'),
    ]
    if ft:
        ours.append(Line2D([], [], color=CHARCOAL, ls='-.', lw=1.5,
                           marker='D', ms=4.0,
                           label='FT-SA (this work, AI off)'))
    leg1 = ax.legend(handles=ours, fontsize=9, frameon=False,
                     loc='upper right', handlelength=2.4,
                     borderaxespad=0.4)
    ax.add_artist(leg1)
    if lit_handles:
        ax.legend(handles=lit_handles, fontsize=8, frameon=False,
                  loc='lower left', ncol=1, handlelength=2.2,
                  handletextpad=0.5, borderaxespad=0.3,
                  labelspacing=0.32)
    fig.tight_layout()
    os.makedirs(PREV, exist_ok=True)
    fig.savefig(os.path.join(OUT, 'dragcrisis_cd_re.pdf'))
    fig.savefig(os.path.join(PREV, 'dragcrisis_cd_re.png'), dpi=140)
    json.dump(dump, open(os.path.join(DATA, 'dragcrisis_cd_re_computed.json'),
                         'w'), indent=1)
    n_lc = sum(1 for e in dump.values() if e["limit_cycle"])
    print(f"wrote {OUT}/dragcrisis_cd_re.pdf ({len(dump)} cases, "
          f"{n_lc} limit-cycle bands, lit={'off' if args.no_lit else 'on'})")


if __name__ == "__main__":
    main()
