"""LaTeX tables for the 2026-08 literature-comparability extension matrix.

Reads paper/data/ext2026_matrix.json (64 L2-pair solutions; see
sa-ai/scripts/ext2026/cases_ext.py for the matrix definition and
sa-ai/scripts/ext2026/harvest.py for the extraction) and writes complete table
environments into paper/tables/, following the repo convention that a table
file is \\input as a whole (a bare row body \\input inside a tabular breaks the
alignment at \\bottomrule). Paper and whitepaper get separate files -- same
numbers, different captions and label prefixes -- as tab_dragcrisis{,_wp} do.

  python3 emit_ext2026_tables.py
"""
import os, json

_H = os.path.dirname(os.path.abspath(__file__))
PD = os.path.abspath(os.path.join(_H, "..", ".."))
D = json.load(open(f"{PD}/data/ext2026_matrix.json"))["cases"]
OUT = f"{PD}/tables"
os.makedirs(OUT, exist_ok=True)


def f(v, n=4, dash="--"):
    return dash if v is None else f"{v:.{n}f}"


def atex(a):
    return f"${a:g}$" if a >= 0 else f"$-{abs(a):g}$"


def grouped(pred, key):
    rows = {}
    for n, r in D.items():
        if pred(r):
            rows.setdefault(key(r), {})[r["family"]] = r
    return [rows[k] for k in sorted(rows)]


def emit(fn, label, caption, header, colspec, rows, placement="tp"):
    body = "\n".join(rows)
    txt = (f"\\begin{{table}}[{placement}]\n  \\centering\\small\n"
           f"  \\caption{{{caption}}}\n  \\label{{{label}}}\n"
           f"  \\begin{{tabular}}{{{colspec}}}\n    \\toprule\n"
           f"{header}\n    \\midrule\n{body}\n    \\bottomrule\n"
           f"  \\end{{tabular}}\n\\end{{table}}\n")
    open(f"{OUT}/{fn}", "w").write(txt)
    print(f"wrote {OUT}/{fn}")


H4 = ("    & \\multicolumn{4}{c}{structured O-grid} & \\multicolumn{4}{c}{unstructured cavity} \\\\\n"
      "    \\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
      "    $\\alpha$ & $c_l$ & $c_d$ & $x_\\mathrm{tr}^\\mathrm{up}$ & $x_\\mathrm{tr}^\\mathrm{lo}$\n"
      "             & $c_l$ & $c_d$ & $x_\\mathrm{tr}^\\mathrm{up}$ & $x_\\mathrm{tr}^\\mathrm{lo}$ \\\\")
H4R = ("    & & \\multicolumn{4}{c}{structured O-grid} & \\multicolumn{4}{c}{unstructured cavity} \\\\\n"
       "    \\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\n"
       "    $\\alpha$ & $Re/10^6$ & $c_l$ & $c_d$ & $x_\\mathrm{tr}^\\mathrm{up}$ & $x_\\mathrm{tr}^\\mathrm{lo}$\n"
       "             & $c_l$ & $c_d$ & $x_\\mathrm{tr}^\\mathrm{up}$ & $x_\\mathrm{tr}^\\mathrm{lo}$ \\\\")
H5 = ("    & \\multicolumn{5}{c}{structured O-grid} & \\multicolumn{5}{c}{unstructured cavity} \\\\\n"
      "    \\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\n"
      "    $\\alpha$ & $c_l$ & $c_d$ & $x_\\mathrm{tr}$ & $x_\\mathrm{sep}$ & $x_\\mathrm{reatt}$\n"
      "             & $c_l$ & $c_d$ & $x_\\mathrm{tr}$ & $x_\\mathrm{sep}$ & $x_\\mathrm{reatt}$ \\\\")
WP = lambda h: h.replace("structured O-grid", "structured").replace(
    "unstructured cavity", "cavity").replace("$c_l$", "$C_L$").replace("$c_d$", "$C_D$")


def r4(g):
    s, c = g["str"], g["cav"]
    return (f"    {atex(s['alpha'])} & {f(s['CL'])} & {f(s['CD'],5)} & "
            f"{f(s['xtr_upper'],3)} & {f(s['xtr_lower'],3)} & "
            f"{f(c['CL'])} & {f(c['CD'],5)} & "
            f"{f(c['xtr_upper'],3)} & {f(c['xtr_lower'],3)} \\\\")


def r4re(g):
    s, c = g["str"], g["cav"]
    return (f"    {atex(s['alpha'])} & ${s['Re']/1e6:g}$ & {f(s['CL'])} & {f(s['CD'],5)} & "
            f"{f(s['xtr_upper'],3)} & {f(s['xtr_lower'],3)} & "
            f"{f(c['CL'])} & {f(c['CD'],5)} & "
            f"{f(c['xtr_upper'],3)} & {f(c['xtr_lower'],3)} \\\\")


def r5(g):
    s, c = g["str"], g["cav"]
    return (f"    {atex(s['alpha'])} & {f(s['CL'])} & {f(s['CD'],5)} & "
            f"{f(s['xtr_upper'],3)} & {f(s['x_sep_wide'],3)} & {f(s['x_reatt_wide'],3)} & "
            f"{f(c['CL'])} & {f(c['CD'],5)} & "
            f"{f(c['xtr_upper'],3)} & {f(c['x_sep_wide'],3)} & {f(c['x_reatt_wide'],3)} \\\\")


STA = ("$x_\\mathrm{tr}$ is the near-wall $\\chi\\!=\\!1$ front; $x_\\mathrm{sep}$ and "
       "$x_\\mathrm{reatt}$ are the signed-$C_f$ crossings, with reattachment required "
       "to lie downstream of separation; ``--'' means the upper surface carries no "
       "closed bubble ahead of the trailing edge.")
UNS = ("The structured $\\alpha\\!=\\!8^\\circ$ entry is the one unsteady solution of the "
       "matrix: its front cycles over $0.24$--$0.57\\,c$ without settling through "
       "$8\\!\\times\\!10^4$ pseudo-steps and the final value is tabulated; its cavity "
       "counterpart is steady at $0.279\\,c$.")

SETS = [
    ("nlf_alpha", lambda r: r["airfoil"] == "nlf0416" and r["block"] in
     ("P1_nlf_alpha", "S_nlf_deepneg"), lambda r: r["alpha"], r4, H4, "c cccc cccc",
     ("NLF(1)-0416 at $Re\\!=\\!4\\!\\times\\!10^6$, extended incidence set on the L2 "
      "pair: forces and near-wall $\\chi\\!=\\!1$ front stations. The five incidences "
      "$-6^\\circ$--$8^\\circ$ complete the overlap with the workshop and "
      "Piotrowski--Zingg sweeps; $-10^\\circ$ and $-12^\\circ$ are the deep-negative "
      "extension. At the four that fall inside the workshop's abscissa both computed "
      "fronts lie within the band spanned by the fourteen submittals on both surfaces."),
     "NLF(1)-0416 at $Re=4\\times10^6$, extended incidence set on the L2 pair."),
    ("nlf_resweep", lambda r: r["block"] == "P4_nlf_resweep",
     lambda r: (r["alpha"], r["Re"]), r4re, H4R, "cc cccc cccc",
     ("NLF(1)-0416 Reynolds sweep on the L2 pair at the two paper incidence anchors, "
      "$M\\!=\\!0.1$ and the paper-wide seed throughout, grids unchanged and only "
      "$\\mu_\\mathrm{ref}$ varied. These join the $Re\\!=\\!4\\!\\times\\!10^6$ "
      "benchmark of Sec.~\\ref{sec:nlfval} continuously."),
     "NLF(1)-0416 Reynolds sweep on the L2 pair at the two incidence anchors, $M=0.1$ and the paper-wide seed."),
    ("epp_a200k", lambda r: r["block"] == "P2_epp_alpha200k", lambda r: r["alpha"],
     r5, H5, "c ccccc ccccc",
     ("Eppler 387 at $Re\\!=\\!2\\!\\times\\!10^5$, the three incidences completing the "
      "$-2^\\circ$--$9^\\circ$ grid of Refs.~\\cite{shahjahan_2024, cole_mueller_1990}, "
      "L2 pair. " + STA),
     "Eppler 387 at $Re=2\\times10^5$, the three incidences completing the published $-2^\\circ$--$9^\\circ$ grid, L2 pair."),
    ("epp_re300k", lambda r: r["block"] == "P3_epp_resweep" and r["Re"] == 3.0e5,
     lambda r: r["alpha"], r5, H5, "c ccccc ccccc",
     "Eppler 387 at $Re\\!=\\!3\\!\\times\\!10^5$, incidence sweep on the L2 pair. " + STA,
     "Eppler 387 at $Re=3\\times10^5$, incidence sweep on the L2 pair. " + STA),
    ("epp_re100k", lambda r: r["block"] == "P3_epp_resweep" and r["Re"] == 1.0e5,
     lambda r: r["alpha"], r5, H5, "c ccccc ccccc",
     "Eppler 387 at $Re\\!=\\!10^5$, incidence sweep on the L2 pair, columns as in "
     "Table~\\ref{tab:extepp300k}. " + UNS,
     "Eppler 387 at $Re=10^5$, incidence sweep on the L2 pair, columns as in "
     "Table~\\ref{t:extepp300k}. " + UNS),
]

LBL = {"nlf_alpha": ("tab:extnlfalpha", "t:extnlfalpha"),
       "nlf_resweep": ("tab:extnlfresweep", "t:extnlfresweep"),
       "epp_a200k": ("tab:exteppalpha", "t:exteppa200k"),
       "epp_re300k": ("tab:extepp300k", "t:extepp300k"),
       "epp_re100k": ("tab:extepp100k", "t:extepp100k")}

for key, pred, sk, rowf, hdr, spec, cap, capwp in SETS:
    rows = [rowf(g) for g in grouped(pred, sk) if len(g) == 2]
    lp, lw = LBL[key]
    emit(f"ext2026_{key}.tex", lp, cap, hdr, spec, rows)
    emit(f"ext2026_{key}_wp.tex", lw, capwp, WP(hdr), spec, rows, placement="H")

conv = [r for r in D.values() if r["converged"]]
fam = {}
for n, r in D.items():
    fam.setdefault((r["block"], r["Re"], r["alpha"]), {})[r["family"]] = r
dx = sorted(abs(v["str"]["xtr_upper"] - v["cav"]["xtr_upper"]) for v in fam.values()
            if len(v) == 2 and v["str"]["xtr_upper"] is not None and v["cav"]["xtr_upper"] is not None)
dcd = sorted(abs(v["str"]["CD"] - v["cav"]["CD"]) for v in fam.values() if len(v) == 2)
print(f"\ncases {len(D)}, converged {len(conv)}, median |dx_tr| {dx[len(dx)//2]:.4f}, "
      f"median |dC_d| {dcd[len(dcd)//2]:.5f}, max |dx_tr| {dx[-1]:.4f}")


# --- 6. NLF Reynolds sweep against the Somers Fig. 9 brackets ---------------
# Joins the computed sweep with data/somers1981_nlf0416_transition_by_Re.json
# (repro/cfd/digitize_somers_fig9.py). The measurement resolves transition only
# to the orifice spacing, so the comparison is computed-front vs measured
# bracket at MATCHED LIFT (the bracket ends are interpolated in c_l).
SOM = json.load(open(f"{PD}/data/somers1981_nlf0416_transition_by_Re.json"))


def _brk(panel, surf, cl, key):
    import numpy as _np
    r = [z for z in SOM[panel][surf] if z["complete"]]
    c = _np.array([z["cl"] for z in r]); v = _np.array([z[key] for z in r])
    o = _np.argsort(c)
    if cl < c[o][0] or cl > c[o][-1]:
        return None
    return float(_np.interp(cl, c[o], v[o]))


rows = []
for a in (0, 4):
    for re, panel in ((1, "re_1e6"), (2, "re_2e6"), (3, "re_3e6"), (4, "re_4e6")):
        if re == 4:                                   # the Sec. V benchmark pair
            s = {"CL": 0.5139 if a == 0 else 0.9805,
                 "xtr_upper": 0.387 if a == 0 else 0.254,
                 "xtr_lower": 0.566 if a == 0 else 0.610}
            c = None
        else:
            s = D[f"nlfsweep_strL2_Re{re}M_a{a}"]
            c = D[f"nlfsweep_cavL2_Re{re}M_a{a}"]
        cell = []
        for surf, k in (("upper", "xtr_upper"), ("lower", "xtr_lower")):
            lo = _brk(panel, surf, s["CL"], "x_last_laminar")
            hi = _brk(panel, surf, s["CL"], "x_first_turbulent")
            cell.append(f(s[k], 3))
            cell.append("--" if c is None else f(c[k], 3))
            cell.append("--" if lo is None else f"${f(lo,3)}$--${f(hi,3)}$")
        tag = "$4$\\rlap{$^\\dagger$}" if re == 4 else f"${re}$"
        rows.append(f"    {atex(a)} & {tag} & {f(s['CL'])} & " + " & ".join(cell) + " \\\\")

hdr = ("    & & & \\multicolumn{3}{c}{upper surface $x_\\mathrm{tr}$} & "
       "\\multicolumn{3}{c}{lower surface $x_\\mathrm{tr}$} \\\\\n"
       "    \\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\n"
       "    $\\alpha$ & $Re/10^6$ & $c_l$ & str & cav & measured & str & cav & measured \\\\")
cap = ("NLF(1)-0416 Reynolds sweep against the Somers TP-1861 Fig.~9 transition "
       "measurements, compared at matched lift. The measurement resolves transition only "
       "to the orifice spacing, so ``measured'' is the \\emph{bracket} between the last "
       "laminar and first turbulent orifice (digitized in "
       "\\texttt{repro/cfd/digitize\\_somers\\_fig9.py}; $0.05\\,c$ wide over most of the "
       "chord). Of the $27$ comparisons the measurement covers, $20$ fall inside the "
       "bracket and every miss is within $0.03\\,c$ of a bracket end. "
       "$^\\dagger$the $Re\\!=\\!4\\!\\times\\!10^6$ rows are the Sec.~\\ref{sec:nlfval} "
       "benchmark (structured L2), shown for continuity; ``--'' on the upper surface at "
       "$\\alpha\\!=\\!4^\\circ$ because the measured upper-surface branch of Fig.~9(d) "
       "ends at $c_l\\!\\approx\\!0.68$.")
capwp = ("NLF(1)-0416 Reynolds sweep against the Somers TP-1861 Fig.~9 transition brackets "
         "at matched lift (last laminar to first turbulent orifice). $20$ of the $27$ "
         "covered comparisons fall inside the bracket. $^\\dagger$Sec.~V benchmark rows.")
emit("ext2026_nlf_somers.tex", "tab:extnlfsomers", cap, hdr, "cc c ccc ccc", rows)
emit("ext2026_nlf_somers_wp.tex", "t:extnlfsomers", capwp, hdr, "cc c ccc ccc", rows, placement="H")
