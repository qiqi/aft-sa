"""Assert the numbers that the prose of Section III derives from its own tables.

The tables in sa-ai.tex and tables/*_wp.tex are emitted by the campaign scripts.
The prose quotes differences and ratios of their entries. This script recomputes
each quoted difference from the table rows and fails if the prose and the tables
disagree.

    python3 repro/tools/check_results_prose.py
"""
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.normpath(os.path.join(HERE, "..", ".."))
NAN = float("nan")


def num(x):
    x = x.replace("$", "").replace("\\\\", "").replace("\\!", "").strip()
    try:
        return float(x)
    except ValueError:
        return NAN


def table_rows(text, label):
    """Rows of the tabular that follows \\label{label}, as lists of cell strings."""
    i = text.index("\\label{" + label + "}")
    body = text[i:text.index("\\end{tabular}", i)]
    return [[c.strip() for c in ln.split("&")] for ln in body.split("\n")
            if "&" in ln and re.match(r"\s*\$?-?[0-9]", ln)]


def between(name, value, lo, hi):
    assert lo - 1e-9 <= value <= hi + 1e-9, f"{name}: {value:.4g} not in [{lo}, {hi}]"
    print(f"  {name}: {value:.4g}")


def main():
    tex = open(os.path.join(PAPER, "sa-ai.tex")).read()

    # ---------------------------------------------------------------- flat plate
    print("flat plate")
    fp = np.array([[num(c) for c in r] for r in table_rows(tex, "t:flatplate")])
    tu, ags, c1, cv, o1 = fp[:, 0], fp[:, 1], fp[:, 2], fp[:, 3], fp[:, 4]
    between("largest |chi=cv1 / AGS - 1|", np.abs(cv / ags - 1).max(), 0.09, 0.10)
    between("chi=1 / AGS - 1 at Tu=0.04%", c1[0] / ags[0] - 1, 0.015, 0.025)
    between("chi=1 / AGS - 1 at Tu=0.6%", c1[-1] / ags[-1] - 1, -0.275, -0.26)
    between("crossings apart at Tu=0.04%", cv[0] - c1[0], 54, 54)
    between("crossings apart at Tu=0.6%", cv[-1] - c1[-1], 161, 161)
    low = 1 - o1 / c1
    between("OpenFOAM chi=1 below Flow360, min", low.min(), 0.04, 0.05)
    between("OpenFOAM chi=1 below Flow360, max", low.max(), 0.07, 0.075)
    ncrit = -8.43 - 2.4 * np.log(tu / 100)
    per_n = -np.diff(c1) / -np.diff(ncrit)
    between("Re_theta per unit N_crit, min", per_n.min(), 87.5, 88.5)
    between("Re_theta per unit N_crit, max", per_n.max(), 105.5, 106.5)

    # --------------------------------------------------------------- NLF(1)-0416
    print("NLF(1)-0416")
    rows = table_rows(tex, "t:nlf")
    exp, lev = {}, {}
    for r in rows:
        a = num(r[0])
        e = [num(c) for c in r[14:18]]
        if any(v == v for v in e):
            exp[a] = e
        lev[(a, r[1])] = ([num(c) for c in r[2:6]], [num(c) for c in r[6:10]],
                          [num(c) for c in r[10:14]])
    alphas = sorted({a for a, g in lev if g == "L2"})
    assert len(alphas) == 13 and alphas[0] == -12 and alphas[-1] == 15
    s = {a: lev[(a, "L2")][0] for a in alphas}
    c = {a: lev[(a, "L2")][1] for a in alphas}
    o = {a: lev[(a, "L2")][2] for a in alphas}
    between("upper front at -12 deg", s[-12][2], 0.945, 0.945)
    between("upper front at 15 deg", s[15][2], 0.005, 0.005)
    between("lower front at -12 deg", s[-12][3], 0.001, 0.001)
    between("lower front at 15 deg", s[15][3], 0.659, 0.659)
    fam = max(max(abs(s[a][k] - c[a][k]) for k in (2, 3)) for a in alphas)
    between("largest front difference of the families at L2", fam, 0.0215, 0.0225)
    of = [a for a in alphas if o[a][0] == o[a][0]]
    between("OpenFOAM front difference at L2",
            max(max(abs(o[a][k] - s[a][k]) for k in (2, 3)) for a in of), 0.0155, 0.0165)
    between("OpenFOAM lift-coefficient difference at L2",
            max(abs(o[a][0] - s[a][0]) for a in of), 0.016, 0.017)
    ea = sorted(exp)
    ecl = [exp[a][0] for a in ea]
    ecd = [exp[a][1] for a in ea]
    rel = [fam_[a][1] / np.interp(fam_[a][0], ecl, ecd) - 1
           for fam_ in (s, c) for a in alphas if -6 <= a <= 6]
    between("L2 drag / measured at the same lift - 1, min", min(rel), -0.03, -0.02)
    between("L2 drag / measured at the same lift - 1, max", max(rel), 0.09, 0.10)
    ratio = [lev[(a, "L0")][k][1] / lev[(a, "L2")][k][1]
             for a in alphas if (a, "L0") in lev for k in (0, 1)]
    between("L0 drag / L2 drag, min", min(ratio), 1.3, 1.35)
    between("L0 drag / L2 drag, max", max(ratio), 4.15, 4.25)
    between("computed - measured lift at -8 deg", s[-8][0] - exp[-8][0], 0.0095, 0.0105)
    between("computed - measured lift at 8 deg", s[8][0] - exp[8][0], 0.115, 0.125)
    between("computed lift at 15 deg", s[15][0], 1.99, 2.01)

    # ---------------------------------------------------------------- Eppler 387
    print("Eppler 387, Re = 2e5")
    rows = table_rows(tex, "t:epp")
    exp, s, c, o = {}, {}, {}, {}
    for r in rows:
        a = num(r[0])
        e = [num(x) for x in r[15:19]]
        if any(v == v for v in e):
            exp[a] = e
        if r[1] == "L2":
            s[a] = [num(x) for x in r[2:6]]
            c[a] = [num(x) for x in r[6:10]]
            o[a] = [num(x) for x in r[10:15]]
    oil = [a for a in sorted(exp) if -2 <= a <= 6 and exp[a][2] == exp[a][2]]
    dsep = [f[a][2] - exp[a][2] for f in (s, c) for a in oil]
    drea = [f[a][3] - exp[a][3] for f in (s, c) for a in oil]
    between("separation aft of oil flow, min", min(dsep), 0.0, 0.005)
    between("separation aft of oil flow, max", max(dsep), 0.035, 0.045)
    between("reattachment aft of oil flow, min", min(drea), 0.025, 0.035)
    between("reattachment aft of oil flow, max", max(drea), 0.065, 0.07)
    between("bubble length at 7 deg, structured", s[7][3] - s[7][2], 0.035, 0.045)
    between("bubble length at 7 deg, cavity", c[7][3] - c[7][2], 0.025, 0.035)
    between("measured bubble length at 7 deg", exp[7][3] - exp[7][2], 0.15, 0.15)
    ea = [a for a in sorted(exp) if exp[a][0] == exp[a][0]]
    ecl = [exp[a][0] for a in ea]
    ecd = [exp[a][1] for a in ea]
    rel = {a: [f[a][1] / np.interp(f[a][0], ecl, ecd) - 1 for f in (s, c)] for a in s}
    lo3 = [v for a in rel if 0 <= a <= 3 for v in rel[a]]
    hi6 = [v for a in rel if 4 <= a <= 6 for v in rel[a]]
    between("drag vs measured at the same lift, 0-3 deg, largest", max(map(abs, lo3)), 0.0, 0.03)
    between("drag vs measured at the same lift, 4-6 deg, min", min(hi6), 0.02, 0.03)
    between("drag vs measured at the same lift, 4-6 deg, max", max(hi6), 0.11, 0.12)
    between("largest drag difference of the families at L2, -2 to 7 deg",
            max(abs(s[a][1] - c[a][1]) for a in s if -2 <= a <= 7), 6.5e-4, 7.0e-4)
    # drag without the f_v1 bypass minus drag with it, structured L2, in counts
    mig = open(os.path.join(PAPER, "data", "fv1_migration_comparison.txt")).read().split("\n")
    for a in (0, 2, 5):
        ln = next(x for x in mig if x.startswith(f"strL2prop_eppler387_Re200k_a{a} "))
        between(f"drag without the bypass minus with it, alpha={a} [counts]", -float(ln.split()[2]), 5.5, 10.5)
    of = [a for a in o if o[a][0] == o[a][0]]
    between("OpenFOAM lift difference", max(abs(o[a][0] / s[a][0] - 1) for a in of), 0.005, 0.006)
    ahead = [s[a][2] - o[a][3] for a in of]
    behind = [o[a][4] - s[a][3] for a in of]
    between("OpenFOAM separates ahead, min", min(ahead), 0.02, 0.025)
    between("OpenFOAM separates ahead, max", max(ahead), 0.055, 0.06)
    between("OpenFOAM reattaches behind, min", min(behind), 0.01, 0.015)
    between("OpenFOAM reattaches behind, max", max(behind), 0.04, 0.045)

    # ----------------------------------------------------------- Reynolds study
    print("Eppler 387, Reynolds study")
    rs = {num(r[0].split("\\times")[0].replace("\\!", "")): [num(x) for x in r[1:]]
          for r in table_rows(tex, "t:eppresweep")}
    r1 = rs[1.0]
    between("structured drag / measured - 1 at 1e5", r1[1] / r1[10] - 1, 0.725, 0.735)
    between("unstructured drag / measured - 1 at 1e5", r1[4] / r1[10] - 1, 0.29, 0.30)
    t3 = open(os.path.join(PAPER, "tables", "ext2026_epp_re300k_wp.tex")).read()
    t1 = open(os.path.join(PAPER, "tables", "ext2026_epp_re100k_wp.tex")).read()
    d3 = [num(r[k + 4]) - num(r[k + 2]) for r in table_rows(t3, "t:extepp300k")
          for k in (1, 6) if -2 <= num(r[0]) <= 6]
    between("3e5: reattachment downstream of the front, min", min(d3), 0.03, 0.035)
    between("3e5: reattachment downstream of the front, max", max(d3), 0.07, 0.075)
    d1 = [num(r[k + 4]) - num(r[k + 2]) for r in table_rows(t1, "t:extepp100k")
          for k in (1, 6) if -2 <= num(r[0]) <= 6]
    d1 = [v for v in d1 if v == v]
    between("1e5: reattachment downstream of the front, min", min(d1), 0.165, 0.17)
    between("1e5: reattachment downstream of the front, max", max(d1), 0.385, 0.39)

    # ------------------------------------------------------------------ cylinder
    print("cylinder")
    td = open(os.path.join(PAPER, "tables", "tab_dragcrisis_wp.tex")).read()
    rows = table_rows(td, "t:dragcrisis")
    assert len(rows) == 25
    assert sum("[" in r[2] for r in rows) == 5
    cd = [num(r[2].split("\\,")[0]) for r in rows]
    tr = [num(r[3]) for r in rows]
    sep = [num(r[4]) for r in rows]
    between("C_d at 1e4-1e5, min", min(cd[8:11]), 0.835, 0.84)
    between("C_d at 1e4-1e5, max", max(cd[8:11]), 0.845, 0.85)
    for i in (10, 11, 12):
        between(f"separation angle, row {i}", sep[i], 73.0, 74.0)
        between(f"transition minus separation angle, row {i}", tr[i] - sep[i], 5.0, 8.0)
    between("first separation at 1e6", sep[14], 94.5, 95.0)
    between("first separation at 1.78e6", sep[15], 98.0, 98.5)
    between("turbulent separation from 3e6, min", min(sep[16:]), 117.5, 118.0)
    between("turbulent separation from 3e6, max", max(sep[16:]), 121.0, 121.5)
    between("transition angle at 1.78e6", tr[15], 98.0, 98.0)
    between("transition angle at 1e10", tr[24], 2.5, 2.5)

    # ------------------------------------------------------------------ Daedalus
    print("Daedalus")
    i = tex.index("\\label{t:daetotals}")
    body = tex[i:tex.index("\\end{tabular}", i)]
    l2 = [[num(c) for c in ln.split("&")[2:6]] for ln in body.split("\n") if "& L2 &" in ln]
    ref = [num(ln.split("&")[3]) for ln in body.split("\n") if "AVL" in ln and "&" in ln]
    dcl = [abs(r[0] / r[2] - 1) for r in l2]
    dcd = [abs(r[1] - r[3]) * 1e4 for r in l2]
    between("families, C_L difference at L2, min", min(dcl), 0.0035, 0.0037)
    between("families, C_L difference at L2, max", max(dcl), 0.0063, 0.0064)
    between("families, C_D difference at L2 [counts], min", min(dcd), 0.65, 0.75)
    between("families, C_D difference at L2 [counts], max", max(dcd), 1.65, 1.75)
    for lev_, cl_lo, cl_hi, cd_lo, cd_hi in (("L1", 0.0045, 0.0054, 0.35, 1.65), ("L0", 0.0074, 0.0098, 17.5, 21.0)):
        rr = [[num(c) for c in ln.split("&")[2:6]] for ln in body.split("\n") if f"& {lev_} &" in ln]
        assert len(rr) == 3
        for r in rr:
            between(f"families, C_L difference at {lev_}", abs(r[0] / r[2] - 1), cl_lo, cl_hi)
            between(f"families, C_D difference at {lev_} [counts]", abs(r[1] - r[3]) * 1e4, cd_lo, cd_hi)
    for r, v, want in zip(l2, ref, (15.6, 10.3, 5.6)):
        between("counts below the reference", (v - r[1]) * 1e4, want - 0.05, want + 0.05)
    between("N_crit of the wing's chi_inf", np.log(7.1 / 8.76e-6), 13.55, 13.65)

    # ------------------------------------------------------------------ spheroid
    print("spheroid")
    ts = open(os.path.join(PAPER, "tables", "tab_spheroid_tunnel_wp.tex")).read()
    rows = [[c.replace("$", "").replace("\\\\", "").strip() for c in ln.split("&")]
            for ln in ts.split("\n") if ln.count("&") == 11 and "circ" in ln]
    get = lambda a, re, tun, seed: next(r for r in rows if r[0].startswith(a + "^")
                                        and r[1] == re and tun in r[2] and r[3] == seed)
    wind = lambda *k: float(get(*k)[9])
    for a, re, want in (("0", "7.2", 0.174), ("2.5", "7.2", 0.181), ("5", "6.49", 0.239),
                        ("10", "6.56", 0.012)):
        d = wind(a, re, "ttingen", "calibrated") - wind(a, re, "ttingen", "measured")
        between(f"windward front moves between the two chi_inf, alpha={a}", d, want - 1e-3, want + 1e-3)
    between("ONERA F1 windward front moves",
            abs(wind("10", "6.56", "ONERA", "calibrated") - wind("10", "6.56", "ONERA", "measured")), 0, 0)
    for tu, n in ((0.106, 8.0), (0.161, 7.0)):
        between(f"N_crit of Tu={tu}%", -8.43 - 2.4 * np.log(tu / 100), n - 0.02, n + 0.02)
    print("All quoted numbers reproduced.")


if __name__ == "__main__":
    main()
