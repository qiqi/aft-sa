"""tab:data_dragcrisis / t:dragcrisis -> paper/tables/tab_dragcrisis*.tex.

Emit the drag-crisis data table from the clean systematic Tu=0.2% UP-LADDER
campaign (systematic_Tu0.2_summary.jsonl, 25 matched-Re points 1..1e10),
replacing the old 84-case 3-seed matrix table. One row per up-ladder Re_D:
Cd (median over the final-stage tail window, [peak-to-peak] bracketed for
cases the steady monitor did not accept, i.e. verdict != converged), the
transition angle theta_tr (chi=1 radial ray) and the first wall-separation
angle. |C_L| is negligible on every case (symmetric protocol).

Writes two wrappers around one identical tabular body so the main paper and
the whitepaper stay single-source:
  tables/tab_dragcrisis.tex     (label tab:data_dragcrisis, main paper)
  tables/tab_dragcrisis_wp.tex  (label t:dragcrisis, whitepaper)

Run:  python3 repro/cfd/regen_dragcrisis_table.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.abspath(os.path.join(HERE, "..", "..", "tables"))
SYS = ("/local_data/qiqi/sa-ai/dragcrisis_matrix/"
       "systematic_Tu0.2_summary.jsonl")


def load_up():
    by = {}
    for ln in open(SYS):
        ln = ln.strip()
        if not ln:
            continue
        d = json.loads(ln)
        by[d["case"]] = d
    up = [r for r in by.values() if r["dir"] == "up"]
    up.sort(key=lambda r: r["re"])
    return up


def re_tex(re):
    """Compact LaTeX for a decade/half-decade Reynolds number."""
    if re < 1e4:
        return f"${int(round(re))}$"
    m, e = f"{re:.2e}".split("e")
    m = m.rstrip("0").rstrip(".")
    e = int(e)
    base = f"10^{{{e}}}"
    return f"${base}$" if m == "1" else f"${m}\\!\\times\\!{base}$"


def first_sep(r):
    for a, k in (r.get("crossings_upper") or []):
        if k == "separation":
            return a
    return None


def cd_cell(r):
    cd = f"{r['Cd']:.3f}"
    if r.get("verdict") != "converged":
        cd += f"\\,[{r['Cd_tail_p2p']:.3f}]"
    return cd


def body():
    up = load_up()
    lines = []
    for r in up:
        ttr = (f"{r['theta_tr_chi1']:.1f}"
               if r.get("theta_tr_chi1") is not None else "--")
        sep = first_sep(r)
        sepc = f"{sep:.1f}" if sep is not None else "--"
        lines.append(f"    {re_tex(r['re'])} & {r['mesh']} & {cd_cell(r)} "
                     f"& {ttr} & {sepc} \\\\")
    return "\n".join(lines), up


def write(path, label, caption, placement="tp"):
    rows, up = body()
    n_flag = sum(1 for r in up if r.get("verdict") != "converged")
    tex = (
        f"\\begin{{table}}[{placement}]\n"
        "  \\centering\\small\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n"
        "  \\begin{tabular}{l l c cc}\n"
        "    \\toprule\n"
        "    $Re_D$ & grid & $C_d$ & $\\theta_{tr}\\,(^\\circ)$ "
        "& $\\theta_{sep}\\,(^\\circ)$ \\\\\n"
        "    \\midrule\n"
        f"{rows}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(tex)
    os.replace(tmp, path)
    print(f"wrote {path}  ({len(up)} rows, {n_flag} flagged non-converged)")


CAP_MAIN = (
    r"Circular cylinder, steady-branch drag-crisis traverse: the clean "
    r"systematic $Tu\!=\!0.2\%$ up-ladder of Fig.~\ref{fig:dragcrisiscd} "
    r"(single seed, matched $Re_D$ grid, warm-started up-continuation over "
    r"four grid families). $C_d$ is the median over the final-stage tail "
    r"window; bracketed values are the tail-window $C_d$ peak-to-peak for the "
    r"cases the steady monitor did not accept (the open rings of the figure). "
    r"$\theta_{tr}$ is the $\chi\!=\!1$ radial-ray transition angle and "
    r"$\theta_{sep}$ the first wall-separation angle; $|C_L|\!<\!2\times"
    r"10^{-6}$ on every case (symmetric protocol)."
)
CAP_WP = (
    r"Cylinder steady drag-crisis traverse, systematic $Tu\!=\!0.2\%$ "
    r"up-ladder ($C_d$ median over the tail window, [peak-to-peak] bracketed "
    r"where the steady monitor did not accept; $\theta_{tr}$ the $\chi\!=\!1$ "
    r"transition angle, $\theta_{sep}$ the first separation; $|C_L|\!<\!2"
    r"\times10^{-6}$ throughout)."
)

if __name__ == "__main__":
    write(os.path.join(TABLES, "tab_dragcrisis.tex"),
          "tab:data_dragcrisis", CAP_MAIN, placement="tp")
    write(os.path.join(TABLES, "tab_dragcrisis_wp.tex"),
          "t:dragcrisis", CAP_WP, placement="H")
