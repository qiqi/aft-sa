"""Write the paper's copies of the shared results tables.

The tables tables/*_wp.tex are emitted by the campaign scripts and are \\input by the
whitepaper. The paper uses the same rows with its own captions, in the paper's words,
and lets the tables float. This script copies each *_wp.tex to *_paper.tex with the
caption replaced, so the numbers have one source and an edit of a caption here does
not change the whitepaper.

    python3 repro/tools/make_paper_tables.py
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.normpath(os.path.join(HERE, "..", "..", "tables"))

CAPTIONS = {
    "ext2026_nlf_somers": (
        r"NLF(1)-0416: computed transition locations on the structured (str) and cavity "
        r"(cav) L2 grids at four Reynolds numbers, and the brackets of "
        r"Somers~\cite{somers_1981}, Fig.~9, at the same lift. A bracket extends from the "
        r"last laminar orifice to the first turbulent orifice. $^\dagger$Also in "
        r"Table~\ref{t:nlf}."),
    "ext2026_nlf_resweep": (
        r"NLF(1)-0416 at $Re=1$, $2$ and $3\times10^6$, on the L2 grids: forces and "
        r"transition locations."),
    "ext2026_epp_re300k": (
        r"Eppler 387 at $Re=3\times10^5$, on the L2 grids. $x_\mathrm{tr}$ is the station "
        r"where the largest $\chi$ near the wall crosses one. $x_\mathrm{sep}$ and "
        r"$x_\mathrm{reatt}$ are zero crossings of the signed $C_f$. A dash means that no "
        r"bubble closes ahead of the trailing edge."),
    "ext2026_epp_re100k": (
        r"Eppler 387 at $Re=10^5$, on the L2 grids; columns as in "
        r"Table~\ref{t:extepp300k}. The structured solution at $\alpha=8^\circ$ is "
        r"unsteady: its transition location varies between $0.24\,c$ and $0.57\,c$ over "
        r"$8\times10^4$ iterations, and the last value is tabulated."),
    "tab_twoelement_geom": (
        r"Two-element section, in units of the total chord. Main element: mean line "
        r"$z_c=x_e[\hat x(1-\hat x)\sum_{i=0}^{10}A^{c}_iB^{10}_i(\hat x)]"
        r"+\hat x\,\Delta z_{te}$ and half-thickness "
        r"$t=x_e[\hat x^{1/2}(1-\hat x)\sum_{i=0}^{9}A^{t}_iB^{9}_i(\hat x)"
        r"+\hat x\,t_{te}/(2x_e)]$, with $\hat x=x/x_e$ and $B^{n}_i$ the Bernstein "
        r"polynomials. The surfaces are offset by $\pm t$ normal to the mean line, and "
        r"the leading edge is at the origin."),
    "tab_dragcrisis": (
        r"Circular cylinder, $\chi_\infty=1.1\times10^{-2}$: drag coefficient, transition "
        r"angle $\theta_{tr}$, and angle of the first separation $\theta_{sep}$. $C_d$ is "
        r"the median over the last iterations. Brackets: the range of $C_d$ where the "
        r"criterion for a steady solution is not met. $|C_L|<2\times10^{-6}$ in every "
        r"computation."),
    "tab_spheroid_tunnel": (
        r"6:1 spheroid: computed minus measured transition location, in units of $L$, at "
        r"the azimuths with a measurement. The computed location is the station where the "
        r"largest $\chi$ near the wall crosses one. The fitted $\chi_\infty$ is that of "
        r"the $N_\mathrm{crit}$ that Stock~\cite{stock_2006} fitted for the tunnel, and "
        r"the measured $\chi_\infty$ is that of the measured turbulence intensity $Tu$. "
        r"The column $n$ is the number of azimuths at which $\chi$ reaches one ahead of "
        r"$x/L=0.97$, over the number of measured azimuths. The columns lee and wind are "
        r"the differences at the most leeward and the most windward measured azimuth. The "
        r"last two columns are the same differences for the $e^N$ computation of Stock, "
        r"digitized from his figures. Upper block: $Re_L\approx1.5\times10^6$, where the "
        r"measured line is the laminar separation line. Lower block: "
        r"$Re_L\ge6.49\times10^6$."),
}

# Words in the table bodies that differ between the two documents.
BODY = {
    "tab_twoelement_geom": [(r"\emph{Fore element}", r"\emph{Main element}")],
    "tab_spheroid_tunnel": [("& seed &", r"& $\chi_\infty$ &"), ("calibrated", "fitted")],
}


def replace_caption(text, caption):
    m = re.search(r"\\caption\{", text)
    i, depth = m.end(), 1
    while depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[:m.end()] + caption + text[i - 1:]


def main():
    for name, caption in CAPTIONS.items():
        text = open(os.path.join(TABLES, name + "_wp.tex")).read()
        assert text.count("\\begin{table}[H]") == 1, name
        text = text.replace("\\begin{table}[H]", "\\begin{table}[!htbp]")
        text = replace_caption(text, caption)
        for old, new in BODY.get(name, []):
            assert old in text, (name, old)
            text = text.replace(old, new)
        out = os.path.join(TABLES, name + "_paper.tex")
        with open(out + ".tmp", "w") as f:
            f.write("%% Written by repro/tools/make_paper_tables.py from " + name
                    + "_wp.tex. Do not edit.\n" + text)
        os.replace(out + ".tmp", out)
        print("wrote", os.path.relpath(out, os.path.join(TABLES, "..")))


if __name__ == "__main__":
    main()
