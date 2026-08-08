"""The 2026-08 literature-comparability extension matrix (L2 pair only).

Five blocks, all on the finest (L2) grid of BOTH mesh families, all cold-started
at the paper-wide canon environment and front-converged with the campaign's
converge_by_xtr protocol -- i.e. the same recipe as run_negalpha_nlf.py /
run_nlfneg_case.py, only the (alpha, muRef) pair changes.

  P1  NLF(1)-0416 @ Re=4e6, alpha = -6,-2,2,6,8
        -> exact one-to-one overlap with all 7 Transition-Workshop Case-2B
           incidences and all 7 Piotrowski-Zingg incidences (we previously had
           only -4, 0, 4 inside that window).
  S   NLF(1)-0416 @ Re=4e6, alpha = -12,-10  (the Somers add-on)
        -> Somers TP-1861 Fig. 9 measures upper-surface fronts down to
           c_l ~ -1.04 (alpha ~ -13); our old floor was -8 (c_l ~ -0.49).
  P4  NLF(1)-0416 Reynolds sweep at alpha = 0, 4; Re = 1,2,3 x10^6
        -> Somers Fig. 9(a)-(c) measures transition at exactly these Re at
           M=0.10. The favorable-gradient/TS counterpart to the Eppler bubble
           Re-sweep; nothing currently tests K_lambda across Re.
  P2  Eppler 387 @ Re=2e5, alpha = -1, 8, 9
        -> completes the -2..9 integer grid of Shahjahan's LM / SA-BC and of
           Cole & Mueller Fig. 4.38. 8 and 9 are the bursting region where the
           two published models diverge hardest.
  P3  Eppler 387 alpha sweeps at Re = 1e5 and 3e5 (previously alpha=5 only)
        Re=1e5: Cole & Mueller Fig. 4.37 (11 alphas, all with sep+reattach --
                the best-instrumented E387 bubble dataset in the literature),
                Shahjahan Fig. 9(a) x_tr, IJSRP trans-SST.
        Re=3e5: Ghimire 2025 Tables 6-8 (three OpenFOAM transitional models +
                experiment, LSB sep/tr/reattach) at alpha=2,4,6; Cole & Mueller
                Fig. 4.39 at -2,0,1,3; IJSRP at even alpha.

Re=6e4 is deliberately EXCLUDED (below the E387 design Reynolds number; the
model's low-Re bubble behaviour is not the claim under test), as is Re=4.6e5
(no published model results to compare against -- McGhee only).

Every case reuses the existing L2 mesh unchanged: alpha is freestream
alphaAngle and Re is muRef = Mach/Re. No remeshing anywhere in this matrix.
"""
from __future__ import annotations

MACH = 0.1
FAMS = ("str", "cav")

# Base clone sources (per family). Both are converged L2 canon cases.
BASE_NLF = "{fam}L2prop_nlf0416_Re4M_a4"
BASE_EPP = "{fam}L2prop_eppler387_Re200k_a5"


def _atag(a: float) -> str:
    """Existing tree convention: a5, a8p5, am2 (negatives use the 'am' prefix)."""
    s = f"{abs(a):g}".replace(".", "p")
    return ("am" if a < 0 else "a") + s


def _mu(re: float) -> float:
    return MACH / re


def _c(name, base, alpha, re, block):
    return dict(name=name, base=base, alpha=float(alpha), re=float(re),
                muRef=_mu(re), block=block)


CASES = []

# ---- P1 + Somers add-on: NLF incidence extension at Re=4e6 -----------------
for a in (-12, -10, -6, -2, 2, 6, 8):
    block = "S_nlf_deepneg" if a <= -10 else "P1_nlf_alpha"
    for fam in FAMS:
        CASES.append(_c(f"{fam}L2prop_nlf0416_Re4M_{_atag(a)}",
                        BASE_NLF.format(fam=fam), a, 4.0e6, block))

# ---- P4: NLF Reynolds sweep at alpha = 0, 4 -------------------------------
for re, rtag in ((1.0e6, "Re1M"), (2.0e6, "Re2M"), (3.0e6, "Re3M")):
    for a in (0, 4):
        for fam in FAMS:
            CASES.append(_c(f"nlfsweep_{fam}L2_{rtag}_{_atag(a)}",
                            BASE_NLF.format(fam=fam), a, re, "P4_nlf_resweep"))

# ---- P2: Eppler incidence completion at Re=2e5 ----------------------------
for a in (-1, 8, 9):
    for fam in FAMS:
        CASES.append(_c(f"{fam}L2prop_eppler387_Re200k_{_atag(a)}",
                        BASE_EPP.format(fam=fam), a, 2.0e5, "P2_epp_alpha200k"))

# ---- P3: Eppler incidence sweeps at the other Reynolds numbers ------------
for re, rtag, alphas in (
    (1.0e5, "Re100k", (-2, 0, 2, 4, 6, 8, 10)),
    (3.0e5, "Re300k", (-2, 0, 1, 2, 3, 4, 6, 8, 10)),
):
    for a in alphas:
        for fam in FAMS:
            CASES.append(_c(f"sweep_{fam}L2_{rtag}_{_atag(a)}",
                            BASE_EPP.format(fam=fam), a, re, "P3_epp_resweep"))

BY_NAME = {c["name"]: c for c in CASES}


def queues(n: int = 12):
    """Round-robin the matrix into n serial queues (one per GPU stream).

    Round-robin (not block) assignment so every queue carries a mix of blocks
    and of both mesh families -- a host dying mid-run then costs coverage
    evenly rather than wiping out one whole block.
    """
    qs = [[] for _ in range(n)]
    for i, c in enumerate(CASES):
        qs[i % n].append(c["name"])
    return qs


if __name__ == "__main__":
    import collections
    print(f"{len(CASES)} cases")
    for b, n in collections.Counter(c["block"] for c in CASES).items():
        print(f"  {b:22s} {n}")
    for i, q in enumerate(queues()):
        print(f"q{i:02d} ({len(q)}): {' '.join(q)}")
