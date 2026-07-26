"""Regenerate every Flow360-dependent paper figure and table.

Runs the nine CFD generators in ``sa-ai/paper/`` against the case tree named
by ``SAAI_CFD_ROOT`` (default: the shipped ``sa-ai/flow360_fr``). Run
``prepare.py`` first if the tree holds fresh solves (derived slice fields).

    SAAI_CFD_ROOT=/path/to/tree python3 regenerate_cfd.py

Figures land in ``sa-ai/paper/figs/`` under the exact names sa-ai.tex includes;
table generators print their LaTeX rows to stdout.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

CFD = Path(__file__).resolve().parent                 # paper/repro/cfd
PAPER = CFD.parent.parent                             # paper/ (figures land in paper/figs)

# Ordered to follow the paper. xfoil_ncrit_sweep.py (Sec. VI alpha=7
# N_crit discussion) is on-demand (needs xfoil + xvfb-run) and not run here.
SCRIPTS = [  # (script, paper floats/passages it feeds)
    # Sec. IV -- flat plate
    ("regen_flatplate_flow360.py", "fig:flatplate_batch (+ ONSET_DIAG=1: AGS onset numbers)"),
    # Sec. V -- NLF(1)-0416
    ("regen_nlf_v2.py",            "fig nlf_cf_lowalpha/highalpha + fig:nlfnegalpha (neg mode)"),
    ("regen_nlf_transition.py",    "Sec. V transition-front prose numbers"),
    ("regen_nlf_polar.py",         "fig:nlfpolar"),
    ("regen_nlf_aft_comparison.py","fig:nlfaft (headline transition comparison)"),
    # Sec. VI -- Eppler 387
    ("regen_eppler_v2.py",         "fig eppler_cf_lowalpha/highalpha, fig:epppolar"),
    ("regen_epp_reattach.py",      "Sec. VI separation/reattachment prose numbers"),
    ("regen_epp_bubble_figure.py", "fig:eppbubble"),
    # Sec. VI, Reynolds sweep
    ("regen_epp_resweep_suite.py", "fig:eppresweep_low, fig:eppresweep_high"),
    ("regen_epp_resweep_forces.py","fig:eppresweepforces"),
    ("regen_resweep_table.py",     "tab:eppresweep / Table C1 (printed LaTeX rows)"),
    # Sec. VII -- Daedalus
    ("regen_daedalus_section_sheets.py", "fig:daesurf4/5/6"),
    # Appendices -- wall-anchored contour sheets
    ("regen_chi_sheets.py",        "fig:sheet_* (6 grids x 2 columns per case-surface)"),
    ("regen_negalpha_sheets.py",   "fig:sheet_nlf0416_negalpha_* (L2 pair)"),
    ("regen_epp_ext_sheets.py",    "fig:sheet_eppler387_{am2,a8p5}_upper (L2 pair)"),
]


def main():
    root = os.environ.get("SAAI_CFD_ROOT", "(default: flow360_fv1 canon)")
    print(f"SAAI_CFD_ROOT = {root}")
    results = []
    for script, feeds in SCRIPTS:
        print(f"\n{'='*70}\n>> {script}   [{feeds}]\n{'='*70}", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, str(CFD / script)], cwd=PAPER)
        results.append((script, r.returncode == 0, time.time() - t0))
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    npass = sum(ok for _, ok, _ in results)
    for script, ok, dt in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {script:<28} {dt:6.1f}s")
    print(f"\n{npass}/{len(results)} passed.")
    sys.exit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
