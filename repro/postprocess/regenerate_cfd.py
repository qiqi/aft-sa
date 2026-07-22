"""Regenerate every Flow360-dependent paper figure and table.

Runs the nine CFD generators in ``sa-ai/paper/`` against the case tree named
by ``SAAI_CFD_ROOT`` (default: the shipped ``sa-ai/flow360_a3``). Run
``prepare.py`` first if the tree holds fresh solves (derived slice fields).

    SAAI_CFD_ROOT=/path/to/tree python3 regenerate_cfd.py

Figures land in ``sa-ai/paper/figs/`` under the exact names main.tex includes;
table generators print their LaTeX rows to stdout.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent.parent / "paper"

SCRIPTS = [  # (script, paper floats it feeds)
    ("regen_flatplate_flow360.py", "fig:flatplate_batch"),
    ("regen_nlf_v2.py",            "fig:nlfcflow, fig:nlfcfhigh"),
    ("regen_nlf_transition.py",    "tab:nlftrans (printed rows)"),
    ("regen_nlf_polar.py",         "fig:nlfpolar"),
    ("regen_eppler_v2.py",         "fig:eppcflow, fig:eppcfhigh, fig:epppolar"),
    ("regen_epp_reattach.py",      "tab:eppxtr (printed rows)"),
    ("regen_epp_L1compare.py",     "fig:eppresweep_low, fig:eppresweep_high"),
    ("regen_resweep_table.py",     "tab:eppresweep (printed LaTeX rows)"),
    ("regen_l0_artifact.py",       "(diagnostic; not included in main.tex)"),
]


def main():
    root = os.environ.get("SAAI_CFD_ROOT", "(default: shipped flow360_a3)")
    print(f"SAAI_CFD_ROOT = {root}")
    results = []
    for script, feeds in SCRIPTS:
        print(f"\n{'='*70}\n>> {script}   [{feeds}]\n{'='*70}", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, script], cwd=PAPER)
        results.append((script, r.returncode == 0, time.time() - t0))
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    npass = sum(ok for _, ok, _ in results)
    for script, ok, dt in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {script:<28} {dt:6.1f}s")
    print(f"\n{npass}/{len(results)} passed.")
    sys.exit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
