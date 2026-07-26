"""Run every analytic/ repro script (figures + tables + constant checks) and print
a pass/fail summary. Each is executed in-process via runpy so import side effects
(sys.path, cwd to paper/) are shared. Figures land in sa-ai/paper/figs/."""
import os
import runpy
import sys
import time
import traceback

ANALYTIC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ANALYTIC)

# Ordered to follow the paper (read each script alongside the passage it
# backs). The two slow Tier-1/N-level scans (scan_background_constants.py,
# scan_anchor_level.py) are on-demand and not part of this suite.
SCRIPTS = [
    # Sec. II -- the model and its committed figures
    "fig01_indicator_sphere.py", # fig:model (rate coordinate on the sphere)
    "fig02_onset_graze.py",      # fig:onsetgraze (soft-min onset curve)
    "fig02_model_calibrate.py",  # fig:calibrate
    "amax_rayleigh.py",          # a_max = tanh-layer eigenvalue (asserts)
    "fig03_fs_transport_rows.py",# fig:nuhat (instrument on three wedges)
    "fig04_shapefactor.py",      # Sec. II shape-factor family (prose numbers)
    # Sec. II.E -- turbulent layer and handover (prose verification)
    "verify_wall_layer_tie.py",  # tie exactness (asserted; no figure)
    "tab02_yplus.py",            # II.E round-off statement (table dropped)
    # receptivity + assembled constants
    "tu_map.py",                 # eq:tumap (seed list in Sec. IV)
    "constants_report.py",       # Appendix-E constants block (asserts)
    # Historical v2-record scripts (fig01_indicator_plane, fig02_kernel_maps,
    # fig05_06_klambda, fit_fpg_rate_slope, verify_three_anchors) are retired
    # from the suite: their floats/constants are no longer in the paper.
]


def run(name):
    path = os.path.join(ANALYTIC, name)
    g = runpy.run_path(path, run_name="_loaded")
    main = g.get("main")
    if callable(main):
        main()


def main():
    results = []
    for name in SCRIPTS:
        print(f"\n{'='*70}\n>> {name}\n{'='*70}", flush=True)
        t0 = time.time()
        try:
            run(name)
            results.append((name, True, time.time() - t0, ""))
        except Exception as e:
            traceback.print_exc()
            results.append((name, False, time.time() - t0, repr(e)))
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    npass = sum(1 for _, ok, _, _ in results if ok)
    for name, ok, dt, err in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<28} {dt:6.1f}s  {err}")
    print(f"\n{npass}/{len(results)} passed.")
    sys.exit(0 if npass == len(results) else 1)


if __name__ == '__main__':
    main()
