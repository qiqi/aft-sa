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
    # Sec. II -- the model and its indicators
    "fig01_indicator_plane.py",  # fig:indicatorplane (FS family in the indicator planes)
    "fig02_kernel_maps.py",      # fig:kernel (S(z) map + cliff)
    # Sec. III.A -- the rate ceiling and the three-anchor solve
    "amax_rayleigh.py",          # a_max = tanh-layer eigenvalue (asserts)
    "verify_three_anchors.py",   # canonical triple satisfies the 3 conditions (asserts)
    "fig03_fs_transport_rows.py",# fig:nuhat (instrument on three wedges)
    "fig04_shapefactor.py",      # fig:shapefactor (family, cliff-only + factored)
    # Sec. III.B -- favorable-gradient pair (Tier 3)
    "fig05_06_klambda.py",       # fig:worstpoint, fig:klambda_sc (K_lambda fixed point)
    "fit_fpg_rate_slope.py",     # eq:kr one-point fit at beta=0.35 (asserts)
    # Sec. III.C -- turbulent layer and handover
    "verify_wall_layer_tie.py",       # Sec III.E tie exactness (asserted; no figure)
    "tab02_yplus.py",            # tab:yplus
    # Sec. III.D -- receptivity
    "tu_map.py",                 # eq:tumap
    # Sec. III.E -- assembled constants
    "constants_report.py",       # constants block (asserts vs paper Table)
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
