"""Run every analytic/ repro script (figures + tables + constant checks) and print
a pass/fail summary. Each is executed in-process via runpy so import side effects
(sys.path, cwd to paper/) are shared. Figures land in sa-ai/paper/figs/."""
import os
import runpy
import sys
import time
import traceback

ANALYTIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analytic")
sys.path.insert(0, ANALYTIC)

SCRIPTS = [
    "constants_report.py",       # paper Table   (asserts)
    "amax_rayleigh.py",          # a_max = tanh-layer eigenvalue (asserts)
    "tu_map.py",                 # eq:tumap
    "fig01_indicator_plane.py",  # fig:indicatorplane
    "fig02_kernel_maps.py",      # fig:kernel
    "fig03_fs_transport_rows.py",# fig:nuhat
    "fig04_shapefactor.py",      # fig:shapefactor
    "fig05_06_klambda.py",       # fig:worstpoint, fig:klambda_sc
    "fig07_wall_layer.py",       # fig:walllayer
    "tab02_yplus.py",            # tab:yplus
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
