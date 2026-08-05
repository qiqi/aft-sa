"""CLI: build (if needed) + solve one SA-AI case, then extract results.

    python3 run.py <case_name> [--outdir DIR] [--gpu N] [--build-only]

``<case_name>`` is a key in ``cases.py`` (the paper matrix). The solve runs
IN-SESSION (no mpirun): ``rans.solve.run_solver`` sets OMPI_COMM_WORLD_LOCAL_RANK=0
and launches Flow360Solver directly, so this must run in a normal (non-sandboxed)
shell -- the GPU solver is killed inside a restricted agent sandbox.

The canonical SA-AI env comes from ONE place -- ``saai_env.canonical_env`` -- and
is merged into the solver subprocess env here. Nothing else sets AI_* vars.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import CaseConfig
from saai_env import canonical_env, classical_sa_env
import case as _casebuild

_DRIVER = Path(__file__).resolve().parent

# The solve/extract layer lives in flexfoil/rans (sibling repo checkout).
_RANS = _DRIVER.parent.parent.parent.parent / "flexfoil" / "rans"
if str(_RANS) not in sys.path:
    sys.path.insert(0, str(_RANS))


def _extract_xtr(case_dir: Path, wall_name: str) -> float | None:
    """Transition location x/c from the max Cf jump on the upper surface.

    Best-effort: needs the surface .pvtu output + vtk/numpy. Returns None if the
    output or the vtk dependency is unavailable (e.g. a build-only run).
    """
    pvtu = case_dir / f"surface_fluid_{wall_name}.pvtu"
    if not pvtu.exists():
        return None
    try:
        import numpy as np
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except Exception:
        return None
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(str(pvtu)); r.Update()
    g = r.GetOutput()
    pts = vtk_to_numpy(g.GetPoints().GetData())
    pd = g.GetPointData()
    names = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    cf_name = next((n for n in names if n.lower().startswith("cf")), None)
    if cf_name is None:
        return None
    a = vtk_to_numpy(pd.GetArray(cf_name))
    cf = np.linalg.norm(a, axis=1) if a.ndim > 1 else a
    x, z = pts[:, 0], pts[:, 2]
    up = z > 1e-6
    xs, cfs = x[up], cf[up]
    o = np.argsort(xs); xs, cfs = xs[o], cfs[o]
    bins = np.linspace(0, 1, 51); xc = 0.5 * (bins[1:] + bins[:-1])
    cb = np.array([np.median(cfs[(xs >= bins[i]) & (xs < bins[i + 1])])
                   if ((xs >= bins[i]) & (xs < bins[i + 1])).any() else np.nan
                   for i in range(50)])
    ok = np.isfinite(cb); xc, cb = xc[ok], cb[ok]
    d = np.diff(cb); dm = 0.5 * (xc[1:] + xc[:-1])
    w = (dm > 0.04) & (dm < 0.95)
    if not w.any():
        return None
    return float(dm[w][np.argmax(d[w])])


def run_case(cfg: CaseConfig, outdir: str | Path, *, gpu: int = 0,
             build_only: bool = False) -> dict:
    """Build the case, set the canonical SA-AI env, solve in-session, extract."""
    outdir = Path(outdir)
    print(f"[{cfg.name}] building case dir -> {outdir}", flush=True)
    case_dir = _casebuild.build(cfg, outdir)

    chi = cfg.resolved_chi_inf()
    print(f"[{cfg.name}] chi_inf = {chi:.4e} "
          f"(Tu={cfg.Tu_pct}% )" if cfg.Tu_pct else f"[{cfg.name}] chi_inf = {chi:.4e}",
          flush=True)

    if build_only:
        print(f"[{cfg.name}] build-only: case ready at {case_dir}", flush=True)
        return {"case_dir": str(case_dir), "chi_inf": chi, "built": True}

    # ---- solve under the paper's convergence protocol (single source) ----
    from solve import extract_forces
    import convergence
    if getattr(cfg, "turbulent", False):
        env = classical_sa_env(chi)                # AI_SA=0 turbulent baseline
    else:
        env = canonical_env(chi, laminar_slowdown=cfg.laminar_slowdown)

    mdl = "classical SA (AI_SA=0)" if getattr(cfg, "turbulent", False) else "canonical SA-AI"
    proto = convergence.protocol_for(cfg)
    print(f"[{cfg.name}] solving on GPU {gpu} ({mdl} env, protocol={proto})...", flush=True)
    convergence.solve_to_convergence(cfg, case_dir, env, gpu=gpu)

    forces = extract_forces(case_dir)
    wall = cfg.geometry if cfg.geometry != "flatplate" else "wall"
    xtr = _extract_xtr(case_dir, wall)

    result = {
        "case_dir": str(case_dir),
        "chi_inf": chi,
        "CL": forces["CL"], "CD": forces["CD"], "L_over_D": forces["L_over_D"],
        "xtr": xtr,
        "csv": {
            "forces": str(case_dir / "total_forces_v2.csv"),
            "surface_Cf": str(case_dir / f"surface_fluid_{wall}.pvtu"),
            "residual": str(case_dir / "nonlinear_residual_v2.csv"),
        },
    }
    print(f"[{cfg.name}] DONE  CL={forces['CL']:+.4f}  CD={forces['CD']:.5f}"
          + (f"  xtr={xtr:.3f}" if xtr is not None else ""), flush=True)
    print(f"[{cfg.name}] outputs in {case_dir}", flush=True)
    (case_dir / "saai_result.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run one SA-AI reproduction case.")
    ap.add_argument("case_name", help="key in cases.py (or 'list' to show all)")
    ap.add_argument("--outdir", default=None,
                    help="output dir (default: repro/driver/out/<case_name>)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--build-only", action="store_true",
                    help="build the case dir but do not solve")
    args = ap.parse_args(argv)

    import cases
    if args.case_name == "list":
        for k in cases.all_cases():
            print(k)
        return
    cfg = cases.get(args.case_name)
    outdir = Path(args.outdir) if args.outdir else _DRIVER / "out" / cfg.name
    run_case(cfg, outdir, gpu=args.gpu, build_only=args.build_only)


if __name__ == "__main__":
    main()
