"""vg-kernel REGRESSION runs (2026-07-29): confirm the low-H vg change does not
break calibrated cases. Clones a canon converged case, warm-restarts from its
own restartOutput, and re-solves with the vg kernel ON (env AI_A_VISC=0.0276,
AI_REOMC_BC=130) under a bounded staged-fSlow protocol -- so canon vs vg differ
ONLY in the kernel (same mesh, config, seed, warm start). The canon original is
left untouched; the vg clone is written to <src>_vg.

Extraction:
  flatplate : chi=1 crossing in Re_x from volume.pvtu (regen_flatplate_flow360
              convention chi = nuHat/(M*NU), M=0.1, NU=1e-6).
  eppler    : upper-surface transition x/c from the max Cf jump (run.py xtr).

Usage:
  python run_vg_regression.py --case flatplate_sphere_Tu0160 --kind flatplate --gpu 2
  python run_vg_regression.py --case strL2prop_eppler387_Re200k_a2 --kind eppler --gpu 3
"""
from __future__ import annotations
import argparse, json, os, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_dragcrisis_matrix import solve_stage, read_cd_trace          # noqa: E402
from run_dragcrisis_pilot import make_env                             # noqa: E402
sys.path.insert(0, str(HERE.parent / "driver"))
from saai_env import canonical_env                                    # noqa: E402

AI = Path("/local_data/qiqi/sa-ai/flow360_fv1")
STAGES = [(4000, 20000, 0.1), (3000, 12000, 0.01)]   # warm migrate then settle
FLAT_TOL = 1e-4


def clone(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in os.listdir(src):
        s = src / f
        if not s.is_file():
            continue
        if f.endswith((".pvtu", ".vtu", ".gltf")) or f.startswith(
                ("total_forces", "surface_forces", "nonlinear_res",
                 "linear_", "max_residual", "minmax", "progress", "cfl_",
                 "performance", "ipc", "stop")):
            continue
        shutil.copy2(s, dst / f)
    # warm restart from the canon converged state
    ro = src / "restartOutput"
    for f in os.listdir(ro):
        shutil.copy2(ro / f, dst / f)


def patch(case: Path, chi_inf: float, fslow: float, cap: int):
    """Set the BC seed = chi_inf*fslow (pre-compensation) for this stage.
    chi_inf is fixed (recovered ONCE from the original JSON), NOT re-derived
    from the mutated seed -- otherwise stage-to-stage rescaling compounds."""
    p = case / "Flow360.json"
    d = json.loads(p.read_text())
    d.setdefault("runControl", {})["restart"] = True
    d.setdefault("timeStepping", {})["maxPseudoSteps"] = cap
    seed = chi_inf * fslow
    d["freestream"].setdefault("turbulenceQuantities", {})
    d["freestream"]["turbulenceQuantities"] = {
        "modelType": "ModifiedTurbulentViscosityRatio",
        "modifiedTurbulentViscosityRatio": seed}
    for bc in d.get("boundaries", {}).values():
        if bc.get("type") == "Freestream" and bc.get("turbulenceQuantities"):
            bc["turbulenceQuantities"]["modifiedTurbulentViscosityRatio"] = seed
    p.write_text(json.dumps(d, indent=1))


def ensure_mesh_dump(case: Path, find, env):
    """Some canon dirs keep only the partitionerData, not the solver-ready
    mesh.cgns_rank_1_of_1.dmp. Regenerate it with MeshPartitioner+MeshProcessor."""
    if (case / "mesh.cgns_rank_1_of_1.dmp").exists():
        return
    import subprocess
    cenv = {k: v for k, v in env.items() if not callable(v)}
    for cmd in (["MeshPartitioner", "--meshfile", "mesh.cgns", "--partitions", "1"],
                ["MeshProcessor", "--threads", "1", "mesh.cgns"]):
        subprocess.run([find(cmd[0])] + cmd[1:], cwd=str(case), env=cenv,
                       check=True, stdout=open(case / "preprocess.log", "a"),
                       stderr=subprocess.STDOUT)


def run(case: Path, gpu: int, vg: bool):
    env, find = make_env(); env["_find"] = find
    ensure_mesh_dump(case, find, env)
    # chi_inf recovered ONCE from the ORIGINAL JSON seed (seed = chi_inf*0.01,
    # the canon final-stage pre-compensation) -- fixed for all stages
    d = json.loads((case / "Flow360.json").read_text())
    seed = d["freestream"]["turbulenceQuantities"]["modifiedTurbulentViscosityRatio"]
    chi_inf = seed / 0.01
    print(f"  chi_inf (fixed) = {chi_inf:.4e}", flush=True)
    for si, (mn, cap, fslow) in enumerate(STAGES):
        patch(case, chi_inf, fslow, cap)
        e = dict(env)
        e.update(canonical_env(chi_inf, laminar_slowdown=fslow))
        if vg:
            e["AI_A_VISC"] = "0.0276"; e["AI_REOMC_BC"] = "130"
        v = solve_stage(case, e, gpu, mn, cap, FLAT_TOL)
        ro = case / "restartOutput"
        if ro.is_dir():
            for f in os.listdir(ro):
                shutil.copy2(ro / f, case / f)
        print(f"  stage{si} fSlow{fslow}: {v}", flush=True)


def front_flatplate(case: Path):
    """chi=1 crossing in Re_x from volume.pvtu (M*NU=1e-7 chi convention)."""
    import numpy as np
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    f = case / "volume.pvtu"
    if not f.exists():
        return None
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(str(f)); r.Update()
    g = r.GetOutput()
    pts = vtk_to_numpy(g.GetPoints().GetData())
    pd = g.GetPointData()
    nuhat = vtk_to_numpy(pd.GetArray("nuHat"))
    x, z = pts[:, 0], pts[:, 1]
    chi = nuhat / 1e-7                       # M*NU = 0.1*1e-6
    # near-wall band, max chi per x-bin
    band = z < 5e-4
    xb, cb = x[band], chi[band]
    bins = np.linspace(0, xb.max(), 400)
    xc = 0.5 * (bins[1:] + bins[:-1])
    cmax = np.array([cb[(xb >= bins[i]) & (xb < bins[i + 1])].max()
                     if ((xb >= bins[i]) & (xb < bins[i + 1])).any() else np.nan
                     for i in range(len(bins) - 1)])
    ok = np.isfinite(cmax)
    xc, cmax = xc[ok], cmax[ok]
    cr = np.where(cmax >= 1.0)[0]
    if len(cr) == 0:
        return None
    x_tr = xc[cr[0]]
    return {"x_tr": float(x_tr), "Re_x_tr": float(x_tr / 1e-6)}   # Re_unit=1e6


def front_eppler(case: Path):
    """upper-surface xtr from max Cf jump (run.py convention)."""
    sys.path.insert(0, str(HERE.parent / "driver"))
    from run import _extract_xtr
    return {"xtr": _extract_xtr(case, "eppler387")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--kind", choices=["flatplate", "eppler"], required=True)
    ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()
    src = AI / args.case
    dst = AI / f"{args.case}_vg"
    front = front_flatplate if args.kind == "flatplate" else front_eppler
    canon_front = front(src)
    print(f"[{args.case}] canon front: {canon_front}", flush=True)
    clone(src, dst)
    run(dst, args.gpu, vg=True)
    vg_front = front(dst)
    ps, cl, cd = read_cd_trace(dst)
    out = {"case": args.case, "kind": args.kind,
           "canon_front": canon_front, "vg_front": vg_front,
           "vg_Cd": float(cd[-1]) if cd else None,
           "vg_CL": float(cl[-1]) if cl else None}
    (dst / "vg_regression.json").write_text(json.dumps(out, indent=2))
    print(f"[{args.case}] RESULT {json.dumps(out)}", flush=True)


if __name__ == "__main__":
    main()
