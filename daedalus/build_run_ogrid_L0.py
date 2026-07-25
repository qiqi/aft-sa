"""Stage/run the canon L0 cases (the canon campaign only ran L1/L2).
BOTH archived L0 meshes were recovered from 017-v100-dev
(~/flexcompute/sa-ai/scripts/daedalus/case_{ogrid,cavity}_saai/mesh.cgns
-- the predecessor tree survives there): _ogrid_L0_mesh/mesh_archived.cgns
and _cavity_L0_mesh/mesh.cgns. The ogrid L0 also regenerates
byte-equivalently from the tracked ogrid_wing.py ladder (same size;
md5 differs only through HDF5 metadata); the archived originals are used
for provenance continuity with tab:daemesh.

  python3 build_run_ogrid_L0.py mesh          # (optional) regenerate ogrid L0
  python3 build_run_ogrid_L0.py stage         # stage all six case dirs
  python3 build_run_ogrid_L0.py run <gpu>     # run all six sequentially on <gpu>
"""
import os
import sys
import json
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
MESH_DIR = f"{HERE}/_ogrid_L0_mesh"


def build_mesh():
    os.makedirs(MESH_DIR, exist_ok=True)
    subprocess.run([sys.executable, f"{HERE}/ogrid_wing.py", "0",
                    f"{MESH_DIR}/wing_ogrid_L0"], check=True)
    from rans import mesh as _mesh
    from rans.env import make_env
    env, find = make_env()
    _mesh.gmsh_to_cgns(f"{MESH_DIR}/wing_ogrid_L0.msh", f"{MESH_DIR}/mesh.cgns",
                       find("flow360gmshtocgns"), env)
    print("mesh.cgns:", os.path.getsize(f"{MESH_DIR}/mesh.cgns"), "bytes")


def stage():
    meshes = {"ogrid": f"{MESH_DIR}/mesh_archived.cgns"
              if os.path.exists(f"{MESH_DIR}/mesh_archived.cgns")
              else f"{MESH_DIR}/mesh.cgns",
              "cavity": f"{HERE}/_cavity_L0_mesh/mesh.cgns"}
    for fam, msh in meshes.items():
        if not os.path.exists(msh):
            print(f"SKIP {fam}: no L0 mesh at {msh}")
            continue
        for a in (4, 5, 6):
            src = f"{HERE}/case_{fam}_L1_saai_a{a}"
            dst = f"{HERE}/case_{fam}_L0_saai_a{a}"
            os.makedirs(dst, exist_ok=True)
            for f in ("Flow360.json", "Flow360Mesh.json", "gpubind.sh"):
                p = f"{src}/{f}"
                if os.path.exists(p):
                    shutil.copy(p, f"{dst}/{f}")
            mdst = f"{dst}/mesh.cgns"
            os.path.exists(mdst) and os.remove(mdst)
            os.link(msh, mdst)
            print("staged", dst)


def run(gpu):
    for fam in ("ogrid", "cavity"):
        for a in (4, 5, 6):
            case = f"case_{fam}_L0_saai_a{a}"
            if not os.path.exists(f"{HERE}/{case}/mesh.cgns"):
                continue
            subprocess.run([sys.executable, f"{HERE}/run_solution.py",
                            case, str(gpu), "saai"], cwd=HERE, check=False)
            print(f"L0 {fam} a{a} done", flush=True)
    print("L0-CANON-DONE", flush=True)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == "mesh":
        build_mesh()
    elif mode == "stage":
        stage()
    elif mode == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
