# SA-AI reproduction driver

A minimal, clarified driver for re-running the paper's SA-AI transition-model
Flow360 CFD suite. It is a **thin facade** over the proven meshing/solving
modules in `flexfoil/rans/` -- it does not reinvent meshing or solving, it just
pins every run to the canonical SA-AI model from one place and drops the
per-case script clutter.

## Layout

| file          | role |
|---------------|------|
| `saai_env.py` | **single source** of the canonical SA-AI (`VARIANT="ai"`) env; kernel constants imported from `sa-ai/scripts/calibrate_kernel.py` |
| `config.py`   | `CaseConfig` dataclass (geometry / Re / Mach / alpha / Tu or chi_inf / mesh knobs); lowers to `flexfoil.rans` config |
| `case.py`     | build the Flow360 case dir (fresh cavity mesh via the rans pipeline, or clone a pre-meshed base) + patch the chi seed |
| `run.py`      | CLI: build + solve one case in-session (delegates to `rans.solve`), extract Cf / forces / transition location |
| `cases.py`    | the **paper case matrix** (flat-plate Tu sweep, NACA0012 / NLF0416 / Eppler387 alpha sweeps) |

## Canonical env is centralized

Every SA-AI run is pinned to the `VARIANT="ai"` model (the final
compact-Laplacian gate). That configuration lives in exactly one function,
`saai_env.canonical_env()`, and the numeric kernel constants are **imported**
from `sa-ai/scripts/calibrate_kernel.py` (`A_MAX`, `SIGMOID_SLOPE`,
`SIGMOID_CENTER`, `RE_OMEGA_FLOOR`, `K_LAMBDA`) -- not re-typed. Recalibrating
the kernel there flows through automatically. Nothing else in the driver sets
`AI_*` variables.

```
AI_SA=1  AI_VG_GATE=3  AI_RATESCALE=0.19  AI_GCRIT=1.055
AI_SIGMOIDSLOPE=8.0  AI_REOMEGA_FLOOR=290.0  AI_CLIFF_LAMBDA_SLOPE=5.9
+ per-case: AI_CHI_INF (from Tu via the Mack map) and AI_LAMINAR_SLOWDOWN
```

### AI_ / AFT_ naming caveat

The C++ solver and the paper's run scripts read **`AI_CHI_INF`**, but
`flexfoil/rans/rans/case.py` currently reads the legacy **`AFT_CHI_INF`** to
inject the freestream / interior chi seed. Until those are unified, the driver
sets **both** names to the same value (see `saai_env.canonical_env` and the
env export in `case.build`) so the freestream chi is applied no matter which
code path reads it. This is flagged as a known naming split to be unified.

## Running

The Flow360 solver runs **in-session** (no `mpirun`): `rans.solve.run_solver`
sets `OMPI_COMM_WORLD_LOCAL_RANK=0` and launches `Flow360Solver` directly. Run
from a **normal (non-sandboxed) shell** on a GPU box -- the solver is killed
inside a restricted agent sandbox.

```bash
cd sa-ai/repro/driver

# list the paper matrix
python3 run.py list

# build + solve one case on GPU 0
python3 run.py naca0012_Re1M_a4 --gpu 0

# just build the case dir (mesh + Flow360.json), no solve
python3 run.py nlf0416_Re4M_a0 --build-only
```

Run the whole matrix by looping over `run.py list` (assign GPUs as you like);
the paper's `run_vg_all.py` parallelizes 8 GPUs, which you can wrap around
`run_case` if needed. Results (CL, CD, xtr) and output paths are printed and
written to `<case_dir>/saai_result.json`.

## Import from the repo root

`saai_env` / `config` / `cases` import standalone (the `flexfoil.rans` and
Flow360-SDK imports are deferred/guarded so they only fire when you actually
build or solve). Add `flexfoil/rans` and `sa-ai` to `sys.path`:

```bash
python3 -c "import sys; sys.path[:0]=['flexfoil/rans','sa-ai/repro','sa-ai/repro/driver']; \
  from repro.driver import config, saai_env, cases"
```

## TODOs before a production re-run

`cases.py` carries `# TODO` markers where the exact paper values need
confirming: NLF0416 Re/Mach/alpha set, Eppler387 Re/Mach/seed + which base case
dir to clone its mesh from, the flat-plate mesh-donor dir, and the final solver
step budgets.
