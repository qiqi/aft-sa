# SA-AI project — onboarding for a new agent

Context for working on the **SA-AI** paper and its Flow360 solver port. SA-AI =
*Spalart–Allmaras with Autogenous Inception*: a **single-equation** RANS transition
model. Read this before touching the code; it front-loads the non-obvious decisions
and the mistakes we already made so you don't repeat them.

> Companion docs: `paper/tools/README.md` (experimental-data digitization),
> `paper/annotation-review-log.md` (PDF review rounds),
> `paper/experimental-transition-extraction.md` (NLF transition digitization).
> Per-user memory also exists at `~/.claude/.../memory/` (MEMORY.md index) and may
> be **stale** — this file is the authoritative, in-repo source of truth.

---

## 1. What the model is (one paragraph)

The SA working variable `ν̃`, in its sub-O(1) range (`χ = ν̃/ν ≲ 1`), is repurposed
as an e^N-style amplification factor for laminar instability. The **same** transport
equation handles both laminar TS amplification and turbulent SA via a blended
production `P = max[(1−σ_t)·P_AI, σ_t·P_SA]`, with `σ_t` a handover function of `χ`
and `P_AI = a(Re_Ω, Γ, λ_p)·ω·ν̃` a Drela-style local amplification rate. **No extra
transport equation** (unlike Coder AFT, Langtry–Menter γ–Re_θ, or the algebraic
Bas–Çakmakçıoğlu SA-BCM). Freestream turbulence enters as the freestream seed
`χ_∞ = c_v1·e^−N_crit ≈ 8.76e-4` at `N_crit=9`. Onset delay in favorable pressure
gradient is a cliff `Re_Ω^c(λ_p)`.

---

## 2. Repository map

```
aft-sa/
├── ONBOARDING.md              ← you are here
├── paper/                     ← the LaTeX paper + all figure-generation
│   ├── main.tex               single self-contained source (no \input); ~42 pp
│   ├── new-aiaa.cls/.bst      AIAA journal class
│   ├── build_paper.sh         canonical: regen key figures + 3× pdflatex + bibtex
│   ├── references/            cited PDFs (Coder, Langtry, Mack, Drela, ...)
│   ├── figs/                  all figure PDFs (\graphicspath); figs/verify/ = data checks
│   ├── data/                  digitized experimental + reference data (see §9)
│   │   └── archive/           superseded data files (unreferenced)
│   ├── tools/                 experimental-Cp digitization + verification (see its README)
│   ├── regen_*.py             figure generators (see §5)
│   ├── diag_*.py              one-off boundary-layer / convergence diagnostics
│   ├── annotation-review-log.md   log of every PDF-review round (see §12)
│   └── *.md                   method notes (experimental extraction, etc.)
├── flow360/                   ← Flow360 case dirs + run/reference scripts
│   ├── sweep_Re{60,100,300,460}k_a5/        α=5 Re-sweep, unstructured (cavity) L1
│   ├── sweep_str_Re*k_a5/                    same, structured O-grid L1
│   ├── {cav,str}L{0,1,2}prop_eppler387_Re200k_a{0,2,5,7}/  200k benchmark cases
│   ├── run_xfoil_eppler_sweep.py            XFOIL reference (needs xvfb — see §8)
│   ├── run_mfoil_eppler.py, run_mfoil_*.py  mfoil reference generators
│   └── *.pkl                                 cached mfoil/xfoil reference solutions
└── external/construct2d/      airfoil .dat coordinates + Construct2D structured mesher
```

The **solver source** lives outside this tree, in the Flow360 monorepo:
`/home/qiqi/flexcompute/compute/src/Flow360Core/Applications/Solver/SpalartAllmaras/`
(key files: `ModelConstants.h`, `SAAiTransition.h`, `SpalartAllmaras.h`,
`SATurbulenceSolver.cpp`, `SATurbulenceSolverResidual.cpp`).

---

## 3. Interpreters & environments — READ THIS FIRST

Different tasks need different Python / setup. Using the wrong one is a classic
time-sink here.

| Task | Interpreter / setup |
|------|---------------------|
| **Figure generation** (vtk, matplotlib, numpy; reads Flow360 `.pvtu`) | compute venv `/home/qiqi/flexcompute/compute/.venv/bin/python` (what `build_paper.sh` uses). System `python3` also works in this env. |
| **PDF digitization** (PyMuPDF/`fitz`, PIL, pypdf, pdfminer) | **system `python3`** (has these; the venv may not). |
| **XFOIL** (`/usr/bin/xfoil`) | must run under **`xvfb-run`** — its `CPWR`/`DUMP` writers need an X display (see §8). |
| **LaTeX** | system `pdflatex` + `bibtex`. |

Rule of thumb: **run figure scripts from the `paper/` directory** (many use
relative paths like `data/...`, `figs/...`). The `tools/` scripts are CWD-independent
(they `chdir` to `paper/`); the `regen_*.py` scripts are not — `cd paper` first.

---

## 4. Building the paper

```bash
cd paper
./build_paper.sh          # regen a few figures + 3× pdflatex + bibtex; reports LaTeX errors
```

Or manually after editing `main.tex`:
```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex ... ; pdflatex ...    # 3 total passes to settle refs/citations
```
Sanity checks: `python3 -c "import fitz; print(fitz.open('main.pdf').page_count)"`
and `grep -ci undefined main.log` (want 0 undefined refs/citations). The bibtex
"empty pages / no volume" warnings are pre-existing bib-entry noise, not errors.

To eyeball a figure in context without a PDF viewer: render a page with `fitz`
(`d[pageidx].get_pixmap(matrix=fitz.Matrix(1.6,1.6)).save(...)`) and Read the PNG.

---

## 5. Figures: what generates what, and what is reproducible

**Every `\includegraphics` figure in `main.tex` → its generator.** To re-derive this
map (or find a new figure's generator), grep the basename across the source dirs:
`grep -rl "eppler_polar_compare" paper/*.py flow360/*.py scripts/**/*.py`.

| Figure PDF | Generator (dir/script) | Data needed |
|------------|------------------------|-------------|
| `blasius_nuHat_solution` | `scripts/models/run_blasius_transport.py` | self-contained (ODE) |
| `kernel_maps` | `paper/regen_kernel_maps.py` | self-contained (model) |
| `klambda_profiles` | `paper/regen_klambda.py` | self-contained (model) |
| `shapefactor_amplification` | `paper/regen_shapefactor.py` | self-contained (model) |
| `wall_layer` | `paper/regen_wall_layer.py` | self-contained (model) |
| `indicator_plane` | `paper/regen_indicator_plane.py` | self-contained (model) |
| `eppler_cf_lowalpha`, `eppler_cf_highalpha` | `paper/regen_eppler_v2.py` (`make_cf_figure`; run `python regen_eppler_v2.py {low\|high\|polar\|all}`) | **CFD tree** + mfoil/xfoil pkl |
| `eppler_polar_compare` | `paper/regen_eppler_v2.py` (`make_polar_figure`) | **CFD tree** + mfoil pkl |
| `eppler_L1compare_lowRe`, `_highRe` | `paper/regen_epp_L1compare.py` | **CFD tree** + mfoil/xfoil pkl |
| `mesh_eppler_cav/str`, `mesh_nlf_cav/str` | `paper/regen_mesh_figures.py` (filenames built from `fam_key`×`mesh_key`) | **CFD mesh files** |
| `nlf_cf_lowalpha`, `nlf_cf_highalpha` | `paper/regen_nlf_v2.py` | **CFD tree** + mfoil pkl |
| `nlf_polar_compare` | `paper/regen_nlf_polar.py` | **CFD tree** + mfoil pkl |
| `flat_plate_batch_flow360` | `paper/regen_flatplate_flow360.py` | **CFD tree** |

Tables: `tab:eppxtr` (reattachment) ← `paper/regen_epp_reattach.py`; `y+`/mesh ladders
← `paper/regen_yplus_table.py` and hand-entered mesh stats. `build_paper.sh` additionally
runs `regen_naca.py`, `regen_final.py`, and two `flow360/plot_*.py` mesh/grid scripts.

### ⚠️ Reproducibility — read before assuming "just re-run it"

1. **`*.pdf` is git-ignored** (`.gitignore`). Figure PDFs are only in the repo if
   force-added (`git add -f`). If a clone is missing figures, the paper won't compile —
   force-add the 19 used PDFs (they are committed as of the onboarding commit).
2. **6 figures are self-contained** ("self-contained" above) — a fresh clone can
   regenerate them from repo code alone.
3. **13 figures need the Flow360 case tree** (`flow360/sweep_*`, `*prop_*`, `.pvtu`
   slices) — that tree is **~124 GB and is NOT in git**. From a clone you can rebuild
   the *plot* (the scripts + the mfoil/xfoil reference `.pkl` are committed) **only if
   the CFD case data is present on disk**. Otherwise the committed PDF is the artifact.
   Re-running the CFD itself means re-meshing (§10) and re-solving (§6) each case.
4. So: **the paper builds from a clone** (all figure PDFs committed); **model figures
   fully reproduce**; **CFD figures reproduce only on a machine that still holds the
   case tree.** Don't delete `flow360/sweep_*`/`*prop_*` expecting to `git checkout` them.

`regen_eppler_v2.py` holds the shared machinery the CFD-figure scripts import:
`airfoil_walk_contour` (surface Cp/Cf from `.pvtu`), `max_chi_vs_x` (near-wall max χ;
needs `slice_centerSpan.pvtu` with `nuHat`+`wallDistance`), and the plotting constants
(`CHI_INF`, `C_V1=7.1`, `N_LO/N_HI`, `NU`). **Key convention:** `χ = nuHat/NU`,
`NU = M/Re = 0.1/Re` (per-Re in the sweep: `NU = 0.1/(Rk*1000)`).

`v2`/`v3`/`_final` script suffixes are historical iterations — the latest-numbered is
usually authoritative, but confirm which one actually writes your figure (grep).

---

## 6. The Flow360 solver — runtime knobs

SA-AI is **default-ON** in this branch. Gate (`SATurbulenceSolverResidual.cpp`):
```cpp
const char* aiEnv = getenv("AI_SA");
const bool aiSaTransition = (aiEnv == nullptr || atoi(aiEnv) != 0);  // ON unless AI_SA=0
```
So **`AI_SA=0` → classical fully-turbulent SA** (baseline comparisons); anything else
(or unset) → SA-AI. All other knobs are read in `SATurbulenceSolver.cpp` via
`envD("AI_...", <default from ModelConstants.h>)`:

| Env var | Default | Meaning |
|---------|---------|---------|
| `AI_SA` | on | `0` = classical SA |
| `AI_LAMINAR_SLOWDOWN` | 1.0 (off) | `fSlow`: pseudo-time slowdown of laminar `ν̃`; **0.01** damps the natural-transition limit cycle. See §7. |
| `AI_SIGMA_FPG` | 0.0 (off) | enable favorable-PG rate suppression `σ_FPG(λ_p)`. Verified **no-op** on adverse-PG cases (σ_FPG→1); the cliff carries FPG. |
| `AI_LAMBDA_STAR` / `AI_LAMBDA_SLOPE` | 0.64 / 4.56 | σ_FPG sigmoid center/slope |
| `AI_CLIFF_LAMBDA_SLOPE` | — | FPG onset-delay cliff `Re_Ω^c = floor·exp(K_λ·max(0,λ_p))` |
| `AI_REOMEGA_FLOOR`,`AI_RATESCALE`,`AI_GCRIT`,`AI_BARRIER_M`,`AI_MAXBLEND`, … | see `ModelConstants.h` | kernel calibration; **the paper's calibration = the defaults**, don't change without re-validating. |

> ⚠️ **Memory may say `AFT_SA`** — that was renamed. Everything is `AI_*` now, and
> SA-AI is default-on (it used to be opt-in). Trust the source, not old memory.

### Build consistency (non-negotiable)
After editing any layout-affecting solver header, **clean-rebuild + install**:
```bash
cmake --build build/release --target install --clean-first
```
Incremental builds silently produce ODR-mismatched binaries → ~1 GB/s memory
"leak"/OOM (this actually happened; it was a stack overflow from a corrupted tree,
not a heap leak). `ccache` makes clean rebuilds fast. RUNPATH means
`install/release/lib` is what runs — installing is mandatory, not optional.

### Running a case in-session
The GPU solver runs in-session: set `OMPI_COMM_WORLD_LOCAL_RANK=0` and invoke
`Flow360Solver` directly (no `mpirun`). Every SA-AI run/restart/diagnostic runs with
SA-AI on by default now — but if you script `make_env`-style launches, double-check
`AI_SA` isn't being forced to 0.

---

## 7. Convergence & the natural-transition limit cycle — ALWAYS CHECK

**Before drawing any physical conclusion from a solution, verify convergence.**
Check the residual trajectory AND the force-coefficient trajectory:
```python
# nonlinear_residual_v2.csv (cols: physical_step, pseudo_step, 0_cont..5_nuHat)
# total_forces_v2.csv (cols incl. CL, CD)  -- trailing comma → strip empty cells
```
A **converged steady** LSB case has CL/CD flat to `Δ < 1e-8` over the last ~2000
iterations (no limit cycle). Report unconverged runs prominently; default to
suspicion on marginal transition / low-`χ_∞` / unstructured cases.

The natural-transition front can migrate ~100× slower than the residual suggests,
so the residual alone can look "converged" while the transition front is still
moving. Mitigations: the **staged-fSlow protocol** — run `AI_LAMINAR_SLOWDOWN=1`
to migrate the front fast, then `=0.01` to damp the limit cycle. When you lower
fSlow, the in-domain seed rescales: `χ_∞(in-domain) = BC_input/fSlow`
(so fSlow=0.01 needs BC input `8.76e-6` to keep `χ_∞=8.76e-4`).

---

## 8. Reference solvers: mfoil & XFOIL

Both are e^9 viscous-inviscid panel references, run on the **same** `eppler387.dat`
contour as the CFD (`external/construct2d/eppler387.dat`, `M=0.1`, `Ncrit=9`).

- **mfoil** (Python, `run_mfoil_eppler.py`): gives Cp, Cf, **and the N envelope**.
  Cached in `flow360/mfoil_eppler387_*.pkl`. **Unreliable near stall**: at Re=200k
  α=7° it stalls numerically (Cl=0.928 < its α=5 value). The α=5 Re-sweep pkl
  (`mfoil_eppler387_sweep_a5.pkl`) has **60/100/200/300k only** — it diverges at 460k
  (thin high-Re bubble, "stagpoint_move").
- **XFOIL** (`/usr/bin/xfoil`, `run_xfoil_eppler_sweep.py`): used where mfoil fails
  (Re=200k α=7, and Re=460k sweep). Gives Cp + Cf but **no N envelope**. Cached in
  `xfoil_eppler387_*.pkl`.

Figures therefore substitute XFOIL for mfoil at those points and simply omit the
reference-N curve there (documented in the Fig. 16 / Fig. 19 captions).

### XFOIL headless recipe (this cost us real time — do it exactly)
XFOIL's `CPWR`/`DUMP` file writers call the plot library, so they need a display:
- Run under **`xvfb-run -a xfoil`** with graphics **on** (do NOT `PLOP`/`G F` it off —
  that makes `CPWR`/`DUMP` silently write nothing; only `PACC` works without a display).
- Drive it via a **Python `subprocess` `input=` pipe**, NOT `xfoil < file` (the `<`
  redirect trips a floating-point exception on EOF in this build).
- `PLOP` **before** `LOAD` crashes (FPE) — load the airfoil first.
- Keep the OPER-menu commands contiguous: a blank line after `ALFA` exits OPER, so a
  following `CPWR` lands at the top menu and no file is written.

`run_xfoil_eppler_sweep.py` already encodes all of this — copy its pattern.

---

## 9. Experimental & reference data (`paper/data/`)

- **`exp_cp_tables.json`** — the authoritative digitized experimental Cp (NASA
  TM-4062 Appendix D). Structure:
  ```
  { "<Re>": { "upper": { "xc":[29], "<alpha_col>":[29 Cp], ... }, "lower": {...} } }
  ```
  Coverage: 200k `{-0.01, 0.01, 2.04, 5.05, 7.01}`; 60/100/300/460k the nearest-5°
  column (`4.99/5.01/5.00/5.01`). Every value is verified against the scan
  (`figs/verify/*.png`). Nearest-5° map used by the sweep figure:
  `{60:"4.99", 100:"5.01", 200:"5.05", 300:"5.00", 460:"5.01"}`.
- `*.csv`, `mfoil_*.csv/.pkl` — reference solver outputs / flat-plate & NACA data.
- `data/archive/` — superseded files (`exp_cp_200k.json`, `exp_cp_alpha5.json`),
  kept only so old runs are traceable; nothing references them.

To add/extend experimental Cp, follow **`paper/tools/README.md`** (source = the
tables, not the plots; two orientations; digit-for-digit verification).

---

## 10. Mesh generation practices (non-negotiable for grid studies)

For any airfoil grid-convergence study:
- **One** smooth-spline contour (the `flow360/{naca,eppler,nlf}_contour.py`
  generators) shared by BOTH the unstructured cavity mesher AND Construct2D
  (`external/construct2d/`) — never two different contours.
- **All** length scales refine simultaneously (LE/TE spacing, wall-normal, TE growth)
  on a **fixed far-field** (100c for Eppler). A "coarser-surface but finer-elsewhere"
  grid is not a valid coarser level.
The June-2026 NLF mishap came from non-matching far-fields + a mislabeled "L2".
Eppler blunt TE: `z_te_half = 0.000833c` (blended from x/c=0.95), half the NLF value.

---

## 11. Validation cases (what the paper rests on)

- **Flat plate** — laminar→turbulent transition; FD-verified Jacobian.
- **NACA 0012** quasi-2D, Re=1e6 — transition ~x/c≈0.45, ~34% lower drag vs turbulent SA.
- **NLF(1)-0416** Re=4e6 — cambered NLF; transition table (`experimental-transition-extraction.md`).
- **Eppler 387** — the adverse-PG / LSB workhorse: Re=2e5 benchmark (α=0,2,5,7, three
  mesh levels × two families) + an α=5 Re-sweep 6e4→4.6e5. This is where most recent
  work happened.

---

## 12. Pitfalls we already hit — don't repeat these

1. **`AI_SA` must be effectively on.** A whole "sigma_FPG regression / drift-to-turbulent"
   false alarm was just re-run scripts omitting the flag → silent plain-turbulent SA.
   (It's default-on now, but verify nothing forces `AI_SA=0`.)
2. **Never conclude from an unconverged run.** See §7. Forces flat to 1e-8, not just
   a settled residual.
3. **Digitize from the tables, not the plots**, and verify every digit (§9, tools/README).
   Our first eye-read Cp was wrong; OCR failed entirely on the scan.
4. **The 200k Appendix-D pages are transposed** vs the other Reynolds numbers — wrong
   verifier tool → garbage crop.
5. **XFOIL headless** needs `xvfb` + pipe (not `<`) + graphics-on + contiguous OPER
   commands (§8).
6. **mfoil is unreliable at α=7 and diverges at Re=460k** — use XFOIL, drop the N curve.
7. **Solver: clean-rebuild + install** after header edits, or you get ODR-mismatch
   OOM (§6).
8. **`-Cp` axes**: bottom limit is always `-1` (Cp ≤ +1 at stagnation); don't let it
   auto-scale below.
9. **Memory can be stale** (esp. `AFT_*`→`AI_*`, opt-in→default-on). Verify against
   source before acting on a recalled fact.

---

## 13. Paper-review workflow

When addressing PDF review comments (`annotated.pdf`), **always append the verbatim
comments + your resolutions to `paper/annotation-review-log.md`** — the annotated PDF
gets overwritten each round, so the log is the only history. Extract annotations with
system `python3`: pypdf for `/Contents`, pdfminer for `QuadPoints` (the highlighted
text under each note).

---

## 14. Working conventions

- **Commit/push only when asked.** End commit messages with
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Do **not** fabricate experimental data — trace every number to a source.
- The repo root is a git repo (`aft-sa/`); many `paper/*.py` are currently untracked.
