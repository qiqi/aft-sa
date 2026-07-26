# OpenFOAM on the SA-AI unstructured (cavity) meshes: two failed attempts

*2026-07-26. Context: cross-solver replication of the SA-AI paper cases in
OpenFOAM v2412 (`sa-ai/openfoam/`, cell-centered, pressure-based simpleFoam).
The structured (Construct2D O-grid) family works: identical-mesh NLF strL0
α=0 transition agrees with Flow360 within 2.5–4% x/c. The unstructured
"cavity" family does not run at all. Both attempts documented here so nobody
repeats them.*

## Attempt 1: the primal cavity mesh (nlfprop cavL0, 27,787 tri-prism cells)

Converted exactly from the committed gmsh source (`gmshToFoam mesh.msh`,
span planes → `empty`, true 2D). `checkMesh` passes the default checks, but
`-allGeometry` reveals the pathology of the anisotropic sliver-triangle
boundary layer:

- max non-orthogonality **89.8°** (avg 47.9°)
- **6,598 cells (24%) under-determined** (cell determinant down to 1e-6)
- face interpolation weights down to **0.002**; 500× volume jumps across faces

Escalation ladder, all fatal by iteration 14–49 (FPE, or pressure solve with
*rising* residual over 1000+ sweeps):

1. SIMPLEC, linearUpwind, `limited corrected 0.5` — Ux singularity, FPE @14
2. SIMPLE (p 0.3 / U 0.7), `limited corrected 0.33`, 3–4 non-orth correctors,
   PCG/DIC instead of GAMG, upwind divs, potentialFoam init — FPE @49
3. Fully `uncorrected` Laplacians (unconditionally stable M-matrix) — FPE @21
4. smoothSolver/symGaussSeidel instead of DILU (the FPE site was
   `DILUPreconditioner::calcReciprocalD`, a zero matrix diagonal) — no help

**Control experiment: stock `SpalartAllmaras` crashes identically** (23 vs 20
iterations). The failure is mesh × discretization, not the SA-AI model.

## Attempt 2: the median dual of the same triangulation

Rationale: Flow360 is node-centered/median-dual, so "the same mesh" for it
arguably *means* the dual control volumes; OpenFOAM handles polyhedral cells
natively. A converter (`sa-ai/openfoam/scripts/dual_mesh.py`) was built and
verified exact: area conservation to machine zero, 14,234 dual cells = 2D
node count, closed topology, all volumes positive.

Result: the dual of a sliver triangulation is *worse* for a compact
cell-centered scheme:

- max non-orthogonality **120°** — 384 faces whose owner–neighbour vector
  *opposes* the face normal (1,411 misoriented face pyramids)
- skewness up to 881; interpolation weights down to 1.2e-5
- worst cells: ribbon-shaped duals around the leading-edge BL nodes

Same scheme ladder: FPE at iteration 99–132, or (with everything first-order
and uncorrected) a "stable" run whose continuity error grows ×2.5 per
iteration. Not scheme-fixable.

## Conclusion

Flow360's robustness on the cavity meshes does **not** come from the shape of
its control volumes — the dual CVs are geometrically awful too. It comes from
the node-centered **operators**: least-squares gradients and dual-face flux
assembly that never reduce to a compact owner–neighbour difference across a
degenerate face. OpenFOAM's cell-centered compact-stencil FV cannot form a
convergent discretization on either the primal or the dual of this
triangulation. The cavity family is therefore out of scope for the OpenFOAM
cross-solver study; the structured family carries it. (This is itself a
useful robustness statement for the paper: the mesh family exercises exactly
the operator differences between the two solver classes.)

## Can snappyHexMesh substitute as OpenFOAM's native unstructured route?

Not directly for 2D. snappyHexMesh is inherently 3D (castellated
refine–snap–layer on a 3D background hex mesh): run on a one-cell-thick slab
it refines and snaps in the span direction too, destroying the one-cell 2D
structure that `empty` patches require. Known workarounds (mesh a thick slab,
slice/extrude a mid-plane via `extrudeMesh`) produce mediocre boundary-layer
quality for wall-resolved transition work. If an OpenFOAM-native unstructured
2D mesh is wanted for a follow-up robustness test, the right tools are:

- **cfMesh's `cartesian2DMesh`** (quad-dominant true 2D with BL layers; ships
  as an OpenFOAM module but was not built in our tree — a small extra build), or
- **gmsh in quad/BL mode** on the same contour (then `gmshToFoam`, as now).

Either produces well-conditioned quad-dominant BL meshes — a different mesh
from the paper's cavity family, so it would test "OpenFOAM on its own
unstructured mesh," not "same mesh as Flow360." Worth doing only if the
structured-family cross-solver results leave an unstructured-specific
question open.
