# SA-AI numerics: discrete evaluation of the model's indicators

This document records the discrete schemes behind the SA-AI transition model's
local indicators — everything the paper deliberately does not carry. It covers
the nodal gradient, the dual-mesh geometry, the compact velocity-Laplacian
kernel that evaluates the profile-curvature indicator Γ_g, the sliver-mesh
noise mechanism that motivated the final ring-averaged form, the alternative
evaluations that were tested and rejected, and the validation of the adopted
scheme. Every number below is reproducible from the scripts in
`paper/repro/numerics/`.

Solver source references are to the Flow360 `compute` repository, branch
`favorable-rate` (which extends `three-anchor-kernel`; the favorable-rate
factor is algebraic and does not touch any discrete operator described
here).

---

## 1. The indicators and where derivatives enter

The amplification kernel uses four local mean-flow quantities (paper Sec. II):

| indicator | formula | highest derivative of u |
|---|---|---|
| Re_Ω | d²ω/ν | first (ω) |
| Γ | 2(ωd)²/(\|u\|²+(ωd)²) | first |
| λ_p | −d²(u·∇p)/(ρν\|u\|²) | first (of p) |
| Γ_g | (d²\|∇²u\|)²/(\|u\|²+(ωd)²) | **second** |

Γ_g is the model's only second derivative, and therefore its most
grid-sensitive quantity. The band gate Q = 1 − √(Γ(2−Γ))/(1+c_A·Γ_g) relies on
Γ_g ≪ 1 near the wall (the "pinch" that keeps amplification out of the
sublayer) — a threshold that mesh-scale error in ∇²u can defeat, with the
consequences documented in Sec. 4.

## 2. Nodal gradients (`gradPrimitive`)

All first derivatives come from the solver's nodal gradient
(`MeshProcessor/LeastSquareCoefficients.cpp`, applied per
`PDESolver.h::computeGradient` with the `Gradient` operator of
`StencilOperations/SharedOperators/Gradient.h`):

- **One-ring weighted least squares with a constant column.** For node i with
  edge vectors e_j = x_j − x_i, the fit solves rows
  [1, e_j] · [a, g]ᵀ ≈ u_j − u_i plus a zeroth row [1/w_c, 0] ≈ 0 that
  penalizes the constant a (center weight w_c = 0.1). The gradient
  coefficients are extracted from the QR pseudo-inverse, so
  g_i = Σ_j c_j (u_j − u_i) with precomputed c_j.
- **Per-direction SVD weighting.** The edge matrix is SVD-rotated; for
  gradient direction d the row weights are max(σ_d/√n, |e_j|)^0.5. If the fit
  is ill-conditioned the exponent is relaxed (÷1.8 per retry) and the center
  weight raised (×1.2) until condition-number thresholds are met.
- **Near-wall modified edge vectors.** Within the anisotropic boundary-layer
  region the wall-normal component of each edge vector is replaced by the wall
  distance difference −(d_R − d_L), blended by exp(−d/ℓ_avg)
  (`MeshProcessor/ModifiedEdgeVector.h`).

## 3. Dual-mesh geometry

Median dual (`MeshProcessor/EdgeProcessing.h::processEdge`): for each element
and each of its edges, the dual face is the pair of triangles
(edge-midpoint, cell-centroid, left-face-centroid) and (edge-midpoint,
right-face-centroid, cell-centroid); the dual volume accrues S⃗·e⃗/6 to each
edge node. For the quasi-2D wedge meshes used throughout the paper (one cell
in span, spanwise-uniform solution), the 3D scheme reduces exactly to the 2D
median dual on the triangulation: dual-face normal for edge (i,j) =
Σ_{T∋(i,j)} rot90(centroid_T − midpoint_ij), dual area A_i = Σ_{T∋i} area_T/3.
The reduction is verified to machine precision in
`repro/numerics/replay_gate3_kernel.py` (dual-cell closure ~2×10⁻¹⁵).

## 4. The compact Laplacian (gate mode 3) and the sliver-noise mechanism

### 4.1 Scheme

`SpalartAllmaras.h`, `AI_VG_GATE=3` branch — the viscous-flux pattern:

```
for each edge (i,j) of node i:
    ê   = (x_j − x_i)/L,   S⃗_f = dual-face area (outward)
    ḡ   = ½(g_i + g_j)                       # averaged nodal LSQ gradients
    g_f = ḡ + [ (u_j − u_i)/L − ḡ·ê ] ê      # edge-direction replacement
    lap += (g_f · S⃗_f)
lap /= V_i        →  Γ_g numerator |∇²u|
```

The edge-direction replacement makes the wall-normal second derivative a
nearest-neighbor two-point difference **on meshes that have wall-normal
edges** (structured BL grids), suppressing odd–even modes there.

### 4.2 The mechanism that breaks it on sliver triangulations

Traced end-to-end at cavity-L2 NLF α=4° (x=0.0736, d=3.7×10⁻⁴; scripts
`replay_gate3_kernel.py` + `trace_spike_node.py`, replay validated against the
solver's own ω output to 3 significant digits at every stencil node):

1. **The solution carries a mesh-scale staggering oscillation.** The cavity
   mesher's near-degenerate triangles (aspect 57–165 in the traced stencil)
   organize the BL nodes into staggered tangential rows; the resolved
   tangential velocity carries a row offset of ~0.4–1% of the local u_t (the
   same structural staggering behind the cavity C_f wiggle).
2. **First derivative: ±27% row-alternating error.** A node's one-ring lies
   mostly on the *other* row, so the LSQ gradient reads the row offset ε over
   the row gap Δn ≈ 40×10⁻⁶ as shear: the solver's own ω alternates
   208.9/208.3/193.2 (one row) vs 120.6/117.5/143.2 (the other) around the
   traced node.
3. **Second derivative: O(1) corruption.** The divergence of the *averaged*
   gradients turns the alternating error Δg ≈ 26 across Δn into ~6×10⁵ of
   spurious Laplacian, funneled through the short inter-row diagonal edge that
   carries the stencil's largest dual-face area with no mirror partner to
   cancel it. Result: |∇²u| = 2.1×10⁶ vs ~1×10⁶ physical; Γ_g = 3.13 vs ~0.3;
   deep-band Q = 0.16 where the structured grid reads 0.005.
4. **The general scaling.** A solution wiggle of relative amplitude ε at mesh
   wavelength ℓ has curvature ~εU(2π/ℓ)², overtaking the physical U/δ² when
   ε(2πδ/ℓ)² ≳ 1 — sub-percent ε suffices at δ/ℓ ~ 5–10. Each derivative
   multiplies the wiggle-to-signal ratio by ~2πδ/ℓ, which is why u and ω (and
   hence Re_Ω, Γ, λ_p) stay grid-clean (band maxima agree to 3 digits across
   all six grids) while the one second-derivative indicator corrupts at O(1).
   Refinement shortens ℓ, so the excess *grows* under refinement: the
   anti-convergent cavity-L2 marginal fronts (early by 0.05–0.09c at the
   three-anchor constants, gate mode 3).
5. **Not an operator artifact.** Two independent second-derivative operators
   (the replayed solver kernel and a plain 1D finite difference of the
   resampled wall-normal profile) report the same excess: the curvature is in
   the discrete solution. No consistent stencil evades it.
6. **Two structural notes.** (a) The two-point edge correction is *not* the
   noise carrier on these meshes (0.4% of the traced spike) — sliver
   triangulations have no wall-normal edges for it to act along, so the
   compact-evaluation design intent is silently defeated and the normal
   curvature is assembled from the staggered-row LSQ gradients instead.
   (b) The raw kernel additionally carries a +45% consistency error on a
   smooth quadratic at the traced aspect-100 stencil (LSQ gradient bias +
   non-cancelling sliver dual faces).

### 4.3 Rejected evaluations (replay study, cavL2 a4 solution)

| variant | deep-band Γ_g P99 | band-top Γ_g median | smooth-quadratic error | verdict |
|---|---|---|---|---|
| mode 3 (baseline) | 0.97 | 0.59 | +45% | spikes open the gate |
| drop edge correction | 0.95 | 0.55 | +45% | no effect — correction isn't the carrier; not biased low (45% of nodes increase) |
| edge-only (drop ḡ) | 76 | 3.1 | −20% | catastrophic: loses even *linear* consistency on slivers (Σ(g·ê)(êᵀS⃗) ≠ 0), shear leaks into curvature ×175 |
| one-ring min of \|lap\| | 0.35 | 0.32 | −32% | kills noise but biases smooth fields −30% → would shift the calibration on good meshes |
| **one-ring average of lap vector** | **0.69** | **0.64 (+10%)** | **+6%** | adopted (mode 4) |

## 5. The adopted scheme (gate mode 4): ring-averaged Laplacian

`AI_VG_GATE=4` = mode 3 plus a dual-volume-weighted average of the Laplacian
*vector* over {i} ∪ ring(i) before its magnitude enters Γ_g:

- lap_avg(i) = Σ_{j∈{i}∪ring(i)} lapRaw_j / Σ_j V_j, with lapRaw the
  *undivided* dual-face divergence sums (so the volume-weighted average of
  lap_j = lapRaw_j/V_j reduces to a single ratio).
- On smooth fields the vector average is second-order benign — and in the
  replay it *repairs* the sliver consistency error (+45% → +6% on the traced
  smooth quadratic) because that error also alternates row-to-row.
- On mesh-scale oscillation the alternating contributions cancel: the filter's
  failure mode is **underestimation**, which the gate tolerates (an
  underestimated Γ_g leaves Q pinched and merely withholds amplification —
  conservative, never spuriously early).

Implementation (`compute` commit `34d2166fe7`): the source kernel has only
one-ring access, so a pre-pass kernel (`VelocityLaplacianRaw`,
`SpalartAllmaras.h`) stores lapRaw + V as a 4-component nodal field
(`SATurbulenceSolver::lapVelocityRaw`), communicated across ranks like the
gradients; the source kernel then ring-averages that field — two-ring
information through a one-ring read. The gate is frozen with respect to ν̃, so
the source Jacobian is unchanged.

## 6. Validation

- **Structured identity.** Flat plate Tu=0.16%: χ=1 onset Re_θ 901.6 (mode 4)
  vs 910 (mode 3), −0.9%. Every structured-L2 NLF front matches its mode-3
  value to ≤0.0006c. The calibration (all constants derived on smooth/analytic
  fields, paper Sec. III) carries over unchanged.
- **Restart probe.** cavL2 NLF α=4° restarted from its converged mode-3 state:
  the spuriously early upper front retreats 0.198 → 0.253 within 6,000
  pseudo-steps, onto the structured value 0.251.
- **Fleet (61 cases, three-anchor constants + mode 4, 0 failures).**
  - NLF L2 cavity-vs-structured splits: α=0° lower 0.010c (was 0.087c under
    mode 3), α=4° upper 0.002c (was 0.054c), α=9° upper 0.018c, α=15° lower
    0.008c — every entry within 0.02c.
  - The anti-convergence is gone: cavity L1→L2 drift on the marginal fronts
    is now 0.003c (α=0° lower, was 0.093c) and 0.016c (α=4° upper, was
    0.055c).
  - Family drag agreement at the NLF bucket restored: ΔC_d = 0.6–0.7×10⁻³
    at α=0°/4° (was 0.9–1.2×10⁻³ under mode 3); both families within
    ~5×10⁻⁴ of the digitized experimental polar through the bucket.
  - Eppler 387: the two families' reattachment agrees to 0.001c at L2 on
    all four incidences (α=7°: 0.422/0.423). The α=7° early-reattachment
    miss vs the oil flow (0.42 vs 0.48) persists identically on both
    families — a kernel property (the raised adverse-branch amplification
    of the three-anchor calibration), not mesh noise.
  - Re sweep: recovery locations unchanged (0.71 / 0.52–0.54 / 0.48–0.49);
    at 3×10⁵ and 4.6×10⁵ the computed C_f minimum now grazes zero
    (+1–4×10⁻⁵) instead of barely reversing — the one visible mode-3→4
    physical difference beyond the marginal fronts.
  - Structured results: unchanged throughout (fronts to ≤0.001c at L2,
    flat-plate onsets −0.9% to −1.2%; the AGS band becomes +2/−13%,
    was +3/−12%).

## 7. The calibration instrument (Falkner–Skan marches)

The disturbance transport (paper eq:transport) is solved by an implicit
march in x on each Falkner–Skan similarity field: at each station the
cross-stream operator (reduced diffusion c_ν,ai/σ, wall value 0,
freestream inflow ν̂_∞ = 1) is assembled as a tridiagonal system and
solved directly; the wedge's own λ_p(x, y) enters the kernel (cliff and
rate factor). Grid: nx = 800 × ny = 600 cells, wall-normal extent
y_top = 8 η₉₉ √(x_max/U_e(x_max)); x_max is adapted until the envelope
reaches N ≈ 9.5 (family measures) or ≈ 14 (figure domains). The
coarser 400 × 300 grid is NOT converged; at 800 × 600 the reported
secant slopes are grid-converged to under 2%. These facts were carried
in a paper footnote until 2026-07-13; this document is now their sole
record.

## 8. Provenance of every paper figure

- Analytic figures/tables (paper Figs. 1–7, Tables 1–2): `paper/repro/analytic/`
  — single kernel implementation in `paper/repro/saai_kernel.py`, textual
  consistency with the solver defaults enforced by
  `sa-ai/tests/test_constants_consistency.py`.
- CFD figures/tables: `paper/repro/cfd/` generators reading one case tree
  (`SAAI_CFD_ROOT`); case matrix + convergence protocols in
  `paper/repro/driver/`.
- Numerics diagnostics (this document): `paper/repro/numerics/`.
