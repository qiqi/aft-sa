# The cascade case: what it is, and what Flow360 already supports

Companion to `10-untouched-literature-regimes.md` §1, which proposes a
turbomachinery cascade as the one new campaign worth running. Two questions were
asked (user, 2026-08-05): **is a cascade linear or rotational — and if linear,
i.e. 2D, how does the experiment actually work?** — and **how well does Flow360
support it?** Answers below; all solver/client claims carry a file:line so they
can be re-checked after a merge.

Paths are relative to `/home/qiqi/flexcompute/`.

---

## 1. Linear, and stationary

In the transition-model literature "cascade" means, almost always, a **linear
(rectilinear) cascade**: a straight row of identical blades, **stationary**, in a
wind tunnel. Both canonical cases are of this kind — Zierke & Deutsch (compressor,
double circular arc, Penn State low-speed rig) and the T106 family (LPT). Rotating
rigs and annular cascades exist and are used, but they are a different and much
smaller literature, and none of the standard transition-model validation cases
comes from one.

That matters for us in two ways: the case has **no rotating frame**, so it does
not carry the frame-dependence hazard of `10` §2 and can be run before that
question is settled; and it is genuinely 2D at midspan, so it fits the existing
quasi-2D (`nspan=1`) pipeline.

### 1.1 The geometric idea

Take a real blade row and cut it at one radius. Unroll that annulus into a plane.
What survives is the **blade-to-blade (S1) plane**: a periodic row of identical
sections, described by five numbers —

| symbol | meaning |
|---|---|
| `c` | chord |
| `s` | pitch (blade spacing); `c/s` is the solidity |
| `γ` | stagger angle (chord line vs. the row normal / axial direction) |
| `β₁` | inlet flow angle |
| `β₂` | exit flow angle (an outcome, not an input) |

The unrolling is exact only if the radius does not change through the row, so the
linear cascade reproduces blade-to-blade aerodynamics while deliberately throwing
away radius change, streamline curvature in the meridional plane, and rotation.
That is the point: it isolates the blade-section problem, which is exactly the
problem a transition model is being asked about.

### 1.2 How the experiment is actually run

- **Blade count.** Typically 7–13 identical blades, machined or cast to one
  profile, spanning between two endwalls. More blades than you need, because the
  outermost passages are *not* periodic and are sacrificed.
- **Setting incidence.** The whole cascade is mounted on a **turntable** and
  rotated relative to the tunnel axis. Incidence is swept by turning the entire
  row, not by pitching individual blades — so `β₁` changes while the blade-to-blade
  geometry stays rigid.
- **Enforcing periodicity.** This is the central experimental difficulty. A finite
  row leaks: without help, the flow turns differently in the outer passages and the
  measured passage is not representative. Rigs correct it with **adjustable
  tailboards** downstream (set to steer the exit flow to the intended `β₂`) and
  **upper/lower wall bleed or suction** upstream. Periodicity is then *verified*,
  not assumed: an exit-plane pitchwise traverse must show the loss and `Cp`
  distributions repeating passage to passage. Published cascade data usually
  reports how well this was achieved.
- **Keeping it two-dimensional.** Endwall boundary layers grow and contract the
  passage, so the axial velocity density ratio (AVDR) drifts from 1. Rigs use
  spanwise suction or a converging endwall to hold AVDR ≈ 1 at midspan. A residual
  AVDR ≠ 1 is a known source of CFD-to-experiment `Cp` mismatch and is worth
  reading off the source before matching pressure distributions.
- **Disturbance environment.** An upstream **turbulence grid** sets `Tu`. This is a
  scope decision for us, not a detail: LPT cases are usually run at `Tu` = 0.5–4 %,
  i.e. into the bypass regime the paper declines. Pick a low-`Tu` case or a
  low-`Tu` point of a swept case.
- **Measurements.** Midspan only, treated as 2D. Inlet traverse ~1 chord upstream
  (pitot / hot wire) for `β₁` and `Tu`; blade surface static taps for `Cp`;
  exit traverse ~40 % chord downstream with a three- or five-hole probe on
  pitchwise traverse gear for total-pressure loss and `β₂`; boundary layers by LDV
  or hot wire (Zierke & Deutsch is an LDV data set, which is why it is valuable);
  transition, separation and reattachment locations by surface hot films
  (intermittency), oil flow, liquid crystals, or IR.

### 1.3 Unsteady wake passing, without a rotor

The wake-induced transition literature keeps the cascade linear and stationary and
instead adds a **moving-bar rig**: cylindrical bars carried on a belt or wheel
across the inlet plane, upstream of the blade row, travelling at the tangential
speed that models the rotor. Each bar sheds a wake that sweeps across the passage
at the correct reduced frequency, producing the wake-induced turbulent strip and,
behind it, the **becalmed region**. Cambridge (Stieger & Hodson) and the
Karlsruhe/Stuttgart rigs paired with DNS (Wissink & Rodi) are the references.

For us this is phase two, and it needs URANS. It is also the reason the cascade is
worth running at all — see `10` §1 for why SA-AI has a structural claim to make
there that the algebraic family cannot.

---

## 2. The CFD domain

A single blade passage:

```
        ┌──────── periodic patch B (upper) ────────┐
  inlet │                  ╱▔▔▔▔╲                 │ outlet
   (β₁) │        ╱▔▔▔▔▔▔▔▔╱  blade ╲               │  (p_static)
        └──────── periodic patch A (lower) ────────┘
              A and B offset by exactly the pitch s
```

- **Pitchwise pair (A, B):** translationally periodic, offset by the pitch vector.
  This replaces the airfoil case's circular farfield on two sides.
- **Inlet:** ~1 chord upstream, total pressure + total temperature + flow
  direction `β₁`, plus the SA-AI seed as a turbulence quantity.
- **Outlet:** 1–2 chords downstream, static pressure.
- **Span:** extrude one cell (`nspan=1`) with slip or periodic end faces — exactly
  what the existing pipeline already does for airfoils.

So the delta from an existing Eppler case is: swap `Freestream` on a circular
farfield for `Inflow` + `Outflow` + one `Periodic` pair, and swap the circular
outer boundary in the mesher for a periodic-conforming H-topology.

---

## 3. Flow360 support: verdict

**Supported, end to end, with one meshing constraint and one open model-specific
question.** Every piece the cascade needs already exists and is exercised
somewhere in the tree.

| Need | Status | Evidence |
|---|---|---|
| Translational periodic BC, v2 API | present; multiple pairs in one model | `flow360_schema/models/simulation/models/surface_models.py:768` (`Periodic`, alias `surface_pairs`), `:275` (`Translational`) |
| ... v1 API | present | `compute/src/flex/public/Flow360/flow360/component/v1/boundaries.py:253` |
| Already used in *our* toolchain | yes — the spanwise pair | `flexfoil/rans/rans/case.py:102`, `fl.Periodic(surface_pairs=[(sym1, sym2)], spec=fl.Translational())` |
| Inlet with flow angle + turbulence input | present | `surface_models.py:614` — `Inflow(total_temperature, spec=TotalPressure, velocity_direction, turbulence_quantities)` |
| Outlet static pressure | present | `surface_models.py:571` — `Outflow(spec=Pressure)` |
| SA-AI seed at the inlet | works — same mechanism as the farfield seed | `case.py:41`, `TurbulenceQuantities(modified_viscosity_ratio=χ_∞)`; `Inflow` derives from `BoundaryBaseWithTurbulenceQuantities` |
| Periodicity in the mean-flow solver | implemented | `Flow360Core/Applications/Solver/NavierStokes/NavierStokesSolverWeakBCs.cpp:467` |
| Periodicity for the **transported scalar** (i.e. `ν̃`, i.e. SA-AI) | implemented — `callPeriodicPhiBC` in the advection-diffusion base that SA is built on | `Applications/Solver/AdvectionDiffusionSolver.h:272`; SA includes `SAAdvectionDiffusion.h` at `SpalartAllmaras/SpalartAllmaras.h:6` |
| Mesh-level stitching | periodic faces become interior via a global node-pair map | `Applications/DistributedMeshPartitioner/periodicBoundaryConnectivity.cpp:457`; `Applications/MeshProcessor/PeriodicHelpers.h` |
| An official turbomachinery example | yes — a **stator** blade row with periodic BCs | `flow360/examples/tutorial_periodic_BC.py` (TU Berlin stator mesh) |

### 3.1 The meshing constraint — hard, and worth knowing before starting

Periodicity is implemented by **matching nodes one-to-one**, not by interpolation.
From `periodicBoundaryConnectivity.cpp:366-415`:

- the two patches must have **equal node counts**, or the partitioner throws
  `InvalidPeriodicBoundary` with a node-count message;
- matching tolerance is **local minimum edge length / 20** (`safetyFactor = 20`);
- the **translation vector is auto-detected** from the difference of the two patch
  centroids — an input `translationVector` is only cross-checked and warned about,
  so `fl.Translational()` with no arguments is correct.

Consequence: the two pitchwise boundaries must carry **identical point
distributions**, translated by the pitch. Practical routes, in order of effort:

1. **Structured / gmsh H-grid via `flexfoil/rans/rans/mesh.py`.** The existing
   writer already emits a periodic-conforming pair for the span faces and says so
   (`mesh.py:113`: *"The two END faces keep the same 2D pattern, so a periodic
   pairing is unaffected by the interior spacing"*). The same discipline applied to
   the pitchwise boundaries is straightforward — generate one boundary curve,
   translate it by `s` for the other, and mesh between. **Recommended.**
2. **The unstructured cavity mesher.** Metric-driven, so the two periodic curves
   would need their point distributions forced to match. More work, and the mesh
   independence argument for the cascade could reasonably be made on one family
   plus a refinement ladder rather than on two families, as the airfoil campaign
   did.

### 3.2 The build to run it in

Note carefully: **the `flexfoil/rans` pipeline's Flow360 build carries production
AFT, not SA-AI** (`case.py:44-58` says so explicitly — the SA-AI kernel "is NOT
compiled into this binary"). SA-AI lives in the local `compute` build and is
default-on there, gated by `AI_SA` (`ONBOARDING.md` §6). So the cascade must run
the same way the rest of the paper's campaign runs: build the mesh and
`simulation.json`, then invoke `Flow360Solver` in-session. `flexfoil/rans` is
useful as the source of the `simulation.json` builder and the mesher, not as the
execution path.

### 3.3 Test-coverage gap worth closing first

The local regression tests that exercise periodic BCs all run **laminar**:
`localTests/gravity/poiseuilleFlow/test.py:155`,
`localTests/ConvergingDivergingNozzle/test.py:301`,
`localTests/rotationalPeriodicBC/test.py:61` — all
`turbulence_model_solver=fl.NoneSolver()`. The periodic path for the transported
scalar exists in code (§3, row 7) but is not covered by these tests. So:

> **First step is a smoke test, not the cascade.** A turbulent plane channel or a
> simple periodic duct with `AI_SA=0` (classical SA), checking that `ν̃` is
> continuous across the periodic pair and that the result matches a
> single-domain reference. Then repeat with SA-AI on. Cheap, and it separates
> "the BC works for scalars" from "the kernel does something odd in a passage".

---

## 4. The one model-specific open question: wall distance under periodicity

This is the part that is specific to SA-AI rather than to Flow360, and it should
be settled before any cascade result is believed.

**What the code does.** Wall distance is computed in the mesh processor by an
octree search over the solid-wall triangulation
(`Applications/MeshProcessor/WallDistance.h`, `allocateAndComputeWallDistance`
→ `computeSolidWallConnectivity`); the solver only *smooths* what it is handed
(`Applications/Solver/SolverFunctions.cpp:147`, `preprocessWallDistance`).
`WallDistance.h` contains **no periodic handling** — periodic node maps are built
elsewhere (`PeriodicHelpers`, `periodicBoundaryConnectivity.cpp`) and, on the
reading above, are not consulted by the distance search.

**Why that is a problem here and not for standard SA.** In a single-passage
cascade mesh there is one blade. For a node near a periodic boundary, the
geometrically nearest wall *present in the mesh* may not be the physically nearest
wall — the adjacent blade is the periodic image of the one in the mesh, and its
distance is generally different. Standard SA barely cares: `d` enters the
destruction term, which is negligible away from the wall, and `f_v1` is saturated
there anyway. SA-AI reads `d` **everywhere**, in every indicator:

```
(X, Y, Z) = ( |u|, d|ω|, ½ d²∇²u·û ) ,   Re_Ω = d²ω/ν
```

and `Re_Ω` is the onset gate. A `d` that is too large inflates `Re_Ω` and can open
the gate early; too small holds it shut. This is the same mechanism already
flagged for confinement in `02` §1 — *"where `d` is truncated below the
shear-layer thickness the onset gate can hold a shear layer subcritical that is
not"* — arriving here by a different route.

**Honest magnitude estimate.** Probably second order for a well-posed cascade
passage: the suction-side bubble, which is what we would be measuring, sits deep
inside the passage where the local blade *is* the nearest wall; and the inlet
region, where the discrepancy is largest in absolute terms, is nearly irrotational
so `Ω̂` is small and the kernel is inert regardless. Exposure is concentrated in
the **blade wake** if it convects near a periodic plane, and in any
tip-clearance-like geometry. But "probably second order" is not a result.

**The test, which is cheap.** Dump `d` from a built cascade mesh and compare
against a periodic-aware nearest-wall distance computed offline — a KD-tree over
the blade surface *plus its images at ±s* — and report the field of
`(d_flow360 − d_periodic)/d_periodic`, and the resulting shift in `Re_Ω` relative
to the gate threshold. Same instrument as the Daedalus tip-vortex audit in
`02` §2, so the tooling is shared.

Follow-on question, worth deciding explicitly rather than discovering: if the
discrepancy is not negligible, is the fix a periodic-aware distance in the mesh
processor, or a cascade mesh built with two half-blades so both bounding surfaces
are physically present? The second is a mesh choice we control and needs no
solver change.

---

## 5. Work items, in order

1. **Smoke test** periodic + SA on a turbulent channel, then + SA-AI (§3.3).
2. **Pick the case.** Zierke & Deutsch compressor cascade, low speed, LDV
   boundary-layer data; verify the reported `Tu`, `Re` definition (chord with
   inlet or exit velocity — cascade papers differ), incidence, and AVDR before
   committing (§1.2).
3. **Mesh** an H-topology single passage through `rans/mesh.py` with a
   translated-copy periodic pair, at two or three refinement levels (§3.1).
4. **`simulation.json`**: `Inflow(β₁, TotalPressure, TurbulenceQuantities(
   modified_viscosity_ratio=χ_∞))`, `Outflow(Pressure)`, one `Periodic` for the
   pitchwise pair, span faces slip or periodic (§2, §3).
5. **Wall-distance audit** on the built mesh, before interpreting any solution
   (§4).
6. **Steady result**: `Cp`, loss, and the suction-side transition / separation /
   reattachment locations against the measured set, in the style of
   §`sec:eppval`.
7. **Only then** the moving-bar unsteady phase, which is where the interesting
   claim lives (§1.3, `10` §1).

---

## 6. Verification debt

- All Flow360 claims above were read from source on **2026-08-05** at
  `flow360` 25.10.0b1 / `flow360-schema` 25.11.2b1 and the local `compute`
  checkout; re-check the line numbers after any merge.
- §4 is a **code-reading inference, not a measured defect.** The claim is that
  `WallDistance.h` shows no periodic handling; the consequence for `d` in a
  passage is argued, not demonstrated. Do the test in §4 before repeating the
  claim anywhere else.
- The experimental description in §1.2 is standard cascade practice, from general
  knowledge of these rigs, not from the Zierke & Deutsch report itself. Read the
  source report for the specifics of that rig (blade count, bleed arrangement,
  AVDR, achieved periodicity) before writing anything case-specific.
