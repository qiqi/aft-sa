# Plan — Daedalus tip vortex, and the drag-crisis wake / lift-off study

Two processing campaigns. Both are **CPU-only** — file reads, KD-trees and
gradient reconstruction. No GPU. The machine is shared; crop before loading.

---

## A. Daedalus tip vortex

### The claim under test

The paper reports (§`sec:daedalus`) that *"the tip vortex trips transition to the
leading edge for η ≳ 0.992."* This is presented as a spanwise mechanism no
sectional method sees — a capability claim.

The alternative is that the kernel reads a vortex core the way it reads any
lifted free shear layer: at a core `d|ω| ≫ |u|` so `Ω̂ → 1`,
`Re_Ω = d²ω/ν` is far above the `1851` ceiling so the gate is wide open, and `Î`
saturates on the sign of the curvature projection. That is the same saturated
free-shear reading the cylinder's lifted shear layer gets (§`dragcrisis`) — a
reading the model is entitled to for a *shear layer*, but that has never been
checked for a *vortex core*, which is a different object.

**Decision the analysis must make:** is the χ that trips η > 0.992
(a) generated in the surface boundary layer, or
(b) generated in the vortex core and delivered to the wall by diffusion/convection?

- (a) → the claim stands; close the item.
- (b) → the claim needs qualifying, and it becomes a discussion-section entry
  ("coherent vortices near a surface read as free shear").

Either outcome is publishable characterization. **It touches a claim already in
the paper, which is why this is first.**

### Data status (verified 2026-08-02)

Rsynced to **`/local_data_2/qiqi_daedalus_a5/`** on this machine:

```
volume_proc0.vtu   3.75 GB   36 702 361 points
   fields: rho, p, Mach, solutionTurbulence (= ν̃), velocity, privateNodePosition
chi_surface.npz    wall (124033,3) float32 ; chi (124033,) float64
volume.pvtu, Flow360.json
```

Source: `014-v100-dev:/local_data/qiqi/sa-ai/daedalus/case_ogrid_L2_saai_a5`
(α = 5°, structured O-grid, L2). Also available on 014 but not yet pulled:
`case_cavity_L2_saai_a4`, `case_cavity_L2_saai_a5` — the unstructured family,
needed for the cross-check and for α = 4° (the incidence of the paper's surface
map figure).

**Missing:** wall distance `d` is not a stored field. Reconstruct by KD-tree
against the 124 k surface points.

### Procedure

1. **Crop first.** Stream the VTU and retain only `η > 0.97`, within a few chords
   of the tip. Never materialise 36.7 M points. Target ≲ 2 M points.
2. **Wall distance and normal.** `scipy.spatial.cKDTree` on the surface points →
   `d` and the nearest-wall point; `n̂` from the local surface normal (or from
   `∇d` reconstructed on the cropped cloud — cross-check the two).
3. **Gradients.** Least-squares gradient reconstruction on the cropped cloud for
   `ω = ∇×u` and for `n̂·∇|ω|`. **Use `n̂·∇ω`, not `∇²u`** — that is what the
   repro chain uses and what is consistent with the derivation (file `01-*.md`
   §2). Compute both and report the difference; near a vortex core `δ/r₀` is
   O(1) and they will not agree, which is itself a result.
4. **Kernel.** Evaluate `Ω̂`, `Î`, `Ω̂Î`, `Re_Ω`, the gate and `a·S` with
   `paper/repro/lib/sphere_kernel.py` unmodified.
5. **Attribution.** The decisive diagnostic: integrate `a·S·ω` along streamlines
   entering the η > 0.992 surface layer and compare the banked `N` against the
   `χ` actually arriving. If the arriving `χ` far exceeds what the surface-layer
   path can bank, the source is the core.
6. **Control.** Repeat at η ≈ 0.75 (a clean sectional station, well inboard) —
   where the mechanism should be ordinary boundary-layer amplification — as the
   null case.

### Deliverable — the figure

A three-panel cross-flow plane at a station just aft of the tip:

- **(a)** `|u|` with in-plane streamlines, vortex core marked (λ₂ or Q).
- **(b)** `Ω̂Î` with the `Re_Ω` gate contour overlaid — does the core sit at the
  saturated free-shear reading?
- **(c)** `log₁₀ χ` with the `χ = 1` and `χ = c_v1` contours, showing whether the
  supercritical region is attached to the wall or attached to the core.

Plus a spanwise line plot of the wall `χ = 1` front station vs `η` through the
trip, with the core's `χ` overlaid. If (b) and (c) show the supercritical region
originating in the core and reaching the wall, that is the answer in one image.

This figure goes in the **Daedalus section** if the claim survives, and in the
**discussion** if it does not.

---

## B. Drag-crisis cylinder: wake, lift-off height, free-shear reading

### Why this campaign, and what is new

§`dragcrisis` already *contains* the sweep — 25 steady cases, `Re_D = 1`–`10¹⁰`,
one seed, one closure, four grid families. What it does not yet do is **read the
kernel out of those solutions**. The traverse spans, at a single closure:

| `Re_D` | configuration | what it probes |
|---|---|---|
| 10¹–10³ | broad laminar wake, detached layers amplifying downstream | far-wake reading, no wall nearby |
| 10⁵ | laminar separation at 74°, lifted shear layer over a dead-air wake | **large** lift-off `d/δ_s` |
| 5×10⁵ | transition reaching the separation point | intermediate |
| 2×10⁶ | 1.5° laminar separation bubble at 98–100° | **small** lift-off, wall-anchored |

That is a lift-off-height sweep of orders of magnitude at one closure — a wider
range than any purpose-built wake-over-plate case could give, and it is already
computed. The processing is the missing half.

### Questions

1. **Does `Î` saturate as the layer lifts?** Plot `Ω̂`, `Î`, `Ω̂Î` through the
   separated shear layer at each `Re_D` against `d/δ_s`, where `δ_s` is the local
   shear-layer thickness (from `|ω|/|∇ω|`, or the vorticity-thickness `ΔU/ω_max`).
   Prediction from file `01-*.md`: both saturate for `d/δ_s ≫ 1`, so the reading
   becomes scale-free and the KH ceiling is approached. **Confirm or refute.**
2. **Is the onset gate ever binding in a lifted layer?** `Re_Ω = d²ω/ν` uses the
   wall distance, not `δ_s`, so it over-reads by `(d/δ_s)²`. Tabulate
   `Re_Ω`, `Re_Ω^c`, and the *true* shear-layer Reynolds number `δ_s ΔU/ν` at each
   `Re_D`. The question is whether the gate ever discriminates, or is simply
   always open once the layer separates.
3. **Where does the wake cross `χ = 1`, and what is the handover length there?**
   The paper notes the far-wake front registers from `Re_D ≈ 10²`. Measure the
   distance from the `χ = 1` crossing to free-shear-equilibrium `χ`, the
   free-shear analogue of `analyze_handover_length.py`. This is the number that
   would govern a wake impinging on a downstream surface — the quantity file
   `02-*.md` §1 wants and that the multi-element case would otherwise have to
   supply.
4. **Intermediate `d/δ_s`.** The uncharacterized band (file `02-*.md`) is
   `d/δ_s ~ 2–20`. Find which `Re_D` stations sit there and report the reading.

### Data status

`014-v100-dev:/local_data/qiqi/sa-ai/dragcrisis_matrix` (25 G) — **not yet
inspected**. First step is to check which cases retain volume output; if only
surface/slice data survives, the slices may be enough (the flow is
quasi-2D and `add_derived_to_slice.py` already computes the whole kernel chain on
a slice — this is exactly the tool it was written for).

### Procedure

1. Inventory `dragcrisis_matrix` on 014; identify cases with volume or slice output.
2. Run `paper/repro/cfd/add_derived_to_slice.py` on the centre-span slice of each
   retained case — it already emits `Re_Omega`, `Omega_hat`, `I_hat`, `OmegaI`,
   `sph_X/Y/Z`, `domega_dn`. No new kernel code needed.
3. Add a `δ_s` reconstruction (`ΔU/ω_max` along shear-layer-normal cuts) and the
   `d/δ_s` ratio.
4. Extract along-shear-layer traverses at `Re_D = 10³, 10⁵, 5×10⁵, 2×10⁶`.

### Deliverable — the figure

Two panels:

- **(a)** the collapse test: `Ω̂Î` through the separated shear layer plotted
  against `d/δ_s` for all four `Re_D`. If the curves collapse onto a saturating
  trend approaching 1, the free-shear limit is scale-free as designed — the
  strongest possible answer to "does the kernel need a length scale away from the
  wall?" **This panel is the whole argument in one plot.**
- **(b)** `Re_Ω` (wall-distance based) and `δ_s ΔU/ν` (true shear-layer based)
  versus `Re_D`, with `Re_Ω^c` overlaid — showing by how much the gate over-reads
  and whether it ever binds.

Panel (a) belongs in the paper — it converts a stated design property into a
measured one, on flows already computed and already published.

---

## Sequencing and cost

| step | machine | cost | blocking? |
|---|---|---|---|
| A1–A4 Daedalus crop + kernel | this machine (data local) | ~1 h CPU | no |
| A5–A6 attribution + control | this machine | ~1 h | needs A4 |
| A figure | this machine | ~1 h | needs A5 |
| B1 inventory `dragcrisis_matrix` | 014 (ssh) | minutes | no |
| B2–B4 slice processing | 014 (data is there; 25 G) | ~2 h | needs B1 |
| B figure | either | ~1 h | needs B4 |

A and B are independent and can interleave. **A first** — it touches a live claim
in the paper; B strengthens a section that already stands.

Neither needs a solver run, so neither competes for GPUs. If B1 finds that the
drag-crisis cases kept no field output, B becomes a re-run request and should be
re-scoped before committing GPU time.
