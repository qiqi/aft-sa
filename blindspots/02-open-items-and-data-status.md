# Blindspots — open characterization items, with data/feasibility status

Companion to `01-parabolic-profiles-pipe-channel.md`. These are the items that
survived an audit against what the paper already contains (see §0). Each carries
its data status as checked **2026-08-02**.

---

## 0. Audit: what is already characterized in the paper

Recorded so these are not re-proposed. All of the following were on a first-pass
blindspot list and are **already done**, in most cases better than a purpose-built
case would have done:

| Item | Where | Note |
|---|---|---|
| Free-shear / lifted-layer handover length | §`epphandover`, `repro/cfd/analyze_handover_length.py` | Quantified: half-saturation 0.025–0.05 c at Re=2–4.6e5, 0.05–0.10 c at 1e5, never completes at 6e4 |
| Lift-off-height sweep (d/δ_shear) | §`dragcrisis` | The 10-decade traverse spans dead-air wake at 1e5 through a 1.5° bubble at 2e6 on one closure — a wider sweep than any contrived wake-over-plate case |
| Wake amplification | §`dragcrisis` | Far-wake χ=1 front registers from Re_D≈1e2; detached shear layers amplifying at 1e3. Consequence entangled with the (stated) steady-symmetric-wake concession |
| Recirculation residence / re-seeding | §`epphandover`, `repro/cfd/measure_lsb_reseeding.py` | Reversed-region χ is 21–27× **above** the incoming envelope at Re≤1e5, flipping to 27–170× below at 3–4.6e5; mechanism identified as transport topology, not rate |
| Farfield seed integrity | campaign-wide | Stagnation-point χ agrees with farfield; settled, not an issue |
| Monotone N (no negative rate) | — | Shared with Coder's AFT and with e^N itself. Class property, not a defect |

**Correction on record:** the NLF L0 front/drag scatter (e.g. α=−8°, x_tr 0.195→0.538 L0→L1)
is a boundary-layer *resolution* artifact — the layer leaving the suction peak is
too thick before any turbulence model acts, which is why it stalls early. It is
**not** evidence for kernel conditioning and should not be cited as such.

---

## 1. Confinement / multi-element — no case in the campaign has a second wall

**Why it is the highest-value open case.** Every computed case is external and
unbounded. Confinement is the one regime where `d` being the wrong length
*suppresses* rather than saturating harmlessly: where `d` is truncated below the
shear-layer thickness (slat gap, tip clearance, duct), `Re_Ω = d²ω/ν` under-reads
and the onset gate can hold a shear layer subcritical that is not. A
multi-element geometry additionally delivers, in one case:

- the confinement/gap regime (above)
- the slat wake convecting over the main element — the wake-on-downstream-surface
  question, with the wake's χ arriving inside the same variable the main
  element's boundary layer uses as its amplification factor
- the confluent boundary layer (wake merging with the flap layer, two shear
  layers sharing one `d`)
- the slat cove — a long-residence-time closed recirculation, the general case of
  the LSB re-seeding already quantified at §`epphandover`
- the medial axis between elements, where the nearest-wall point jumps and `∇d`
  is discontinuous

**Geometry status: NOT in the repo.** `data/` holds only
`naca0006/0012/2412.dat` and `nlf0416.dat`. No multi-element coordinates.

**Recommended source:** the McDonnell-Douglas **30P30N** three-element landing
configuration (slat and flap both deflected 30°), tested at NASA Langley LTPT —
the standard multi-element validation case, with published coordinates and LTPT
pressure/force data. Confirm whether the LTPT campaign carries transition
measurements before committing; if not, the comparison target is XFOIL/MSES
strip `e^N` per element plus the measured Cp, in the same style as the Daedalus
AVL+XFOIL reference.

**Meshing.** The structured O-grid family does not carry over. Unstructured only.
The metric should be the **intersection of the per-element metrics**, each
element sized as if it were an isolated airfoil — i.e. build the airfoil metric
(wall spacing, chordwise clustering, LE/TE refinement) independently for slat,
main and flap and take the pointwise intersection (finest requirement wins) so
that gaps and coves inherit the finer of the two neighbouring elements'
requirements. This also automatically refines the medial axis region, which is
where the kernel's `d` discontinuity lives — useful, since we want that region
resolved rather than smeared.

**Cost note:** three elements at Eppler-class Reynolds number, unstructured, at
two or three levels. Not cheap but not a campaign — one incidence at L1/L2 would
already answer the confinement and wake-transport questions.

---

## 2. Daedalus tip vortex — FEASIBLE, data exists on 014-v100-dev

The paper reports "the tip vortex trips transition to the leading edge for
η ≳ 0.992" as a captured spanwise mechanism. The open question is whether that is
physics or the kernel reading a vortex core as a free shear layer: at a vortex
core `d|ω| ≫ |u|` so `Ω̂ → 1`, `Re_Ω = d²ω/ν` is far above the 1851 ceiling, and
`Î` saturates on the sign of `∇²u·û` — the same saturated free-shear reading the
cylinder's lifted shear layer gets.

**Data status — checked by ssh 2026-08-02:**

- `/local_data/qiqi/sa-ai/daedalus` = 54 G on **014-v100-dev** (not on 019; the
  local `daedalus_fv1` symlink here is dangling).
- Most cases retain **surface only** (`chi_surface.npz` = wall coords + χ,
  `surface_fluid_wing_proc0.vtu`). Not sufficient.
- **Volume output exists for three cases:**
  `case_ogrid_L2_saai_a5`, `case_cavity_L2_saai_a4`, `case_cavity_L2_saai_a5`
  (`volume_proc0.vtu`, 3.75 GB, 36.7 M points).
- Fields present: `rho, p, Mach, solutionTurbulence (=ν̃), velocity,
  privateNodePosition`. Enough to form χ, ω = ∇×u, and ∇²u = −∇×ω.

**What is missing:** wall distance `d` is not in the file. Reconstruct it by
KD-tree against the 124 k wall points in `chi_surface.npz` — cheap if the volume
is cropped to the tip region first.

**Plan (CPU only, no GPU — mind the shared machine):**
1. On 014, stream `volume_proc0.vtu` for `case_ogrid_L2_saai_a5` and crop to
   η > 0.97, a few chords around the tip. Do **not** load 36.7 M points at once.
2. KD-tree `d` and nearest-wall point against the surface mesh.
3. Finite-difference/least-squares `ω` and `∇²u` on the cropped cloud.
4. Evaluate `Ω̂`, `Î`, `Ω̂Î`, `Re_Ω`, gate, and `a·S` through the vortex core and
   through the boundary layer just inboard of the trip, using
   `paper/repro/lib/sphere_kernel.py`.
5. Decide: is the χ that trips η > 0.992 (a) generated in the surface layer, or
   (b) generated in the vortex core and diffused/convected to the wall?

**Answer (b) would mean the reported spanwise mechanism is a kernel artifact and
the claim needs qualifying.** Answer (a) leaves the claim intact and closes the
item. Either outcome is publishable characterization.

Note α=4° (the incidence the paper's surface-map figure uses) has volume output
only on the **cavity** family; the ogrid volume is at α=5°. Fine for this
question — do both and cross-check.

---

## 3. Non-ideal walls — the `u'''_w = 0` anchor

The construction that holds amplification off the no-slip line is
`Î → 0` at the wall, which rests on the **steady impermeable** wall
compatibility condition `u'''_w = 0` (paper §`sec:live`, citing White).
Two deployment-relevant cases break it:

- **Transpiration** (suction/blowing, HLFC, porous walls, film cooling).
  At the wall `v_w u'_w = −(1/ρ)dp/dx + ν u''_w`; differentiating in y with
  `∂_y v|_w = −∂_x u|_w = 0` gives

  ```
  u'''_w = (v_w/ν) u''_w   ≠ 0
  ```

  so `Î` departs zero **at** the wall. With suction (`v_w < 0`) under an adverse
  gradient (`u''_w > 0`) this gives `I = −∫ ½ s² u''' ds > 0` immediately —
  positive `Î` at the no-slip line, which the paper states never happens.
  The `Re_Ω` gate is a partial backstop (`Re_Ω → 0` as `d → 0`), so this is
  probably not catastrophic, but the band placement under transpiration is
  untested and HLFC is a first-tier NLF application.

- **Deforming / accelerating surfaces** (pitching, plunging, aeroelastic).
  `ν u''_w = ∂_t u_w + (1/ρ)dp/dx` ⇒ `ν u'''_w = ∂_t u'_w ≠ 0`. A rigidly
  translating wall in its own frame is fine; a *deforming* one is not.

Both are statements, derivable, no case needed. They belong in the limitations
section.

---

## 4. Roughness — a seed-level boundary condition compatible with SA's rough-wall extension

### 4.1 Why the existing SA roughness extension is the wrong tool here

Aupoix & Spalart (2003) (Boeing and ONERA variants, equivalent-sand-grain `k_s`)
work by, in structure:

```
d      →  d + d0 ,          d0 = 0.03 k_s
wall BC:  ∂ν̃/∂n = ν̃ / d0        (Robin — ν̃ extrapolates to zero at depth d0)
χ      →  χ + c_R1 k_s/(d + d0)   in f_v1   (c_R1 ≈ 0.5)
```

*(constants above are from recollection — verify against Aupoix & Spalart 2003,
IJHFF 24(4):454, and the NASA TMR SA page before use.)*

The Robin condition plants a finite ν̃ at the wall of order `κ u_τ k_s`, i.e.

```
χ_wall ~ κ k_s⁺
```

For a genuine trip (`k_s⁺ ≳ 70`) that is `χ_wall ~ 30` — already past `c_v1`.
**This is exactly right for a trip and useless for gentle roughness**: it does
not raise the disturbance level, it installs fully-developed turbulence at the
wall. That matches the "brutal" characterization.

### 4.2 What SA-AI actually needs

SA-AI's distinguishing property is that the sub-`O(1)` range of χ *is* the
amplitude ladder. Gentle roughness physically acts on **receptivity** — it raises
the initial disturbance amplitude, equivalently lowers the remaining N budget —
and does not itself create turbulence. In this model that is the same object as
the freestream seed, delivered at the wall instead of the farfield:

```
N_remaining = ln( c_v1 / χ_wall,rough )      instead of      ln( c_v1 / χ_∞ )
```

So the natural construction is **the same Robin boundary condition, with the
imposed wall level held below c_v1** — a *local seed injection*. Because
`f_v1(χ) ≈ 2.8e-3 χ³`, a wall value of χ ~ O(0.1–1) contributes negligible eddy
viscosity (paper §`sec:model`) while consuming real e-folds of budget. No new PDE
term, no new field: **one boundary condition and one input (`k_s`)**, which fits
the "stop adding stuff" constraint.

### 4.3 A compatible interpolation

Requirements:
1. `k_s → 0` must recover `ν̃_wall = 0` exactly (smooth wall untouched).
2. Small `k_s⁺` must act as a seed: `χ_wall ≪ c_v1`, no eddy viscosity.
3. Large `k_s⁺` must saturate onto the Aupoix–Spalart value, so the model is
   continuous with the existing rough-wall extension and reproduces a trip.

The single anchor for where (2) becomes (3) is already standard: the **Braslow
trip criterion**, `Re_kk = u_k k/ν ≈ 600`, the roughness Reynolds number at which
a discrete element trips immediately. Anchoring the saturation there would follow
the paper's own style — one physically-anchored constant, not a fit.

For the seed branch, the calibration target is the `e^N` roughness literature's
`ΔN(k/δ₁)` correlations: distributed roughness of relative height `k/δ₁` costs a
known number of `N`-units, which maps directly onto `χ_wall = c_v1 e^{−(N_crit − ΔN)}`.

### 4.4 A real interaction to check first

Aupoix–Spalart shifts the **wall distance** itself, `d → d + d0`. The SA-AI
indicators are *built on* `d`:

```
(X, Y, Z) = ( |u|, d|ω|, ½ d² ∇²u·û ) ,     Re_Ω = d²ω/ν
```

so adopting the AS origin shift is **not neutral for the transition kernel** —
it changes `Î`, `Ω̂` and `Re_Ω` even at roughness heights where it should be
nearly inert. Before designing anything, decide (and document) whether the
kernel consumes `d` or `d + d0`. My reading is that the kernel should keep the
geometric `d`, and only SA's destruction/`f_v1` machinery should see `d + d0` —
but that is a choice the current formulation does not make explicitly.

### 4.5 Scope statement that comes with it

A wall BC can only model roughness as **pure receptivity**. Real roughness also
alters the mean profile and hence the growth *rate* (and, for large elements,
generates its own wake instability). A seed-level BC will not capture that, and
the model should say so.

---

## 5. Remaining items — statements, not campaigns

- **Görtler / centrifugal instability.** Wall curvature radius enters no sensor;
  concave surfaces will be predicted too laminar.
- **Wall temperature.** `Î` is the incompressible inflection, not Lees–Lin's
  generalized inflection; wall cooling/heating is invisible.
- **Attachment-line transition.** No concept of it — and standard SA's spurious
  attachment-anchored branch (§`sec:bistability`) mimics it, so genuine
  contamination and the numerical artifact are not distinguishable under the
  current protocol. Relevant to any swept-wing deployment.
- **Relaminarization.** No quench term; once handed over, never handed back.
- **`Î` numerical conditioning.** `Î` is third-order small near the wall but is
  formed as a difference of `O(1)` quantities, so relative error grows like
  `d⁻³` as `d → 0`. This is a *plausible* concern with **no supporting evidence**
  (see the L0 correction in §0). If it is worth testing, the test is direct:
  evaluate `Î` on an analytic Falkner–Skan profile at coarsening wall-normal
  spacing and find where the sign goes bad. Cheap, and it either produces a
  resolution guideline or removes the item.
