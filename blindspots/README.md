# SA-AI blindspot characterization

Working notes on the model's operating envelope — **characterization, not
repair**. The premise (user, 2026-08-02): SA itself gives up on things (it has
no length scale away from the wall, and is famously wrong when a propeller
re-ingests its own wake) and became the most-used turbulence model in
aerodynamics anyway. The goal here is that people know the limits and adopt the
model despite — or because of — knowing them.

These files are deliberately **independent of `paper/sa-ai.tex`**; other agents
are editing the paper. Integrate later.

## Contents

| File | Subject | Status |
|---|---|---|
| `01-parabolic-profiles-pipe-channel.md` | Exactly parabolic profiles. Channel read as exactly neutral (misses TS at Re=5772); pipe reads as 1.9× Blasius from transverse curvature alone, net growth from Re_D≈530, growth rate saturating at 0.09 e-folds/diameter | **Complete.** Numbers reproducible |
| `02-open-items-and-data-status.md` | Audit of what the paper already covers; the surviving open items with data/feasibility status | Complete as a plan |
| `pipe_kernel_analysis.py` | Repro for 01. numpy/scipy, ~20 s, no GPU | Runs: `python3 -u blindspots/pipe_kernel_analysis.py` |

## Headline findings

1. **The parabola class.** `Î = 0` on any exactly parabolic profile is the
   model's neutral locus. Plane Poiseuille *is* exactly parabolic, so the model
   reads it as neutral at every Reynolds number and misses its `Re_c = 5772` TS
   instability outright (verified: `max|Î| = 1.8e-13`).

2. **Transverse curvature manufactures an inflection.** The solver forms
   `Z = ½d²(∇²u)·û` with `∇²u = −∇×ω`, which in a pipe carries the
   `(1/r)du/dr` term and is **2×** the planar `½d²u''`. Result:
   `max Ω̂Î = 0.150` in Hagen–Poiseuille — 1.9× the Blasius peak of 0.078 —
   with no inflection point present. The effect scales as **δ/r₀**: negligible
   on slender external bodies (≈1.6 % at the spheroid mid-body, so the external
   campaign is uncontaminated), O(1) internally, and *stabilizing* rather than
   destabilizing on convex bodies of revolution.

3. **A Reynolds threshold becomes a length threshold.** Pipe growth rate
   saturates at `s·D ≈ 0.09` e-folds per diameter above `Re_D ≈ 3000`, so the
   model predicts every pipe longer than ~70–160 diameters is turbulent at every
   `Re_D` above ~600, and every shorter one laminar at every Reynolds number.
   It therefore cannot reproduce the defining fact of pipe flow — laminar
   maintained to `Re = 10⁵` (Pfenniger) and `44 000` (Ekman) with careful
   inlets — nor the sharp critical point at `Re = 2040 ± 10` (Avila et al. 2011).
   Real pipe transition is subcritical and finite-amplitude; the correct linear
   answer is `s ≡ 0`.

4. **Most of a first-pass blindspot list was already characterized in the paper**
   — free-shear handover length, lift-off-height sweep, wake amplification,
   recirculation re-seeding, seed integrity. See §0 of file 02, including a
   correction: the NLF L0 scatter is a boundary-layer resolution artifact, not
   evidence of kernel conditioning.

## Next actions, in priority order

1. **Daedalus tip-vortex audit** — feasible now; volume data exists on
   **014-v100-dev** for `case_ogrid_L2_saai_a5` and `case_cavity_L2_saai_a4/a5`
   (3.75 GB, 36.7 M pts, fields `velocity` + `solutionTurbulence`). CPU-only.
   Decides whether the reported η>0.992 tip trip is physics or the kernel
   reading a vortex core as free shear. Highest value because it touches a claim
   already in the paper.
2. **Multi-element** — geometry not in repo; 30P30N is the recommended source.
   Unstructured only, metric = intersection of per-element airfoil metrics.
   Delivers confinement, wake-on-downstream-surface, confluent layer, cove
   recirculation and the medial axis in one case.
3. **Roughness seed BC** — design note in file 02 §4. One BC, one input (`k_s`),
   no new PDE term. First decide whether the kernel consumes `d` or `d + d0`
   under an Aupoix–Spalart origin shift: the indicators are built on `d`, so the
   shift is **not** neutral for transition.
4. **Non-ideal walls** — `u'''_w = 0` breaks under transpiration
   (`u'''_w = (v_w/ν)u''_w`) and on deforming surfaces (`ν u'''_w = ∂_t u'_w`).
   Derivable; limitations-section material.

## GPU note

Nothing here has needed a GPU. The pipe analysis is local numpy/scipy; the
Daedalus audit is a file read plus a KD-tree on 014. Crop the volume before
loading — do not pull 36.7 M points into memory on a shared machine.
