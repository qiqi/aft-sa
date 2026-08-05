# Blindspots — regimes the RANS transition-model literature invested in, that we have not touched

Companion to `02-open-items-and-data-status.md`. That file audits the items we
generated ourselves. This one asks a different question, posed by the user
(2026-08-05):

> we already looked at internal flow, wake / lifted shear layer (cylinder drag
> crisis, two-element airfoil), 3D including crossflow (Daedalus / spheroid).
> Are there any other blindspots, something *other people* investigated with a
> RANS transition model, that we haven't touched?

So the filter is **literature coverage**, not our own imagination: what does the
γ–Re_θ / AFT / BC-BCM validation record contain that the SA-AI campaign has no
counterpart for?

**Method.** Every item below was checked against (a) all twelve case sections of
`paper/sa-ai.tex` plus its nine appendices, (b) the stated out-of-scope list in
the introduction, (c) the future-work paragraph of the conclusion, and (d) files
`01`–`09` of this directory. Anything already covered by those was dropped.
Dropped for that reason, recorded so they are not re-proposed: bypass/high-`Tu`
(explicitly out of scope, T3A/T3B named and declined at `sa-ai.tex:1451`),
crossflow (§`sec:spheroid` + the Gram-algebra sensor sketched in the
conclusion), LSB and bubble bursting, the favorable rooftop, steady
wake-over-surface (§`sec:twoelement`), pipe/channel (file `01`), tip vortex
(file `02` §2, file `06`), transpiration and deforming walls (file `02` §3),
roughness receptivity (file `02` §4), Görtler, wall temperature,
attachment-line, relaminarization, `Î` conditioning (file `02` §5, file `07`),
sphere and the unsteady cylinder (conclusion).

Same premise as the rest of this directory: **characterization, not repair.**
Each item therefore carries a deliverable that is a statement or a bounded test,
not a campaign, except where flagged.

---

## Summary

| # | Item | Status in our work | Kind | Cost |
|---|---|---|---|---|
| 1 | Turbomachinery cascades, incl. periodic wake passing | absent (`cascade`×1, `compressor`×0, `T106`×0, `Pak-B`×0) | campaign | see file `11` |
| 2 | Rotating frames — rotors, propellers, rotating disk | absent (`rotating`×0, `Coriolis`×0, `propeller`×0) | formulation statement + smoke test | days |
| 3 | Heat transfer as the quantity of interest | absent (`heat`×0) | campaign or statement | medium |
| 4 | Steps, gaps, surface waviness (excrescences) | absent; distinct from roughness-as-receptivity | bounded test → criterion curve | low–medium |
| 5 | Geometry-fixed separation (blunt LE, backward step, hump) | absent | bounded test, diagnostic value | low |
| 6 | Max-lift / stall with free transition | incidence ranges stop short of stall | campaign | medium |
| 7 | Differentiability for gradient-based design | absent (`adjoint`×0) | bounded test | low |
| 8 | Receptivity beyond a single `Tu` scalar | `χ_∞` is one number from Mack | statement | low |
| 9 | Unsteady inflow, pitching / dynamic stall | no unsteady case in the campaign | statement (partly file `02` §3) | low |
| 10 | Community benchmark sets (AIAA transition workshops) | not used; conclusion promises it | curation | low |

Recommended order — 2, 7, then 1. Rationale in §11.

---

## 1. Turbomachinery cascades, especially periodic wake passing

**What others did.** This is the largest single application literature for
correlation-based transition models. Langtry & Menter's own validation set
includes a compressor cascade (Zierke & Deutsch, double-circular-arc, Penn State
low-speed rig, LDV boundary-layer data, suction-side bubble) and a low-pressure
turbine blade (Pak-B). The LPT side then carries the T106A/C/D family and the
**moving-bar rigs** that reproduce upstream rotor wakes, with matched DNS/LES
(Wissink & Rodi; Stieger & Hodson; Michelassi) and a dedicated
separated-flow-transition correlation (Praisner & Clark). None of it has a
counterpart in our campaign.

**Why this is more than one more case.** The physics is *wake-induced
transition*: a wake sweeps across the passage, a turbulent strip forms and
convects, and behind it a **becalmed region** stays laminar longer than the
steady state would. That is transition with memory.

- SA-AI is the only entry in the paper's Table 1 taxonomy that carries history in
  the working variable *without* an extra transport equation. Periodic wake
  passing is therefore the regime where SA-AI has a structural argument against
  the algebraic family (BC/BCM), which is memoryless by construction. This is an
  **opportunity item**, not only a risk item — unusual in this directory.
- The matching risk is equally specific and is the unsteady generalization of
  §`sec:twoelement`: the arriving wake delivers `χ` into the very variable the
  blade layer is using as its amplification ladder. The two-element section
  tested that with a **steady** wake at one lift setting. A passing wake asks
  whether the handover survives being re-seeded periodically, and whether the
  becalmed region survives at all when the seed arriving from upstream never
  drops back to `χ_∞`.

**Where the wake actually comes from.** A plain steady cascade has **no upstream
blades and no wakes at all** — uniform inlet plus grid turbulence. The wake-passing
physics comes from a separate rig option: a **moving-bar generator** upstream of
the still-stationary blade row (file `11` §1.0, §1.3). So the steady cascade on
its own is another confined bubble case, overlapping §`sec:eppval` and
§`sec:twoelement`; its independent assets are the LDV boundary-layer data and the
passage geometry. **The unsteady bar-passing phase is not a bonus on top — it is
the item.** Price the campaign accordingly.

**Scope-honest entry point: Zierke & Deutsch, not T106.** T106 exit Mach numbers
put it outside the paper's stated incompressible scope, and LPT cases usually run
at `Tu` of 0.5–4 %, i.e. into the declined bypass regime. The compressor cascade
is low speed and can be run at low `Tu`. Run the steady case first as the
validation base, then the bar-passing case, which needs URANS and a prescribed
time-varying inlet wake.

**Feasibility, solver support, and what a linear cascade actually is:** see
`11-cascade-linear-and-flow360-support.md`. Short version: Flow360 supports it,
the mesh is the work item, and one open question (wall distance across a periodic
boundary) touches the SA-AI indicators specifically.

## 2. Rotating frames — rotors, propellers, rotating disk

**Ranked highest on value per unit compute**, because it is a formulation
question rather than a campaign.

**The specific hazard.** The indicator sphere is built on the vorticity:

```
(X, Y, Z) = ( |u|, d|ω|, ½ d²∇²u·û ) ,   Ω̂ from d|ω|/R ,   Re_Ω = d²ω/ν
```

In a rotating reference frame the absolute vorticity carries the `2Ω` term. Unless
the kernel is explicitly fed **relative** vorticity, a solid-body-rotating fluid
reads as sheared: `Ω̂ → 1` and `Re_Ω` inflates with no shear present, and the
amplification rate becomes **frame-dependent** — the same physical flow gives a
different transition location computed in the blade-fixed frame than in the
inertial frame. Standard SA has the same exposure in `S̃`, which is why SA-RC and
the rotation/curvature corrections exist; but for SA-AI the consequence lands on
the transition location, not on a modest eddy-viscosity level.

**Deliverable.** (i) Read the solver's rotating-frame path and state, in one
paragraph, which vorticity the kernel consumes. (ii) One smoke case: rigid-body
rotation of a fluid in a rotating frame with a wall — the rate must be identically
zero. (iii) If the answer is "absolute", say so as a scope statement, since it
gates every rotor deployment.

**Why it matters beyond the algebra.** The paper's own introduction sells the
application: "small — increasingly electric — rotors and drones", where the
laminar run is a large fraction of chord. Rotating-frame behaviour is currently
the gap between the advertised market and the tested envelope. The literature
others built here is substantial: NREL Phase VI in Langtry's set, helicopter-rotor
applications of Coder's AFT, the current eVTOL/UAV-propeller low-Re transition
literature, and the rotating disk as the canonical rotating-frame stability
problem (Type I/II crossflow instability, and a clean experiment).

Note `flexfoil/rans/rotating/PLAN.md` exists — check it before starting, that
work may already have set up a rotating-frame case.

## 3. Heat transfer as the quantity of interest

`heat` does not appear once in `sa-ai.tex`. Yet a large fraction of industrial
transition-model use is thermal rather than drag: turbine blade and vane heat
load (the Arts/VKI LS89 cascade is the reference case), stagnation-region heat
transfer, film-cooling interaction.

**Why it bites SA-AI specifically.** The deliverable in those studies is the
**Nusselt overshoot through the transitional zone** — i.e. the internal structure
of the zone, which the paper deliberately does not model (memoryless fixed-`χ`-width
handover, §`sec:blend`; limitation #2 in the conclusion). Skin friction integrates
the error and drag partly hides it; `h(x)` does not. This is the QoI that turns
"the transition zone is unmodeled" from a caveat into a number, and it feeds
directly into the outer-scaled, memory-carrying `σ_t` closure the conclusion
already proposes.

Cheapest honest version: state the limitation with the transitional-`h(x)`
argument, and note that the fixed-`χ`-width interpolation has no mechanism to
produce the measured overshoot. A real case needs a thermal BC path in the
SA-AI build — check before promising.

## 4. Steps, gaps, and surface waviness

**Distinct from file `02` §4**, which treats roughness as pure receptivity via a
wall seed. A forward- or backward-facing step is a *geometric* excrescence, and
the literature is a first-tier NLF/HLFC concern — manufacturing and assembly
tolerance criteria (Drela's step criteria; Perraud & Séraudie; the recent
step/gap studies with AFT and γ–Re_θ). Deliverable in that literature is a
criterion: allowable `Δh/δ₁` before the front jumps forward, equivalently the
`ΔN` a step costs.

**Why it bites the kernel.** A step plants its own small separation bubble and
breaks the assumption the entire indicator sphere rests on — that to second order
about a wall-normal point the profile is captured by `(u, d u', ½d²u'')`. At the
corner the expansion is not that, and `d` is simultaneously discontinuous in
direction. This is the same class as the medial-axis issue noted for
multi-element (file `02` §1), but on a single surface and much cheaper to set up:
flat plate plus one step, sweep `Δh/δ₁`, report where the front moves.

## 5. Geometry-fixed separation: blunt leading edge, backward step, wall-mounted hump

Every separated case we have — Eppler bubble, cylinder, Daedalus mid-span bubble
— has a separation point the model must **predict**. A corner-fixed separation
pins it by geometry.

**Why that is worth a case even though it adds no new physics.** It is the
missing decomposition for the Eppler bursting analysis. §`sec:epphandover`
traces the low-`Re` failure to two opposite-signed defects that partially cancel
(handover length starving the bubble; closed-cell re-seeding). With the
separation point pinned, the shear-layer amplification rate is measurable in
isolation from the separation-location error. Classic cases: the blunt-nosed flat
plate (Ota), a backward-facing step at low `Re`, the NASA wall-mounted hump.
Cheap, 2D, and diagnostic rather than merely additive.

## 6. Max-lift / stall with free transition

Our incidence ranges stop short of stall (Eppler to 7°, NLF to 9°). The standard
case is the Aérospatiale-A airfoil (ERCOFTAC, `Re = 2.1×10⁶`, α = 13.3°), where
the transition location sets the turbulent trailing-edge separation and therefore
`C_L,max`; the wind-energy literature (S809, DU sections, clean vs. rough) exists
almost entirely to answer "how does transition move `C_L,max`". Our bubble-bursting
boundary is adjacent to this but is not the same deliverable.

Caution: this regime is where standard SA's attachment-anchored branch
(§`sec:bistability`) and the convergence protocol will be under most strain, so
it may be better attempted after the relaminarization quench of file `07`.

## 7. Differentiability for gradient-based design

`adjoint` appears zero times in `sa-ai.tex`. Halila's AFT-S exists **because**
AFT was not smooth enough for implicit Newton–Krylov solution and adjoint-driven
optimization, and NLF shape optimization is where Coder's and Martins' groups took
these models. Nobody has asked whether SA-AI's front position is a smooth
function of a design variable — and SA-AI contains a soft-min onset threshold, a
tanh onset ramp, a blend switch, and a projective coordinate with a pole at the
curvature axis.

**Deliverable, cheap.** Finite-difference the transition-front location against a
single design variable (α, or one shape mode) at fixed grid and fixed protocol,
step size swept over two decades; look for staircasing or non-monotone response.
This is small, it is an adoption argument rather than a defect hunt, and the
kernel already exists in a JAX form (`paper/repro/lib/sphere_kernel.py`,
`.jax_cache`) so the derivative may be directly available for comparison against
the finite difference.

## 8. Receptivity beyond a single `Tu` scalar

`χ_∞` comes from Mack's correlation — one number from one number. The literature
others built includes acoustic receptivity, leading-edge receptivity, and
facility-specific `ΔN` calibration from measured disturbance spectra (Crouch,
Drela). Consequence: the same measured `Tu` implies different N-budgets in
different facilities, and the flat-plate agreement of §`sec:flatplate` is stated
*at* the Mack closure. A statement, not a campaign; it also frames the roughness
seed of file `02` §4 as one member of a family of receptivity inputs the model
currently collapses onto a single scalar.

## 9. Unsteady inflow, pitching, dynamic stall

Partly anticipated by the `u'''_w ≠ 0` deforming-wall derivation in file `02` §3,
but worth separating: dynamic-stall transition is an application literature in its
own right (rotor blades again, and pitching-airfoil rigs), and **no unsteady case
exists anywhere in the campaign** — the cylinder traverse is deliberately a
steady branch. `flexfoil/rans` already carries a URANS builder
(`build_urans_simulation_json`) and `validate_unsteady_sweep.py`, so the
infrastructure is not the obstacle.

## 10. Community benchmark sets

The conclusion already promises "a direct comparison with the algebraic SA-BCM
model on community-standard benchmarks". The shortcut to answering *"what have
others investigated"* with authority is to adopt the AIAA Transition Modeling &
CFD workshop case list rather than curating our own: that is where the crossflow
case (DLR sickle wing) and the transonic NLF case (CRM-NLF) live, and using its
cases makes our envelope legible to reviewers without having to defend case
selection. Low cost, high credibility return.

---

## 11. Suggested order, and why

1. **Item 2 (rotating frames).** Algebra plus one smoke case; may force a scope
   statement regardless of outcome; and it closes the gap between the paper's
   advertised application (rotors, drones) and its tested envelope.
2. **Item 7 (differentiability).** Nearly free, and it is an adoption argument.
3. **Item 1 (Zierke & Deutsch cascade).** The one real new campaign worth
   running: it opens the largest untouched application literature at our existing
   incompressible scope, and periodic wake passing is where SA-AI's
   single-variable memory is a structural advantage to claim rather than a
   limitation to confess.

Items 4, 5, 8, 9 are cheap and can be interleaved. Items 3 and 6 are campaigns
and should wait for the `σ_t` closure and the relaminarization quench
respectively, since both would otherwise be measuring known-open defects.

---

## 12. Provenance and verification debt

- The keyword counts in the summary table are `grep -ic` over
  `paper/sa-ai.tex` as of 2026-08-05.
- **Citation debt:** the exact composition of Langtry & Menter's published
  validation set (§1), and the stated motivation for AFT-S (§7), are from recall
  and must be verified against the sources before either enters the paper. The
  physical arguments do not depend on the attributions.
- Nothing in this file has been computed. Every item is a proposal, and every
  "why it bites" is an argument from the model's form, not an observation.
