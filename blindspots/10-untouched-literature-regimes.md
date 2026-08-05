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

## The selection criterion (user, 2026-08-05)

> I'm looking for blindspots only to be **comparable** to other transition
> models, not going into un-explored water.

This is a sharper filter than "is it a blindspot", and it re-ranks the list.
A regime qualifies only if **other transition models have published numbers on a
named case**, so that running it produces a comparison rather than a first
result. Novelty is a *cost*, not a benefit. The "comparable?" column below is
that filter; the ordering of §11 obeys it, and it demotes two items I had ranked
on interest (bar-passing wake rigs, geometry-fixed separation) and promotes one
sharply (the A-airfoil, §6 — now the recommended first case).

The canonical anchor for "what everyone runs" is Langtry & Menter's own
test-case paper, whose three 2D cases are the **Aérospatiale-A airfoil**, the
**Zierke & Deutsch compressor cascade**, and the **VKI BRITE large-scale turbine
cascade** (Genoa). Sources in §13.

## Summary

| # | Item | Status in our work | Comparable? | Kind | Cost |
|---|---|---|---|---|---|
| 6 | **A-airfoil / max-lift with free transition** | incidence ranges stop short of stall | **yes — canonical, and used by the algebraic (BCM-family) papers** | campaign, existing infrastructure | **low** |
| 1a | Steady turbomachinery cascade (Zierke & Deutsch) | absent (`cascade`×1, `compressor`×0) | **yes — canonical** | campaign, new mesh topology | medium |
| 10 | Community benchmark sets (AIAA transition workshops) | not used; conclusion promises it | **yes, by construction** | curation | low |
| 3 | Heat transfer as the quantity of interest | absent (`heat`×0) | yes, large literature — but the canonical cases are transonic | campaign or statement | medium |
| 2 | Rotating frames — rotors, propellers, rotating disk | absent (`rotating`×0, `Coriolis`×0, `propeller`×0) | partly — self-characterization, not a comparison | formulation statement + smoke test | days |
| 7 | Differentiability for gradient-based design | absent (`adjoint`×0) | in kind (AFT-S precedent), not a case | bounded test | low |
| 1b | Unsteady bar-passing wake rig (T106D-EIZ) | absent (`T106`×0) | yes, but compressible, unsteady, and a deliberate stress case | campaign | high |
| 4 | Steps, gaps, surface waviness | absent; distinct from roughness-as-receptivity | thin — studies exist, no canonical benchmark | bounded test → criterion curve | low–medium |
| 8 | Receptivity beyond a single `Tu` scalar | `χ_∞` is one number from Mack | statement only | statement | low |
| 9 | Unsteady inflow, pitching / dynamic stall | no unsteady case in the campaign | moderate | statement (partly file `02` §3) | low |
| 5 | Geometry-fixed separation (blunt LE, backward step, hump) | absent | **weak for transition models specifically** — keep only for its diagnostic value | bounded test | low |

Recommended order — **6, then 10, then 1a**. Rationale in §11. Item numbering
below is the original one, kept so cross-references from other files survive.

---

## 1. Turbomachinery cascades, especially periodic wake passing

**What others did.** This is the largest single application literature for
correlation-based transition models. **Zierke & Deutsch** (compressor,
double-circular-arc, Penn State low-speed rig, LDV boundary-layer data,
suction-side bubble) is one of the three 2D cases in Langtry & Menter's own
test-case paper and recurs in the calibration and re-implementation literature
(Malan & Suluksna's commercial-CFD calibration; the SU2 γ–Re_θ implementation).
The LPT side carries the T106A/C/D family, and the **moving-bar rigs** that
reproduce upstream rotor wakes, with matched DNS/LES (Wissink & Rodi; Stieger &
Hodson; Michelassi). None of it has a counterpart in our campaign.

**Is the bar-passing case comparable, or unexplored?** Checked 2026-08-05:
**comparable — the benchmark was purpose-built for exactly that.** T106D-EIZ
takes the T106A geometry and opens pitch/chord from 0.799 to 1.05 (≈30 % more
loading), adds a **moving-bar wake generator**, and was measured at the
University of the German Armed Forces Munich with the data published for download
*so that different turbulence and transition models could be run against it*.
URANS transition-model results exist on it and on T106A with SST γ–θ, plus URANS
studies of upstream wakes on high-lift LPT cascades using transition-sensitive
closures. So the wake-passing item is **not** unexplored water.

Two reasons it is still the wrong first move (hence its demotion to 1b in the
summary):

- **Mach.** T106 rigs are high-speed facilities. This sits outside the paper's
  stated incompressible scope, so the comparison would arrive bundled with an
  unvalidated compressibility extension.
- **It is deliberately an adversarial case.** Most transition models are
  calibrated on the moderately-loaded T106A; T106D-EIZ was constructed *because*
  it is demanding for them. Unsteady + compressible + adversarial is a poor place
  to make a first comparison, and a poor place to distinguish a model defect from
  a scope violation.

**Scope-honest entry point: Zierke & Deutsch (steady, low speed), item 1a.**

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

**Where the wake actually comes from — and the tension this creates.** A plain
steady cascade has **no upstream blades and no wakes at all**: uniform inlet plus
grid turbulence. The wake-passing physics comes only from the separate moving-bar
rig upstream of the still-stationary blade row (file `11` §1.0, §1.3). So the two
halves of this item pull in opposite directions:

- The **steady** cascade is the comparable half (canonical case, low speed, in
  scope, many published transition-model results) but its physics overlaps
  §`sec:eppval` and §`sec:twoelement`; its independent assets are the LDV
  boundary-layer data and the confined passage geometry.
- The **bar-passing** case is where the distinctive claim lives — transition with
  memory, which the algebraic family cannot represent — but it costs URANS,
  compressibility, and an adversarial benchmark.

Under the comparability criterion the steady half wins, and the memory claim
stays an argument rather than a demonstration until someone is willing to pay for
1b. That is a defensible position: the argument follows from the model's
structure and can be stated in the paper without a case, as long as it is stated
*as* an argument.

Note also that LPT cases typically run at `Tu` = 0.5–4 %, i.e. into the declined
bypass regime — another reason the low-speed compressor cascade, which can be run
at low `Tu`, is the entry point.

**Feasibility, solver support, and what a linear cascade actually is:** see
`11-cascade-linear-and-flow360-support.md`. Short version: Flow360 supports it,
the mesh is the work item, and one open question (wall distance across a periodic
boundary) touches the SA-AI indicators specifically.

## 2. Rotating frames — rotors, propellers, rotating disk

**Cheapest item in the list**, because it is a formulation question rather than a
campaign. Under the comparability criterion it is *not* a comparison case — the
deliverable is self-characterization, a statement about what the kernel reads in a
rotating frame. Keep it for that reason, not as a benchmark. (`Tu`-swept
rotating-disk stability and NREL Phase VI do give comparison points if the
formulation answer turns out to be "absolute vorticity", but that is a
contingency, not the plan.)

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

## 6. The Aérospatiale-A airfoil — max-lift with free transition

**Promoted to the recommended first case** by the comparability criterion. It was
ranked mid-list on a novelty axis, which was the wrong axis.

**Conditions (verified 2026-08-05):** `Re = 2.1×10⁶`, `M = 0.15`,
α = 13.3°, `Tu = 0.05 %`. The suction-side laminar layer **separates at
0.12 c and reattaches turbulent**, with the bubble and transition located by
**oil-flow visualization**.

**Why it is the cheapest comparable case we have available:**

- **In scope on both axes that matter.** `M = 0.15` against our `M = 0.1`, and
  `Tu = 0.05 %` is a natural-transition disturbance environment — *not* the
  bypass regime the paper declines. Almost nothing else on this list is in scope
  without an extension.
- **It is the regime already validated, relocated.** A leading-edge laminar
  separation bubble with an oil-flow-measured front is exactly what
  §`sec:eppval` compares against — at 10× the Reynolds number, and near stall
  where the front's position sets the turbulent trailing-edge separation and
  therefore `C_L,max`.
- **Zero new infrastructure.** It is an airfoil: existing contour machinery,
  both existing mesh families, the existing refinement ladder, no periodic BCs,
  no new topology, no new BC types. Contrast item 1a, which needs an H-grid,
  inflow/outflow, a periodic pair, and a wall-distance audit first.
- **The comparison set is exactly the one the conclusion promises.** The
  A-airfoil is one of the three 2D cases in Langtry & Menter's test-case paper,
  it carries the EUROVAL / ECARP / LESFOIL history, and it is used in the
  **algebraic zero-equation (BCM-family) transition-model** papers — which is
  precisely the "direct comparison with the algebraic SA-BCM model on
  community-standard benchmarks" the conclusion already commits to.

The adjacent wind-energy literature (S809, DU-91-W2-250, clean vs. rough) exists
almost entirely to answer "how does transition move `C_L,max`", and γ–Re_θ results
on those sections are published — a second tier of comparison if the A-airfoil
lands well.

**The one real risk, and it is ours not the case's.** This is the regime where
standard SA's attachment-anchored branch (§`sec:bistability`) and the convergence
protocol will be under most strain. That argues for running it *with* the
protocol as published and reporting what the protocol admits, rather than waiting
on the relaminarization quench of file `07` — but expect the bistability question
to resurface, and decide in advance how it will be reported.

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

Ordered by the comparability criterion, not by interest.

1. **Item 6 (Aérospatiale-A airfoil).** In scope on Mach *and* on `Tu`, a
   leading-edge bubble with an oil-flow front, zero new infrastructure, and it sits
   in both the canonical γ–Re_θ test-case set and the algebraic-model papers we
   have promised to compare against. Cheapest *and* most comparable — an unusual
   combination, and the reason it displaces everything below.
2. **Item 10 (adopt a published benchmark list).** Curation, not compute. Decides
   the rest of the roadmap by borrowing someone else's case selection, which is
   itself the comparability argument.
3. **Item 1a (Zierke & Deutsch, steady).** The cascade entry point: canonical,
   low speed, LDV boundary-layer data. Costs a new mesh topology, new BC types,
   and the wall-distance audit of file `11` §4 — so it follows the A-airfoil
   rather than leading.

Cheap self-characterization items to interleave whenever convenient, none of which
produce a comparison: **2** (rotating-frame vorticity — still worth doing early
because it is nearly free and the paper's abstract sells rotors), **7**
(differentiability), **8** (receptivity), **9** (unsteady statement).

**Deferred:** item 3 (heat transfer) until the `σ_t` closure exists, since it
would otherwise measure a known-open defect; item 1b (bar-passing) until there is
appetite for URANS plus compressibility; item 4 (steps/gaps) and item 5
(geometry-fixed separation) last, since neither has a canonical
transition-model benchmark to be compared against — item 5 survives only on its
diagnostic value for the Eppler bursting decomposition.

---

## 12. Provenance and verification debt

- The keyword counts in the summary table are `grep -ic` over
  `paper/sa-ai.tex` as of 2026-08-05.
- **Citation debt, reduced but not cleared 2026-08-05.** The three 2D cases of
  Langtry & Menter's test-case paper (A-airfoil, Zierke & Deutsch, VKI BRITE /
  Genoa), the A-airfoil conditions and its 0.12 c separation, and the existence
  and purpose of T106D-EIZ were checked against the sources in §13. **Pak-B is
  removed** — it was in the recalled version of the Langtry & Menter set and did
  not survive checking; treat any earlier note that names it as unverified. The
  stated motivation for AFT-S (§7) is **still from recall**.
- Nothing in this file has been computed. Every item is a proposal, and every
  "why it bites" is an argument from the model's form, not an observation.

---

## 13. Sources checked (2026-08-05)

- Langtry & Menter, *A Correlation-Based Transition Model Using Local
  Variables — Part II: Test Cases and Industrial Applications*, J. Turbomach.
  128(3):423 —
  <https://asmedigitalcollection.asme.org/turbomachinery/article-abstract/128/3/423/476613/A-Correlation-Based-Transition-Model-Using-Local>
  (the three 2D cases).
- Menter et al., *Transition Modelling for General Purpose CFD Codes*, Flow
  Turbul. Combust. — <https://link.springer.com/article/10.1007/s10494-006-9047-1>
- Malan & Suluksna, *Calibrating the γ-Re_θ Transition Model for Commercial CFD*
  — <https://www.semanticscholar.org/paper/Calibrating-the-%CE%B3-Re-%CE%B8-Transition-Model-for-CFD-Malan-Suluksna/cf3d02b55d65cdaeaf2a39884ce92802ba932b2b>
  (ERCOFTAC cases + Zierke & Deutsch).
- γ–Re_θ implementation in SU2 —
  <https://re.public.polimi.it/bitstream/11311/1242117/3/RAUSA_OA_01-23.pdf>
  (Zierke & Deutsch at −1.5° incidence, mesh and y+ detail).
- Stadtmüller & Fottner, *A Test Case for the Numerical Investigation of Wake
  Passing Effects on a Highly Loaded LP Turbine Cascade Blade*, ASME GT2001 —
  <https://asmedigitalcollection.asme.org/GT/proceedings-abstract/GT2001/78507/V001T03A015/249324>
  (T106D-EIZ, moving-bar wake generator, data published for model comparison).
- *Unsteady simulation of the LP turbine test case T106D-EIZ*, DGLR 2013 —
  <https://www.dglr.de/publikationen/2013/281279.pdf> (URANS with modern
  transition models; T106A as the usual calibration case).
- *Unsteady RANS modelling of wake-induced transition in a linear LP turbine*,
  Imperial College —
  <https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/turbulent-flow-modeling-and-simulation/47859705.PDF>
- *URANS Prediction of the Effects of Upstream Wakes on High-Lift LP Turbine
  Cascades Using Transition-Sensitive Turbulence Closures* —
  <https://www.academia.edu/4544519/URANS_Prediction_of_the_Effects_of_Upstream_Wakes_on_High_Lift_LP_Turbine_Cascades_Using_Transition_Sensitive_Turbulence_Closures>
- A-airfoil conditions and the 0.12 c oil-flow separation: ERCOFTAC KBwiki
  UFR 2-05 — <https://www.kbwiki.ercoftac.org/w/index.php/UFR_2-05_Test_Case> —
  and *A local correlation-based zero-equation transition model* —
  <https://www.sciencedirect.com/science/article/abs/pii/S0045793020303285>
  (the algebraic-family paper that runs it).
- γ–Re_θ on a wind-turbine section, DU-91-W2-250 —
  <https://www.mdpi.com/1996-1073/14/24/8224>

Read the primary sources before any of this is cited in the paper; the above were
read as search results and abstracts, not in full.
