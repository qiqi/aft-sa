# Anchoring the viscous amplification *rate*

The whitepaper appendix anchors the viscous branch's **onset** (`B_c`) on
Orszag's `Re_c = 5772` and then waves at "its peak envelope rate" for
`a_visc`. That second half is not yet an anchor. This note says what it should
be, why DNS is the wrong instrument, and surfaces a structural fork the
question exposes.

---

## 1. The right instrument is Orr–Sommerfeld, not DNS

Plane Poiseuille is the *best-characterised* linear stability problem in fluid
mechanics. The eigenvalue problem is exactly solvable and has been to eight
digits since Orszag (1971): `Re_c = 5772.22`, `α_c = 1.02056`, `c_r = 0.264`.
Above `Re_c` the spatial amplification `−α_i(Re, ω)` is a smooth computable
surface, and its **envelope over frequency** is precisely the same object as the
Drela–Giles envelope for Blasius — which is itself built from Orr–Sommerfeld
solutions.

So the viscous branch should be anchored exactly as the rest of the model is:

| branch | quantity | anchor |
|---|---|---|
| inviscid rate | `a_max = 0.19` | Michalke tanh KH temporal eigenvalue |
| inviscid onset | `k = 0.712` | Blasius march onto the Drela–Giles `N=1` station |
| **viscous onset** | `B_c` | plane-Poiseuille `Re_c = 5772` (Orszag) |
| **viscous rate** | `a_visc` | **plane-Poiseuille OS spatial envelope `dN/dRe`** |

No new class of evidence is introduced — it is the same linear-stability
currency the paper already trades in.

**The tool exists in the repo.** `repro/analytic/explore_pinchpoint_shooting.py`
is an Orr–Sommerfeld / Briggs–Bers shooting solver already validated against
Hammond–Redekopp (`R_crit = 1.3157`). Adding a plane-Poiseuille base flow and a
frequency sweep is a small job, and it should reproduce `5772.22` as its own
verification before anything is anchored on it.

## 2. Why DNS is the wrong instrument here

The question is fair — Moser and others have decades of channel DNS — but it is
aimed at the wrong regime:

- **The channel DNS canon is fully-turbulent statistics.** Kim–Moin–Moser (1987),
  Moser–Kim–Mansour (1999), Lee & Moser (2015) to `Re_τ = 5200` are
  *statistically stationary* turbulence. The initial transient is a nuisance to
  be discarded, not an object of measurement, and growth rates are not reported.
- **Transition DNS in channels studies the wrong stage.** Zang & Krist, Sandham &
  Kleiser (1992), Gilbert & Kleiser examine *secondary* instability and breakdown
  (K-type / H-type). The *primary* TS growth in those simulations is just linear
  theory — which OS gives more accurately and far more cheaply.
- **Real channel transition is subcritical.** It happens well below `Re_c = 5772`
  via transient (lift-up) growth and streak breakdown. A DNS "watching
  disturbances grow" is mostly watching nonlinear transient growth — precisely
  the regime this model excludes by scope.
- Temporal→spatial conversion is not the obstacle. Gaster's transformation
  (`ω_i ≈ −α_i c_g`, valid for small growth) would do it. It is simply a detour:
  OS gives the spatial rate directly.

**Where DNS *is* the right instrument for this model** is the other end — the
**handover**: nonlinear saturation and breakdown, which `σ_t` and the
`χ = 1 → c_v1` ladder currently interpolate with no physics (Drela feedback §3;
§`epphandover`). Sandham & Kleiser, and Alam & Sandham / Spalart & Strelets for
bubbles, have exactly the data linear theory cannot supply. That is where a DNS
campaign would buy something.

## 3. The fork this question exposes

Checking what `a_visc` each flow wants produced a convergence — and a problem.

**Blasius has no inflection point.** `u''_w = 0` for zero pressure gradient and
`u'' < 0` throughout, so by Rayleigh it is inviscidly *stable*. Its TS
instability is **purely viscous** — the entire `e^N`/NLF enterprise rests on a
viscous instability of a non-inflectional profile. Yet the current kernel carries
Blasius on the coordinate labelled *inflectional*.

Two independent numbers:

```
a_visc that would let the VISCOUS branch alone carry Blasius:
      0.19 × max P_I / max P_curv  =  0.19 × 0.0782 / 0.2292  =  0.0648

a_visc that the PLANE CHANNEL anchor wants at B_c = 130:
      0.060   (gives Re_c = 5834 vs the true 5772 — 1.1 % off)
```

**`a_visc ≈ 0.06` satisfies the plane-channel critical Reynolds number and the
Blasius rate simultaneously, at `B_c ≈ 130`.** Two unrelated flows, one constant,
no fit. That is a far better story than `a_visc = 0.0276` from a low-H
Falkner–Skan residual.

### But it forces a choice

At `a_visc = 0.0276` the "changes nothing" property (file `05-*.md`) holds — and
the reason it holds is uncomfortable. On Blasius the viscous threshold is
`thr_visc = 2598`, while `Re_Ω ≈ 2.19 Re_θ`:

| `Re_θ` | `Re_Ω` | `S_visc` |
|---|---|---|
| 600 | 1312 | 0.056 |
| 1000 | 2187 | 0.288 |
| **1200** | **2625** | **0.515** |
| 1500 | 3281 | 0.818 |

The viscous gate reaches half-open at `Re_θ ≈ 1190` — essentially **exactly** the
Blasius `N = 9` station (model 1181, Drela 1108). So the viscous branch
contributes nothing to Blasius's accumulated `N` only because the layer has
already transitioned by the time it switches on.

That means the decoupled form as parameterised **does not assign Blasius to the
viscous branch**. It leaves Blasius on `Î` and adds a viscous branch that fires
only in the strongly-favorable/parabolic corner. So:

| | position | `a_visc` | consequence |
|---|---|---|---|
| **(a) pragmatic** | `Î` is a *shape proxy* calibrated on Drela; `P_curv` extends it where `Î` dies. No mechanistic claim. | ≈0.028 | "changes nothing" holds; the appendix must not claim the branches are the two *mechanisms* |
| **(b) mechanistic** | `P_I` carries genuinely inviscid (inflectional) instability; `P_curv` carries viscous TS. | ≈0.065 | Blasius moves to the viscous branch — so `k = 0.712`, anchored on Blasius, is a *viscous* anchor and must be re-derived. Much larger change. |

Position (b) is more principled and is where the `a_visc ≈ 0.06` coincidence
points. Position (a) is what files `05-*.md` and the whitepaper appendix
currently propose. **The appendix's language should be fixed either way**: as
written it implies (b) while the constants implement (a).

## 4. A structural concern the OS envelope would settle

The model's viscous rate `a_visc · P_curv` is **Reynolds-independent** —
`P_curv` is a pure shape coordinate. That assumption is inherited from the
inviscid branch, where it is justified: Drela's envelope rate is
Reynolds-independent beyond the viscous cutoff, because the mechanism is
inviscid.

**It does not obviously transfer to a viscous mechanism.** A viscous instability's
growth rate should depend on Reynolds number — rising from zero at `Re_c`,
peaking, and decaying as `Re → ∞` where viscosity (its energy source) weakens.
The model's computed channel growth does the opposite:

```
s·h  =  0.0026 (Re 2e4)   0.0046 (5e4)   0.0054 (1e5)   0.0065 (1e6)
```

— monotonically saturating, never decaying. If the OS envelope for plane
Poiseuille decays at high `Re`, the model will over-predict channel instability
there, and the mismatch is *structural*, not a constant. **This is the single
most important thing the OS computation would tell us, and it should be checked
before `a_visc` is fixed at all** — there is little point anchoring a rate whose
Reynolds dependence is wrong.

## 5. "Channel transition is subcritical — so is the anchor useful?"

Two different things are being conflated, and separating them answers it:

- **(A) Does the linear mechanism exist, and at what rate?** For plane Poiseuille:
  yes, exactly, `Re_c = 5772.22`, rates computable to eight digits.
- **(B) Is that mechanism the route by which the flow actually transitions?**
  For plane Poiseuille: **no.** Observed transition sits well below `Re_c` —
  by a factor of several — via transient growth and streak breakdown.

We are anchoring on **(A)**, not (B). The model is an envelope method: it claims
only to carry *linear* amplification. Calibrating its kernel against a linear
eigenvalue is exactly in scope — the same status as `a_max = 0.19`, where nobody
claims a mixing layer transitions "at" the Michalke rate; it is the growth rate of
the mechanism.

**But (B) does impose a hard limit: the channel can never be a validation case.**
There is no measured channel transition location the model could be compared
against, because the observed one is produced by physics the model excludes. The
channel is a *calibration and verification device only*. That should be stated
plainly wherever it appears.

## 6. Do favorable-pressure-gradient boundary layers transition linearly?

This is the question that decides whether the viscous branch earns its place,
since FPG layers — not channels — are the in-scope flows where it fires.

**Mild-to-moderate FPG, low disturbance: yes, linear TS is the route.** This is
the natural-laminar-flow regime, and the entire `e^N` design practice rests on
it. The paper's own NLF(1)-0416 campaign matching the Somers fronts across the
incidence matrix is direct evidence: a rooftop held by a favorable gradient,
igniting at the recovery, at `N_crit = 9`. Note also that **Blasius itself is
non-inflectional**, so the canonical TS instability of the entire NLF enterprise
*is* a viscous instability of a non-inflectional profile. The mechanism the
viscous branch represents is not exotic — it is the main one.

**Strong FPG: no, and it should not matter.** As `β → 0.5` the layer relaminarizes
or simply stays laminar; if it transitions it is by bypass, roughness, crossflow
or Görtler, none of which the model carries. But the *correct* answer there is
"stays laminar," which is what the model already returns. So the strong-FPG corner
is not where the branch is needed.

**High freestream turbulence: no.** FPG suppresses streak growth but does not
eliminate it; transition can occur with no TS at all. Explicitly out of scope.

### Consequence: the primary anchor should be Falkner–Skan, not the channel

The flow class that is (i) non-inflectional, (ii) linearly TS-unstable, (iii)
well characterised by OS-derived data, **and (iv) actually transitions that way in
practice** is the *favorable Falkner–Skan family against Drela's envelope* — not
plane Poiseuille, which fails (iv).

Drela's `dN/dRe_θ(H)` is itself built from Orr–Sommerfeld solutions and covers
the family down to the stagnation profile (`H ≈ 2.2`), so the low-H floor near
`5×10⁻³` per unit `Re_θ` that the current kernel misses is **OS-derived linear
theory, not an extrapolated fit.** This rehabilitates the original tuning: the
"low-H Falkner–Skan fit" that produced `a_visc = 0.0276` was anchored on the most
practically relevant flow class available, using OS data.

So the recommendation in file `05-*.md` should be **inverted**:

| | role | flow |
|---|---|---|
| **primary anchor** | rate `a_visc` | favorable Falkner–Skan vs the Drela (OS) envelope |
| **independent verification** | onset `B_c` | plane Poiseuille `Re_c = 5772` (exact, but not a validation case) |

And that verification already passes: the FS-anchored constants put the channel's
neutral point at `7409` against the true `5772` — `1.28×`, from a completely
independent flow with no refitting. **That agreement is the result to report**,
not "replace the FS anchor with the channel."

### One caution that follows

If the physical answer in strong FPG is "stays laminar," then a viscous branch
booking `12.4×` more amplification at `β = 0.35` could in principle transition
flows that should not. The defence is quantitative: at a floor of `5×10⁻³` per
unit `Re_θ`, reaching `N = 9` needs `ΔRe_θ ≈ 1800`, which on a real rooftop is
often beyond the chord — consistent with Drela's judgement that the low-H
mismatch does not matter much for transition location. **The branch should be
checked against that, not assumed harmless**: run the favorable wedges to
convergence and confirm the fronts do not march forward.

## 7. Proposed procedure

1. Extend `explore_pinchpoint_shooting.py` to plane Poiseuille; verify it returns
   `Re_c = 5772.22`, `α_c = 1.02056`.
2. Sweep frequency at each `Re` to build the **spatial envelope** `dN/dRe`,
   the plane-channel analogue of Drela–Giles.
3. **Check the Reynolds dependence** (§4) against the model's frozen-profile
   eigenvalue before fixing anything.
4. Anchor `a_visc` on the envelope rate and `B_c` on `Re_c` — both from one flow,
   both linear-stability eigenvalues.
5. Cross-check against Blasius (§3) and decide the (a)/(b) fork explicitly.
6. Only then rewrite the whitepaper appendix's anchor paragraph.
