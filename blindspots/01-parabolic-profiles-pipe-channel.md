# Blindspot 01 — exactly parabolic profiles: pipe and channel

**Status:** complete. **Revised 2026-08-02** after checking the actual
curvature realization in the repro chain — an earlier draft of this note drew
the opposite conclusion for the pipe. See §2, which is the substantive finding.

**Repro:** `blindspots/pipe_kernel_analysis.py` (numpy/scipy, ~20 s, no GPU),
using `paper/repro/lib/sphere_kernel.py` unmodified.

---

## 1. Why this class exists

The model's neutral locus is the parabola great circle: `u = B y + C y²` gives
`Î = Y − X − Z = 0` identically (paper Eq. `eq:parabola`). Two canonical flows
are **exactly** parabolic in wall distance, not merely near-wall parabolic:

- plane Poiseuille, `u = U_c(1 − y²/h²)`, `d = h − |y|`
- Hagen–Poiseuille, `u = U_c(1 − r²/R²)`, `d = R − r`

So the kernel's reading of both follows from algebra, with no solver and no
calibration ambiguity — the sharpest available probe of the geometric construction.

## 2. THE FINDING: two curvature realizations are in play, and they disagree

Three expressions for the curvature indicator coincide on a plane parallel layer
and **do not coincide on a curved wall**:

| realization | pipe value | where it is used |
|---|---|---|
| planar `½d² u_yy` | `−2U_c/R²` | the RP² derivation, §`sec:sphere` |
| `½d² (n̂·∇ω)`, `n̂ = ∇d` | `−2U_c/R²` | **`paper/repro/cfd/add_derived_to_slice.py:57`** |
| `½d² (∇²u)·û`, `∇²u = −∇×ω` | `−4U_c/R²` | **paper Appendix E / whitepaper Eq. `w:gram`** |

The Laplacian carries the transverse-curvature term
`(1/r)du/dr`, so it is a **factor 2** larger for Hagen–Poiseuille. The paper is
not wrong about this — Appendix E says `n̂·∇ω` is what `∇²u·û` reduces to *"on a
parallel layer… in that limit"* — but it presents them as the same object, and
they differ at relative order **δ/r₀**.

Consequence, computed:

```
REPRO   Z = ½d² n̂·∇ω  :  max Ω̂Î = 0.00000        <- exactly neutral
APP-E   Z = ½d² ∇²u   :  max Ω̂Î = 0.14961 at d/R = 0.642
```

### RESOLVED 2026-08-02 — the solver uses the full Laplacian

Read from `~/flexcompute/compute` (the compute repo is checked out locally):

- `SpalartAllmaras/SpalartAllmaras.h:305–333` forms
  `uppAi = lap(u)·û` from a compact one-ring-averaged velocity Laplacian, then
  `dwdn = −uppAi`; `SAAiTransition.h:211` sets `Z0 = −½d²·dwdn = +½d²(lap(u)·û)`.
- The source comment is explicit that the alternative was *rejected*: the compact
  Laplacian "drops the noisy nested grad|omega| + its |.| sign-kink."
- `ModelConstants.h:145` `ai_invariantKernel = 0.0` (default) and the campaign log
  `daedalus/ai_constants/case_ogrid_L2_saai_a5.log` lists no override, so
  `__aiRate` — the full-Laplacian path — is what produced every published number.

**So the paper's Appendix E is correct about the solver, and `add_derived_to_slice.py`
is the outlier.** The pipe therefore reads `max Ω̂Î = 0.1496` in the solver, not
0 — my earlier revision of this file was wrong and is retracted. Sections 5–6
below stand as *linear-theory* statements; §5's claim that the model "gets pipe
flow right" applies only to the `n̂·∇ω` diagnostic, not to the solver.

### Why it is implemented this way, and a one-line correction

The documented reason is **numerical, not physical**
(`SpalartAllmaras.h:311–318`): the compact one-ring dual-volume Laplacian is
"robust to near-degenerate triangulation" and "drops the noisy nested
grad|omega| + its |.| sign-kink." The identity `∇²u = −∇×ω` is exact for
divergence-free flow; what is an *approximation* is `∇²u·û ≈ ∂²u/∂n²`, which
holds only on a **plane** parallel layer.

The RP² derivation asks for the wall-normal Taylor coefficients
`(u, d ∂u/∂n, ½d² ∂²u/∂n²)`. In a pipe, verified exactly:

```
u  = B y + C y²   with B = 2U_c/R, C = −U_c/R²      exactly parabolic in y
ω  = ∂u/∂y                                          X and Y are both EXACT
∂²u/∂n²   = 2C  = −2U_c/R²
∇²u·û     = ∂²u/∂r² + (1/r)∂u/∂r = 4C = −4U_c/R²    only Z is corrupted
```

The parabola identity is an exact cancellation that the extra term destroys:

```
Y − X − Z  = (By+2Cy²) − (By+Cy²) − Cy²  = 0        wall-normal Z
Y − X − Z  = (By+2Cy²) − (By+Cy²) − 2Cy² = −Cy² > 0  Laplacian Z
```

The extra `(1/r)∂u/∂r` is the **area metric** of the cylindrical divergence —
momentum diffuses through shells of area `2πrL` that shrink inward. That is a
statement about viscous momentum flux, not about the shape of the profile, and
`Î` is by construction a pure profile-shape quantity (the residual of a
parabolic fit).

**Correction (verified numerically, exact for the pipe):**

```
∂²u/∂n²  =  ∇²u·û  −  (∇²d)(n̂·∇|u|)
```

`∇²d` is one further compact divergence of the wall-distance field the solver
already stores — no nested gradient of `|ω|`, no `|·|` sign-kink, so every
numerical advantage cited in the source comment is retained.

`∇²d` is the sum of principal curvatures of the wall-distance level sets: it is
`~1/R_wall` near a curved wall, but **it does not vanish in the far field** —
level sets become spherical, `∇²d ~ (n−1)/d`. So the correction matters most in
wakes, free shear layers and around vortex cores: precisely the regions this
blindspot discussion is about.

This reframes the pipe result. It is **not a blindspot of the model as derived**
— which gives `Î ≡ 0` for Hagen–Poiseuille, the correct linear answer — but an
implementation/derivation mismatch of relative order `δ/r₀`, invisible on the
validated external cases and O(1) internally. The blindspot that survives is the
**plane channel** (planar, both forms identical, `Î ≡ 0`, misses `Re_c = 5772`),
which is exactly what the two-source viscous branch repairs (file `05-*.md`).

**New action item — diagnostic fidelity.** `add_derived_to_slice.py` computes a
*different* `Ω̂Î` from the one the solver used, differing at O(δ/r₀). Every χ-sheet
figure and the spheroid flank kernel audit go through it. On an airfoil leading
edge `δ/R_LE` can reach ~0.1, and in wakes and vortex cores it is O(1). The
diagnostic should be brought into line with the solver (use the velocity
Laplacian projected on `û`), and any figure that reads a curvature-sensitive
region should be re-checked.

**Below, the "repro is right" reading is superseded.** Hagen–Poiseuille is
linearly stable at every Reynolds number, so "exactly neutral" is the correct
linear-stability result — and the model gets it for the right structural reason:
the pipe profile sits on the parabola locus, and the `n̂·∇ω` realization
preserves the identity that defines that locus. The `∇²u` realization *breaks*
the identity that makes a parabola neutral, purely from wall curvature.

Scaling of the discrepancy:

| flow | δ/r₀ | effect | consequence |
|---|---|---|---|
| slender external body (spheroid mid-body, Re_L=7.2e6) | ≈ 0.016 | ~1.6 % in `Z` | **every published result is unaffected** |
| airfoil leading edge (δ/R_LE) | few % | small | unaffected |
| body of revolution near a closing tail | → O(1) | large | untested |
| pipe / duct | O(1) | factor 2 | flips the answer |

Sign note (this caught me out and is worth stating): `Î = (Y − X − Z)/R`, so `Z`
enters **negatively**. Positive curvature stabilizes; it is *excess negative*
curvature that destabilizes. A parabola carries exactly the amount of negative
`Z` that balances `Y − X` to zero — that *is* the neutral locus. The `∇²u` form
gives the pipe twice that amount, over-balancing it to `Î > 0`.

### Action for the paper

State explicitly which realization is canonical. The derivation's neutral locus
is the planar parabola in wall distance, and `½d²(n̂·∇ω)` is the realization
consistent with it; `∇²u·û` is a convenience that drifts from it at O(δ/r₀).
**Also needed: confirm what the Flow360 solver actually evaluates** — the header
(`SAAiTransition.h`) is in the compute repo and is not on this machine, so this
note cannot close that. If the solver uses `∇²u`, the external results still
stand (1.6 %), but the internal-flow statement inverts.

## 3. Plane Poiseuille — the genuine blindspot

```
max |Î| over the section = 1.8e-13      (machine zero)
max |P|                  = 1.3e-13
```

Both realizations agree here (planar flow), so this is realization-independent.
The channel sits **on** the neutral circle at every Reynolds number; the model
produces identically zero amplification.

Truth: plane Poiseuille is linearly **unstable** at `Re_c = 5772.22`
(Orszag 1971) — a purely **viscous** Tollmien–Schlichting instability, with no
inflection point anywhere. This is the real, exact, unfixable-by-tuning miss.

## 4. Why the channel is unstable at all — and what it says about `Î`

Rayleigh's theorem rules out *inviscid* instability without an inflection point.
It says nothing about viscous instability, and viscosity is not purely damping:
the phase shift it introduces in the critical layer and the wall layer permits a
net Reynolds stress `−⟨u'v'⟩` that feeds the wave. Blasius has **no inflection
point either** (`u''_w = 0`, `u'' < 0` throughout) and is TS-unstable — the whole
`e^N`/NLF enterprise rests on a viscous instability of a non-inflectional profile.

So `Î` is not a Rayleigh criterion. It is the residual of a **parabolic fit**
(the accumulated third derivative, Eq. `eq:g_integral`), which is nonzero for
Blasius (`max Ω̂Î = 0.078`) and is calibrated against the Drela–Giles envelope —
a correlation that already contains viscous TS growth. On the Falkner–Skan
family the inviscid-inflectional content and the viscous-TS rate **co-vary
monotonically**, so one shape coordinate carries both.

The parabola is the single profile where they **decouple**: departure-from-
parabolic exactly zero, viscous TS growth not zero. That is precisely where the
model returns zero.

> **The single coordinate `Ω̂Î` conflates the inviscid-inflectional and
> viscous-TS mechanisms, which co-vary on the family it was calibrated against.
> Exactly parabolic profiles are where they separate, and there the model has no
> amplification channel at all.**

## 5. Why the pipe transitions, and why the model is right not to predict it

Hagen–Poiseuille is stable to *all* infinitesimal disturbances — not inviscidly,
not viscously (established numerically to very high Re). Its transition is
**subcritical**: it requires a finite-amplitude disturbance, threshold amplitude
scaling roughly as `Re⁻¹` (Hof, Juel & Mullin 2003 — *recalled, verify*). The
route is non-modal transient growth (lift-up: streamwise vortices → streaks,
algebraic amplification), secondary instability of the streaks, then puffs; and
*sustained* turbulence requires puff splitting to outpace puff decay, giving
`Re_c = 2040 ± 10` (Avila et al. 2011).

None of that is in the linear-envelope class, so an `e^N`-type model should
return zero — and SA-AI does. **This is a scope statement, not a defect.**

Literature confirmed this session:

- **Reynolds (1883)** — with an undisturbed inlet, natural transition near
  **Re ≈ 13 000**; the familiar "≈2000" is the value below which disturbances
  always decayed, not his observed natural transition.
- **Ekman (1910)** — laminar to **Re ≈ 44 000** using Reynolds' apparatus.
- **Pfenniger (1961)** — laminar to **Re ≈ 100 000**.
- **Wygnanski & Champagne (1973)**, JFM 59:281 — natural slugs above
  **Re ≈ 5×10⁴**; puffs detectable to **Re ≈ 1500–2000**.
- **Avila et al. (2011)**, Science 333:192 — **Re = 2040 ± 10**, from the
  crossover of puff-decay and puff-splitting timescales.

The spread from 2040 to 100 000 across facilities is itself the point: pipe
transition is set by disturbance amplitude, not by a linear threshold. A model
with a linear amplification channel cannot and should not reproduce it.

## 6. The route the model *does* have: the entrance region

The laminar entrance length is `x_e/D ≈ 0.05 Re_D`, and the developing wall
layer there is Blasius-like — squarely inside the calibrated channel. The
entrance route beats the (absent) fully-developed route above:

| N_crit | Re_θ,tr | Re_x,tr | entrance route active above |
|---|---|---|---|
| 9 (quiet) | 1108 | 2.8e6 | Re_D ≈ 7 500 |
| 7 | 908 | 1.9e6 | Re_D ≈ 6 100 |
| 5 (noisy) | 708 | 1.1e6 | Re_D ≈ 4 800 |

So the model's complete pipe prediction is: **laminar in the fully developed
region at every Reynolds number; transition possible only in the entrance
boundary layer, above `Re_D ≈ 5 000–7 500` depending on disturbance level.**
Set against measured natural transition of 13 000 (Reynolds) to 100 000
(Pfenniger), that is a defensible, conservative answer from the right mechanism.

## 7. Does inflection help? — yes, strongly, and the model would see it

Physically, inviscid inflectional growth is O(ΔU/δ) — the Kelvin–Helmholtz
scale, the model's own `a_max = 0.19` — while viscous TS growth is smaller by a
factor scaling like `Re^{-1/2}`. Any duct feature that produces a real
inflection point therefore dominates: divergence/diffusion, expansions, bends,
swirl, an adverse gradient, developing flow. (Curvature-driven Dean and Görtler
vortices are *centrifugal*, not inflectional, and the model has no channel for
them — same gap as Görtler on concave walls.)

In the model, the rate is linear in `Ω̂Î` with the KH ceiling at the vorticity
pole, so a modest inflection is worth a lot:

| profile | `max Ω̂Î` | rate `a = 0.19·Ω̂Î` | vs Blasius |
|---|---|---|---|
| parabola (channel, pipe) | 0 | 0 | — |
| Blasius | 0.078 | 0.0148 | 1× |
| moderate adverse (β=−0.10) | 0.162 | 0.0308 | 2.1× |
| separation limit | 0.510 | 0.0969 | **6.6×** |

**This bounds the practical reach of the blindspot.** Real internal flows of
engineering interest — diffusers, bends, expansions, entrance regions, anything
with a streamwise pressure gradient — are essentially never exactly parabolic,
and the model would see them through the geometric construction with no new
term. The blindspot's true footprint is *exactly parallel, exactly parabolic,
fully developed duct flow*: plane Poiseuille, Hagen–Poiseuille, and little else.

That is simultaneously a narrow limitation and a real one, because fully
developed duct flow is the base state that every internal-flow perturbation is
measured against. A cheap positive test that would demonstrate the good half:
a 2D planar diffuser, where a genuine inflection appears and the model should
respond sharply and correctly.

## 8. Suggested paper statement

> The model's neutral locus is the parabola, and plane Poiseuille is exactly
> parabolic in wall distance: the model reads the channel as neutral at every
> Reynolds number and does not reproduce its `Re_c = 5772` Tollmien–Schlichting
> instability. This is structural. The coordinate `Ω̂Î` measures departure from
> the osculating parabola, a proxy that carries both the inviscid-inflectional
> and the viscous-TS content because the two co-vary across the Falkner–Skan
> family it is calibrated on; the exactly parabolic profile is where they
> separate. Hagen–Poiseuille is likewise exactly parabolic and is read as
> neutral, which is here the correct answer — pipe flow is linearly stable at
> all Reynolds numbers and its transition is subcritical and finite-amplitude,
> outside the envelope class. Amplification in a pipe is available to the model
> only through the developing entrance layer, above `Re_D ≈ 5×10³`. Any duct
> feature producing a genuine inflection — divergence, expansion, adverse
> gradient — restores the model's amplification, at up to `6.6×` the Blasius
> rate at the separation limit.
