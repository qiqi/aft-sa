# Results: four analytic bounds, and the differentiability audit

Repro: `python3 -u blindspots/exact_profile_bounds.py` (numpy/scipy, ~15 s, no
GPU). Same instrument as file `01`: the canonical kernel from
`paper/repro/lib/sphere_kernel.py`, evaluated on profiles whose linear-stability
answer is exact, and on exact perturbations of profiles the campaign already
computed. **No CFD was run for any number below.**

Validation of the instrument, before anything else: Blasius `f''(0) = 0.33206`,
Hiemenz `f''(0) = 1.23259`, spanwise `g'(0) = 0.57047`, and the kernel's Blasius
peak `max P = 0.0781` against the `0.078` file `01` quotes. All four agree.

## Headline

| # | Question | Answer |
|---|---|---|
| 1 | Does frame rotation break the kernel? | **No.** Solid-body rotation is read as *stable* exactly (`P < 0`), and it is the `|u|` term that does it. A genuine layer's rate shifts by `41 x (ell/L)`: **1.3 %** on the NLF, **6.6 %** on the Eppler, **12 %** at the Eppler's low-`Re` end, at the pessimistic `Omega = U/c`. |
| 2 | Is the kernel blind to wall curvature? | **No — `02` §5 is wrong.** Curvature enters through the metric term in `Z`. Already present in every computed case: **−0.17 %** (NLF), **−0.4/−0.5 %** (Eppler), **−0.09 %** (cylinder at `Re_D=1e8`), rising to **−10 %** at `Re_D=1e4` and **total extinction** at `Re_D <= 3e2`. Concave flips the sign. |
| 3 | Attachment line? | **`02` §5 is wrong here too.** The kernel reads the swept-Hiemenz spanwise profile as amplifying at **82 %** of the Blasius rate and fires at **`Rbar = 601`** against the linear-stability **583** — unfitted. |
| 4 | Asymptotic suction layer? | Read as **stable at every Reynolds number** (`P < 0` strictly), against a true critical `Re_d* ~ 5.44e4`. Conservative error; over-credits HLFC suction. Also **closes the suction half** of the transpiration item (`02` §3). |
| 5 | Differentiable? | **Continuous everywhere, differentiable almost everywhere** — confirmed. But there are **five** non-smooth surfaces, not one, and the deliberate handover max is not the important one. |

Two of the five items in `02` §5 turn out to be wrong as written. That file's
"statements, not campaigns" list was never checked against the algebra; three
of these four bounds took an afternoon.

---

## 1. Frame rotation

**Structure.** Rotation enters the triple *only* through the vorticity:
`Y = d|omega|` picks up `2 Omega d`, while `X = |u|` is untouched and `Z` is
untouched (the Laplacian of a solid-body field is zero).

**1a. Solid-body rotation is read as stable, exactly.** With
`X = Omega r`, `Y = 2 Omega d`, `Z = 0`:

| `d/r` | `P` | rate |
|---|---|---|
| 1e-4 | −2.000e-4 | 0 |
| 1e-3 | −1.996e-3 | 0 |
| 1e-2 | −1.959e-2 | 0 |
| 1e-1 | −1.538e-1 | 0 |

`g = (Y − X − Z)/R < 0` because `Y/X = 2d/r << 1`, so `P <= 0` and the rate is
clipped to zero. **The `|u|` term is what saves it.** This is the SA-AI
counterpart of the property baseline SA relies on — production built on
vorticity, which vanishes in an irrotational freestream — reached by a different
route: under frame rotation the vorticity does *not* vanish, but the rate still
does.

*(Attribution note: the user asked for Spalart's exact words on freestream
vorticity from the original paper. Four searches did not surface a verbatim,
accessible quote — AIAA 92-0439 is paywalled and the secondary reviews
paraphrase. The substance is uncontroversial and well attested, but the sentence
must be pulled from the paper itself before it is quoted. `references.bib` has
the entry; no PDF is in the repo.)*

**1b. A genuine layer on a rotating body.** In the normalized triple
(`U_e = 1`, `y` in units of `ell = sqrt(nu x/U)`) the perturbation is
`dY = 2 eta (ell/L)` for `Omega = U_inf/L`, one small parameter, entering exactly
as the curvature term of §2 does, with either sign depending on the sense of
rotation. Propagated through to the rate:

```
d(max P)/P  =  41.33 * (ell/L)
```

while the shear itself only moves by `dY/Y = 9.61 (ell/L)` at the peak-`P`
height. **The kernel amplifies the perturbation by a factor 4.3.** That is a
finding in its own right: `g = (Y−X−Z)/R` is a near-cancellation in the
amplifying band, so relative perturbations of the shear indicator arrive at `P`
magnified several-fold. It is the same conditioning property that makes a
`ell/R ~ 1e-3` curvature term visible at all.

| case | `Re_L` | `x/c` | `Re_theta` | `ell/c` | `dP/P`, `L=c` | `dP/P`, `L=10c` |
|---|---|---|---|---|---|---|
| NLF(1)-0416, α=0 | 4.0e6 | 0.39 | 829 | 3.12e-4 | +1.29 % | +0.13 % |
| Eppler 387 | 2.0e5 | 0.50 | 210 | 1.58e-3 | +6.56 % | +0.65 % |
| Eppler 387, low-`Re` end | 6.0e4 | 0.50 | 115 | 2.89e-3 | +12.00 % | +1.19 % |
| Daedalus mid-span | 5.0e5 | 0.50 | 332 | 1.00e-3 | +4.14 % | +0.41 % |
| thick layer, `Re_L=1e3` | 1.0e3 | 0.50 | 15 | 2.24e-2 | +95 % | +9.3 % |

`ell/c = sqrt((x/c)/Re_c)` and `theta/c = Re_theta/Re_c` are the same small
parameter, so the bound **is** `O(Re_theta/Re_L)` as the user proposed. Two
columns because `L` is the *rotation* scale: `L = c` means one radian of frame
rotation per chord of travel (deliberately pessimistic); `L = 10c` is a
rotor-like chord-to-radius ratio.

**Reading.** The bound is real, small at flight Reynolds number, and **not
negligible at the low-`Re` end** — which is exactly where the small-rotor
application the introduction advertises lives. A drone rotor at chord `Re = 6e4`
carries a ~1 % rate error from frame rotation alone. Nothing here threatens the
campaign (no case rotates), and nothing here needed a rotating-frame CFD run.
What it does is convert "the kernel has no rotation sensor" from an unquantified
gap into a number that scales, plus a statement of where the number stops being
small.

---

## 2. Wall curvature — and a correction to `02` §5

`02` §5 says: *"Görtler / centrifugal instability. Wall curvature radius enters
no sensor; concave surfaces will be predicted too laminar."* **The first clause
is false.** Curvature enters through the metric term in the curvature indicator,
exactly as transverse curvature does in the pipe of file `01`:

```
convex   (fluid at r = R+d):   lap(u).uhat = u'' + u'/(R+d) - u/(R+d)^2
concave  (fluid at r = R-d):   lap(u).uhat = u'' - u'/(R-d) - u/(R-d)^2
leading term:                  dZ = +- (d/2R) Y
```

`+` convex (stabilizing, `g` decreases), `−` concave (destabilizing).
**`n.grad|omega|` gives the same leading term for streamwise curvature**, so
unlike the pipe this result does *not* depend on which curvature realization the
solver uses — verified in the script's docstring algebra.

Sensitivity, both signs:

| `ell/R` | `dP/P` concave | `dP/P` convex |
|---|---|---|
| 1e-4 | +0.048 % | −0.048 % |
| 1e-3 | +0.481 % | −0.481 % |
| 1e-2 | +4.80 % | −4.81 % |
| 3e-2 | +14.36 % | −14.48 % |
| 1e-1 | +47.3 % | −48.6 % |

The computed cases (`R` splined from the coordinate file where we have one; all
convex, so all stabilizing):

| case | `Re_L` | `x/c` | `R/c` | `ell/R` | `dP/P` |
|---|---|---|---|---|---|
| NLF(1)-0416 upper, α=0 | 4.0e6 | 0.39 | 0.91 | 3.45e-4 | −0.166 % |
| Eppler 387 upper, pre-sep | 2.0e5 | 0.40 | 1.35 | 1.05e-3 | −0.503 % |
| Eppler 387 upper, sep point | 2.0e5 | 0.52 | 2.02 | 7.97e-4 | −0.383 % |
| cylinder, `Re_D=1e10` | — | θ=92° | 0.50 (`/D`) | 1.79e-5 | −0.009 % |
| cylinder, `Re_D=1e8` | — | 95° | 0.50 | 1.82e-4 | −0.088 % |
| cylinder, `Re_D=1e6` | — | 100° | 0.50 | 1.87e-3 | −0.898 % |
| cylinder, `Re_D=1e4` | — | 120° | 0.50 | 2.05e-2 | **−9.87 %** |
| cylinder, `Re_D=3e2` | — | 140° | 0.50 | 1.27e-1 | **−62 %** |
| cylinder, `Re_D=1e2` | — | 155° | 0.50 | 2.33e-1 | **−100 %** |
| spheroid, `x/L=0.3` (transverse `r0`) | 1.5e6 | 0.30 | 0.076 | 5.86e-3 | −2.82 % |
| spheroid, `x/L=0.6` | 1.5e6 | 0.60 | 0.082 | 7.75e-3 | −3.73 % |
| spheroid, `x/L=0.9` | 1.5e6 | 0.90 | 0.050 | 1.55e-2 | −7.46 % |

**Four readings, in order of importance.**

1. **This is not a missing term — it is already in the computed answers.** The
   solver builds `Z` from the actual curved geometry. Every number above is a
   *decomposition* of what the kernel already did, not a correction to apply. So
   the airfoil numbers say: less than half a percent of each computed
   amplification rate is the curvature metric rather than the profile shape.
   Negligible against the campaign's own 0.01–0.02 c mesh-to-mesh scatter.

2. **It is not negligible at the low-`Re` end of the cylinder traverse.** At
   `Re_D = 1e4` the curvature metric removes 10 % of the rate; at `Re_D <= 3e2`
   it removes essentially all of it, and `ell/R = 0.13–0.23` means it is no
   longer a perturbation at all. The paper claims that traverse over ten
   decades, so the honest remark is that at its low-`Re` end the transition the
   kernel reports is set as much by the wall's curvature metric as by the
   profile shape. This is about *why*, not *whether* — the traverse did produce
   a transition angle there.

3. **On a body of revolution the transverse term is realization-dependent**,
   which is file `01`'s open inconsistency. So the spheroid's −3 to −7 % is
   better read as a *realization uncertainty* in the rate than as a bias. (File
   `01` quotes ~1.6 % for the spheroid mid-body; that is `delta*/r0` against
   `ell/r0 = 0.8 %` here — the same geometry in a different thickness measure,
   not a disagreement.)

4. **Görtler: right sign, wrong law.** The concave response exists and
   destabilizes, so the kernel is not blind to Görtler-type geometry. But its
   response is *linear* in `delta/R`, whereas the centrifugal instability is
   governed by `G = Re_delta sqrt(delta/R)`. At fixed `delta/R` the physical
   instability strengthens with Reynolds number and the kernel's response does
   not move at all, so the gap widens with `Re`. That is the correct replacement
   for `02` §5's sentence.

---

## 3. Attachment line — swept Hiemenz

Exact similarity solution; on the attachment line the chordwise velocity
vanishes, so the triple is built from the spanwise profile alone:
`X = g`, `Y = eta g'`, `Z = (1/2) eta^2 g''`, and `Re_Omega = Rbar eta^2 g'`,
with `Rbar = W_e delta / nu`, `delta = sqrt(nu/a)`.

`g'' = -f g' < 0` everywhere: **no inflection point.** And yet:

| `Rbar` | `max P` | `Re_Omega` at peak | `Re_Om,crit` | `max a` |
|---|---|---|---|---|
| 100 | +0.0639 | 76.6 | 461.8 | 1.02e-4 |
| 245 | +0.0639 | 188.7 | 463.3 | 3.94e-4 |
| **583** | +0.0639 | 449.5 | 463.7 | 5.50e-3 |
| 1000 | +0.0639 | 757.1 | 459.5 | 1.18e-2 |
| ≥3000 | +0.0639 | — | 458.9 | 1.214e-2 (saturated) |

**The kernel fires at `Rbar = 601`.** Linear stability of swept Hiemenz (Hall,
Malik & Poll) puts the critical value near **583**, and Poll's transition
criterion near 650. The agreement is within a few percent and was **not fitted**
— the threshold's only free scale is the Blasius `N=1` anchor.

So `02` §5's *"no concept of"* attachment-line transition is wrong. The kernel
reads the attachment-line profile as an amplifying shear layer: `max P = 0.0639`
against the Blasius `0.0781`, saturated rate `1.214e-2` against the Blasius
`1.484e-2` — **82 % of the Blasius rate** — switching on at very nearly the right
`Rbar`.

**Why it can work.** The attachment-line boundary layer *is* a shear layer and
its primary instability is a viscous instability of that layer, so a kernel built
to read viscous instability of a shear profile is not being asked for anything
foreign. What it cannot know is the crossflow content away from the attachment
line.

**Caveats that must travel with the number.**
- The reduction assumes the attachment line exactly at `x=0`; away from it the
  chordwise strain enters and this triple is not the whole story.
- The real Görtler–Hämmerlin mode has chordwise structure a wall-normal-profile
  kernel cannot represent. Agreement on the *threshold* is not agreement on the
  *mechanism*.
- The confound of `02` §5 is now **sharper, not resolved**: standard SA's
  spurious attachment-anchored branch and this genuine attachment-line reading
  would both trip a swept leading edge, and separating them in a computed case
  still needs file `07`'s quench or a protocol that excludes the spurious branch.
- `Rbar = 583` and `650` are from recall and **must be checked against Hall,
  Malik & Poll and Poll** before either is quoted.

---

## 4. Asymptotic suction layer

`u = 1 − exp(−y/delta)`, `delta = nu/|v_w|` = displacement thickness.

```
max over the profile of P = 0 (attained only at the wall)
P at s = 0.5, 1, 2, 4     = -1.75e-2, -5.36e-2, -1.02e-1, -5.70e-2
```

`P <= 0` everywhere and strictly negative away from the wall, so the **rate is
identically zero and no Reynolds number can make this profile amplify.** The
gate does open (it reaches 1 by `Re_d* = 1e4`) but it multiplies a zero — which
is why quoting a "critical `Re`" from the gate alone would be wrong, and worth
remembering as a general trap in this kind of audit.

So the suction layer joins the parabola family in the kernel's **stable class**,
against a true critical `Re_d*` of about `5.44e4` (roughly 105× the Blasius 520
— verify against Hocking). The error is **conservative** (too laminar), and its
practical face is that the model will **over-credit HLFC suction**: a
suction-shaped profile is read as indefinitely laminar.

Two more things this profile settles for free:

- It is the exact realization of the transpiration concern of `02` §3
  (`u'''_w = (v_w/nu) u''_w != 0`). The worry was that `P` might go **positive at
  the no-slip line**, which the paper states never happens. Here `P(0) = 0` and
  `P < 0` just off the wall, so for **suction** the feared sign flip does not
  occur. **That closes the suction half of item 3**; blowing (`v_w > 0`) is the
  half still open.
- With plane Poiseuille and plane Couette it fixes the shape of the stable
  class: full, single-signed-curvature, inflection-free profiles are read as
  stable. That is the model's design intent; the ASBL is the case where the
  intent costs a real, if weak, instability.

---

## 5. Differentiability audit

**Confirmed: the model is continuous everywhere and differentiable almost
everywhere.** The 90 % prior was right. But the non-smooth set has **five**
members, and the deliberate handover max is the least consequential of them.
Read from `SAAiTransition.h` and `SpalartAllmaras.h` (canon branch,
`aVisc = 0`, `reOmBc = 0`, `maxBlend > 0.5`).

| # | Where | Surface | Active in | Deliberate? |
|---|---|---|---|---|
| 1 | `Pp = (P>0) ? P : 0` — rate | `P = 0` | **every case**: the whole amplification boundary | no |
| 2 | `aLin = (Pp<1) ? Pp : 1` — rate ceiling | `P = 1` | free shear: the cylinder wake, mixing layers | no |
| 3 | `__aiIsTurb`: `chi <= switchCenter ? 0 : 1-exp(...)` | `chi = 1` | every transition front | no — and it is a kink **in the solution variable** |
| 4 | `max((1-isTurb) P_ai, isTurb P_sa)` | crossover | handover | **yes**, the one you named |
| 5 | `omegaMag = |omega|` | `omega = 0` | freestream | inherited from baseline SA |

Notes that matter:

- **#1 is the dominant one.** The boundary between amplifying and
  non-amplifying flow is a codimension-1 surface threaded through every
  solution, and the rate is `C^0` but not `C^1` across it. Any adjoint or
  Newton linearization sees a discontinuous derivative there.
- **#3 is a kink in `chi`, i.e. in `nu_tilde` itself.** `isTurb` has derivative
  `0` for `chi < 1` and `1/(switchWidth * nu)` at `chi = 1+`. That is the worst
  placement for an implicit solve, because it is a kink in the residual as a
  function of the unknown, not of the geometry.
- **#4 is easy to soften and you already have the tool.** The codebase's own
  `softmin_2(x,y) = xy/sqrt(x^2+y^2)` and `softmax2(x,y) = sqrt(x^2+y^2)` forms
  are exactly the right replacements, and #1/#2 can use the same, so all three
  soften with machinery already in the file and no new constants beyond a width.
- The `Pf = max(Pp,1e-6)` floor inside the threshold is a sixth kink but is
  **masked**: where it acts the rate is `O(1e-6 * a_max)`, so it multiplies a
  zero. Not worth softening.
- `sqrt(X^2+Y^2)` and `sqrt(X^2+Y^2+Z^2)` are smooth except at the origin, which
  is approached at the wall; `P` is degree-0 homogeneous so it has a directional
  limit there, and the physical approach path (`X/Y -> 1`, `Z/Y -> 0`) is
  definite, so this is removable rather than a real kink.
- The invariant-kernel form's `s_hat = (omega x n)/|omega x n|` is ill-defined
  where `omega || n` or `omega = 0`, but `Y -> 0` there so the product stays
  continuous.
- Wall distance `d` is non-differentiable across the medial axis — geometric,
  inherited, and the multi-element item of `02` §1 already flags it.

**One thing that is not about smoothness at all, and matters more than any of
the above for gradient-based design:** `SpalartAllmaras.h` clips the stored
Jacobian, `production.grad[0] = min(Pai.grad[0], 0)`. So the linearization the
solver carries is deliberately *not* the derivative of the residual. A discrete
adjoint built on it would be inconsistent regardless of how smooth the residual
is. This is standard practice for SA positivity and baseline SA does it too, but
it is the first thing to fix if the model is ever driven by an adjoint, and it
should be stated rather than discovered.

---

## Verification debt

- Every number above is reproducible: `python3 -u blindspots/exact_profile_bounds.py`.
- **Literature values used as comparison targets are from recall and must be
  checked before they are quoted in the paper:** swept-Hiemenz critical
  `Rbar ≈ 583` (Hall, Malik & Poll) and Poll's `~650`; ASBL critical
  `Re_d* ≈ 5.44e4` (Hocking); Blasius `Re_d*_crit = 520`. The Spalart freestream
  vorticity quote was **not** retrieved (see §1a).
- The attachment-line result is the one most worth a second pair of eyes: it is
  a *positive* result, it is the kind of agreement that can be coincidence, and
  it rests on a reduction (§3 caveats) that a reviewer will probe.
