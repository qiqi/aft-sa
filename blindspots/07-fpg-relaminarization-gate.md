# Favorable-pressure-gradient stabilization of baseline SA

> ## TESTED 2026-08-06 — step 2 PASSES, step 1 still open
>
> Executed on the frozen-Hiemenz rig via
> `paper/repro/analytic/stagnation_kgate_test.py` (gate added to
> `stagnation_bistability.run_case` behind `K_crit=None`, ungated path asserted
> bit-identical). Data: `paper/data/stagnation_kgate_test.json`.
>
> **The gate removes the attachment-anchored root, completely, at both Reynolds
> numbers tested and across the whole literature range of `K_c`.**
>
> | `L` | `Re_r` | ungated | gated (`K_c` = 2.5/3.0/3.5e-6) |
> |---|---|---|---|
> | 688.5 (critical) | 4.7e5 | sustained, `χ`=15.3 | **collapsed**, `χ`~3e-4, all three |
> | 3000 | 9.0e6 | sustained, `χ`=75.67 | sustained, `χ`=75.61 |
>
> The `L=3000` row looks like a failure and is not one. Localizing the state
> settles it — near-wall `χ` versus `x`, ungated → gated:
>
> | `x` | 50 | 200 | 577 | 1000 | 2900 |
> |---|---|---|---|---|---|
> | ungated | 1e-12 | **2.24** | **11.9** | 24.3 | 73.7 |
> | gated | 1e-123 | 1e-95 | 9e-15 | 22.4 | 73.6 |
>
> Inside the quench zone (`x < 577`) the fraction of stations carrying `χ > 1`
> goes from **76 % to exactly 0**. The nose is annihilated. What survives at
> `L=3000` is the layer at `x ≳ 1000`, outside the quench zone, where `K < K_c`
> and the flow is genuinely not relaminarizing — a turbulent layer SA *should*
> sustain, and never the trap. The rig's pass/fail criterion is `max χ` over the
> whole domain, so it stops testing the trap once `L` exceeds the quench
> station; that is a limitation of the test, not of the gate.
>
> **Why the quench zone does not grow with `L`:** the closed form is exact here.
> On the edge of this field `U_e = k x` gives `K = ν/(k x²)`, i.e. `K = 1/x̂²`
> in rig units, so the quench station sits at `x = 1/√K_c = 577δ` regardless of
> `L`, while the domain grows as `L = √Re_r`. Gated fraction of the layer falls
> from 86 % at `L=688` to 26 % at `L=3000`. Since Hiemenz convects *outward*
> (`u = x f' > 0`), the ungated outer region is downstream and cannot be cleared
> from within the quench zone — correctly.
>
> **First evidence on inertness (§6), favourable but not the real test.** At the
> outer station `x=2900`, `q ≥ 0.991` everywhere above `y = 0.2δ`; only the two
> wall-adjacent rows are gated, and `χ = 0` there by boundary condition anyway.
> So the `K_loc ~ 1/(2 f''₀ x² y)` divergence at the wall is real but is
> confined below `0.2δ` in this flow. **This does not discharge step 1**: a
> Hiemenz layer is not a ZPG log layer, and the inertness that has to be
> demonstrated is on the flat plate and the NLF rooftop.
>
> **What this does and does not license.** It does *not* license dropping the
> protocol. `χ = 0` remains an exact fixed point, so at `L=3000` two stable
> states still exist with the gate on — but the second is SA's known seedless
> sustainment of an already-turbulent layer (`analytic/sa_sustain.py`), a
> different mechanism the gate is not meant to touch. So §8's "the
> initialization protocol becomes unnecessary" is right about the attachment
> trap and wrong as stated about the model problem as a whole.
>
> Verification order below, updated: **step 2 done and passed**; step 1 is now
> the deciding one; steps 3–5 unchanged.


**Goal:** remove the attachment-anchored spurious turbulent branch
(paper §`sec:bistability`) so the model determines its own fixed point, instead
of excluding the branch by an initialization protocol.

This is a *simplification* in the sense that matters: today the printed transport
equation does not determine which steady state an implementation finds, and a
reader porting the model must reproduce a convergence ritual. A quench gate makes
the equation self-sufficient. That is worth one constant.

---

## 1. The problem, restated

Standard SA sustains a turbulent wedge anchored at a leading-edge attachment
point, with no help from the amplification kernel and — above a critical nose
Reynolds number — none from the freestream either. The paper's frozen Hiemenz
model problem bisects that critical band at `Re_r = 4.66–4.74 × 10⁵`
(Table `t:stagbistab`). The branch belongs to SA, not to SA-AI, so removing it
means modifying SA's behaviour at high `χ`.

## 2. The physical gate: acceleration quench

Real turbulent boundary layers relaminarize under strong acceleration. The
governing parameter is

```
K = (ν/U_e²) dU_e/ds
```

with the accepted quench threshold `K_crit ≈ 3×10⁻⁶` (Launder 1964; Narasimha &
Sreenivasan 1979 give 3–3.5×10⁻⁶ for complete reversion). **This constant is
taken wholesale from the literature**, in the same spirit as Mack's `Tu → N_crit`
map and Michalke's `a_max = 0.19` — not fitted here.

`K` diverges at an attachment point (`U_e → 0` while `dU_e/ds → a`, the strain
rate), which is exactly where the spurious branch anchors.

## 3. A purely local form

`K` as written needs an edge velocity and a streamwise derivative — both
non-local. But the directional derivative of speed along the streamline *is*
`dU/ds`, so

```
K_local = ν ( û · ∇|u| ) / |u|²
```

is local, needs no pressure and no edge quantities, and coincides with `K` in a
thin layer. It uses the same `|u|` convention the kernel already uses (relative
to the nearest wall point), so it inherits the same Galilean treatment.

Equivalently, via `ρU dU/ds = −dp/ds`, it relates to the legacy local
pressure-gradient sensor `λ_p = −d²(u·∇p)/(ρν|u|²)` already present in
`ModelConstants.h` (currently unused by the sphere kernel) as

```
K = λ_p / Re_d² ,      Re_d = |u| d / ν
```

so either route is available with machinery that already exists.

## 4. Where to apply it

Gate the **standard-SA production only** — the branch is sustained by
`P_SA`, not by `P_AI`:

```
P = max[ (1 − σ_P) P_AI ,  σ_P · Q(K) · P_SA ]
Q(K) = smooth ramp, 1 for K ≪ K_crit, 0 for K ≫ K_crit
```

Leave `P_AI` alone: the amplification branch is already geometrically
extinguished in favorable gradients (`Ω̂Î → 0`), and if the two-source viscous
branch is revived (file `05-*.md`) it must be checked separately (§7).

**Confine it to attached wall layers.** `K` is a wall-layer parameter; applied
pointwise in the outer flow it could quench things it should not. The `q` sensor
from the `f_v1` bypass (paper §`sec:fv1bypass`) is exactly the discriminator
already in hand — `q = 1` in an attached equilibrium layer, `q ≫ 1` in a lifted
one. Multiplying the quench by a `q ≈ 1` indicator reuses existing machinery and
adds no constant.

## 5. Does it reach the branch? A quantitative check

For potential flow over a cylinder, `U_e = 2U_∞ sin θ`, `s = (D/2)θ`, so near the
attachment point `U_e ≈ 2U_∞θ` and `dU_e/ds = 4U_∞/D`, giving the clean result

```
K = 1 / (Re_D θ²)        ⇒        θ_quench = 1 / √(Re_D K_crit)
```

| `Re_D` | `θ_quench` |
|---|---|
| 10⁵ | 106° (whole forebody) |
| 4.7×10⁵ (the paper's bisected band) | **48°** |
| 10⁶ | 33° |
| 10⁸ | 3.3° |

At the Reynolds number where the paper bisects the spurious branch, the quench
zone covers ~48° of arc — comfortably the whole region in which the wedge
anchors. And the zone **shrinks as `Re` rises**, which is the correct trend:
relaminarization gets harder at high Reynolds number, and the model should not
be quenching a genuine high-`Re` turbulent nose.

## 6. Margin against equilibrium turbulent layers

A zero-pressure-gradient turbulent boundary layer has `K = 0`. A favorable
rooftop on an NLF section runs `K ~ 10⁻⁷–10⁻⁸`, one to two orders below
`K_crit` — the margin the paper already anticipates. The gate should therefore be
inert everywhere the turbulent calibration lives, which is the same style of
argument as §`sec:calib_diff` makes for `c_ν,ai` and the handover ties, and should
be verified the same way (evaluate `K` through the log layer on the flat plate
and the NLF rooftop and confirm `Q ≡ 1` to round-off).

## 7. Two interactions to check before adopting

1. **Swept attachment lines — a feature, not a bug.** Because `K_local` uses the
   local `|u|`, which on a swept leading edge retains the spanwise component,
   `|u|` does not vanish along the attachment line and `K` stays finite. So the
   gate does **not** fire there, and genuine leading-edge contamination is not
   suppressed. This is a real discriminator the present protocol lacks: today,
   spurious attachment turbulence and genuine contamination are indistinguishable
   (file `02-*.md` §5). Worth verifying explicitly on a swept case.

2. **The two-source viscous branch at stagnation.** `P_curv = Ω̂⟨−Ẑ⟩₊` is
   *large* on a filled Hiemenz profile (`u'' < 0` throughout), so reviving the
   viscous branch risks igniting the stagnation region — the opposite of what
   this gate is for. The FPG study's `β = 1` (Hiemenz, `H = 2.216`) row is the
   test; commit `5a0a47e` already records "two-eps variant infeasible at
   stagnation," so this is a known sharp edge. **Check `β = 1` in the
   decoupled+ceiling form before adopting either change.**

## 8. What it buys, and what it costs

**Buys:** the initialization protocol becomes unnecessary; the printed equations
determine the answer; the bistability concession in §`sec:bistability` and the
"a reader implementing the model in any solver will need it in some form"
caveat both go away; and the model gains genuine relaminarization physics, which
is currently a listed gap (file `02-*.md` §5) relevant to accelerating ducts,
turbine passages and high-lift reattachment.

**Costs:** one constant, `K_crit ≈ 3×10⁻⁶`, taken from the literature; one smooth
ramp on `P_SA`; and the obligation to demonstrate inertness in equilibrium
turbulent layers.

## 9. Suggested verification order

1. **NOW THE DECIDING STEP.** `K` through the log layer, flat plate and NLF
   rooftop → confirm `Q ≡ 1`. First evidence from the Hiemenz rig is favourable
   (`q ≥ 0.991` above `0.2δ` at the outer station) but a Hiemenz layer is not a
   ZPG log layer.
2. ~~Frozen Hiemenz model problem (paper Table `t:stagbistab`) with the gate on~~
   → **DONE 2026-08-06, PASSED** (see the block at the top). Note the criterion
   as originally written — "`max χ → 0` at all `Re_r`" — is the wrong test above
   `Re_r ≈ (577)²`: the correct statement is that `χ` inside the quench zone
   goes to zero, which it does, exactly.
3. Cylinder up-ladder and down-ladder with the gate on → confirm the supercritical
   two-state spread (concession six, `C_d ≈ 0.11–0.19`) collapses, or does not.
4. NLF and Eppler campaigns → confirm nothing moves.
5. Only then: drop the protocol from the paper.
