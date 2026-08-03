# The retired two-source kernel on channel and pipe

**Question (user, 2026-08-02):** the two-source amplification kernel proposed
just before the Drela meeting — inviscid and viscous rates, soft-maxed — how
would it do on plane Poiseuille and Hagen–Poiseuille? Orders of magnitude wrong
too?

**Answer: no. It lands plane Poiseuille's critical Reynolds number within a
factor 1.75, and leaves a laboratory-length pipe laminar.**

**Repro:** `blindspots/twosource_pipe_channel.py` (numpy/scipy, ~1 min, no GPU).

---

## 1. The kernel, recovered from git

From `2010535` (FPG floor Part III), `4e8961c` (Part V), and the Part VIII `vg`
form, all in `paper/repro/analytic/fpg_recalibration_study.py`:

**Rate** (form `zc`/`zb`) — a soft-max (quadratic mean) of two branches:

```
P_r = Ω̂ · sqrt( ⟨Î⟩₊²  +  ( ε_r ⟨−Ẑ⟩₊ )² ),        Ẑ = Z/R
a   = a_max · clip(P_r, 0, 1)
    = sqrt( (a_inv Ω̂⟨Î⟩₊)² + (a_visc Ω̂⟨−Ẑ⟩₊)² )
a_inv = a_max = 0.19 ,   a_visc = ε_r a_max = 0.0276  (ε_r = 0.1455)
```

**Gate** (form `vg`, two-branch, *no* constant ceiling `C`):

```
Re_Ω^c = softmin_n( A + B/P_I² ,  A_c + B_c/P_curv² )
P_I = Ω̂⟨Î⟩₊ ,  P_curv = Ω̂⟨−Ẑ⟩₊ ,  A_c = A = 124.6 , B_c = 130   (k-carrying)
```

The viscous coordinate is `⟨−Z⟩₊/R`: positive wherever `u'' < 0` (filled,
non-inflected profiles), clipped to zero near an adverse wall, vanishing like
`y` toward the wall and zero in the free stream.

## 2. Why it has anything to say here

On an exactly parabolic profile `Î ≡ 0`, so the **canonical** single-coordinate
kernel returns exactly zero (file `01-*.md`). But a parabola has `u'' < 0`, hence
`−Z > 0`, so the **viscous branch is alive**. The two-source form is the only
variant in the project's history that gives channel or pipe flow a rate at all.

Both flows reduce to the *same* profile in wall units, `s = d/h` or `d/R`:

```
u = 2s − s² ,   ω = |u'| = 2(1−s) ,   u''(wall-normal) = −2
```

(the pipe using the repro realization `Z = ½d² n̂·∇ω`, which equals `½d² u''`).
So the indicator profiles are **identical**; only the diffusion operator
(axisymmetric vs planar) and the Reynolds bookkeeping differ.

## 3. Results

### Coordinates (Reynolds-independent)

```
max P_I    (inflectional) = 1.2e-16          <- exactly zero, as it must be
max P_curv (viscous)      = 0.18406  at d/h = 0.7017
max rate a                = 0.005088 at d/h = 0.7017
Re_Ω^c at that point      = 3961.8
```

The rate is **0.34× the canonical Blasius rate** (`0.19 × 0.078 = 0.0148`).
Caveat: that comparison uses the *single-source* Blasius value; under the
two-source form Blasius's own rate also rises slightly (its `u'' < 0` too), so
the true ratio is somewhat below 0.34. Directionally: **the channel gets roughly
a third of a Blasius layer's amplification rate.**

### Onset and net growth (frozen eigenvalue, diffusion retained, grid-converged)

| | gate 0.5 | net growth `s > 0` | truth |
|---|---|---|---|
| plane Poiseuille | Re = 1.35×10⁴ | **Re_c = 7 409** | **5772** (Orszag 1971) |
| Hagen–Poiseuille | Re = 1.35×10⁴ | **Re_D = 8 821** | linearly **stable at all Re** |

> **Correction (2026-08-02):** the channel entry originally read `1.011×10⁴`.
> `twosource_pipe_channel.py` imposed a Dirichlet condition at the channel
> centreline instead of symmetry, over-damping the eigenfunction. The pipe
> operator was correct (the `r`-weight kills the axis flux automatically), so
> `Re_D = 8 821` stands. The corrected channel value is **7 409**, i.e. the
> two-source kernel is **1.28×** the true critical Reynolds number, not 1.75×.
> Corrected value from `twosource_retune_channel.py`, which is the authority.

Growth rates, and the development length they imply:

| Re | channel `s·h` | pipe `s·D` | pipe L/D for N=9 |
|---|---|---|---|
| 1×10⁴ | −1.6e−5 | 0.00026 | 35 000 |
| 2×10⁴ | 0.00257 | 0.00520 | 1 730 |
| 5×10⁴ | 0.00460 | 0.00921 | 977 |
| 1×10⁵ | 0.00536 | 0.01073 | 839 |

## 4. Reading

**Channel — good.** Onset at `Re_c ≈ 1.0×10⁴` against the true `5772`: a factor
**1.75** late. For a kernel with no knowledge of the Orr–Sommerfeld problem,
built from three local profile moments and calibrated on Falkner–Skan, landing
the canonical viscous-TS critical Reynolds number within a factor of two is a
real result. The canonical kernel misses it by ∞ (returns exactly zero).

**Pipe — soft, harmless error.** The kernel predicts weak growth from
`Re_D ≈ 8 800`, from a linear mechanism that does not exist. But the rate is so
small that a real pipe never accumulates: Reynolds' glass tubes were `L/D` of
order 50–250, which at `Re_D = 2×10⁴` banks only

```
N ≈ 0.0052 × 250 ≈ 1.3 e-folds
```

— nowhere near any handover. Even at `Re_D = 10⁵` a 250-diameter pipe banks
`N ≈ 2.7`. So **the two-source kernel leaves a laboratory-length pipe laminar at
every Reynolds number reached in the classical experiments**, which is the
observationally right answer (Ekman `4.4×10⁴`, Pfenniger `10⁵`), and it still
cannot produce Reynolds' subcritical transition — correctly, since that is
finite-amplitude physics outside the envelope class.

Contrast with the earlier estimate for the canonical kernel *under the
Appendix-E `∇²u` realization* (file `01-*.md` §2), which gave `s·D ≈ 0.09` —
**17× larger** — and would have turned every long pipe turbulent. That variant
is the one that would have been badly wrong; neither the canonical kernel as
actually implemented nor the two-source kernel is.

## 5. Why this matters beyond the low-H question

Drela advised de-prioritizing the low-H (strong-FPG) mismatch, and the two-branch
form was parked on that basis (`expert_feedback.md` §1). This analysis is an
**independent argument for the same construction**, from a different direction:

> The single coordinate `Ω̂Î` conflates the inviscid-inflectional and viscous-TS
> mechanisms, which co-vary across the Falkner–Skan family it was calibrated on.
> Exactly parabolic profiles are where they separate — and the two-source kernel
> is exactly the form that separates them. Its viscous branch `⟨−Z⟩₊/R` is not a
> low-H patch; it is the missing *mechanism*, and the channel's `Re_c = 5772` is
> a canonical, Falkner–Skan-independent target it hits within 1.75×.

Whether that is worth reopening is a judgement call — the model's stated scope is
external, two-dimensional, low-speed natural transition, and the parabola class
is narrow in practice (file `01-*.md` §7). But the finding changes the *character*
of the low-H item from "a residual we chose not to chase" to "the visible corner
of a mechanism the kernel does not carry," which is a more honest framing for
the discussion section either way.

## 6. Caveats

- `ε_r = 0.1455` was tuned jointly with `C` on the Falkner–Skan family
  (commit `2010535`), not on internal flow. The channel/pipe numbers here are a
  *prediction* of that tuning, not a fit — which is what makes them meaningful.
- Adopting the two-source rate would require re-anchoring `k` (the Blasius
  `N = 1` anchor moves, since Blasius's own rate rises) and re-running the
  regression fleet. The `vg` fleet results exist (`de14994`, `2fac060`,
  `agent-paper-review/2026-07-29-0320-vg-kernel-validation.md`).
- The `vg` gate replaces the constant ceiling `C` with the curvature branch, so
  the constant count is unchanged — worth restating if this is revived.
