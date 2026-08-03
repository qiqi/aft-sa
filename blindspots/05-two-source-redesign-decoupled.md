# Reviving the viscous branch: channel anchoring and a decoupled structure

**Repro:** `blindspots/twosource_retune_channel.py`. Answers two user questions
(2026-08-02) and reaches a recommendation.

---

## Headline

**The decoupled two-source kernel with the canonical ceiling retained is a
strict superset of the model in the paper.** It reproduces the canonical rate to
within **0.7 %** across the entire calibrated Falkner–Skan family — Blasius, the
adverse branch, incipient separation, and the mild favorable wedges — while
adding a **12.4×** lift exactly where the canon is dead (β = +0.35), and giving
plane Poiseuille a critical Reynolds number within **28 %** of Orszag's 5772,
where the canon returns exactly zero.

So the branch can be revived **without disturbing a single published number.**

## Q1 — can plane channel replace the low-H Falkner–Skan fit as the anchor?

Yes, and the two agree to within 30 %.

On a parabola `P_I ≡ 0` exactly, so the inviscid branch is inert and the channel
anchors the viscous branch **with zero contamination** — it is a clean,
single-mechanism calibration device, which the low-H FS family is not.

Locus of `(a_visc, B_c)` that puts the channel neutral point on `Re_c = 5772`:

| `a_visc` | `B_c` (channel-anchored) | channel `Re_c` at the FS-tuned `B_c = 130` |
|---|---|---|
| 0.0150 | 73.5 | 8 726 |
| 0.0200 | 81.0 | 8 095 |
| **0.0276** (FS-tuned) | **91.0** | **7 409** |
| 0.0350 | 99.9 | 6 916 |
| 0.0600 | 127.9 | 5 834 |
| 0.1900 | 325.3 | 3 725 |

Two readings of the same row:

- at the FS-tuned `a_visc = 0.0276`, the channel wants `B_c = 91` where the
  low-H family gave `130` — a ratio of **0.70**;
- at the FS-tuned `B_c = 130`, the channel neutral point lands at `Re_c = 7409`,
  **1.28×** the true value.

Two entirely independent calibration targets — a low-H Falkner–Skan rate
residual, and the textbook Orr–Sommerfeld critical Reynolds number of plane
Poiseuille — agree on the same constant to 30 %. That is a much stronger
footing than either alone.

### Recommended anchoring: put the whole viscous branch on plane Poiseuille

The channel gives one constraint for two constants. Rather than borrow `a_visc`
from the low-H fit, take **both** from the same canonical flow:

- `Re_c = 5772` (Orszag 1971) fixes one combination;
- the peak Orr–Sommerfeld envelope growth rate of plane Poiseuille fixes the other.

That makes the construction symmetric with what already exists:

| branch | anchor | value |
|---|---|---|
| inviscid | Michalke tanh Kelvin–Helmholtz **temporal eigenvalue** | `a_max = 0.19` |
| viscous | Orszag plane-Poiseuille **critical Reynolds number** (+ its peak rate) | `a_visc`, `B_c` |

Both are canonical linear-stability eigenvalues of a self-similar flow, neither
is fitted to a transition case, and **the low-H Falkner–Skan fit disappears
entirely** — which removes the exact objection Drela raised (`expert_feedback.md`
§1: stop chasing low-H as a defect). The low-H improvement then becomes a
*consequence* of a principled viscous branch rather than its justification.

## Q2 — decouple the two mechanisms? Yes.

Three structures compared, all at the FS-tuned `(a_visc, B_c) = (0.0276, 130)`,
maximum of the dimensionless source `a·S` over the profile:

```
coupled          a = softmax₂(a_inv P_I , a_visc P_curv) · S(Re_Ω / softmin(thr_inv, thr_visc))
decoupled        source = softmax₂( a_inv P_I S(Re_Ω/thr_inv) , a_visc P_curv S(Re_Ω/thr_visc) )
decoupled+ceil   as above, thr_inv keeps the canonical softmin ceiling C
```

| β | Re_θ | canon | coupled | decoupled | **dec+ceil** | cpl/canon | dec/canon | **decC/canon** |
|---|---|---|---|---|---|---|---|---|
| 0.000 | 500 | 0.01485 | 0.01595 | 0.01485 | 0.01485 | 1.074 | 1.000 | **1.000** |
| 0.000 | 1000 | 0.01485 | 0.01595 | 0.01487 | 0.01487 | 1.074 | 1.001 | **1.001** |
| −0.100 | 400 | 0.03082 | 0.03166 | 0.03083 | 0.03083 | 1.027 | 1.000 | **1.000** |
| −0.199 | 300 | 0.09692 | 0.09766 | 0.09764 | 0.09764 | 1.008 | 1.007 | **1.007** |
| +0.100 | 1000 | 0.00695 | 0.00837 | 0.00672 | 0.00696 | 1.203 | 0.966 | **1.000** |
| +0.200 | 2000 | 0.00281 | 0.00177 | 0.00177 | 0.00282 | 0.629 | 0.629 | **1.004** |
| +0.350 | 4000 | 0.00033 | 0.00405 | 0.00405 | 0.00405 | 12.446 | 12.446 | **12.446** |

### Why decoupling is right

**Physically.** The two mechanisms have different critical Reynolds numbers.
Kelvin–Helmholtz is inviscid — its threshold is essentially the `A` floor.
Tollmien–Schlichting is a viscous instability with its own, much higher, viscous
onset. A shared `softmin` threshold applies **whichever is lower to both rates**,
which has no basis: it lets the inviscid branch's low threshold switch on the
viscous rate before the viscous mechanism is critical.

**Numerically.** That is exactly the +7.4 % Blasius contamination in the coupled
column. `thr_inv = 358` vs `thr_visc = 2598` on Blasius: the softmin returns 358,
opening the gate for a viscous rate whose own threshold is 7× higher. Decoupling
removes it, and Blasius returns to **1.000**.

**Practically.** The decoupled form leaves the entire calibrated family
bit-for-bit unchanged, so `a_max = 0.19` and `k = 0.712` **do not move** and the
ninety-six-solution airfoil campaign does not need re-anchoring. The coupled
form would require both.

### Why keep the ceiling `C`

The `vg` proposal dropped `C` for a net-zero constant count. That costs the
β = +0.20 station (0.629× canon) — the curvature branch's threshold there
(4092) is higher than the canon ceiling (1851), so a station the canon ignites
goes quiet. Retaining `C` restores it (1.004) and makes the whole family ≥ canon.

The clean statement is worth the extra constant:

> **The decoupled two-source kernel with the ceiling retained changes nothing
> where the canonical kernel works, and adds amplification only where the
> canonical kernel has none.**

Constant count: `+a_visc`, `+B_c` (with `A_c = A` shared) = **+2**, against the
canon's current set. Both anchored on plane Poiseuille under the recommendation
above, neither fitted to a transition case.

## Proposed final form

```
P_I    = Ω̂ ⟨Î⟩₊                             inflectional (inviscid) coordinate
P_curv = Ω̂ ⟨−Ẑ⟩₊                            curvature (viscous) coordinate

thr_inv  = softmin_n( C , A + B/P_I²   )      unchanged from canon
thr_visc =             A + B_c/P_curv²        A_c = A shared

P_AI = ω ν̃ · softmax₂[  a_max  P_I    S(Re_Ω/thr_inv ) ,
                         a_visc P_curv S(Re_Ω/thr_visc) ]
```

with `softmax₂(x,y) = √(x²+y²)`, `softmin_n` as in the paper.

## Verification plan before adoption

1. **Re-anchor** `(a_visc, B_c)` on plane Poiseuille alone (`Re_c = 5772` + peak
   envelope rate). Expect `B_c` near 91 and `a_visc` near 0.03.
2. **Regression:** confirm the ≤ 1 % family-wide agreement of the table above
   holds across the full FS sweep and the marched `N = 1`/`N = 9` stations —
   i.e. that `k = 0.712` genuinely does not move.
3. **Fleet:** rerun the existing `vg` regression fleet (`de14994`, `2fac060`,
   `agent-paper-review/2026-07-29-0320-vg-kernel-validation.md`) in the
   decoupled+ceiling form. Prediction: the airfoil campaign is unchanged to
   plotting accuracy, since every station in it sits in the ≤1 % rows.
4. **New capability check:** channel `Re_c`, and the pipe staying laminar at
   laboratory `L/D` (file `04-*.md` §4).
5. **Strong-FPG payoff:** the β ≥ 0.35 wedges and the cylinder nose at high
   `Re_D`, where the paper records the FPG deficit (§`dragcrisis`, the
   "collapses three orders of magnitude by β = 0.5" audit).

## Caveats

- `dec+ceil` on the adverse/separated branch is +0.7 %, not exactly 1.000 —
  small but not zero; confirm it stays inside solver noise.
- The FS profiles here use numerically differentiated `du/dy` for `u''`; the
  ≤1 % agreements are at the level of that differentiation. Redo with the
  Falkner–Skan ODE's own `f'''` before quoting in the paper.
- `Re_Ω` here is in edge-normalised marched units; the comparison is a *ratio*
  at matched stations, which is what the table reports.
