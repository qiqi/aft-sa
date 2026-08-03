# Proposal: three model changes, for joint evaluation

Status: **proposal, not adopted.** Written for a joint evaluation session.
Each item is stated as equations + evidence + cost + risk, so it can be accepted,
rejected, or deferred independently — subject to the ordering constraint in §4,
which is not optional.

| # | Change | Constants | Touches published results? |
|---|---|---|---|
| **P1** | Compact, 3D-safe curvature indicator | **±0** | expected ≲1 % (must verify) |
| **P2** | Inviscid / viscous two-source amplification | **+2** | ≤0.7 % family-wide (measured) |
| **P3** | Favorable-gradient relaminarization quench | **+1** | expected none (must verify) |

Net: **+3 constants, none fitted to a transition case** — one realization fix at
zero cost, two constants from canonical linear-stability eigenvalues, one from the
relaminarization literature. What is *removed* is the convergence protocol of
§`sec:bistability`.

---

## P1 — Compact, 3D-safe curvature indicator

### Statement

The indicator triple asks for wall-normal Taylor coefficients,
`(u, d ∂u/∂n, ½d² ∂²u/∂n²)`. The solver supplies the third as the velocity
Laplacian projected on the flow direction. Replace it with the **Hessian double
contraction**

```
∂²u/∂n²  =  n̂ᵀ H(u) n̂ ,        n̂ = ∇d          (exact — see below)
```

leaving `X`, `Y`, `Re_Ω` and every constant untouched.

> **Revised 2026-08-02.** An earlier draft proposed
> `∂²u/∂n² = ∇²u·û − (∇²d)(n̂·∇|u|)`. That form is algebraically correct but
> **should not be used**: `∇²d` is genuinely singular on medial axes, not merely
> discontinuous. For a distance field the Hessian eigenvalues are
> `κᵢ/(1 + κᵢ d)`, so `∇²d = Σᵢ κᵢ/(1 + κᵢ d)` **diverges at the focal point
> `d = −1/κ` on any concave side** — which is exactly where a multi-body mesh
> puts its medial axis. The contraction below removes `∇²d` entirely.

### The identity that removes `∇²d`

A true distance field satisfies `|∇d| = 1`. Differentiating,

```
∇(|∇d|²) = 0   ⇒   2 H(d) ∇d = 0   ⇒   (n̂·∇) n̂ = 0
```

i.e. **the `∇d` field lines are straight** (they are geodesics normal to the
wall). Therefore the directional derivative commutes cleanly:

```
n̂·∇( n̂·∇u )  =  n̂ᵢn̂ⱼ ∂ᵢ∂ⱼu  +  n̂ᵢ(∂ᵢn̂ⱼ)∂ⱼu  =  n̂ᵀ H(u) n̂
```

with the second term vanishing identically. No `∇²d` appears anywhere.

**The geometric diagnosis of the original error is now sharp:** `∇²u` is the
**trace** of the Hessian — it sums all three directions and so drags in the
tangential and transverse metric. What the derivation wants is a single
**component**, `n̂n̂`. Taking a trace where a component is required is the actual
mistake, and the pipe simply makes it visible:

```
pipe:   n̂ᵀH n̂ = ∂²u/∂r²                = −2U_c/R²      ← what we want
        ∇²u    = ∂²u/∂r² + (1/r)∂u/∂r  = −4U_c/R²      ← trace, carries the metric
```

### Compact numerics — yes, and it reuses the stored gradient

A directional second derivative does **not** require the full Hessian, and does
not require a nested gradient of a derived field. Taylor-expand over the existing
one-ring, using the gradient the solver already stores (`gradPrimitive`):

```
u_j − u_i − ∇u_i·r_ij  ≈  ½ r_ijᵀ H r_ij
```

Project onto the wall normal and solve a **1×1 weighted least squares** for the
single scalar `n̂ᵀHn̂`, weighting edges by their alignment `a_ij = n̂·r̂_ij`:

```
n̂ᵀHn̂  =  Σⱼ wⱼ bⱼ Rⱼ / Σⱼ wⱼ bⱼ² ,    bⱼ = ½(r_ij·n̂)² ,  wⱼ = a_ij⁴
```

Verified on an irregular, anisotropic 3D point cloud sampling the pipe (4
normal-aligned neighbours + 14 skewed tangential ones, random perturbations):

| probe | LS directional 2nd derivative | exact `∂²u/∂n²` | `∇²u` |
|---|---|---|---|
| `r₀ = 0.50` | **−2.0000** | −2.0000 | −4.0000 |
| `r₀ = 0.80` | **−2.0001** | −2.0000 | −4.0000 |

Properties, against the objections in `SpalartAllmaras.h:311–318`:

| requirement | status |
|---|---|
| compact | ✅ same one-ring as the existing Laplacian |
| no nested gradient of a derived field | ✅ uses the stored `∇u` directly |
| no `\|·\|` sign-kink | ✅ no absolute value anywhere |
| no `∇²d` | ✅ removed by the geodesic identity |
| bounded across medial axes | ✅ `n̂` *jumps* but stays unit; the LS operator is bounded |
| degenerates well | ✅ in a prism/hex layer the normal-aligned edges dominate and it becomes the exact 3-point wall-normal difference |
| robust to bad triangulation | ⚠️ needs a conditioning guard — see risk |

### Alternatives, ranked (for the evaluation)

| | option | verdict |
|---|---|---|
| **A** | Hessian contraction via one-ring LS (above) | **recommended** |
| B | `∇²u·û − (∇²d)(n̂·∇\|u\|)` with `∇²d` clipped to the exact bound `Σκᵢ/(1+κᵢd)` | cheap, keeps current structure, but the clip is arbitrary and the singularity is only masked |
| C | smooth *signed* shear `s = −ω·(û×n̂)`, then `n̂·∇s` | kills the sign-kink but keeps the nested gradient the source comment objected to |
| D | change nothing in the solver; use the corrected form in **diagnostics only** | preserves the campaign exactly at zero cost, but leaves the wake/vortex studies measuring a partly-artifactual quantity |

D is a legitimate position if the evaluation concludes the O(δ/r₀) property should
be documented rather than fixed.

### Why

`∇²u = −∇×ω` is exact for divergence-free flow. What is an approximation is
`∇²u·û ≈ ∂²u/∂n²`, which holds only on a **plane** parallel layer. `∇²u` carries
the transverse/tangential metric of the divergence operator; `∂²u/∂n²` is the
profile-shape coefficient the derivation is built on. They differ at relative
order **δ/r₀**.

Demonstration in Hagen–Poiseuille, where the discrepancy is O(1) and every step
is exact (`blindspots/pipe_kernel_analysis.py`, verified numerically):

```
u = B y + C y²,  B = 2U_c/R,  C = −U_c/R²        exactly parabolic in wall distance
ω = ∂u/∂y                                        X and Y are EXACT; only Z is affected
∂²u/∂n² = 2C = −2U_c/R²
∇²u·û   = ∂²u/∂r² + (1/r)∂u/∂r = 4C = −4U_c/R²
```

The parabola identity is an exact cancellation that the extra term destroys:

```
Y − X − Z = (By+2Cy²) − (By+Cy²) − Cy²  = 0         wall-normal Z   → Î ≡ 0  (correct)
Y − X − Z = (By+2Cy²) − (By+Cy²) − 2Cy² = −Cy² > 0  Laplacian Z     → max Ω̂Î = 0.150
```

The extra `(1/r)∂u/∂r` is the **area metric** of the cylindrical divergence —
momentum diffuses through shells of area `2πrL` that shrink inward. That is a
statement about viscous momentum flux, not about whether the profile is inflected.

### Why this form, and why it is "3D-safe"

`∇²d = ∇·(∇d)` is the **sum of the principal curvatures** of the wall-distance
level sets. It therefore handles, with no special-casing:

- 2D wall curvature (`∇²d ≈ ±1/R_wall`, sign set by convex/concave);
- axisymmetric bodies (both principal curvatures, ≈ `2/r₀` — twice the 2D effect);
- wing tips, junctions, fuselages — anywhere the level sets are doubly curved;
- the far field, where level sets become spherical and `∇²d ~ (n−1)/d` — it does
  **not** vanish away from the wall.

It also preserves every numerical property the current choice was made for
(`SpalartAllmaras.h:311–318`: *"robust to near-degenerate triangulation"*,
*"drops the noisy nested grad|omega| + its |.| sign-kink"*). `∇²d` is one further
compact divergence of a field the solver already stores. **No nested gradient of
`|ω|`, no `|·|` sign-kink.**

### Sign and expected effect on the campaign

| geometry | `∇²d` | bias in `Î` | effect |
|---|---|---|---|
| convex external wall (airfoil, spheroid) | `+1/(R+d)` | current form **under**-reads `Î` | slight *late* transition |
| concave wall / internal | negative | current form **over**-reads `Î` | early |
| far field, wake, vortex core | `~(n−1)/d`, not small | O(1) | **uncharacterized** |

Magnitude in the boundary layer is `O(δ/R_wall)`: ≈1.6 % at the spheroid
mid-body, a few percent near an airfoil leading edge. So the published campaign
should move ≲1 %. The direction is consistent with the spheroid's late
zero-incidence front, but 1.6 % cannot explain a `0.42 L` gap — the seed
explanation in §`sec:spheroid` stands; this is a bounded second-order check, not
a competing cause.

**Where it is not second-order** is exactly where the open questions live: wakes,
free shear layers, vortex cores. The Daedalus tip-vortex audit and the
drag-crisis wake study (file `06-*.md`) must use the corrected form or they will
measure the artifact.

### Notes

- Applies to **both** kernel paths. `__aiRateInvariant` (`ai_invariantKernel = 1`)
  uses `Z₀ = ½d²(lap u)·ŝ` — the same raw Laplacian — so it inherits the same
  issue and the same correction.
- `paper/repro/cfd/add_derived_to_slice.py` currently uses a *third* form,
  `½d²(n̂·∇ω)`, which happens to be correct for a plane parallel layer but is
  not what the solver ran. Every χ-sheet figure and the spheroid flank audit go
  through it. It should be brought onto the corrected form too, and
  curvature-sensitive figures re-checked.
- Appendix E's text is correct about the *current* solver; if P1 is adopted the
  appendix and Eq. `w:gram` need updating.

### Risk — one open, two retired (checked against the compute repo 2026-08-02)

**R1 — conditioning on isotropic tets. OPEN, but it is a solved problem in-house.**
Far from the wall few one-ring edges align with `n̂`, so `Σ wⱼbⱼ²` gets small.
This must **not** be implemented as bespoke code: Flow360 already carries the
whole apparatus in
`MeshProcessor/LeastSquareCoefficients.{h,cpp}` (+ `LsqArrays.h`), and the
directional stencil should be built on it:

- **distance weighting with a tunable exponent** —
  `weights[idx] = pow(max(Sigma[d]/sqrt(edgeCount), …), exponent)`, plus a
  separate centre weight `ctrWeight`; defaults `exponent = 0.50`,
  `crtWeight = 0.1` (`LeastSquareCoefficients.cpp:474`);
- **QR factorisation with an explicit singularity assert** (`:93`, `:107`);
- **condition numbers computed and stored per node** —
  `calculateLsqCoeffCondNumbers` → `grid.nodes.lsqStat[nodeId]`
  (`stats[0] = condNbr`, `stats[1] = condCtr`);
- **condition-number bounding with a stability-over-accuracy fallback** —
  `scaleLsqNodeBLDblAndUpdateCondNum` (`:222`): thresholds
  `condNumThreshold = 1.8`, `condCtrThreshold = 1.0` (`:475`); when exceeded it
  rebuilds the edge vectors and applies `scaleFactor = condNumThreshold/condNum`,
  then re-verifies. Shrinking the coefficients degrades the reconstruction
  **toward zeroth order** rather than emitting a wild value — exactly the
  "prefer stability over accuracy when both are impossible" fallback this stencil
  needs;
- **mirrored / modified edge vectors** for boundary-layer and symmetry nodes
  (`populateModifiedEdgeVectors` with `boundaryUnitNormal`).

Action: reuse this path with the alignment weight `wⱼ = a_ij⁴` folded into the
existing distance weight, and inherit its condition guard verbatim. The
zeroth-order fallback means the degenerate case returns `Z → 0`, i.e. the state
falls back onto the neutral parabola circle — a *safe* direction: no spurious
amplification, no spurious suppression of an already-inflectional profile.

**R2 — approximate wall distance. RETIRED.** The geodesic identity needs
`|∇d| = 1`. Verified: `MeshProcessor/WallDistance.h` builds an octree over the
solid-wall triangulation (`buildOctreeFromTriangles`) and does an exact
closest-point search per node (`minPointToTriDistance`). It is the true distance
function, not an Eikonal or Poisson surrogate — the Eikonal code in the repo
(`CavityBasedMesher`, `MeshOperations`) is mesher-side and does not produce the
solver's `wallDistance`. So `H(d)n̂ = 0` holds to geometric accuracy and no
residual term reappears.

**R3 — medial-axis jump. RETIRED.** `n̂` flips across a medial axis, so `n̂ᵀHn̂`
jumps — but `d` and `∇d` are **frozen geometry, computed once and never updated
during the Newton solve**. The discontinuity is therefore a fixed spatial
coefficient, like any mesh-dependent quantity, not a moving front the Newton
iteration has to track. The Jacobian never sees it. (The same is already true of
`d` and `Re_Ω` in the current kernel.)

---

## P2 — Inviscid / viscous two-source amplification

### Statement

```
P_I    = Ω̂ ⟨Î⟩₊                              inflectional (inviscid) coordinate
P_curv = Ω̂ ⟨−Ẑ⟩₊                             curvature (viscous) coordinate

thr_inv  = k · softmin_n( C , A + B/P_I² )    unchanged from canon
thr_visc =      A + B_c/P_curv²               A_c = A shared

P_AI = ω ν̃ · softmax₂[ a_max  P_I    S(Re_Ω/thr_inv ) ,
                        a_visc P_curv S(Re_Ω/thr_visc) ]
```

Full derivation, tables and repro: file `05-*.md`,
`blindspots/twosource_retune_channel.py`. Whitepaper appendix already drafted
(`paper/whitepaper.tex`, `\label{app:twosource}`).

### Why

`Ω̂Î` conflates the inviscid-inflectional and the viscous Tollmien–Schlichting
mechanisms, which co-vary across the Falkner–Skan family it is calibrated on.
They separate on exactly one profile — a parabola — where `Î ≡ 0`. Plane
Poiseuille *is* exactly parabolic and is linearly unstable at `Re_c = 5772`
(Orszag 1971) by a purely viscous mechanism. The current kernel returns
identically zero there, and no profile in the calibration family exposes it.

**Decoupling the gates is the substantive choice.** Kelvin–Helmholtz is inviscid
(threshold ≈ the `A` floor); TS is viscous (threshold far higher). A shared
`softmin` applies whichever is lower to *both* rates, letting the inviscid
branch's low threshold switch on the viscous rate before the viscous mechanism is
critical — worth +7.4 % on the Blasius anchor. Decoupling removes it.

### Evidence: it changes nothing that works

Peak dimensionless source `a·S`, at `(a_visc, B_c) = (0.0276, 130)`:

| β | Re_θ | canon | two-source | ratio |
|---|---|---|---|---|
| 0.000 | 500 | 0.01485 | 0.01485 | **1.000** |
| 0.000 | 1000 | 0.01485 | 0.01487 | **1.001** |
| −0.100 | 400 | 0.03082 | 0.03083 | **1.000** |
| −0.199 | 300 | 0.09692 | 0.09764 | 1.007 |
| +0.100 | 1000 | 0.00695 | 0.00696 | **1.000** |
| +0.200 | 2000 | 0.00281 | 0.00282 | **1.004** |
| +0.350 | 4000 | 0.00033 | 0.00405 | **12.4** |

`a_max = 0.19` and `k = 0.712` do not move; the 96-solution campaign needs no
re-anchoring. Retaining the ceiling `C` is what buys the β = +0.20 row (0.629 →
1.004) and the clean claim: **changes nothing where the canon works, adds
amplification only where the canon has none.**

### Anchoring

Take both constants from plane Poiseuille — `Re_c = 5772` fixes one combination,
its peak Orr–Sommerfeld envelope rate fixes the other — making the construction
symmetric with what exists:

| branch | anchor |
|---|---|
| inviscid | Michalke tanh KH temporal eigenvalue → `a_max = 0.19` |
| viscous | Orszag plane-Poiseuille critical Reynolds number → `a_visc`, `B_c` |

On a parabola `P_I ≡ 0`, so the channel calibrates the viscous branch with **zero
contamination** from the inflectional one — which the strongly favorable
Falkner–Skan wedges cannot do. And the low-H Falkner–Skan fit disappears, which
removes exactly the objection Drela raised (`expert_feedback.md` §1).

The two independent targets already agree: the low-H family returns `B_c = 130`,
Orszag's `Re_c` returns `91` — ratio 0.70. At `B_c = 130` the channel neutral
point lands at 7409, `1.28×` the true value, against a canon that never
destabilizes it at all.

### Risk

**The stagnation edge.** `P_curv` is *large* on a filled Hiemenz profile
(`u'' < 0` throughout), so the viscous branch could ignite the stagnation region
— the opposite of what P3 is for. Commit `5a0a47e` records "two-eps variant
infeasible at stagnation." **The β = 1 (Hiemenz, H = 2.216) row must be checked
in the decoupled+ceiling form before adoption.** This is the single highest
open risk in the whole proposal.

Secondary: the FS `u''` in the table above comes from numerically differentiated
`du/dy`; redo with the Falkner–Skan ODE's own `f'''` before quoting.

---

## P3 — Favorable-gradient relaminarization quench

### Statement

```
K_local = ν ( û·∇|u| ) / |u|²
P = max[ (1 − σ_P) P_AI ,  σ_P · Q(K) · W(q) · P_SA ]
```

`Q` a smooth ramp, 1 for `K ≪ K_crit`, 0 for `K ≫ K_crit`, with
`K_crit ≈ 3×10⁻⁶` taken wholesale from the relaminarization literature
(Launder 1964; Narasimha & Sreenivasan 1979). `W(q)` confines the quench to
attached wall layers using the existing `f_v1`-bypass sensor `q` (§`sec:fv1bypass`)
— `q ≈ 1` attached, `q ≫ 1` lifted — adding no constant.

Gates **standard-SA production only**; `P_AI` is untouched. Full note: file `07-*.md`.

### Why

Standard SA sustains a spurious turbulent wedge anchored at a leading-edge
attachment point (§`sec:bistability`, critical band `Re_r = 4.66–4.74×10⁵`).
Today the printed transport equation does not determine which steady state an
implementation finds, and a reader porting the model must reproduce a convergence
ritual. This is a *simplification* in the sense that matters.

`K_local` is purely local — no edge velocity, no pressure — and coincides with the
classical `K = (ν/U_e²)dU_e/ds` in a thin layer. Via `ρU dU/ds = −dp/ds` it also
equals `λ_p/Re_d²`, so the legacy `λ_p` machinery in `ModelConstants.h` is an
alternative route.

### Does it reach the branch?

Potential flow over a cylinder gives `K = 1/(Re_D θ²)`, hence
`θ_quench = 1/√(Re_D K_crit)`:

| `Re_D` | `θ_quench` |
|---|---|
| 4.7×10⁵ (the bisected band) | **48°** |
| 10⁶ | 33° |
| 10⁸ | 3.3° |

At the Reynolds number where the paper bisects the branch, the quench covers ~48°
of arc — the whole anchoring region — and it shrinks with `Re`, the correct trend.
Equilibrium turbulent layers sit 1–2 orders below `K_crit` (ZPG: `K = 0`; NLF
rooftop: `K ~ 10⁻⁷–10⁻⁸`).

### A property worth having

`K_local` uses the local `|u|`, which on a **swept** leading edge retains the
spanwise component. `|u|` does not vanish along the attachment line, `K` stays
finite, and the gate does **not** fire — so genuine leading-edge contamination is
preserved while the spurious branch dies. The present protocol cannot make that
distinction (file `02-*.md` §5).

### Risk

`K` is a wall-layer parameter; the `W(q)` confinement is what keeps it from
quenching the outer flow, and that confinement is itself unvalidated for this
use. Needs the same smooth-ramp treatment as the other gates for Newton
convergence.

---

## 4. Interactions and required ordering

**These are not independent. The ordering below is a constraint, not a preference.**

1. **P1 before P2.** The curvature fix changes `Z`, and `Z` enters *both*
   coordinates — `Î` (hence `P_I`) and `−Ẑ` (hence `P_curv`). Any calibration of
   `(a_visc, B_c)` done on the current `Z` is invalidated by P1. The channel
   anchor itself is safe (plane flow, `∇²d = 0`, P1 is the identity there), but
   the Falkner–Skan agreement table must be recomputed after P1.

2. **P2 and P3 collide at the stagnation point.** P3 exists to kill turbulence at
   an attachment point; P2's viscous branch is *strongest* on the filled profile
   found there. They act on different terms (`P_SA` vs `P_AI`), so they are not
   formally in conflict — but they must be evaluated together at β = 1, not
   separately.

3. **P1 changes what the wake/vortex studies measure.** File `06-*.md`'s two
   campaigns should run on the corrected kernel, or run both and report the
   difference as the O(δ/r₀) measurement it is.

Recommended sequence: **P1 → verify campaign unchanged → P2 (re-anchor on the
channel) → β = 1 joint check → P3 → drop the protocol.**

---

## 5. Joint verification plan

| # | Check | Passes if |
|---|---|---|
| V1 | `∇²d` behaviour across a medial axis on a two-body mesh | bounded; limiter adequate |
| V2 | Recompute FS family + marched `N=1`/`N=9` under P1 | ≲1 % shift; `k` unmoved |
| V3 | Rerun NLF + Eppler L2 under P1 | fronts within 0.01 c |
| V4 | Re-anchor `(a_visc, B_c)` on plane Poiseuille after P1 | `B_c ≈ 90`, `a_visc ≈ 0.03` |
| V5 | FS agreement table under P1+P2 | still ≤1 % on the calibrated family |
| V6 | **β = 1 Hiemenz under P1+P2** | viscous branch does not ignite stagnation |
| V7 | `K` through the log layer, flat plate + NLF rooftop | `Q ≡ 1` to round-off |
| V8 | Frozen Hiemenz model problem with P3 | branch gone; no critical band |
| V9 | Cylinder up- and down-ladder with P3 | supercritical two-state spread collapses |
| V10 | Full fleet under P1+P2+P3 | campaign unchanged to plotting accuracy |

V6 is the gate on the whole proposal. If the viscous branch ignites stagnation
and cannot be made not to, P2 does not proceed in this form.

---

## 6. Open questions for the evaluation session

1. Is P1 a *bug fix* or a *model change*? It alters published numbers by ~1 %.
   If bug fix, the paper needs a corrected Appendix E and a re-run; if model
   change, it needs its own justification paragraph. My reading: bug fix — the
   derivation is unambiguous about wanting `∂²u/∂n²`.
2. Is the medial-axis behaviour of `∇²d` acceptable, or does P1 need a smoothed
   wall distance? This decides whether P1 is cheap or expensive.
3. Does P2 earn its two constants given that the model's stated scope is
   external, 2D, low-speed natural transition — where the parabola class is
   narrow? The counter-argument is that it also repairs the strong-FPG corner
   (12.4× at β = 0.35), which *is* in scope and *is* a documented deficiency
   (§`dragcrisis` FPG audit).
4. Does P3 belong in this paper at all, or is "the protocol is stated and
   reproducible" an acceptable position for now? P3 is the only one of the three
   that changes SA itself rather than SA-AI.
