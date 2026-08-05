# Expert feedback log — SA-AI transition model

Chronological record of feedback from outside experts on the SA-AI
(Spalart–Allmaras with Autogenous Inception) model and paper, with the
resulting analysis tasks and design directions. Newest entries go at the
**bottom**. Each entry captures, as faithfully as possible: who, when, the
raw feedback (paraphrased from the meeting/email, not verbatim), how it
squares with what we already know, and the concrete action items it spawns.

This is a running memory file — it is committed so any agent or collaborator
picking up the work can see the standing expert guidance and its status.

---

## 2026-07-29 — Mark Drela (MIT)

Context: meeting to walk Drela through the SA-AI whitepaper. Overall verdict:
**he considers the approach highly promising.** The substance below is his
feedback plus the directions we agreed on. Two logistics items:

- Drela is sending his **database of separated boundary-layer profiles**
  (see the high-H item). He promised it at the meeting; we followed up by
  email.
- His thesis is saved at `paper/drela.pdf` (159 pp). Note: it is a
  **scanned/image PDF with no text layer** — it needs OCR before we can mine
  it programmatically.

### 1. Low-H (strong-FPG) mismatch — de-prioritized

Drela: **the low-H mismatch is not very important**, based on his experience.
This is consistent with what our own experimentation already showed:

- The low-H residual is **structural**, not an untuned knob — neither
  `a_visc` (largest safe value 0.0276 under a 5% Blasius drift) nor `c_nu,ai`
  removes it.
- Mechanism: in the inflectional coordinate `P_I = Ω̂ Î` the low-H (strong
  favorable-gradient) profiles collapse toward `Î ≈ 0`, so a single-branch
  onset gate has nothing to graze there. The `vg` two-branch curvature-gate
  proposal (`P_curv = ⟨Ω̂ Ẑ⟩₊`, `Ẑ = −Z/R`; second branch `A + B_c/P_curv²`,
  `B_c = 130`) is what lifts the low-H rate toward Drela, and it is
  env-gated / default-off.

Action: **stop chasing the low-H rate as a defect.** Keep the `vg` two-branch
form available but do not treat the remaining low-H miss as a blocker. Soften
any paper language that frames low-H as an open problem.

### 2. High-H limit — important, and it HAS a literature/physics basis

This is the correction that matters most. We had postulated (in the paper)
that the high-H behavior lacks direct literature evidence. **That was wrong.**

- Drela's **high-H limit comes from Orr–Sommerfeld solutions computed on a
  database of separated profiles**, all taken from laminar solutions of
  representative airfoil profiles. It is a real, grounded limit, not a
  correlation guess.
- His high-H limit is **also consistent with the Kelvin–Helmholtz
  (inviscid, inflectional) rate**, just as ours is: our free-shear
  amplification ceiling `a_max = 0.19` is exactly the Michalke KH eigenvalue
  of the tanh mixing layer (`amax_rayleigh.py`). So the high-H limit is
  anchored to the same inviscid inflectional instability from two independent
  directions.

Actions:
- [ ] Obtain Drela's separated-profile database (pending email).
- [ ] Re-derive / validate **our** high-H limit against Orr–Sommerfeld on
  those separated profiles, using the in-house eigenvalue machinery:
  `repro/analytic/explore_pinchpoint_shooting.py` (Briggs–Bers / OS shooting,
  already validated to HM85 `R_crit = 1.3157`), `explore_lsb_pinchpoint.py`,
  and `amax_rayleigh.py` (Rayleigh eigenvalue for `a_max`). See
  `repro/analytic/notes_absolute_instability_lsb.md`.
- [ ] Correct the paper: remove/qualify the "lack of literature evidence"
  framing for the high-H limit; cite the Orr–Sommerfeld-on-separated-profiles
  basis (Drela's database + the corroborating source above).

### 3. Handover — do NOT let SA "do whatever it does" once χ just exceeds 1

Drela's key modeling insight on the handover:

- It is **wrong** to hand the state over to unmodified SA the moment `χ`
  climbs just above 1.
- **TS waves at high amplitude already carry significant Reynolds stress** —
  so the transition physics is not "done" at `χ ≈ 1`.
- The ideal handover should be at **`χ = 30` or above**, and it should scale
  with **bubble size**: the bigger the bubble, the larger the handover `χ`
  should be.
- Note this dovetails with the current model: the paper already states that
  for `χ ≳ 30`, `f_v1 → 1` and the eddy-viscosity blend becomes the identity
  (`sec:fv1bypass`). Drela's number lands where our lifted-layer assembly
  already stops intervening — worth making the handover target explicit and
  bubble-size-aware.
- **Quantified** (see `expert_feedback/handover_viscosity_ratio.tex`, repro
  `repro/analytic/handover_viscosity_ratio.py`, cross-checked to 1e-9 against
  the XFOIL closure in `src/validation/mfoil.py`): XFOIL initializes the
  turbulent station at `Cτ(s_t) = c²·Cτ,eq` with `c = 1.8 e^{-3.3/(Hk-1)}` —
  a function of `Hk` **alone** (no `N` dependence; applied once at the
  `N = N_crit` station), and `c = 1` only at `Hk = 1 + 3.3/ln1.8 = 6.6`. So the
  layer is handed over holding just `c²` of its equilibrium stress: 4% at
  Hk=2.5, 23% at Hk=3.5.
- Mapping `Cτ → χ` via `χ = Re_θ·Cτ/G`, where the peak shear
  `G = (θ/u_e)(du/dy)_max` is **calibrated on Falkner–Skan** (the right family,
  since at a bubble's transition station the profile is still the laminar
  near-/post-separation shear layer): `G = 0.207 ± 2%` and is essentially
  **independent of Hk** over Hk = 2.7–4.9 (mild APG → separation → reversed
  flow). It does *not* scale as `(Hk−1)/Hk`. Hence `χ ≈ 4.8 Re_θ Cτ`.
- Result: the handover level is **`χ* = χ_init = O(0.2–7)`**, rising ~8× from
  Hk=2.5 to Hk=3.5 at fixed Re_θ (mostly via `c²`). So "bigger bubble → later
  handover" is confirmed **quantitatively**.
- ⚠ **Tension to raise with Drela:** this is an order of magnitude *below* his
  stated `χ≈30`. XFOIL's own initialization hands over at χ of order a few and
  expects the layer to develop the rest — which is what SA is supposed to do.
  His 30 may refer to the developed layer (Clauser level `≈0.017 Hk Re_θ`,
  which is O(10–50)), or to a target for a model without a lag equation.

### 4. χ recirculation inside the bubble → growth faster than e^N

Known issue (corrected 2026-07-30 from the Eppler Re=1e5 χ-sheet): the model
does **not** over-amplify along the unstable boundary-layer streamline.
Tracing χ only through the most-unstable BL region matches XFOIL's \(e^N\)
closely, reaching \(c_{v1}\) near \(x/c\approx0.6\) where XFOIL declares
transition. The open bubble is a different failure: at the oil-flow closure
station (\(x/c\approx0.67\)) we already have \(\chi\sim50\) lofted in the
shear layer, but that large χ **does not propagate toward the wall** — it
sits above the peak reverse velocity and never closes the bubble. A prior
reseeding / reverse-layer sink study (negative amplification, RP² upper-branch
barrier) is parked on `deadend/rp2-reseeding-barrier` and should not return to
canon.

- **Real problem: lofted χ fails to diffuse/mix to the wall** inside the
  reverse-flow region, so turbulent viscosity never acts where it would
  reattach the bubble.
- [ ] **Analysis to do:** characterize wall-normal transport of χ across the
  reverse layer (why \(\chi\sim50\) stays aloft); what `Re_θ` looks like when
  computed **from the reversed-flow portion of the profile only** (a
  reverse-flow `Re_θ`). This sets the argument for the stable/unstable switch.

**Concern to quantify before committing to a decay term:** if we add an
exponential decay of `χ` in regions where `Re_Ω` is well below `Re_Ω^c`, it
will likely **erode the growth slope near onset** — only a small band of the
profile has local `Re_Ω > Re_Ω^c`, and the rest of the profile would be
decaying, eating away the net growth.

- [ ] Quantify this near-onset slope penalty with the eigenvalue solvers
  (above), and **verify with the parabolized marching tool**
  `repro/analytic/march_sa_handover.py` (frozen-profile parabolized march that
  already carries the full blended SA handover RHS; can watch `d(ln χ)/dx`
  slope continuity across the `χ = 1` crossing).

### 5. Boosted handover via the q-based lifted-layer assembly

To realize the `χ = 30` (bubble-size-dependent) handover, go **aggressive on
the eddy-viscosity assembly in lifted transitional layers** (`sec:fv1bypass`,
the q-based gate: `μ_t = [(1−b) f_v1(χ) + b] ρ ν̃`, `b = s(χ) G(q)`,
`q = |u| d/ν / (y⁺ U⁺(y⁺))`, `G` ramping over `q ∈ (2,4)`,
`s = clip(χ−1, 0, 1)`).

- Idea: **generalize the q-based formulation** so it does not merely *prevent*
  `f_v1` from suppressing the turbulent viscosity in a lifted layer, but also
  **boosts the growth of SA itself** — enough to overwhelm the destruction and
  self-diffusion terms — until `χ` climbs to the Drela-intuited `≈ 30` (or the
  local bubble-size-set value).
- [ ] **Test tool:** a parabolized solver **with reverse advection dropped**
  is a natural instrument for this "boosted handover" construction. Extend /
  reuse `march_sa_handover.py` (its `x_freeze` "bubble caricature" mode pins
  `Re_Ω`; add a reverse-advection-dropped mode).

### Consolidated action items

- [ ] Get Drela's separated-profile database (email follow-up sent).
- [ ] Validate our high-H limit vs Orr–Sommerfeld on those profiles
  (`explore_pinchpoint_shooting.py`, `amax_rayleigh.py`).
- [ ] Fix the paper's high-H "no literature" framing; cite the OS-on-separated
  -profiles basis and the Kelvin–Helmholtz (inviscid inflectional) corroboration.
- [ ] Analyze reverse-flow-only `Re_θ` across bubble profiles.
- [ ] Quantify the near-onset slope penalty of a `Re_Ω ≪ Re_Ω^c` decay term
  (eigenvalue solvers + `march_sa_handover.py`).
- [ ] Prototype the negative-growth-where-`Re_θ`-stable rate law.
- [ ] Prototype the boosted q-based handover to `χ ≈ 30` (bubble-scaled),
  tested on a reverse-advection-dropped parabolized march.
- [ ] Make the handover target explicit and bubble-size-aware in the model:
  XFOIL-implied `χ* = O(0.2–7)`, rising steeply with Hk, per
  `expert_feedback/handover_viscosity_ratio.{tex,pdf}`.
- [ ] Reconcile that `χ* = O(0.2–7)` with Drela's stated `χ≈30` (ask him which
  level he meant).

### Tools referenced

- Eigenvalue / stability: `repro/analytic/explore_pinchpoint_shooting.py`
  (Briggs–Bers OS shooting), `explore_lsb_pinchpoint.py`, `amax_rayleigh.py`;
  criteria review in `repro/analytic/notes_absolute_instability_lsb.md`.
- Parabolized marching (handover): `repro/analytic/march_sa_handover.py`.
- q-based lifted-layer assembly: paper `sec:fv1bypass`;
  `repro/analytic/fv1_qmargin_composite.py`, `explore_q2_*.py`.
- Handover viscosity ratio (χ at handover: XFOIL `Cτ` initialization + the
  Falkner–Skan peak-shear `G` calibration): note
  `expert_feedback/handover_viscosity_ratio.tex` (+ `.pdf`), repro
  `repro/analytic/handover_viscosity_ratio.py` (cross-checked against
  `src/validation/mfoil.py` `get_cteq`/`get_cttr`).
