# Spheroid alpha=0 physics deep-dive: what actually starves the amplifying band (re72a0 L2)

*2026-07-28 00:40, implementation-agent thread. Follow-on to the kill-chain
audit (2026-07-27-2142, Part 1.2). Script:
`paper/repro/cfd/spheroid_a0_physics.py` (committed); figures + JSON:
`paper/repro/cfd/figs_explore/spheroid_a0_physics_{blchar,eN,chifield,band,
balance}.png`, `..._captions.md`, `...json` (exploratory, NOT paper figures).
NO tex edits. All numbers below re-extracted from the committed L2/L1/L0/plate
fields by this script; fronts: measured 0.4384 and Stock e^N(8) 0.4248 from
the committed digitization, model chi=1 = 0.9244 re-extracted from this
sweep's own chi field (matches the committed front summary's 0.9245).*

The question posed: the c_nu_ai = 1/6 selection (Sec II.D frozen-profile
eigenvalue) was tuned precisely so laminar diffusion would NOT overwhelm the
amplifying band — so what overwhelms it here?

**Headline: nothing overwhelms the band.** The frozen-profile eigenvalue on
the extracted spheroid profiles stays positive at every station and at every
c_nu_ai down the ladder including c_nu_ai = 1 (minimum 0.48 e-folds/L at the
x/L = 0.70 trough). The front miss is not a diffusion-vs-production defeat;
it is (1) the RATE being roughly halved at its source because the RANS
laminar mean profile is anomalously FULL (H = 2.49 where the physical layer
is Blasius-class 2.59), compounded by (2) a transported-realization shortfall
that grows as the band thins, and (3) the flight-quiet 11.65-e-fold seed
requirement. Given its own mean flow, the model's front is essentially what
the e^N envelope itself would predict.

## 1. BL character (fig blchar)

Re_theta at the audit station x/L = 0.42: **900** (audit said 902 —
verified); at the measured front 0.438: **926**, H = 2.491. The extracted
Re_theta(x) lies on top of a laminar Thwaites–Mangler march driven by the
same field's u_e(x) all the way to ~0.7 (momentum bookkeeping validated),
but the SHAPE does not: field H = 2.44–2.51 across 0.1–0.85 vs the
Thwaites-equilibrium 2.59–2.61 for the same u_e (local lambda ~ +0.007
mid-body: near-Blasius). Operator control: the SAME extraction on the
verified Sec-III laminar flat plate reads H = 2.610/2.605/2.600 at
Re_theta = 641/909/1116 (Blasius 2.59, +0.5%), and H at 0.42 is stable to
the edge-height convention (2.493–2.501 over edge = 1.2–3 delta99). Grid
ladder at 0.42: H = 2.508 (L0) / 2.525 (L1) / 2.493 (L2); at 0.70: 2.542 /
2.511 / 2.44 with the solver-read P falling 0.060 / 0.046 / 0.022. **The
fullness is a stable-to-refining property of the RANS laminar solution (and
the aft stall region deepens with refinement — not grid-converged), not an
extraction artifact and not under-resolution of the integrals.**

## 2. e^N on this field's edge conditions (fig eN)

Axisymmetric (Mangler-weighted) Thwaites march on the FIELD's extracted
u_e(x), Drela–Giles envelope (chain rule dN/ds = dN/dRe_theta *
dRe_theta/ds):

- amplification onset (Re_theta crosses Re_theta0(H)): **x/L = 0.059**
  (the layer is envelope-unstable almost from the nose at these shapes);
- N = 6 at 0.374, **N = 8 at 0.492** (Stock's own N_TS = 8 front: 0.425;
  same class, +0.07 method offset), **N = 11.65 at 0.661**;
- the FS-spatial-conversion variant reads earlier (N8 at 0.297) — that
  identity assumes 2D theta-growth and over-reads on an axisymmetric body;
  the chain rule is the honest bookkeeping. The two bracket Stock.

So: on equilibrium-shape laminar physics, this body's edge conditions demand
transition at 0.43–0.49 for a tunnel-calibrated N = 8, and even the
flight-quiet 11.65 e-folds would commit by ~0.66. The measured 0.438 is
exactly this class. **The e^N target is reproducible from our own field's
u_e — the failure is not in the pressure distribution.**

The same envelope evaluated on the field's OWN extracted shapes
(H = 2.44–2.51, its own Re_theta(x)): onset moves to **0.197**, N = 6 only
at 0.755, **N = 8 at 0.842**, and **N = 11.65 is never reached by x = 0.88**
— i.e. an exact Drela–Giles envelope tracker running on the RANS mean flow
as it actually is would put the front at ~0.84 (tunnel criterion) to >0.9
(flight-quiet criterion). The model's chi = 1 front is 0.924. **Given its
mean flow, the model is e^N-consistent; the mean flow itself carries the
miss.** (The model's realized N even runs AHEAD of the field-H envelope
until ~0.7: 4.5 vs 2.4 at the measured front.)

## 3. Band width, the FS beta=+0.10 class comparison, and the c_nu_ai ladder (fig band)

Planar-convention estimator note (numbers discipline): the raw
double-np.gradient profile curvature used for the audit's planar variants is
noise-dominated on the probed profiles (maxP at 0.42 reads
0.28/0.13/0.056/0.050/0.048 at savgol windows 15/31/61/91/121; the same
estimator recovers exact FS profiles sampled onto the same ray to <1% at
W = 61). Converged planar values below (W = 61); **the flank audit's raw
P_ii ~ 0.075-class planar numbers were noise-inflated ~1.3–1.6x** (its
magnitude-triple P_i chain and verdicts are unaffected).

| x/L | Re_th | H | maxΩ̂Î planar | maxP solver | gate | w/θ | w/δ99 | sup aω/u [/L] | s_eig c=1 | c=1/3 | c=1/6 | c=1/12 | c→0 | realized [/L] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.20 | 584 | 2.51 | 0.063 | 0.058 | 1.00 | 4.42 | 0.59 | 30.1 | 6.0 | 13.2 | **16.9** | 20.0 | 30.1 | 14.4 |
| 0.30 | 735 | 2.51 | 0.061 | 0.054 | 1.00 | 4.33 | 0.58 | 22.4 | 5.6 | 10.8 | **13.5** | 15.7 | 22.4 | 11.2 |
| 0.42 | 900 | 2.49 | 0.056 | 0.046 | 1.00 | 4.20 | 0.56 | 15.2 | 3.9 | 7.4 | **9.2** | 10.7 | 15.2 | 6.5 |
| 0.55 | 1084 | 2.47 | 0.043 | 0.036 | 0.99 | 3.71 | 0.50 | 9.3 | 2.0 | 4.1 | **5.3** | 6.2 | 9.3 | 3.1 |
| 0.70 | 1360 | 2.44 | 0.029 | 0.022 | 0.97 | 2.75 | 0.37 | 4.1 | 0.5 | 1.5 | **2.0** | 2.5 | 4.1 | 0.7 |
| 0.85 | 1900 | 2.47 | 0.038 | 0.030 | 1.00 | 3.17 | 0.43 | 4.3 | 1.6 | 2.6 | **3.0** | 3.4 | 4.3 | 1.6 |

(s_eig = leading eigenvalue of the gated Sec-II.D problem
[a_max clip(P) S_gate ω + (c_nu_ai ν/σ) d²/dy²] v = s u v on the extracted
profiles, e-folds per L; realized = d ln(chi_max)/dx from the field.)

FS calibration geometry at the same machinery (verified against committed
Table 2): beta=+0.10: H = 2.481, maxΩ̂Î = 0.0370, band w/θ = 4.98; Blasius:
2.591 / 0.0782 / 5.62. Eigen retention s(1/6)/sup at Re_theta = 900:
FS-favorable 0.72, Blasius 0.80; spheroid stations 0.50–0.61.

**The sharpened FS beta=+0.10 verdict (user directive):** the spheroid
mid-body profile IS the FS-mild-favorable class — H = 2.49 vs the FS
profile's 2.48 — and its amplifying coordinate (planar 0.056, solver 0.046
at the front station) sits ABOVE the FS beta=+0.10 value 0.037 for that
class (about half-way to Blasius 0.078). The kernel reads its mean profile
fairly, even generously; and on that class the paper's own marcher
verifiably delivers e^N growth. **The pointwise profile physics is fine; the
deficit is (i) the profile itself being the stabler class (where the
physical layer is Blasius-class), and (ii) transport delivery.**

**Confinement-penalty verdict (the c_nu_ai question):** the band is 15–25%
relatively thinner than the calibration geometry (w/θ 2.8–4.4 vs 5.0–5.6)
and P is halved, so the drain costs more than at calibration — retention
0.50–0.61 vs 0.72–0.80 at matched Re_theta — but the eigenvalue never goes
negative at ANY rung of the ladder (heuristic check at the peak, per unit
time: penalty (c_nu ν/σ)(π/w)² = 13% of production a_max P S ω at 0.42, 44%
at 0.70). Removing the drain entirely (c→0) only doubles the trough growth
(2.0 → 4.1 /L at 0.70) while the front deficit is ~e^7. **The stall is NOT
"the band too thin for any c_nu_ai"; no c_nu_ai retune addresses this
case.**

## 4. Transport balance at the stalled station x/L = 0.70 (fig balance)

Frozen-field laminar-branch budget (Appendix E form, sigma_D floor
1 − 0.2489; chi_max = 1.56e-3 at y/δ99 = 0.44, safely laminar), normalized
by peak AI production:

| term | at the chi peak | band-integrated |
|---|---|---|
| AI production | +1.000 | +1.000 |
| wall-normal diffusion (c_nu ν + ν̃) | −0.794 | −0.266 |
| c_b2 gradient term | +0.0001 | +0.001 |
| streamwise diffusion | −7e-7 | −3e-7 |
| destruction (σ_D-tied floor) | −2e-9 | −2e-9 |
| = streamwise convection u_s ∂s ν̃ | +0.357 | +1.176 |
| = wall-normal convection u_n ∂y ν̃ | −0.237 | −0.710 |
| residual (reconstruction error) | +0.086 | +0.270 |

Reading: destruction and c_b2 are inert (as designed — the II.D disclosure
holds pointwise, D/P < 1e-8 here). At the chi peak the transported profile
pays **79% of its production to wall-normal diffusion**, where the confined
leading eigenmode of the same operator would pay only ~50% (s_eig/sup =
2.02/4.06): the actual chi profile is NOT the eigenmode — its shape is
inherited from the upstream, wider band (w/θ 4.4 at 0.2 thinning to 2.8 at
0.70 as P decays 0.058 → 0.022) and it must continuously re-shape as δ99
grows, with the wall-normal convection term redistributing it outward
(−0.24 at the peak, −0.71 band-integrated acting as a feed at fixed y).
Net realized growth at the peak: d ln ν̃/ds = 1.36/L (smoothed field value
0.67/L), vs the pointwise production-only bound ~4.6/L at that height and
the eigen-limit 2.0/L. **The "realization stall" of the audit is therefore
the equation honestly transporting a sub-eigenmode-shaped chi through a
thinning, decaying band — a non-parallel/history effect — not a leak and not
the diffusion coefficient.** (Residual caveat: 8.6% of peak production at
the peak height, 27% band-integrated — probe interpolation plus omitted
axisymmetric metric terms; profile-level statements only.)

## 5. The composed answer

At the measured front (0.438) the model needs 11.65 e-folds (flight-quiet
seed) and has realized 4.5 (kernel sup-bound 9.4 from x = 0.02; the audit's
8.2 from x = 0.10 — consistent). The shortfall decomposes as:

1. **Mean-flow fullness (the dominant, newly isolated link):** the RANS
   laminar profile runs H ≈ 2.49 where Thwaites-on-the-same-u_e (and Stock's
   own BL solutions, evidenced by his 0.425 e^N front) say Blasius-class
   2.59. That single shape shift halves the amplifying coordinate (0.046 vs
   0.078) and, fed to ANY H-based envelope method, moves the N = 8 station
   from 0.49 to 0.84 and pushes N = 11.65 past 0.9. The model's front at
   0.924 is what instability physics says about THIS mean flow.
2. **Transport realization:** on top of the halved rate, the transported
   chi delivers a decreasing fraction of even the frozen-profile eigenvalue
   (0.85 at 0.2 → 0.33 at 0.70) as the band thins — the diffusion
   over-payment quantified in Sec 4.
3. **Seed requirement:** 11.65 e-folds vs Stock's tunnel 8 (the committed
   linear-remap fronts already showed the front barely moves with seed here
   BECAUSE of the stall — consistent).

The c_nu_ai = 1/6 calibration physics itself transfers correctly: at matched
Re_theta and P-class the drain behaves exactly as in Sec II.D, and the band
is never overwhelmed.

## Caveats (recorded)

1. Balance residual 27% band-integrated (8.6% at the peak) — frozen-field
   reconstruction, not solver-residual level.
2. Planar-P estimator: smoothed values quoted (calibration in the script
   docstring); raw-gradient planar values anywhere in this thread's history
   should be treated as upper-noise readings.
3. The e^N Thwaites variant has method spread (chain rule vs FS conversion:
   N8 at 0.49 vs 0.30); both bracket Stock's 0.425 and neither changes any
   verdict.
4. H-based envelope on a non-FS-shaped profile is an approximation; the
   kernel's own P reading independently supports the same class shift.
5. The aft stall region is not grid-converged (P at 0.70: 0.046 L1 → 0.022
   L2, H 2.51 → 2.44): refinement so far deepens the fullness anomaly.

## Follow-on this points at (no action taken)

The pointed question is now the LAMINAR MEAN FLOW: why does the RANS
laminar BL on this body run H ≈ 2.49 / P ≈ 0.05 where laminar-BL theory on
its own u_e says 2.59 / 0.078 — solver numerics (O-grid, low-Mach
dissipation), axisymmetric metric handling, or real non-equilibrium physics
Thwaites misses? A standalone laminar BL march (or an OpenFOAM laminar
cross-solve) on this geometry would discriminate in hours and bears on every
curved-body front the model computes; the airfoil validations (attached,
2D, finer chordwise grids) evidently do not suffer the same profile bias.
