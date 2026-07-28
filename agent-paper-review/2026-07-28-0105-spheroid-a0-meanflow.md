# Spheroid alpha=0 mean-flow anomaly: WHY the RANS laminar BL runs full (re72a0)

*2026-07-28 01:05, implementation-agent thread. Follow-on to the physics
deep-dive (2026-07-28-0040, Sec 1 + closing paragraph), which isolated the
mean-flow fullness (field H = 2.44-2.51 where Thwaites-on-the-same-u_e says
Blasius-class) as the dominant carrier of the alpha=0 front miss and posed
the cause question.  Script: `paper/repro/cfd/spheroid_a0_meanflow.py`
(committed); figures + JSON:
`paper/repro/cfd/figs_explore/spheroid_a0_meanflow_{march,profiles,balance,
resolution,wiggle,cp_pg}.png`, `...json`.  NO tex edits, NO GPU launches.
Candidates from the task: (a) solver numerics, (b) axisymmetric/transverse-
curvature BL physics, (c) non-equilibrium/history, (d) extraction geometry.*

**Headline: (a) is convicted.** A validated axisymmetric laminar BL PROFILE
march (full history, Mangler, first-order transverse curvature) on the
field's own u_e says H = 2.56-2.63 across 0.2-0.7 where the RANS field reads
2.49-2.44; a frozen-field momentum-balance probe (which closes to 0.5% on
the verified flat plate) finds a structured NEAR-WALL STREAMWISE MOMENTUM
SOURCE of numerical origin in the spheroid field — the force that holds the
profile full — and the discrete solution carries a cell-locked tangential
staggering, dominated by a ONE-AZIMUTHAL-CELL wiggle (du/u peak-to-peak 2.6%
at y = 0.15 delta99) that should be exactly zero on this axisymmetric flow.
The airfoil control exonerates the airfoils for a sharper reason than the
0040 note assumed: their favorable-region H is equally below equilibrium but
matches the BL march on their own u_e to 0.02 — genuine LE-history physics,
not the anomaly.

## 1. The laminar BL march (fig march, profiles)

Implicit space-march of the axisymmetric laminar BL equations in physical
(s, y), driven by the field's extracted u_e(x) and the analytic r0(x);
r-ladder: planar (r=1) / axisymmetric (r=r0, exact Mangler content) /
+ transverse curvature (r = r0 + y n_r).  Validation: constant-u_e run gives
H = 2.5906 (Blasius 2.5905), cf*theta*u_e/nu = 0.2206 (0.2205), theta within
0.7% (IC virtual origin); insensitive to IC shape (tanh vs quartic), IC
theta0 (+30%), and y/s grid halving.  Discretization gotcha recorded in the
script: the pressure forcing must be the BACKWARD difference of u_e (the
marcher's own stencil) — a central-difference forcing at the nose-stub kink
plants a permanent inviscid outer deficit that silently doubles theta.

**H table** (march numbers "operator-matched" = the field's own
edge/integral operator on the RAY grid applied to the marched profiles;
raw march integrals in parentheses):

| x/L | RANS L0 | RANS L1 | RANS L2 | march planar | march axi | march axi+TC | H_eq(lambda) | L2 gap |
|---|---|---|---|---|---|---|---|---|
| 0.20 | 2.446 | 2.507 | 2.512 | 2.545 | 2.533 (2.564) | (2.563) | 2.596 | −0.02 |
| 0.42 | 2.508 | 2.525 | 2.493 | 2.573 | 2.561 (2.581) | (2.579) | 2.606 | −0.07 |
| 0.70 | 2.542 | 2.511 | 2.438 | 2.626 | 2.613 (2.626) | (2.623) | 2.648 | −0.18 |

- Transverse curvature is worth −0.002 in H; Mangler vs planar +0.01-0.02;
  full non-equilibrium history −0.02 to −0.04 vs the equilibrium H_eq at the
  fore stations.  **Candidates (b, first order) and (c) are real, captured,
  and an order too small.**  The march separates at x ~ 0.87 (strong aft
  adverse gradient); claims stop at 0.85.
- The march also corrects the 0040 note's "Thwaites-equilibrium 2.59-2.61,
  flat across 0.1-0.85": the proper laminar reference DIPS below Blasius on
  the front half (2.50-2.58, favorable), crosses 2.59 just past mid-body and
  RISES aft (2.63 at 0.7) — see Sec 4.  The fore-body gap is therefore
  smaller than the note implied (−0.02 at 0.2), and the aft gap larger
  (−0.18 at 0.7); at the measured-front station 0.42 it is −0.07 to −0.09.
- Momentum bookkeeping: field Re_theta(x) tracks the march to ~0.55, then
  falls below it — the field aft layer carries LESS momentum deficit while
  being FULLER: both symptoms of a near-wall momentum source (Sec 2).

## 2. Momentum-balance probe (fig balance)

Frozen-field steady laminar streamwise momentum, term-by-term through the
layer from 5-ray FD stencils (rho u.grad(u).t_s + dp/ds − mu[d2u_s/dy2 +
(1/r)dr/dy du_s/dy + d2u_s/ds2]), smoothed with the physics-pass-calibrated
savgol (W=61 on 600-pt uniform resample).  All cases share IDENTICAL
numerics (Roe, lowMachPreconditioner=false, M=0.1, 2nd order MUSCL
kappa=−1), so differences indict the grid/geometry interaction.

| case | nu_eff/nu median (0.1-0.75 d99) | IQR | imbalance/max-viscous | near-wall imbalance at 0.15 d99 |
|---|---|---|---|---|
| plate x=2 (Re_th 909, matched) | 1.005 | 0.994-1.013 | 0.005 | +0.006 |
| NLF0416 a0 x/c=0.25 | 1.098 | 0.896-1.292 | 0.114 | −0.05 (sign-mixed noise) |
| spheroid L1 x/L=0.42 | 1.042 | 0.973-1.101 | 0.038 | +0.08 |
| spheroid L2 x/L=0.42 | 0.813 | 0.589-0.883 | 0.160 | **+0.33** |
| spheroid L2 x/L=0.70 | 0.670 | 0.257-0.844 | 0.309 | **+0.83** |

- The plate row validates the instrument to half a percent.
- The spheroid L2 imbalance is a systematic POSITIVE near-wall lobe
  (+0.33/+0.83 of max|viscous| at 0.15 delta99, decaying to zero by ~0.75
  delta99), robust to the stencil (hx = 0.003/0.006/0.012 gives median
  nu_eff/nu 0.79/0.81/0.90): **a numerical streamwise momentum source in
  the lower half of the layer — precisely the force needed to hold the
  profile fuller than laminar physics allows, largest where the anomaly is
  deepest (aft).**  The answer to the task's "is the solver adding O(10-20%)
  effective viscosity" is: not a uniform viscosity excess — a structured
  near-wall source (the mid-layer nu_eff/nu < 1 reading is that source's
  signature in the quotient, not genuine anti-diffusion).
- The measured dp/ds is CONSTANT across the layer at both stations
  (−0.019 and +0.20 of max|viscous| at 0.42/0.70): no second-order
  normal-pressure-gradient physics is present — kills the remaining part of
  candidate (b).

## 3. The staggering (fig wiggle) — the discrete solution's own signature

Fine-dx probe of u_s at fixed wall heights (12 samples per cell):

- MERIDIONAL: one-cell sawtooth (dominant wavelength 2.6e-3 = exactly the
  L2 meridional cell), rms du/u = 6.3e-4 at 0.15 delta99, decaying to 2e-6
  above the layer.
- AZIMUTHAL (should be exactly zero for this axisymmetric flow): locked to
  the azimuthal cell (1.128 vs cell 1.125 deg), rms du/u = 7.7e-3,
  peak-to-peak 2.6% at 0.15 delta99 — **12x the meridional mode**.  At L1:
  rms 2.8e-2 (3.7x the L2 amplitude, locked to the L1 cell).
- Geometric driver scale: the discrete surface is azimuthally faceted; the
  facet sagitta is ~10 first-cell heights at L2 (21 at L1, 43 at L0) — the
  near-wall cells always see the polygonized surface at o(10) wall units,
  which is why refinement reduces the wiggle amplitude but does not converge
  the H deficit (L0/L1/L2 at 0.42: 2.508/2.525/2.493; at 0.70 it DEEPENS:
  2.542/2.511/2.438).
- H(x) itself is smooth (station-to-station scatter 0.001): the staggering
  does not contaminate the extraction; its rectified (mean) footprint is
  what the Sec-2 balance measures.

## 4. The user challenge (2026-07-28), point by point (fig cp_pg)

1. **Cp(x)/u_e(x) extraction sanity**: extracted Cp (probed p, far-field
   referenced) overlays 1−(u_e/U_inf)^2 to plotting accuracy; the u_e peak
   sits at **x/L = 0.489** — at/slightly upstream of 0.5 exactly as
   expected for the displacement-shifted fore-aft-symmetric body.  The
   extraction geometry is right; u_e does NOT keep accelerating past 0.5.
2. **Equilibrium/march H shape**: confirmed as the user reasoned — the
   proper laminar H dips below Blasius over the favorable front half
   (march: 2.50-2.58; equilibrium H_eq(lambda): 2.55-2.61), crosses near
   the peak, and rises past it (march 2.63, H_eq 2.65 at 0.7).  The
   slender-body PG is weak: lambda in [+0.010, −0.010] over 0.06-0.75
   (−0.06 by 0.83, marcher separation ~0.87).  The 0040 note's "2.59-2.61
   flat" overstated flatness; its verdict survives but the honest gap is
   smaller fore (−0.02) and larger aft (−0.18).
3. **Aft re-verification (0.55-0.85)**: dense rays (400 pts to 0.12L) +
   edge-convention ladder (first-local-max; fixed 1.5/2/3 delta99 edges):
   H(0.70) = 2.438/2.442/2.453, H(0.80) = 2.433/2.441/2.458 — the aft fall
   is extraction-real to ±0.02.  maxP (planar kernel, same estimator class
   as the physics pass) re-extracted aft: 0.044/0.039/0.030/0.026/0.042 at
   0.55/0.60/0.70/0.80/0.85.  On the MARCHED laminar profiles the same
   kernel reads 0.078/0.081/0.091/0.127/0.246 — **rising monotonically into
   the adverse region, exactly the user's expectation.  Stated plainly: the
   RANS layer gets FULLER (H 2.49 -> 2.43, maxP 0.044 -> 0.026) while
   marching INTO an adverse pressure gradient — the wrong-signed response,
   a strong and strange signature that is extraction-robust and coincides
   with the largest measured numerical momentum source (+0.83 at 0.70).**
   The downstream FALL of maxP in the band table of the 0040 note is
   therefore real (its Sec-3 numbers stand) but is itself part of the
   anomaly, not physics.
4. The band-table trend vs physics: resolved as above — the physical
   expectation does rise aft; the field's opposite trend is the numerics
   artifact, now directly instrumented.

## 5. Airfoil control at matched Re_theta

| station | Re_theta | field H | march-on-own-u_e H | verdict |
|---|---|---|---|---|
| NLF0416 a0 x/c=0.12 | 459 | 2.437 | 2.458 | history-consistent (−0.02) |
| NLF0416 a0 x/c=0.20 | 625 | 2.510 | 2.538 | history-consistent (−0.03) |
| NLF0416 a0 x/c=0.30 | 807 | 2.640 | 2.650 | history-consistent (−0.01) |
| spheroid x/L=0.42 | 900 | 2.493 | 2.561-2.581 | **anomalous (−0.07..−0.09)** |
| spheroid x/L=0.70 | 1360 | 2.438 | 2.613-2.626 | **anomalous (−0.18)** |

Eppler387 a2 (Re 2e5): H = 2.623 at x/c=0.25 (lambda −0.021) and 3.135 at
0.40 (approaching the LSB) — adverse/bubble physics, no fullness anomaly.
So "the airfoils don't suffer the profile bias" (0040 closing paragraph) is
CONFIRMED, but the reason is subtler than assumed: their below-equilibrium
favorable-region H is genuine LE-acceleration history, quantitatively
reproduced by the same marcher that the spheroid field disagrees with.

Streamwise resolution does NOT discriminate: cells per delta99 are the same
class everywhere (spheroid L2 0.27-0.56, NLF L2 0.31-0.37, plate 0.22 — and
the plate is the COARSEST yet exact).  What is spheroid-specific is the 3D
azimuthal direction: curved, faceted (sagitta ~10 h0), high-aspect near-wall
cells — the direction whose staggering mode dominates Sec 3.

## 6. Ranked verdict

1. **(a) solver numerics on the axisymmetric O-grid — the carrier.**
   Direct footprint (Sec 2), direct mode signature (Sec 3), wrong-signed
   aft response (Sec 4.3), all lower-order alternatives eliminated by a
   validated instrument (Sec 1).  Sub-attribution left open between (i) the
   azimuthal faceting/staggering truncation stress and (ii) unpreconditioned
   low-Mach Roe dissipation (config shared with the clean airfoil/plate
   cases, but only the spheroid has 3D curved anisotropic near-wall cells
   for it to act on).  Note the wiggle amplitude drops 3.7x from L1 to L2
   while the aft H deficit deepens — amplitude alone is not the dose metric
   (the rectified stress rides on gradients that steepen with refinement),
   which is why the discriminating run below is worth one GPU slot.
2. **(c) non-equilibrium/history** — real, correctly captured by the march,
   ~−0.02..−0.04 in H fore-body; fully explains the AIRFOIL readings;
   cannot explain the spheroid (wrong magnitude fore, wrong sign aft).
3. **(b) axisymmetric/transverse-curvature physics** — Mangler content
   +0.01-0.02, transverse curvature −0.002, measured dp/ds constant across
   the layer: eliminated at the relevant magnitude.
4. **(d) extraction geometry** — eliminated: plate operator control (2.60-
   2.61 at matched Re_theta), edge-height/edge-convention ladders (±0.02),
   u_e peak position physical, Cp Bernoulli-consistent, outer profile flat
   to 0.1%, H(x) smooth to 0.001.  (For calibration: H(0.42) would need the
   normalizing u_e to be read 1% low to reach 2.59 — the profile gives no
   basis for that; sensitivity recorded in the JSON.)

**Consequence for the paper's alpha=0 story:** the 0040 note's chain stands
(fullness halves the amplifying coordinate and moves the e^N N=8 station
from 0.49 to 0.84; the model is e^N-consistent on its own mean flow), and
its cause is now measured: a solver-numerics mean-flow artifact specific to
this axisymmetric O-grid class, NOT model physics, NOT laminar-BL theory,
NOT the pressure distribution.  The model constants owe this case nothing
(consistent with the c_nu_ai ladder finding); the honest paper-level framing
is a solver/grid limitation carrying the front miss, pending the
discriminating run below.

## 7. Recommended discriminating run (NOT launched — user decision)

**re72a0 L1 rerun with `lowMachPreconditioner: true`, everything else
byte-identical** (config-only change, no new mesh; L1 costs ~1/8 of an L2
case under the campaign's converge_by_xtr protocol; L1's H deficit at 0.42
is 0.056 ± 0.01, comfortably above the ±0.01 instrument noise, and its
balance footprint is currently below probe resolution, so any movement is
attributable).  Readout: H(0.42)/H(0.70) via this script's extraction.
- H -> ~2.56-2.58: low-Mach Roe dissipation convicted; expect the L2 rerun
  to restore P ~ 0.07-class and pull the chi=1 front strongly forward
  (0040's field-H envelope: N=8 at 0.49 vs 0.84).
- H unchanged: faceting/staggering truncation convicted; the follow-up is
  an azimuthally-refined (N_C x2) L1 stripe (mesh regen via
  ogrid_spheroid.py, same cost class).

## Caveats

1. Momentum-balance residual on the spheroid includes wiggle-aliasing noise
   in the 5-ray FD (est. ~2% of scale near the wall); the near-wall lobe
   (+0.33/+0.83) is far above it and hx-robust, but the mid-layer
   nu_eff/nu quotient should not be read as a literal viscosity.
2. The march assumes first-order BL (no normal pressure gradient); the
   measured dp/ds constancy justifies it a posteriori at 0.42/0.70.
3. Aft of x ~ 0.85 the march approaches separation and the tail region is
   out of scope (the model front at 0.92 sits there; claims about the front
   LOCATION still route through the 0040 e^N-on-field-H argument).
4. The L0/L1 probed profiles carry larger surface-faceting sagitta offsets
   (up to 43 h0 at L0); their H station values are correspondingly less
   certain than L2's (±0.02-0.03 est.), though the L0-L2 non-convergence is
   far outside that.
5. Eppler stations are an adverse-gradient/bubble control, not a
   matched-Re_theta control (Re_theta 164-224).
