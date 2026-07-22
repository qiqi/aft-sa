# Absolute vs convective instability of separated shear layers — literature criteria

Context: the frozen-profile study (`explore_lsb_frozen_profile.py`) showed the
model's LSB over-amplification is an advection-denominator effect: in-place
(temporal-in-lab-frame) growth at the u=0 line where e^N assumes convective
growth at the wave speed c ≈ 0.42 U_e. A soft gate correction must NOT shut
this off unconditionally, because a sufficiently thick/deep bubble genuinely
becomes ABSOLUTELY unstable and in-place growth is then physical. This note
collects the rigorous criteria that separate the two regimes.

## 1. The rigorous definition (Briggs–Bers)

Briggs (1964); Bers; review: Huerre & Monkewitz, Annu. Rev. Fluid Mech. 22,
473 (1990). For a parallel profile with dispersion relation D(k, w) = 0:

* Convectively unstable: the impulse response grows but is swept away —
  along every ray x/t = V = 0 the response decays; growth only along a band
  of rays V in [V-, V+] with 0 < V- (wavepacket trailing edge moves
  downstream).
* Absolutely unstable: the response grows AT the source — the V = 0 ray
  grows. Criterion: the saddle point w0 (dw/dk = 0 at k0, formed by the
  PINCHING of a downstream k+ and an upstream k- branch) has Im(w0) > 0.
  Im(w0) is the absolute growth rate. (Saddles that are not pinch points do
  not count — branch collision must be verified, e.g. by continuation from a
  known convective state.)

The distinction is frame-dependent; the wall/bubble frame is the physically
selected frame here (the bubble is attached to the geometry).

## 2. Free shear layer: the counterflow threshold

Huerre & Monkewitz, J. Fluid Mech. 159, 151 (1985): the tanh mixing layer
u = U_m + (DU/2) tanh(y), velocity ratio R = DU / (2 U_m):

* Convective for R < 1.315; absolutely unstable for R > 1.315.
* R = 1 is a single stream ending at zero velocity; R > 1 requires
  COUNTERFLOW. R = 1.315 means back-flow of (R-1)/(R+1) = 13.6% of the
  forward stream. This is the canonical order of magnitude: a separated
  shear layer needs O(15%) reverse flow before in-place growth is physical.

## 3. Separated-boundary-layer profiles (wall present)

Reverse-flow thresholds for the onset of 2D KH-branch absolute instability,
all expressed as peak reverse flow u_rev / U_e on frozen local profiles:

* Hammond & Redekopp, Eur. J. Mech. B/Fluids 17, 145 (1998): model
  separated profiles; the inflection-point mode near the dividing
  streamline becomes absolutely unstable at u_rev ~ 20–30%. Wall proximity
  (image vorticity) is stabilizing.
* Alam & Sandham, J. Fluid Mech. 403, 223 (2000): DNS of short LSBs;
  absolute instability expected for u_rev ~ 15–20%.
* Rist & Maucher, ~2002 (and Maucher, Rist & Wagner): u_rev ~ 16–20% AND a
  geometric requirement: the height of the u = 0 line above the wall must
  exceed ~0.6 of the local displacement thickness — the shear layer must
  sit high enough off the wall. (The "thick enough bubble" effect.)
* Fasel & Postl; Diwan & Ramesh, J. Fluid Mech. 629, 263 (2009); Embacher &
  Fasel, J. Fluid Mech. 748 (2014): same 12–25% band; Diwan & Ramesh
  emphasize the wall distance of the inflection point for the instability
  character.

The scatter (12–30%) exists because peak reverse flow is NOT the collapsing
parameter — wall distance and profile shape enter independently.

## 4. The collapsed criterion (Avanci–Rodríguez–Alves)

Avanci, Rodríguez & Alves, Phys. Fluids 31, 014103 (2019), "A geometrical
criterion for absolute instability in separated boundary layers":

    ABSOLUTE  <=>  y_i < y_b,

where y_i is the wall-normal position of the inflection point of u(y) and
y_b the zero net mass-flux height (integral of u from 0 to y_b = 0). I.e.
the profile is absolutely unstable iff the inflection point lies INSIDE the
recirculation region (below the zero-mass-flux line). This collapses the
convective/absolute boundary across wide profile families where the
%-reverse-flow thresholds scatter. Wall proximity and shape effects are
automatically encoded (deep bubble -> shear layer and its inflection sink
into the recirculation; shallow bubble -> inflection rides above it).

Physical reading: the instability is driven at the inflection point; if that
point is embedded in the back-flow, the wave's own critical layer is being
advected upstream — the wavepacket can no longer be washed downstream.

## 5. From local to global: when does the flow self-excite?

Chomaz, Huerre & Redekopp, Phys. Rev. Lett. 60, 25 (1988); Stud. Appl. Math.
84, 119 (1991); Monkewitz, Huerre & Chomaz, J. Fluid Mech. 251 (1993):

* Local absolute instability is NECESSARY but not sufficient for
  self-sustained (global) oscillation: a finite POCKET of absolute
  instability of sufficient streamwise extent (order several instability
  wavelengths) is required.
* The global frequency is selected by the saddle of the local absolute
  frequency w0(X) continued to complex X.
* Hammond & Redekopp (1998) applied exactly this framework to bubbles:
  a bubble with a long-enough absolutely-unstable pocket hosts a global
  mode (self-sustained shedding, "bursting" dynamics).

## 6. The OTHER self-excited route: 3D centrifugal global instability

Rodríguez & Theofilis (2010); Rodríguez, Gennaro & Juniper, "The two
classes of primary modal instability in laminar separation bubbles" (2013);
Rodríguez, Gennaro & Souza, J. Fluid Mech. (2021):

* A steady, three-dimensional, zero-frequency global mode (centrifugal
  mechanism at the recirculation) self-excites at peak reverse flow as low
  as ~7% — well below the 2D KH-absolute threshold. It breaks spanwise
  homogeneity (supercritical pitchfork) and modulates the bubble in z.
* This is a DIFFERENT mechanism from the 2D absolute instability: it does
  not by itself produce in-place high-frequency amplification of the KH
  branch; it deforms the bubble and can promote secondary transition.

## 7. What this means for the model (design constraints, not yet a design)

1. e^N-consistent behavior (convective): for u_rev below the absolute
   threshold the correct steady growth is along the advection path at the
   wave speed c ~ 0.4 U_e. The model's in-place growth at the u ~ 0 line is
   unphysical there — this is the identified over-amplification.
2. The gate must reopen (allow in-place growth) when the LOCAL profile
   crosses the absolute threshold; the rigorous local switch is the Avanci
   geometric criterion y_i < y_b (equivalently: the inflection point of the
   shear layer sits inside the recirculation), NOT a fixed reverse-flow
   percentage.
3. The switch must be SOFT: Im(w0) rises continuously from 0 at the
   threshold (HM85: Im(w0) ~ (R - R_crit) near onset), so a physically
   scaled soft gate can use the margin past the criterion as its argument.
4. Pointwise-locality is the hard constraint: y_i is where d2u/dn2 = 0
   (the Lambda_v = 0 crossing!), and "inside the recirculation" involves
   the sign of u relative to the outer stream — a nonlocal integral
   (y_b) or a signed-velocity surrogate (e.g. u . (n x omega) sign
   convention distinguishes the back-flow side under a shear layer) must be
   found. To be settled AFTER the in-house Briggs-Bers verification.

## 8. In-house verification plan (explore_lsb_pinchpoint.py)

Compute the Briggs-Bers pinch point on our frozen Stewartson profiles with
a reverse-flow-deepening knob: track the KH-branch saddle w0 in the complex
wavenumber plane by continuation from the convective side; find the
u_rev where Im(w0) = 0; check the crossing coincides with the Avanci
y_i = y_b condition on the same family; extract Im(w0) vs margin for the
soft-gate scaling.

## 9. In-house verification results (explore_pinchpoint_shooting.py, 2026-07-18)

Shooting dispersion function D(alpha, c) = phi(wall) (analytic in both
arguments; complex contour below the critical layer; Chebyshev continuation
for tabulated profiles). Validation: HM85 counterflow threshold reproduced
to R_crit = 1.3157 vs 1.315 (0.05%).

Avanci-label verification on the Q2 calibration family: 29/32 agree.
The 3 disagreements are all at h = 8 (very tall detached layer), where the
geometric criterion OVER-predicts absolute instability: Briggs-Bers keeps
u_r = 6-14% convective (Im w0 < 0) although the inflection already sits
below the zero-mass-flux line; true onset there is ~16-17% reverse flow.
For LSB-typical heights (h = 3-6, the FS-like regime) agreement is perfect
and the threshold tracks Im w0 = 0 closely (|Im w0| < 0.003 at the labeled
boundary). Conclusion: the Avanci label is exact where the pocket operates
(thin/moderate layers) and conservative for tall detached layers -- the
pocket design is unaffected (tall-layer crossings lie outside it anyway).
