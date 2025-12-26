# Source term in turbulent inner layer

The goal is to check whether using the *small* part of the SA turbulence variable as an $e^N$ or AFT-type transition variable would contaminate the **fully turbulent** region of the boundary layer.

In a standard SA model, the working variable is proportional to an eddy viscosity $\nu_t$.  In laminar and very early transitional flow this variable is essentially unused: $\nu_t$ is nearly zero and contributes nothing to the mean flow.  The idea is to let this low range encode a TS-wave amplification quantity $\hat\nu$ (or $e^{N-N_{crit}}$) and then, once $\hat\nu$ reaches order one, switch over to the standard SA dynamics so that $\nu_t$ grows to its usual turbulent level.

To see whether this could disturb an already-turbulent wall layer, we examine a **canonical turbulent inner profile** given by Spaulding’s law of the wall.

---

### Setup with Spaulding profile

Spaulding’s curve is written in wall units as a relation between $u^+$ and $y^+$

$$
y^+(u^+)
= u^+ + \exp(-\kappa B)
\Big(
\exp(\kappa u^+)
- 1
- \kappa u^+
- (\kappa u^+)^2 / 2
- (\kappa u^+)^3 / 6
- (\kappa u^+)^4 / 24
\Big),
$$

with $\kappa = 0.41$ and $B = 5.2$.  From this we compute:

* the gradient in wall units
  $$\frac{\mathrm{d}u^+}{\mathrm{d}y^+},$$
* the implied eddy viscosity in wall units
  $$\nu_t^+ = \frac{1}{\mathrm{d}u^+/\mathrm{d}y^+} - 1,$$
  using the usual relation $\mathrm{d}u^+/\mathrm{d}y^+ = 1/(1 + \nu_t^+)$,
* the "normal Reynolds number" used in the transition model
  $$Re_\nu = y^{+2}\,\frac{\mathrm{d}u^+}{\mathrm{d}y^+},$$
* and the local amplification source
  $$s(y^+) = C_\text{trans}\,\frac{\mathrm{d}u^+}{\mathrm{d}y^+}
  \max(Re_\nu - Re_\text{crit},0),$$
  with the same $C_\text{trans}$ and $Re_\text{crit}$ used in the Blasius-based AFT model.

The figure has three panels:

1. left: $u^+$ and $\mathrm{d}u^+/\mathrm{d}y^+$ versus $y^+$
2. middle: $Re_\nu$ and the source $s$ versus $y^+$
3. right: $\nu_t^+$ versus $y^+$

All axes for $y^+$ are logarithmic, covering the viscous sublayer, buffer layer, and log layer.

---

### What the plot shows

**Panel 1: inner profile and gradient**

* The red curve $u^+(y^+)$ smoothly connects the linear viscous region (slope one) to the log layer.
* The blue dashed curve $\mathrm{d}u^+/\mathrm{d}y^+$ starts near one at $y^+ \approx 1$, decreases through the buffer layer, and behaves roughly like $1/(\kappa y^+)$ in the log layer.

This is the standard universal near-wall behavior, independent of pressure gradient and outer flow, which is exactly where we want the SA model to remain untouched.

**Panel 2: $Re_\nu$ and amplification source**

* The red curve shows that $Re_\nu = y^{+2},\mathrm{d}u^+/\mathrm{d}y^+$ is small in the viscous and buffer layers and then grows approximately proportional to $y^+$ in the log layer.  With the chosen $Re_\text{crit} \approx 400$, the condition $Re_\nu > Re_\text{crit}$ is not met until $y^+$ is well into the log region, around a few hundred.
* The blue dashed curve is the amplification source $s(y^+)$.  It is exactly zero for $y^+ \lesssim 100$ and only becomes significant once $Re_\nu$ exceeds the critical value.  That is, the **transition model would not “turn on’’ until relatively far from the wall**, in the middle of the log layer.

**Panel 3: eddy viscosity level where amplification starts**

* The red curve $\nu_t^+(y^+)$ increases from zero near the wall and reaches order one in the buffer layer, then grows roughly like $\kappa y^+$ in the log layer.
* At the $y^+$ where $Re_\nu$ first exceeds $Re_\text{crit}$ and $s$ becomes nonzero, $\nu_t^+$ is already large, on the order of $50$ to $60$.  In other words, by the time the AFT-style source would be active in this turbulent profile, the SA eddy viscosity has already reached a value very much larger than the laminar viscosity.

---

### Interpretation and consequence for the SA-based transition idea

These results support the key safety check for the proposed transition model:

* In the **inner and buffer layers** ($y^+ \lesssim 50$), which are universal and crucial for wall shear stress and near-wall turbulence structure, the AFT-style amplification source $s$ is effectively zero.  The SA solution there would behave exactly as in the standard fully turbulent model.
* The **onset of nonzero amplification** occurs only in the **middle of the log layer**, where $\nu_t^+ \gg 1$.  At those locations the SA working variable is already dominated by turbulence physics, and any transient use of its *low* subrange as a transition marker should be turned off, based on a $\nu_t$ sensor, so that it has negligible influence on the equilibrium turbulent state.
* In the **outer layer** at very large $y^+$, both $Re_\nu$ and any associated $\hat\nu$ or $N$ become even larger.  However, our transition logic, based on a $\nu_t$ sensor, should already have switched over to the pure SA model before $\hat\nu$ reaches order 50, so the outer layer of an established turbulent boundary layer should be even less sensitive to the transitional modification.

Therefore, this Spaulding-based test indicates that our SA-based transition idea has a fairly large **safe range** in the inner layer: the modified dynamics do not disturb the canonical turbulent profile until the transition variable reaches $\hat\nu \sim 50$.  In other words, we can not only use the low range $\hat\nu < 1$ to represent linear TS-wave amplification in an $e^N$ or AFT sense, but also use the intermediate range $1 < \hat\nu < 50$ to model the **rapid late stage of transition**, driving $\hat\nu$ quickly up to its usual turbulent SA level and preventing unphysical relaminarization once breakdown has begun.
