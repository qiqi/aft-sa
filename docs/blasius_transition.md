# Evolution of nuHat in Blasius laminar boundary layer

## Blasius profile

In this plot we show the **self-similar Blasius boundary-layer solution** using the similarity coordinate $\eta = y\sqrt{U_\infty/(\nu x)}$. The red curve is the non-dimensional streamwise velocity $u/U_\infty = f'(\eta)$, which starts from zero at the wall (no-slip condition) and smoothly approaches $1$ as $\eta$ increases, meaning the flow recovers to the free-stream velocity outside the boundary layer. The blue curve is the velocity gradient $\partial u/\partial\eta = f''(\eta)$; its large value near $\eta = 0$ corresponds to strong wall shear and skin friction, and it rapidly decays to nearly zero in the outer region where the flow is almost uniform.

The green curve shows the wall-normal velocity scaled as $v\sqrt{Re_x}$. From the Blasius similarity streamfunction
$$
\psi = \sqrt{\nu U_\infty x},f(\eta), \qquad \eta = y\sqrt{\frac{U_\infty}{\nu x}},
$$
one finds
$$
u = \frac{\partial\psi}{\partial y} = U_\infty f'(\eta),
\qquad
v = -\frac{\partial\psi}{\partial x}
= \tfrac12\sqrt{\frac{\nu U_\infty}{x}},[\eta f'(\eta) - f(\eta)],
$$
and therefore
$$
v\sqrt{Re_x}
= v\sqrt{\frac{U_\infty x}{\nu}}
= \tfrac12\bigl(\eta f'(\eta) - f(\eta)\bigr),
$$
which is exactly what is plotted (with $U_\infty = 1$). This combination is independent of $x$ and collapses to a single universal curve, just like $u/U_\infty$ and $f''(\eta)$. Physically, $v$ is much smaller than $U_\infty$ by a factor of $1/\sqrt{Re_x}$: near the wall it is zero (no penetration), then becomes slightly positive and finally tends toward a small nearly constant value of opposite sign far from the wall, representing the weak cross-stream motion that feeds slower fluid in the boundary layer by entraining fluid from the outer flow. Together, these three curves summarize how the Blasius profile simultaneously satisfies no-slip, conserves mass, and matches the outer inviscid flow.


In this figure we show Blasius flat-plate boundary layers for several streamwise Reynolds numbers $Re_x = U_\infty x/\nu$ from $10^4$ to $5\times10^5$.  On the bottom horizontal axis are the streamwise velocity profiles $u/U_\infty$ (solid lines), and on the top horizontal axis are the corresponding wall-normal velocities $v/U_\infty$ (dashed lines).  The vertical axis is the wall-normal coordinate made dimensionless by the viscous length scale, $yU_\infty/\nu$.  For each $Re_x$ the Blasius similarity solution gives $u/U_\infty = f'(\eta)$ and $v\sqrt{Re_x} = \tfrac12\bigl(\eta f'(\eta) - f(\eta)\bigr)$ with $\eta = y\sqrt{U_\infty/(\nu x)}$, which we then convert to $(yU_\infty/\nu, u/U_\infty, v/U_\infty)$ using $yU_\infty/\nu = \eta\sqrt{Re_x}$ and $v/U_\infty = \tfrac12\bigl(\eta f'(\eta) - f(\eta)\bigr)/\sqrt{Re_x}$.

The solid curves illustrate how the boundary-layer thickness grows roughly like $\delta^+ \sim 5\sqrt{Re_x}$: as you move to higher $Re_x$, the same Blasius shape is stretched vertically in $yU_\infty/\nu$.  The legend also lists the corresponding momentum-thickness Reynolds numbers $Re_\theta \approx 0.665\sqrt{Re_x}$, which run from about $Re_\theta \approx 70$ to $Re_\theta \approx 470$.  This range covers the regime where Tollmien–Schlichting waves first become unstable and then grow to amplitudes large enough that transition to turbulence is likely in low-disturbance environments, so these profiles are exactly the “base flows’’ used in classical linear stability and $e^N$ transition analyses.

The dashed curves show the wall-normal velocity $v/U_\infty$, which is everywhere much smaller than the streamwise velocity, with peak values of only a few $\times 10^{-3}$.  The reason $v$ is nonzero is conservation of mass, expressed in the boundary-layer continuity equation $\partial u/\partial x + \partial v/\partial y = 0$.  Inside the boundary layer, $u$ decreases in $x$ (the flow is slowed by viscosity); to satisfy continuity, a small outward $v$ is induced.

Near the wall, all dashed curves start from $v = 0$ due to the no-penetration condition.  Moving away from the wall, $v$ rises and then levels off as the flow approaches the nearly uniform outer stream.  The scaling $v/U_\infty \sim \delta/x \sim 1/\sqrt{Re_x}$ is evident: at larger $Re_x$ the boundary layer is thicker in viscous units, but the associated wall-normal velocities are even smaller in units of $U_\infty$.  Together, the solid and dashed families of curves give a complete picture of the laminar Blasius boundary layer in the Reynolds-number range that is most relevant for the onset and growth of instability waves leading to transition.

## Calculation of source term (pure laminar)


In this three-panel figure the Blasius boundary layers from the previous plots are now expressed in the language used by modern transition models. The left panel shows a “normal Reynolds number” $Re_\nu$ (a Reynolds-like quantity based on the local wall-normal coordinate and velocity) rescaled so that $Re_\theta \approx Re_\nu/2.2$ for a Blasius profile. Because the laminar flat-plate boundary layer is self-similar, the curves at different $Re_x$ nearly collapse when plotted in this way, so that the state of the boundary layer at a given height can be characterized mostly by $Re_\theta$ and $Re_\nu$ instead of the streamwise coordinate $x$. This is exactly the set of variables used in Drela’s approximate $e^N$ correlations for Tollmien–Schlichting (TS) growth in XFOIL/MSES, which were tuned to classical linear-stability results and flat-plate experiments.

The middle panel restates the same profiles in terms of the wall-normal shear $\mathrm{d}u/\mathrm{d}y$, highlighting the thin region of moderate shear above the no-slip wall where TS waves live and extract energy from the mean flow. In the classical $e^N$ “envelope method,” one solves the Orr–Sommerfeld problem on these mean profiles, integrates the local complex growth rate $\alpha_i(x)$ downstream, and defines an amplification factor
$$
N(x) = \int \alpha_i \,\mathrm{d}x.
$$
Transition is empirically observed when the envelope of $N$ reaches values of order $8$–$10$ for low free-stream turbulence. For a zero–pressure-gradient Blasius boundary layer this process produces nearly universal $N$-factor curves when expressed in terms of $Re_\nu$, with TS growth concentrated in a band of $Re_\nu$ roughly corresponding to the “bulge” in the shear profiles shown here.

The right panel shows the result of an AFT-like model evaluated on these Blasius profiles and calibrated to match Drela’s Blasius $e^N$ curve. Here the “local production” of $N$ is modeled as
$$
Re_\nu = y^2 \left|\frac{\mathrm{d}u}{\mathrm{d}y}\right| + 1,\qquad
S = C_{\text{trans}}\,\frac{\mathrm{d}u}{\mathrm{d}y}\,\max\bigl(Re_\nu - Re_{\text{crit}},\,0\bigr),
$$
with $Re_{\text{crit}} \approx 400$ and $C_{\text{trans}} = 10^{-4}$, and then converted to a growth rate with respect to $Re_\theta$ via the known Blasius relation $\mathrm{d}Re_\theta/\mathrm{d}x = 0.665/(2\sqrt{Re_x})$. When plotted as $\mathrm{d}N/\mathrm{d}Re_\theta$, the resulting profiles show a peak around $\mathrm{d}N/\mathrm{d}Re_\theta \sim 0.02$ in the unstable part of the layer and nearly zero elsewhere, in good agreement with Drela’s fitted Blasius envelope. This is very much in the spirit of Coder’s Amplification Factor Transport (AFT) model: instead of repeatedly solving the full stability eigenproblem, one solves a transport equation for $N$ whose production term is expressed as a simple function of $Re_\nu$ and $\mathrm{d}u/\mathrm{d}y$, but still reproduces the classic $e^N$ behavior when applied to canonical cases like the Blasius boundary layer.

### Comparison of growth rate and diffusion time scale

From these profiles we can also estimate the relative timescales of **growth** and **cross–stream diffusion** in the AFT equation.  With the chosen critical value $Re_\nu^\text{crit} \approx 400$, amplification first becomes positive when $Re_\nu \approx Re_\nu^\text{crit}$, which for a Blasius layer corresponds to $Re_\theta \approx Re_\nu^\text{crit}/2.2 \approx 180$ and, using $Re_\theta \approx 0.665\sqrt{Re_x}$, to $Re_x \approx 7\times 10^4$.  In the core of the unstable region near $Re_x \sim 10^5$ the correlation gives $\mathrm{d}N/\mathrm{d}Re_\theta \approx 0.02$ and $\mathrm{d}Re_\theta/\mathrm{d}Re_x \approx 0.665/(2\sqrt{Re_x})$, so $\mathrm{d}N/\mathrm{d}Re_x \approx 2\times 10^{-5}$ and an e–fold growth length of $\Delta Re_x^\text{grow} \sim 5\times 10^4$.  The TS–active band in the wall–normal direction has a thickness of order $\Delta Re_y \sim 10^3$ in these units, so the diffusive distance needed to mix a scalar across this band is $\Delta Re_x^\text{diff} \sim \Delta Re_y^2 \sim 10^6$.  Thus the **diffusion timescale is roughly one order of magnitude longer** than the linear growth timescale: disturbances experience many e–folds of amplification in $x$ before diffusion can homogenize them across the entire unstable layer.  In the AFT picture, diffusion mainly smooths the vertical shape of the envelope, while the onset and streamwise march of transition are controlled primarily by the growth rate encoded in $\mathrm{d}N/\mathrm{d}Re_\theta$.


## Transport and diffusion

Here we are solving a **parabolic advection–diffusion equation with amplification** on a Blasius flat-plate boundary layer, and then plotting the evolution of a scalar field $\hat\nu(x,y)$ that represents a **TS–wave–like disturbance intensity**.

---

### PDE and numerical model

The semi-discrete system in the code is

$$
M\,\frac{\partial \hat\nu}{\partial x} + A \hat\nu = b,
$$

where:

* $M = \mathrm{diag}(u)$ is a diagonal matrix of the local Blasius streamwise velocity $u(x,y)$,
* $A$ contains:

  * an upwind discretization of $v,\partial\hat\nu/\partial y$ using the Blasius wall-normal velocity $v(x,y)$, and
  * a second-derivative term $\partial^2\hat\nu/\partial y^2$ modeling cross-stream diffusion,
* $b$ is a source term $b(y) = \text{amplification}( \partial u/\partial y, y)$ that is nonzero only where the Blasius profile is TS-unstable.

In continuous form this corresponds to

$$
u\,\frac{\partial\hat\nu}{\partial x}
* v\,\frac{\partial\hat\nu}{\partial y}
  = \frac{\partial^2 \hat\nu}{\partial y^2}
* s(x,y),
$$

with $s(x,y)$ given by your amplification model based on $Re_\nu$:

$$
Re_\nu = y^2 \left|\frac{\partial u}{\partial y}\right| + 1, \qquad
s = C_\text{trans} \frac{\partial u}{\partial y}
\max\bigl(Re_\nu - Re_{\text{crit}}, 0\bigr).
$$

The marching scheme

$$
\bigl(M/\Delta x + A\bigr)\,\hat\nu^{n+1}
= (M/\Delta x)\,\hat\nu^n + b
$$

is an implicit Euler step in $x$ (treating $x$ like a “time” variable) with advection by $(u,v)$ and diffusion in $y$.

At the inlet you prescribe a very small initial disturbance

$$
\hat\nu(x=0,y) = e^{-9},
$$

so $\log \hat\nu \approx -9$ everywhere at the leading edge.

---

### What the contours show

You then contour the solution in $(Re_x, Re_y)$ coordinates, where

$$
Re_x = \frac{U_\infty x}{\nu}, \qquad
Re_y = \frac{U_\infty y}{\nu},
$$

and overlay the classical Blasius thickness scalings

* $Re_y = 0.665 \sqrt{Re_x}$ (dashed black) ≈ momentum thickness,
* $Re_y = 1.72 \sqrt{Re_x}$ (solid black) ≈ displacement thickness,
* $Re_y = 5 \sqrt{Re_x}$ (dotted black) ≈ $99%$ boundary-layer edge.

**Top panel:** you plot $\log \hat\nu$ with a red contour where $\log \hat\nu = 0$, i.e. $\hat\nu = 1$.
**Bottom panel:** you plot $\hat\nu$ itself, again highlighting the $\hat\nu = 1$ contour in red.

The color field shows how a tiny initial disturbance is:

* **advected downstream** by $u(x,y)$,
* **shifted vertically** by the weak wall-normal advection $v(x,y)$,
* **spread in $y$** by diffusion,
* and **amplified** in the TS-unstable region of the Blasius profile through the source term $s(x,y)$.

The unstable “pocket” in $(Re_x,Re_y)$ space appears as the bright region in the top plot: disturbances grow most strongly at intermediate $Re_y$ and in the streamwise range where the amplification model is active. Upstream of that pocket $\hat\nu$ remains close to its tiny inlet value; downstream, the amplification and diffusion have produced an $O(1)$ disturbance level over a significant part of the boundary layer.

---

### Transition criterion and link to $e^N$

The **transition criterion** in this model is simply

$$
\hat\nu \sim O(1).
$$

You start from $\hat\nu_0 = e^{-9}$, so reaching $\hat\nu \approx 1$ corresponds to roughly 9 e-folds of amplification, i.e. an $e^N$ factor with $N \approx 9$, which is very close to the classic $e^N$ transition thresholds used in low-disturbance wind-tunnel data for TS-driven transition.

Thus:

* The **red contour** $\hat\nu = 1$ marks where the flow is predicted to leave the linear regime and enter the nonlinear breakdown stage.
* Its position relative to the Blasius thickness curves shows that transition is triggered within the boundary layer, at $Re_\theta$ of a few hundred and $Re_y$ well below the outer edge, consistent with classical TS-wave theory.
* The overall pattern is exactly what one expects from an AFT-type transport model: a scalar envelope that starts tiny, is amplified where the base flow is unstable, and is used to declare transition once it reaches order unity.
