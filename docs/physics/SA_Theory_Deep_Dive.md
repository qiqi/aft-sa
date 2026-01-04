# Spalart-Allmaras Turbulence Model: Theory and Derivation

Based on: "A One-Equation Turbulence Model for Aerodynamic Flows" (Spalart & Allmaras, AIAA 92-0439, 1992)

---

## 1. The Transport Equation

The SA model solves for $\tilde{\nu}$, a working variable closely related to turbulent kinematic viscosity:

$$
\frac{D\tilde{\nu}}{Dt} = \underbrace{c_{b1} \tilde{S} \tilde{\nu}}_{\text{Production}} - \underbrace{c_{w1} f_w \left(\frac{\tilde{\nu}}{d}\right)^2}_{\text{Destruction}} + \underbrace{\frac{1}{\sigma}\nabla \cdot [(\nu + \tilde{\nu})\nabla\tilde{\nu}]}_{\text{Diffusion (FV form)}} + \underbrace{\frac{c_{b2}}{\sigma}|\nabla\tilde{\nu}|^2}_{\text{cb2 term}}
$$

**Key insight**: The diffusion is written with $(\nu + \tilde{\nu})$ **inside** the divergence. This is the correct form for finite volume implementation.

---

## 2. Relationship to Turbulent Viscosity

The physical turbulent viscosity $\nu_t$ is related to $\tilde{\nu}$ by:

$$
\nu_t = \tilde{\nu} f_{v1}(\chi), \quad \chi = \frac{\tilde{\nu}}{\nu}
$$

$$
f_{v1} = \frac{\chi^3}{\chi^3 + c_{v1}^3}
$$

**Near-wall behavior**: As $\chi \to 0$, $f_{v1} \sim \chi^3/c_{v1}^3$, so $\nu_t \sim \chi^4 \sim y^4$. This ensures proper wall damping.

---

## 3. Model Constants

| Constant | Value | Origin |
|----------|-------|--------|
| $c_{b1}$ | 0.1355 | Calibrated to mixing length |
| $c_{b2}$ | 0.622 | Calibrated to free shear layers |
| $\sigma$ | 2/3 | Diffusion Prandtl number |
| $\kappa$ | 0.41 | Von Karman constant |
| $c_{v1}$ | 7.1 | Wall damping ($f_{v1} = 0.5$ at $\chi = c_{v1}$) |
| $c_{w2}$ | 0.3 | Controls $f_w$ shape |
| $c_{w3}$ | 2.0 | Limits destruction at large $r$ |
| $c_{w1}$ | **derived** | From log-layer balance (see Section 4) |

---

## 4. The Log-Layer Calibration ($c_{w1}$)

In the log layer ($y^+ > 30$), the balance includes Production, Destruction, **AND Diffusion**.

**Unlike other models, Diffusion does NOT vanish here.**

$$\text{Production} + \text{Diffusion} = \text{Destruction}$$

### The Derivation

1. **Assume Log Law**: $\tilde{\nu} = \kappa u_\tau y$.

2. **Production**: $P = c_{b1} \tilde{S} \tilde{\nu} \approx c_{b1} u_\tau^2$.

3. **Destruction**: $D = c_{w1} (\tilde{\nu}/d)^2 \approx c_{w1} \kappa^2 u_\tau^2$.

4. **Diffusion**:
   - Since $\tilde{\nu} \propto y$, the gradient $\nabla \tilde{\nu}$ is constant ($\kappa u_\tau$).
   - $\text{Diff} = \frac{1}{\sigma} [ \nabla \cdot (\tilde{\nu} \nabla \tilde{\nu}) + c_{b2} (\nabla \tilde{\nu})^2 ]$.
   - The divergence term: $\nabla \cdot (\tilde{\nu} \nabla \tilde{\nu}) = \tilde{\nu} \nabla^2 \tilde{\nu} + |\nabla \tilde{\nu}|^2 = 0 + \kappa^2 u_\tau^2$.
   - The cb2 term: $c_{b2} |\nabla \tilde{\nu}|^2 = c_{b2} \kappa^2 u_\tau^2$.
   - Total Diff $= \frac{1+c_{b2}}{\sigma} \kappa^2 u_\tau^2$.

5. **The Identity** ($P + \text{Diff} = D$):

$$c_{b1} + \frac{1+c_{b2}}{\sigma} \kappa^2 = c_{w1} \kappa^2$$

Rearranging gives the definition of $c_{w1}$:

$$\boxed{c_{w1} = \frac{c_{b1}}{\kappa^2} + \frac{1+c_{b2}}{\sigma}}$$

### Numerical Verification

With standard constants:
- $c_{b1}/\kappa^2 = 0.1355 / 0.41^2 = 0.806$
- $(1+c_{b2})/\sigma = (1+0.622)/(2/3) = 2.433$
- $c_{w1} = 0.806 + 2.433 = 3.239$

### The Logic Hierarchy (CRITICAL FOR DEBUGGING)

**The SA model does NOT cause the Log Law. The Log Law is a requirement of the Momentum Equation; the SA model is merely calibrated to support it.**

This distinction is vital for debugging: if the velocity profile isn't logarithmic, debugging the SA production terms is a waste of time.

1. **The Origin (Momentum Equation):**
   In the log layer, Total Stress $\tau \approx \tau_w$ (constant).
   Since $\tau \approx \nu_t \frac{\partial u}{\partial y}$, and we observe $u \propto \ln y$, physics **DEMANDS** that $\nu_t \propto y$.
   *This requirement comes from the Navier-Stokes equations, not the SA model.*

2. **The Response (SA Model):**
   The SA model terms ($P, D, \text{Diff}$) are **engineered** specifically so that the linear profile $\tilde{\nu} = \kappa u_\tau y$ is a stable solution to the transport equation.

### Debugging Flowchart

If you observe $\chi \neq \kappa y^+$ (Linear Law violated):

```
┌─────────────────────────────────────────────────────────────┐
│  1. CHECK MOMENTUM FIRST: Is u⁺ logarithmic?                │
│     ├─ NO  → Error is in FLOW SOLVER (Fluxes/BCs)          │
│     │        The SA model is INNOCENT. Stop here.           │
│     └─ YES → Proceed to step 2                              │
├─────────────────────────────────────────────────────────────┤
│  2. CHECK GEOMETRY: Is wall distance d correct?             │
│     ├─ NO  → Fix wall distance calculation                  │
│     └─ YES → Proceed to step 3                              │
├─────────────────────────────────────────────────────────────┤
│  3. CHECK SA MODEL: P, D, Diff failing to balance           │
│     • Check signs of all terms                              │
│     • Check constants (cb1, cw1, sigma)                     │
│     • Check diffusion coefficient (ν + ν̃)/σ                 │
└─────────────────────────────────────────────────────────────┘
```

**Key insight**: Most "turbulence model bugs" are actually flow solver bugs. The momentum equation must produce a logarithmic velocity profile *first*, then the SA model maintains the corresponding linear $\tilde{\nu}$ profile.

---

## 5. The P/D Ratio in the Log Layer

**Debugging Implication**: If you check the ratio $P/D$ in the log layer, it will **NOT** be 1.0.

From the balance $P + \text{Diff} = D$:

$$\frac{P}{D} = 1 - \frac{\text{Diff}}{D} = 1 - \frac{(1+c_{b2})/\sigma \cdot \kappa^2 u_\tau^2}{c_{w1} \kappa^2 u_\tau^2} = 1 - \frac{1+c_{b2}}{\sigma c_{w1}}$$

Numerically:

$$\frac{P}{D} = 1 - \frac{2.433}{3.239} = 1 - 0.751 = 0.249$$

**Expected P/D ratio in log layer: ~0.25** (not 1.0!)

### What to Assert Instead

**Do not assert P=D.** Assert the full balance:

$$P + \text{Diff} - D \approx 0$$

Or equivalently, check:

$$\frac{P + \text{Diff}}{D} \approx 1.0$$

---

## 6. The Equilibrium Parameter $r$

$$r = \frac{\tilde{\nu}}{\tilde{S} \kappa^2 d^2}$$

**At equilibrium ($P + \text{Diff} = D$):**
- In the log layer with $\tilde{\nu} = \kappa d u_\tau$ and $\tilde{S} \approx u_\tau/(\kappa d)$:

$$r = \frac{\kappa d u_\tau}{(u_\tau/\kappa d) \kappa^2 d^2} = \frac{\kappa d u_\tau}{\kappa u_\tau d} = 1$$

So $r \approx 1$ still holds in the log layer, which means $f_w \approx 1$.

**Interpretation**:
- $r < 1$: $\tilde{\nu}$ is "too low" relative to $\tilde{S}$ → $f_w < 1$ → reduced destruction → $\tilde{\nu}$ increases
- $r > 1$: $\tilde{\nu}$ is "too high" → $f_w > 1$ → enhanced destruction → $\tilde{\nu}$ decreases
- $r = 1$: Equilibrium (but equilibrium is $P + \text{Diff} = D$, not $P = D$!)

---

## 7. Expected $\chi$ Profile

In the log layer, $\chi = \tilde{\nu}/\nu$ should follow:

$$\chi = \kappa y^+ = 0.41 \cdot y^+$$

| $y^+$ | Expected $\chi$ |
|-------|----------------|
| 30    | 12.3           |
| 50    | 20.5           |
| 100   | 41.0           |

**Debugging**: If $\chi$ is significantly lower than $\kappa y^+$, the model is not producing enough turbulent viscosity. Check:
1. Vorticity calculation $\tilde{S}$
2. Wall distance $d$
3. Production term implementation
4. Diffusion term sign (should be positive, adding $\tilde{\nu}$)

---

## 8. The Viscous Sublayer ($y^+ < 5$)

In the viscous sublayer:
- $f_{v1} \to 0$ (cubic damping)
- $\nu_t = \tilde{\nu} f_{v1} \propto y^4$
- Destruction dominates production

**Wall boundary condition**: $\tilde{\nu} = 0$ at the wall.

This means $\tilde{\nu}$ must grow from zero at the wall. The diffusion term drives this growth by transporting $\tilde{\nu}$ from the outer region toward the wall.

---

## 9. Implementation Notes

### Finite Volume Form

The diffusion term should be implemented as:

$$\text{Diff} = \frac{1}{\sigma} \nabla \cdot [(\nu + \tilde{\nu}) \nabla \tilde{\nu}] + \frac{c_{b2}}{\sigma} |\nabla \tilde{\nu}|^2$$

In FV form, for each face:
$$F_{\text{diff}} = \frac{1}{\sigma} (\nu + \tilde{\nu})_{\text{face}} \cdot (\nabla \tilde{\nu} \cdot \mathbf{n}) \cdot S$$

The $c_{b2}$ term is added as a cell-centered source:
$$S_{cb2} = \frac{c_{b2}}{\sigma} |\nabla \tilde{\nu}|^2$$

**Why separate?** The FV divergence naturally produces the $|\nabla \tilde{\nu}|^2$ term from the product rule when $(\nu + \tilde{\nu})$ is inside. The $c_{b2}$ term adds an additional $c_{b2} \cdot |\nabla \tilde{\nu}|^2$ contribution, for a total of:

$$\text{Total } |\nabla \tilde{\nu}|^2 \text{ coefficient} = \frac{1 + c_{b2}}{\sigma}$$

### Momentum Equations

The momentum viscous stress follows the same FV pattern:

$$\tau_{ij} = \mu_{\text{eff}} \left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right)$$

With $\mu_{\text{eff}} = \mu + \mu_t$ placed at faces (averaged from cells) and multiplied by the velocity gradient dotted with the face normal.

---

## 10. Debugging Checklist

1. **Check $\chi$ profile**: Should be $\chi \approx \kappa y^+$ in log layer.

2. **Check P/D ratio**: Should be ~0.25, NOT ~1.0.

3. **Check full balance**: $(P + \text{Diff})/D$ should be ~1.0.

4. **Check $r$ parameter**: Should be ~1.0 in log layer.

5. **Check $f_w$**: Should be ~1.0 in log layer.

6. **Check diffusion sign**: Diffusion should generally be positive (adding $\tilde{\nu}$) in the log layer where $\tilde{\nu}$ is linear.

7. **Check wall damping**: $f_{v1}$ should be small ($< 0.1$) for $y^+ < 5$.

8. **Check vorticity in outer BL**: If Production → 0 in the log layer (y+ = 30-200), check vorticity calculation. Should have $\Omega \approx u_\tau / (\kappa y)$.

---

## 11. Grid Requirements for SA Model

**CRITICAL**: The SA model requires adequate grid resolution throughout the boundary layer, not just at the wall!

### Common Problem: Aggressive Grid Stretching

If the grid stretching ratio is too high (e.g., > 1.5), the outer boundary layer cells become too large to resolve velocity gradients accurately. This causes:

1. **Vorticity → 0** in the outer BL (gradient calculation error)
2. **Production → 0** (since $P = c_{b1} \tilde{S} \tilde{\nu}$ and $\tilde{S} \approx \Omega$)
3. **$\tilde{\nu}$ cannot grow** → $\chi$ plateaus at initial/freestream value
4. **Incorrect turbulent viscosity** in the log layer

### Symptoms

- $u^+$ profile may be approximately correct (momentum adapts)
- But $\chi$ plateaus at ~5-10 instead of growing as $\kappa y^+$
- $P/D > 1$ in log layer but $\chi$ still doesn't grow
- Vorticity ratio $\Omega / \Omega_{expected} \ll 1$ at $y^+ > 100$

### Grid Requirements

| Region | Requirement |
|--------|-------------|
| Viscous sublayer ($y^+ < 5$) | First cell $y^+ \approx 1$ |
| Buffer layer ($5 < y^+ < 30$) | 5-10 cells |
| Log layer ($30 < y^+ < 200$) | 10-15 cells |
| Outer BL ($200 < y^+ < \delta^+$) | 5-10 cells |

**Recommended stretching ratio**: 1.15-1.25 (NOT 2.0+!)

### Verification

Check vorticity at various $y^+$ locations:
```
Expected: Ω = u_τ / (κ y)  for y+ > 30
Acceptable: Ω_computed / Ω_expected > 0.5
Bad: Ω_computed / Ω_expected < 0.1
```

If vorticity drops below 10% of expected in the log layer, the grid is too coarse.

---

## Appendix: Rigorous Derivation of the Log Layer Calibration

This derivation proves that the linear slope of $\hat{\nu}$ is determined by the balance of constants. We calculate the slope produced by the model ($S_{\hat{\nu}}$) and then tune $c_{w1}$ to match the physical requirement ($S_{\hat{\nu}} = \kappa u_\tau$).

### Part 1: Momentum Equation (The Physical Requirement)

In the constant stress layer ($y^+ > 30$), $\tau_{total} \approx \rho u_\tau^2$.

If we assume a linear viscosity profile $\hat{\nu} = S_{\hat{\nu}} y$, the momentum equation dictates the velocity gradient:

$$\tau \approx \rho (S_{\hat{\nu}} y) \frac{\partial u}{\partial y} = \rho u_\tau^2 \implies \frac{\partial u}{\partial y} = \frac{u_\tau^2}{S_{\hat{\nu}} y}$$

Integrating this yields a logarithmic profile. For this profile to match the Kármán Log Law ($u = \frac{u_\tau}{\kappa} \ln y$), we must satisfy:

$$\boxed{S_{\hat{\nu}} = \kappa u_\tau}$$

**Key insight**: The slope $\kappa u_\tau$ is **demanded by the momentum equation**, not chosen arbitrarily.

### Part 2: SA Transport Equation (The Model's Output)

We solve the SA equation ($P + Diff = D$) for the unknown slope $S_{\hat{\nu}}$, assuming $\hat{\nu} = S_{\hat{\nu}} y$.

**1. Production ($P$):**

Substitute $\tilde{S} \approx \partial u / \partial y = u_\tau^2 / (S_{\hat{\nu}} y)$:

$$P = c_{b1} \tilde{S} \hat{\nu} = c_{b1} \left( \frac{u_\tau^2}{S_{\hat{\nu}} y} \right) (S_{\hat{\nu}} y) = c_{b1} u_\tau^2$$

**Critical observation**: The slope $S_{\hat{\nu}}$ cancels out in Production! This means $P$ is independent of the viscosity profile slope.

**2. Diffusion ($Diff$):**

$$Diff = \frac{1}{\sigma} \left[ \nabla \cdot (\hat{\nu} \nabla \hat{\nu}) + c_{b2} (\nabla \hat{\nu})^2 \right]$$

With $\nabla \hat{\nu} = S_{\hat{\nu}}$ (constant):

$$Diff = \frac{1}{\sigma} [ S_{\hat{\nu}}^2 + c_{b2} S_{\hat{\nu}}^2 ] = \frac{1+c_{b2}}{\sigma} S_{\hat{\nu}}^2$$

**3. Destruction ($D$):**

$$D = c_{w1} \left( \frac{\hat{\nu}}{d} \right)^2 = c_{w1} S_{\hat{\nu}}^2$$

**4. Solving for $S_{\hat{\nu}}$:**

From $P + Diff = D$:

$$c_{b1} u_\tau^2 + \frac{1+c_{b2}}{\sigma} S_{\hat{\nu}}^2 = c_{w1} S_{\hat{\nu}}^2$$

Rearranging:

$$c_{b1} u_\tau^2 = \left( c_{w1} - \frac{1+c_{b2}}{\sigma} \right) S_{\hat{\nu}}^2$$

$$\boxed{S_{\hat{\nu}} = u_\tau \sqrt{ \frac{c_{b1}}{c_{w1} - \frac{1+c_{b2}}{\sigma}} }}$$

### Part 3: Calibration of $c_{w1}$

To match the physical requirement from Part 1 ($S_{\hat{\nu}} = \kappa u_\tau$), we equate:

$$\kappa = \sqrt{ \frac{c_{b1}}{c_{w1} - \frac{1+c_{b2}}{\sigma}} }$$

Squaring both sides:

$$\kappa^2 = \frac{c_{b1}}{c_{w1} - \frac{1+c_{b2}}{\sigma}}$$

Solving for $c_{w1}$:

$$c_{w1} - \frac{1+c_{b2}}{\sigma} = \frac{c_{b1}}{\kappa^2}$$

$$\boxed{c_{w1} = \frac{c_{b1}}{\kappa^2} + \frac{1+c_{b2}}{\sigma}}$$

### Numerical Verification

| Constant | Value | Computation |
|----------|-------|-------------|
| $c_{b1}$ | 0.1355 | Given |
| $\kappa$ | 0.41 | Von Kármán |
| $c_{b2}$ | 0.622 | Given |
| $\sigma$ | 2/3 | Given |
| $c_{b1}/\kappa^2$ | 0.806 | $0.1355 / 0.1681$ |
| $(1+c_{b2})/\sigma$ | 2.433 | $1.622 / 0.667$ |
| $c_{w1}$ | **3.239** | $0.806 + 2.433$ |

### Conclusion

The constant $c_{w1}$ is **not arbitrary**. It is the "tuning knob" that forces the SA model to produce the correct Von Kármán slope $\kappa = 0.41$.

**Debugging implication**: If you change any of $c_{b1}$, $c_{b2}$, $\sigma$, or $\kappa$, you **must** recalculate $c_{w1}$ using the formula above, or the model will produce the wrong viscosity slope and the momentum equation will not yield the log law.
