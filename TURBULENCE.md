# SA Turbulence Model Integration Plan

This document outlines the plan to integrate the Spalart-Allmaras (SA) turbulence equation into the RANS solver, including the AFT (Amplification Factor Transport) transition model.

## Current State Analysis

### RANS Solver (`src/solvers/rans_solver.py`)

The RANS solver already has the 4th state variable `ν̃` (nuHat) in the state vector:

```
Q = [p, u, v, ν̃]
```

**What it currently does:**
- Computes `μ_t = ν̃ · fv1(ν̃)` for effective viscosity
- Convects `ν̃` through the JST flux scheme
- Applies boundary conditions for `ν̃`

**What it does NOT do:**
- Compute SA production term: `P = cb1 · S̃ · ν̃`
- Compute SA destruction term: `D = cw1 · fw · (ν̃/d)²`
- Compute SA diffusion with cb2 term: `(cb2/σ)(∇ν̃)·(∇ν̃)`

### SA Model Implementation (`src/physics/spalart_allmaras.py`)

The SA model is **fully implemented and tested** in JAX:

| Function | Purpose | Status |
|----------|---------|--------|
| `fv1`, `fv2` | Damping functions with analytical gradients | ✅ Tested |
| `_S_tilde` | Modified vorticity S̃ = Ω + (ν̃/κ²d²)·fv2 | ✅ Tested |
| `r`, `g`, `fw` | Destruction term helper functions | ✅ Tested |
| `spalart_allmaras_amplification` | Production & Destruction with gradients | ✅ Tested |
| `compute_sa_production` | Safe production (handles negative ν̃) | ✅ Tested |
| `compute_sa_destruction` | Safe destruction | ✅ Tested |
| `compute_diffusion_coefficient_safe` | D = (ν + ν̃)/σ | ✅ Tested |
| `effective_viscosity_safe` | ν_eff = ν + max(0, ν̃·fv1) | ✅ Tested |

### Boundary Layer Solver Reference

The `NuHatFlatPlateSolver` in `src/solvers/boundary_layer_solvers.py` demonstrates working SA+AFT integration:

```python
# Turbulence indicator based on nuHat level
is_turb = jnp.clip(1 - jnp.exp(-(nuHat - 1) / 4), min=0.0)

# AFT amplification for laminar regime
Gamma = jnp.abs(dudy) * y_cell / jnp.abs(u)
a_aft = compute_nondimensional_amplification_rate(Re_Omega(dudy, y_cell), Gamma)

# Blend AFT (laminar) with SA (turbulent)
P_blended = P_sa * is_turb + a_aft * nuHat * (1 - is_turb)
```

---

## SA Transport Equation

The Spalart-Allmaras equation for the working variable `ν̃`:

```
∂ν̃/∂t + u·∇ν̃ = Production - Destruction + Diffusion
```

Where:
- **Production**: `P = cb1 · S̃ · ν̃`
- **Destruction**: `D = cw1 · fw · (ν̃/d)²`
- **Diffusion**: `(1/σ)∇·[(ν + ν̃)∇ν̃] + (cb2/σ)(∇ν̃)·(∇ν̃)`

### Model Constants

```python
cb1 = 0.1355
cb2 = 0.622
sigma = 2/3
kappa = 0.41
cw1 = cb1/kappa² + (1 + cb2)/sigma
cw2 = 0.3
cw3 = 2.0
cv1 = 7.1
```

---

## Integration Plan

### Phase 1: SA Source Term Module

**Create `src/numerics/sa_sources.py`:**

```python
from src.physics.jax_config import jax, jnp
from src.physics.spalart_allmaras import compute_sa_production, compute_sa_destruction
from src.numerics.gradients import compute_vorticity_jax

@jax.jit
def compute_sa_source_jax(nuHat, grad, wall_dist, nu_laminar, sigma=2.0/3.0, cb2=0.622):
    """
    Compute SA turbulence model source terms for 2D FVM.
    
    Parameters
    ----------
    nuHat : jnp.ndarray (NI, NJ)
        SA working variable from interior cells
    grad : jnp.ndarray (NI, NJ, 4, 2)
        Cell-centered gradients of Q
    wall_dist : jnp.ndarray (NI, NJ)
        Distance to nearest wall
    nu_laminar : float
        Kinematic viscosity (1/Re)
    
    Returns
    -------
    source : jnp.ndarray (NI, NJ)
        Net source term: P - D + cb2_term
    """
    # 1. Vorticity magnitude from velocity gradients
    omega = compute_vorticity_jax(grad)
    
    # 2. SA production & destruction
    P = compute_sa_production(omega, nuHat, wall_dist)
    D = compute_sa_destruction(omega, nuHat, wall_dist)
    
    # 3. cb2 gradient term: (cb2/σ)(∇ν̃)·(∇ν̃)
    grad_nuHat = grad[:, :, 3, :]  # (NI, NJ, 2)
    grad_nuHat_sq = jnp.sum(grad_nuHat**2, axis=-1)
    cb2_term = (cb2 / sigma) * grad_nuHat_sq
    
    return P - D + cb2_term
```

### Phase 2: Extend Viscous Fluxes

**Modify `src/numerics/viscous_fluxes.py`:**

The existing `compute_viscous_fluxes_jax` only handles momentum (indices 1, 2). Extend to include SA diffusion on index 3:

```python
@jax.jit
def compute_viscous_fluxes_with_sa_jax(grad, Si_x, Si_y, Sj_x, Sj_y, 
                                        mu_eff, nu_laminar, nuHat, sigma=2.0/3.0):
    """
    Viscous fluxes including SA diffusion.
    
    Adds (1/σ)∇·[(ν + ν̃)∇ν̃] to residual index 3.
    """
    NI, NJ = mu_eff.shape
    
    # ... existing momentum viscous flux code for indices 1, 2 ...
    
    # SA diffusion coefficient: (ν + ν̃)/σ
    nu_eff_sa = (nu_laminar + jnp.maximum(nuHat, 0.0)) / sigma
    
    # Compute SA diffusion using existing infrastructure
    sa_diff = compute_nu_tilde_diffusion_jax(
        grad[:, :, 3, :], Si_x, Si_y, Sj_x, Sj_y, nu_eff_sa, sigma=1.0
    )
    
    residual = residual.at[:, :, 3].set(sa_diff)
    
    return residual
```

### Phase 3: RANS Solver Modification

**Modify `src/solvers/rans_solver.py`:**

#### 3.1 Add Wall Distance to JAX Arrays

In `_initialize_jax()`:

```python
def _initialize_jax(self):
    # ... existing code ...
    
    # Transfer wall distance to GPU (static, computed once)
    self.wall_dist_jax = jax.device_put(jnp.array(self.metrics.wall_distance))
```

#### 3.2 Modify `step()` Method

```python
def step(self) -> Tuple[float, np.ndarray]:
    nghost = NGHOST
    cfl = self._get_cfl(self.iteration)
    k4 = self.config.jst_k4
    beta = self.config.beta
    nu = self.mu_laminar
    
    # ... existing timestep computation ...
    
    for alpha in alphas:
        Q_np = self.bc.apply(Q_np)
        Q_jax = jax.device_put(jnp.array(Q_np))
        
        # Convective flux (unchanged)
        R = compute_fluxes_jax(
            Q_jax, self.Si_x_jax, self.Si_y_jax,
            self.Sj_x_jax, self.Sj_y_jax, beta, k4, nghost
        )
        
        # Gradients (unchanged)
        grad = compute_gradients_jax(
            Q_jax, self.Si_x_jax, self.Si_y_jax,
            self.Sj_x_jax, self.Sj_y_jax, self.volume_jax, nghost
        )
        
        # Turbulent viscosity from SA variable
        Q_int = Q_jax[nghost:-nghost, nghost:-nghost, :]
        nu_tilde = jnp.maximum(Q_int[:, :, 3], 0.0)
        chi = nu_tilde / nu
        cv1 = 7.1
        fv1_val = chi**3 / (chi**3 + cv1**3)
        mu_t = nu_tilde * fv1_val
        mu_eff = nu + mu_t
        
        # Viscous fluxes (extended for SA diffusion)
        R_visc = compute_viscous_fluxes_with_sa_jax(
            grad, self.Si_x_jax, self.Si_y_jax,
            self.Sj_x_jax, self.Sj_y_jax,
            mu_eff, nu, nu_tilde
        )
        R = R + R_visc
        
        # NEW: SA source terms (P - D + cb2)
        sa_source = compute_sa_source_jax(
            nu_tilde, grad, self.wall_dist_jax, nu
        )
        # Scale by volume for FVM formulation
        R = R.at[:, :, 3].add(sa_source * self.volume_jax)
        
        # Smoothing (unchanged)
        if epsilon > 0 and n_passes > 0:
            R = smooth_explicit_jax(R, epsilon, n_passes)
        
        # Update state (unchanged)
        R_np = np.array(R)
        dt_np = np.array(dt)
        Q_np = Q0_np.copy()
        Q_np[nghost:-nghost, nghost:-nghost, :] += \
            alpha * (dt_np / self.metrics.volume)[:, :, np.newaxis] * R_np
    
    # ... rest unchanged ...
```

### Phase 4: Boundary Conditions

The existing BCs already handle `ν̃` correctly:

#### Wall BC (in `apply_surface_bc_jax`)
- `ν̃ = 0` at wall (reflection gives zero at wall face)

#### Farfield BC (in `apply_farfield_bc_jax`)
- **Inflow**: `ν̃ = ν̃_∞` (small freestream value, ~0.001·ν)
- **Outflow**: Zero-gradient extrapolation

No modifications needed.

### Phase 5: AFT-SA Transition Model (Optional)

For transition prediction, blend AFT amplification with SA:

```python
@jax.jit
def compute_aft_sa_source_jax(nuHat, grad, wall_dist, nu_laminar):
    """
    Blended AFT-SA source for transition prediction.
    
    - Low nuHat: AFT amplification dominates (laminar growth)
    - High nuHat: SA production/destruction dominates (turbulent)
    """
    omega = compute_vorticity_jax(grad)
    
    # SA terms
    P_sa = compute_sa_production(omega, nuHat, wall_dist)
    D_sa = compute_sa_destruction(omega, nuHat, wall_dist)
    
    # AFT amplification (laminar regime)
    # Shape factor Gamma approximation from local gradients
    u = grad[:, :, 1, 0]  # Approximate u from dudx context
    dudy = grad[:, :, 1, 1]
    Gamma = jnp.abs(dudy) * wall_dist / (jnp.abs(u) + 1e-10)
    Gamma = jnp.clip(Gamma, 0.0, 2.0)  # Physical bounds
    
    Re_omega = wall_dist**2 * omega
    a_aft = compute_nondimensional_amplification_rate(Re_omega, Gamma) * omega
    
    # Smooth blending based on nuHat level
    # is_turb ≈ 0 for nuHat << 1 (laminar)
    # is_turb ≈ 1 for nuHat >> 1 (turbulent)
    is_turb = jnp.clip(1 - jnp.exp(-(nuHat - 1) / 4), min=0.0)
    
    P_blended = P_sa * is_turb + a_aft * nuHat * (1 - is_turb)
    
    return P_blended - D_sa
```

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/numerics/sa_sources.py` | **Create** | SA source term computation (P, D, cb2) |
| `src/numerics/viscous_fluxes.py` | **Modify** | Add SA diffusion to residual index 3 |
| `src/solvers/rans_solver.py` | **Modify** | Call SA sources in `step()`, add `wall_dist_jax` |
| `tests/numerics/test_sa_sources.py` | **Create** | Unit tests for SA source terms |
| `tests/solver/test_sa_rans.py` | **Create** | Integration test (flat plate transition) |

---

## JAX/GPU Considerations

Since another agent is converting the entire solver to JAX:

1. **Wall Distance**: Static array, transfer once at initialization with `jax.device_put`

2. **SA Sources**: Point-local computation → naturally parallelizes on GPU

3. **Gradient Reuse**: Already computed for viscous fluxes → pass to SA source computation

4. **Memory**: SA source terms are computed in-place, no additional large arrays

5. **Full GPU Loop Structure**:
   ```python
   @functools.partial(jax.jit, static_argnums=(8, 9, 10))
   def rk_stage_with_sa(Q, metrics, wall_dist, beta, k4, nu, cfl, alpha,
                        nghost, n_wake, epsilon, n_passes):
       """Single RK stage with SA turbulence - fully on GPU."""
       Q = apply_bc_jax(Q, ...)
       R = compute_fluxes_jax(Q, ...)
       grad = compute_gradients_jax(Q, ...)
       R_visc = compute_viscous_fluxes_with_sa_jax(grad, ...)
       R = R + R_visc
       sa_src = compute_sa_source_jax(...)
       R = R.at[:, :, 3].add(sa_src * volume)
       R = smooth_explicit_jax(R, ...)
       dt = compute_local_timestep_jax(...)
       Q_new = Q + alpha * dt/volume * R
       return Q_new
   ```

6. **Convergence Monitoring**: Only return residual norm (scalar) to CPU

---

## Validation Strategy

### Unit Tests

1. **SA Source Terms**: Compare `compute_sa_source_jax` output against boundary layer solver
2. **cb2 Term**: Verify gradient-squared computation
3. **Diffusion**: Check conservation properties

### Integration Tests

1. **Flat Plate**: 
   - Start with laminar flow (low `ν̃_∞`)
   - Verify Cf follows Blasius initially
   - Observe transition to turbulent Cf

2. **NACA 0012 at Re=6M**:
   - Compare Cp distribution to experiments
   - Check Cf levels (laminar vs turbulent regions)
   - Verify force coefficients (CL, CD)

### Convergence Checks

- Monitor `ν̃` residual alongside pressure residual
- Check for negative `ν̃` (should be handled by safe functions)
- Verify `μ_t/μ` ratio is physical (typically 10-1000 in turbulent regions)

---

## Implementation Order

1. **Phase 1**: Create `sa_sources.py` with unit tests
2. **Phase 2**: Extend `viscous_fluxes.py` for SA diffusion
3. **Phase 3**: Modify `rans_solver.py` to call SA sources
4. **Phase 4**: Run flat plate validation
5. **Phase 5**: Add AFT blending (optional, for transition)
6. **Phase 6**: Full airfoil validation (NACA 0012)

---

## References

- Spalart, P. R., & Allmaras, S. R. (1992). "A one-equation turbulence model for aerodynamic flows." AIAA Paper 92-0439.
- Drela, M., & Giles, M. B. (1987). "Viscous-inviscid analysis of transonic and low Reynolds number airfoils." AIAA Journal, 25(10), 1347-1355.
