# SA Turbulence Model Integration Plan

This document outlines the plan to integrate the Spalart-Allmaras (SA) turbulence equation into the RANS solver, including the AFT (Amplification Factor Transport) transition model.

## Current State (January 2025)

### RANS Solver Architecture

The solver (`src/solvers/rans_solver.py`) is fully GPU-accelerated using JAX:

- **State vector**: `Q = [p, u, v, ν̃]` (4 variables)
- **Full GPU execution**: JIT-compiled RK5 stages, no CPU transfers during iteration
- **Batch support**: `BatchRANSSolver` in `src/solvers/batch.py` for parallel AoA sweeps

**Current turbulence handling:**
- Computes `μ_t = ν̃ · fv1(ν̃)` for effective viscosity
- Convects `ν̃` through the JST flux scheme
- Applies boundary conditions for `ν̃`

**What is NOT yet implemented:**
- SA production term: `P = cb1 · S̃ · ν̃`
- SA destruction term: `D = cw1 · fw · (ν̃/d)²`
- SA diffusion with cb2 term: `(cb2/σ)(∇ν̃)·(∇ν̃)`

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

### Grid Metrics & Wall Distance

The `MetricComputer` in `src/grid/metrics.py` computes:
- Face normals and areas (Si, Sj)
- Cell volumes
- **Wall distance** with proper wake exclusion via `n_wake` parameter

Wall distance correctly identifies only the airfoil surface (excluding C-grid wake cut):
```python
# Airfoil surface spans i = n_wake to NI - n_wake at j = 0
metrics = MetricComputer(X, Y, n_wake=config.n_wake)
```

### Boundary Conditions

Current BC implementation (`src/solvers/boundary_conditions.py`):

| Boundary | Treatment |
|----------|-----------|
| **Wall** | No-slip (u=v=0), zero pressure gradient, ν̃=0 |
| **Wake cut** | Periodic/averaging for C-grid topology |
| **Farfield** | Dirichlet (ghost = freestream) + sponge layer damping |

The sponge layer (`src/numerics/dissipation.py`) gradually damps solution to freestream near outer boundaries, improving stability.

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

### Phase 3: RANS Solver Integration

**Modify the JIT-compiled RK stage in `src/solvers/rans_solver.py`:**

The current `_make_rk_stage_jit()` returns a JIT-compiled function. Add SA sources:

```python
def _make_rk_stage_jit(self):
    # Capture metrics, wall distance, etc.
    Si_x, Si_y = self.Si_x_jax, self.Si_y_jax
    Sj_x, Sj_y = self.Sj_x_jax, self.Sj_y_jax
    volume = self.volume_jax
    wall_dist = self.metrics.wall_distance_jax  # Pre-transferred to GPU
    nu = self.mu_laminar
    
    @jax.jit
    def rk_stage(Q, Q0, dt, alpha):
        # Apply BCs
        Q = apply_bc_jax(Q, ...)
        
        # Convective fluxes (JST)
        R = compute_fluxes_jax(Q, Si_x, Si_y, Sj_x, Sj_y, ...)
        
        # Gradients
        grad = compute_gradients_jax(Q, Si_x, Si_y, Sj_x, Sj_y, volume, ...)
        
        # Turbulent viscosity
        nuHat = jnp.maximum(Q[..., 3], 0.0)
        chi = nuHat / nu
        fv1_val = chi**3 / (chi**3 + 7.1**3)
        mu_eff = nu + nuHat * fv1_val
        
        # Viscous fluxes (extended for SA)
        R_visc = compute_viscous_fluxes_with_sa_jax(grad, ..., mu_eff, nu, nuHat)
        R = R + R_visc
        
        # SA source terms (P - D + cb2)
        sa_source = compute_sa_source_jax(nuHat, grad, wall_dist, nu)
        R = R.at[..., 3].add(sa_source * volume)
        
        # Explicit smoothing
        R = smooth_explicit_jax(R, ...)
        
        # RK update
        Q_new = Q0 + alpha * (dt / volume)[..., None] * R
        return Q_new, R
    
    return rk_stage
```

### Phase 4: Batch Solver Extension

Extend `BatchRANSSolver` similarly - the `vmap` structure naturally handles per-case SA sources since wall distance is shared.

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
    u = grad[:, :, 1, 0]
    dudy = grad[:, :, 1, 1]
    Gamma = jnp.abs(dudy) * wall_dist / (jnp.abs(u) + 1e-10)
    Gamma = jnp.clip(Gamma, 0.0, 2.0)
    
    Re_omega = wall_dist**2 * omega
    a_aft = compute_nondimensional_amplification_rate(Re_omega, Gamma) * omega
    
    # Smooth blending based on nuHat level
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
| `src/solvers/rans_solver.py` | **Modify** | Add SA sources to RK stage JIT function |
| `src/solvers/batch.py` | **Modify** | Add SA sources to batch kernels |
| `tests/numerics/test_sa_sources.py` | **Create** | Unit tests for SA source terms |
| `tests/solver/test_sa_rans.py` | **Create** | Integration test (flat plate transition) |

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

### Batch Validation

Use `BatchRANSSolver` for efficient parameter studies:
```bash
python scripts/solver/run_batch.py data/naca0012.dat --alpha-sweep -5 15 21 --re 6e6
```

Output: `output/batch/batch_results.csv` with CL, CD vs α

---

## References

- Spalart, P. R., & Allmaras, S. R. (1992). "A one-equation turbulence model for aerodynamic flows." AIAA Paper 92-0439.
- Drela, M., & Giles, M. B. (1987). "Viscous-inviscid analysis of transonic and low Reynolds number airfoils." AIAA Journal, 25(10), 1347-1355.
