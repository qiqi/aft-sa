# Jacobian-Free Newton-Krylov (JFNK) Implementation Plan

> **Status: ✅ IMPLEMENTED** (2026-01-05)
>
> All five phases have been completed. The solver now supports three modes:
> - `rk5`: Standard explicit RK5 (default)
> - `rk5_precond`: RK5 with block-Jacobi preconditioned residual
> - `newton`: Newton-GMRES with block-Jacobi preconditioning
>
> See `config/examples/naca0012_newton.yaml` for example usage.

## Overview

This document outlines the implementation plan for adding a Jacobian-Free Newton-Krylov solver with GMRES(m) and block-Jacobi preconditioning to the AFT-SA RANS solver.

**Goal:** Improve convergence rate compared to explicit RK5 time-stepping, especially for stiff problems (high Re, complex geometries).

**Key Design Decisions:**
- Use JAX's native forward-mode AD (`jax.jvp`) for exact Jacobian-vector products
- Block-Jacobi preconditioner (4×4 blocks per cell) computed via `jax.jacfwd`
- GMRES(m) with m=20 (configurable), all operations on GPU
- Retain Patankar scheme for nuHat updates (stability for stiff SA destruction term)
- Pseudo-transient continuation via CFL ramping (exponential for Newton, linear for RK)

---

## Solver Modes (End State)

After implementation, three solver modes will be available:

| Mode | Time Stepping | Preconditioner | Residual Smoothing | CFL Ramping |
|------|--------------|----------------|-------------------|-------------|
| `rk5` | Explicit RK5 | None | Applied after R(Q) | Linear to target |
| `rk5_precond` | Explicit RK5 | Block-Jacobi | Precondition → Smooth | Linear to target |
| `newton` | Implicit Newton-GMRES | Block-Jacobi | Part of R(Q) for Jacobian | Exponential to ∞ |

**Residual smoothing order for RK:** Precondition first, then smooth.

**Residual smoothing for Newton:** Must be included in the residual function R(Q) so that the Jacobian J = ∂R/∂Q includes the smoothing operator.

---

## Pseudo-Transient Continuation (CFL Strategy)

The CFL number controls the diagonal regularization of the implicit system:

**System being solved at each Newton iteration:**
```
(V/(dt) * I + J) @ dQ = -R
```

where `dt = CFL * V / λ` (local timestep from CFL condition).

This is equivalent to:
```
(λ/CFL * I + J) @ dQ = -R
```

**CFL ramping strategies:**

| Method | CFL Behavior | Rationale |
|--------|------------|-----------|
| RK5 | Linear: `CFL(n) = CFL_start + n/N * (CFL_target - CFL_start)` | Explicit stability limit |
| Newton | Exponential: `CFL(n) = CFL_start * growth^n` | Pseudo-transient → steady state |

For Newton, as CFL → ∞, the diagonal vanishes and we recover pure Newton's method for steady state.

---

## Configuration Structure

```yaml
flow:
  alpha: 4.0
  reynolds: 1.0e6

solver:
  method: "newton"  # "rk5", "rk5_precond", or "newton"
  max_iter: 1000
  tol: 1.0e-10
  
  # Residual smoothing (applied to all methods)
  smoothing:
    epsilon: 0.2
    passes: 2
  
  # CFL strategy (method-specific defaults)
  cfl:
    # For RK methods:
    rk:
      start: 0.5
      target: 5.0
      ramp_iters: 500  # Linear ramp over this many iterations
    
    # For Newton method:
    newton:
      start: 1.0
      growth: 1.5      # Exponential growth factor per iteration
      max: 1.0e12      # Cap to avoid overflow (effectively ∞)
  
  # GMRES settings (Newton only)
  gmres:
    m: 20              # Krylov subspace dimension
    tol: 1.0e-4        # Relative tolerance for linear solve
    max_restarts: 3    # Maximum GMRES restarts per Newton step
  
  # Preconditioner settings (rk5_precond and newton)
  preconditioner:
    type: "block_jacobi"  # Currently only option
```

---

## Phase 1: Block-Jacobi Preconditioner

**Goal:** Create a 4×4 block-diagonal preconditioner that approximates the local Jacobian at each cell.

### Files to Create

```
src/numerics/preconditioner.py
```

### API Design

```python
@dataclass
class BlockJacobiPreconditioner:
    """Block-Jacobi preconditioner with 4x4 blocks per cell."""
    
    P_inv: jnp.ndarray  # (NI, NJ, 4, 4) - inverted blocks
    
    @staticmethod
    def compute(residual_fn, Q, dt, volume) -> 'BlockJacobiPreconditioner':
        """Compute block-Jacobi preconditioner at current state.
        
        The diagonal block approximates:
            P = V/dt * I + ∂R_i/∂Q_i
        
        For implicit Euler, P approximates the system matrix.
        
        Parameters
        ----------
        residual_fn : callable
            R(Q) -> (NI, NJ, 4) residual function (includes smoothing for Newton)
        Q : jnp.ndarray
            Current state (NI+2*nghost, NJ+2*nghost, 4)
        dt : jnp.ndarray
            Local timestep (NI, NJ) from CFL condition
        volume : jnp.ndarray
            Cell volumes (NI, NJ)
            
        Returns
        -------
        BlockJacobiPreconditioner
            Preconditioner with inverted blocks
        """
        pass
    
    def apply(self, v: jnp.ndarray) -> jnp.ndarray:
        """Apply P^{-1} to vector v.
        
        Parameters
        ----------
        v : jnp.ndarray
            Input vector (NI, NJ, 4) or flattened
            
        Returns
        -------
        jnp.ndarray
            P^{-1} @ v, same shape as input
        """
        pass
```

### Implementation Details

1. **Extract cell-local Jacobian using JAX autodiff:**
   ```python
   def cell_jacobian(Q_cell, i, j):
       """Get 4x4 Jacobian ∂R[i,j]/∂Q[i,j] via forward-mode AD."""
       def single_cell_residual(q):
           # Inject q into Q at (i,j), compute R, extract R[i,j]
           ...
       return jax.jacfwd(single_cell_residual)(Q_cell)
   ```

2. **Batch computation over all cells:**
   ```python
   # Vectorize over (i,j) using vmap
   all_jacobians = jax.vmap(jax.vmap(cell_jacobian))(Q_interior)  # (NI, NJ, 4, 4)
   ```

3. **Form and invert preconditioner blocks:**
   ```python
   # P = V/dt * I + J_diag  (implicit Euler system matrix)
   diag_term = (volume / dt)[:, :, None, None] * jnp.eye(4)
   P = diag_term + all_jacobians
   P_inv = jax.vmap(jax.vmap(jnp.linalg.inv))(P)  # Batched 4x4 inversion
   ```

4. **Apply preconditioner:**
   ```python
   def apply(self, v):
       v_reshaped = v.reshape(NI, NJ, 4)
       # Batched matrix-vector: P_inv[i,j] @ v[i,j]
       result = jnp.einsum('ijkl,ijl->ijk', self.P_inv, v_reshaped)
       return result.reshape(v.shape)
   ```

### Testing

- [ ] Unit test: `P_inv @ P ≈ I` for each block
- [ ] Unit test: Batched inversion handles well-conditioned 4×4 matrices
- [ ] Integration test: Preconditioner reduces condition number estimate

---

## Phase 2: Preconditioned RK (Validation)

**Goal:** Test the preconditioner by using it to accelerate the existing RK5 scheme.

### Files to Modify

```
src/solvers/rans_solver.py
src/config/schema.py
```

### Changes to RK Stage

The preconditioned RK stage applies: **precondition → smooth**

```python
@jax.jit
def jit_rk_stage_precond(Q, Q0, dt, alpha_rk, P_inv):
    """Single RK stage with preconditioning."""
    Q = apply_bc(Q)
    
    # Compute raw residual (no smoothing yet)
    R = compute_residual_raw(Q)
    
    # 1. Apply preconditioner
    R_precond = apply_block_jacobi(P_inv, R)
    
    # 2. Apply smoothing AFTER preconditioning
    if smoothing_epsilon > 0:
        R_precond = smooth_explicit_jax(R_precond, smoothing_epsilon, smoothing_passes)
    
    # Update with preconditioned+smoothed residual
    dQ = alpha_rk * (dt * volume_inv)[:, :, None] * R_precond
    
    # ... Patankar for nuHat, same as before ...
    
    return Q_new, R_precond
```

### Preconditioner Update

For preconditioned RK, the preconditioner is computed based on the **raw residual** (without smoothing), since smoothing is applied afterward.

```python
# In run_steady_state():
if self.config.method == "rk5_precond":
    # Recompute preconditioner periodically (every 50 iters is fine for RK)
    if iteration % 50 == 0:
        self.preconditioner = BlockJacobiPreconditioner.compute(
            residual_fn=self.compute_residual_raw,  # Raw, no smoothing
            Q=self.Q_jax,
            dt=self.dt_local,
            volume=self.volume_jax
        )
```

### Testing

- [ ] Compare convergence: preconditioned RK vs unpreconditioned RK
- [ ] Verify no regression in solution accuracy
- [ ] Profile: preconditioner computation cost vs speedup

---

## Phase 3: GMRES(m) Implementation

**Goal:** Implement restarted GMRES with JAX, keeping all operations on GPU.

### Files to Create

```
src/numerics/gmres.py
```

### API Design

```python
@dataclass
class GMRESResult:
    """Result of GMRES solve."""
    x: jnp.ndarray        # Solution
    residual_norm: float  # Final residual norm
    iterations: int       # Total Krylov iterations used
    converged: bool       # Whether tolerance was achieved


def gmres_m(
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    b: jnp.ndarray,
    x0: jnp.ndarray = None,
    precond: Callable[[jnp.ndarray], jnp.ndarray] = None,
    m: int = 20,
    tol: float = 1e-4,
    max_restarts: int = 3,
) -> GMRESResult:
    """
    Solve A @ x = b using restarted GMRES(m).
    
    Parameters
    ----------
    matvec : callable
        Function computing A @ v for a vector v
    b : jnp.ndarray
        Right-hand side vector (flattened)
    x0 : jnp.ndarray, optional
        Initial guess (default: zeros)
    precond : callable, optional
        Right preconditioner P^{-1}(v). Solves (A @ P^{-1}) @ (P @ x) = b
    m : int
        Krylov subspace dimension (restart after m iterations). Default: 20
    tol : float
        Convergence tolerance on relative residual ||r|| / ||b||
    max_restarts : int
        Maximum number of GMRES restarts
        
    Returns
    -------
    GMRESResult
        Solution and convergence information
    """
    pass
```

### Implementation Details

1. **Arnoldi iteration with modified Gram-Schmidt:**
   ```python
   @jax.jit
   def arnoldi_iteration(matvec, r0, m):
       """Build orthonormal Krylov basis V and upper Hessenberg H."""
       n = r0.shape[0]
       V = jnp.zeros((n, m + 1))
       H = jnp.zeros((m + 1, m))
       
       beta = jnp.linalg.norm(r0)
       V = V.at[:, 0].set(r0 / beta)
       
       def arnoldi_step(carry, j):
           V, H = carry
           w = matvec(V[:, j])
           
           # Modified Gram-Schmidt
           def orthogonalize(carry, i):
               w, H = carry
               h_ij = jnp.dot(V[:, i], w)
               w = w - h_ij * V[:, i]
               H = H.at[i, j].set(h_ij)
               return (w, H), None
           
           (w, H), _ = jax.lax.scan(orthogonalize, (w, H), jnp.arange(j + 1))
           
           h_jp1_j = jnp.linalg.norm(w)
           H = H.at[j + 1, j].set(h_jp1_j)
           V = V.at[:, j + 1].set(w / (h_jp1_j + 1e-30))
           
           return (V, H), None
       
       (V, H), _ = jax.lax.scan(arnoldi_step, (V, H), jnp.arange(m))
       return V, H, beta
   ```

2. **Solve least squares via Givens rotations:**
   ```python
   @jax.jit
   def solve_hessenberg_ls(H, beta, m):
       """Solve min ||beta*e1 - H*y|| using Givens rotations."""
       # Apply Givens rotations to reduce H to upper triangular
       # Then back-substitute to get y
       ...
   ```

3. **Main GMRES loop with restarts:**
   ```python
   def gmres_m(matvec, b, x0, precond, m, tol, max_restarts):
       x = x0 if x0 is not None else jnp.zeros_like(b)
       b_norm = jnp.linalg.norm(b)
       
       for restart in range(max_restarts):
           r = b - matvec(x)
           r_norm = jnp.linalg.norm(r)
           
           if r_norm / b_norm < tol:
               return GMRESResult(x, r_norm, restart * m, True)
           
           # Apply right preconditioning: solve (A @ P^{-1}) @ z = r
           if precond is not None:
               matvec_precond = lambda v: matvec(precond(v))
           else:
               matvec_precond = matvec
           
           V, H, beta = arnoldi_iteration(matvec_precond, r, m)
           y = solve_hessenberg_ls(H, beta, m)
           
           # Update solution
           z = V[:, :m] @ y
           if precond is not None:
               z = precond(z)
           x = x + z
       
       return GMRESResult(x, r_norm, max_restarts * m, False)
   ```

### Testing

- [ ] Unit test: Solve small dense system, compare with `jnp.linalg.solve`
- [ ] Unit test: Converges on well-conditioned sparse system
- [ ] Unit test: Preconditioning reduces iteration count
- [ ] Integration test: Works with JVP-based matvec

---

## Phase 4: Newton-GMRES Solver

**Goal:** Wrap GMRES in a Newton iteration with pseudo-transient continuation and Patankar update for nuHat.

### Files to Create

```
src/solvers/newton_solver.py
```

### Key Concepts

**Pseudo-transient continuation:**
At each Newton iteration, we solve:
```
(V/dt * I + J) @ dQ = -R
```
where `dt` comes from the current CFL. As CFL → ∞, `dt → ∞`, and this becomes pure Newton.

**Residual function for Newton:**
Must include smoothing so that `J = ∂R_smoothed/∂Q`:
```python
def residual_with_smoothing(Q):
    R = compute_residual_raw(Q)
    R = smooth_explicit_jax(R, epsilon, passes)
    return R
```

**Preconditioner recomputation:**
Every Newton iteration (it's cheap, and the Jacobian changes).

### API Design

```python
@dataclass
class NewtonConfig:
    """Configuration for Newton-GMRES solver."""
    gmres_m: int = 20
    gmres_tol: float = 1e-4
    gmres_max_restarts: int = 3
    tol: float = 1e-10          # Newton convergence tolerance
    max_iters: int = 100        # Max Newton iterations
    cfl_start: float = 1.0      # Initial CFL
    cfl_growth: float = 1.5     # Exponential growth factor
    cfl_max: float = 1e12       # Cap (effectively infinity)


class NewtonGMRESSolver:
    """Newton-GMRES solver for steady-state RANS with pseudo-transient continuation."""
    
    def __init__(self, config: NewtonConfig, residual_fn, ...):
        """
        Parameters
        ----------
        residual_fn : callable
            R(Q) including smoothing, i.e., R_smoothed(Q)
        """
        pass
    
    def solve(self, Q0: jnp.ndarray) -> Tuple[jnp.ndarray, bool]:
        """Solve R(Q) = 0 starting from Q0."""
        pass
```

### Implementation Details

1. **Newton iteration with pseudo-transient continuation:**
   ```python
   def solve(self, Q0):
       Q = Q0
       cfl = self.config.cfl_start
       
       for n in range(self.config.max_iters):
           # Compute residual (includes smoothing)
           R = self.residual_fn(Q)
           R_norm = jnp.linalg.norm(R)
           
           if R_norm < self.config.tol:
               return Q, True  # Converged
           
           # Compute local timestep from current CFL
           dt = self.compute_local_dt(Q, cfl)
           
           # Recompute preconditioner (every iteration)
           self.preconditioner = BlockJacobiPreconditioner.compute(
               self.residual_fn, Q, dt, self.volume
           )
           
           # Form the system matrix operator: (V/dt * I + J) @ v
           def system_matvec(v):
               # Diagonal term: V/dt * v
               v_shaped = v.reshape(NI, NJ, 4)
               diag_v = (self.volume / dt)[:, :, None] * v_shaped
               
               # Jacobian term: J @ v via JVP
               _, Jv = jax.jvp(
                   lambda q: self.residual_fn(q).flatten(),
                   (Q,), (v.reshape(Q.shape),)
               )
               
               return diag_v.flatten() + Jv
           
           # Solve (V/dt * I + J) @ dQ = -R using GMRES
           result = gmres_m(
               matvec=system_matvec,
               b=-R.flatten(),
               precond=self.preconditioner.apply,
               m=self.config.gmres_m,
               tol=self.config.gmres_tol,
               max_restarts=self.config.gmres_max_restarts,
           )
           
           dQ = result.x.reshape(Q.shape)
           
           # Apply update with Patankar for nuHat
           Q = self.apply_update_with_patankar(Q, dQ)
           
           # Exponential CFL growth
           cfl = min(cfl * self.config.cfl_growth, self.config.cfl_max)
           
           # Print progress
           print(f"Newton {n}: ||R|| = {R_norm:.6e}, CFL = {cfl:.2e}")
       
       return Q, False  # Did not converge
   ```

2. **Patankar update for nuHat:**
   ```python
   def apply_update_with_patankar(self, Q, dQ):
       """Apply Newton update with Patankar scheme for nuHat."""
       Q_new = Q.at[..., :3].add(dQ[..., :3])  # Normal update for p, u, v
       
       # Patankar update for nuHat (index 3)
       nuHat_old = Q[..., 3]
       dNuHat = dQ[..., 3]
       
       # Patankar: nuHat_new = nuHat_old / (1 - dNuHat/nuHat_old)
       # Only when nuHat_old > 0 and dNuHat < 0 (destruction-dominated)
       nuHat_patankar = jnp.where(
           nuHat_old > 0,
           nuHat_old / (1.0 - dNuHat / nuHat_old),
           nuHat_old + dNuHat
       )
       nuHat_new = jnp.where(dNuHat < 0, nuHat_patankar, nuHat_old + dNuHat)
       nuHat_new = jnp.maximum(nuHat_new, 0.0)
       
       return Q_new.at[..., 3].set(nuHat_new)
   ```

### Testing

#### Implicit Euler System Tests

The core of the Newton solver is the implicit Euler system:
```
(V/dt · I + J) · dQ = -R
```

These tests verify the system is correctly formed and behaves as expected:

- [ ] **Unit test: Diagonal term formation**
  - Verify `(V/dt · I) @ v` is correctly computed for test vector v
  - Check dimensions: V is (NI, NJ), dt is (NI, NJ), result is (NI, NJ, 4)

- [ ] **Unit test: System matvec correctness**
  - For a simple residual R(Q) = A @ Q - b (linear), verify:
    - `system_matvec(v) = (V/dt · I + A) @ v`
  - Compare against explicit matrix construction for small test case

- [ ] **Unit test: CFL limiting behavior**
  - CFL → 0 (dt → 0): System becomes `(V/dt · I) @ dQ ≈ -R`, i.e., `dQ ≈ -dt/V · R`
    - This is explicit Euler, should give small stable steps
  - CFL → ∞ (dt → ∞): System becomes `J @ dQ = -R`
    - This is pure Newton, should give quadratic convergence near solution

- [ ] **Unit test: Exponential CFL ramping**
  - Verify `CFL(n) = CFL_start * growth^n` until `CFL_max`
  - Test that CFL is capped correctly at `CFL_max`

- [ ] **Unit test: JVP correctness for Jacobian-vector product**
  - For known residual function, compare `jax.jvp` result against finite difference:
    - `J @ v ≈ (R(Q + ε·v) - R(Q)) / ε` for small ε
  - Verify JVP is exact (not finite difference approximation)

#### Newton Solver Tests

- [ ] **Unit test: Newton converges quadratically on simple nonlinear system**
  - Use R(Q) = Q² - c, verify ||R|| decreases quadratically near solution

- [ ] **Unit test: Pseudo-transient helps convergence from bad initial guess**
  - Start far from solution, verify low CFL allows progress
  - Compare: pure Newton (CFL=∞) may diverge, pseudo-transient converges

- [ ] **Unit test: Patankar update preserves nuHat >= 0**
  - Test with negative dNuHat, verify nuHat_new >= 0
  - Test with nuHat_old = 0, verify no division by zero

- [ ] **Unit test: Preconditioner is recomputed every Newton iteration**
  - Mock preconditioner.compute(), verify it's called each iteration

#### Integration Tests

- [ ] **Integration test: Converges on NACA 0012 case**
  - Start from freestream initialization
  - Verify final residual < tolerance

- [ ] **Comparison test: Newton vs RK5**
  - Count total residual evaluations (matvecs + explicit R calls)
  - Newton should require fewer total evaluations for same final residual

---

## Phase 5: Integration & Configuration

**Goal:** Integrate all solver modes and update configuration schema.

### Files to Modify

```
src/config/schema.py
src/solvers/rans_solver.py
config/examples/*.yaml
```

### Updated Configuration Schema

```python
@dataclass
class CFLConfigRK:
    """CFL configuration for explicit RK methods."""
    start: float = 0.5
    target: float = 5.0
    ramp_iters: int = 500  # Linear ramp

@dataclass
class CFLConfigNewton:
    """CFL configuration for Newton (pseudo-transient continuation)."""
    start: float = 1.0
    growth: float = 1.5    # Exponential growth factor
    max: float = 1.0e12    # Cap

@dataclass
class GMRESConfig:
    """GMRES settings for Newton solver."""
    m: int = 20
    tol: float = 1e-4
    max_restarts: int = 3

@dataclass
class SmoothingConfig:
    """Residual smoothing settings."""
    epsilon: float = 0.2
    passes: int = 2

@dataclass
class SolverConfig:
    """Main solver configuration."""
    method: str = "rk5"  # "rk5", "rk5_precond", or "newton"
    max_iter: int = 10000
    tol: float = 1e-10
    
    # Sub-configs
    cfl_rk: CFLConfigRK = field(default_factory=CFLConfigRK)
    cfl_newton: CFLConfigNewton = field(default_factory=CFLConfigNewton)
    gmres: GMRESConfig = field(default_factory=GMRESConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
```

### Example YAML Configurations

**Standard RK5 (current behavior):**
```yaml
solver:
  method: "rk5"
  max_iter: 10000
  tol: 1.0e-10
  cfl_rk:
    start: 0.5
    target: 5.0
    ramp_iters: 500
  smoothing:
    epsilon: 0.2
    passes: 2
```

**Preconditioned RK5:**
```yaml
solver:
  method: "rk5_precond"
  max_iter: 10000
  tol: 1.0e-10
  cfl_rk:
    start: 0.5
    target: 10.0    # Can be more aggressive with preconditioning
    ramp_iters: 300
  smoothing:
    epsilon: 0.2
    passes: 2
```

**Newton-GMRES:**
```yaml
solver:
  method: "newton"
  max_iter: 100     # Newton iterations, not RK steps
  tol: 1.0e-10
  cfl_newton:
    start: 1.0
    growth: 1.5
    max: 1.0e12
  gmres:
    m: 20
    tol: 1.0e-4
    max_restarts: 3
  smoothing:
    epsilon: 0.2
    passes: 2
```

### Solver Dispatch in RANSSolver

```python
def run_steady_state(self):
    if self.config.method == "newton":
        return self._run_newton()
    elif self.config.method == "rk5_precond":
        return self._run_rk5_preconditioned()
    else:
        return self._run_rk5()  # Current implementation
```

### Testing

- [ ] All three modes work on NACA 0012 test case
- [ ] YAML configuration correctly selects mode
- [ ] Backward compatibility: existing configs still work with default RK5

---

## Summary: File Changes

| File | Action | Description |
|------|--------|-------------|
| `src/numerics/preconditioner.py` | **CREATE** | Block-Jacobi preconditioner |
| `src/numerics/gmres.py` | **CREATE** | GMRES(m) solver |
| `src/solvers/newton_solver.py` | **CREATE** | Newton-GMRES with pseudo-transient |
| `src/solvers/rans_solver.py` | MODIFY | Add rk5_precond mode, Newton integration |
| `src/config/schema.py` | MODIFY | New config structure with CFL sub-fields |
| `config/examples/newton.yaml` | **CREATE** | Example Newton config |
| `tests/numerics/test_preconditioner.py` | **CREATE** | Preconditioner tests |
| `tests/numerics/test_gmres.py` | **CREATE** | GMRES tests |
| `tests/solvers/test_newton.py` | **CREATE** | Newton solver tests |
| `tests/solvers/test_implicit_euler.py` | **CREATE** | Implicit Euler system & pseudo-transient tests |

---

## Implementation Order

1. **Phase 1:** `preconditioner.py` + tests
2. **Phase 2:** Preconditioned RK in `rans_solver.py` + validation  
3. **Phase 3:** `gmres.py` + tests
4. **Phase 4:** `newton_solver.py` + tests + integration
5. **Phase 5:** Config schema updates + example configs + documentation

Each phase is independently testable. We can validate and merge each phase before proceeding to the next.

---

## Decisions Made

| Question | Decision |
|----------|----------|
| Preconditioner recomputation | Every Newton iteration (cheap) |
| GMRES restart dimension | Fixed m=20 in config |
| Hybrid RK→Newton | No. Start directly with Newton |
| Continuation strategy | Pseudo-transient via exponential CFL ramp |
| Residual smoothing with Newton | Part of R(Q) so Jacobian includes it |
| Precondition vs smooth order (RK) | Precondition first, then smooth |

---

## References

- Knoll & Keyes (2004). "Jacobian-free Newton-Krylov methods: a survey of approaches and applications." JCP 193.
- Kelley & Keyes (1998). "Convergence analysis of pseudo-transient continuation." SIAM J. Numer. Anal.
- Saad & Schultz (1986). "GMRES: A generalized minimal residual algorithm for solving nonsymmetric linear systems." SIAM J. Sci. Stat. Comput.
- JAX documentation: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
