"""
GMRES(m) solver for Newton-Krylov methods.

Implements restarted GMRES with:
- Configurable restart parameter (m)
- Left preconditioning support
- Modified Gram-Schmidt orthogonalization
- Givens rotations for least-squares
- All operations on GPU via JAX (using lax control flow)

Reference: Saad & Schultz (1986), "GMRES: A Generalized Minimal Residual
Algorithm for Solving Nonsymmetric Linear Systems"
"""

from dataclasses import dataclass
from typing import Callable, Tuple, Optional
from functools import lru_cache

from src.physics.jax_config import jax, jnp


# Cache for JIT-compiled GMRES cycles to avoid retracing
_gmres_cycle_cache = {}


@dataclass
class GMRESResult:
    """Result of GMRES solve.
    
    Attributes
    ----------
    x : jnp.ndarray
        Solution vector.
    residual_norm : float
        Final residual norm ||b - Ax||.
    converged : bool
        Whether the solver converged to tolerance.
    iterations : int
        Number of iterations performed.
    residual_history : list
        History of residual norms per iteration.
    """
    x: jnp.ndarray
    residual_norm: float
    converged: bool
    iterations: int
    residual_history: list


def gmres(
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    b: jnp.ndarray,
    x0: Optional[jnp.ndarray] = None,
    tol: float = 1e-6,
    restart: int = 20,
    maxiter: int = 100,
    preconditioner: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
) -> GMRESResult:
    """Solve Ax = b using restarted GMRES.
    
    Solves the linear system Ax = b using the Generalized Minimal Residual
    method with restarts. Supports left preconditioning.
    
    Parameters
    ----------
    matvec : callable
        Matrix-vector product function A @ v. Should accept and return
        flat JAX arrays.
    b : jnp.ndarray
        Right-hand side vector (flattened).
    x0 : jnp.ndarray, optional
        Initial guess. Defaults to zeros.
    tol : float
        Relative tolerance for convergence: ||r|| < tol * ||b||.
    restart : int
        Number of iterations before restart (GMRES(m) parameter).
    maxiter : int
        Maximum total iterations across all restarts.
    preconditioner : callable, optional
        Left preconditioner P^{-1}. Solves P^{-1} @ A @ x = P^{-1} @ b.
        
    Returns
    -------
    GMRESResult
        Solution and convergence information.
        
    Notes
    -----
    The algorithm uses modified Gram-Schmidt for orthogonalization and
    Givens rotations to solve the least-squares problem. All operations
    are performed on GPU using JAX lax control flow.
    """
    n = b.size
    
    # Initial guess
    if x0 is None:
        x = jnp.zeros(n)
    else:
        x = x0.flatten()
    
    # Preprocess b
    if preconditioner is not None:
        b_precond = preconditioner(b)
    else:
        b_precond = b
    
    b_norm = jnp.linalg.norm(b_precond)
    if b_norm < 1e-30:
        return GMRESResult(
            x=x.reshape(b.shape),
            residual_norm=0.0,
            converged=True,
            iterations=0,
            residual_history=[0.0]
        )
    
    tol_abs = tol * float(b_norm)
    
    # Create JIT-compiled GMRES cycle (with caching to avoid retracing)
    # Use underlying cache keys if available (from make_jfnk_matvec)
    matvec_key = getattr(matvec, '_cache_key', id(matvec))
    precond_key = getattr(preconditioner, '_cache_key', id(preconditioner)) if preconditioner else None
    cache_key = (matvec_key, precond_key, n, restart)
    
    if cache_key not in _gmres_cycle_cache:
        _gmres_cycle_cache[cache_key] = _make_gmres_cycle_jit(matvec, preconditioner, n, restart)
    gmres_cycle_jit = _gmres_cycle_cache[cache_key]
    
    # Run GMRES with restarts
    residual_history = []
    total_iters = 0
    r_norm = float(b_norm)
    
    max_restarts = (maxiter + restart - 1) // restart
    
    for _ in range(max_restarts):
        if total_iters >= maxiter:
            break
            
        # Compute initial residual r = P^{-1}(b - A @ x)
        r = b - matvec(x)
        if preconditioner is not None:
            r = preconditioner(r)
        
        r_norm_jax = jnp.linalg.norm(r)
        r_norm = float(r_norm_jax)
        residual_history.append(r_norm)
        
        if r_norm < tol_abs:
            return GMRESResult(
                x=x.reshape(b.shape),
                residual_norm=r_norm,
                converged=True,
                iterations=total_iters,
                residual_history=residual_history
            )
        
        # Run one GMRES cycle (JIT-compiled)
        x_new, final_res, iters = gmres_cycle_jit(x, r, r_norm_jax, tol_abs)
        
        x = x_new
        total_iters += int(iters)
        r_norm = float(final_res)
        
        if r_norm < tol_abs:
            residual_history.append(r_norm)
            return GMRESResult(
                x=x.reshape(b.shape),
                residual_norm=r_norm,
                converged=True,
                iterations=total_iters,
                residual_history=residual_history
            )
    
    # Final residual
    r = b - matvec(x)
    if preconditioner is not None:
        r = preconditioner(r)
    r_norm = float(jnp.linalg.norm(r))
    residual_history.append(r_norm)
    
    return GMRESResult(
        x=x.reshape(b.shape),
        residual_norm=r_norm,
        converged=r_norm < tol_abs,
        iterations=total_iters,
        residual_history=residual_history
    )


def _make_gmres_cycle_jit(
    matvec: Callable,
    preconditioner: Optional[Callable],
    n: int,
    m: int,
) -> Callable:
    """Create a JIT-compiled GMRES cycle function.
    
    This function creates a closure that captures matvec and preconditioner,
    then JIT-compiles the GMRES cycle using JAX lax control flow.
    """
    
    # Define the preconditioned matvec
    if preconditioner is not None:
        def prec_matvec(v):
            return preconditioner(matvec(v))
    else:
        def prec_matvec(v):
            return matvec(v)
    
    @jax.jit
    def gmres_cycle(x, r, r_norm, tol_abs):
        """Run one GMRES cycle using JAX lax control flow."""
        
        # Initialize Arnoldi basis V (m+1 x n)
        V = jnp.zeros((m + 1, n))
        V = V.at[0, :].set(r / r_norm)
        
        # Upper Hessenberg matrix H (m+1 x m)
        H = jnp.zeros((m + 1, m))
        
        # Givens rotation coefficients
        cs = jnp.zeros(m)
        sn = jnp.zeros(m)
        
        # RHS for least-squares
        g = jnp.zeros(m + 1)
        g = g.at[0].set(r_norm)
        
        # State for the loop: (j, V, H, cs, sn, g, converged)
        init_state = (0, V, H, cs, sn, g, False, r_norm)
        
        def cond_fn(state):
            j, V, H, cs, sn, g, converged, res_est = state
            return (j < m) & (~converged)
        
        def body_fn(state):
            j, V, H, cs, sn, g, converged, res_est = state
            
            # Arnoldi step: w = P^{-1} A v_j
            v_j = V[j, :]
            w = prec_matvec(v_j)
            
            # Modified Gram-Schmidt using fori_loop
            def mgs_body(i, carry):
                H_cur, w_cur = carry
                h_ij = jnp.dot(w_cur, V[i, :])
                H_new = H_cur.at[i, j].set(h_ij)
                w_new = w_cur - h_ij * V[i, :]
                return (H_new, w_new)
            
            H, w = jax.lax.fori_loop(0, j + 1, mgs_body, (H, w))
            
            # Normalize
            h_jp1_j = jnp.linalg.norm(w)
            H = H.at[j + 1, j].set(h_jp1_j)
            
            # Update V (handle near-zero case)
            v_new = jnp.where(h_jp1_j > 1e-14, w / h_jp1_j, jnp.zeros(n))
            V = V.at[j + 1, :].set(v_new)
            
            # Apply previous Givens rotations using fori_loop
            def apply_givens_body(i, carry):
                H_cur, = carry
                temp = cs[i] * H_cur[i, j] + sn[i] * H_cur[i + 1, j]
                H_new = H_cur.at[i + 1, j].set(-sn[i] * H_cur[i, j] + cs[i] * H_cur[i + 1, j])
                H_new = H_new.at[i, j].set(temp)
                return (H_new,)
            
            (H,) = jax.lax.fori_loop(0, j, apply_givens_body, (H,))
            
            # Compute new Givens rotation
            a = H[j, j]
            b = H[j + 1, j]
            r_givens = jnp.sqrt(a**2 + b**2)
            c = a / (r_givens + 1e-30)
            s = b / (r_givens + 1e-30)
            
            cs = cs.at[j].set(c)
            sn = sn.at[j].set(s)
            
            # Apply to H
            H = H.at[j, j].set(r_givens)
            H = H.at[j + 1, j].set(0.0)
            
            # Apply to g
            temp = c * g[j] + s * g[j + 1]
            g = g.at[j + 1].set(-s * g[j] + c * g[j + 1])
            g = g.at[j].set(temp)
            
            # Check convergence
            res_est_new = jnp.abs(g[j + 1])
            converged_new = res_est_new < tol_abs
            
            return (j + 1, V, H, cs, sn, g, converged_new, res_est_new)
        
        # Run the Arnoldi process
        final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
        iters, V, H, cs, sn, g, converged, res_est = final_state
        
        # Solve upper triangular system using fori_loop (back substitution)
        # y = solve(H[:iters, :iters], g[:iters])
        y = jnp.zeros(m)
        
        def back_sub_body(i_rev, y_cur):
            # i goes from iters-1 down to 0
            i = iters - 1 - i_rev
            # s = g[i] - sum(H[i, i+1:iters] * y[i+1:iters])
            s = g[i] - jnp.dot(H[i, :], y_cur)  # H[i, i+1:] * y[i+1:]
            y_new = y_cur.at[i].set(s / (H[i, i] + 1e-30))
            return y_new
        
        y = jax.lax.fori_loop(0, iters, back_sub_body, y)
        
        # Update solution: x_new = x + V[:iters, :].T @ y[:iters]
        # Only use first 'iters' components
        y_masked = jnp.where(jnp.arange(m) < iters, y, 0.0)
        x_new = x + V[:m, :].T @ y_masked
        
        return x_new, res_est, iters
    
    return gmres_cycle


# =============================================================================
# Jacobian-free Newton-Krylov matvec
# =============================================================================

# Cache for JVP-based matvec functions (keyed by residual_fn identity)
_jfnk_matvec_cache = {}


def make_jfnk_matvec(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    Q: jnp.ndarray,
    dt: jnp.ndarray,
    volume: jnp.ndarray,
    nghost: int,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Create Jacobian-free matvec function for Newton-Krylov.
    
    The Newton system for implicit Euler is:
        (V/dt · I - J) · ΔQ = R(Q)
    
    This function creates a matvec for the LHS:
        matvec(v) = V/dt · v - J @ v
    
    where J @ v is computed via automatic differentiation (JVP).
    
    The minus sign on J ensures stability: for source terms like SA destruction
    where dR/d(nuHat) < 0, the matrix (V/dt - J) = V/dt + |J| is always positive
    definite, giving a well-conditioned system.
    
    Uses caching to avoid JAX retracing when called multiple times with the
    same residual function.
    
    Parameters
    ----------
    residual_fn : callable
        R(Q) -> (NI, NJ, 4) residual function.
    Q : jnp.ndarray
        Current state (NI+2*nghost, NJ+2*nghost, 4).
    dt : jnp.ndarray
        Local timestep (NI, NJ).
    volume : jnp.ndarray
        Cell volumes (NI, NJ).
    nghost : int
        Number of ghost cells.
        
    Returns
    -------
    matvec : callable
        Function that computes (V/dt · I - J) @ v.
    """
    NI = volume.shape[0]
    NJ = volume.shape[1]
    n = NI * NJ * 4
    
    # Get or create cached matvec implementation for this residual function
    cache_key = (id(residual_fn), NI, NJ, nghost)
    
    if cache_key not in _jfnk_matvec_cache:
        # Create JIT-compiled matvec that takes Q, dt, volume as arguments
        @jax.jit
        def _matvec_impl(Q_state, dt_state, volume_state, v):
            """JIT-compiled matvec implementation."""
            # Precompute V/dt diagonal
            diag = (volume_state / dt_state).flatten()
            diag_full = jnp.repeat(diag, 4)
            
            # Get interior Q
            Q_int_flat = Q_state[nghost:-nghost, nghost:-nghost, :].flatten()
            
            # Create residual function for JVP
            def R_flat(Q_int):
                Q_int_reshaped = Q_int.reshape(NI, NJ, 4)
                Q_full = Q_state.at[nghost:-nghost, nghost:-nghost, :].set(Q_int_reshaped)
                return residual_fn(Q_full).flatten()
            
            # Diagonal term - Jacobian term via JVP (note: MINUS for stability)
            diag_term = diag_full * v
            _, Jv = jax.jvp(R_flat, (Q_int_flat,), (v,))
            
            return diag_term - Jv
        
        _jfnk_matvec_cache[cache_key] = _matvec_impl
    
    _matvec_impl = _jfnk_matvec_cache[cache_key]
    
    # Create wrapper that captures current Q, dt, volume
    def matvec(v):
        return _matvec_impl(Q, dt, volume, v)
    
    # Store reference to the cached impl for cache key matching
    matvec._cache_key = cache_key
    
    return matvec


def make_newton_rhs(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    Q: jnp.ndarray,
    nghost: int,
) -> jnp.ndarray:
    """Compute RHS for Newton system: R(Q).
    
    The Newton system is (V/dt - J) · ΔQ = R, so the RHS is simply R(Q).
    
    Parameters
    ----------
    residual_fn : callable
        R(Q) residual function.
    Q : jnp.ndarray
        Current state.
    nghost : int
        Number of ghost cells.
        
    Returns
    -------
    rhs : jnp.ndarray
        Flattened R(Q), shape (NI*NJ*4,).
    """
    R = residual_fn(Q)
    return R.flatten()


# =============================================================================
# Legacy helper functions (for testing)
# =============================================================================

def _modified_gram_schmidt(
    H: jnp.ndarray,
    V: jnp.ndarray,
    w: jnp.ndarray,
    j: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Modified Gram-Schmidt orthogonalization (legacy, for testing)."""
    for i in range(j + 1):
        h_ij = jnp.dot(w, V[i, :])
        H = H.at[i, j].set(h_ij)
        w = w - h_ij * V[i, :]
    return H, w


def _back_substitute(R: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Solve upper triangular system Rx = b (legacy, for testing)."""
    n = b.size
    x = jnp.zeros(n)
    
    for i in range(n - 1, -1, -1):
        s = b[i] - jnp.dot(R[i, i + 1:], x[i + 1:])
        x = x.at[i].set(s / (R[i, i] + 1e-30))
    
    return x
