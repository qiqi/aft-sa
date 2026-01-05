"""
GMRES(m) solver for Newton-Krylov methods.

Implements restarted GMRES with:
- Configurable restart parameter (m)
- Left preconditioning support
- Modified Gram-Schmidt orthogonalization
- Givens rotations for least-squares
- All operations on GPU via JAX

Reference: Saad & Schultz (1986), "GMRES: A Generalized Minimal Residual
Algorithm for Solving Nonsymmetric Linear Systems"
"""

from dataclasses import dataclass
from typing import Callable, Tuple, Optional

from src.physics.jax_config import jax, jnp


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
    are performed on GPU using JAX.
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
    
    tol_abs = tol * b_norm
    residual_history = []
    total_iters = 0
    
    while total_iters < maxiter:
        # Compute initial residual r = b - A @ x
        if preconditioner is not None:
            r = preconditioner(b - matvec(x))
        else:
            r = b - matvec(x)
        
        r_norm = jnp.linalg.norm(r)
        residual_history.append(float(r_norm))
        
        if r_norm < tol_abs:
            return GMRESResult(
                x=x.reshape(b.shape),
                residual_norm=float(r_norm),
                converged=True,
                iterations=total_iters,
                residual_history=residual_history
            )
        
        # Run one GMRES cycle (up to 'restart' iterations)
        x, r_norm, iters = _gmres_cycle(
            matvec, b, x, r, r_norm, tol_abs, restart, preconditioner
        )
        
        total_iters += iters
        residual_history.append(float(r_norm))
        
        if r_norm < tol_abs:
            return GMRESResult(
                x=x.reshape(b.shape),
                residual_norm=float(r_norm),
                converged=True,
                iterations=total_iters,
                residual_history=residual_history
            )
    
    # Max iterations reached
    return GMRESResult(
        x=x.reshape(b.shape),
        residual_norm=float(r_norm),
        converged=False,
        iterations=total_iters,
        residual_history=residual_history
    )


def _gmres_cycle(
    matvec: Callable,
    b: jnp.ndarray,
    x: jnp.ndarray,
    r: jnp.ndarray,
    r_norm: float,
    tol_abs: float,
    m: int,
    preconditioner: Optional[Callable] = None,
) -> Tuple[jnp.ndarray, float, int]:
    """Run one GMRES cycle (Arnoldi + solve).
    
    Parameters
    ----------
    matvec : callable
        A @ v function.
    b : jnp.ndarray
        RHS vector.
    x : jnp.ndarray
        Current solution estimate.
    r : jnp.ndarray
        Current residual.
    r_norm : float
        Norm of current residual.
    tol_abs : float
        Absolute tolerance.
    m : int
        Maximum iterations in this cycle (restart parameter).
    preconditioner : callable, optional
        Left preconditioner.
        
    Returns
    -------
    x_new : jnp.ndarray
        Updated solution.
    r_norm_new : float
        New residual norm.
    iterations : int
        Iterations performed in this cycle.
    """
    n = r.size
    
    # Arnoldi basis vectors (m+1 vectors of length n)
    V = jnp.zeros((m + 1, n))
    V = V.at[0, :].set(r / r_norm)
    
    # Upper Hessenberg matrix (m+1 x m)
    H = jnp.zeros((m + 1, m))
    
    # Givens rotation coefficients
    cs = jnp.zeros(m)  # cosines
    sn = jnp.zeros(m)  # sines
    
    # RHS for least-squares (m+1)
    g = jnp.zeros(m + 1)
    g = g.at[0].set(r_norm)
    
    iterations = 0
    
    for j in range(m):
        # Arnoldi step: compute A @ v_j
        v_j = V[j, :]
        if preconditioner is not None:
            w = preconditioner(matvec(v_j))
        else:
            w = matvec(v_j)
        
        # Modified Gram-Schmidt orthogonalization
        H, w = _modified_gram_schmidt(H, V, w, j)
        
        h_jp1_j = jnp.linalg.norm(w)
        H = H.at[j + 1, j].set(h_jp1_j)
        
        # Check for breakdown
        if h_jp1_j > 1e-14:
            V = V.at[j + 1, :].set(w / h_jp1_j)
        else:
            # Lucky breakdown - solution found
            iterations = j + 1
            break
        
        # Apply previous Givens rotations to new column of H
        H, g = _apply_givens_rotations(H, g, cs, sn, j)
        
        # Compute new Givens rotation
        cs, sn, H, g = _compute_givens_rotation(H, g, cs, sn, j)
        
        iterations = j + 1
        
        # Check convergence using |g[j+1]| as residual estimate
        residual_estimate = jnp.abs(g[j + 1])
        if residual_estimate < tol_abs:
            break
    
    # Solve upper triangular system H[:j, :j] @ y = g[:j]
    # Using back-substitution
    y = _back_substitute(H[:iterations, :iterations], g[:iterations])
    
    # Update solution: x_new = x + V[:iterations, :].T @ y
    x_new = x + V[:iterations, :].T @ y
    
    # Compute actual residual norm
    if preconditioner is not None:
        r_new = preconditioner(b - matvec(x_new))
    else:
        r_new = b - matvec(x_new)
    r_norm_new = float(jnp.linalg.norm(r_new))
    
    return x_new, r_norm_new, iterations


def _modified_gram_schmidt(
    H: jnp.ndarray,
    V: jnp.ndarray,
    w: jnp.ndarray,
    j: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Modified Gram-Schmidt orthogonalization.
    
    Orthogonalizes w against V[0:j+1, :] and stores coefficients in H[:j+1, j].
    """
    for i in range(j + 1):
        h_ij = jnp.dot(w, V[i, :])
        H = H.at[i, j].set(h_ij)
        w = w - h_ij * V[i, :]
    return H, w


def _apply_givens_rotations(
    H: jnp.ndarray,
    g: jnp.ndarray,
    cs: jnp.ndarray,
    sn: jnp.ndarray,
    j: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply previous Givens rotations to column j of H."""
    for i in range(j):
        temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
        H = H.at[i + 1, j].set(-sn[i] * H[i, j] + cs[i] * H[i + 1, j])
        H = H.at[i, j].set(temp)
    return H, g


def _compute_givens_rotation(
    H: jnp.ndarray,
    g: jnp.ndarray,
    cs: jnp.ndarray,
    sn: jnp.ndarray,
    j: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute and apply j-th Givens rotation."""
    a = H[j, j]
    b = H[j + 1, j]
    
    # Compute rotation to eliminate b
    r = jnp.sqrt(a**2 + b**2)
    c = a / (r + 1e-30)
    s = b / (r + 1e-30)
    
    cs = cs.at[j].set(c)
    sn = sn.at[j].set(s)
    
    # Apply to H
    H = H.at[j, j].set(r)
    H = H.at[j + 1, j].set(0.0)
    
    # Apply to g
    temp = c * g[j] + s * g[j + 1]
    g = g.at[j + 1].set(-s * g[j] + c * g[j + 1])
    g = g.at[j].set(temp)
    
    return cs, sn, H, g


def _back_substitute(R: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Solve upper triangular system Rx = b via back substitution."""
    n = b.size
    x = jnp.zeros(n)
    
    for i in range(n - 1, -1, -1):
        s = b[i] - jnp.dot(R[i, i + 1:], x[i + 1:])
        x = x.at[i].set(s / (R[i, i] + 1e-30))
    
    return x


# =============================================================================
# Jacobian-free Newton-Krylov matvec
# =============================================================================

def make_jfnk_matvec(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    Q: jnp.ndarray,
    dt: jnp.ndarray,
    volume: jnp.ndarray,
    nghost: int,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Create Jacobian-free matvec function for Newton-Krylov.
    
    The Newton system for implicit Euler is:
        (V/dt · I + J) · ΔQ = -R(Q)
    
    This function creates a matvec for the LHS:
        matvec(v) = V/dt · v + J @ v
    
    where J @ v is computed via automatic differentiation (JVP).
    
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
        Function that computes (V/dt · I + J) @ v.
    """
    NI = volume.shape[0]
    NJ = volume.shape[1]
    
    # Precompute V/dt diagonal
    diag = (volume / dt).flatten()  # (NI*NJ,)
    
    # We need to extend diag to handle all 4 variables
    diag_full = jnp.repeat(diag, 4)  # (NI*NJ*4,)
    
    # Create the residual function that works on interior only
    def R_full(Q_flat):
        """Residual as a function of flat interior Q."""
        # Reshape to (NI, NJ, 4)
        Q_int = Q_flat.reshape(NI, NJ, 4)
        # Insert into full Q array
        Q_full = Q.at[nghost:-nghost, nghost:-nghost, :].set(Q_int)
        # Compute residual
        R = residual_fn(Q_full)
        return R.flatten()
    
    # Get current interior Q
    Q_int_flat = Q[nghost:-nghost, nghost:-nghost, :].flatten()
    
    def matvec(v):
        """Compute (V/dt · I + J) @ v."""
        # Diagonal term: V/dt * v
        diag_term = diag_full * v
        
        # Jacobian term: J @ v via JVP
        # jax.jvp(f, (x,), (v,)) returns (f(x), df/dx @ v)
        _, Jv = jax.jvp(R_full, (Q_int_flat,), (v,))
        
        return diag_term + Jv
    
    return jax.jit(matvec)


def make_newton_rhs(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    Q: jnp.ndarray,
    nghost: int,
) -> jnp.ndarray:
    """Compute RHS for Newton system: -R(Q).
    
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
        Flattened -R(Q), shape (NI*NJ*4,).
    """
    R = residual_fn(Q)
    return -R.flatten()

