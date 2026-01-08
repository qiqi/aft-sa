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



# =============================================================================
# JAX-native GMRES with restarts
# =============================================================================

from functools import partial

@partial(jax.jit, static_argnums=(0, 4, 5, 6))
def gmres_solve(
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    b: jnp.ndarray,
    x0: jnp.ndarray,
    tol_abs: float,
    restart: int,
    maxiter: int,
    preconditioner: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
) -> Tuple[jnp.ndarray, float, int, bool]:
    """Fully JIT-able restarted GMRES solver.
    
    This function uses jax.lax control flow for both the internal Arnoldi
    cycle and the external restart loop, allowing it to be compiled into
    a single XLA executable without Python overhead.
    """
    n = b.size
    
    # Define the preconditioned matvec
    if preconditioner is not None:
        def prec_matvec(v):
            return preconditioner(matvec(v))
        b_precond = preconditioner(b)
    else:
        def prec_matvec(v):
            return matvec(v)
        b_precond = b
    
    b_norm = jnp.linalg.norm(b_precond)
    
    # Internal Arnoldi cycle (m iterations)
    def gmres_cycle(x, r, r_norm):
        # Initialize Arnoldi basis V (m+1 x n)
        V = jnp.zeros((restart + 1, n))
        V = V.at[0, :].set(r / (r_norm + 1e-30))
        
        # Upper Hessenberg matrix H (m+1 x m)
        H = jnp.zeros((restart + 1, restart))
        
        # Givens rotation coefficients
        cs = jnp.zeros(restart)
        sn = jnp.zeros(restart)
        
        # RHS for least-squares
        g = jnp.zeros(restart + 1)
        g = g.at[0].set(r_norm)
        
        # State: (j, V, H, cs, sn, g, converged, res_est)
        def cycle_cond(state):
            j, _, _, _, _, _, converged, _ = state
            return (j < restart) & (~converged)
        
        def cycle_body(state):
            j, V, H, cs, sn, g, converged, res_est = state
            
            # Arnoldi step
            v_j = V[j, :]
            w = prec_matvec(v_j)
            
            # Modified Gram-Schmidt
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
            v_new = jnp.where(h_jp1_j > 1e-14, w / h_jp1_j, jnp.zeros(n))
            V = V.at[j + 1, :].set(v_new)
            
            # Apply previous Givens rotations
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
            
            # Apply to H and g
            H = H.at[j, j].set(r_givens)
            H = H.at[j + 1, j].set(0.0)
            temp = c * g[j] + s * g[j + 1]
            g = g.at[j + 1].set(-s * g[j] + c * g[j + 1])
            g = g.at[j].set(temp)
            
            res_est_new = jnp.abs(g[j + 1])
            converged_new = res_est_new < tol_abs
            
            return (j + 1, V, H, cs, sn, g, converged_new, res_est_new)
        
        init_cycle = (0, V, H, cs, sn, g, False, r_norm)
        final_cycle = jax.lax.while_loop(cycle_cond, cycle_body, init_cycle)
        iters, V, H, cs, sn, g, _, res_est = final_cycle
        
        # Back substitution
        y = jnp.zeros(restart)
        def back_sub_body(i_rev, y_cur):
            i = iters - 1 - i_rev
            s = g[i] - jnp.dot(H[i, :], y_cur)
            y_new = y_cur.at[i].set(s / (H[i, i] + 1e-30))
            return y_new
        
        y = jax.lax.fori_loop(0, iters, back_sub_body, y)
        
        # Update solution
        y_masked = jnp.where(jnp.arange(restart) < iters, y, 0.0)
        x_new = x + V[:restart, :].T @ y_masked
        
        return x_new, res_est, iters

    # Restart loop: (total_iters, x, r_norm, converged)
    def restart_cond(state):
        total_iters, _, r_norm, converged = state
        return (total_iters < maxiter) & (~converged) & (r_norm > tol_abs)
    
    def restart_body(state):
        total_iters, x, _, _ = state
        
        # Initial residual for this cycle
        r = b - matvec(x)
        if preconditioner is not None:
            r = preconditioner(r)
        r_norm = jnp.linalg.norm(r)
        
        # Run Arnoldi cycle
        x_new, r_norm_new, cycle_iters = gmres_cycle(x, r, r_norm)
        
        return (total_iters + cycle_iters, x_new, r_norm_new, r_norm_new < tol_abs)
    
    init_restart = (0, x0, b_norm, b_norm < tol_abs)
    final_state = jax.lax.while_loop(restart_cond, restart_body, init_restart)
    total_iters, x_final, final_r_norm, converged = final_state
    
    return x_final, final_r_norm, total_iters, converged


def gmres(
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    b: jnp.ndarray,
    x0: Optional[jnp.ndarray] = None,
    tol: float = 1e-6,
    restart: int = 20,
    maxiter: int = 100,
    preconditioner: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
) -> GMRESResult:
    """Public wrapper for JAX-native GMRES solver."""
    n = b.size
    if x0 is None:
        x0 = jnp.zeros(n)
    else:
        x0 = x0.flatten()
    
    # Compute tol_abs based on preconditioned b
    if preconditioner is not None:
        b_precond = preconditioner(b)
    else:
        b_precond = b
    b_norm = jnp.linalg.norm(b_precond)
    tol_abs = tol * float(b_norm)
    
    x, r_norm, iters, converged = gmres_solve(
        matvec, b, x0, tol_abs, restart, maxiter, preconditioner
    )
    
    return GMRESResult(
        x=x,
        residual_norm=float(r_norm),
        converged=bool(converged),
        iterations=int(iters),
        residual_history=[] # Not implemented in JIT version
    )

