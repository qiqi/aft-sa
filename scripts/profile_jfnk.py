#!/usr/bin/env python
"""
Performance profiling for JFNK operations.
Used to identify bottlenecks and compare implementation strategies.
"""

import time
import jax
import jax.numpy as jnp

print(f"JAX devices: {jax.devices()}")

from src.physics.jax_config import jnp
from src.numerics.preconditioner import (
    BlockJacobiPreconditioner, 
    _compute_block_jacobians,
    _compute_block_jacobians_jvp,
)
from src.numerics.gmres import gmres, make_jfnk_matvec, make_newton_rhs
from src.constants import NGHOST


def create_test_problem(NI=64, NJ=32):
    """Create a simple test problem for timing."""
    nghost = NGHOST
    
    # Create a simple state array
    Q = jnp.ones((NI + 2*nghost, NJ + 2*nghost, 4))
    Q = Q.at[:, :, 0].set(1.0)   # rho
    Q = Q.at[:, :, 1].set(0.5)   # rho*u
    Q = Q.at[:, :, 2].set(0.0)   # rho*v
    Q = Q.at[:, :, 3].set(1e-5)  # nuHat
    
    # Simple volume and dt
    volume = jnp.ones((NI, NJ))
    dt = jnp.ones((NI, NJ)) * 0.01
    
    return Q, volume, dt, nghost


def simple_residual(Q):
    """A simple residual for testing - just returns interior cells."""
    nghost = NGHOST
    Q_int = Q[nghost:-nghost, nghost:-nghost, :]
    target = jnp.ones_like(Q_int) * 0.5
    target = target.at[:, :, 3].set(1e-6)
    return Q_int - target


def time_operation(name, fn, n_warmup=2, n_iter=5):
    """Time an operation with warmup."""
    # Warmup (includes JIT compilation)
    for _ in range(n_warmup):
        result = fn()
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, jnp.ndarray):
            result.block_until_ready()
    
    # Timed runs
    times = []
    for _ in range(n_iter):
        start = time.perf_counter()
        result = fn()
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, jnp.ndarray):
            result.block_until_ready()
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    print(f"  {name}: {avg_time*1000:.2f} ms (avg of {n_iter})")
    return avg_time


def main():
    print("=" * 70)
    print("JFNK Performance Profiling")
    print("=" * 70)
    
    # Test different grid sizes
    for NI, NJ in [(32, 16), (64, 32), (128, 64)]:
        print(f"\n{'='*70}")
        print(f"Grid size: {NI} x {NJ} ({NI*NJ*4} DOFs)")
        print("=" * 70)
        
        Q, volume, dt, nghost = create_test_problem(NI, NJ)
        
        # JIT-compile residual
        residual_jit = jax.jit(simple_residual)
        
        # =====================================================================
        # 1. Basic operations
        # =====================================================================
        print("\n1. Basic Operations:")
        time_operation("Residual eval", lambda: residual_jit(Q))
        
        # =====================================================================
        # 2. Block Jacobian computation - FD vs JVP
        # =====================================================================
        print("\n2. Block Jacobian Computation:")
        
        # Finite difference (current implementation)
        def compute_jacobians_fd():
            return _compute_block_jacobians(residual_jit, Q, nghost)
        t_fd = time_operation("FD (16 perturbations)", compute_jacobians_fd)
        
        # JVP-based
        def compute_jacobians_jvp():
            return _compute_block_jacobians_jvp(residual_jit, Q, nghost)
        t_jvp = time_operation("JVP (16 tangents)", compute_jacobians_jvp)
        
        print(f"  -> JVP vs FD speedup: {t_fd/t_jvp:.2f}x" if t_jvp < t_fd else f"  -> FD is {t_jvp/t_fd:.2f}x faster")
        
        # =====================================================================
        # 3. Full preconditioner
        # =====================================================================
        print("\n3. Full Preconditioner:")
        def compute_precond():
            return BlockJacobiPreconditioner.compute(
                residual_fn=residual_jit,
                Q=Q, dt=dt, volume=volume, nghost=nghost,
            )
        time_operation("Compute (JVP + inversion, cached)", compute_precond)
        
        # Get preconditioner for later tests
        precond = compute_precond()
        precond_apply = precond.apply_jit()
        
        # =====================================================================
        # 4. GMRES components
        # =====================================================================
        print("\n4. GMRES Components:")
        
        matvec = make_jfnk_matvec(residual_jit, Q, dt, volume, nghost)
        rhs = make_newton_rhs(residual_jit, Q, nghost)
        v = jnp.ones_like(rhs)
        
        time_operation("JFNK matvec (1 JVP)", lambda: matvec(v))
        time_operation("Precond apply", lambda: precond_apply(rhs))
        
        # =====================================================================
        # 5. Full GMRES
        # =====================================================================
        print("\n5. GMRES:")
        
        def run_gmres(restart, maxiter):
            return gmres(
                matvec=matvec, b=rhs, x0=None,
                tol=1e-10, restart=restart, maxiter=maxiter,
                preconditioner=precond_apply,
            )
        
        time_operation("GMRES (10 iters, m=10)", 
                      lambda: run_gmres(10, 10), n_warmup=1, n_iter=3)
        time_operation("GMRES (20 iters, m=20)", 
                      lambda: run_gmres(20, 20), n_warmup=1, n_iter=3)
        
        # =====================================================================
        # 6. Full Newton step
        # =====================================================================
        print("\n6. Full Newton Step:")
        
        def newton_step_reused():
            """Newton step with reused matvec (after warmup)."""
            r = make_newton_rhs(residual_jit, Q, nghost)
            return gmres(
                matvec=matvec, b=r, x0=None,
                tol=1e-3, restart=20, maxiter=100,
                preconditioner=precond_apply,
            ).x
        
        time_operation("GMRES only (reused funcs)", newton_step_reused, n_warmup=1, n_iter=3)
        
        def newton_step_with_precond():
            """Full Newton step including preconditioner computation."""
            p = BlockJacobiPreconditioner.compute(
                residual_fn=residual_jit, Q=Q, dt=dt, volume=volume, nghost=nghost
            )
            r = make_newton_rhs(residual_jit, Q, nghost)
            return gmres(
                matvec=matvec, b=r, x0=None,
                tol=1e-3, restart=20, maxiter=100,
                preconditioner=precond_apply,
            ).x
        
        time_operation("With precond compute (JVP)", newton_step_with_precond, n_warmup=1, n_iter=3)
    
    print("\n" + "=" * 70)
    print("Profiling complete")
    print("=" * 70)


if __name__ == "__main__":
    main()

