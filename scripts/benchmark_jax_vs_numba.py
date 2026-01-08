#!/usr/bin/env python3
"""
Benchmark: Full Solver Iteration - JAX GPU vs Numba CPU

This script simulates a complete solver iteration (without I/O overhead)
to compare the performance of JAX GPU vs Numba CPU backends.
"""

import sys
import time
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

# Check for JAX
try:
    from src.physics.jax_config import jax, jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("JAX not available, running Numba-only benchmark")

from src.constants import NGHOST
from src.numerics.fluxes import (
    compute_fluxes, FluxConfig, GridMetrics as FluxMetrics,
)
from src.numerics.gradients import compute_gradients, GradientMetrics
from src.numerics.viscous_fluxes import compute_viscous_fluxes
from src.numerics.explicit_smoothing import smooth_explicit_numba
from src.solvers.time_stepping import compute_local_timestep, TimeStepConfig

if JAX_AVAILABLE:
    from src.numerics.fluxes import compute_fluxes_jax
    from src.numerics.gradients import compute_gradients_jax
    from src.numerics.viscous_fluxes import compute_viscous_fluxes_jax
    from src.numerics.explicit_smoothing import smooth_explicit_jax
    from src.solvers.time_stepping import compute_local_timestep_jax


def create_test_data(NI, NJ, nghost=NGHOST):
    """Create realistic test data for solver benchmark."""
    np.random.seed(42)
    
    # State array (freestream-like)
    Q = np.zeros((NI + 2*nghost, NJ + 2*nghost, 4), dtype=np.float64)
    Q[:, :, 0] = 0.0  # pressure
    Q[:, :, 1] = 1.0 + np.random.randn(NI + 2*nghost, NJ + 2*nghost) * 0.05  # u
    Q[:, :, 2] = np.random.randn(NI + 2*nghost, NJ + 2*nghost) * 0.05  # v
    Q[:, :, 3] = 1e-5  # nu_tilde
    
    # Grid metrics (C-grid-like)
    Si_x = np.random.randn(NI + 1, NJ) * 0.01 + 0.1
    Si_y = np.random.randn(NI + 1, NJ) * 0.01
    Sj_x = np.random.randn(NI, NJ + 1) * 0.01
    Sj_y = np.random.randn(NI, NJ + 1) * 0.01 + 0.1
    volume = np.random.rand(NI, NJ) * 0.1 + 0.01
    
    return Q, Si_x, Si_y, Sj_x, Sj_y, volume


def run_numba_iteration(Q, Si_x, Si_y, Sj_x, Sj_y, volume, beta, cfg_flux, cfg_ts, nu, n_stages=5):
    """Run one complete RK iteration using Numba/NumPy."""
    NI, NJ = volume.shape
    
    # Timestep
    dt = compute_local_timestep(Q, Si_x, Si_y, Sj_x, Sj_y, volume, beta, cfg_ts, nu=nu)
    
    Q0 = Q.copy()
    Qk = Q.copy()
    
    alphas = [0.25, 0.166666667, 0.375, 0.5, 1.0][:n_stages]
    
    flux_metrics = FluxMetrics(Si_x, Si_y, Sj_x, Sj_y, volume)
    grad_metrics = GradientMetrics(Si_x, Si_y, Sj_x, Sj_y, volume)
    
    for alpha in alphas:
        # Flux
        R = compute_fluxes(Qk, flux_metrics, beta, cfg_flux)
        
        # Gradient + viscous
        grad = compute_gradients(Qk, grad_metrics)
        mu_eff = np.full((NI, NJ), nu)
        R_visc = compute_viscous_fluxes(Qk, grad, grad_metrics, nu)
        R = R + R_visc
        
        # Smoothing
        R = smooth_explicit_numba(R, 0.2, 2)
        
        # Update
        Qk = Q0.copy()
        Qk[NGHOST:-NGHOST, NGHOST:-NGHOST, :] += alpha * (dt / volume)[:, :, np.newaxis] * R
    
    return Qk


def run_jax_iteration(Q, Si_x, Si_y, Sj_x, Sj_y, volume, beta, k4, cfl, nu, nghost, n_stages=5):
    """Run one complete RK iteration using JAX on GPU."""
    NI, NJ = volume.shape
    
    # Timestep
    dt = compute_local_timestep_jax(Q, Si_x, Si_y, Sj_x, Sj_y, volume, beta, cfl, nghost, nu=nu)
    
    Q0 = Q
    Qk = Q
    
    alphas = jnp.array([0.25, 0.166666667, 0.375, 0.5, 1.0][:n_stages])
    
    mu_eff = jnp.full((NI, NJ), nu)
    
    for i in range(n_stages):
        alpha = alphas[i]
        
        # Flux
        R = compute_fluxes_jax(Qk, Si_x, Si_y, Sj_x, Sj_y, beta, k4, nghost)
        
        # Gradient + viscous
        grad = compute_gradients_jax(Qk, Si_x, Si_y, Sj_x, Sj_y, volume, nghost)
        R_visc = compute_viscous_fluxes_jax(grad, Si_x, Si_y, Sj_x, Sj_y, mu_eff)
        R = R + R_visc
        
        # Smoothing
        R = smooth_explicit_jax(R, 0.2, 2)
        
        # Update (JAX functional style)
        Q_int_new = Q0[nghost:-nghost, nghost:-nghost, :] + alpha * (dt / volume)[:, :, None] * R
        Qk = Q0.at[nghost:-nghost, nghost:-nghost, :].set(Q_int_new)
    
    return Qk


def main():
    parser = argparse.ArgumentParser(description="Benchmark JAX GPU vs Numba CPU")
    parser.add_argument("--ni", type=int, default=256, help="Grid cells in I-direction")
    parser.add_argument("--nj", type=int, default=64, help="Grid cells in J-direction")
    parser.add_argument("--iters", type=int, default=100, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    args = parser.parse_args()
    
    NI, NJ = args.ni, args.nj
    n_iter = args.iters
    n_warmup = args.warmup
    nghost = NGHOST
    beta = 10.0
    nu = 1e-6
    
    print(f"\n{'='*70}")
    print(f"SOLVER ITERATION BENCHMARK: JAX GPU vs Numba CPU")
    print(f"{'='*70}")
    print(f"Grid size: {NI} x {NJ} cells ({(NI*NJ)/1e6:.2f}M cells)")
    print(f"Iterations: {n_iter} (warmup: {n_warmup})")
    
    # Create test data
    Q_np, Si_x, Si_y, Sj_x, Sj_y, volume = create_test_data(NI, NJ)
    
    cfg_flux = FluxConfig()
    cfg_ts = TimeStepConfig(cfl=0.8)
    
    # =========================================================================
    # Numba CPU Benchmark
    # =========================================================================
    print(f"\n--- Numba CPU ---")
    
    # Warmup
    print("Warming up Numba JIT...")
    for _ in range(n_warmup):
        _ = run_numba_iteration(Q_np.copy(), Si_x, Si_y, Sj_x, Sj_y, volume, 
                                 beta, cfg_flux, cfg_ts, nu)
    
    # Benchmark
    print(f"Running {n_iter} iterations...")
    t0 = time.perf_counter()
    Q_numba = Q_np.copy()
    for i in range(n_iter):
        Q_numba = run_numba_iteration(Q_numba, Si_x, Si_y, Sj_x, Sj_y, volume,
                                       beta, cfg_flux, cfg_ts, nu)
    t_numba = time.perf_counter() - t0
    
    ms_per_iter_numba = t_numba / n_iter * 1000
    print(f"Total time: {t_numba:.2f}s")
    print(f"Per iteration: {ms_per_iter_numba:.2f} ms")
    
    # =========================================================================
    # JAX GPU Benchmark
    # =========================================================================
    if JAX_AVAILABLE:
        jax.config.update('jax_platform_name', 'cuda')
        print(f"\n--- JAX GPU ({jax.devices()[0]}) ---")
        
        # Transfer to GPU
        Q_jax = jax.device_put(jnp.array(Q_np))
        Si_x_jax = jax.device_put(jnp.array(Si_x))
        Si_y_jax = jax.device_put(jnp.array(Si_y))
        Sj_x_jax = jax.device_put(jnp.array(Sj_x))
        Sj_y_jax = jax.device_put(jnp.array(Sj_y))
        vol_jax = jax.device_put(jnp.array(volume))
        
        # Warmup (JIT compilation)
        print("Warming up JAX JIT...")
        for _ in range(n_warmup):
            Q_jax = run_jax_iteration(Q_jax, Si_x_jax, Si_y_jax, Sj_x_jax, Sj_y_jax,
                                       vol_jax, beta, cfg_flux.k4, cfg_ts.cfl, nu, nghost)
        jax.block_until_ready(Q_jax)
        
        # Reset
        Q_jax = jax.device_put(jnp.array(Q_np))
        
        # Benchmark
        print(f"Running {n_iter} iterations...")
        t0 = time.perf_counter()
        for i in range(n_iter):
            Q_jax = run_jax_iteration(Q_jax, Si_x_jax, Si_y_jax, Sj_x_jax, Sj_y_jax,
                                       vol_jax, beta, cfg_flux.k4, cfg_ts.cfl, nu, nghost)
        jax.block_until_ready(Q_jax)
        t_jax = time.perf_counter() - t0
        
        ms_per_iter_jax = t_jax / n_iter * 1000
        print(f"Total time: {t_jax:.2f}s")
        print(f"Per iteration: {ms_per_iter_jax:.2f} ms")
        
        # Compare results
        Q_jax_np = np.array(Q_jax)
        Q_numba_final = Q_numba
        
        diff = np.abs(Q_jax_np - Q_numba_final).max()
        rel_diff = diff / (np.abs(Q_numba_final).max() + 1e-12)
        
        print(f"\n{'='*70}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"Grid: {NI}x{NJ} = {NI*NJ:,} cells")
        print(f"")
        print(f"{'Backend':<15} {'Time/iter (ms)':<18} {'Speedup':<12}")
        print(f"{'-'*45}")
        print(f"{'Numba CPU':<15} {ms_per_iter_numba:<18.2f} {'1.0x (baseline)':<12}")
        print(f"{'JAX GPU':<15} {ms_per_iter_jax:<18.2f} {ms_per_iter_numba/ms_per_iter_jax:.1f}x")
        print(f"")
        print(f"Numerical difference: {rel_diff:.2e} (max relative)")
        status = "✅ MATCH" if rel_diff < 1e-6 else ("⚠️  CLOSE" if rel_diff < 1e-3 else "❌ MISMATCH")
        print(f"Status: {status}")
        print(f"")
        print(f"Estimated time for 10,000 iterations:")
        print(f"  Numba CPU: {ms_per_iter_numba * 10000 / 1000 / 60:.1f} minutes")
        print(f"  JAX GPU:   {ms_per_iter_jax * 10000 / 1000 / 60:.1f} minutes")
        print(f"{'='*70}")
    else:
        print("\nJAX not available - skipping GPU benchmark")


if __name__ == "__main__":
    main()


