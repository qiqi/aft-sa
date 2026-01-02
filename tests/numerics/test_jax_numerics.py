"""
Tests comparing JAX vs Numba implementations of numerical kernels.

Validates numerical equivalence and benchmarks performance.
"""

import pytest
import numpy as np
import time
from dataclasses import dataclass

from src.physics.jax_config import jax, jnp
from src.constants import NGHOST


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def small_grid():
    """Small grid for quick tests."""
    NI, NJ = 32, 24
    return NI, NJ, NGHOST


@pytest.fixture
def medium_grid():
    """Medium grid for validation."""
    NI, NJ = 64, 48
    return NI, NJ, NGHOST


@pytest.fixture
def large_grid():
    """Large grid for performance tests."""
    NI, NJ = 256, 128
    return NI, NJ, NGHOST


def create_test_state(NI, NJ, nghost, seed=42):
    """Create random state array with ghost cells."""
    np.random.seed(seed)
    Q = np.random.randn(NI + 2*nghost, NJ + 2*nghost, 4) * 0.1
    Q[:, :, 0] += 0.0  # Pressure perturbation
    Q[:, :, 1] += 1.0  # u ~ 1
    Q[:, :, 2] += 0.0  # v ~ 0
    Q[:, :, 3] = np.abs(Q[:, :, 3]) * 1e-3  # nu_t > 0
    return Q


def create_test_metrics(NI, NJ, seed=42):
    """Create random grid metrics."""
    np.random.seed(seed + 1)
    # I-face normals: (NI+1, NJ)
    Si_x = np.random.randn(NI + 1, NJ) * 0.01 + 0.1
    Si_y = np.random.randn(NI + 1, NJ) * 0.01
    # J-face normals: (NI, NJ+1)
    Sj_x = np.random.randn(NI, NJ + 1) * 0.01
    Sj_y = np.random.randn(NI, NJ + 1) * 0.01 + 0.1
    # Volumes: (NI, NJ)
    volume = np.random.rand(NI, NJ) * 0.1 + 0.01
    return Si_x, Si_y, Sj_x, Sj_y, volume


# =============================================================================
# Flux Tests
# =============================================================================

class TestFluxComparison:
    """Compare JAX and Numba flux computations."""
    
    def test_flux_numerical_equivalence(self, medium_grid):
        """JAX and Numba flux residuals should match."""
        NI, NJ, nghost = medium_grid
        from src.numerics.fluxes import compute_fluxes, FluxConfig, GridMetrics
        from src.numerics.fluxes import compute_fluxes_jax
        
        Q = create_test_state(NI, NJ, nghost)
        Si_x, Si_y, Sj_x, Sj_y, volume = create_test_metrics(NI, NJ)
        
        beta = 10.0
        k4 = 0.016
        
        # Numba version
        metrics = GridMetrics(Si_x, Si_y, Sj_x, Sj_y, volume)
        cfg = FluxConfig(k4=k4)
        res_numba = compute_fluxes(Q, metrics, beta, cfg)
        
        # JAX version
        res_jax = compute_fluxes_jax(
            jnp.array(Q),
            jnp.array(Si_x), jnp.array(Si_y),
            jnp.array(Sj_x), jnp.array(Sj_y),
            beta, k4, nghost
        )
        res_jax_np = np.array(res_jax)
        
        # Compare
        max_diff = np.abs(res_numba - res_jax_np).max()
        rel_diff = max_diff / (np.abs(res_numba).max() + 1e-12)
        
        print(f"Flux max diff: {max_diff:.2e}, rel diff: {rel_diff:.2e}")
        assert rel_diff < 1e-10, f"Flux mismatch: rel_diff = {rel_diff:.2e}"
    
    def test_flux_performance(self, large_grid):
        """Benchmark flux computation."""
        NI, NJ, nghost = large_grid
        from src.numerics.fluxes import compute_fluxes, FluxConfig, GridMetrics
        from src.numerics.fluxes import compute_fluxes_jax
        
        Q = create_test_state(NI, NJ, nghost)
        Si_x, Si_y, Sj_x, Sj_y, volume = create_test_metrics(NI, NJ)
        beta = 10.0
        k4 = 0.016
        
        metrics = GridMetrics(Si_x, Si_y, Sj_x, Sj_y, volume)
        cfg = FluxConfig(k4=k4)
        
        # JAX arrays
        Q_jax = jnp.array(Q)
        Si_x_jax, Si_y_jax = jnp.array(Si_x), jnp.array(Si_y)
        Sj_x_jax, Sj_y_jax = jnp.array(Sj_x), jnp.array(Sj_y)
        
        # Warm up
        _ = compute_fluxes(Q, metrics, beta, cfg)
        _ = compute_fluxes_jax(Q_jax, Si_x_jax, Si_y_jax, Sj_x_jax, Sj_y_jax, beta, k4, nghost)
        jax.block_until_ready(_)
        
        n_iter = 50
        
        # Numba timing
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = compute_fluxes(Q, metrics, beta, cfg)
        t_numba = (time.perf_counter() - t0) / n_iter * 1000
        
        # JAX timing
        t0 = time.perf_counter()
        for _ in range(n_iter):
            res = compute_fluxes_jax(Q_jax, Si_x_jax, Si_y_jax, Sj_x_jax, Sj_y_jax, beta, k4, nghost)
            jax.block_until_ready(res)
        t_jax = (time.perf_counter() - t0) / n_iter * 1000
        
        print(f"\nFlux computation ({NI}x{NJ}):")
        print(f"  Numba: {t_numba:.3f} ms")
        print(f"  JAX:   {t_jax:.3f} ms")
        print(f"  Speedup: {t_numba/t_jax:.2f}x")


# =============================================================================
# Gradient Tests
# =============================================================================

class TestGradientComparison:
    """Compare JAX and Numba gradient computations."""
    
    def test_gradient_numerical_equivalence(self, medium_grid):
        """JAX and Numba gradients should match."""
        NI, NJ, nghost = medium_grid
        from src.numerics.gradients import compute_gradients, GradientMetrics
        from src.numerics.gradients import compute_gradients_jax
        
        Q = create_test_state(NI, NJ, nghost)
        Si_x, Si_y, Sj_x, Sj_y, volume = create_test_metrics(NI, NJ)
        
        # Numba version
        metrics = GradientMetrics(Si_x, Si_y, Sj_x, Sj_y, volume)
        grad_numba = compute_gradients(Q, metrics)
        
        # JAX version
        grad_jax = compute_gradients_jax(
            jnp.array(Q),
            jnp.array(Si_x), jnp.array(Si_y),
            jnp.array(Sj_x), jnp.array(Sj_y),
            jnp.array(volume), nghost
        )
        grad_jax_np = np.array(grad_jax)
        
        max_diff = np.abs(grad_numba - grad_jax_np).max()
        rel_diff = max_diff / (np.abs(grad_numba).max() + 1e-12)
        
        print(f"Gradient max diff: {max_diff:.2e}, rel diff: {rel_diff:.2e}")
        assert rel_diff < 1e-10, f"Gradient mismatch: rel_diff = {rel_diff:.2e}"
    
    def test_gradient_performance(self, large_grid):
        """Benchmark gradient computation."""
        NI, NJ, nghost = large_grid
        from src.numerics.gradients import compute_gradients, GradientMetrics
        from src.numerics.gradients import compute_gradients_jax
        
        Q = create_test_state(NI, NJ, nghost)
        Si_x, Si_y, Sj_x, Sj_y, volume = create_test_metrics(NI, NJ)
        
        metrics = GradientMetrics(Si_x, Si_y, Sj_x, Sj_y, volume)
        
        Q_jax = jnp.array(Q)
        Si_x_jax, Si_y_jax = jnp.array(Si_x), jnp.array(Si_y)
        Sj_x_jax, Sj_y_jax = jnp.array(Sj_x), jnp.array(Sj_y)
        vol_jax = jnp.array(volume)
        
        # Warm up
        _ = compute_gradients(Q, metrics)
        _ = compute_gradients_jax(Q_jax, Si_x_jax, Si_y_jax, Sj_x_jax, Sj_y_jax, vol_jax, nghost)
        
        n_iter = 50
        
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = compute_gradients(Q, metrics)
        t_numba = (time.perf_counter() - t0) / n_iter * 1000
        
        t0 = time.perf_counter()
        for _ in range(n_iter):
            res = compute_gradients_jax(Q_jax, Si_x_jax, Si_y_jax, Sj_x_jax, Sj_y_jax, vol_jax, nghost)
            jax.block_until_ready(res)
        t_jax = (time.perf_counter() - t0) / n_iter * 1000
        
        print(f"\nGradient computation ({NI}x{NJ}):")
        print(f"  Numba: {t_numba:.3f} ms")
        print(f"  JAX:   {t_jax:.3f} ms")
        print(f"  Speedup: {t_numba/t_jax:.2f}x")


# =============================================================================
# Viscous Flux Tests
# =============================================================================

class TestViscousFluxComparison:
    """Compare JAX and Numba viscous flux computations."""
    
    def test_viscous_flux_numerical_equivalence(self, medium_grid):
        """JAX and Numba viscous fluxes should match."""
        NI, NJ, nghost = medium_grid
        from src.numerics.gradients import compute_gradients, GradientMetrics
        from src.numerics.viscous_fluxes import compute_viscous_fluxes
        from src.numerics.viscous_fluxes import compute_viscous_fluxes_jax
        
        Q = create_test_state(NI, NJ, nghost)
        Si_x, Si_y, Sj_x, Sj_y, volume = create_test_metrics(NI, NJ)
        mu_laminar = 1e-3
        
        metrics = GradientMetrics(Si_x, Si_y, Sj_x, Sj_y, volume)
        grad = compute_gradients(Q, metrics)
        mu_eff = np.full((NI, NJ), mu_laminar)
        
        # Numba version
        res_numba = compute_viscous_fluxes(Q, grad, metrics, mu_laminar)
        
        # JAX version
        res_jax = compute_viscous_fluxes_jax(
            jnp.array(grad),
            jnp.array(Si_x), jnp.array(Si_y),
            jnp.array(Sj_x), jnp.array(Sj_y),
            jnp.array(mu_eff)
        )
        res_jax_np = np.array(res_jax)
        
        # Only compare u, v components (indices 1, 2)
        max_diff = np.abs(res_numba[:, :, 1:3] - res_jax_np[:, :, 1:3]).max()
        rel_diff = max_diff / (np.abs(res_numba[:, :, 1:3]).max() + 1e-12)
        
        print(f"Viscous flux max diff: {max_diff:.2e}, rel diff: {rel_diff:.2e}")
        assert rel_diff < 1e-8, f"Viscous flux mismatch: rel_diff = {rel_diff:.2e}"
    
    def test_viscous_flux_performance(self, large_grid):
        """Benchmark viscous flux computation."""
        NI, NJ, nghost = large_grid
        from src.numerics.gradients import compute_gradients, GradientMetrics
        from src.numerics.viscous_fluxes import compute_viscous_fluxes
        from src.numerics.viscous_fluxes import compute_viscous_fluxes_jax
        
        Q = create_test_state(NI, NJ, nghost)
        Si_x, Si_y, Sj_x, Sj_y, volume = create_test_metrics(NI, NJ)
        mu_laminar = 1e-3
        
        metrics = GradientMetrics(Si_x, Si_y, Sj_x, Sj_y, volume)
        grad = compute_gradients(Q, metrics)
        mu_eff = np.full((NI, NJ), mu_laminar)
        
        grad_jax = jnp.array(grad)
        Si_x_jax, Si_y_jax = jnp.array(Si_x), jnp.array(Si_y)
        Sj_x_jax, Sj_y_jax = jnp.array(Sj_x), jnp.array(Sj_y)
        mu_eff_jax = jnp.array(mu_eff)
        
        # Warm up
        _ = compute_viscous_fluxes(Q, grad, metrics, mu_laminar)
        _ = compute_viscous_fluxes_jax(grad_jax, Si_x_jax, Si_y_jax, Sj_x_jax, Sj_y_jax, mu_eff_jax)
        
        n_iter = 50
        
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = compute_viscous_fluxes(Q, grad, metrics, mu_laminar)
        t_numba = (time.perf_counter() - t0) / n_iter * 1000
        
        t0 = time.perf_counter()
        for _ in range(n_iter):
            res = compute_viscous_fluxes_jax(grad_jax, Si_x_jax, Si_y_jax, Sj_x_jax, Sj_y_jax, mu_eff_jax)
            jax.block_until_ready(res)
        t_jax = (time.perf_counter() - t0) / n_iter * 1000
        
        print(f"\nViscous flux computation ({NI}x{NJ}):")
        print(f"  Numba: {t_numba:.3f} ms")
        print(f"  JAX:   {t_jax:.3f} ms")
        print(f"  Speedup: {t_numba/t_jax:.2f}x")


# =============================================================================
# IRS Tests
# =============================================================================

class TestIRSComparison:
    """Compare JAX and Numba Implicit Residual Smoothing."""
    
    def test_irs_numerical_equivalence(self, medium_grid):
        """JAX and Numba IRS should match."""
        NI, NJ, _ = medium_grid
        from src.numerics.smoothing import apply_residual_smoothing
        from src.numerics.smoothing import apply_residual_smoothing_jax
        
        np.random.seed(42)
        residual = np.random.randn(NI, NJ, 4)
        epsilon = 0.5
        
        # Numba version (in-place)
        res_numba = residual.copy()
        apply_residual_smoothing(res_numba, epsilon)
        
        # JAX version
        res_jax = apply_residual_smoothing_jax(jnp.array(residual), epsilon)
        res_jax_np = np.array(res_jax)
        
        max_diff = np.abs(res_numba - res_jax_np).max()
        rel_diff = max_diff / (np.abs(res_numba).max() + 1e-12)
        
        print(f"IRS max diff: {max_diff:.2e}, rel diff: {rel_diff:.2e}")
        assert rel_diff < 1e-10, f"IRS mismatch: rel_diff = {rel_diff:.2e}"
    
    def test_irs_different_epsilon(self, small_grid):
        """Test IRS with different epsilon values."""
        NI, NJ, _ = small_grid
        from src.numerics.smoothing import apply_residual_smoothing
        from src.numerics.smoothing import apply_residual_smoothing_jax
        
        np.random.seed(42)
        residual = np.random.randn(NI, NJ, 4)
        
        for epsilon in [0.1, 0.5, 1.0, 2.0]:
            res_numba = residual.copy()
            apply_residual_smoothing(res_numba, epsilon)
            
            res_jax = apply_residual_smoothing_jax(jnp.array(residual), epsilon)
            res_jax_np = np.array(res_jax)
            
            max_diff = np.abs(res_numba - res_jax_np).max()
            rel_diff = max_diff / (np.abs(res_numba).max() + 1e-12)
            
            assert rel_diff < 1e-10, f"IRS mismatch at epsilon={epsilon}: {rel_diff:.2e}"
        
        print("IRS matches for all epsilon values")
    
    def test_irs_performance(self, large_grid):
        """Benchmark IRS computation."""
        NI, NJ, _ = large_grid
        from src.numerics.smoothing import apply_residual_smoothing
        from src.numerics.smoothing import apply_residual_smoothing_jax
        
        np.random.seed(42)
        residual = np.random.randn(NI, NJ, 4)
        epsilon = 0.5
        
        res_jax_input = jnp.array(residual)
        
        # Warm up
        res_numba = residual.copy()
        apply_residual_smoothing(res_numba, epsilon)
        _ = apply_residual_smoothing_jax(res_jax_input, epsilon)
        
        n_iter = 50
        
        t0 = time.perf_counter()
        for _ in range(n_iter):
            res_numba = residual.copy()
            apply_residual_smoothing(res_numba, epsilon)
        t_numba = (time.perf_counter() - t0) / n_iter * 1000
        
        t0 = time.perf_counter()
        for _ in range(n_iter):
            res = apply_residual_smoothing_jax(res_jax_input, epsilon)
            jax.block_until_ready(res)
        t_jax = (time.perf_counter() - t0) / n_iter * 1000
        
        print(f"\nIRS computation ({NI}x{NJ}):")
        print(f"  Numba: {t_numba:.3f} ms")
        print(f"  JAX:   {t_jax:.3f} ms")
        print(f"  Speedup: {t_numba/t_jax:.2f}x")


# =============================================================================
# Combined Performance Summary
# =============================================================================

class TestPerformanceSummary:
    """Combined performance test for all kernels."""
    
    def test_all_kernels_performance(self, large_grid):
        """Benchmark all numerical kernels together."""
        NI, NJ, nghost = large_grid
        
        from src.numerics.fluxes import compute_fluxes, FluxConfig, GridMetrics
        from src.numerics.fluxes import compute_fluxes_jax
        from src.numerics.gradients import compute_gradients, GradientMetrics
        from src.numerics.gradients import compute_gradients_jax
        from src.numerics.viscous_fluxes import compute_viscous_fluxes
        from src.numerics.viscous_fluxes import compute_viscous_fluxes_jax
        from src.numerics.smoothing import apply_residual_smoothing
        from src.numerics.smoothing import apply_residual_smoothing_jax
        
        Q = create_test_state(NI, NJ, nghost)
        Si_x, Si_y, Sj_x, Sj_y, volume = create_test_metrics(NI, NJ)
        beta = 10.0
        k4 = 0.016
        mu_laminar = 1e-3
        epsilon = 0.5
        
        flux_metrics = GridMetrics(Si_x, Si_y, Sj_x, Sj_y, volume)
        grad_metrics = GradientMetrics(Si_x, Si_y, Sj_x, Sj_y, volume)
        cfg = FluxConfig(k4=k4)
        
        # JAX arrays
        Q_jax = jnp.array(Q)
        Si_x_jax, Si_y_jax = jnp.array(Si_x), jnp.array(Si_y)
        Sj_x_jax, Sj_y_jax = jnp.array(Sj_x), jnp.array(Sj_y)
        vol_jax = jnp.array(volume)
        
        n_iter = 20
        
        print(f"\n{'='*60}")
        print(f"Performance Summary ({NI}x{NJ} grid)")
        print(f"{'='*60}")
        
        results = []
        
        # Flux
        _ = compute_fluxes(Q, flux_metrics, beta, cfg)
        _ = compute_fluxes_jax(Q_jax, Si_x_jax, Si_y_jax, Sj_x_jax, Sj_y_jax, beta, k4, nghost)
        
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = compute_fluxes(Q, flux_metrics, beta, cfg)
        t_numba = (time.perf_counter() - t0) / n_iter * 1000
        
        t0 = time.perf_counter()
        for _ in range(n_iter):
            res = compute_fluxes_jax(Q_jax, Si_x_jax, Si_y_jax, Sj_x_jax, Sj_y_jax, beta, k4, nghost)
            jax.block_until_ready(res)
        t_jax = (time.perf_counter() - t0) / n_iter * 1000
        results.append(("Flux", t_numba, t_jax))
        
        # Gradient
        grad = compute_gradients(Q, grad_metrics)
        _ = compute_gradients_jax(Q_jax, Si_x_jax, Si_y_jax, Sj_x_jax, Sj_y_jax, vol_jax, nghost)
        
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = compute_gradients(Q, grad_metrics)
        t_numba = (time.perf_counter() - t0) / n_iter * 1000
        
        t0 = time.perf_counter()
        for _ in range(n_iter):
            res = compute_gradients_jax(Q_jax, Si_x_jax, Si_y_jax, Sj_x_jax, Sj_y_jax, vol_jax, nghost)
            jax.block_until_ready(res)
        t_jax = (time.perf_counter() - t0) / n_iter * 1000
        results.append(("Gradient", t_numba, t_jax))
        
        # Viscous
        mu_eff = np.full((NI, NJ), mu_laminar)
        mu_eff_jax = jnp.array(mu_eff)
        grad_jax = jnp.array(grad)
        
        _ = compute_viscous_fluxes(Q, grad, grad_metrics, mu_laminar)
        _ = compute_viscous_fluxes_jax(grad_jax, Si_x_jax, Si_y_jax, Sj_x_jax, Sj_y_jax, mu_eff_jax)
        
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = compute_viscous_fluxes(Q, grad, grad_metrics, mu_laminar)
        t_numba = (time.perf_counter() - t0) / n_iter * 1000
        
        t0 = time.perf_counter()
        for _ in range(n_iter):
            res = compute_viscous_fluxes_jax(grad_jax, Si_x_jax, Si_y_jax, Sj_x_jax, Sj_y_jax, mu_eff_jax)
            jax.block_until_ready(res)
        t_jax = (time.perf_counter() - t0) / n_iter * 1000
        results.append(("Viscous", t_numba, t_jax))
        
        # IRS
        residual = np.random.randn(NI, NJ, 4)
        res_jax_input = jnp.array(residual)
        
        res_numba = residual.copy()
        apply_residual_smoothing(res_numba, epsilon)
        _ = apply_residual_smoothing_jax(res_jax_input, epsilon)
        
        t0 = time.perf_counter()
        for _ in range(n_iter):
            res_numba = residual.copy()
            apply_residual_smoothing(res_numba, epsilon)
        t_numba = (time.perf_counter() - t0) / n_iter * 1000
        
        t0 = time.perf_counter()
        for _ in range(n_iter):
            res = apply_residual_smoothing_jax(res_jax_input, epsilon)
            jax.block_until_ready(res)
        t_jax = (time.perf_counter() - t0) / n_iter * 1000
        results.append(("IRS", t_numba, t_jax))
        
        # Print results
        print(f"\n{'Kernel':<12} {'Numba (ms)':<12} {'JAX (ms)':<12} {'Speedup':<10}")
        print("-" * 46)
        total_numba = 0
        total_jax = 0
        for name, t_n, t_j in results:
            speedup = t_n / t_j
            print(f"{name:<12} {t_n:<12.3f} {t_j:<12.3f} {speedup:<10.2f}x")
            total_numba += t_n
            total_jax += t_j
        print("-" * 46)
        print(f"{'TOTAL':<12} {total_numba:<12.3f} {total_jax:<12.3f} {total_numba/total_jax:<10.2f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

