"""
Tests for explicit residual smoothing implementations.

Validates:
1. JAX implementation produces correct results
2. Boundary conditions are correctly applied (Neumann)
3. Performance of explicit smoothing
"""

import pytest
import numpy as np
import time

from src.physics.jax_config import jax, jnp
from src.numerics.explicit_smoothing import (
    smooth_explicit_jax,
    apply_explicit_smoothing,
    _smooth_pass_i_jax,
    _smooth_pass_j_jax,
)


@pytest.fixture
def medium_grid():
    """Medium grid for validation."""
    NI, NJ = 64, 48
    return NI, NJ


@pytest.fixture
def large_grid():
    """Large grid for performance tests."""
    NI, NJ = 256, 128
    return NI, NJ


class TestExplicitSmoothingNumerics:
    """Test numerical correctness of explicit smoothing."""
    
    def test_smoothing_reduces_oscillations(self, medium_grid):
        """Smoothing should reduce high-frequency oscillations."""
        NI, NJ = medium_grid
        
        # Create oscillatory residual
        i, j = np.meshgrid(np.arange(NI), np.arange(NJ), indexing='ij')
        R = np.zeros((NI, NJ, 4))
        R[:, :, 0] = np.sin(10 * np.pi * i / NI) * np.sin(10 * np.pi * j / NJ)
        
        R_jax = jnp.array(R)
        R_smooth = smooth_explicit_jax(R_jax, epsilon=0.2, n_passes=3)
        R_smooth_np = np.array(R_smooth)
        
        # Smoothed residual should have lower amplitude
        assert np.abs(R_smooth_np).max() < np.abs(R).max()
        print(f"Amplitude reduction: {np.abs(R).max():.3f} -> {np.abs(R_smooth_np).max():.3f}")
    
    def test_neumann_bc(self, medium_grid):
        """Verify Neumann boundary conditions are applied."""
        NI, NJ = medium_grid
        
        # Constant residual should be unchanged by smoothing
        R = np.ones((NI, NJ, 4)) * 5.0
        R_jax = jnp.array(R)
        R_smooth = smooth_explicit_jax(R_jax, epsilon=0.2, n_passes=3)
        R_smooth_np = np.array(R_smooth)
        
        max_diff = np.abs(R - R_smooth_np).max()
        assert max_diff < 1e-14, f"Constant field changed: diff = {max_diff}"
    
    def test_zero_epsilon(self, medium_grid):
        """Zero epsilon should return unchanged residual."""
        NI, NJ = medium_grid
        
        np.random.seed(42)
        R = np.random.randn(NI, NJ, 4)
        R_jax = jnp.array(R)
        
        R_smooth = smooth_explicit_jax(R_jax, epsilon=0.0, n_passes=2)
        R_smooth_np = np.array(R_smooth)
        
        max_diff = np.abs(R - R_smooth_np).max()
        assert max_diff < 1e-14, f"Zero epsilon changed field: diff = {max_diff}"


class TestDispatch:
    """Test dispatch function."""
    
    def test_dispatch_numpy_input(self, medium_grid):
        """Test automatic dispatch with NumPy input."""
        NI, NJ = medium_grid
        
        np.random.seed(42)
        R = np.random.randn(NI, NJ, 4)
        
        # NumPy input returns NumPy
        R_out = apply_explicit_smoothing(R, epsilon=0.2, n_passes=2)
        assert isinstance(R_out, np.ndarray)
    
    def test_dispatch_jax_input(self, medium_grid):
        """Test automatic dispatch with JAX input."""
        NI, NJ = medium_grid
        
        np.random.seed(42)
        R = np.random.randn(NI, NJ, 4)
        R_jax = jnp.array(R)
        
        # JAX input returns JAX
        R_out = apply_explicit_smoothing(R_jax, epsilon=0.2, n_passes=2)
        
        # Results should be same
        R_out_np = apply_explicit_smoothing(R, epsilon=0.2, n_passes=2)
        max_diff = np.abs(R_out_np - np.array(R_out)).max()
        assert max_diff < 1e-12


class TestPerformance:
    """Compare explicit vs implicit smoothing performance."""
    
    def test_explicit_vs_implicit_performance(self, large_grid):
        """Benchmark explicit vs implicit smoothing."""
        from src.numerics.smoothing import apply_residual_smoothing_jax
        
        NI, NJ = large_grid
        np.random.seed(42)
        R = np.random.randn(NI, NJ, 4)
        R_jax = jnp.array(R)
        
        n_iter = 50
        
        # Warm up
        _ = smooth_explicit_jax(R_jax, 0.2, 2)
        jax.block_until_ready(_)
        
        _ = apply_residual_smoothing_jax(R_jax, 0.5)
        jax.block_until_ready(_)
        
        # Benchmark explicit JAX
        t0 = time.perf_counter()
        for _ in range(n_iter):
            res = smooth_explicit_jax(R_jax, 0.2, 2)
            jax.block_until_ready(res)
        t_explicit_jax = (time.perf_counter() - t0) / n_iter * 1000
        
        # Benchmark implicit JAX
        t0 = time.perf_counter()
        for _ in range(n_iter):
            res = apply_residual_smoothing_jax(R_jax, 0.5)
            jax.block_until_ready(res)
        t_implicit_jax = (time.perf_counter() - t0) / n_iter * 1000
        
        print(f"\nSmoothing Performance ({NI}x{NJ}):")
        print(f"  Explicit JAX:   {t_explicit_jax:.3f} ms")
        print(f"  Implicit JAX:   {t_implicit_jax:.3f} ms")
        print(f"\n  Explicit vs Implicit speedup: {t_implicit_jax/t_explicit_jax:.2f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
