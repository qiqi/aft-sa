"""
Tests for explicit residual smoothing implementations.

Validates:
1. JAX and Numba implementations produce identical results
2. Boundary conditions are correctly applied (Neumann)
3. Performance comparison vs implicit smoothing
"""

import pytest
import numpy as np
import time

from src.numerics.explicit_smoothing import (
    smooth_explicit_numba,
    apply_explicit_smoothing,
)

# JAX imports
try:
    from src.physics.jax_config import jax, jnp
    from src.numerics.explicit_smoothing import (
        smooth_explicit_jax,
        _smooth_pass_i_jax,
        _smooth_pass_j_jax,
    )
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


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
    
    def test_numba_smoothing_reduces_oscillations(self, medium_grid):
        """Smoothing should reduce high-frequency oscillations."""
        NI, NJ = medium_grid
        
        # Create oscillatory residual
        i, j = np.meshgrid(np.arange(NI), np.arange(NJ), indexing='ij')
        R = np.zeros((NI, NJ, 4))
        R[:, :, 0] = np.sin(10 * np.pi * i / NI) * np.sin(10 * np.pi * j / NJ)
        
        R_smooth = smooth_explicit_numba(R, epsilon=0.2, n_passes=3)
        
        # Smoothed residual should have lower amplitude
        assert np.abs(R_smooth).max() < np.abs(R).max()
        print(f"Amplitude reduction: {np.abs(R).max():.3f} -> {np.abs(R_smooth).max():.3f}")
    
    def test_numba_neumann_bc(self, medium_grid):
        """Verify Neumann boundary conditions are applied."""
        NI, NJ = medium_grid
        
        # Constant residual should be unchanged by smoothing
        R = np.ones((NI, NJ, 4)) * 5.0
        R_smooth = smooth_explicit_numba(R, epsilon=0.2, n_passes=3)
        
        max_diff = np.abs(R - R_smooth).max()
        assert max_diff < 1e-14, f"Constant field changed: diff = {max_diff}"
    
    def test_numba_zero_epsilon(self, medium_grid):
        """Zero epsilon should return unchanged residual."""
        NI, NJ = medium_grid
        
        np.random.seed(42)
        R = np.random.randn(NI, NJ, 4)
        
        R_smooth = smooth_explicit_numba(R, epsilon=0.0, n_passes=2)
        
        max_diff = np.abs(R - R_smooth).max()
        assert max_diff < 1e-14, f"Zero epsilon changed field: diff = {max_diff}"


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXNumbAgreement:
    """Test that JAX and Numba implementations agree."""
    
    def test_jax_numba_equivalence(self, medium_grid):
        """JAX and Numba should produce identical results."""
        NI, NJ = medium_grid
        
        np.random.seed(42)
        R = np.random.randn(NI, NJ, 4)
        
        R_numba = smooth_explicit_numba(R, epsilon=0.2, n_passes=2)
        R_jax = smooth_explicit_jax(jnp.array(R), epsilon=0.2, n_passes=2)
        R_jax_np = np.array(R_jax)
        
        max_diff = np.abs(R_numba - R_jax_np).max()
        rel_diff = max_diff / (np.abs(R_numba).max() + 1e-12)
        
        print(f"Max diff: {max_diff:.2e}, Rel diff: {rel_diff:.2e}")
        assert rel_diff < 1e-12, f"JAX/Numba mismatch: {rel_diff:.2e}"
    
    def test_jax_numba_equivalence_various_params(self, medium_grid):
        """Test equivalence for various epsilon and n_passes values."""
        NI, NJ = medium_grid
        
        np.random.seed(42)
        R = np.random.randn(NI, NJ, 4)
        
        for epsilon in [0.1, 0.2, 0.3]:
            for n_passes in [1, 2, 3, 4]:
                R_numba = smooth_explicit_numba(R, epsilon, n_passes)
                R_jax = smooth_explicit_jax(jnp.array(R), epsilon, n_passes)
                R_jax_np = np.array(R_jax)
                
                max_diff = np.abs(R_numba - R_jax_np).max()
                assert max_diff < 1e-12, f"Mismatch at eps={epsilon}, passes={n_passes}"
        
        print("All parameter combinations match")
    
    def test_dispatch_function(self, medium_grid):
        """Test automatic dispatch based on array type."""
        NI, NJ = medium_grid
        
        np.random.seed(42)
        R = np.random.randn(NI, NJ, 4)
        
        # NumPy input -> Numba
        R_np = apply_explicit_smoothing(R, epsilon=0.2, n_passes=2)
        assert isinstance(R_np, np.ndarray)
        
        # JAX input -> JAX
        R_jax = apply_explicit_smoothing(jnp.array(R), epsilon=0.2, n_passes=2)
        R_jax_np = np.array(R_jax)
        
        max_diff = np.abs(R_np - R_jax_np).max()
        assert max_diff < 1e-12


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestPerformanceComparison:
    """Compare explicit vs implicit smoothing performance."""
    
    def test_explicit_vs_implicit_performance(self, large_grid):
        """Benchmark explicit vs implicit smoothing."""
        from src.numerics.smoothing import (
            apply_residual_smoothing,
            apply_residual_smoothing_jax,
        )
        
        NI, NJ = large_grid
        np.random.seed(42)
        R = np.random.randn(NI, NJ, 4)
        R_jax = jnp.array(R)
        
        n_iter = 50
        
        # Warm up
        _ = smooth_explicit_numba(R, 0.2, 2)
        _ = smooth_explicit_jax(R_jax, 0.2, 2)
        jax.block_until_ready(_)
        
        R_temp = R.copy()
        apply_residual_smoothing(R_temp, 0.5)
        _ = apply_residual_smoothing_jax(R_jax, 0.5)
        jax.block_until_ready(_)
        
        # Benchmark explicit Numba
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = smooth_explicit_numba(R, 0.2, 2)
        t_explicit_numba = (time.perf_counter() - t0) / n_iter * 1000
        
        # Benchmark explicit JAX
        t0 = time.perf_counter()
        for _ in range(n_iter):
            res = smooth_explicit_jax(R_jax, 0.2, 2)
            jax.block_until_ready(res)
        t_explicit_jax = (time.perf_counter() - t0) / n_iter * 1000
        
        # Benchmark implicit Numba
        t0 = time.perf_counter()
        for _ in range(n_iter):
            R_temp = R.copy()
            apply_residual_smoothing(R_temp, 0.5)
        t_implicit_numba = (time.perf_counter() - t0) / n_iter * 1000
        
        # Benchmark implicit JAX
        t0 = time.perf_counter()
        for _ in range(n_iter):
            res = apply_residual_smoothing_jax(R_jax, 0.5)
            jax.block_until_ready(res)
        t_implicit_jax = (time.perf_counter() - t0) / n_iter * 1000
        
        print(f"\nSmoothing Performance ({NI}x{NJ}):")
        print(f"  Explicit Numba: {t_explicit_numba:.3f} ms")
        print(f"  Explicit JAX:   {t_explicit_jax:.3f} ms")
        print(f"  Implicit Numba: {t_implicit_numba:.3f} ms")
        print(f"  Implicit JAX:   {t_implicit_jax:.3f} ms")
        print(f"\n  Explicit vs Implicit speedup (Numba): {t_implicit_numba/t_explicit_numba:.2f}x")
        print(f"  Explicit vs Implicit speedup (JAX):   {t_implicit_jax/t_explicit_jax:.2f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

