"""
Pytest tests for Spalart-Allmaras analytical gradients.

Tests verify that hand-coded analytical gradients match JAX autograd.
"""

import pytest
import numpy as np

from src.physics.jax_config import jax, jnp
from src.physics.spalart_allmaras import (
    spalart_allmaras_amplification, fv1, fv1_value, fv2
)


class TestSAGradientAccuracy:
    """Test that analytical gradients match JAX autograd."""
    
    def test_production_gradient(self):
        """Production gradient should match autograd within tolerance."""
        dudy = jnp.array([0.1, 10.0, 100.0, 50.0])  # Avoid zero for stability
        y = jnp.array([1.0, 0.1, 0.01, 0.5])
        nuHat = jnp.array([1e-5, 0.5, 5.0, 20.0])
        
        # Analytical gradient
        (prod_val, prod_grad_analytic), _ = spalart_allmaras_amplification(dudy, nuHat, y)
        
        # Autograd: sum over production values for scalar output
        def prod_sum(nuHat_):
            (prod, _), _ = spalart_allmaras_amplification(dudy, nuHat_, y)
            return jnp.sum(prod)
        
        prod_grad_autograd = jax.grad(prod_sum)(nuHat)
        
        diff = jnp.abs(prod_grad_analytic - prod_grad_autograd).max()
        assert float(diff) < 1e-8, f"Production gradient mismatch: {diff:.2e}"
    
    def test_destruction_gradient(self):
        """Destruction gradient should match autograd within tolerance."""
        dudy = jnp.array([0.1, 10.0, 100.0, 50.0])  # Avoid zero for stability
        y = jnp.array([1.0, 0.1, 0.01, 0.5])
        nuHat = jnp.array([1e-5, 0.5, 5.0, 20.0])
        
        # Analytical gradient
        _, (dest_val, dest_grad_analytic) = spalart_allmaras_amplification(dudy, nuHat, y)
        
        # Autograd: sum over destruction values for scalar output
        def dest_sum(nuHat_):
            _, (dest, _) = spalart_allmaras_amplification(dudy, nuHat_, y)
            return jnp.sum(dest)
        
        dest_grad_autograd = jax.grad(dest_sum)(nuHat)
        
        diff = jnp.abs(dest_grad_analytic - dest_grad_autograd).max()
        assert float(diff) < 1e-8, f"Destruction gradient mismatch: {diff:.2e}"


class TestSAPhysicalConstraints:
    """Test that SA functions satisfy physical constraints."""
    
    def test_fv1_bounds(self):
        """fv1 should be in [0, 1] and monotonically increasing."""
        nuHat = jnp.linspace(0.0, 100.0, 101)
        fv1_val, fv1_grad = fv1(nuHat)
        
        assert (fv1_val >= 0).all(), "fv1 should be non-negative"
        assert (fv1_val <= 1).all(), "fv1 should be <= 1"
        assert (fv1_grad >= 0).all(), "fv1 gradient should be non-negative"
    
    def test_fv1_limits(self):
        """fv1(0) → 0, fv1(∞) → 1."""
        fv1_zero, _ = fv1(jnp.array([0.0]))
        fv1_large, _ = fv1(jnp.array([1000.0]))
        
        assert float(fv1_zero[0]) < 1e-10, "fv1(0) should be ~0"
        assert float(fv1_large[0]) > 0.99, "fv1(∞) should approach 1"
    
    def test_fv2_bounds(self):
        """fv2 should be <= 1."""
        nuHat = jnp.linspace(0.0, 100.0, 101)
        fv2_val, _ = fv2(nuHat)
        
        assert (fv2_val <= 1.001).all(), "fv2 should be <= 1"
    
    def test_fv2_at_zero(self):
        """fv2(0) = 1."""
        fv2_zero, _ = fv2(jnp.array([0.0]))
        assert abs(float(fv2_zero[0]) - 1.0) < 1e-10, "fv2(0) should be 1"
    
    def test_fv2_decreasing_laminar(self):
        """fv2 should be decreasing for small chi (laminar regime)."""
        _, fv2_grad = fv2(jnp.linspace(0.0, 1.0, 11))
        assert (fv2_grad <= 0).all(), "fv2 should decrease for χ < 1"
    
    def test_production_destruction_positive(self):
        """Production and destruction should be non-negative for positive inputs."""
        dudy = jnp.array([10.0, 50.0, 100.0])
        y = jnp.array([0.1, 0.5, 1.0])
        nuHat = jnp.array([0.1, 1.0, 10.0])
        
        (prod, _), (dest, _) = spalart_allmaras_amplification(dudy, nuHat, y)
        
        assert (prod >= 0).all(), "Production should be non-negative"
        assert (dest >= 0).all(), "Destruction should be non-negative"


class TestSADimensionAgnostic:
    """Test SA functions work with different array shapes."""
    
    def test_1d_input(self):
        """1D inputs should work."""
        dudy = jnp.array([10.0, 20.0, 30.0])
        nuHat = jnp.array([0.5, 1.0, 2.0])
        y = jnp.array([0.1, 0.2, 0.3])
        
        (prod, _), _ = spalart_allmaras_amplification(dudy, nuHat, y)
        assert prod.shape == (3,)
    
    def test_2d_input(self):
        """2D inputs should work."""
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        dudy = jax.random.uniform(k1, (5, 7))
        nuHat = jax.random.uniform(k2, (5, 7)) + 0.1
        y = jax.random.uniform(k3, (5, 7)) + 0.01
        
        (prod, _), _ = spalart_allmaras_amplification(dudy, nuHat, y)
        assert prod.shape == (5, 7)
    
    def test_scalar_input(self):
        """Scalar inputs should work."""
        (prod, _), _ = spalart_allmaras_amplification(
            jnp.array(10.0), jnp.array(0.5), jnp.array(0.1)
        )
        assert prod.shape == ()


class TestJAXAutograd:
    """Test JAX autograd consistency with analytical gradients."""
    
    def test_fv1_autograd(self):
        """fv1 analytical gradient should match JAX autograd."""
        nuHat = jnp.array([0.1, 1.0, 5.0, 10.0, 50.0])
        
        # Analytical
        _, fv1_grad_analytic = fv1(nuHat)
        
        # Autograd using value-only function
        fv1_grad_auto = jax.grad(lambda x: jnp.sum(fv1_value(x)))(nuHat)
        
        diff = jnp.abs(fv1_grad_analytic - fv1_grad_auto).max()
        assert float(diff) < 1e-12, f"fv1 gradient mismatch: {diff:.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
