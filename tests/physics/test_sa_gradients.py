"""
Pytest tests for Spalart-Allmaras analytical gradients.

Tests verify that hand-coded analytical gradients match PyTorch autograd.
"""

import pytest
import torch
import numpy as np

from src.physics.spalart_allmaras import spalart_allmaras_amplification, fv1, fv2


class TestSAGradientAccuracy:
    """Test that analytical gradients match autograd."""
    
    def test_production_gradient(self):
        """Production gradient should match autograd within tolerance."""
        torch.set_default_dtype(torch.float64)
        
        dudy = torch.tensor([0.0, 10.0, 100.0, 50.0])
        y = torch.tensor([1.0, 0.1, 0.01, 0.5])
        nuHat = torch.tensor([1e-5, 0.5, 5.0, 20.0], requires_grad=True)
        
        (prod_val, prod_grad_analytic), _ = spalart_allmaras_amplification(dudy, nuHat, y)
        
        prod_grad_autograd = torch.autograd.grad(
            outputs=prod_val, inputs=nuHat,
            grad_outputs=torch.ones_like(prod_val),
            retain_graph=True
        )[0]
        
        diff = (prod_grad_analytic - prod_grad_autograd).abs().max().item()
        assert diff < 1e-8, f"Production gradient mismatch: {diff:.2e}"
    
    def test_destruction_gradient(self):
        """Destruction gradient should match autograd within tolerance."""
        torch.set_default_dtype(torch.float64)
        
        dudy = torch.tensor([0.0, 10.0, 100.0, 50.0])
        y = torch.tensor([1.0, 0.1, 0.01, 0.5])
        nuHat = torch.tensor([1e-5, 0.5, 5.0, 20.0], requires_grad=True)
        
        _, (dest_val, dest_grad_analytic) = spalart_allmaras_amplification(dudy, nuHat, y)
        
        dest_grad_autograd = torch.autograd.grad(
            outputs=dest_val, inputs=nuHat,
            grad_outputs=torch.ones_like(dest_val)
        )[0]
        
        diff = (dest_grad_analytic - dest_grad_autograd).abs().max().item()
        assert diff < 1e-8, f"Destruction gradient mismatch: {diff:.2e}"


class TestSAPhysicalConstraints:
    """Test that SA functions satisfy physical constraints."""
    
    def test_fv1_bounds(self):
        """fv1 should be in [0, 1] and monotonically increasing."""
        torch.set_default_dtype(torch.float64)
        
        nuHat = torch.linspace(0.0, 100.0, 101)
        fv1_val, fv1_grad = fv1(nuHat)
        
        assert (fv1_val >= 0).all(), "fv1 should be non-negative"
        assert (fv1_val <= 1).all(), "fv1 should be <= 1"
        assert (fv1_grad >= 0).all(), "fv1 gradient should be non-negative"
    
    def test_fv1_limits(self):
        """fv1(0) → 0, fv1(∞) → 1."""
        torch.set_default_dtype(torch.float64)
        
        fv1_zero, _ = fv1(torch.tensor([0.0]))
        fv1_large, _ = fv1(torch.tensor([1000.0]))
        
        assert fv1_zero.item() < 1e-10, "fv1(0) should be ~0"
        assert fv1_large.item() > 0.99, "fv1(∞) should approach 1"
    
    def test_fv2_bounds(self):
        """fv2 should be <= 1."""
        torch.set_default_dtype(torch.float64)
        
        nuHat = torch.linspace(0.0, 100.0, 101)
        fv2_val, _ = fv2(nuHat)
        
        assert (fv2_val <= 1.001).all(), "fv2 should be <= 1"
    
    def test_fv2_at_zero(self):
        """fv2(0) = 1."""
        torch.set_default_dtype(torch.float64)
        
        fv2_zero, _ = fv2(torch.tensor([0.0]))
        assert abs(fv2_zero.item() - 1.0) < 1e-10, "fv2(0) should be 1"
    
    def test_fv2_decreasing_laminar(self):
        """fv2 should be decreasing for small chi (laminar regime)."""
        torch.set_default_dtype(torch.float64)
        
        _, fv2_grad = fv2(torch.linspace(0.0, 1.0, 11))
        assert (fv2_grad <= 0).all(), "fv2 should decrease for χ < 1"
    
    def test_production_destruction_positive(self):
        """Production and destruction should be non-negative for positive inputs."""
        torch.set_default_dtype(torch.float64)
        
        dudy = torch.tensor([10.0, 50.0, 100.0])
        y = torch.tensor([0.1, 0.5, 1.0])
        nuHat = torch.tensor([0.1, 1.0, 10.0])
        
        (prod, _), (dest, _) = spalart_allmaras_amplification(dudy, nuHat, y)
        
        assert (prod >= 0).all(), "Production should be non-negative"
        assert (dest >= 0).all(), "Destruction should be non-negative"


class TestSADimensionAgnostic:
    """Test SA functions work with different tensor shapes."""
    
    def test_1d_input(self):
        """1D inputs should work."""
        torch.set_default_dtype(torch.float64)
        
        dudy = torch.tensor([10.0, 20.0, 30.0])
        nuHat = torch.tensor([0.5, 1.0, 2.0])
        y = torch.tensor([0.1, 0.2, 0.3])
        
        (prod, _), _ = spalart_allmaras_amplification(dudy, nuHat, y)
        assert prod.shape == (3,)
    
    def test_2d_input(self):
        """2D inputs should work."""
        torch.set_default_dtype(torch.float64)
        
        dudy = torch.rand(5, 7)
        nuHat = torch.rand(5, 7) + 0.1
        y = torch.rand(5, 7) + 0.01
        
        (prod, _), _ = spalart_allmaras_amplification(dudy, nuHat, y)
        assert prod.shape == (5, 7)
    
    def test_scalar_input(self):
        """Scalar inputs should work."""
        torch.set_default_dtype(torch.float64)
        
        (prod, _), _ = spalart_allmaras_amplification(
            torch.tensor(10.0), torch.tensor(0.5), torch.tensor(0.1)
        )
        assert prod.shape == ()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

