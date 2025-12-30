#!/usr/bin/env python3
"""
Test Spalart-Allmaras Analytical Gradients.

This script verifies that the hand-coded analytical gradients in the SA model
match PyTorch's automatic differentiation (autograd).

Assertions:
- Production gradient matches autograd within tolerance
- Destruction gradient matches autograd within tolerance
- All gradients are non-negative for positive inputs
"""

import torch
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.physics.spalart_allmaras import spalart_allmaras_amplification, fv1, fv2


def test_gradient_accuracy():
    """Test that analytical gradients match autograd."""
    print("--- Verifying Spalart-Allmaras Analytical Gradients ---")

    # Use double precision for strict numerical checking
    torch.set_default_dtype(torch.float64)

    # 1. Create Inputs - variety of conditions
    dudy = torch.tensor([0.0, 10.0, 100.0, 50.0])
    y = torch.tensor([1.0, 0.1, 0.01, 0.5])
    nuHat = torch.tensor([1e-5, 0.5, 5.0, 20.0], requires_grad=True)

    # 2. Run the Modified Forward Pass (Analytical)
    (prod_val, prod_grad_analytic), (dest_val, dest_grad_analytic) = \
        spalart_allmaras_amplification(dudy, nuHat, y)

    # 3. Compute PyTorch Autograd (Automatic)
    prod_grad_autograd = torch.autograd.grad(
        outputs=prod_val, inputs=nuHat,
        grad_outputs=torch.ones_like(prod_val),
        retain_graph=True
    )[0]

    dest_grad_autograd = torch.autograd.grad(
        outputs=dest_val, inputs=nuHat,
        grad_outputs=torch.ones_like(dest_val),
        retain_graph=False
    )[0]

    # 4. Compare Results
    tol = 1e-8
    prod_diff = (prod_grad_analytic - prod_grad_autograd).abs().max().item()
    dest_diff = (dest_grad_analytic - dest_grad_autograd).abs().max().item()

    print(f"\nProduction Gradient Difference (Max Abs): {prod_diff:.2e}")
    print(f"Destruction Gradient Difference (Max Abs): {dest_diff:.2e}")

    print("\n--- Detailed Dump ---")
    print("NuHat:          ", nuHat.detach().numpy())
    print("Prod Analytic:  ", prod_grad_analytic.detach().numpy())
    print("Prod Autograd:  ", prod_grad_autograd.detach().numpy())
    print("Dest Analytic:  ", dest_grad_analytic.detach().numpy())
    print("Dest Autograd:  ", dest_grad_autograd.detach().numpy())

    # Assertions
    assert prod_diff < tol, f"Production gradient mismatch: {prod_diff:.2e} > {tol}"
    assert dest_diff < tol, f"Destruction gradient mismatch: {dest_diff:.2e} > {tol}"
    
    print("\n✅ Production gradient matches Autograd.")
    print("✅ Destruction gradient matches Autograd.")
    
    return True


def test_physical_constraints():
    """Test that SA functions satisfy physical constraints."""
    print("\n--- Verifying Physical Constraints ---")
    
    torch.set_default_dtype(torch.float64)
    
    # Test fv1: should be in [0, 1] and monotonically increasing
    nuHat = torch.linspace(0.0, 100.0, 101)
    fv1_val, fv1_grad = fv1(nuHat)
    
    assert (fv1_val >= 0).all(), "fv1 should be non-negative"
    assert (fv1_val <= 1).all(), "fv1 should be <= 1"
    assert (fv1_grad >= 0).all(), "fv1 gradient should be non-negative (monotonic)"
    print("✅ fv1 ∈ [0, 1] and monotonically increasing")
    
    # Test fv1 limits
    fv1_at_zero, _ = fv1(torch.tensor([0.0]))
    fv1_at_large, _ = fv1(torch.tensor([1000.0]))
    assert fv1_at_zero.item() < 1e-10, "fv1(0) should be ~0"
    assert fv1_at_large.item() > 0.99, "fv1(∞) should approach 1"
    print("✅ fv1(0) → 0, fv1(∞) → 1")
    
    # Test fv2: should be <= 1, decreasing initially then can increase
    # (fv2 is NOT monotonic - it goes negative then back up for large chi)
    fv2_val, fv2_grad = fv2(nuHat)
    assert (fv2_val <= 1.001).all(), "fv2 should be <= 1"
    print("✅ fv2 ≤ 1")
    
    # Test fv2 limits
    fv2_at_zero, _ = fv2(torch.tensor([0.0]))
    assert abs(fv2_at_zero.item() - 1.0) < 1e-10, "fv2(0) should be 1"
    print("✅ fv2(0) = 1")
    
    # Test fv2 is decreasing for small chi (laminar regime)
    fv2_small, fv2_grad_small = fv2(torch.linspace(0.0, 1.0, 11))
    assert (fv2_grad_small <= 0).all(), "fv2 should be decreasing for small chi"
    print("✅ fv2 decreasing for χ < 1 (laminar regime)")
    
    # Test production/destruction positivity for positive inputs
    dudy = torch.tensor([10.0, 50.0, 100.0])
    y = torch.tensor([0.1, 0.5, 1.0])
    nuHat_pos = torch.tensor([0.1, 1.0, 10.0])
    
    (prod, _), (dest, _) = spalart_allmaras_amplification(dudy, nuHat_pos, y)
    
    assert (prod >= 0).all(), "Production should be non-negative"
    assert (dest >= 0).all(), "Destruction should be non-negative"
    print("✅ Production ≥ 0, Destruction ≥ 0 for positive inputs")
    
    return True


def test_dimension_agnostic():
    """Test that SA functions work with different tensor shapes."""
    print("\n--- Verifying Dimension-Agnostic Behavior ---")
    
    torch.set_default_dtype(torch.float64)
    
    # Test 1D
    dudy_1d = torch.tensor([10.0, 20.0, 30.0])
    nuHat_1d = torch.tensor([0.5, 1.0, 2.0])
    y_1d = torch.tensor([0.1, 0.2, 0.3])
    (prod_1d, _), (dest_1d, _) = spalart_allmaras_amplification(dudy_1d, nuHat_1d, y_1d)
    assert prod_1d.shape == (3,), "1D input should give 1D output"
    print("✅ 1D inputs work")
    
    # Test 2D
    dudy_2d = torch.rand(5, 7)
    nuHat_2d = torch.rand(5, 7) + 0.1
    y_2d = torch.rand(5, 7) + 0.01
    (prod_2d, _), (dest_2d, _) = spalart_allmaras_amplification(dudy_2d, nuHat_2d, y_2d)
    assert prod_2d.shape == (5, 7), "2D input should give 2D output"
    print("✅ 2D inputs work")
    
    # Test scalar (0D)
    dudy_0d = torch.tensor(10.0)
    nuHat_0d = torch.tensor(0.5)
    y_0d = torch.tensor(0.1)
    (prod_0d, _), (dest_0d, _) = spalart_allmaras_amplification(dudy_0d, nuHat_0d, y_0d)
    assert prod_0d.shape == (), "Scalar input should give scalar output"
    print("✅ Scalar inputs work")
    
    return True


def run():
    """Run all tests."""
    all_passed = True
    
    try:
        test_gradient_accuracy()
    except AssertionError as e:
        print(f"❌ FAILED: {e}")
        all_passed = False
    
    try:
        test_physical_constraints()
    except AssertionError as e:
        print(f"❌ FAILED: {e}")
        all_passed = False
    
    try:
        test_dimension_agnostic()
    except AssertionError as e:
        print(f"❌ FAILED: {e}")
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("All tests PASSED ✅")
        return 0
    else:
        print("Some tests FAILED ❌")
        return 1


if __name__ == "__main__":
    exit(run())
