"""
Pytest tests for Green-Gauss gradient reconstruction.

These tests verify:
1. Exactness for linear/quadratic fields
2. 2nd order convergence for higher-order fields
3. Correctness of vorticity and strain rate
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.numerics.gradients import (
    compute_gradients, compute_vorticity, compute_strain_rate, GradientMetrics
)


def create_uniform_grid(NI: int, NJ: int, Lx: float = 1.0, Ly: float = 1.0):
    """Create uniform Cartesian grid with metrics."""
    dx = Lx / NI
    dy = Ly / NJ
    
    x = np.linspace(-dx/2, Lx + dx/2, NI + 2)
    y = np.linspace(-dy/2, Ly + dy/2, NJ + 2)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    Si_x = np.ones((NI + 1, NJ)) * dy
    Si_y = np.zeros((NI + 1, NJ))
    Sj_x = np.zeros((NI, NJ + 1))
    Sj_y = np.ones((NI, NJ + 1)) * dx
    volume = np.ones((NI, NJ)) * dx * dy
    
    metrics = GradientMetrics(Si_x, Si_y, Sj_x, Sj_y, volume)
    return X, Y, metrics


class TestLinearExactness:
    """Test that linear fields give exact gradients."""
    
    @pytest.mark.parametrize("NI,NJ", [(8, 8), (16, 16), (10, 20)])
    def test_linear_field(self, NI, NJ):
        """Linear field u = 2x + 3y + 1 should give exact gradients."""
        a, b, c = 2.0, 3.0, 1.0
        X, Y, metrics = create_uniform_grid(NI, NJ)
        
        Q = np.zeros((NI + 2, NJ + 2, 4))
        Q[:, :, 1] = a * X + b * Y + c
        
        grad = compute_gradients(Q, metrics)
        
        # Check exact gradients
        assert np.allclose(grad[:, :, 1, 0], a, atol=1e-12), "du/dx should equal a"
        assert np.allclose(grad[:, :, 1, 1], b, atol=1e-12), "du/dy should equal b"
    
    @pytest.mark.parametrize("NI,NJ", [(8, 8), (16, 16)])
    def test_quadratic_field(self, NI, NJ):
        """Quadratic field u = x² + y² should give exact gradients on uniform grid."""
        X, Y, metrics = create_uniform_grid(NI, NJ)
        
        Q = np.zeros((NI + 2, NJ + 2, 4))
        Q[:, :, 1] = X**2 + Y**2
        
        grad = compute_gradients(Q, metrics)
        
        X_int = X[1:-1, 1:-1]
        Y_int = Y[1:-1, 1:-1]
        
        # Green-Gauss is exact for quadratic on uniform grid
        assert np.allclose(grad[:, :, 1, 0], 2 * X_int, atol=1e-12), "du/dx should equal 2x"
        assert np.allclose(grad[:, :, 1, 1], 2 * Y_int, atol=1e-12), "du/dy should equal 2y"


class TestConvergenceOrder:
    """Test 2nd order convergence for higher-order fields."""
    
    def compute_error(self, field_func, grad_x_func, grad_y_func, NI, NJ):
        """Compute gradient error for a given field."""
        X, Y, metrics = create_uniform_grid(NI, NJ)
        
        Q = np.zeros((NI + 2, NJ + 2, 4))
        Q[:, :, 1] = field_func(X, Y)
        
        grad = compute_gradients(Q, metrics)
        
        X_int = X[1:-1, 1:-1]
        Y_int = Y[1:-1, 1:-1]
        
        err_x = np.sqrt(np.mean((grad[:, :, 1, 0] - grad_x_func(X_int, Y_int))**2))
        err_y = np.sqrt(np.mean((grad[:, :, 1, 1] - grad_y_func(X_int, Y_int))**2))
        
        return max(err_x, err_y)
    
    def test_cubic_convergence(self):
        """Cubic field u = x³ + y³ should show 2nd order convergence."""
        field = lambda X, Y: X**3 + Y**3
        grad_x = lambda X, Y: 3 * X**2
        grad_y = lambda X, Y: 3 * Y**2
        
        errors = []
        sizes = [16, 32, 64]
        
        for N in sizes:
            err = self.compute_error(field, grad_x, grad_y, N, N)
            errors.append(err)
        
        # Compute convergence orders
        for i in range(1, len(sizes)):
            order = np.log(errors[i-1] / errors[i]) / np.log(2)
            assert order > 1.9, f"Expected 2nd order convergence, got {order:.2f}"
    
    def test_sinusoidal_convergence(self):
        """Sinusoidal field should show 2nd order convergence."""
        k = 2 * np.pi
        field = lambda X, Y: np.sin(k * X) * np.cos(k * Y)
        grad_x = lambda X, Y: k * np.cos(k * X) * np.cos(k * Y)
        grad_y = lambda X, Y: -k * np.sin(k * X) * np.sin(k * Y)
        
        errors = []
        sizes = [16, 32, 64]
        
        for N in sizes:
            err = self.compute_error(field, grad_x, grad_y, N, N)
            errors.append(err)
        
        # Compute convergence orders
        for i in range(1, len(sizes)):
            order = np.log(errors[i-1] / errors[i]) / np.log(2)
            assert order > 1.9, f"Expected 2nd order convergence, got {order:.2f}"


class TestVorticityAndStrain:
    """Test vorticity and strain rate computation."""
    
    def test_rigid_body_rotation(self):
        """Rigid body rotation: u = -y, v = x → ω = 2, |S| = 0."""
        NI, NJ = 32, 32
        X, Y, metrics = create_uniform_grid(NI, NJ)
        
        Q = np.zeros((NI + 2, NJ + 2, 4))
        Q[:, :, 1] = -Y  # u = -y
        Q[:, :, 2] = X   # v = x
        
        grad = compute_gradients(Q, metrics)
        omega = compute_vorticity(grad)
        S = compute_strain_rate(grad)
        
        assert np.allclose(omega, 2.0, atol=1e-10), "Vorticity should be 2"
        assert np.allclose(S, 0.0, atol=1e-10), "Strain should be 0"
    
    def test_pure_shear(self):
        """Pure shear: u = y, v = 0 → |ω| = 1, |S| = 1."""
        NI, NJ = 32, 32
        X, Y, metrics = create_uniform_grid(NI, NJ)
        
        Q = np.zeros((NI + 2, NJ + 2, 4))
        Q[:, :, 1] = Y  # u = y
        Q[:, :, 2] = 0  # v = 0
        
        grad = compute_gradients(Q, metrics)
        omega = compute_vorticity(grad)
        S = compute_strain_rate(grad)
        
        assert np.allclose(omega, 1.0, atol=1e-10), "Vorticity should be 1"
        assert np.allclose(S, 1.0, atol=1e-10), "Strain should be 1"
    
    def test_pure_stretch(self):
        """Pure stretching: u = x, v = -y → ω = 0, |S| = 2."""
        NI, NJ = 32, 32
        X, Y, metrics = create_uniform_grid(NI, NJ)
        
        Q = np.zeros((NI + 2, NJ + 2, 4))
        Q[:, :, 1] = X   # u = x
        Q[:, :, 2] = -Y  # v = -y
        
        grad = compute_gradients(Q, metrics)
        omega = compute_vorticity(grad)
        S = compute_strain_rate(grad)
        
        assert np.allclose(omega, 0.0, atol=1e-10), "Vorticity should be 0"
        assert np.allclose(S, 2.0, atol=1e-10), "Strain should be 2"


class TestMultipleVariables:
    """Test gradients are computed for all state variables."""
    
    def test_all_variables_computed(self):
        """All 4 state variables should have gradients computed."""
        NI, NJ = 16, 16
        X, Y, metrics = create_uniform_grid(NI, NJ)
        
        Q = np.zeros((NI + 2, NJ + 2, 4))
        Q[:, :, 0] = X          # p = x
        Q[:, :, 1] = Y          # u = y
        Q[:, :, 2] = X + Y      # v = x + y
        Q[:, :, 3] = 2*X - Y    # nu_t = 2x - y
        
        grad = compute_gradients(Q, metrics)
        
        assert grad.shape == (NI, NJ, 4, 2), "Output shape should be (NI, NJ, 4, 2)"
        
        # Check each variable's gradient
        assert np.allclose(grad[:, :, 0, 0], 1.0, atol=1e-12), "dp/dx = 1"
        assert np.allclose(grad[:, :, 0, 1], 0.0, atol=1e-12), "dp/dy = 0"
        
        assert np.allclose(grad[:, :, 1, 0], 0.0, atol=1e-12), "du/dx = 0"
        assert np.allclose(grad[:, :, 1, 1], 1.0, atol=1e-12), "du/dy = 1"
        
        assert np.allclose(grad[:, :, 2, 0], 1.0, atol=1e-12), "dv/dx = 1"
        assert np.allclose(grad[:, :, 2, 1], 1.0, atol=1e-12), "dv/dy = 1"
        
        assert np.allclose(grad[:, :, 3, 0], 2.0, atol=1e-12), "dnu_t/dx = 2"
        assert np.allclose(grad[:, :, 3, 1], -1.0, atol=1e-12), "dnu_t/dy = -1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

