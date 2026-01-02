"""
Pytest tests for Green-Gauss gradient reconstruction.

Tests verify:
1. Exactness for linear/quadratic fields
2. 2nd order convergence for higher-order fields
3. Correctness of vorticity and strain rate
"""

import numpy as np
import pytest
import os

from src.constants import NGHOST
from src.numerics.gradients import (
    compute_gradients, compute_vorticity, compute_strain_rate, GradientMetrics
)


def create_uniform_grid(NI: int, NJ: int, Lx: float = 1.0, Ly: float = 1.0):
    """Create uniform Cartesian grid with metrics.
    
    With NGHOST ghost layers: X, Y have shape (NI + 2*NGHOST, NJ + 2*NGHOST)
    """
    dx = Lx / NI
    dy = Ly / NJ
    
    # Cell centers including ghost cells
    x = np.linspace(-(NGHOST - 0.5) * dx, Lx + (NGHOST - 0.5) * dx, NI + 2 * NGHOST)
    y = np.linspace(-(NGHOST - 0.5) * dy, Ly + (NGHOST - 0.5) * dy, NJ + 2 * NGHOST)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    metrics = GradientMetrics(
        Si_x=np.ones((NI + 1, NJ)) * dy,
        Si_y=np.zeros((NI + 1, NJ)),
        Sj_x=np.zeros((NI, NJ + 1)),
        Sj_y=np.ones((NI, NJ + 1)) * dx,
        volume=np.ones((NI, NJ)) * dx * dy
    )
    return X, Y, metrics


def create_stretched_grid(NI: int, NJ: int, Lx: float = 1.0, Ly: float = 1.0,
                          stretch_x: float = 1.2, stretch_y: float = 1.1):
    """Create stretched (non-uniform) Cartesian grid with NGHOST ghost layers."""
    def stretched_nodes(N, L, ratio):
        if abs(ratio - 1.0) < 1e-10:
            return np.linspace(0, L, N + 1)
        h0 = L * (1 - ratio) / (1 - ratio**N)
        nodes = np.zeros(N + 1)
        for i in range(1, N + 1):
            nodes[i] = nodes[i-1] + h0 * ratio**(i-1)
        return nodes
    
    x_nodes = stretched_nodes(NI, Lx, stretch_x)
    y_nodes = stretched_nodes(NJ, Ly, stretch_y)
    
    x_cell = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    y_cell = 0.5 * (y_nodes[:-1] + y_nodes[1:])
    
    dx_first, dx_last = x_nodes[1] - x_nodes[0], x_nodes[-1] - x_nodes[-2]
    dy_first, dy_last = y_nodes[1] - y_nodes[0], y_nodes[-1] - y_nodes[-2]
    
    # Add NGHOST ghost cells on each side
    x_ghosts_left = [x_cell[0] - (NGHOST - k) * dx_first for k in range(NGHOST)]
    x_ghosts_right = [x_cell[-1] + (k + 1) * dx_last for k in range(NGHOST)]
    x_full = np.concatenate([x_ghosts_left, x_cell, x_ghosts_right])
    
    y_ghosts_left = [y_cell[0] - (NGHOST - k) * dy_first for k in range(NGHOST)]
    y_ghosts_right = [y_cell[-1] + (k + 1) * dy_last for k in range(NGHOST)]
    y_full = np.concatenate([y_ghosts_left, y_cell, y_ghosts_right])
    
    X, Y = np.meshgrid(x_full, y_full, indexing='ij')
    
    dx_arr, dy_arr = np.diff(x_nodes), np.diff(y_nodes)
    
    Si_x = np.zeros((NI + 1, NJ))
    for j in range(NJ):
        Si_x[:, j] = dy_arr[j]
    
    Sj_y = np.zeros((NI, NJ + 1))
    for i in range(NI):
        Sj_y[i, :] = dx_arr[i]
    
    metrics = GradientMetrics(
        Si_x=Si_x, Si_y=np.zeros((NI + 1, NJ)),
        Sj_x=np.zeros((NI, NJ + 1)), Sj_y=Sj_y,
        volume=np.outer(dx_arr, dy_arr)
    )
    return X, Y, metrics


# =============================================================================
# Test Classes
# =============================================================================

class TestLinearExactness:
    """Test that linear fields give exact gradients."""
    
    @pytest.mark.parametrize("NI,NJ", [(8, 8), (16, 16), (10, 20), (32, 32)])
    def test_linear_field(self, NI, NJ):
        """Linear field u = ax + by + c should give exact gradients."""
        a, b, c = 2.5, -1.7, 3.0
        X, Y, metrics = create_uniform_grid(NI, NJ)
        
        Q = np.zeros((NI + 2*NGHOST, NJ + 2*NGHOST, 4))
        Q[:, :, 1] = a * X + b * Y + c
        
        grad = compute_gradients(Q, metrics)
        
        assert np.allclose(grad[:, :, 1, 0], a, atol=1e-12), "du/dx should equal a"
        assert np.allclose(grad[:, :, 1, 1], b, atol=1e-12), "du/dy should equal b"
    
    @pytest.mark.parametrize("NI,NJ", [(8, 8), (16, 16)])
    def test_quadratic_field(self, NI, NJ):
        """Quadratic field u = x² + y² should give exact gradients on uniform grid."""
        X, Y, metrics = create_uniform_grid(NI, NJ)
        
        Q = np.zeros((NI + 2*NGHOST, NJ + 2*NGHOST, 4))
        Q[:, :, 1] = X**2 + Y**2
        
        grad = compute_gradients(Q, metrics)
        
        X_int, Y_int = X[NGHOST:-NGHOST, NGHOST:-NGHOST], Y[NGHOST:-NGHOST, NGHOST:-NGHOST]
        
        assert np.allclose(grad[:, :, 1, 0], 2 * X_int, atol=1e-12)
        assert np.allclose(grad[:, :, 1, 1], 2 * Y_int, atol=1e-12)


class TestConvergenceOrder:
    """Test 2nd order convergence for higher-order fields."""
    
    def compute_error(self, field_func, grad_x_func, grad_y_func, NI, NJ):
        """Compute gradient error for a given field."""
        X, Y, metrics = create_uniform_grid(NI, NJ)
        
        Q = np.zeros((NI + 2*NGHOST, NJ + 2*NGHOST, 4))
        Q[:, :, 1] = field_func(X, Y)
        
        grad = compute_gradients(Q, metrics)
        X_int, Y_int = X[NGHOST:-NGHOST, NGHOST:-NGHOST], Y[NGHOST:-NGHOST, NGHOST:-NGHOST]
        
        err_x = np.sqrt(np.mean((grad[:, :, 1, 0] - grad_x_func(X_int, Y_int))**2))
        err_y = np.sqrt(np.mean((grad[:, :, 1, 1] - grad_y_func(X_int, Y_int))**2))
        return max(err_x, err_y)
    
    def test_cubic_convergence(self):
        """Cubic field u = x³ + y³ should show 2nd order convergence."""
        field = lambda X, Y: X**3 + Y**3
        grad_x = lambda X, Y: 3 * X**2
        grad_y = lambda X, Y: 3 * Y**2
        
        errors = [self.compute_error(field, grad_x, grad_y, N, N) for N in [16, 32, 64]]
        
        for i in range(1, len(errors)):
            order = np.log(errors[i-1] / errors[i]) / np.log(2)
            assert order > 1.9, f"Expected 2nd order, got {order:.2f}"
    
    def test_sinusoidal_convergence(self):
        """Sinusoidal field should show 2nd order convergence."""
        k = 2 * np.pi
        field = lambda X, Y: np.sin(k * X) * np.cos(k * Y)
        grad_x = lambda X, Y: k * np.cos(k * X) * np.cos(k * Y)
        grad_y = lambda X, Y: -k * np.sin(k * X) * np.sin(k * Y)
        
        errors = [self.compute_error(field, grad_x, grad_y, N, N) for N in [16, 32, 64]]
        
        for i in range(1, len(errors)):
            order = np.log(errors[i-1] / errors[i]) / np.log(2)
            assert order > 1.9, f"Expected 2nd order, got {order:.2f}"
    
    def test_exponential_convergence(self):
        """Exponential field u = exp(x+y) should show 2nd order convergence."""
        field = lambda X, Y: np.exp(X + Y)
        grad_x = lambda X, Y: np.exp(X + Y)
        grad_y = lambda X, Y: np.exp(X + Y)
        
        errors = [self.compute_error(field, grad_x, grad_y, N, N) for N in [16, 32, 64]]
        
        for i in range(1, len(errors)):
            order = np.log(errors[i-1] / errors[i]) / np.log(2)
            assert order > 1.9, f"Expected 2nd order, got {order:.2f}"


class TestVorticityAndStrain:
    """Test vorticity and strain rate computation."""
    
    def test_rigid_body_rotation(self):
        """Rigid body rotation: u = -y, v = x → ω = 2, |S| = 0."""
        NI, NJ = 32, 32
        X, Y, metrics = create_uniform_grid(NI, NJ)
        
        Q = np.zeros((NI + 2*NGHOST, NJ + 2*NGHOST, 4))
        Q[:, :, 1] = -Y
        Q[:, :, 2] = X
        
        grad = compute_gradients(Q, metrics)
        omega = compute_vorticity(grad)
        S = compute_strain_rate(grad)
        
        assert np.allclose(omega, 2.0, atol=1e-10), "Vorticity should be 2"
        assert np.allclose(S, 0.0, atol=1e-10), "Strain should be 0"
    
    def test_pure_shear(self):
        """Pure shear: u = y, v = 0 → |ω| = 1, |S| = 1."""
        NI, NJ = 32, 32
        X, Y, metrics = create_uniform_grid(NI, NJ)
        
        Q = np.zeros((NI + 2*NGHOST, NJ + 2*NGHOST, 4))
        Q[:, :, 1] = Y
        Q[:, :, 2] = 0
        
        grad = compute_gradients(Q, metrics)
        
        assert np.allclose(compute_vorticity(grad), 1.0, atol=1e-10)
        assert np.allclose(compute_strain_rate(grad), 1.0, atol=1e-10)
    
    def test_pure_stretch(self):
        """Pure stretching: u = x, v = -y → ω = 0, |S| = 2."""
        NI, NJ = 32, 32
        X, Y, metrics = create_uniform_grid(NI, NJ)
        
        Q = np.zeros((NI + 2*NGHOST, NJ + 2*NGHOST, 4))
        Q[:, :, 1] = X
        Q[:, :, 2] = -Y
        
        grad = compute_gradients(Q, metrics)
        
        assert np.allclose(compute_vorticity(grad), 0.0, atol=1e-10)
        assert np.allclose(compute_strain_rate(grad), 2.0, atol=1e-10)


class TestMultipleVariables:
    """Test gradients are computed for all state variables."""
    
    def test_all_variables_computed(self):
        """All 4 state variables should have gradients computed."""
        NI, NJ = 16, 16
        X, Y, metrics = create_uniform_grid(NI, NJ)
        
        Q = np.zeros((NI + 2*NGHOST, NJ + 2*NGHOST, 4))
        Q[:, :, 0] = X          # p = x
        Q[:, :, 1] = Y          # u = y
        Q[:, :, 2] = X + Y      # v = x + y
        Q[:, :, 3] = 2*X - Y    # nu_t = 2x - y
        
        grad = compute_gradients(Q, metrics)
        
        assert grad.shape == (NI, NJ, 4, 2)
        
        assert np.allclose(grad[:, :, 0, 0], 1.0, atol=1e-12)  # dp/dx
        assert np.allclose(grad[:, :, 0, 1], 0.0, atol=1e-12)  # dp/dy
        assert np.allclose(grad[:, :, 1, 0], 0.0, atol=1e-12)  # du/dx
        assert np.allclose(grad[:, :, 1, 1], 1.0, atol=1e-12)  # du/dy
        assert np.allclose(grad[:, :, 2, 0], 1.0, atol=1e-12)  # dv/dx
        assert np.allclose(grad[:, :, 2, 1], 1.0, atol=1e-12)  # dv/dy
        assert np.allclose(grad[:, :, 3, 0], 2.0, atol=1e-12)  # dnu_t/dx
        assert np.allclose(grad[:, :, 3, 1], -1.0, atol=1e-12) # dnu_t/dy


class TestStretchedGrid:
    """Test gradients on non-uniform grids."""
    
    def test_linear_on_stretched_interior(self):
        """Linear fields should be reasonably accurate on stretched grids."""
        # Note: Simple face averaging is not exact for non-uniform grids
        # but should still be accurate for smooth fields
        NI, NJ = 16, 16
        a, b = 2.0, 3.0
        
        X, Y, metrics = create_stretched_grid(NI, NJ, stretch_x=1.1, stretch_y=1.05)
        
        Q = np.zeros((NI + 2*NGHOST, NJ + 2*NGHOST, 4))
        Q[:, :, 1] = a * X + b * Y
        
        grad = compute_gradients(Q, metrics)
        
        # Interior cells (away from boundaries) should be more accurate
        interior = grad[2:-2, 2:-2, 1, :]
        assert np.allclose(interior[:, :, 0], a, rtol=0.05)  # 5% tolerance
        assert np.allclose(interior[:, :, 1], b, rtol=0.05)


# =============================================================================
# Standalone runner for generating plots
# =============================================================================

def run_convergence_study(output_dir: str = None):
    """
    Generate convergence plot for documentation.
    
    Parameters
    ----------
    output_dir : str, optional
        Directory to save plots. If None, uses output/numerics.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                   'output', 'numerics')
    os.makedirs(output_dir, exist_ok=True)
    
    grid_sizes = [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128)]
    
    fields = [
        ('Quadratic', lambda X, Y: X**2 + Y**2, 
         lambda X, Y: 2*X, lambda X, Y: 2*Y),
        ('Sinusoidal', lambda X, Y: np.sin(2*np.pi*X) * np.cos(2*np.pi*Y),
         lambda X, Y: 2*np.pi * np.cos(2*np.pi*X) * np.cos(2*np.pi*Y),
         lambda X, Y: -2*np.pi * np.sin(2*np.pi*X) * np.sin(2*np.pi*Y)),
        ('Cubic', lambda X, Y: X**3 + Y**3,
         lambda X, Y: 3*X**2, lambda X, Y: 3*Y**2),
        ('Exponential', lambda X, Y: np.exp(X + Y),
         lambda X, Y: np.exp(X + Y), lambda X, Y: np.exp(X + Y)),
    ]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(fields)))
    
    for (name, field, grad_x, grad_y), color in zip(fields, colors):
        errors, h_list = [], []
        
        for NI, NJ in grid_sizes:
            X, Y, metrics = create_uniform_grid(NI, NJ)
            Q = np.zeros((NI + 2*NGHOST, NJ + 2*NGHOST, 4))
            Q[:, :, 1] = field(X, Y)
            grad = compute_gradients(Q, metrics)
            
            X_int, Y_int = X[NGHOST:-NGHOST, NGHOST:-NGHOST], Y[NGHOST:-NGHOST, NGHOST:-NGHOST]
            err_x = np.sqrt(np.mean((grad[:, :, 1, 0] - grad_x(X_int, Y_int))**2))
            err_y = np.sqrt(np.mean((grad[:, :, 1, 1] - grad_y(X_int, Y_int))**2))
            
            errors.append(max(err_x, err_y))
            h_list.append(1.0 / NI)
        
        ax.loglog(h_list, errors, 'o-', color=color, linewidth=2, 
                  markersize=8, label=name)
    
    h_ref = np.array([h_list[0], h_list[-1]])
    ax.loglog(h_ref, 0.5 * h_ref**2, 'k--', linewidth=1, label='O(h²)')
    
    ax.set_xlabel('Mesh size h', fontsize=12)
    ax.set_ylabel('Gradient L2 Error', fontsize=12)
    ax.set_title('Green-Gauss Gradient Convergence', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'gradient_convergence.pdf')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
