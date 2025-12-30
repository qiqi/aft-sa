#!/usr/bin/env python3
"""
Test Green-Gauss Gradient Reconstruction.

This script validates the gradient computation by:
1. Testing exact reproduction of linear fields
2. Testing convergence order for polynomial and trigonometric fields
3. Verifying vorticity and strain rate computations

Expected Results:
- Linear fields: Machine precision (exact)
- Quadratic/trig fields: 2nd order convergence on uniform grids
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.numerics.gradients import (
    compute_gradients, compute_vorticity, compute_strain_rate, GradientMetrics
)


def get_output_dir():
    """Get output directory for test results."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = os.path.join(project_root, 'output', 'numerics')
    os.makedirs(out, exist_ok=True)
    return out


def create_cartesian_grid(NI: int, NJ: int, Lx: float = 1.0, Ly: float = 1.0):
    """
    Create a uniform Cartesian grid with metrics.
    
    Returns
    -------
    X, Y : ndarray, shape (NI+2, NJ+2)
        Cell center coordinates (including ghost cells).
    metrics : GradientMetrics
        Grid metrics for gradient computation.
    dx, dy : float
        Grid spacing.
    """
    dx = Lx / NI
    dy = Ly / NJ
    
    # Cell centers (including ghost cells)
    x = np.linspace(-dx/2, Lx + dx/2, NI + 2)
    y = np.linspace(-dy/2, Ly + dy/2, NJ + 2)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Metrics for uniform Cartesian grid
    # I-faces: normal = (1, 0), area = dy
    Si_x = np.ones((NI + 1, NJ)) * dy
    Si_y = np.zeros((NI + 1, NJ))
    
    # J-faces: normal = (0, 1), area = dx
    Sj_x = np.zeros((NI, NJ + 1))
    Sj_y = np.ones((NI, NJ + 1)) * dx
    
    # Cell volumes
    volume = np.ones((NI, NJ)) * dx * dy
    
    metrics = GradientMetrics(Si_x, Si_y, Sj_x, Sj_y, volume)
    
    return X, Y, metrics, dx, dy


def create_stretched_grid(NI: int, NJ: int, Lx: float = 1.0, Ly: float = 1.0, 
                          stretch_x: float = 1.2, stretch_y: float = 1.1):
    """
    Create a stretched (non-uniform) Cartesian grid.
    
    Uses geometric stretching: dx_i+1 = stretch * dx_i
    """
    # Generate stretched node locations
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
    
    # Cell centers
    x_cell = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    y_cell = 0.5 * (y_nodes[:-1] + y_nodes[1:])
    
    # Add ghost cells
    dx_first = x_nodes[1] - x_nodes[0]
    dx_last = x_nodes[-1] - x_nodes[-2]
    dy_first = y_nodes[1] - y_nodes[0]
    dy_last = y_nodes[-1] - y_nodes[-2]
    
    x_full = np.concatenate([[x_cell[0] - dx_first], x_cell, [x_cell[-1] + dx_last]])
    y_full = np.concatenate([[y_cell[0] - dy_first], y_cell, [y_cell[-1] + dy_last]])
    
    X, Y = np.meshgrid(x_full, y_full, indexing='ij')
    
    # Metrics (non-uniform)
    dx_arr = np.diff(x_nodes)
    dy_arr = np.diff(y_nodes)
    
    # I-faces: area = dy at that j-level
    Si_x = np.zeros((NI + 1, NJ))
    Si_y = np.zeros((NI + 1, NJ))
    for j in range(NJ):
        Si_x[:, j] = dy_arr[j]
    
    # J-faces: area = dx at that i-level
    Sj_x = np.zeros((NI, NJ + 1))
    Sj_y = np.zeros((NI, NJ + 1))
    for i in range(NI):
        Sj_y[i, :] = dx_arr[i]
    
    # Cell volumes
    volume = np.outer(dx_arr, dy_arr)
    
    metrics = GradientMetrics(Si_x, Si_y, Sj_x, Sj_y, volume)
    
    return X, Y, metrics


# =============================================================================
# Analytical Test Fields
# =============================================================================

class LinearField:
    """u = a*x + b*y + c"""
    def __init__(self, a=2.0, b=3.0, c=1.0):
        self.a, self.b, self.c = a, b, c
        self.name = f"Linear: u = {a}x + {b}y + {c}"
    
    def value(self, X, Y):
        return self.a * X + self.b * Y + self.c
    
    def grad_x(self, X, Y):
        return np.full_like(X, self.a)
    
    def grad_y(self, X, Y):
        return np.full_like(X, self.b)


class QuadraticField:
    """u = a*x^2 + b*y^2 + c*x*y"""
    def __init__(self, a=1.0, b=2.0, c=0.5):
        self.a, self.b, self.c = a, b, c
        self.name = f"Quadratic: u = {a}x² + {b}y² + {c}xy"
    
    def value(self, X, Y):
        return self.a * X**2 + self.b * Y**2 + self.c * X * Y
    
    def grad_x(self, X, Y):
        return 2 * self.a * X + self.c * Y
    
    def grad_y(self, X, Y):
        return 2 * self.b * Y + self.c * X


class SinusoidalField:
    """u = sin(k*x) * cos(m*y)"""
    def __init__(self, k=2*np.pi, m=2*np.pi):
        self.k, self.m = k, m
        self.name = f"Sinusoidal: u = sin({k:.2f}x)·cos({m:.2f}y)"
    
    def value(self, X, Y):
        return np.sin(self.k * X) * np.cos(self.m * Y)
    
    def grad_x(self, X, Y):
        return self.k * np.cos(self.k * X) * np.cos(self.m * Y)
    
    def grad_y(self, X, Y):
        return -self.m * np.sin(self.k * X) * np.sin(self.m * Y)


class CubicField:
    """u = x^3 + y^3"""
    def __init__(self):
        self.name = "Cubic: u = x³ + y³"
    
    def value(self, X, Y):
        return X**3 + Y**3
    
    def grad_x(self, X, Y):
        return 3 * X**2
    
    def grad_y(self, X, Y):
        return 3 * Y**2


class ExponentialField:
    """u = exp(a*x + b*y)"""
    def __init__(self, a=1.0, b=1.0):
        self.a, self.b = a, b
        self.name = f"Exponential: u = exp({a}x + {b}y)"
    
    def value(self, X, Y):
        return np.exp(self.a * X + self.b * Y)
    
    def grad_x(self, X, Y):
        return self.a * np.exp(self.a * X + self.b * Y)
    
    def grad_y(self, X, Y):
        return self.b * np.exp(self.a * X + self.b * Y)


# =============================================================================
# Test Functions
# =============================================================================

def test_single_field(field, NI, NJ, grid_type='uniform'):
    """
    Test gradient computation for a single analytical field.
    
    Returns
    -------
    err_x, err_y : float
        L2 norm of gradient errors.
    """
    if grid_type == 'uniform':
        X, Y, metrics, dx, dy = create_cartesian_grid(NI, NJ)
    else:
        X, Y, metrics = create_stretched_grid(NI, NJ)
    
    # Create state array with field in u-component
    Q = np.zeros((NI + 2, NJ + 2, 4))
    Q[:, :, 1] = field.value(X, Y)
    
    # Compute gradients
    grad = compute_gradients(Q, metrics)
    
    # Get interior coordinates
    X_int = X[1:-1, 1:-1]
    Y_int = Y[1:-1, 1:-1]
    
    # Analytical gradients at interior cell centers
    grad_x_exact = field.grad_x(X_int, Y_int)
    grad_y_exact = field.grad_y(X_int, Y_int)
    
    # Computed gradients
    grad_x_comp = grad[:, :, 1, 0]
    grad_y_comp = grad[:, :, 1, 1]
    
    # L2 error
    err_x = np.sqrt(np.mean((grad_x_comp - grad_x_exact)**2))
    err_y = np.sqrt(np.mean((grad_y_comp - grad_y_exact)**2))
    
    return err_x, err_y


def test_linear_exactness():
    """Test that linear fields give exact gradients."""
    print("=" * 60)
    print("Test 1: Linear Field Exactness")
    print("=" * 60)
    
    field = LinearField(a=2.5, b=-1.7, c=3.0)
    print(f"Field: {field.name}")
    
    all_passed = True
    tol = 1e-12
    
    for NI, NJ in [(8, 8), (16, 16), (32, 32), (10, 20)]:
        err_x, err_y = test_single_field(field, NI, NJ)
        status = "✅" if err_x < tol and err_y < tol else "❌"
        print(f"  Grid {NI}×{NJ}: err_x = {err_x:.2e}, err_y = {err_y:.2e} {status}")
        
        if err_x > tol or err_y > tol:
            all_passed = False
    
    if all_passed:
        print("✅ Linear fields give machine-precision gradients")
    else:
        print("❌ Linear field test FAILED")
    
    return all_passed


def test_convergence_order(field, grid_sizes, expected_order=2, grid_type='uniform'):
    """
    Test convergence order of gradient computation.
    
    Parameters
    ----------
    field : analytical field object
    grid_sizes : list of (NI, NJ) tuples
    expected_order : expected convergence order
    grid_type : 'uniform' or 'stretched'
    
    Returns
    -------
    passed : bool
    orders : list of computed convergence orders
    """
    errors_x = []
    errors_y = []
    h_list = []
    
    for NI, NJ in grid_sizes:
        err_x, err_y = test_single_field(field, NI, NJ, grid_type)
        errors_x.append(err_x)
        errors_y.append(err_y)
        h_list.append(1.0 / NI)  # characteristic mesh size
    
    # Compute convergence orders
    orders_x = []
    orders_y = []
    for i in range(1, len(grid_sizes)):
        if errors_x[i] > 1e-14 and errors_x[i-1] > 1e-14:
            order_x = np.log(errors_x[i-1] / errors_x[i]) / np.log(h_list[i-1] / h_list[i])
            orders_x.append(order_x)
        if errors_y[i] > 1e-14 and errors_y[i-1] > 1e-14:
            order_y = np.log(errors_y[i-1] / errors_y[i]) / np.log(h_list[i-1] / h_list[i])
            orders_y.append(order_y)
    
    return errors_x, errors_y, orders_x, orders_y, h_list


def test_polynomial_convergence():
    """Test convergence order for polynomial fields."""
    print("\n" + "=" * 60)
    print("Test 2: Polynomial Field Convergence")
    print("=" * 60)
    
    grid_sizes = [(8, 8), (16, 16), (32, 32), (64, 64)]
    all_passed = True
    
    fields = [
        QuadraticField(a=1.0, b=2.0, c=0.5),
        CubicField(),
    ]
    
    for field in fields:
        print(f"\nField: {field.name}")
        errors_x, errors_y, orders_x, orders_y, h_list = test_convergence_order(
            field, grid_sizes, expected_order=2
        )
        
        print("  Grid       Error (∂/∂x)    Error (∂/∂y)    Order")
        print("  " + "-" * 50)
        for i, (NI, NJ) in enumerate(grid_sizes):
            order_str = f"{orders_x[i-1]:.2f}" if i > 0 and len(orders_x) >= i else "---"
            print(f"  {NI:3d}×{NJ:<3d}    {errors_x[i]:.4e}      {errors_y[i]:.4e}      {order_str}")
        
        # Check that order is at least 1.8 (allowing some margin)
        if orders_x and min(orders_x) < 1.8:
            print(f"  ⚠️  Order in x ({min(orders_x):.2f}) below expected 2.0")
            all_passed = False
        if orders_y and min(orders_y) < 1.8:
            print(f"  ⚠️  Order in y ({min(orders_y):.2f}) below expected 2.0")
            all_passed = False
    
    if all_passed:
        print("\n✅ Polynomial fields show 2nd order convergence")
    else:
        print("\n❌ Convergence order test FAILED")
    
    return all_passed


def test_trigonometric_convergence():
    """Test convergence order for trigonometric fields."""
    print("\n" + "=" * 60)
    print("Test 3: Trigonometric Field Convergence")
    print("=" * 60)
    
    grid_sizes = [(8, 8), (16, 16), (32, 32), (64, 64)]
    all_passed = True
    
    fields = [
        SinusoidalField(k=2*np.pi, m=2*np.pi),
        SinusoidalField(k=4*np.pi, m=2*np.pi),  # Higher frequency in x
    ]
    
    for field in fields:
        print(f"\nField: {field.name}")
        errors_x, errors_y, orders_x, orders_y, h_list = test_convergence_order(
            field, grid_sizes, expected_order=2
        )
        
        print("  Grid       Error (∂/∂x)    Error (∂/∂y)    Order")
        print("  " + "-" * 50)
        for i, (NI, NJ) in enumerate(grid_sizes):
            order_str = f"{orders_x[i-1]:.2f}" if i > 0 and len(orders_x) >= i else "---"
            print(f"  {NI:3d}×{NJ:<3d}    {errors_x[i]:.4e}      {errors_y[i]:.4e}      {order_str}")
        
        if orders_x and min(orders_x) < 1.8:
            print(f"  ⚠️  Order in x ({min(orders_x):.2f}) below expected 2.0")
            all_passed = False
    
    if all_passed:
        print("\n✅ Trigonometric fields show 2nd order convergence")
    else:
        print("\n❌ Trigonometric convergence test FAILED")
    
    return all_passed


def test_vorticity_and_strain():
    """Test vorticity and strain rate computation."""
    print("\n" + "=" * 60)
    print("Test 4: Vorticity and Strain Rate")
    print("=" * 60)
    
    all_passed = True
    NI, NJ = 32, 32
    X, Y, metrics, dx, dy = create_cartesian_grid(NI, NJ)
    
    # Test 1: Rigid body rotation (u = -y, v = x)
    # Vorticity = dv/dx - du/dy = 1 - (-1) = 2
    print("\nTest 4a: Rigid body rotation (u = -y, v = x)")
    Q = np.zeros((NI + 2, NJ + 2, 4))
    Q[:, :, 1] = -Y  # u = -y
    Q[:, :, 2] = X   # v = x
    
    grad = compute_gradients(Q, metrics)
    omega = compute_vorticity(grad)
    S_mag = compute_strain_rate(grad)
    
    omega_exact = 2.0
    S_exact = 0.0  # Pure rotation has no strain
    
    omega_err = np.abs(omega.mean() - omega_exact)
    S_err = S_mag.mean()  # Should be ~0
    
    print(f"  Vorticity: expected = {omega_exact}, computed = {omega.mean():.6f}, error = {omega_err:.2e}")
    print(f"  Strain rate: expected = {S_exact}, computed = {S_mag.mean():.6f}")
    
    if omega_err < 1e-10:
        print("  ✅ Vorticity correct")
    else:
        print("  ❌ Vorticity error too large")
        all_passed = False
    
    if S_err < 1e-10:
        print("  ✅ Strain rate correct (zero for pure rotation)")
    else:
        print("  ❌ Strain rate should be zero for rotation")
        all_passed = False
    
    # Test 2: Pure shear (u = y, v = 0)
    # Vorticity = 0 - 1 = -1 → |ω| = 1
    # Strain: S_xy = 0.5 * (1 + 0) = 0.5, |S| = sqrt(2 * 2 * 0.25) = 1
    print("\nTest 4b: Pure shear (u = y, v = 0)")
    Q = np.zeros((NI + 2, NJ + 2, 4))
    Q[:, :, 1] = Y   # u = y
    Q[:, :, 2] = 0   # v = 0
    
    grad = compute_gradients(Q, metrics)
    omega = compute_vorticity(grad)
    S_mag = compute_strain_rate(grad)
    
    omega_exact = 1.0
    S_exact = 1.0
    
    omega_err = np.abs(omega.mean() - omega_exact)
    S_err = np.abs(S_mag.mean() - S_exact)
    
    print(f"  Vorticity: expected = {omega_exact}, computed = {omega.mean():.6f}, error = {omega_err:.2e}")
    print(f"  Strain rate: expected = {S_exact}, computed = {S_mag.mean():.6f}, error = {S_err:.2e}")
    
    if omega_err < 1e-10 and S_err < 1e-10:
        print("  ✅ Both vorticity and strain rate correct")
    else:
        print("  ❌ Error too large")
        all_passed = False
    
    # Test 3: Pure stretching (u = x, v = -y)
    # Vorticity = 0, Strain: S_xx = 1, S_yy = -1, |S| = sqrt(2*(1 + 1)) = 2
    print("\nTest 4c: Pure stretching (u = x, v = -y)")
    Q = np.zeros((NI + 2, NJ + 2, 4))
    Q[:, :, 1] = X   # u = x
    Q[:, :, 2] = -Y  # v = -y
    
    grad = compute_gradients(Q, metrics)
    omega = compute_vorticity(grad)
    S_mag = compute_strain_rate(grad)
    
    omega_exact = 0.0
    S_exact = 2.0
    
    omega_err = omega.mean()
    S_err = np.abs(S_mag.mean() - S_exact)
    
    print(f"  Vorticity: expected = {omega_exact}, computed = {omega.mean():.6f}")
    print(f"  Strain rate: expected = {S_exact}, computed = {S_mag.mean():.6f}, error = {S_err:.2e}")
    
    if omega_err < 1e-10 and S_err < 1e-10:
        print("  ✅ Both vorticity and strain rate correct")
    else:
        print("  ❌ Error too large")
        all_passed = False
    
    return all_passed


def plot_convergence_study():
    """Generate convergence plot for documentation."""
    print("\n" + "=" * 60)
    print("Generating Convergence Plot")
    print("=" * 60)
    
    grid_sizes = [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128)]
    
    fields = [
        ('Quadratic', QuadraticField()),
        ('Sinusoidal', SinusoidalField()),
        ('Cubic', CubicField()),
        ('Exponential', ExponentialField()),
    ]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(fields)))
    
    for (name, field), color in zip(fields, colors):
        errors_x, errors_y, _, _, h_list = test_convergence_order(field, grid_sizes)
        errors = [max(ex, ey) for ex, ey in zip(errors_x, errors_y)]
        
        ax.loglog(h_list, errors, 'o-', color=color, linewidth=2, 
                  markersize=8, label=name)
    
    # Reference lines
    h_ref = np.array([h_list[0], h_list[-1]])
    ax.loglog(h_ref, 0.5 * h_ref**2, 'k--', linewidth=1, label='O(h²)')
    ax.loglog(h_ref, 0.1 * h_ref**3, 'k:', linewidth=1, label='O(h³)')
    
    ax.set_xlabel('Mesh size h', fontsize=12)
    ax.set_ylabel('Gradient L2 Error', fontsize=12)
    ax.set_title('Green-Gauss Gradient Convergence', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(get_output_dir(), 'gradient_convergence.pdf')
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


def run():
    """Run all gradient tests."""
    print("=" * 60)
    print("GREEN-GAUSS GRADIENT RECONSTRUCTION TESTS")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Linear exactness
    if not test_linear_exactness():
        all_passed = False
    
    # Test 2: Polynomial convergence
    if not test_polynomial_convergence():
        all_passed = False
    
    # Test 3: Trigonometric convergence
    if not test_trigonometric_convergence():
        all_passed = False
    
    # Test 4: Vorticity and strain rate
    if not test_vorticity_and_strain():
        all_passed = False
    
    # Generate convergence plot
    plot_convergence_study()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if all_passed:
        print("✅ All gradient tests PASSED")
        print("\nKey results:")
        print("  - Linear fields: exact (machine precision)")
        print("  - Higher-order fields: 2nd order convergence")
        print("  - Vorticity and strain rate: exact for linear velocity")
        return 0
    else:
        print("❌ Some tests FAILED")
        return 1


if __name__ == "__main__":
    exit(run())


