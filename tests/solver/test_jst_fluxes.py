#!/usr/bin/env python
"""
Unit tests for JST flux scheme implementation.

Tests the Jameson-Schmidt-Turkel central-difference flux scheme
for artificial compressibility formulation.
"""

import os
import sys
import numpy as np

# Add project root to path (go up two levels: solver/ -> scripts/ -> project root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.numerics.fluxes import (
    compute_fluxes,
    compute_convective_flux,
    compute_spectral_radius,
    compute_time_step,
    FluxConfig,
    GridMetrics,
)


def create_cartesian_grid(NI: int, NJ: int, dx: float = 0.1, dy: float = 0.1) -> GridMetrics:
    """Create simple Cartesian grid metrics for testing."""
    # I-face normals (pointing in +x direction), scaled by face area (dy)
    Si_x = np.ones((NI + 1, NJ)) * dy
    Si_y = np.zeros((NI + 1, NJ))
    
    # J-face normals (pointing in +y direction), scaled by face area (dx)
    Sj_x = np.zeros((NI, NJ + 1))
    Sj_y = np.ones((NI, NJ + 1)) * dx
    
    # Cell volumes
    volume = np.ones((NI, NJ)) * dx * dy
    
    return GridMetrics(Si_x=Si_x, Si_y=Si_y, Sj_x=Sj_x, Sj_y=Sj_y, volume=volume)


def create_uniform_state(NI: int, NJ: int, p: float = 1.0, u: float = 1.0, 
                         v: float = 0.0, nu_t: float = 1e-4) -> np.ndarray:
    """Create uniform flow state with ghost cells."""
    NI_ghost, NJ_ghost = NI + 2, NJ + 2
    Q = np.zeros((NI_ghost, NJ_ghost, 4))
    Q[:, :, 0] = p
    Q[:, :, 1] = u
    Q[:, :, 2] = v
    Q[:, :, 3] = nu_t
    return Q


class TestResults:
    """Simple test result tracker."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def check(self, condition: bool, name: str, details: str = ""):
        if condition:
            self.passed += 1
            self.tests.append((name, "PASS", details))
        else:
            self.failed += 1
            self.tests.append((name, "FAIL", details))
    
    def summary(self):
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        for name, status, details in self.tests:
            print(f"  [{status}] {name}")
            if details and status == "FAIL":
                print(f"         {details}")
        print(f"\nTotal: {self.passed} passed, {self.failed} failed")
        return self.failed == 0


def test_output_shape():
    """Test that output has correct shape."""
    results = TestResults()
    print("\n" + "=" * 60)
    print("Test: Output Shape")
    print("=" * 60)
    
    NI, NJ = 10, 8
    Q = create_uniform_state(NI, NJ)
    metrics = create_cartesian_grid(NI, NJ)
    beta = 10.0
    cfg = FluxConfig()
    
    residual = compute_fluxes(Q, metrics, beta, cfg)
    
    expected_shape = (NI, NJ, 4)
    results.check(
        residual.shape == expected_shape,
        "Residual shape",
        f"Expected {expected_shape}, got {residual.shape}"
    )
    
    print(f"  Input Q shape: {Q.shape}")
    print(f"  Output residual shape: {residual.shape}")
    print(f"  Expected: {expected_shape}")
    
    return results


def test_uniform_flow_zero_residual():
    """Test that uniform flow has zero residual (to machine precision)."""
    results = TestResults()
    print("\n" + "=" * 60)
    print("Test: Uniform Flow Zero Residual")
    print("=" * 60)
    
    NI, NJ = 10, 8
    Q = create_uniform_state(NI, NJ, p=1.0, u=1.0, v=0.5, nu_t=1e-4)
    metrics = create_cartesian_grid(NI, NJ)
    beta = 10.0
    cfg = FluxConfig()
    
    residual = compute_fluxes(Q, metrics, beta, cfg)
    
    max_residual = np.abs(residual).max()
    tolerance = 1e-12
    
    results.check(
        max_residual < tolerance,
        "Uniform flow residual ~ 0",
        f"Max residual = {max_residual:.2e}, tolerance = {tolerance:.2e}"
    )
    
    print(f"  Max |residual|: {max_residual:.6e}")
    print(f"  Tolerance: {tolerance:.6e}")
    
    return results


def test_conservation():
    """Test that fluxes are conservative (sum of residuals = 0)."""
    results = TestResults()
    print("\n" + "=" * 60)
    print("Test: Conservation (Sum of Residuals = 0)")
    print("=" * 60)
    
    NI, NJ = 10, 8
    Q = create_uniform_state(NI, NJ)
    
    # Add a non-uniform perturbation
    Q[5, 4, 0] = 2.0  # Pressure spike
    Q[7, 6, 1] = 1.5  # Velocity perturbation
    
    metrics = create_cartesian_grid(NI, NJ)
    beta = 10.0
    cfg = FluxConfig()
    
    residual = compute_fluxes(Q, metrics, beta, cfg)
    
    tolerance = 1e-12
    all_conserved = True
    
    for i, name in enumerate(['p', 'u', 'v', 'nu_t']):
        total = residual[:, :, i].sum()
        is_conserved = np.abs(total) < tolerance
        results.check(
            is_conserved,
            f"Conservation of {name}",
            f"Sum = {total:.2e}"
        )
        print(f"  Sum of {name} residuals: {total:.6e}")
        all_conserved = all_conserved and is_conserved
    
    return results


def test_pressure_spike_dissipation():
    """Test that pressure spike triggers dissipation."""
    results = TestResults()
    print("\n" + "=" * 60)
    print("Test: Pressure Spike Triggers Dissipation")
    print("=" * 60)
    
    NI, NJ = 10, 8
    Q = create_uniform_state(NI, NJ)
    
    # Add a pressure spike
    spike_i, spike_j = 6, 5  # In ghost-cell indexing
    Q[spike_i, spike_j, 0] = 2.0
    
    metrics = create_cartesian_grid(NI, NJ)
    beta = 10.0
    cfg = FluxConfig(k2=0.5, k4=0.016)
    
    residual = compute_fluxes(Q, metrics, beta, cfg)
    
    # Interior cell index for spike location
    interior_i, interior_j = spike_i - 1, spike_j - 1
    
    max_loc = np.unravel_index(np.abs(residual[:, :, 0]).argmax(), (NI, NJ))
    
    results.check(
        np.abs(residual).max() > 0.01,
        "Non-zero residual from perturbation",
        f"Max residual = {np.abs(residual).max():.4f}"
    )
    
    results.check(
        max_loc == (interior_i, interior_j),
        "Max residual at spike location",
        f"Expected ({interior_i}, {interior_j}), got {max_loc}"
    )
    
    print(f"  Max |residual|: {np.abs(residual).max():.6e}")
    print(f"  Location of max: {max_loc}")
    print(f"  Expected location: ({interior_i}, {interior_j})")
    print(f"  Residual at spike: {residual[interior_i, interior_j, :]}")
    
    return results


def test_spectral_radius():
    """Test spectral radius computation."""
    results = TestResults()
    print("\n" + "=" * 60)
    print("Test: Spectral Radius Computation")
    print("=" * 60)
    
    beta = 10.0
    
    # Test case 1: Flow aligned with normal
    Q1 = np.array([1.0, 1.0, 0.0, 1e-4])
    nx1, ny1 = 1.0, 0.0
    lam1 = compute_spectral_radius(Q1, nx1, ny1, beta)
    
    u, v = Q1[1], Q1[2]
    expected1 = np.abs(u * nx1 + v * ny1) + np.sqrt(u**2 + v**2 + beta)
    
    results.check(
        np.abs(lam1 - expected1) < 1e-10,
        "Spectral radius (flow aligned with normal)",
        f"Got {lam1:.6f}, expected {expected1:.6f}"
    )
    print(f"  Case 1: u=1, v=0, n=(1,0)")
    print(f"    λ = {lam1:.6f}, expected = {expected1:.6f}")
    
    # Test case 2: Flow perpendicular to normal
    Q2 = np.array([1.0, 0.0, 1.0, 1e-4])
    nx2, ny2 = 1.0, 0.0
    lam2 = compute_spectral_radius(Q2, nx2, ny2, beta)
    
    u, v = Q2[1], Q2[2]
    expected2 = np.abs(u * nx2 + v * ny2) + np.sqrt(u**2 + v**2 + beta)
    
    results.check(
        np.abs(lam2 - expected2) < 1e-10,
        "Spectral radius (flow perpendicular to normal)",
        f"Got {lam2:.6f}, expected {expected2:.6f}"
    )
    print(f"  Case 2: u=0, v=1, n=(1,0)")
    print(f"    λ = {lam2:.6f}, expected = {expected2:.6f}")
    
    # Test case 3: Diagonal flow and normal
    Q3 = np.array([1.0, 1.0, 1.0, 1e-4])
    nx3, ny3 = 1.0, 1.0  # Diagonal normal (not unit)
    lam3 = compute_spectral_radius(Q3, nx3, ny3, beta)
    
    u, v = Q3[1], Q3[2]
    S = np.sqrt(nx3**2 + ny3**2)
    Un = (u * nx3 + v * ny3) / S
    c_art = np.sqrt(u**2 + v**2 + beta)
    expected3 = (np.abs(Un) + c_art) * S
    
    results.check(
        np.abs(lam3 - expected3) < 1e-10,
        "Spectral radius (diagonal)",
        f"Got {lam3:.6f}, expected {expected3:.6f}"
    )
    print(f"  Case 3: u=1, v=1, n=(1,1)")
    print(f"    λ = {lam3:.6f}, expected = {expected3:.6f}")
    
    return results


def test_convective_flux():
    """Test convective flux computation."""
    results = TestResults()
    print("\n" + "=" * 60)
    print("Test: Convective Flux Computation")
    print("=" * 60)
    
    beta = 10.0
    
    # Test case: Simple 1D flow
    Q = np.array([1.5, 2.0, 0.5, 1e-3])  # p, u, v, nu_t
    nx, ny = 0.1, 0.0  # Face normal scaled by area
    
    F = compute_convective_flux(Q, nx, ny, beta)
    
    p, u, v, nu_t = Q
    Un = u * nx + v * ny  # = 2.0 * 0.1 = 0.2
    
    expected = np.array([
        beta * Un,          # = 10 * 0.2 = 2.0
        u * Un + p * nx,    # = 2 * 0.2 + 1.5 * 0.1 = 0.55
        v * Un + p * ny,    # = 0.5 * 0.2 + 1.5 * 0 = 0.1
        nu_t * Un           # = 1e-3 * 0.2 = 2e-4
    ])
    
    for i, (name, computed, exp) in enumerate(zip(['continuity', 'x-mom', 'y-mom', 'SA'], F, expected)):
        error = np.abs(computed - exp)
        results.check(
            error < 1e-10,
            f"Convective flux: {name}",
            f"Got {computed:.6e}, expected {exp:.6e}"
        )
    
    print(f"  State: p={p}, u={u}, v={v}, nu_t={nu_t}")
    print(f"  Normal: nx={nx}, ny={ny}")
    print(f"  Computed flux: {F}")
    print(f"  Expected flux: {expected}")
    
    return results


def test_time_step():
    """Test local time step computation."""
    results = TestResults()
    print("\n" + "=" * 60)
    print("Test: Local Time Step Computation")
    print("=" * 60)
    
    NI, NJ = 5, 5
    Q = create_uniform_state(NI, NJ, u=1.0, v=0.5)
    metrics = create_cartesian_grid(NI, NJ, dx=0.1, dy=0.1)
    beta = 10.0
    cfl = 0.8
    
    dt = compute_time_step(Q, metrics, beta, cfl)
    
    results.check(
        dt.shape == (NI, NJ),
        "Time step shape",
        f"Expected ({NI}, {NJ}), got {dt.shape}"
    )
    
    results.check(
        np.all(dt > 0),
        "Time step positive",
        f"Min dt = {dt.min():.6e}"
    )
    
    # For uniform Cartesian grid, all time steps should be equal
    dt_variation = (dt.max() - dt.min()) / dt.mean()
    results.check(
        dt_variation < 1e-10,
        "Uniform dt on uniform grid",
        f"Variation = {dt_variation:.2e}"
    )
    
    print(f"  Time step shape: {dt.shape}")
    print(f"  Min dt: {dt.min():.6e}")
    print(f"  Max dt: {dt.max():.6e}")
    print(f"  Mean dt: {dt.mean():.6e}")
    
    return results


def test_different_grid_sizes():
    """Test flux computation on various grid sizes."""
    results = TestResults()
    print("\n" + "=" * 60)
    print("Test: Different Grid Sizes")
    print("=" * 60)
    
    grid_sizes = [(5, 5), (10, 8), (20, 15), (50, 30)]
    beta = 10.0
    cfg = FluxConfig()
    
    for NI, NJ in grid_sizes:
        Q = create_uniform_state(NI, NJ, u=1.0, v=0.5)
        metrics = create_cartesian_grid(NI, NJ)
        
        residual = compute_fluxes(Q, metrics, beta, cfg)
        max_res = np.abs(residual).max()
        
        results.check(
            residual.shape == (NI, NJ, 4),
            f"Shape for {NI}x{NJ} grid",
            f"Got {residual.shape}"
        )
        
        results.check(
            max_res < 1e-12,
            f"Zero residual for {NI}x{NJ} uniform flow",
            f"Max = {max_res:.2e}"
        )
        
        print(f"  {NI}x{NJ}: shape={residual.shape}, max|R|={max_res:.2e}")
    
    return results

def test_jst_switch_logic():
    """
    CRITICAL TEST: Verifies the 'Sensor' mechanism of JST.
    
    JST should use 4th order damping (k4) in smooth regions,
    but switch to 2nd order damping (k2) near strong gradients.
    
    We compare the PRESSURE (continuity) residual only, since the momentum
    residual contains physical contributions from dp/dx that are not
    related to the dissipation switch.
    """
    results = TestResults()
    print("\n" + "=" * 60)
    print("Test: JST Pressure Switch Logic")
    print("=" * 60)
    
    NI, NJ = 10, 5
    Q = create_uniform_state(NI, NJ)
    metrics = create_cartesian_grid(NI, NJ)
    beta = 1.0
    
    # 1. Smooth Flow (Linear Gradient)
    # The 2nd derivative of linear data is 0, so the pressure sensor should be 0.
    # With zero sensor, only k4 (4th order) dissipation acts.
    # For linear pressure, d^4p/dx^4 = 0, so dissipation flux ≈ 0.
    for i in range(NI + 2):
        Q[i, :, 0] = 1.0 + 0.1 * i  # Linear pressure ramp
        
    cfg = FluxConfig(k2=0.5, k4=0.016)
    
    res_smooth = compute_fluxes(Q, metrics, beta, cfg)
    # Focus on pressure residual - this shows dissipation behavior
    # (Momentum residual is dominated by physical dp/dx term)
    mag_smooth_p = np.abs(res_smooth[:, :, 0]).max()
    
    # 2. Shock (Step Function)
    # A sharp jump should trigger the pressure sensor -> high dissipation
    Q_shock = create_uniform_state(NI, NJ)
    Q_shock[0:6, :, 0] = 1.0
    Q_shock[6:, :, 0] = 2.0  # Step at i=6
    
    res_shock = compute_fluxes(Q_shock, metrics, beta, cfg)
    mag_shock_p = np.abs(res_shock[:, :, 0]).max()
    
    # The pressure residual at a shock should be significantly larger
    # than in smooth flow (which should be ~machine precision)
    ratio = mag_shock_p / (mag_smooth_p + 1e-20)
    
    results.check(
        ratio > 1e6,
        "Switch Activation (pressure residual)",
        f"Shock ({mag_shock_p:.2e}) should be >> Smooth ({mag_smooth_p:.2e})"
    )
    
    # Also verify the smooth case has essentially zero pressure residual
    results.check(
        mag_smooth_p < 1e-12,
        "Smooth flow: zero pressure dissipation",
        f"Got {mag_smooth_p:.2e}, expected < 1e-12"
    )
    
    print(f"  Smooth gradient - pressure residual: {mag_smooth_p:.2e}")
    print(f"  Step function   - pressure residual: {mag_shock_p:.2e}")
    print(f"  Ratio: {ratio:.2e}")
    
    return results

def test_symmetry_preservation():
    """
    Visual/Logic Test: Does a symmetric condition result in symmetric output?
    This catches 'off-by-one' indexing errors in flux calculations.
    """
    results = TestResults()
    print("\n" + "=" * 60)
    print("Test: Symmetry Preservation")
    print("=" * 60)
    
    NI, NJ = 11, 11 # Odd number to have a center line
    Q = create_uniform_state(NI, NJ)
    metrics = create_cartesian_grid(NI, NJ)
    
    # Place a pressure spike exactly in the center
    center_i, center_j = 6, 6 # (indices include ghost cells)
    Q[center_i, center_j, 0] = 5.0
    
    beta = 10.0
    cfg = FluxConfig()
    
    res = compute_fluxes(Q, metrics, beta, cfg)
    
    # The X-momentum residual to the left of center should be 
    # equal and opposite to the right of center
    # Center index in output (without ghost) is 5, 5
    res_left  = res[4, 5, 1] # i=4, j=5, var=u
    res_right = res[6, 5, 1] # i=6, j=5, var=u
    
    # Y-momentum top vs bottom
    res_bot = res[5, 4, 2]
    res_top = res[5, 6, 2]
    
    sym_error_x = np.abs(res_left + res_right) # Should sum to 0
    sym_error_y = np.abs(res_bot + res_top)
    
    results.check(
        sym_error_x < 1e-10,
        "X-Symmetry (Left/Right)",
        f"Left: {res_left:.4f}, Right: {res_right:.4f}, Sum: {sym_error_x:.2e}"
    )
    
    results.check(
        sym_error_y < 1e-10,
        "Y-Symmetry (Top/Bottom)",
        f"Bot: {res_bot:.4f}, Top: {res_top:.4f}, Sum: {sym_error_y:.2e}"
    )
    
    return results

def main():
    """Run all tests."""
    print("=" * 60)
    print("JST Flux Scheme Unit Tests")
    print("=" * 60)
    
    all_results = []
    
    # Run all tests
    all_results.append(test_output_shape())
    all_results.append(test_uniform_flow_zero_residual())
    all_results.append(test_conservation())
    all_results.append(test_pressure_spike_dissipation())
    all_results.append(test_spectral_radius())
    all_results.append(test_convective_flux())
    all_results.append(test_time_step())
    all_results.append(test_different_grid_sizes())
    all_results.append(test_jst_switch_logic())
    all_results.append(test_symmetry_preservation())
    
    # Overall summary
    total_passed = sum(r.passed for r in all_results)
    total_failed = sum(r.failed for r in all_results)
    
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"Total tests: {total_passed + total_failed}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    
    if total_failed == 0:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total_failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

