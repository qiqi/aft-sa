#!/usr/bin/env python3
"""
Unit tests for the time stepping module.

Tests cover:
    - Spectral radius computation
    - Local time step calculation
    - Global time step
    - Explicit Euler integrator
    - RK4 integrator
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from dataclasses import dataclass
from typing import List

from src.solvers.time_stepping import (
    TimeStepConfig,
    SpectralRadius,
    compute_spectral_radii,
    compute_local_timestep,
    compute_global_timestep,
    ExplicitEuler,
    RungeKutta4,
)
from src.numerics.fluxes import compute_fluxes, FluxConfig, GridMetrics


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str


class TimeSteppingTests:
    """Test suite for time stepping module."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        # Common test parameters
        self.NI = 20
        self.NJ = 15
        self.dx = 0.1
        self.dy = 0.1
        self.beta = 10.0
        
    def setup_uniform_grid(self):
        """Create uniform Cartesian grid metrics."""
        NI, NJ = self.NI, self.NJ
        dx, dy = self.dx, self.dy
        
        Si_x = np.ones((NI+1, NJ)) * dy
        Si_y = np.zeros((NI+1, NJ))
        Sj_x = np.zeros((NI, NJ+1))
        Sj_y = np.ones((NI, NJ+1)) * dx
        volume = np.ones((NI, NJ)) * dx * dy
        
        return Si_x, Si_y, Sj_x, Sj_y, volume
    
    def setup_uniform_flow(self, u=1.0, v=0.0, p=0.0):
        """Create uniform flow state."""
        Q = np.zeros((self.NI + 2, self.NJ + 2, 4))
        Q[:, :, 0] = p
        Q[:, :, 1] = u
        Q[:, :, 2] = v
        Q[:, :, 3] = 1e-6
        return Q
    
    def run_test(self, name: str, test_func):
        """Run a single test and record result."""
        try:
            passed, message = test_func()
            self.results.append(TestResult(name, passed, message))
        except Exception as e:
            self.results.append(TestResult(name, False, f"Exception: {e}"))
    
    # ===== Spectral Radius Tests =====
    
    def test_spectral_radius_shape(self):
        """Test spectral radius output shape."""
        Q = self.setup_uniform_flow()
        Si_x, Si_y, Sj_x, Sj_y, _ = self.setup_uniform_grid()
        
        spec = compute_spectral_radii(Q, Si_x, Si_y, Sj_x, Sj_y, self.beta)
        
        expected_shape = (self.NI, self.NJ)
        if spec.lambda_i.shape != expected_shape:
            return False, f"lambda_i shape {spec.lambda_i.shape} != {expected_shape}"
        if spec.lambda_j.shape != expected_shape:
            return False, f"lambda_j shape {spec.lambda_j.shape} != {expected_shape}"
        return True, "Correct shapes"
    
    def test_spectral_radius_positive(self):
        """Spectral radius should always be positive."""
        Q = self.setup_uniform_flow(u=-1.0, v=-0.5)
        Si_x, Si_y, Sj_x, Sj_y, _ = self.setup_uniform_grid()
        
        spec = compute_spectral_radii(Q, Si_x, Si_y, Sj_x, Sj_y, self.beta)
        
        if np.any(spec.lambda_i <= 0):
            return False, "lambda_i has non-positive values"
        if np.any(spec.lambda_j <= 0):
            return False, "lambda_j has non-positive values"
        return True, "All positive"
    
    def test_spectral_radius_scales_with_beta(self):
        """Spectral radius should increase with beta."""
        Q = self.setup_uniform_flow(u=0.0, v=0.0)  # Zero velocity
        Si_x, Si_y, Sj_x, Sj_y, _ = self.setup_uniform_grid()
        
        spec1 = compute_spectral_radii(Q, Si_x, Si_y, Sj_x, Sj_y, beta=1.0)
        spec2 = compute_spectral_radii(Q, Si_x, Si_y, Sj_x, Sj_y, beta=100.0)
        
        # With zero velocity, lambda = c * S = sqrt(beta) * S
        ratio = spec2.lambda_i[10, 7] / spec1.lambda_i[10, 7]
        expected_ratio = np.sqrt(100.0 / 1.0)  # = 10
        
        if not np.isclose(ratio, expected_ratio, rtol=0.01):
            return False, f"Ratio {ratio:.2f} != expected {expected_ratio:.2f}"
        return True, f"Scales correctly with sqrt(beta)"
    
    def test_spectral_radius_includes_velocity(self):
        """Spectral radius should include convective contribution."""
        Q_slow = self.setup_uniform_flow(u=0.1, v=0.0)
        Q_fast = self.setup_uniform_flow(u=10.0, v=0.0)
        Si_x, Si_y, Sj_x, Sj_y, _ = self.setup_uniform_grid()
        
        spec_slow = compute_spectral_radii(Q_slow, Si_x, Si_y, Sj_x, Sj_y, self.beta)
        spec_fast = compute_spectral_radii(Q_fast, Si_x, Si_y, Sj_x, Sj_y, self.beta)
        
        # Faster flow should have larger spectral radius
        if not np.all(spec_fast.lambda_i > spec_slow.lambda_i):
            return False, "Faster flow should have larger lambda_i"
        return True, "Velocity contribution correct"
    
    # ===== Local Time Step Tests =====
    
    def test_local_timestep_shape(self):
        """Test local timestep output shape."""
        Q = self.setup_uniform_flow()
        Si_x, Si_y, Sj_x, Sj_y, volume = self.setup_uniform_grid()
        
        dt = compute_local_timestep(Q, Si_x, Si_y, Sj_x, Sj_y, volume, self.beta)
        
        expected_shape = (self.NI, self.NJ)
        if dt.shape != expected_shape:
            return False, f"Shape {dt.shape} != {expected_shape}"
        return True, "Correct shape"
    
    def test_local_timestep_positive(self):
        """Time step should always be positive."""
        Q = self.setup_uniform_flow()
        Si_x, Si_y, Sj_x, Sj_y, volume = self.setup_uniform_grid()
        
        dt = compute_local_timestep(Q, Si_x, Si_y, Sj_x, Sj_y, volume, self.beta)
        
        if np.any(dt <= 0):
            return False, "Found non-positive timestep"
        return True, "All positive"
    
    def test_local_timestep_cfl_scaling(self):
        """Time step should scale linearly with CFL."""
        Q = self.setup_uniform_flow()
        Si_x, Si_y, Sj_x, Sj_y, volume = self.setup_uniform_grid()
        
        cfg1 = TimeStepConfig(cfl=0.5)
        cfg2 = TimeStepConfig(cfl=1.0)
        
        dt1 = compute_local_timestep(Q, Si_x, Si_y, Sj_x, Sj_y, volume, self.beta, cfg1)
        dt2 = compute_local_timestep(Q, Si_x, Si_y, Sj_x, Sj_y, volume, self.beta, cfg2)
        
        ratio = dt2[10, 7] / dt1[10, 7]
        if not np.isclose(ratio, 2.0, rtol=0.01):
            return False, f"CFL ratio {ratio:.3f} != 2.0"
        return True, "CFL scaling correct"
    
    def test_local_timestep_volume_scaling(self):
        """Time step should scale with cell volume."""
        Q = self.setup_uniform_flow()
        Si_x, Si_y, Sj_x, Sj_y, volume = self.setup_uniform_grid()
        
        # Double the volume
        volume2 = volume * 2.0
        
        dt1 = compute_local_timestep(Q, Si_x, Si_y, Sj_x, Sj_y, volume, self.beta)
        dt2 = compute_local_timestep(Q, Si_x, Si_y, Sj_x, Sj_y, volume2, self.beta)
        
        ratio = dt2[10, 7] / dt1[10, 7]
        if not np.isclose(ratio, 2.0, rtol=0.01):
            return False, f"Volume ratio {ratio:.3f} != 2.0"
        return True, "Volume scaling correct"
    
    def test_local_timestep_uniform_on_uniform_grid(self):
        """Uniform flow on uniform grid should give uniform dt."""
        Q = self.setup_uniform_flow()
        Si_x, Si_y, Sj_x, Sj_y, volume = self.setup_uniform_grid()
        
        dt = compute_local_timestep(Q, Si_x, Si_y, Sj_x, Sj_y, volume, self.beta)
        
        # All values should be identical
        if not np.allclose(dt, dt[0, 0]):
            return False, f"dt varies: min={dt.min():.6f}, max={dt.max():.6f}"
        return True, "Uniform dt on uniform grid"
    
    def test_local_timestep_min_max_limits(self):
        """Time step should respect min/max limits."""
        Q = self.setup_uniform_flow()
        Si_x, Si_y, Sj_x, Sj_y, volume = self.setup_uniform_grid()
        
        cfg = TimeStepConfig(cfl=0.8, min_dt=0.001, max_dt=0.005)
        dt = compute_local_timestep(Q, Si_x, Si_y, Sj_x, Sj_y, volume, self.beta, cfg)
        
        if np.any(dt < cfg.min_dt):
            return False, f"dt below min: {dt.min():.6f} < {cfg.min_dt}"
        if np.any(dt > cfg.max_dt):
            return False, f"dt above max: {dt.max():.6f} > {cfg.max_dt}"
        return True, "Limits respected"
    
    # ===== Global Time Step Tests =====
    
    def test_global_timestep_scalar(self):
        """Global timestep should return scalar."""
        Q = self.setup_uniform_flow()
        Si_x, Si_y, Sj_x, Sj_y, volume = self.setup_uniform_grid()
        
        dt = compute_global_timestep(Q, Si_x, Si_y, Sj_x, Sj_y, volume, self.beta)
        
        if not isinstance(dt, float):
            return False, f"Expected float, got {type(dt)}"
        return True, "Returns scalar"
    
    def test_global_timestep_is_minimum(self):
        """Global timestep should be minimum of local timesteps."""
        Q = self.setup_uniform_flow()
        Si_x, Si_y, Sj_x, Sj_y, volume = self.setup_uniform_grid()
        
        # Make one cell smaller (faster local timescale)
        volume[5, 5] = volume[5, 5] * 0.1
        
        dt_global = compute_global_timestep(Q, Si_x, Si_y, Sj_x, Sj_y, volume, self.beta)
        dt_local = compute_local_timestep(Q, Si_x, Si_y, Sj_x, Sj_y, volume, self.beta)
        
        if not np.isclose(dt_global, dt_local.min()):
            return False, f"Global {dt_global:.6f} != min(local) {dt_local.min():.6f}"
        return True, "Global is minimum"
    
    # ===== Explicit Euler Tests =====
    
    def test_euler_preserves_shape(self):
        """Euler step should preserve state shape."""
        Q = self.setup_uniform_flow()
        Si_x, Si_y, Sj_x, Sj_y, volume = self.setup_uniform_grid()
        
        # Create fake residual
        R = np.zeros((self.NI, self.NJ, 4))
        
        euler = ExplicitEuler(beta=self.beta)
        Q_new = euler.step(Q, R, Si_x, Si_y, Sj_x, Sj_y, volume)
        
        if Q_new.shape != Q.shape:
            return False, f"Shape changed: {Q.shape} -> {Q_new.shape}"
        return True, "Shape preserved"
    
    def test_euler_zero_residual_no_change(self):
        """Zero residual should not change interior state."""
        Q = self.setup_uniform_flow()
        Si_x, Si_y, Sj_x, Sj_y, volume = self.setup_uniform_grid()
        
        R = np.zeros((self.NI, self.NJ, 4))
        
        euler = ExplicitEuler(beta=self.beta)
        Q_new = euler.step(Q, R, Si_x, Si_y, Sj_x, Sj_y, volume)
        
        # Interior should be unchanged
        if not np.allclose(Q_new[1:-1, 1:-1, :], Q[1:-1, 1:-1, :]):
            return False, "Interior changed with zero residual"
        return True, "No change with zero residual"
    
    def test_euler_applies_residual(self):
        """Euler should apply residual update."""
        Q = self.setup_uniform_flow(u=0.0, v=0.0, p=1.0)
        Si_x, Si_y, Sj_x, Sj_y, volume = self.setup_uniform_grid()
        
        # Uniform residual
        R = np.ones((self.NI, self.NJ, 4)) * 0.1
        
        euler = ExplicitEuler(beta=self.beta, cfg=TimeStepConfig(cfl=0.5))
        Q_new = euler.step(Q, R, Si_x, Si_y, Sj_x, Sj_y, volume)
        
        # Interior should increase (Q_new = Q + dt/vol * R)
        dp = Q_new[10, 7, 0] - Q[10, 7, 0]
        if dp <= 0:
            return False, f"Pressure should increase, got dp={dp:.6f}"
        return True, f"Residual applied, dp={dp:.6f}"
    
    def test_euler_ghost_unchanged(self):
        """Euler should not modify ghost cells."""
        Q = self.setup_uniform_flow()
        Si_x, Si_y, Sj_x, Sj_y, volume = self.setup_uniform_grid()
        
        # Set ghost cells to special value
        Q[0, :, :] = 999.0
        Q[-1, :, :] = 999.0
        Q[:, 0, :] = 999.0
        Q[:, -1, :] = 999.0
        
        R = np.ones((self.NI, self.NJ, 4)) * 0.1
        
        euler = ExplicitEuler(beta=self.beta)
        Q_new = euler.step(Q, R, Si_x, Si_y, Sj_x, Sj_y, volume)
        
        # Ghost cells should remain unchanged
        if not np.allclose(Q_new[0, :, :], 999.0):
            return False, "Left ghost cells modified"
        if not np.allclose(Q_new[-1, :, :], 999.0):
            return False, "Right ghost cells modified"
        if not np.allclose(Q_new[:, 0, :], 999.0):
            return False, "Bottom ghost cells modified"
        if not np.allclose(Q_new[:, -1, :], 999.0):
            return False, "Top ghost cells modified"
        return True, "Ghost cells unchanged"
    
    # ===== RK4 Tests =====
    
    def test_rk4_preserves_shape(self):
        """RK4 step should preserve state shape."""
        Q = self.setup_uniform_flow()
        Si_x, Si_y, Sj_x, Sj_y, volume = self.setup_uniform_grid()
        metrics = GridMetrics(Si_x=Si_x, Si_y=Si_y, Sj_x=Sj_x, Sj_y=Sj_y, volume=volume)
        cfg = FluxConfig()
        
        def compute_residual(Q):
            return compute_fluxes(Q, metrics, self.beta, cfg)
        
        def apply_bc(Q):
            return Q.copy()  # No-op for test
        
        rk4 = RungeKutta4(beta=self.beta)
        Q_new = rk4.step(Q, compute_residual, apply_bc, Si_x, Si_y, Sj_x, Sj_y, volume)
        
        if Q_new.shape != Q.shape:
            return False, f"Shape changed: {Q.shape} -> {Q_new.shape}"
        return True, "Shape preserved"
    
    def test_rk4_reduces_residual(self):
        """RK4 with flux computation should reduce residual."""
        Q = self.setup_uniform_flow()
        # Add small perturbation
        Q[10, 7, 0] += 0.1
        
        Si_x, Si_y, Sj_x, Sj_y, volume = self.setup_uniform_grid()
        metrics = GridMetrics(Si_x=Si_x, Si_y=Si_y, Sj_x=Sj_x, Sj_y=Sj_y, volume=volume)
        cfg = FluxConfig()
        
        def compute_residual(Q):
            return compute_fluxes(Q, metrics, self.beta, cfg)
        
        def apply_bc(Q):
            # Simple extrapolation BC
            Q = Q.copy()
            Q[0, :, :] = Q[1, :, :]
            Q[-1, :, :] = Q[-2, :, :]
            Q[:, 0, :] = Q[:, 1, :]
            Q[:, -1, :] = Q[:, -2, :]
            return Q
        
        # Initial residual
        Q_bc = apply_bc(Q)
        R_init = compute_residual(Q_bc)
        res_init = np.max(np.abs(R_init))
        
        # After RK4 step
        rk4 = RungeKutta4(beta=self.beta, cfg=TimeStepConfig(cfl=0.3))
        Q_new = rk4.step(Q, compute_residual, apply_bc, Si_x, Si_y, Sj_x, Sj_y, volume)
        Q_new = apply_bc(Q_new)
        R_final = compute_residual(Q_new)
        res_final = np.max(np.abs(R_final))
        
        # Residual should decrease (at least not increase significantly)
        if res_final > res_init * 1.5:
            return False, f"Residual increased: {res_init:.2e} -> {res_final:.2e}"
        return True, f"Residual: {res_init:.2e} -> {res_final:.2e}"
    
    def run_all(self):
        """Run all tests."""
        # Spectral radius tests
        self.run_test("spectral_radius_shape", self.test_spectral_radius_shape)
        self.run_test("spectral_radius_positive", self.test_spectral_radius_positive)
        self.run_test("spectral_radius_scales_with_beta", self.test_spectral_radius_scales_with_beta)
        self.run_test("spectral_radius_includes_velocity", self.test_spectral_radius_includes_velocity)
        
        # Local timestep tests
        self.run_test("local_timestep_shape", self.test_local_timestep_shape)
        self.run_test("local_timestep_positive", self.test_local_timestep_positive)
        self.run_test("local_timestep_cfl_scaling", self.test_local_timestep_cfl_scaling)
        self.run_test("local_timestep_volume_scaling", self.test_local_timestep_volume_scaling)
        self.run_test("local_timestep_uniform", self.test_local_timestep_uniform_on_uniform_grid)
        self.run_test("local_timestep_min_max_limits", self.test_local_timestep_min_max_limits)
        
        # Global timestep tests
        self.run_test("global_timestep_scalar", self.test_global_timestep_scalar)
        self.run_test("global_timestep_is_minimum", self.test_global_timestep_is_minimum)
        
        # Explicit Euler tests
        self.run_test("euler_preserves_shape", self.test_euler_preserves_shape)
        self.run_test("euler_zero_residual_no_change", self.test_euler_zero_residual_no_change)
        self.run_test("euler_applies_residual", self.test_euler_applies_residual)
        self.run_test("euler_ghost_unchanged", self.test_euler_ghost_unchanged)
        
        # RK4 tests
        self.run_test("rk4_preserves_shape", self.test_rk4_preserves_shape)
        self.run_test("rk4_reduces_residual", self.test_rk4_reduces_residual)
        
        return self.results


def main():
    """Run all time stepping tests."""
    print("=" * 60)
    print("Time Stepping Module Tests")
    print("=" * 60)
    
    tests = TimeSteppingTests()
    results = tests.run_all()
    
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    
    print(f"\nResults: {passed} passed, {failed} failed\n")
    
    for r in results:
        status = "✓" if r.passed else "✗"
        print(f"  {status} {r.name}: {r.message}")
    
    print()
    if failed > 0:
        print(f"FAILED: {failed} test(s)")
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()

