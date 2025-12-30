"""
Blasius flat plate validation test.

This test validates the viscous flux implementation against the 
analytical Blasius solution for laminar boundary layer.

Expected: Cf * sqrt(Re_x) = 0.664 (within 15% tolerance)
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.numerics.gradients import compute_gradients, GradientMetrics
from src.numerics.fluxes import compute_fluxes, FluxConfig, GridMetrics as FluxGridMetrics
from src.numerics.viscous_fluxes import add_viscous_fluxes


def create_grid(NI, NJ, L, H):
    """Create uniform Cartesian grid."""
    x = np.linspace(0, L, NI + 1)
    y = np.linspace(0, H, NJ + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return X, Y


def compute_metrics(X, Y):
    """Compute FVM metrics."""
    NI, NJ = X.shape[0] - 1, X.shape[1] - 1
    dx = X[1, 0] - X[0, 0]
    dy = Y[0, 1] - Y[0, 0]
    
    Si_x = np.full((NI + 1, NJ), dy)
    Si_y = np.zeros((NI + 1, NJ))
    Sj_x = np.zeros((NI, NJ + 1))
    Sj_y = np.full((NI, NJ + 1), dx)
    volume = np.full((NI, NJ), dx * dy)
    
    return FluxGridMetrics(Si_x, Si_y, Sj_x, Sj_y, volume), \
           GradientMetrics(Si_x, Si_y, Sj_x, Sj_y, volume), dx, dy


def apply_bc(Q, u_inf=1.0):
    """Apply flat plate boundary conditions."""
    Q = Q.copy()
    # Inlet: Dirichlet
    Q[0, :, 1] = 2 * u_inf - Q[1, :, 1]
    Q[0, :, 2] = -Q[1, :, 2]
    Q[0, :, 0] = Q[1, :, 0]
    # Outlet: Zero gradient
    Q[-1, :, :] = Q[-2, :, :]
    # Wall: No-slip
    Q[:, 0, 0] = Q[:, 1, 0]
    Q[:, 0, 1] = -Q[:, 1, 1]
    Q[:, 0, 2] = -Q[:, 1, 2]
    # Top: Zero gradient (allow outflow)
    Q[:, -1, :] = Q[:, -2, :]
    return Q


class TestBlasiusFlatPlate:
    """Validate viscous solver against Blasius solution."""
    
    def test_skin_friction(self):
        """
        Test that Cf*sqrt(Re_x) ≈ 0.664 (Blasius).
        
        Uses coarse grid (100x30) for fast CI execution.
        Validates only in x ∈ [0.2L, 0.8L] to avoid boundary effects.
        """
        # Parameters
        Re = 100000
        nu = 1.0 / Re
        L = 1.0
        delta_max = 5.0 * L / np.sqrt(Re)  # ~0.016
        H = 3.0 * delta_max  # ~0.047
        
        NI, NJ = 100, 30
        beta = 5.0
        cfl = 0.5
        max_iter = 10000
        
        # Create grid
        X, Y = create_grid(NI, NJ, L, H)
        flux_met, grad_met, dx, dy = compute_metrics(X, Y)
        
        dt = cfl * min(dx, dy) / (1.0 + np.sqrt(beta))
        
        # Initialize
        Q = np.zeros((NI + 2, NJ + 2, 4))
        Q[:, :, 1] = 1.0
        Q = apply_bc(Q)
        
        # Run solver
        flux_cfg = FluxConfig(k2=0.0, k4=0.002)
        
        for _ in range(max_iter):
            Q0 = Q.copy()
            Qk = Q.copy()
            
            for alpha in [0.25, 0.333, 0.5, 1.0]:
                Qk = apply_bc(Qk)
                R = compute_fluxes(Qk, flux_met, beta, flux_cfg)
                grad = compute_gradients(Qk, grad_met)
                R = add_viscous_fluxes(R, Qk, grad, grad_met, mu_laminar=nu)
                
                Qk = Q0.copy()
                Qk[1:-1, 1:-1, :] += alpha * dt / flux_met.volume[:, :, np.newaxis] * R
            
            Q = apply_bc(Qk)
        
        # Compute Cf
        u_wall = Q[1:-1, 1, 1]
        y_first = 0.5 * dy
        tau_w = nu * u_wall / y_first
        Cf = 2.0 * tau_w
        
        x_wall = 0.5 * (X[:-1, 0] + X[1:, 0])
        Re_x = Re * x_wall
        cf_rex = Cf * np.sqrt(Re_x + 1e-12)
        
        # Validate in [0.2L, 0.8L]
        mask = (x_wall > 0.2) & (x_wall < 0.8)
        mean_cf_rex = cf_rex[mask].mean()
        
        # Check within 15% of Blasius value 0.664
        error = abs(mean_cf_rex - 0.664) / 0.664
        assert error < 0.15, f"Cf*sqrt(Re_x) = {mean_cf_rex:.4f}, error = {error*100:.1f}% (expected < 15%)"
    
    def test_no_divergence(self):
        """Test that solver doesn't diverge on flat plate."""
        Re = 100000
        nu = 1.0 / Re
        L, H = 1.0, 0.05
        NI, NJ = 50, 20
        beta = 5.0
        
        X, Y = create_grid(NI, NJ, L, H)
        flux_met, grad_met, dx, dy = compute_metrics(X, Y)
        dt = 0.5 * min(dx, dy) / (1.0 + np.sqrt(beta))
        
        Q = np.zeros((NI + 2, NJ + 2, 4))
        Q[:, :, 1] = 1.0
        Q = apply_bc(Q)
        
        flux_cfg = FluxConfig(k2=0.0, k4=0.002)
        
        # Run 1000 steps
        for _ in range(1000):
            Q0 = Q.copy()
            Qk = Q.copy()
            for alpha in [0.25, 0.333, 0.5, 1.0]:
                Qk = apply_bc(Qk)
                R = compute_fluxes(Qk, flux_met, beta, flux_cfg)
                grad = compute_gradients(Qk, grad_met)
                R = add_viscous_fluxes(R, Qk, grad, grad_met, mu_laminar=nu)
                Qk = Q0.copy()
                Qk[1:-1, 1:-1, :] += alpha * dt / flux_met.volume[:, :, np.newaxis] * R
            Q = apply_bc(Qk)
        
        # Check no NaN or Inf
        assert not np.any(np.isnan(Q)), "Solution contains NaN"
        assert not np.any(np.isinf(Q)), "Solution contains Inf"
        
        # Check velocity is reasonable
        assert Q[1:-1, 1:-1, 1].max() < 2.0, "u velocity unreasonably high"
        assert Q[1:-1, 1:-1, 1].min() > -0.5, "u velocity unreasonably negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

