"""
Blasius flat plate validation test.

This test validates the viscous flux implementation against the 
analytical Blasius solution for laminar boundary layer.

Expected: Cf * sqrt(Re_x) = 0.664 (within 15% tolerance)
"""

import numpy as np
from numba import njit
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


@njit(cache=True)
def apply_bc_numba(Q, u_inf):
    """Apply flat plate boundary conditions (in-place, Numba optimized)."""
    NI_ghost = Q.shape[0]
    NJ_ghost = Q.shape[1]
    
    # Inlet (i=0): Dirichlet velocity
    for j in range(NJ_ghost):
        Q[0, j, 0] = Q[1, j, 0]  # Neumann p
        Q[0, j, 1] = 2.0 * u_inf - Q[1, j, 1]  # Dirichlet u
        Q[0, j, 2] = -Q[1, j, 2]  # Dirichlet v=0
        Q[0, j, 3] = Q[1, j, 3]  # Neumann nu_t
    
    # Outlet (i=-1): Zero gradient
    for j in range(NJ_ghost):
        for k in range(4):
            Q[NI_ghost-1, j, k] = Q[NI_ghost-2, j, k]
    
    # Wall (j=0): No-slip
    for i in range(NI_ghost):
        Q[i, 0, 0] = Q[i, 1, 0]   # Neumann p
        Q[i, 0, 1] = -Q[i, 1, 1]  # No-slip u
        Q[i, 0, 2] = -Q[i, 1, 2]  # No-slip v
        Q[i, 0, 3] = -Q[i, 1, 3]  # nu_t = 0 at wall
    
    # Top (j=-1): Zero gradient (allow outflow)
    for i in range(NI_ghost):
        for k in range(4):
            Q[i, NJ_ghost-1, k] = Q[i, NJ_ghost-2, k]


@njit(cache=True)
def rk4_update(Q, Q0, R, dt_over_vol, alpha):
    """RK4 substep update (Numba optimized)."""
    NI = R.shape[0]
    NJ = R.shape[1]
    
    for i in range(NI):
        for j in range(NJ):
            for k in range(4):
                Q[i+1, j+1, k] = Q0[i+1, j+1, k] + alpha * dt_over_vol[i, j] * R[i, j, k]


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
        delta_max = 5.0 * L / np.sqrt(Re)
        H = 3.0 * delta_max
        
        NI, NJ = 100, 30
        beta = 5.0
        cfl = 0.5
        max_iter = 8000  # Balance speed vs accuracy
        
        # Create grid
        X, Y = create_grid(NI, NJ, L, H)
        flux_met, grad_met, dx, dy = compute_metrics(X, Y)
        
        dt = cfl * min(dx, dy) / (1.0 + np.sqrt(beta))
        dt_over_vol = dt / flux_met.volume
        
        # Initialize
        Q = np.zeros((NI + 2, NJ + 2, 4))
        Q[:, :, 1] = 1.0
        apply_bc_numba(Q, 1.0)
        
        Q0 = np.zeros_like(Q)
        
        # Run solver
        flux_cfg = FluxConfig(k2=0.0, k4=0.002)
        alphas = np.array([0.25, 0.333333333, 0.5, 1.0])
        
        for _ in range(max_iter):
            Q0[:] = Q
            
            for alpha in alphas:
                apply_bc_numba(Q, 1.0)
                R = compute_fluxes(Q, flux_met, beta, flux_cfg)
                grad = compute_gradients(Q, grad_met)
                R = add_viscous_fluxes(R, Q, grad, grad_met, mu_laminar=nu)
                
                rk4_update(Q, Q0, R, dt_over_vol, alpha)
            
            apply_bc_numba(Q, 1.0)
        
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
        dt_over_vol = dt / flux_met.volume
        
        Q = np.zeros((NI + 2, NJ + 2, 4))
        Q[:, :, 1] = 1.0
        apply_bc_numba(Q, 1.0)
        
        Q0 = np.zeros_like(Q)
        flux_cfg = FluxConfig(k2=0.0, k4=0.002)
        alphas = np.array([0.25, 0.333333333, 0.5, 1.0])
        
        # Run 1000 steps
        for _ in range(1000):
            Q0[:] = Q
            for alpha in alphas:
                apply_bc_numba(Q, 1.0)
                R = compute_fluxes(Q, flux_met, beta, flux_cfg)
                grad = compute_gradients(Q, grad_met)
                R = add_viscous_fluxes(R, Q, grad, grad_met, mu_laminar=nu)
                rk4_update(Q, Q0, R, dt_over_vol, alpha)
            apply_bc_numba(Q, 1.0)
        
        # Check no NaN or Inf
        assert not np.any(np.isnan(Q)), "Solution contains NaN"
        assert not np.any(np.isinf(Q)), "Solution contains Inf"
        
        # Check velocity is reasonable
        assert Q[1:-1, 1:-1, 1].max() < 2.0, "u velocity unreasonably high"
        assert Q[1:-1, 1:-1, 1].min() > -0.5, "u velocity unreasonably negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
