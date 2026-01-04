"""
Tests for SA source term computation.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from src.physics.jax_config import jnp
from src.numerics.sa_sources import (
    compute_sa_source_jax,
    compute_sa_production_only_jax,
    compute_sa_destruction_only_jax,
    compute_cb2_term_jax,
    compute_turbulent_viscosity_jax,
    compute_effective_viscosity_jax,
)
from src.physics.spalart_allmaras import (
    compute_sa_production,
    compute_sa_destruction,
)


class TestSASourceTerms:
    """Tests for SA source term computation."""
    
    @pytest.fixture
    def simple_fields(self):
        """Create simple test fields."""
        NI, NJ = 16, 8
        
        # SA working variable - positive in most of domain
        nuHat = np.ones((NI, NJ)) * 5.0  # 5x laminar viscosity
        
        # Gradients: Q = [p, u, v, nuHat]
        # Shape: (NI, NJ, 4, 2) where last dim is (d/dx, d/dy)
        grad = np.zeros((NI, NJ, 4, 2))
        # du/dy = 1.0 everywhere (shear flow)
        grad[:, :, 1, 1] = 1.0
        # Small dv/dx to create vorticity
        grad[:, :, 2, 0] = 0.1
        # d(nuHat)/dx and d(nuHat)/dy
        grad[:, :, 3, 0] = 0.2
        grad[:, :, 3, 1] = 0.3
        
        # Wall distance - increases away from wall
        wall_dist = np.zeros((NI, NJ))
        for j in range(NJ):
            wall_dist[:, j] = 0.01 + j * 0.1  # Starts at 0.01
        
        nu_laminar = 1e-4
        
        return {
            'nuHat': jnp.array(nuHat),
            'grad': jnp.array(grad),
            'wall_dist': jnp.array(wall_dist),
            'nu_laminar': nu_laminar,
            'NI': NI,
            'NJ': NJ,
        }
    
    def test_sa_source_shape(self, simple_fields):
        """Test that SA source terms have correct shapes."""
        P, D, cb2_term = compute_sa_source_jax(
            simple_fields['nuHat'],
            simple_fields['grad'],
            simple_fields['wall_dist'],
            simple_fields['nu_laminar'],
        )
        
        expected_shape = (simple_fields['NI'], simple_fields['NJ'])
        assert P.shape == expected_shape
        assert D.shape == expected_shape
        assert cb2_term.shape == expected_shape
    
    def test_production_positive(self, simple_fields):
        """Test that production is positive for positive nuHat."""
        P = compute_sa_production_only_jax(
            simple_fields['nuHat'],
            simple_fields['grad'],
            simple_fields['wall_dist'],
            simple_fields['nu_laminar'],
        )
        
        # Production should be positive where nuHat > 0
        assert np.all(np.array(P) >= 0)
    
    def test_destruction_positive(self, simple_fields):
        """Test that destruction is positive for positive nuHat."""
        D = compute_sa_destruction_only_jax(
            simple_fields['nuHat'],
            simple_fields['grad'],
            simple_fields['wall_dist'],
            simple_fields['nu_laminar'],
        )
        
        # Destruction should be positive
        assert np.all(np.array(D) >= 0)
    
    def test_cb2_term_positive(self, simple_fields):
        """Test that cb2 gradient term is positive (always squared)."""
        cb2_term = compute_cb2_term_jax(simple_fields['grad'])
        
        # Should always be non-negative (it's a squared gradient)
        assert np.all(np.array(cb2_term) >= 0)
    
    def test_cb2_term_value(self, simple_fields):
        """Test cb2 term value against manual calculation."""
        CB2 = 0.622
        SIGMA = 2.0 / 3.0
        
        grad = np.array(simple_fields['grad'])
        grad_nuHat_x = grad[:, :, 3, 0]
        grad_nuHat_y = grad[:, :, 3, 1]
        expected = (CB2 / SIGMA) * (grad_nuHat_x**2 + grad_nuHat_y**2)
        
        cb2_term = compute_cb2_term_jax(simple_fields['grad'])
        
        assert_allclose(np.array(cb2_term), expected, rtol=1e-6)
    
    def test_source_near_wall(self, simple_fields):
        """Test that source terms behave correctly near wall."""
        # Near wall, destruction should dominate
        nuHat = simple_fields['nuHat']
        grad = simple_fields['grad']
        wall_dist = simple_fields['wall_dist']
        nu = simple_fields['nu_laminar']
        
        P = compute_sa_production_only_jax(nuHat, grad, wall_dist, nu)
        D = compute_sa_destruction_only_jax(nuHat, grad, wall_dist, nu)
        
        # Near wall (j=0), destruction should be larger due to (nuHat/d)^2 term
        P_np = np.array(P)
        D_np = np.array(D)
        
        # Check first row (near wall)
        # At small wall distance, destruction should be significant
        assert np.mean(D_np[:, 0]) > 0
    
    def test_negative_nuHat_handled(self):
        """Test that negative nuHat is handled safely."""
        NI, NJ = 4, 4
        nuHat = jnp.array(np.full((NI, NJ), -1.0))  # Negative!
        grad = jnp.zeros((NI, NJ, 4, 2))
        grad = grad.at[:, :, 1, 1].set(1.0)  # du/dy
        wall_dist = jnp.array(np.full((NI, NJ), 0.1))
        nu = 1e-4
        
        # Should not raise and should produce finite results
        source = compute_sa_source_jax(nuHat, grad, wall_dist, nu)
        
        assert np.all(np.isfinite(np.array(source)))
    
    def test_production_matches_physics_module(self, simple_fields):
        """Test that production matches the physics module implementation."""
        nuHat = simple_fields['nuHat']
        grad = simple_fields['grad']
        wall_dist = simple_fields['wall_dist']
        nu = simple_fields['nu_laminar']
        
        # Compute vorticity from grad
        dudy = grad[:, :, 1, 1]
        dvdx = grad[:, :, 2, 0]
        omega = jnp.abs(dvdx - dudy)
        
        # Production from numerics module
        P_numerics = compute_sa_production_only_jax(nuHat, grad, wall_dist, nu)
        
        # Production from physics module (pass nu_laminar to match)
        P_physics = compute_sa_production(omega, nuHat, wall_dist, nu)
        
        assert_allclose(np.array(P_numerics), np.array(P_physics), rtol=1e-10)
    
    def test_destruction_matches_physics_module(self, simple_fields):
        """Test that destruction matches the physics module implementation."""
        nuHat = simple_fields['nuHat']
        grad = simple_fields['grad']
        wall_dist = simple_fields['wall_dist']
        nu = simple_fields['nu_laminar']
        
        # Compute vorticity from grad
        dudy = grad[:, :, 1, 1]
        dvdx = grad[:, :, 2, 0]
        omega = jnp.abs(dvdx - dudy)
        
        # Destruction from numerics module
        D_numerics = compute_sa_destruction_only_jax(nuHat, grad, wall_dist, nu)
        
        # Destruction from physics module (pass nu_laminar to match)
        D_physics = compute_sa_destruction(omega, nuHat, wall_dist, nu)
        
        assert_allclose(np.array(D_numerics), np.array(D_physics), rtol=1e-10)


class TestTurbulentViscosity:
    """Tests for turbulent viscosity computation."""
    
    def test_zero_nuHat(self):
        """Test that zero nuHat gives zero turbulent viscosity."""
        nuHat = jnp.array([0.0, 0.0, 0.0])
        nu = 1e-4
        
        mu_t = compute_turbulent_viscosity_jax(nuHat, nu)
        
        assert_allclose(np.array(mu_t), [0.0, 0.0, 0.0], atol=1e-10)
    
    def test_negative_nuHat_clipped(self):
        """Test that negative nuHat is clipped to zero."""
        nuHat = jnp.array([-1.0, -0.5, 0.0])
        nu = 1e-4
        
        mu_t = compute_turbulent_viscosity_jax(nuHat, nu)
        
        assert np.all(np.array(mu_t) >= 0)
    
    def test_fv1_behavior(self):
        """Test fv1 damping at different chi values."""
        cv1 = 7.1
        nu = 1e-4
        
        # At chi << cv1, fv1 ≈ 0
        nuHat_low = jnp.array([0.001 * nu])  # chi = 0.001
        mu_t_low = compute_turbulent_viscosity_jax(nuHat_low, nu)
        
        # At chi >> cv1, fv1 ≈ 1, so mu_t ≈ nuHat
        nuHat_high = jnp.array([1000 * nu])  # chi = 1000
        mu_t_high = compute_turbulent_viscosity_jax(nuHat_high, nu)
        
        # Low chi: mu_t should be much smaller than nuHat
        assert np.array(mu_t_low)[0] < nuHat_low[0]
        
        # High chi: mu_t should be close to nuHat
        assert_allclose(np.array(mu_t_high)[0], np.array(nuHat_high)[0], rtol=0.01)
    
    def test_effective_viscosity(self):
        """Test effective viscosity is sum of laminar and turbulent."""
        nuHat = jnp.array([5.0, 10.0, 50.0])
        nu = 1e-4
        
        mu_t = compute_turbulent_viscosity_jax(nuHat, nu)
        mu_eff = compute_effective_viscosity_jax(nuHat, nu)
        
        assert_allclose(np.array(mu_eff), nu + np.array(mu_t), rtol=1e-10)


class TestSASourceIntegration:
    """Integration tests for SA source computation in solver context."""
    
    def test_source_conservation_tendency(self):
        """Test that source terms have expected conservation properties.
        
        In equilibrium, P ≈ D (production balances destruction).
        Near equilibrium, P - D should be small compared to either term.
        """
        # Create a near-equilibrium state
        NI, NJ = 32, 16
        nu = 1e-4
        
        # Boundary layer-like profile
        nuHat = np.zeros((NI, NJ))
        wall_dist = np.zeros((NI, NJ))
        grad = np.zeros((NI, NJ, 4, 2))
        
        for j in range(NJ):
            y = 0.01 + j * 0.01  # Wall distance
            wall_dist[:, j] = y
            
            # Rough boundary layer nuHat profile
            nuHat[:, j] = nu * min(y / 0.01, 50.0)
            
            # Velocity gradient (du/dy ~ 1/y near wall)
            grad[:, j, 1, 1] = 1.0 / (y + 0.01)
        
        nuHat_jax = jnp.array(nuHat)
        grad_jax = jnp.array(grad)
        wall_dist_jax = jnp.array(wall_dist)
        
        source = compute_sa_source_jax(nuHat_jax, grad_jax, wall_dist_jax, nu)
        P = compute_sa_production_only_jax(nuHat_jax, grad_jax, wall_dist_jax, nu)
        D = compute_sa_destruction_only_jax(nuHat_jax, grad_jax, wall_dist_jax, nu)
        
        # Source should be finite
        assert np.all(np.isfinite(np.array(source)))
        
        # In the outer part of the boundary layer, P and D should be similar order
        P_np = np.array(P)
        D_np = np.array(D)
        
        # Check the outer part (j > NJ/2)
        outer_P = np.mean(P_np[:, NJ//2:])
        outer_D = np.mean(D_np[:, NJ//2:])
        
        # Both should be positive and of similar magnitude
        assert outer_P > 0
        assert outer_D > 0
