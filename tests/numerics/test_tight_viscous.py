"""Tests for tight-stencil viscous flux computation."""

import numpy as np
import pytest
from src.physics.jax_config import jnp
from src.grid.metrics import MetricComputer
from src.numerics.viscous_fluxes import (
    compute_viscous_fluxes_tight_jax,
    compute_viscous_fluxes_jax,
    compute_nu_tilde_diffusion_jax,
)


def create_skewed_grid(NI, NJ, skew_factor=0.15):
    """Create a non-Cartesian skewed grid for testing."""
    x_base = np.linspace(0, 2, NI + 1)
    y_base = np.linspace(0, 1.5, NJ + 1)
    X_base, Y_base = np.meshgrid(x_base, y_base, indexing='ij')
    
    X = X_base + skew_factor * np.sin(2 * np.pi * Y_base / 1.5)
    Y = Y_base + skew_factor * np.sin(np.pi * X_base / 2)
    
    return X, Y


class TestTightStencilViscousFlux:
    """Tests for tight-stencil viscous flux computation."""
    
    @pytest.fixture
    def skewed_grid_setup(self):
        """Create a skewed (non-Cartesian) grid with all necessary metrics."""
        NI, NJ = 8, 6
        X, Y = create_skewed_grid(NI, NJ, skew_factor=0.15)
        
        # Compute metrics
        computer = MetricComputer(X, Y)
        metrics = computer.compute()
        face_geom = computer.compute_face_geometry()
        ls_weights = computer.compute_ls_weights(face_geom)
        
        return {
            'NI': NI, 'NJ': NJ,
            'X': X, 'Y': Y,
            'metrics': metrics,
            'face_geom': face_geom,
            'ls_weights': ls_weights,
        }
    
    def test_tight_stencil_zero_velocity_skewed(self, skewed_grid_setup):
        """For zero velocity on skewed grid, viscous flux should be zero."""
        setup = skewed_grid_setup
        NI, NJ = setup['NI'], setup['NJ']
        
        # Zero velocity field
        Q = np.zeros((NI, NJ, 4))
        Q[:, :, 0] = 1.0  # Constant pressure
        
        mu_eff = np.ones((NI, NJ)) * 0.001
        nu_laminar = 1e-5
        nuHat = np.zeros((NI, NJ))
        
        # Convert to JAX
        Q_jax = jnp.asarray(Q)
        Si_x = jnp.asarray(setup['metrics'].Si_x)
        Si_y = jnp.asarray(setup['metrics'].Si_y)
        Sj_x = jnp.asarray(setup['metrics'].Sj_x)
        Sj_y = jnp.asarray(setup['metrics'].Sj_y)
        
        fg = setup['face_geom']
        lsw = setup['ls_weights']
        
        residual = compute_viscous_fluxes_tight_jax(
            Q_jax, Si_x, Si_y, Sj_x, Sj_y,
            jnp.asarray(fg.d_coord_i), jnp.asarray(fg.e_coord_i_x), jnp.asarray(fg.e_coord_i_y),
            jnp.asarray(fg.e_ortho_i_x), jnp.asarray(fg.e_ortho_i_y),
            jnp.asarray(fg.d_coord_j), jnp.asarray(fg.e_coord_j_x), jnp.asarray(fg.e_coord_j_y),
            jnp.asarray(fg.e_ortho_j_x), jnp.asarray(fg.e_ortho_j_y),
            jnp.asarray(lsw.weights_i), jnp.asarray(lsw.weights_j),
            jnp.asarray(mu_eff), nu_laminar, jnp.asarray(nuHat)
        )
        
        # All viscous residuals should be zero for uniform field
        np.testing.assert_allclose(np.asarray(residual), 0.0, atol=1e-10)
    
    def test_tight_stencil_uniform_velocity_skewed(self, skewed_grid_setup):
        """For uniform velocity on skewed grid, interior viscous flux should be zero."""
        setup = skewed_grid_setup
        NI, NJ = setup['NI'], setup['NJ']
        
        # Uniform velocity field
        Q = np.zeros((NI, NJ, 4))
        Q[:, :, 0] = 1.0  # pressure
        Q[:, :, 1] = 2.0  # u = 2
        Q[:, :, 2] = 1.5  # v = 1.5
        Q[:, :, 3] = 0.01  # nuHat
        
        mu_eff = np.ones((NI, NJ)) * 0.001
        nu_laminar = 1e-5
        nuHat = Q[:, :, 3]
        
        Q_jax = jnp.asarray(Q)
        Si_x = jnp.asarray(setup['metrics'].Si_x)
        Si_y = jnp.asarray(setup['metrics'].Si_y)
        Sj_x = jnp.asarray(setup['metrics'].Sj_x)
        Sj_y = jnp.asarray(setup['metrics'].Sj_y)
        
        fg = setup['face_geom']
        lsw = setup['ls_weights']
        
        residual = compute_viscous_fluxes_tight_jax(
            Q_jax, Si_x, Si_y, Sj_x, Sj_y,
            jnp.asarray(fg.d_coord_i), jnp.asarray(fg.e_coord_i_x), jnp.asarray(fg.e_coord_i_y),
            jnp.asarray(fg.e_ortho_i_x), jnp.asarray(fg.e_ortho_i_y),
            jnp.asarray(fg.d_coord_j), jnp.asarray(fg.e_coord_j_x), jnp.asarray(fg.e_coord_j_y),
            jnp.asarray(fg.e_ortho_j_x), jnp.asarray(fg.e_ortho_j_y),
            jnp.asarray(lsw.weights_i), jnp.asarray(lsw.weights_j),
            jnp.asarray(mu_eff), nu_laminar, jnp.asarray(nuHat)
        )
        
        # Interior residuals should be zero for uniform velocity
        # (boundary cells may have non-zero residual due to edge padding)
        interior = slice(2, -2)
        np.testing.assert_allclose(np.asarray(residual)[interior, interior, :], 0.0, atol=1e-10)
    
    def test_tight_stencil_linear_velocity_skewed(self, skewed_grid_setup):
        """For linear velocity profile on skewed grid, verify interior residual is zero."""
        setup = skewed_grid_setup
        NI, NJ = setup['NI'], setup['NJ']
        X, Y = setup['X'], setup['Y']
        
        # Create cell centers
        xc = 0.25 * (X[:-1, :-1] + X[1:, :-1] + X[1:, 1:] + X[:-1, 1:])
        yc = 0.25 * (Y[:-1, :-1] + Y[1:, :-1] + Y[1:, 1:] + Y[:-1, 1:])
        
        # Linear velocity profile: u = y, v = x
        # For linear profile, second derivative is zero, so interior residual should be ~0
        Q = np.zeros((NI, NJ, 4))
        Q[:, :, 0] = 1.0  # pressure
        Q[:, :, 1] = yc   # u = y
        Q[:, :, 2] = xc   # v = x
        Q[:, :, 3] = 0.0  # nuHat
        
        mu = 0.01
        mu_eff = np.ones((NI, NJ)) * mu
        nu_laminar = 1e-5
        nuHat = Q[:, :, 3]
        
        Q_jax = jnp.asarray(Q)
        Si_x = jnp.asarray(setup['metrics'].Si_x)
        Si_y = jnp.asarray(setup['metrics'].Si_y)
        Sj_x = jnp.asarray(setup['metrics'].Sj_x)
        Sj_y = jnp.asarray(setup['metrics'].Sj_y)
        
        fg = setup['face_geom']
        lsw = setup['ls_weights']
        
        residual = compute_viscous_fluxes_tight_jax(
            Q_jax, Si_x, Si_y, Sj_x, Sj_y,
            jnp.asarray(fg.d_coord_i), jnp.asarray(fg.e_coord_i_x), jnp.asarray(fg.e_coord_i_y),
            jnp.asarray(fg.e_ortho_i_x), jnp.asarray(fg.e_ortho_i_y),
            jnp.asarray(fg.d_coord_j), jnp.asarray(fg.e_coord_j_x), jnp.asarray(fg.e_coord_j_y),
            jnp.asarray(fg.e_ortho_j_x), jnp.asarray(fg.e_ortho_j_y),
            jnp.asarray(lsw.weights_i), jnp.asarray(lsw.weights_j),
            jnp.asarray(mu_eff), nu_laminar, jnp.asarray(nuHat)
        )
        
        residual_np = np.asarray(residual)
        
        # Interior cells should have near-zero residual (linear profile -> zero Laplacian)
        interior_u = residual_np[2:-2, 2:-2, 1]
        interior_v = residual_np[2:-2, 2:-2, 2]
        
        np.testing.assert_allclose(interior_u, 0.0, atol=1e-6)
        np.testing.assert_allclose(interior_v, 0.0, atol=1e-6)
    
    def test_tight_stencil_sa_diffusion_constant_skewed(self, skewed_grid_setup):
        """Verify SA diffusion term is zero for constant nuHat on skewed grid."""
        setup = skewed_grid_setup
        NI, NJ = setup['NI'], setup['NJ']
        
        # Constant nuHat profile
        Q = np.zeros((NI, NJ, 4))
        Q[:, :, 0] = 1.0
        Q[:, :, 1] = 0.0
        Q[:, :, 2] = 0.0
        Q[:, :, 3] = 0.01  # Constant nuHat
        
        mu_eff = np.ones((NI, NJ)) * 0.001
        nu_laminar = 1e-5
        nuHat = Q[:, :, 3]
        
        Q_jax = jnp.asarray(Q)
        Si_x = jnp.asarray(setup['metrics'].Si_x)
        Si_y = jnp.asarray(setup['metrics'].Si_y)
        Sj_x = jnp.asarray(setup['metrics'].Sj_x)
        Sj_y = jnp.asarray(setup['metrics'].Sj_y)
        
        fg = setup['face_geom']
        lsw = setup['ls_weights']
        
        residual = compute_viscous_fluxes_tight_jax(
            Q_jax, Si_x, Si_y, Sj_x, Sj_y,
            jnp.asarray(fg.d_coord_i), jnp.asarray(fg.e_coord_i_x), jnp.asarray(fg.e_coord_i_y),
            jnp.asarray(fg.e_ortho_i_x), jnp.asarray(fg.e_ortho_i_y),
            jnp.asarray(fg.d_coord_j), jnp.asarray(fg.e_coord_j_x), jnp.asarray(fg.e_coord_j_y),
            jnp.asarray(fg.e_ortho_j_x), jnp.asarray(fg.e_ortho_j_y),
            jnp.asarray(lsw.weights_i), jnp.asarray(lsw.weights_j),
            jnp.asarray(mu_eff), nu_laminar, jnp.asarray(nuHat)
        )
        
        residual_np = np.asarray(residual)
        
        # For constant nuHat, interior SA diffusion residual should be zero
        interior = slice(2, -2)
        np.testing.assert_allclose(residual_np[interior, interior, 3], 0.0, atol=1e-10)
    
    def test_tight_stencil_pressure_no_diffusion_skewed(self, skewed_grid_setup):
        """Verify pressure (index 0) has no diffusion on skewed grid."""
        setup = skewed_grid_setup
        NI, NJ = setup['NI'], setup['NJ']
        X, Y = setup['X'], setup['Y']
        
        # Create cell centers
        xc = 0.25 * (X[:-1, :-1] + X[1:, :-1] + X[1:, 1:] + X[:-1, 1:])
        yc = 0.25 * (Y[:-1, :-1] + Y[1:, :-1] + Y[1:, 1:] + Y[:-1, 1:])
        
        # Quadratic pressure field (would have non-zero diffusion if present)
        Q = np.zeros((NI, NJ, 4))
        Q[:, :, 0] = xc**2 + yc**2
        Q[:, :, 1] = xc
        Q[:, :, 2] = yc
        Q[:, :, 3] = 0.01
        
        mu_eff = np.ones((NI, NJ)) * 0.01
        nu_laminar = 1e-5
        nuHat = Q[:, :, 3]
        
        Q_jax = jnp.asarray(Q)
        Si_x = jnp.asarray(setup['metrics'].Si_x)
        Si_y = jnp.asarray(setup['metrics'].Si_y)
        Sj_x = jnp.asarray(setup['metrics'].Sj_x)
        Sj_y = jnp.asarray(setup['metrics'].Sj_y)
        
        fg = setup['face_geom']
        lsw = setup['ls_weights']
        
        residual = compute_viscous_fluxes_tight_jax(
            Q_jax, Si_x, Si_y, Sj_x, Sj_y,
            jnp.asarray(fg.d_coord_i), jnp.asarray(fg.e_coord_i_x), jnp.asarray(fg.e_coord_i_y),
            jnp.asarray(fg.e_ortho_i_x), jnp.asarray(fg.e_ortho_i_y),
            jnp.asarray(fg.d_coord_j), jnp.asarray(fg.e_coord_j_x), jnp.asarray(fg.e_coord_j_y),
            jnp.asarray(fg.e_ortho_j_x), jnp.asarray(fg.e_ortho_j_y),
            jnp.asarray(lsw.weights_i), jnp.asarray(lsw.weights_j),
            jnp.asarray(mu_eff), nu_laminar, jnp.asarray(nuHat)
        )
        
        residual_np = np.asarray(residual)
        
        # Pressure residual should always be zero
        np.testing.assert_allclose(residual_np[:, :, 0], 0.0, atol=1e-15)
    
    def test_stencil_isolation_inf_outside(self, skewed_grid_setup):
        """Verify that values outside the 6-cell stencil don't affect result.
        
        Set all values outside the stencil to inf and verify they don't pollute
        the computation for interior cells.
        """
        setup = skewed_grid_setup
        NI, NJ = setup['NI'], setup['NJ']
        X, Y = setup['X'], setup['Y']
        
        # Create cell centers for a well-defined field
        xc = 0.25 * (X[:-1, :-1] + X[1:, :-1] + X[1:, 1:] + X[:-1, 1:])
        yc = 0.25 * (Y[:-1, :-1] + Y[1:, :-1] + Y[1:, 1:] + Y[:-1, 1:])
        
        # Reference computation with normal values
        Q_ref = np.zeros((NI, NJ, 4))
        Q_ref[:, :, 0] = 1.0
        Q_ref[:, :, 1] = xc + yc  # Linear velocity
        Q_ref[:, :, 2] = xc - yc
        Q_ref[:, :, 3] = 0.02
        
        mu_eff = np.ones((NI, NJ)) * 0.01
        nu_laminar = 1e-5
        nuHat = Q_ref[:, :, 3].copy()
        
        Si_x = jnp.asarray(setup['metrics'].Si_x)
        Si_y = jnp.asarray(setup['metrics'].Si_y)
        Sj_x = jnp.asarray(setup['metrics'].Sj_x)
        Sj_y = jnp.asarray(setup['metrics'].Sj_y)
        
        fg = setup['face_geom']
        lsw = setup['ls_weights']
        
        # Compute reference residual
        residual_ref = compute_viscous_fluxes_tight_jax(
            jnp.asarray(Q_ref), Si_x, Si_y, Sj_x, Sj_y,
            jnp.asarray(fg.d_coord_i), jnp.asarray(fg.e_coord_i_x), jnp.asarray(fg.e_coord_i_y),
            jnp.asarray(fg.e_ortho_i_x), jnp.asarray(fg.e_ortho_i_y),
            jnp.asarray(fg.d_coord_j), jnp.asarray(fg.e_coord_j_x), jnp.asarray(fg.e_coord_j_y),
            jnp.asarray(fg.e_ortho_j_x), jnp.asarray(fg.e_ortho_j_y),
            jnp.asarray(lsw.weights_i), jnp.asarray(lsw.weights_j),
            jnp.asarray(mu_eff), nu_laminar, jnp.asarray(nuHat)
        )
        residual_ref_np = np.asarray(residual_ref)
        
        # Now create a field with inf at corners (outside stencil of center cells)
        Q_inf = Q_ref.copy()
        # For center cell (NI//2, NJ//2), the 6-cell stencil spans:
        # I-face: cells (i-1:i+1, j-1:j+2) -> roughly (3:5, 2:5) for NI=8, NJ=6
        # J-face: cells (i-1:i+2, j-1:j+1) -> roughly (2:5, 2:4)
        # So cells at corners (0,0), (0,NJ-1), (NI-1,0), (NI-1,NJ-1) should not affect center
        Q_inf[0, 0, :] = np.inf
        Q_inf[0, -1, :] = np.inf
        Q_inf[-1, 0, :] = np.inf
        Q_inf[-1, -1, :] = np.inf
        
        mu_eff_inf = mu_eff.copy()
        mu_eff_inf[0, 0] = np.inf
        mu_eff_inf[0, -1] = np.inf
        mu_eff_inf[-1, 0] = np.inf
        mu_eff_inf[-1, -1] = np.inf
        
        nuHat_inf = nuHat.copy()
        nuHat_inf[0, 0] = np.inf
        nuHat_inf[0, -1] = np.inf
        nuHat_inf[-1, 0] = np.inf
        nuHat_inf[-1, -1] = np.inf
        
        residual_inf = compute_viscous_fluxes_tight_jax(
            jnp.asarray(Q_inf), Si_x, Si_y, Sj_x, Sj_y,
            jnp.asarray(fg.d_coord_i), jnp.asarray(fg.e_coord_i_x), jnp.asarray(fg.e_coord_i_y),
            jnp.asarray(fg.e_ortho_i_x), jnp.asarray(fg.e_ortho_i_y),
            jnp.asarray(fg.d_coord_j), jnp.asarray(fg.e_coord_j_x), jnp.asarray(fg.e_coord_j_y),
            jnp.asarray(fg.e_ortho_j_x), jnp.asarray(fg.e_ortho_j_y),
            jnp.asarray(lsw.weights_i), jnp.asarray(lsw.weights_j),
            jnp.asarray(mu_eff_inf), nu_laminar, jnp.asarray(nuHat_inf)
        )
        residual_inf_np = np.asarray(residual_inf)
        
        # Center cells should have same residual (corners are outside stencil)
        center_i = slice(3, NI - 3)
        center_j = slice(3, NJ - 3)
        
        np.testing.assert_allclose(
            residual_inf_np[center_i, center_j, :],
            residual_ref_np[center_i, center_j, :],
            atol=1e-10,
            err_msg="Center cell residuals should not be affected by inf at corners"
        )
        
        # Verify that corners are indeed affected (should be inf or nan)
        assert not np.isfinite(residual_inf_np[1, 1, 1]), "Near-corner cell should be affected by inf"
    
    def test_stencil_isolation_nan_outside(self, skewed_grid_setup):
        """Verify that NaN values outside stencil don't affect center cells."""
        setup = skewed_grid_setup
        NI, NJ = setup['NI'], setup['NJ']
        X, Y = setup['X'], setup['Y']
        
        xc = 0.25 * (X[:-1, :-1] + X[1:, :-1] + X[1:, 1:] + X[:-1, 1:])
        yc = 0.25 * (Y[:-1, :-1] + Y[1:, :-1] + Y[1:, 1:] + Y[:-1, 1:])
        
        Q_ref = np.zeros((NI, NJ, 4))
        Q_ref[:, :, 0] = 1.0
        Q_ref[:, :, 1] = xc
        Q_ref[:, :, 2] = yc
        Q_ref[:, :, 3] = 0.01
        
        mu_eff = np.ones((NI, NJ)) * 0.01
        nu_laminar = 1e-5
        nuHat = Q_ref[:, :, 3].copy()
        
        fg = setup['face_geom']
        lsw = setup['ls_weights']
        
        Si_x = jnp.asarray(setup['metrics'].Si_x)
        Si_y = jnp.asarray(setup['metrics'].Si_y)
        Sj_x = jnp.asarray(setup['metrics'].Sj_x)
        Sj_y = jnp.asarray(setup['metrics'].Sj_y)
        
        residual_ref = compute_viscous_fluxes_tight_jax(
            jnp.asarray(Q_ref), Si_x, Si_y, Sj_x, Sj_y,
            jnp.asarray(fg.d_coord_i), jnp.asarray(fg.e_coord_i_x), jnp.asarray(fg.e_coord_i_y),
            jnp.asarray(fg.e_ortho_i_x), jnp.asarray(fg.e_ortho_i_y),
            jnp.asarray(fg.d_coord_j), jnp.asarray(fg.e_coord_j_x), jnp.asarray(fg.e_coord_j_y),
            jnp.asarray(fg.e_ortho_j_x), jnp.asarray(fg.e_ortho_j_y),
            jnp.asarray(lsw.weights_i), jnp.asarray(lsw.weights_j),
            jnp.asarray(mu_eff), nu_laminar, jnp.asarray(nuHat)
        )
        residual_ref_np = np.asarray(residual_ref)
        
        # Set corners to NaN
        Q_nan = Q_ref.copy()
        Q_nan[0, 0, :] = np.nan
        Q_nan[0, -1, :] = np.nan
        Q_nan[-1, 0, :] = np.nan
        Q_nan[-1, -1, :] = np.nan
        
        mu_eff_nan = mu_eff.copy()
        nuHat_nan = nuHat.copy()
        
        residual_nan = compute_viscous_fluxes_tight_jax(
            jnp.asarray(Q_nan), Si_x, Si_y, Sj_x, Sj_y,
            jnp.asarray(fg.d_coord_i), jnp.asarray(fg.e_coord_i_x), jnp.asarray(fg.e_coord_i_y),
            jnp.asarray(fg.e_ortho_i_x), jnp.asarray(fg.e_ortho_i_y),
            jnp.asarray(fg.d_coord_j), jnp.asarray(fg.e_coord_j_x), jnp.asarray(fg.e_coord_j_y),
            jnp.asarray(fg.e_ortho_j_x), jnp.asarray(fg.e_ortho_j_y),
            jnp.asarray(lsw.weights_i), jnp.asarray(lsw.weights_j),
            jnp.asarray(mu_eff_nan), nu_laminar, jnp.asarray(nuHat_nan)
        )
        residual_nan_np = np.asarray(residual_nan)
        
        # Center cells should have same residual
        center_i = slice(3, NI - 3)
        center_j = slice(3, NJ - 3)
        
        np.testing.assert_allclose(
            residual_nan_np[center_i, center_j, :],
            residual_ref_np[center_i, center_j, :],
            atol=1e-10,
            err_msg="Center cell residuals should not be affected by NaN at corners"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
