"""
Tests for batch kernel implementations.

Verifies that batch kernels (using vmap) produce identical results
to single-case computations.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

# JAX imports
from src.physics.jax_config import jax, jnp

# Single-case implementations
from src.numerics.fluxes import compute_fluxes_jax
from src.numerics.gradients import compute_gradients_jax
from src.numerics.viscous_fluxes import compute_viscous_fluxes_jax
from src.numerics.explicit_smoothing import smooth_explicit_jax
from src.solvers.time_stepping import compute_local_timestep_jax
from src.solvers.boundary_conditions import make_apply_bc_jit, FreestreamConditions

# Batch implementations
from src.solvers.batch import (
    BatchFlowConditions,
    BatchState,
    compute_fluxes_batch,
    compute_gradients_batch,
    compute_viscous_fluxes_batch,
    compute_timestep_batch,
    smooth_residual_batch,
    make_apply_bc_batch_jit,
    make_batch_step_jit,
)
from src.constants import NGHOST


@pytest.fixture
def small_grid():
    """Create a small test grid (32x16 cells)."""
    NI, NJ = 32, 16
    nghost = NGHOST

    # Create simple rectangular grid
    x = np.linspace(0, 1, NI + 1)
    y = np.linspace(0, 0.5, NJ + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Compute metrics (simplified for unit cell grid)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Face normal vectors (scaled by face area in 2D = length)
    Si_x = jnp.ones((NI + 1, NJ)) * dy  # Normal in x-direction at i-faces
    Si_y = jnp.zeros((NI + 1, NJ))
    Sj_x = jnp.zeros((NI, NJ + 1))
    Sj_y = jnp.ones((NI, NJ + 1)) * dx  # Normal in y-direction at j-faces

    volume = jnp.ones((NI, NJ)) * dx * dy

    # Wall distance (for SA model) - increases with j
    wall_dist = np.zeros((NI, NJ))
    for j in range(NJ):
        wall_dist[:, j] = (j + 0.5) * dy  # Distance to j=0 wall
    wall_dist = jnp.array(wall_dist)

    # Farfield normals (pointing outward at j=NJ)
    nx_ff = jnp.zeros(NI)
    ny_ff = jnp.ones(NI)

    return {
        'NI': NI,
        'NJ': NJ,
        'nghost': nghost,
        'Si_x': Si_x,
        'Si_y': Si_y,
        'Sj_x': Sj_x,
        'Sj_y': Sj_y,
        'volume': volume,
        'wall_dist': wall_dist,
        'nx_ff': nx_ff,
        'ny_ff': ny_ff,
    }


@pytest.fixture
def flow_params():
    """Common flow parameters."""
    return {
        'beta': 10.0,
        'k4': 0.02,
        'nu': 1e-6,
        'cfl': 1.0,
    }


def create_random_Q(NI, NJ, nghost, seed=42):
    """Create random state array with valid values."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((NI + 2*nghost, NJ + 2*nghost, 4))
    
    # Reasonable values: p near 0, u,v near 1, nu_t small
    Q[:, :, 0] = rng.uniform(-0.1, 0.1, Q.shape[:2])  # pressure
    Q[:, :, 1] = rng.uniform(0.8, 1.2, Q.shape[:2])   # u
    Q[:, :, 2] = rng.uniform(-0.2, 0.2, Q.shape[:2])  # v
    Q[:, :, 3] = rng.uniform(0.0, 1e-4, Q.shape[:2])  # nu_t
    
    return jnp.array(Q)


class TestBatchFluxes:
    """Test batch flux computation matches single-case."""
    
    def test_single_vs_batch_n1(self, small_grid, flow_params):
        """Batch of 1 should match single case exactly."""
        NI = small_grid['NI']
        NJ = small_grid['NJ']
        nghost = small_grid['nghost']
        
        Q_single = create_random_Q(NI, NJ, nghost)
        Q_batch = Q_single[None, ...]  # (1, NI+2g, NJ+2g, 4)
        
        # Single case
        R_single = compute_fluxes_jax(
            Q_single, 
            small_grid['Si_x'], small_grid['Si_y'],
            small_grid['Sj_x'], small_grid['Sj_y'],
            flow_params['beta'], flow_params['k4'], nghost
        )
        
        # Batch of 1
        R_batch = compute_fluxes_batch(
            Q_batch,
            small_grid['Si_x'], small_grid['Si_y'],
            small_grid['Sj_x'], small_grid['Sj_y'],
            flow_params['beta'], flow_params['k4'], nghost
        )
        
        assert R_batch.shape == (1, NI, NJ, 4)
        assert_allclose(np.array(R_batch[0]), np.array(R_single), rtol=1e-12)
    
    def test_batch_independence(self, small_grid, flow_params):
        """Each batch element should be computed independently."""
        NI = small_grid['NI']
        NJ = small_grid['NJ']
        nghost = small_grid['nghost']
        
        # Create 4 different Q states
        Q1 = create_random_Q(NI, NJ, nghost, seed=1)
        Q2 = create_random_Q(NI, NJ, nghost, seed=2)
        Q3 = create_random_Q(NI, NJ, nghost, seed=3)
        Q4 = create_random_Q(NI, NJ, nghost, seed=4)
        
        Q_batch = jnp.stack([Q1, Q2, Q3, Q4], axis=0)
        
        # Compute batch
        R_batch = compute_fluxes_batch(
            Q_batch,
            small_grid['Si_x'], small_grid['Si_y'],
            small_grid['Sj_x'], small_grid['Sj_y'],
            flow_params['beta'], flow_params['k4'], nghost
        )
        
        # Compute singles
        for i, Q_i in enumerate([Q1, Q2, Q3, Q4]):
            R_single = compute_fluxes_jax(
                Q_i,
                small_grid['Si_x'], small_grid['Si_y'],
                small_grid['Sj_x'], small_grid['Sj_y'],
                flow_params['beta'], flow_params['k4'], nghost
            )
            assert_allclose(np.array(R_batch[i]), np.array(R_single), rtol=1e-12)


class TestBatchGradients:
    """Test batch gradient computation matches single-case."""
    
    def test_single_vs_batch_n1(self, small_grid):
        """Batch of 1 should match single case exactly."""
        NI = small_grid['NI']
        NJ = small_grid['NJ']
        nghost = small_grid['nghost']
        
        Q_single = create_random_Q(NI, NJ, nghost)
        Q_batch = Q_single[None, ...]
        
        # Single case
        grad_single = compute_gradients_jax(
            Q_single,
            small_grid['Si_x'], small_grid['Si_y'],
            small_grid['Sj_x'], small_grid['Sj_y'],
            small_grid['volume'], nghost
        )
        
        # Batch of 1
        grad_batch = compute_gradients_batch(
            Q_batch,
            small_grid['Si_x'], small_grid['Si_y'],
            small_grid['Sj_x'], small_grid['Sj_y'],
            small_grid['volume'], nghost
        )
        
        assert grad_batch.shape == (1, NI, NJ, 4, 2)
        assert_allclose(np.array(grad_batch[0]), np.array(grad_single), rtol=1e-12)


class TestBatchTimestep:
    """Test batch timestep computation matches single-case."""
    
    def test_single_vs_batch_n1(self, small_grid, flow_params):
        """Batch of 1 should match single case exactly."""
        NI = small_grid['NI']
        NJ = small_grid['NJ']
        nghost = small_grid['nghost']
        
        Q_single = create_random_Q(NI, NJ, nghost)
        Q_batch = Q_single[None, ...]
        
        # Single case
        dt_single = compute_local_timestep_jax(
            Q_single,
            small_grid['Si_x'], small_grid['Si_y'],
            small_grid['Sj_x'], small_grid['Sj_y'],
            small_grid['volume'],
            flow_params['beta'], flow_params['cfl'], nghost,
            nu=flow_params['nu']
        )
        
        # Batch of 1
        dt_batch = compute_timestep_batch(
            Q_batch,
            small_grid['Si_x'], small_grid['Si_y'],
            small_grid['Sj_x'], small_grid['Sj_y'],
            small_grid['volume'],
            flow_params['beta'], flow_params['cfl'], nghost,
            flow_params['nu']
        )
        
        assert dt_batch.shape == (1, NI, NJ)
        assert_allclose(np.array(dt_batch[0]), np.array(dt_single), rtol=1e-12)


class TestBatchViscousFluxes:
    """Test batch viscous flux computation matches single-case."""
    
    def test_single_vs_batch_n1(self, small_grid, flow_params):
        """Batch of 1 should match single case exactly."""
        NI = small_grid['NI']
        NJ = small_grid['NJ']
        nghost = small_grid['nghost']
        
        Q_single = create_random_Q(NI, NJ, nghost)
        Q_batch = Q_single[None, ...]
        
        # Compute gradients first
        grad_single = compute_gradients_jax(
            Q_single,
            small_grid['Si_x'], small_grid['Si_y'],
            small_grid['Sj_x'], small_grid['Sj_y'],
            small_grid['volume'], nghost
        )
        grad_batch = grad_single[None, ...]
        
        # Create mu_eff
        mu_eff_single = jnp.full((NI, NJ), flow_params['nu'])
        mu_eff_batch = mu_eff_single[None, ...]
        
        # Single case
        R_visc_single = compute_viscous_fluxes_jax(
            grad_single,
            small_grid['Si_x'], small_grid['Si_y'],
            small_grid['Sj_x'], small_grid['Sj_y'],
            mu_eff_single
        )
        
        # Batch of 1
        R_visc_batch = compute_viscous_fluxes_batch(
            grad_batch,
            small_grid['Si_x'], small_grid['Si_y'],
            small_grid['Sj_x'], small_grid['Sj_y'],
            mu_eff_batch
        )
        
        assert R_visc_batch.shape == (1, NI, NJ, 4)
        assert_allclose(np.array(R_visc_batch[0]), np.array(R_visc_single), rtol=1e-12)


class TestBatchSmoothing:
    """Test batch smoothing matches single-case."""
    
    def test_single_vs_batch_n1(self, small_grid):
        """Batch of 1 should match single case exactly."""
        NI = small_grid['NI']
        NJ = small_grid['NJ']
        
        # Create random residual
        rng = np.random.default_rng(42)
        R_single = jnp.array(rng.uniform(-1, 1, (NI, NJ, 4)))
        R_batch = R_single[None, ...]
        
        epsilon = 0.2
        n_passes = 2
        
        # Single case
        R_smooth_single = smooth_explicit_jax(R_single, epsilon, n_passes)
        
        # Batch of 1
        R_smooth_batch = smooth_residual_batch(R_batch, epsilon, n_passes)
        
        assert R_smooth_batch.shape == (1, NI, NJ, 4)
        assert_allclose(np.array(R_smooth_batch[0]), np.array(R_smooth_single), rtol=1e-12)


class TestBatchBoundaryConditions:
    """Test batch BC application with per-case freestream."""
    
    def test_different_alpha_produces_different_bc(self, small_grid):
        """Different angles of attack should produce different BC values."""
        NI = small_grid['NI']
        NJ = small_grid['NJ']
        nghost = small_grid['nghost']
        
        # Create batch BC function
        apply_bc_batch = make_apply_bc_batch_jit(
            NI, NJ, n_wake_points=5,
            nx=small_grid['nx_ff'], ny=small_grid['ny_ff'],
            nghost=nghost
        )
        
        # Two cases: alpha = 0° and alpha = 10°
        Q1 = create_random_Q(NI, NJ, nghost, seed=1)
        Q2 = create_random_Q(NI, NJ, nghost, seed=1)  # Same initial state
        Q_batch = jnp.stack([Q1, Q2], axis=0)
        
        # Different freestream conditions
        alpha1, alpha2 = 0.0, 10.0
        u_inf = jnp.array([np.cos(np.radians(alpha1)), np.cos(np.radians(alpha2))])
        v_inf = jnp.array([np.sin(np.radians(alpha1)), np.sin(np.radians(alpha2))])
        p_inf = jnp.array([0.0, 0.0])
        nu_t_inf = jnp.array([1e-9, 1e-9])
        
        Q_bc = apply_bc_batch(Q_batch, u_inf, v_inf, p_inf, nu_t_inf)
        
        # Results should be different due to different freestream
        # Check farfield ghost cells differ
        diff = jnp.abs(Q_bc[0, :, -1, :] - Q_bc[1, :, -1, :])
        assert jnp.max(diff) > 1e-6, "Different alpha should produce different BCs"
    
    def test_batch_bc_matches_single(self, small_grid):
        """Batch BC should match single BC for same freestream."""
        NI = small_grid['NI']
        NJ = small_grid['NJ']
        nghost = small_grid['nghost']
        
        alpha = 5.0
        freestream = FreestreamConditions.from_mach_alpha(0.0, alpha, reynolds=1e6)
        
        # Single case BC
        apply_bc_single = make_apply_bc_jit(
            NI, NJ, n_wake_points=5,
            nx=small_grid['nx_ff'], ny=small_grid['ny_ff'],
            freestream=freestream,
            nghost=nghost
        )
        
        # Batch BC
        apply_bc_batch = make_apply_bc_batch_jit(
            NI, NJ, n_wake_points=5,
            nx=small_grid['nx_ff'], ny=small_grid['ny_ff'],
            nghost=nghost
        )
        
        Q_single = create_random_Q(NI, NJ, nghost)
        Q_batch = Q_single[None, ...]
        
        u_inf = jnp.array([freestream.u_inf])
        v_inf = jnp.array([freestream.v_inf])
        p_inf = jnp.array([freestream.p_inf])
        nu_t_inf = jnp.array([freestream.nu_t_inf])
        
        Q_bc_single = apply_bc_single(Q_single)
        Q_bc_batch = apply_bc_batch(Q_batch, u_inf, v_inf, p_inf, nu_t_inf)
        
        assert_allclose(np.array(Q_bc_batch[0]), np.array(Q_bc_single), rtol=1e-12)


class TestBatchStep:
    """Test complete batch step function."""
    
    def test_rk_stage_single_vs_batch(self, small_grid, flow_params):
        """Single RK stage should match between batch and single."""
        NI = small_grid['NI']
        NJ = small_grid['NJ']
        nghost = small_grid['nghost']
        
        # Create batch step function
        batch_step, apply_bc_batch, compute_dt = make_batch_step_jit(
            NI, NJ, n_wake_points=5,
            nx=small_grid['nx_ff'], ny=small_grid['ny_ff'],
            Si_x=small_grid['Si_x'], Si_y=small_grid['Si_y'],
            Sj_x=small_grid['Sj_x'], Sj_y=small_grid['Sj_y'],
            volume=small_grid['volume'],
            wall_dist=small_grid['wall_dist'],
            beta=flow_params['beta'], k4=flow_params['k4'],
            nu=flow_params['nu'],
            smoothing_epsilon=0.2, smoothing_passes=2,
            nghost=nghost
        )
        
        # Create initial state
        Q_single = create_random_Q(NI, NJ, nghost)
        Q_batch = Q_single[None, ...]
        
        # Freestream for alpha = 5°
        alpha = 5.0
        u_inf = jnp.array([np.cos(np.radians(alpha))])
        v_inf = jnp.array([np.sin(np.radians(alpha))])
        p_inf = jnp.array([0.0])
        nu_t_inf = jnp.array([1e-9])
        
        # Compute timestep
        dt_batch = compute_dt(Q_batch, flow_params['cfl'])
        
        # One RK stage
        alpha_rk = 0.25
        Q_new, R = batch_step(Q_batch, Q_batch, dt_batch, u_inf, v_inf, p_inf, nu_t_inf, alpha_rk)
        
        assert Q_new.shape == Q_batch.shape
        assert R.shape == (1, NI, NJ, 4)
        
        # Check that state changed
        diff = jnp.max(jnp.abs(Q_new - Q_batch))
        assert diff > 1e-10, "State should change after RK stage"
    
    def test_multiple_cases_batch(self, small_grid, flow_params):
        """Test batch with multiple different cases."""
        NI = small_grid['NI']
        NJ = small_grid['NJ']
        nghost = small_grid['nghost']
        
        batch_step, apply_bc_batch, compute_dt = make_batch_step_jit(
            NI, NJ, n_wake_points=5,
            nx=small_grid['nx_ff'], ny=small_grid['ny_ff'],
            Si_x=small_grid['Si_x'], Si_y=small_grid['Si_y'],
            Sj_x=small_grid['Sj_x'], Sj_y=small_grid['Sj_y'],
            volume=small_grid['volume'],
            wall_dist=small_grid['wall_dist'],
            beta=flow_params['beta'], k4=flow_params['k4'],
            nu=flow_params['nu'],
            smoothing_epsilon=0.2, smoothing_passes=2,
            nghost=nghost
        )
        
        # 5 cases with different initial states
        n_batch = 5
        Q_list = [create_random_Q(NI, NJ, nghost, seed=i) for i in range(n_batch)]
        Q_batch = jnp.stack(Q_list, axis=0)
        
        # Different angles of attack
        alphas = np.array([-5.0, 0.0, 5.0, 10.0, 15.0])
        u_inf = jnp.array(np.cos(np.radians(alphas)))
        v_inf = jnp.array(np.sin(np.radians(alphas)))
        p_inf = jnp.zeros(n_batch)
        nu_t_inf = jnp.full(n_batch, 1e-9)
        
        dt_batch = compute_dt(Q_batch, flow_params['cfl'])
        
        # One RK stage
        alpha_rk = 0.25
        Q_new, R = batch_step(Q_batch, Q_batch, dt_batch, u_inf, v_inf, p_inf, nu_t_inf, alpha_rk)
        
        assert Q_new.shape == (n_batch, NI + 2*nghost, NJ + 2*nghost, 4)
        assert R.shape == (n_batch, NI, NJ, 4)
        
        # Each case should have evolved differently
        for i in range(n_batch - 1):
            diff = jnp.max(jnp.abs(Q_new[i] - Q_new[i+1]))
            assert diff > 1e-6, f"Cases {i} and {i+1} should differ"


class TestBatchStateIntegration:
    """Test integration with BatchState and BatchFlowConditions."""
    
    def test_batch_state_creation(self, small_grid):
        """Test BatchState.from_single_ic works correctly."""
        NI = small_grid['NI']
        NJ = small_grid['NJ']
        nghost = small_grid['nghost']
        
        Q_single = np.array(create_random_Q(NI, NJ, nghost))
        n_batch = 8
        
        state = BatchState.from_single_ic(Q_single, n_batch, nghost=nghost)
        
        assert state.n_batch == n_batch
        assert state.Q.shape == (n_batch, NI + 2*nghost, NJ + 2*nghost, 4)
        
        # All should be identical copies
        for i in range(n_batch):
            assert_allclose(state.Q[i], Q_single, rtol=1e-12)
    
    def test_batch_flow_conditions(self):
        """Test BatchFlowConditions from sweep."""
        conditions = BatchFlowConditions.from_sweep(
            alpha_spec={'sweep': [-5, 10, 6]},  # 6 values: -5, -2, 1, 4, 7, 10
            reynolds=6e6,
            mach=0.0
        )
        
        assert conditions.n_batch == 6
        expected_alphas = np.linspace(-5, 10, 6)
        assert_allclose(conditions.alpha_deg, expected_alphas, rtol=1e-12)
        
        # Check derived quantities
        expected_u_inf = np.cos(np.radians(expected_alphas))
        expected_v_inf = np.sin(np.radians(expected_alphas))
        assert_allclose(conditions.u_inf, expected_u_inf, rtol=1e-12)
        assert_allclose(conditions.v_inf, expected_v_inf, rtol=1e-12)
    
    def test_jax_conversion(self):
        """Test conversion to JAX arrays."""
        conditions = BatchFlowConditions.from_sweep(
            alpha_spec={'values': [0, 5, 10]},
            reynolds=6e6
        )
        
        jax_cond = conditions.to_jax()
        
        assert jax_cond.n_batch == 3
        assert isinstance(jax_cond.u_inf, jnp.ndarray)
        assert jax_cond.u_inf.shape == (3,)
