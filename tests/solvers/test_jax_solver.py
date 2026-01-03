"""
Tests for JAX solver components: time stepping, boundary conditions, forces.
"""

import pytest
import numpy as np
import time

from src.constants import NGHOST
from src.solvers.time_stepping import (
    compute_spectral_radii,
    compute_local_timestep,
    TimeStepConfig,
)
from src.solvers.boundary_conditions import (
    FreestreamConditions,
    BoundaryConditions,
)
from src.numerics.forces import (
    compute_aerodynamic_forces,
    AeroForces,
)

# JAX imports
try:
    from src.physics.jax_config import jax, jnp
    from src.solvers.time_stepping import (
        compute_spectral_radii_jax,
        compute_local_timestep_jax,
    )
    from src.solvers.boundary_conditions import (
        apply_surface_bc_jax,
        apply_farfield_bc_jax,
        apply_bc_jax,
    )
    from src.numerics.forces import (
        compute_surface_forces_jax,
        compute_aerodynamic_forces_jax,
    )
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@pytest.fixture
def medium_grid():
    """Medium grid for validation."""
    NI, NJ = 64, 32
    return NI, NJ


@pytest.fixture
def sample_state(medium_grid):
    """Create sample state array and metrics."""
    NI, NJ = medium_grid
    nghost = NGHOST
    
    np.random.seed(42)
    Q = np.zeros((NI + 2*nghost, NJ + 2*nghost, 4))
    Q[:, :, 0] = 0.0  # pressure
    Q[:, :, 1] = 1.0 + np.random.randn(NI + 2*nghost, NJ + 2*nghost) * 0.1
    Q[:, :, 2] = np.random.randn(NI + 2*nghost, NJ + 2*nghost) * 0.1
    Q[:, :, 3] = 1e-5  # nu_tilde
    
    Si_x = np.random.randn(NI + 1, NJ) * 0.01 + 0.1
    Si_y = np.random.randn(NI + 1, NJ) * 0.01
    Sj_x = np.random.randn(NI, NJ + 1) * 0.01
    Sj_y = np.random.randn(NI, NJ + 1) * 0.01 + 0.1
    volume = np.random.rand(NI, NJ) * 0.1 + 0.01
    
    return Q, Si_x, Si_y, Sj_x, Sj_y, volume, NI, NJ


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestTimeSteppingJAX:
    """Test JAX implementations of time stepping."""
    
    def test_spectral_radii_equivalence(self, sample_state):
        """Test spectral radii JAX vs NumPy."""
        Q, Si_x, Si_y, Sj_x, Sj_y, volume, NI, NJ = sample_state
        beta = 10.0
        nghost = NGHOST
        
        # NumPy
        spec_rad = compute_spectral_radii(Q, Si_x, Si_y, Sj_x, Sj_y, beta)
        lambda_i_np = spec_rad.lambda_i
        lambda_j_np = spec_rad.lambda_j
        
        # JAX
        Q_jax = jnp.array(Q)
        Si_x_jax = jnp.array(Si_x)
        Si_y_jax = jnp.array(Si_y)
        Sj_x_jax = jnp.array(Sj_x)
        Sj_y_jax = jnp.array(Sj_y)
        
        lambda_i_jax, lambda_j_jax = compute_spectral_radii_jax(
            Q_jax, Si_x_jax, Si_y_jax, Sj_x_jax, Sj_y_jax, beta, nghost
        )
        
        np.testing.assert_allclose(lambda_i_np, np.array(lambda_i_jax), rtol=1e-10)
        np.testing.assert_allclose(lambda_j_np, np.array(lambda_j_jax), rtol=1e-10)
    
    def test_local_timestep_equivalence(self, sample_state):
        """Test local timestep JAX vs NumPy."""
        Q, Si_x, Si_y, Sj_x, Sj_y, volume, NI, NJ = sample_state
        beta = 10.0
        cfg = TimeStepConfig(cfl=0.8)
        nghost = NGHOST
        
        # NumPy
        dt_np = compute_local_timestep(Q, Si_x, Si_y, Sj_x, Sj_y, volume, beta, cfg)
        
        # JAX
        Q_jax = jnp.array(Q)
        Si_x_jax = jnp.array(Si_x)
        Si_y_jax = jnp.array(Si_y)
        Sj_x_jax = jnp.array(Sj_x)
        Sj_y_jax = jnp.array(Sj_y)
        vol_jax = jnp.array(volume)
        
        dt_jax = compute_local_timestep_jax(
            Q_jax, Si_x_jax, Si_y_jax, Sj_x_jax, Sj_y_jax, vol_jax,
            beta, cfg.cfl, nghost
        )
        
        np.testing.assert_allclose(dt_np, np.array(dt_jax), rtol=1e-10)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestBoundaryConditionsJAX:
    """Test JAX implementations of boundary conditions."""
    
    def test_surface_bc_equivalence(self, sample_state):
        """Test surface BC JAX vs NumPy."""
        Q, Si_x, Si_y, Sj_x, Sj_y, volume, NI, NJ = sample_state
        n_wake = 8
        
        freestream = FreestreamConditions.from_mach_alpha(0.15, 0.0)
        bc = BoundaryConditions(freestream=freestream, n_wake_points=n_wake)
        
        # NumPy
        Q_np = bc.apply_surface(Q.copy())
        
        # JAX
        Q_jax = apply_surface_bc_jax(jnp.array(Q), n_wake, NGHOST)
        
        # Compare interior + ghost regions affected by surface BC
        max_diff = np.abs(Q_np - np.array(Q_jax)).max()
        rel_diff = max_diff / (np.abs(Q_np).max() + 1e-12)
        
        print(f"Surface BC max diff: {max_diff:.2e}, rel diff: {rel_diff:.2e}")
        assert rel_diff < 1e-10
    
    def test_farfield_bc_equivalence(self, sample_state):
        """Test farfield BC JAX vs NumPy."""
        Q, Si_x, Si_y, Sj_x, Sj_y, volume, NI, NJ = sample_state
        n_wake = 8
        
        # Create farfield normals (outward at j=NJ)
        nx = np.random.randn(NI)
        ny = np.random.randn(NI)
        mag = np.sqrt(nx**2 + ny**2)
        nx = nx / mag
        ny = ny / mag
        
        freestream = FreestreamConditions.from_mach_alpha(0.15, 0.0)
        bc = BoundaryConditions(
            freestream=freestream, 
            n_wake_points=n_wake,
            farfield_normals=(nx, ny)
        )
        
        # NumPy (apply full BC)
        Q_np = bc.apply(Q.copy())
        
        # JAX
        Q_jax = apply_bc_jax(
            jnp.array(Q), 
            (jnp.array(nx), jnp.array(ny)),
            freestream, n_wake, NGHOST
        )
        
        max_diff = np.abs(Q_np - np.array(Q_jax)).max()
        rel_diff = max_diff / (np.abs(Q_np).max() + 1e-12)
        
        print(f"Full BC max diff: {max_diff:.2e}, rel diff: {rel_diff:.2e}")
        assert rel_diff < 1e-10


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestForcesJAX:
    """Test JAX implementations of aerodynamic forces."""
    
    def test_surface_forces_equivalence(self, sample_state):
        """Test surface forces JAX vs NumPy."""
        Q, Si_x, Si_y, Sj_x, Sj_y, volume, NI, NJ = sample_state
        n_wake = 8
        mu_laminar = 1e-5
        
        # Create a simple metrics object
        class Metrics:
            pass
        metrics = Metrics()
        metrics.Sj_x = Sj_x
        metrics.Sj_y = Sj_y
        metrics.volume = volume
        
        # NumPy
        forces_np = compute_aerodynamic_forces(
            Q, metrics, mu_laminar, 
            alpha_deg=0.0, n_wake=n_wake
        )
        
        # JAX
        mu_eff = jnp.full((NI, NJ), mu_laminar)
        Fx_p, Fy_p, Fx_v, Fy_v = compute_surface_forces_jax(
            jnp.array(Q), jnp.array(Sj_x), jnp.array(Sj_y), 
            jnp.array(volume), mu_eff, n_wake, NGHOST
        )
        
        forces_jax = compute_aerodynamic_forces_jax(
            jnp.array(Q), jnp.array(Sj_x), jnp.array(Sj_y), jnp.array(volume),
            mu_laminar, alpha_deg=0.0, n_wake=n_wake
        )
        
        print(f"NumPy CL={forces_np.CL:.6f}, CD={forces_np.CD:.6f}")
        print(f"JAX   CL={forces_jax.CL:.6f}, CD={forces_jax.CD:.6f}")
        
        np.testing.assert_allclose(forces_np.CL, forces_jax.CL, rtol=1e-6)
        np.testing.assert_allclose(forces_np.CD, forces_jax.CD, rtol=1e-6)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestPerformance:
    """Performance comparison between JAX and NumPy."""
    
    def test_timestep_performance(self):
        """Benchmark time step computation."""
        NI, NJ = 256, 128
        nghost = NGHOST
        beta = 10.0
        
        np.random.seed(42)
        Q = np.random.randn(NI + 2*nghost, NJ + 2*nghost, 4) * 0.1
        Q[:, :, 1] += 1.0
        
        Si_x = np.random.randn(NI + 1, NJ) * 0.01 + 0.1
        Si_y = np.random.randn(NI + 1, NJ) * 0.01
        Sj_x = np.random.randn(NI, NJ + 1) * 0.01
        Sj_y = np.random.randn(NI, NJ + 1) * 0.01 + 0.1
        volume = np.random.rand(NI, NJ) * 0.1 + 0.01
        
        cfg = TimeStepConfig(cfl=0.8)
        
        Q_jax = jax.device_put(jnp.array(Q))
        Si_x_jax = jax.device_put(jnp.array(Si_x))
        Si_y_jax = jax.device_put(jnp.array(Si_y))
        Sj_x_jax = jax.device_put(jnp.array(Sj_x))
        Sj_y_jax = jax.device_put(jnp.array(Sj_y))
        vol_jax = jax.device_put(jnp.array(volume))
        
        # Warmup
        for _ in range(3):
            _ = compute_local_timestep(Q, Si_x, Si_y, Sj_x, Sj_y, volume, beta, cfg)
            _ = compute_local_timestep_jax(Q_jax, Si_x_jax, Si_y_jax, Sj_x_jax, 
                                            Sj_y_jax, vol_jax, beta, cfg.cfl, nghost)
        jax.block_until_ready(_)
        
        n_iter = 100
        
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = compute_local_timestep(Q, Si_x, Si_y, Sj_x, Sj_y, volume, beta, cfg)
        t_np = (time.perf_counter() - t0) / n_iter * 1000
        
        t0 = time.perf_counter()
        for _ in range(n_iter):
            res = compute_local_timestep_jax(Q_jax, Si_x_jax, Si_y_jax, Sj_x_jax, 
                                              Sj_y_jax, vol_jax, beta, cfg.cfl, nghost)
            jax.block_until_ready(res)
        t_jax = (time.perf_counter() - t0) / n_iter * 1000
        
        print(f"\nTimestep computation ({NI}x{NJ}):")
        print(f"  NumPy: {t_np:.3f} ms")
        print(f"  JAX:   {t_jax:.3f} ms")
        print(f"  Speedup: {t_np/t_jax:.2f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

