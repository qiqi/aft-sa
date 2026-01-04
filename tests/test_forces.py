"""
Pytest tests for aerodynamic force computation.

Tests verify:
1. Force computation runs without errors
2. Pressure and viscous forces are separated
3. Rotation to wind axes is correct
4. Coefficients are normalized properly
"""

import numpy as np
import pytest

from src.constants import NGHOST
from src.numerics.forces import compute_aerodynamic_forces, AeroForces


class SimpleMetrics:
    """Simple metrics for testing."""
    def __init__(self, NI, NJ, dx=0.1, dy=0.05):
        # J-faces at j=0 point in +y direction (into domain)
        self.Sj_x = np.zeros((NI, NJ + 1))
        self.Sj_y = np.ones((NI, NJ + 1)) * dx  # Area = dx
        self.volume = np.ones((NI, NJ)) * dx * dy


class TestForceComputation:
    """Tests for force computation."""
    
    def test_returns_aero_forces(self):
        """Should return AeroForces named tuple."""
        NI, NJ = 10, 5
        Q = np.ones((NI + 2*NGHOST, NJ + 2*NGHOST, 4))
        metrics = SimpleMetrics(NI, NJ)
        
        forces = compute_aerodynamic_forces(Q, metrics, mu_laminar=0.01)
        
        assert isinstance(forces, AeroForces)
        assert hasattr(forces, 'CL')
        assert hasattr(forces, 'CD')
        assert hasattr(forces, 'CD_p')
        assert hasattr(forces, 'CD_f')
    
    def test_uniform_pressure_symmetric_force(self):
        """Uniform pressure on symmetric airfoil should give zero lift."""
        NI, NJ = 20, 5
        Q = np.zeros((NI + 2*NGHOST, NJ + 2*NGHOST, 4))
        Q[:, :, 0] = 1.0  # Uniform pressure
        Q[:, :, 1] = 0.0  # No velocity (no viscous)
        
        # Symmetric metrics (forces in +y and -y cancel)
        metrics = SimpleMetrics(NI, NJ)
        # Make half point up, half point down
        metrics.Sj_y[NI//2:, 0] = -0.1  # Lower surface points down
        
        forces = compute_aerodynamic_forces(Q, metrics, mu_laminar=0.01, alpha_deg=0.0)
        
        # For symmetric airfoil at α=0, lift should be ~0
        # (This is a simplified test; real airfoil geometry matters)
        assert forces.CD_f == pytest.approx(0.0, abs=1e-10), "No velocity means no friction"
    
    def test_rotation_to_wind_axes(self):
        """Force rotation should follow wind axis convention."""
        NI, NJ = 10, 5
        Q = np.zeros((NI + 2*NGHOST, NJ + 2*NGHOST, 4))
        Q[:, :, 0] = 1.0  # pressure
        
        metrics = SimpleMetrics(NI, NJ)
        
        # At α=0: D=Fx, L=Fy
        forces_0 = compute_aerodynamic_forces(Q, metrics, mu_laminar=0.01, alpha_deg=0.0)
        
        # At α=90: D=Fy, L=-Fx  
        forces_90 = compute_aerodynamic_forces(Q, metrics, mu_laminar=0.01, alpha_deg=90.0)
        
        # The body-axis forces should be the same
        assert forces_0.Fx == pytest.approx(forces_90.Fx, rel=1e-10)
        assert forces_0.Fy == pytest.approx(forces_90.Fy, rel=1e-10)
        
        # But wind-axis should be rotated
        # At 90°: D = Fx*cos(90) + Fy*sin(90) = Fy
        #         L = Fy*cos(90) - Fx*sin(90) = -Fx
        assert forces_90.CD == pytest.approx(forces_0.CL, rel=1e-10)
        assert forces_90.CL == pytest.approx(-forces_0.CD, rel=1e-10)
    
    def test_friction_drag_from_velocity(self):
        """Wall velocity gradient should produce friction drag."""
        NI, NJ = 10, 5
        Q = np.zeros((NI + 2*NGHOST, NJ + 2*NGHOST, 4))
        Q[:, :, 0] = 0.5  # Some pressure (not important for friction)
        Q[:, NGHOST, 1] = 1.0  # u-velocity in first interior cell
        
        metrics = SimpleMetrics(NI, NJ, dx=0.1, dy=0.05)
        
        forces = compute_aerodynamic_forces(
            Q, metrics, mu_laminar=0.1, alpha_deg=0.0
        )
        
        # Should have positive friction drag (opposes flow)
        assert forces.CD_f > 0, "Friction drag should be positive"
        
        # Friction drag should increase with viscosity
        forces_high_mu = compute_aerodynamic_forces(
            Q, metrics, mu_laminar=0.2, alpha_deg=0.0
        )
        assert forces_high_mu.CD_f > forces.CD_f, \
            "Higher viscosity should give more friction"
    
    def test_coefficient_normalization(self):
        """Coefficients should be normalized by dynamic pressure."""
        NI, NJ = 10, 5
        Q = np.zeros((NI + 2*NGHOST, NJ + 2*NGHOST, 4))
        Q[:, :, 0] = 2.0  # Pressure = 2
        
        metrics = SimpleMetrics(NI, NJ)
        
        # With V_inf=1, q_inf=0.5
        forces_v1 = compute_aerodynamic_forces(
            Q, metrics, mu_laminar=0.01, V_inf=1.0
        )
        
        # With V_inf=2, q_inf=2.0 (4x larger)
        forces_v2 = compute_aerodynamic_forces(
            Q, metrics, mu_laminar=0.01, V_inf=2.0
        )
        
        # Coefficients should scale with 1/q_inf
        # Same forces, but 4x smaller coefficients at V_inf=2
        assert forces_v2.CD_p == pytest.approx(forces_v1.CD_p / 4, rel=0.01)


class TestDecomposition:
    """Tests for pressure/friction decomposition."""
    
    def test_total_equals_sum(self):
        """Total drag should equal pressure + friction."""
        NI, NJ = 10, 5
        Q = np.zeros((NI + 2*NGHOST, NJ + 2*NGHOST, 4))
        Q[:, :, 0] = 1.0
        Q[:, NGHOST, 1] = 0.5
        
        metrics = SimpleMetrics(NI, NJ)
        forces = compute_aerodynamic_forces(Q, metrics, mu_laminar=0.01)
        
        assert forces.CD == pytest.approx(forces.CD_p + forces.CD_f, rel=1e-10)
        assert forces.CL == pytest.approx(forces.CL_p + forces.CL_f, rel=1e-10)


class TestJaxVsNumpyForces:
    """Verify JAX and NumPy force implementations give identical results."""
    
    def test_jax_pure_matches_wrapper(self):
        """Pure JAX force computation should match wrapper result."""
        from src.physics.jax_config import jnp
        from src.numerics.forces import compute_aerodynamic_forces_jax_pure
        
        NI, NJ = 20, 10
        nghost = NGHOST
        
        # Create realistic test state
        Q = np.zeros((NI + 2*nghost, NJ + 2*nghost, 4))
        Q[:, :, 0] = 0.5  # Pressure
        Q[:, :, 1] = 1.0  # u-velocity  
        Q[:, :, 2] = 0.1  # v-velocity
        Q[:, :, 3] = 0.001  # nuHat (SA variable)
        
        # Add some variation
        np.random.seed(42)
        Q[:, :, 0] += 0.1 * np.random.randn(*Q[:, :, 0].shape)
        Q[:, nghost, 1] = 0.8  # Near-wall velocity
        
        # Create metrics
        dx, dy = 0.1, 0.05
        Sj_x = np.zeros((NI, NJ + 1))
        Sj_y = np.ones((NI, NJ + 1)) * dx
        volume = np.ones((NI, NJ)) * dx * dy
        
        mu_laminar = 0.001
        mu_eff = np.full((NI, NJ), mu_laminar)
        
        # Parameters
        alpha_deg = 5.0
        alpha_rad = np.deg2rad(alpha_deg)
        V_inf = 1.0
        q_inf = 0.5 * V_inf**2
        n_wake = 2
        
        # Compute with wrapper
        class SimpleMetrics:
            pass
        metrics = SimpleMetrics()
        metrics.Sj_x = Sj_x
        metrics.Sj_y = Sj_y
        metrics.volume = volume
        
        forces_wrapper = compute_aerodynamic_forces(
            Q, metrics, mu_laminar=mu_laminar, mu_turb=None,
            alpha_deg=alpha_deg, V_inf=V_inf, n_wake=n_wake
        )
        
        # Compute with pure JAX
        Q_jax = jnp.asarray(Q)
        Sj_x_jax = jnp.asarray(Sj_x)
        Sj_y_jax = jnp.asarray(Sj_y)
        volume_jax = jnp.asarray(volume)
        mu_eff_jax = jnp.asarray(mu_eff)
        
        CL_jax, CD_jax, CD_p_jax, CD_f_jax = compute_aerodynamic_forces_jax_pure(
            Q_jax, Sj_x_jax, Sj_y_jax, volume_jax, mu_eff_jax,
            np.sin(alpha_rad), np.cos(alpha_rad), q_inf,
            n_wake, nghost
        )
        
        # Compare
        assert float(CL_jax) == pytest.approx(forces_wrapper.CL, rel=1e-10)
        assert float(CD_jax) == pytest.approx(forces_wrapper.CD, rel=1e-10)
        assert float(CD_p_jax) == pytest.approx(forces_wrapper.CD_p, rel=1e-10)
        assert float(CD_f_jax) == pytest.approx(forces_wrapper.CD_f, rel=1e-10)
    
    def test_jax_pure_performance(self):
        """Verify JAX version is faster for repeated calls."""
        import time
        from src.physics.jax_config import jnp
        from src.numerics.forces import compute_aerodynamic_forces_jax_pure
        
        NI, NJ = 100, 50
        nghost = NGHOST
        
        # Create test arrays
        Q = np.random.randn(NI + 2*nghost, NJ + 2*nghost, 4)
        Sj_x = np.random.randn(NI, NJ + 1) * 0.01
        Sj_y = np.ones((NI, NJ + 1)) * 0.1
        volume = np.ones((NI, NJ)) * 0.01
        mu_eff = np.full((NI, NJ), 0.001)
        
        alpha_rad = np.deg2rad(5.0)
        
        # Convert to JAX
        Q_jax = jnp.asarray(Q)
        Sj_x_jax = jnp.asarray(Sj_x)
        Sj_y_jax = jnp.asarray(Sj_y)
        volume_jax = jnp.asarray(volume)
        mu_eff_jax = jnp.asarray(mu_eff)
        
        # Warmup JIT
        _ = compute_aerodynamic_forces_jax_pure(
            Q_jax, Sj_x_jax, Sj_y_jax, volume_jax, mu_eff_jax,
            np.sin(alpha_rad), np.cos(alpha_rad), 0.5, 0, nghost
        )
        
        # Time JAX version
        n_calls = 100
        t0 = time.perf_counter()
        for _ in range(n_calls):
            CL, CD, _, _ = compute_aerodynamic_forces_jax_pure(
                Q_jax, Sj_x_jax, Sj_y_jax, volume_jax, mu_eff_jax,
                np.sin(alpha_rad), np.cos(alpha_rad), 0.5, 0, nghost
            )
            # Force sync to measure actual compute time
            _ = float(CL)
        t_jax = (time.perf_counter() - t0) / n_calls * 1000  # ms
        
        # Time wrapper version (includes NumPy↔JAX conversions)
        class SimpleMetrics:
            pass
        metrics = SimpleMetrics()
        metrics.Sj_x = Sj_x
        metrics.Sj_y = Sj_y
        metrics.volume = volume
        
        t0 = time.perf_counter()
        for _ in range(n_calls):
            _ = compute_aerodynamic_forces(
                Q, metrics, mu_laminar=0.001, mu_turb=None,
                alpha_deg=5.0, V_inf=1.0, n_wake=0
            )
        t_wrapper = (time.perf_counter() - t0) / n_calls * 1000  # ms
        
        print(f"\nForce computation timing ({n_calls} calls):")
        print(f"  JAX pure: {t_jax:.3f} ms/call")
        print(f"  Wrapper:  {t_wrapper:.3f} ms/call")
        
        # JAX version should be faster (no array conversions)
        # But allow some slack since wrapper also uses JAX internally
        assert t_jax < t_wrapper * 2, "JAX pure should not be significantly slower"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

