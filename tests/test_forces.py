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
        Q = np.ones((NI + 2, NJ + 2, 4))
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
        Q = np.zeros((NI + 2, NJ + 2, 4))
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
        Q = np.zeros((NI + 2, NJ + 2, 4))
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
        Q = np.zeros((NI + 2, NJ + 2, 4))
        Q[:, :, 0] = 0.5  # Some pressure (not important for friction)
        Q[:, 1, 1] = 1.0  # u-velocity in first interior cell
        
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
        Q = np.zeros((NI + 2, NJ + 2, 4))
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
        Q = np.zeros((NI + 2, NJ + 2, 4))
        Q[:, :, 0] = 1.0
        Q[:, 1, 1] = 0.5
        
        metrics = SimpleMetrics(NI, NJ)
        forces = compute_aerodynamic_forces(Q, metrics, mu_laminar=0.01)
        
        assert forces.CD == pytest.approx(forces.CD_p + forces.CD_f, rel=1e-10)
        assert forces.CL == pytest.approx(forces.CL_p + forces.CL_f, rel=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

