#!/usr/bin/env python3
"""
Unit tests for the boundary conditions module.

Tests cover:
    - Surface (wall) boundary conditions
    - Farfield boundary conditions
    - Wake cut periodic boundary conditions
    - State initialization
    - Combined BC application
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from dataclasses import dataclass
from typing import List

from src.solvers.boundary_conditions import (
    FreestreamConditions,
    BoundaryConditions,
    apply_boundary_conditions,
    initialize_state,
    InletOutletBC,
)


@dataclass
class ResultData:
    name: str
    passed: bool
    message: str


class BoundaryConditionTests:
    """Test suite for boundary conditions module."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.NI = 20
        self.NJ = 15
        
    def setup_freestream(self):
        """Create freestream conditions."""
        return FreestreamConditions(
            p_inf=0.0,
            u_inf=1.0,
            v_inf=0.0,
            nu_t_inf=3e-6
        )
    
    def setup_state(self, freestream=None):
        """Create initialized state."""
        if freestream is None:
            freestream = self.setup_freestream()
        return initialize_state(self.NI, self.NJ, freestream)
    
    def run_test(self, name: str, test_func):
        """Run a single test and record result."""
        try:
            passed, message = test_func()
            self.results.append(ResultData(name, passed, message))
        except Exception as e:
            self.results.append(ResultData(name, False, f"Exception: {e}"))
    
    # ===== FreestreamConditions Tests =====
    
    def test_freestream_from_mach_alpha(self):
        """Test freestream creation from Mach/alpha."""
        fs = FreestreamConditions.from_mach_alpha(mach=0.2, alpha_deg=5.0)
        
        # Check velocity magnitude is 1
        vel_mag = np.sqrt(fs.u_inf**2 + fs.v_inf**2)
        if not np.isclose(vel_mag, 1.0, rtol=0.01):
            return False, f"Velocity magnitude {vel_mag:.4f} != 1.0"
        
        # Check angle
        alpha_computed = np.degrees(np.arctan2(fs.v_inf, fs.u_inf))
        if not np.isclose(alpha_computed, 5.0, atol=0.01):
            return False, f"Alpha {alpha_computed:.2f} != 5.0"
        
        return True, f"u={fs.u_inf:.4f}, v={fs.v_inf:.4f}"
    
    def test_freestream_zero_alpha(self):
        """Test freestream at zero angle of attack."""
        fs = FreestreamConditions.from_mach_alpha(mach=0.3, alpha_deg=0.0)
        
        if not np.isclose(fs.u_inf, 1.0, rtol=0.01):
            return False, f"u_inf {fs.u_inf:.4f} != 1.0"
        if not np.isclose(fs.v_inf, 0.0, atol=1e-10):
            return False, f"v_inf {fs.v_inf:.4f} != 0.0"
        
        return True, "Correct zero-alpha freestream"
    
    # ===== State Initialization Tests =====
    
    def test_initialize_state_shape(self):
        """Test initialized state shape."""
        Q = self.setup_state()
        
        expected_shape = (self.NI + 2, self.NJ + 2, 4)
        if Q.shape != expected_shape:
            return False, f"Shape {Q.shape} != {expected_shape}"
        return True, "Correct shape"
    
    def test_initialize_state_values(self):
        """Test initialized state values match freestream."""
        fs = FreestreamConditions(p_inf=1.5, u_inf=0.8, v_inf=0.1, nu_t_inf=5e-6)
        Q = initialize_state(self.NI, self.NJ, fs)
        
        if not np.allclose(Q[:, :, 0], fs.p_inf):
            return False, "Pressure doesn't match freestream"
        if not np.allclose(Q[:, :, 1], fs.u_inf):
            return False, "u doesn't match freestream"
        if not np.allclose(Q[:, :, 2], fs.v_inf):
            return False, "v doesn't match freestream"
        if not np.allclose(Q[:, :, 3], fs.nu_t_inf):
            return False, "nu_t doesn't match freestream"
        
        return True, "All values match freestream"
    
    # ===== Surface BC Tests =====
    
    def test_surface_no_slip_velocity(self):
        """Test no-slip velocity condition at surface."""
        Q = self.setup_state()
        bc = BoundaryConditions(freestream=self.setup_freestream())
        
        Q_bc = bc.apply_surface(Q)
        
        # Ghost velocity should be negative of interior
        # So that average at face is zero
        u_interior = Q_bc[10, 1, 1]
        u_ghost = Q_bc[10, 0, 1]
        
        if not np.isclose(u_ghost, -u_interior):
            return False, f"u_ghost {u_ghost:.4f} != -u_interior {-u_interior:.4f}"
        
        # Face average should be zero
        u_face = 0.5 * (u_ghost + u_interior)
        if not np.isclose(u_face, 0.0, atol=1e-10):
            return False, f"Face velocity {u_face:.4e} != 0"
        
        return True, "No-slip velocity correct"
    
    def test_surface_zero_pressure_gradient(self):
        """Test zero pressure gradient at surface."""
        Q = self.setup_state()
        # Add pressure variation
        Q[10, 1, 0] = 2.5
        
        bc = BoundaryConditions(freestream=self.setup_freestream())
        Q_bc = bc.apply_surface(Q)
        
        # Ghost pressure should equal interior (zero gradient)
        p_interior = Q_bc[10, 1, 0]
        p_ghost = Q_bc[10, 0, 0]
        
        if not np.isclose(p_ghost, p_interior):
            return False, f"p_ghost {p_ghost:.4f} != p_interior {p_interior:.4f}"
        
        return True, "Zero pressure gradient correct"
    
    def test_surface_zero_nu_tilde(self):
        """Test ν̃ = 0 at wall."""
        Q = self.setup_state()
        Q[:, :, 3] = 1e-4  # Set non-zero nu_t
        
        bc = BoundaryConditions(freestream=self.setup_freestream())
        Q_bc = bc.apply_surface(Q)
        
        # Face value should be zero
        nu_interior = Q_bc[10, 1, 3]
        nu_ghost = Q_bc[10, 0, 3]
        nu_face = 0.5 * (nu_ghost + nu_interior)
        
        if not np.isclose(nu_face, 0.0, atol=1e-10):
            return False, f"Face nu_t {nu_face:.4e} != 0"
        
        return True, "Zero nu_tilde at wall correct"
    
    # ===== Farfield BC Tests =====
    
    def test_farfield_dirichlet(self):
        """Test farfield Dirichlet conditions."""
        fs = FreestreamConditions(p_inf=1.0, u_inf=0.9, v_inf=0.05, nu_t_inf=2e-6)
        Q = initialize_state(self.NI, self.NJ, fs)
        
        # Modify interior to differ from freestream
        Q[10, -2, :] = [5.0, 2.0, 1.0, 1e-3]
        
        bc = BoundaryConditions(freestream=fs)
        Q_bc = bc.apply_farfield(Q)
        
        # Ghost cells should be set to freestream
        if not np.isclose(Q_bc[10, -1, 0], fs.p_inf):
            return False, f"p_ghost {Q_bc[10, -1, 0]:.4f} != p_inf {fs.p_inf:.4f}"
        if not np.isclose(Q_bc[10, -1, 1], fs.u_inf):
            return False, f"u_ghost {Q_bc[10, -1, 1]:.4f} != u_inf {fs.u_inf:.4f}"
        if not np.isclose(Q_bc[10, -1, 2], fs.v_inf):
            return False, f"v_ghost {Q_bc[10, -1, 2]:.4f} != v_inf {fs.v_inf:.4f}"
        if not np.isclose(Q_bc[10, -1, 3], fs.nu_t_inf):
            return False, f"nu_ghost {Q_bc[10, -1, 3]:.4e} != nu_inf {fs.nu_t_inf:.4e}"
        
        return True, "Farfield Dirichlet correct"
    
    def test_farfield_all_j_max(self):
        """Test farfield BC applies to all i indices."""
        fs = self.setup_freestream()
        Q = self.setup_state(fs)
        Q[:, -1, :] = 99.0  # Corrupt ghost cells
        
        bc = BoundaryConditions(freestream=fs)
        Q_bc = bc.apply_farfield(Q)
        
        # All j=-1 ghost cells should be freestream
        if not np.allclose(Q_bc[:, -1, 0], fs.p_inf):
            return False, "Not all ghost pressures set"
        if not np.allclose(Q_bc[:, -1, 1], fs.u_inf):
            return False, "Not all ghost u set"
        
        return True, "All farfield ghost cells set"
    
    # ===== Wake Cut BC Tests =====
    
    def test_wake_cut_left_ghost(self):
        """Test left ghost cells copy from right interior."""
        Q = self.setup_state()
        
        # Set distinct values at right interior (i = NI, which is Q[-2])
        Q[-2, :, :] = [[1.0, 2.0, 3.0, 4.0]] * (self.NJ + 2)
        
        bc = BoundaryConditions(freestream=self.setup_freestream())
        Q_bc = bc.apply_wake_cut(Q)
        
        # Left ghost (i=0) should match right interior (i=NI)
        if not np.allclose(Q_bc[0, :, :], Q[-2, :, :]):
            return False, "Left ghost doesn't match right interior"
        
        return True, "Left ghost = right interior"
    
    def test_wake_cut_right_ghost(self):
        """Test right ghost cells copy from left interior."""
        Q = self.setup_state()
        
        # Set distinct values at left interior (i = 1, which is Q[1])
        Q[1, :, :] = [[5.0, 6.0, 7.0, 8.0]] * (self.NJ + 2)
        
        bc = BoundaryConditions(freestream=self.setup_freestream())
        Q_bc = bc.apply_wake_cut(Q)
        
        # Right ghost (i=NI+1, which is Q[-1]) should match left interior (i=1)
        if not np.allclose(Q_bc[-1, :, :], Q[1, :, :]):
            return False, "Right ghost doesn't match left interior"
        
        return True, "Right ghost = left interior"
    
    def test_wake_cut_periodic(self):
        """Test wake cut is periodic (both directions)."""
        Q = self.setup_state()
        
        # Set different values at both ends
        Q[1, 5, :] = [1.0, 2.0, 3.0, 4.0]
        Q[-2, 5, :] = [5.0, 6.0, 7.0, 8.0]
        
        bc = BoundaryConditions(freestream=self.setup_freestream())
        Q_bc = bc.apply_wake_cut(Q)
        
        # Check both directions
        if not np.allclose(Q_bc[0, 5, :], Q[-2, 5, :]):
            return False, "Left ghost != right interior"
        if not np.allclose(Q_bc[-1, 5, :], Q[1, 5, :]):
            return False, "Right ghost != left interior"
        
        return True, "Periodic connection correct"
    
    # ===== Combined BC Application Tests =====
    
    def test_apply_all_bcs(self):
        """Test combined BC application."""
        fs = self.setup_freestream()
        Q = self.setup_state(fs)
        
        # Modify some values
        Q[10, 3, 0] = 5.0  # Interior pressure
        
        bc = BoundaryConditions(freestream=fs)
        Q_bc = bc.apply(Q)
        
        # Check surface BC applied
        u_face = 0.5 * (Q_bc[10, 0, 1] + Q_bc[10, 1, 1])
        if not np.isclose(u_face, 0.0, atol=1e-10):
            return False, "Surface BC not applied"
        
        # Check farfield BC applied
        if not np.isclose(Q_bc[10, -1, 1], fs.u_inf):
            return False, "Farfield BC not applied"
        
        # Check wake cut BC applied
        if not np.allclose(Q_bc[0, :, :], Q_bc[-2, :, :]):
            return False, "Wake cut BC not applied"
        
        return True, "All BCs applied correctly"
    
    def test_apply_bc_convenience_function(self):
        """Test convenience function."""
        fs = self.setup_freestream()
        Q = self.setup_state(fs)
        
        Q_bc = apply_boundary_conditions(Q, fs)
        
        # Should produce same result as class method
        bc = BoundaryConditions(freestream=fs)
        Q_bc_class = bc.apply(Q)
        
        if not np.allclose(Q_bc, Q_bc_class):
            return False, "Convenience function differs from class"
        
        return True, "Convenience function works"
    
    # ===== Inlet/Outlet BC Tests =====
    
    def test_inlet_outlet_bc(self):
        """Test inlet/outlet BC for channel flow."""
        Q = self.setup_state()
        
        io_bc = InletOutletBC(inlet_velocity=(1.0, 0.0), p_outlet=0.0)
        Q_bc = io_bc.apply(Q)
        
        # Check inlet velocity (face average should be inlet velocity)
        u_face_inlet = 0.5 * (Q_bc[0, 5, 1] + Q_bc[1, 5, 1])
        if not np.isclose(u_face_inlet, 1.0, rtol=0.01):
            return False, f"Inlet u_face {u_face_inlet:.4f} != 1.0"
        
        # Check outlet extrapolation
        if not np.allclose(Q_bc[-1, :, :], Q_bc[-2, :, :]):
            return False, "Outlet not extrapolated"
        
        # Check wall no-slip (j=0)
        u_face_wall = 0.5 * (Q_bc[5, 0, 1] + Q_bc[5, 1, 1])
        if not np.isclose(u_face_wall, 0.0, atol=1e-10):
            return False, f"Wall u_face {u_face_wall:.4e} != 0"
        
        return True, "Inlet/outlet BCs correct"
    
    # ===== BC Stability Tests =====
    
    def test_bc_repeated_application(self):
        """BCs should be idempotent (applying twice = applying once)."""
        Q = self.setup_state()
        Q[10, 5, :] = [1.0, 2.0, 3.0, 4.0]  # Some interior values
        
        bc = BoundaryConditions(freestream=self.setup_freestream())
        
        Q_once = bc.apply(Q)
        Q_twice = bc.apply(Q_once)
        
        if not np.allclose(Q_once, Q_twice):
            max_diff = np.max(np.abs(Q_once - Q_twice))
            return False, f"Not idempotent, max diff = {max_diff:.2e}"
        
        return True, "BCs are idempotent"
    
    def test_bc_preserves_interior(self):
        """BCs should not modify interior cells."""
        Q = self.setup_state()
        
        # Set specific interior values
        Q[5, 5, :] = [10.0, 20.0, 30.0, 40.0]
        Q[15, 10, :] = [11.0, 21.0, 31.0, 41.0]
        
        Q_original = Q.copy()
        
        bc = BoundaryConditions(freestream=self.setup_freestream())
        Q_bc = bc.apply(Q)
        
        # Interior should be unchanged
        if not np.allclose(Q_bc[1:-1, 1:-1, :], Q_original[1:-1, 1:-1, :]):
            return False, "Interior modified"
        
        return True, "Interior preserved"
    
    def test_wake_cut_nonuniform_flow(self):
        """Wake cut BC should work for non-uniform flow (with lift).
        
        In a C-grid, the wake cut is a PERIODIC boundary where:
        - Cell i=1 (first interior) and cell i=NI (last interior) are neighbors
        - Ghost cells provide data from the opposite side for flux calculation
        - The grid metrics (face normals) handle the geometric reflection
        - Simple copy is correct: Q[0] = Q[-2], Q[-1] = Q[1]
        
        No velocity negation is needed because the face normals encode the geometry.
        """
        Q = self.setup_state()
        
        # Set non-uniform velocity field (simulating circulation)
        # First interior row has different v than last interior row
        Q[1, :, 2] = 0.1   # v = 0.1 on first interior
        Q[-2, :, 2] = -0.05  # v = -0.05 on last interior
        
        bc = BoundaryConditions(freestream=self.setup_freestream())
        Q_bc = bc.apply_wake_cut(Q)
        
        # Ghost cells should copy directly (no negation)
        if not np.allclose(Q_bc[0, :, 2], Q[-2, :, 2]):
            return False, f"Left ghost v={Q_bc[0, 4, 2]:.3f} != last interior v={Q[-2, 4, 2]:.3f}"
        if not np.allclose(Q_bc[-1, :, 2], Q[1, :, 2]):
            return False, f"Right ghost v={Q_bc[-1, 4, 2]:.3f} != first interior v={Q[1, 4, 2]:.3f}"
        
        return True, "Non-uniform flow handled correctly"
    
    # ===== Corner Ghost Cell Tests =====
    
    def test_corner_wall_wake_trailing_edge(self):
        """Corner at wall-wake intersection (trailing edge).
        
        Corners: Q[0, 0, :] and Q[-1, 0, :]
        
        Physics: At the trailing edge, the wake cut connects the upper and lower
        surfaces. The corner ghost cells are set by the wake cut BC (applied last),
        which copies from the opposite interior cell.
        
        This is correct because:
        - The trailing edge is where the two wake branches meet
        - The periodic BC ensures flow continuity across the cut
        """
        Q = self.setup_state()
        
        # Set distinct values to track what gets copied where
        Q[1, 1, 0] = 100.0    # Left interior near wall
        Q[-2, 1, 0] = 200.0   # Right interior near wall
        
        bc = BoundaryConditions(freestream=self.setup_freestream())
        Q_bc = bc.apply(Q)
        
        # Corners should be set by wake cut (last applied)
        # Q[0, 0] should copy from Q[-2, 0] (right interior at wall)
        # Q[-1, 0] should copy from Q[1, 0] (left interior at wall)
        
        # Note: surface BC was applied first, so Q[-2, 0] = -Q[-2, 1] (mirrored)
        # Then wake cut copies Q[0, :] = Q[-2, :], so Q[0, 0] = Q[-2, 0] = -Q[-2, 1]
        
        # Check the corner is set (not undefined/NaN)
        if np.isnan(Q_bc[0, 0, 0]) or np.isnan(Q_bc[-1, 0, 0]):
            return False, "Corner contains NaN"
        
        # Verify wake cut BC was applied to corners
        if not np.allclose(Q_bc[0, 0, :], Q_bc[-2, 0, :]):
            return False, "Left wall-wake corner != right interior at wall"
        if not np.allclose(Q_bc[-1, 0, :], Q_bc[1, 0, :]):
            return False, "Right wall-wake corner != left interior at wall"
        
        return True, "Wall-wake corners set by wake cut BC"
    
    def test_corner_farfield_wake(self):
        """Corner at farfield-wake intersection.
        
        Corners: Q[0, -1, :] and Q[-1, -1, :]
        
        Physics: These corners are far from the airfoil in the wake region.
        The wake cut BC is applied last, so corners copy from the opposite
        interior cells at the farfield level.
        
        This maintains periodicity across the full wake cut.
        """
        Q = self.setup_state()
        
        # Set distinct values in farfield interior cells
        Q[1, -2, 0] = 300.0    # Left interior at farfield
        Q[-2, -2, 0] = 400.0   # Right interior at farfield
        
        bc = BoundaryConditions(freestream=self.setup_freestream())
        Q_bc = bc.apply(Q)
        
        # Farfield BC sets Q[:, -1] = freestream values
        # Wake cut BC then copies Q[0, :] = Q[-2, :] and Q[-1, :] = Q[1, :]
        # So corners Q[0, -1] and Q[-1, -1] come from wake cut (periodic)
        
        # Check corners are set
        if np.isnan(Q_bc[0, -1, 0]) or np.isnan(Q_bc[-1, -1, 0]):
            return False, "Farfield-wake corner contains NaN"
        
        # Verify wake cut BC was applied to corners
        if not np.allclose(Q_bc[0, -1, :], Q_bc[-2, -1, :]):
            return False, "Left farfield-wake corner != right interior at farfield"
        if not np.allclose(Q_bc[-1, -1, :], Q_bc[1, -1, :]):
            return False, "Right farfield-wake corner != left interior at farfield"
        
        return True, "Farfield-wake corners set by wake cut BC"
    
    def test_corner_consistency_after_repeated_application(self):
        """Corners should be consistent after repeated BC application.
        
        If BCs are applied multiple times (as in time-stepping), the corner
        values should remain stable (idempotent).
        """
        Q = self.setup_state()
        
        # Set some non-uniform values
        Q[5, 5, :] = [10.0, 2.0, 0.5, 1e-5]
        
        bc = BoundaryConditions(freestream=self.setup_freestream())
        
        # Apply BCs multiple times
        Q1 = bc.apply(Q)
        Q2 = bc.apply(Q1)
        Q3 = bc.apply(Q2)
        
        # All four corners should be identical across applications
        corners = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
        for i, j in corners:
            if not np.allclose(Q1[i, j, :], Q2[i, j, :]):
                return False, f"Corner ({i},{j}) changed on 2nd application"
            if not np.allclose(Q2[i, j, :], Q3[i, j, :]):
                return False, f"Corner ({i},{j}) changed on 3rd application"
        
        return True, "Corners stable across repeated BC application"
    
    def test_corner_velocity_at_trailing_edge(self):
        """At trailing edge corners, velocity should satisfy wall BC.
        
        Physics check: Even though wake cut BC sets the corner ghost cells,
        the resulting velocity at the trailing edge face should still be
        consistent with no-slip (u ≈ 0).
        
        This is because the wake cut copies from interior cells that have
        already been processed by the surface BC.
        """
        Q = self.setup_state()
        
        # Set interior velocity near trailing edge
        Q[1, 1, 1] = 5.0   # u at left side
        Q[1, 1, 2] = 2.0   # v at left side
        Q[-2, 1, 1] = 5.0  # u at right side
        Q[-2, 1, 2] = -2.0 # v at right side (opposite due to geometry)
        
        bc = BoundaryConditions(freestream=self.setup_freestream())
        Q_bc = bc.apply(Q)
        
        # At the wall face (between j=0 ghost and j=1 interior),
        # the face velocity should be (ghost + interior) / 2
        
        # For the corner at i=0 (left ghost), j=0 (wall ghost):
        # The corner ghost Q[0, 0] was set by wake cut from Q[-2, 0]
        # Q[-2, 0] was set by wall BC to mirror Q[-2, 1]
        
        # Check that wall-adjacent face velocity is ~0
        # Face between Q[1, 0] and Q[1, 1]
        u_face_left = 0.5 * (Q_bc[1, 0, 1] + Q_bc[1, 1, 1])
        u_face_right = 0.5 * (Q_bc[-2, 0, 1] + Q_bc[-2, 1, 1])
        
        if not np.isclose(u_face_left, 0.0, atol=1e-10):
            return False, f"Left wall face u = {u_face_left:.3f} != 0"
        if not np.isclose(u_face_right, 0.0, atol=1e-10):
            return False, f"Right wall face u = {u_face_right:.3f} != 0"
        
        return True, "Wall face velocity = 0 near trailing edge"
    
    def run_all(self):
        """Run all tests."""
        # Freestream tests
        self.run_test("freestream_from_mach_alpha", self.test_freestream_from_mach_alpha)
        self.run_test("freestream_zero_alpha", self.test_freestream_zero_alpha)
        
        # Initialization tests
        self.run_test("initialize_state_shape", self.test_initialize_state_shape)
        self.run_test("initialize_state_values", self.test_initialize_state_values)
        
        # Surface BC tests
        self.run_test("surface_no_slip_velocity", self.test_surface_no_slip_velocity)
        self.run_test("surface_zero_pressure_gradient", self.test_surface_zero_pressure_gradient)
        self.run_test("surface_zero_nu_tilde", self.test_surface_zero_nu_tilde)
        
        # Farfield BC tests
        self.run_test("farfield_dirichlet", self.test_farfield_dirichlet)
        self.run_test("farfield_all_j_max", self.test_farfield_all_j_max)
        
        # Wake cut BC tests
        self.run_test("wake_cut_left_ghost", self.test_wake_cut_left_ghost)
        self.run_test("wake_cut_right_ghost", self.test_wake_cut_right_ghost)
        self.run_test("wake_cut_periodic", self.test_wake_cut_periodic)
        self.run_test("wake_cut_nonuniform_flow", self.test_wake_cut_nonuniform_flow)
        
        # Corner ghost cell tests
        self.run_test("corner_wall_wake", self.test_corner_wall_wake_trailing_edge)
        self.run_test("corner_farfield_wake", self.test_corner_farfield_wake)
        self.run_test("corner_stability", self.test_corner_consistency_after_repeated_application)
        self.run_test("corner_trailing_edge_noslip", self.test_corner_velocity_at_trailing_edge)
        
        # Combined BC tests
        self.run_test("apply_all_bcs", self.test_apply_all_bcs)
        self.run_test("apply_bc_convenience_function", self.test_apply_bc_convenience_function)
        
        # Inlet/outlet BC tests
        self.run_test("inlet_outlet_bc", self.test_inlet_outlet_bc)
        
        # Stability tests
        self.run_test("bc_repeated_application", self.test_bc_repeated_application)
        self.run_test("bc_preserves_interior", self.test_bc_preserves_interior)
        
        return self.results


def main():
    """Run all boundary condition tests."""
    print("=" * 60)
    print("Boundary Conditions Module Tests")
    print("=" * 60)
    
    tests = BoundaryConditionTests()
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

