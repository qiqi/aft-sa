"""Tests for face geometry computation (tight-stencil viscous flux support)."""

import numpy as np
import pytest
from src.grid.metrics import MetricComputer, FaceGeometry, LSWeights, _compute_ls_weights_qr


def create_skewed_grid(NI, NJ, skew_factor=0.15):
    """Create a non-Cartesian skewed grid for testing.
    
    Parameters
    ----------
    NI, NJ : int
        Number of cells in each direction.
    skew_factor : float
        Amount of grid distortion (0 = Cartesian).
    
    Returns
    -------
    X, Y : arrays of shape (NI+1, NJ+1)
        Grid node coordinates.
    """
    # Base grid
    x_base = np.linspace(0, 2, NI + 1)
    y_base = np.linspace(0, 1.5, NJ + 1)
    X_base, Y_base = np.meshgrid(x_base, y_base, indexing='ij')
    
    # Add sinusoidal distortion
    X = X_base + skew_factor * np.sin(2 * np.pi * Y_base / 1.5)
    Y = Y_base + skew_factor * np.sin(np.pi * X_base / 2)
    
    return X, Y


class TestFaceGeometry:
    """Tests for FaceGeometry computation."""
    
    def test_cartesian_grid_i_faces(self):
        """On uniform Cartesian grid, e_coord_i should point in +x direction."""
        # Create uniform 3x3 grid (4x4 nodes)
        NI, NJ = 3, 3
        x = np.linspace(0, 3, NI + 1)
        y = np.linspace(0, 3, NJ + 1)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        computer = MetricComputer(X, Y)
        face_geom = computer.compute_face_geometry()
        
        # I-faces should have e_coord pointing in +x direction
        # (from cell i to cell i+1)
        assert face_geom.e_coord_i_x.shape == (NI + 1, NJ)
        
        # Interior faces (1 to NI-1) should have e_coord = (1, 0)
        np.testing.assert_allclose(face_geom.e_coord_i_x[1:-1, :], 1.0, atol=1e-10)
        np.testing.assert_allclose(face_geom.e_coord_i_y[1:-1, :], 0.0, atol=1e-10)
        
        # e_ortho should be perpendicular: (0, 1) for CCW rotation
        np.testing.assert_allclose(face_geom.e_ortho_i_x[1:-1, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(face_geom.e_ortho_i_y[1:-1, :], 1.0, atol=1e-10)
        
        # Distance should be 1.0 (cell spacing)
        np.testing.assert_allclose(face_geom.d_coord_i[1:-1, :], 1.0, atol=1e-10)
    
    def test_cartesian_grid_j_faces(self):
        """On uniform Cartesian grid, e_coord_j should point in +y direction."""
        NI, NJ = 3, 3
        x = np.linspace(0, 3, NI + 1)
        y = np.linspace(0, 3, NJ + 1)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        computer = MetricComputer(X, Y)
        face_geom = computer.compute_face_geometry()
        
        # J-faces should have e_coord pointing in +y direction
        assert face_geom.e_coord_j_x.shape == (NI, NJ + 1)
        
        # Interior faces should have e_coord = (0, 1)
        np.testing.assert_allclose(face_geom.e_coord_j_x[:, 1:-1], 0.0, atol=1e-10)
        np.testing.assert_allclose(face_geom.e_coord_j_y[:, 1:-1], 1.0, atol=1e-10)
        
        # e_ortho should be perpendicular: (-1, 0) for CCW rotation
        np.testing.assert_allclose(face_geom.e_ortho_j_x[:, 1:-1], -1.0, atol=1e-10)
        np.testing.assert_allclose(face_geom.e_ortho_j_y[:, 1:-1], 0.0, atol=1e-10)
        
        # Distance should be 1.0
        np.testing.assert_allclose(face_geom.d_coord_j[:, 1:-1], 1.0, atol=1e-10)
    
    def test_orthogonality_skewed_grid(self):
        """Verify e_coord · e_ortho = 0 for all faces on skewed grid."""
        NI, NJ = 6, 5
        X, Y = create_skewed_grid(NI, NJ, skew_factor=0.2)
        
        computer = MetricComputer(X, Y)
        face_geom = computer.compute_face_geometry()
        
        # Check orthogonality for I-faces
        dot_i = (face_geom.e_coord_i_x * face_geom.e_ortho_i_x + 
                 face_geom.e_coord_i_y * face_geom.e_ortho_i_y)
        np.testing.assert_allclose(dot_i, 0.0, atol=1e-10)
        
        # Check orthogonality for J-faces
        dot_j = (face_geom.e_coord_j_x * face_geom.e_ortho_j_x + 
                 face_geom.e_coord_j_y * face_geom.e_ortho_j_y)
        np.testing.assert_allclose(dot_j, 0.0, atol=1e-10)
    
    def test_unit_vectors_skewed_grid(self):
        """Verify e_coord and e_ortho are unit vectors on skewed grid."""
        NI, NJ = 5, 4
        X, Y = create_skewed_grid(NI, NJ, skew_factor=0.25)
        
        computer = MetricComputer(X, Y)
        face_geom = computer.compute_face_geometry()
        
        # Check |e_coord| = 1 for I-faces
        mag_coord_i = np.sqrt(face_geom.e_coord_i_x**2 + face_geom.e_coord_i_y**2)
        np.testing.assert_allclose(mag_coord_i, 1.0, atol=1e-10)
        
        # Check |e_ortho| = 1 for I-faces
        mag_ortho_i = np.sqrt(face_geom.e_ortho_i_x**2 + face_geom.e_ortho_i_y**2)
        np.testing.assert_allclose(mag_ortho_i, 1.0, atol=1e-10)
        
        # Check |e_coord| = 1 for J-faces
        mag_coord_j = np.sqrt(face_geom.e_coord_j_x**2 + face_geom.e_coord_j_y**2)
        np.testing.assert_allclose(mag_coord_j, 1.0, atol=1e-10)
        
        # Check |e_ortho| = 1 for J-faces
        mag_ortho_j = np.sqrt(face_geom.e_ortho_j_x**2 + face_geom.e_ortho_j_y**2)
        np.testing.assert_allclose(mag_ortho_j, 1.0, atol=1e-10)
    
    def test_skewed_grid_distances(self):
        """On skewed grid, verify d_coord matches cell center distances."""
        NI, NJ = 4, 4
        X, Y = create_skewed_grid(NI, NJ, skew_factor=0.2)
        
        computer = MetricComputer(X, Y)
        face_geom = computer.compute_face_geometry()
        
        # Compute cell centers manually
        xc = 0.25 * (X[:-1, :-1] + X[1:, :-1] + X[1:, 1:] + X[:-1, 1:])
        yc = 0.25 * (Y[:-1, :-1] + Y[1:, :-1] + Y[1:, 1:] + Y[:-1, 1:])
        
        # Check I-face distances (interior faces only)
        for i in range(1, NI):
            for j in range(NJ):
                expected_d = np.sqrt((xc[i, j] - xc[i-1, j])**2 + (yc[i, j] - yc[i-1, j])**2)
                np.testing.assert_allclose(face_geom.d_coord_i[i, j], expected_d, rtol=1e-10)
        
        # Check J-face distances (interior faces only)
        for i in range(NI):
            for j in range(1, NJ):
                expected_d = np.sqrt((xc[i, j] - xc[i, j-1])**2 + (yc[i, j] - yc[i, j-1])**2)
                np.testing.assert_allclose(face_geom.d_coord_j[i, j], expected_d, rtol=1e-10)


class TestLSWeights:
    """Tests for least-squares weight computation."""
    
    def test_ls_weights_sum_zero_skewed(self):
        """Verify sum of weights = 0 for interior faces on skewed grid."""
        NI, NJ = 6, 5
        X, Y = create_skewed_grid(NI, NJ, skew_factor=0.2)
        
        computer = MetricComputer(X, Y)
        face_geom = computer.compute_face_geometry()
        ls_weights = computer.compute_ls_weights(face_geom)
        
        # Sum of weights should be zero for interior faces
        # (boundary faces may have issues due to edge padding)
        np.testing.assert_allclose(
            np.sum(ls_weights.weights_i[1:-1, 1:-1, :], axis=-1), 0.0, atol=1e-10
        )
        np.testing.assert_allclose(
            np.sum(ls_weights.weights_j[1:-1, 1:-1, :], axis=-1), 0.0, atol=1e-10
        )
    
    def test_ls_weights_constant_field_skewed(self):
        """For constant field on skewed grid, weighted sum should be zero."""
        NI, NJ = 5, 4
        X, Y = create_skewed_grid(NI, NJ, skew_factor=0.25)
        
        computer = MetricComputer(X, Y)
        face_geom = computer.compute_face_geometry()
        ls_weights = computer.compute_ls_weights(face_geom)
        
        # Create constant field
        phi = np.ones((NI, NJ)) * 7.3
        phi_padded = np.pad(phi, ((1, 1), (1, 1)), mode='edge')
        
        # Check I-faces (interior only to avoid boundary issues)
        for i_face in range(1, NI):
            for j in range(NJ):
                # Extract 6-cell stencil
                stencil_i = [i_face, i_face, i_face, i_face + 1, i_face + 1, i_face + 1]
                stencil_j = [j, j + 1, j + 2, j, j + 1, j + 2]
                phi_stencil = np.array([phi_padded[si, sj] for si, sj in zip(stencil_i, stencil_j)])
                
                grad_ortho = np.dot(ls_weights.weights_i[i_face, j, :], phi_stencil)
                assert abs(grad_ortho) < 1e-10, f"Constant field should give zero gradient at I-face ({i_face}, {j})"
        
        # Check J-faces
        for i in range(NI):
            for j_face in range(1, NJ):
                stencil_i = [i, i + 1, i + 2, i, i + 1, i + 2]
                stencil_j = [j_face, j_face, j_face, j_face + 1, j_face + 1, j_face + 1]
                phi_stencil = np.array([phi_padded[si, sj] for si, sj in zip(stencil_i, stencil_j)])
                
                grad_ortho = np.dot(ls_weights.weights_j[i, j_face, :], phi_stencil)
                assert abs(grad_ortho) < 1e-10, f"Constant field should give zero gradient at J-face ({i}, {j_face})"
    
    def test_ls_weights_linear_field_skewed(self):
        """For linear field on skewed grid, verify exact derivative."""
        NI, NJ = 5, 5
        X, Y = create_skewed_grid(NI, NJ, skew_factor=0.2)
        
        computer = MetricComputer(X, Y)
        face_geom = computer.compute_face_geometry()
        ls_weights = computer.compute_ls_weights(face_geom)
        
        # Compute cell centers
        xc = 0.25 * (X[:-1, :-1] + X[1:, :-1] + X[1:, 1:] + X[:-1, 1:])
        yc = 0.25 * (Y[:-1, :-1] + Y[1:, :-1] + Y[1:, 1:] + Y[:-1, 1:])
        
        # Linear field: phi = 2*x + 3*y
        a, b = 2.0, 3.0
        phi = a * xc + b * yc
        phi_padded = np.pad(phi, ((1, 1), (1, 1)), mode='edge')
        
        # For a linear field, the ortho derivative should be a*e_ortho_x + b*e_ortho_y
        
        # Check interior I-faces
        for i_face in range(1, NI):
            for j in range(1, NJ - 1):  # Avoid boundary stencil issues
                stencil_i = [i_face, i_face, i_face, i_face + 1, i_face + 1, i_face + 1]
                stencil_j = [j, j + 1, j + 2, j, j + 1, j + 2]
                phi_stencil = np.array([phi_padded[si, sj] for si, sj in zip(stencil_i, stencil_j)])
                
                grad_ortho = np.dot(ls_weights.weights_i[i_face, j, :], phi_stencil)
                
                # Expected ortho derivative = grad_phi · e_ortho = (a, b) · (e_ortho_x, e_ortho_y)
                expected = a * face_geom.e_ortho_i_x[i_face, j] + b * face_geom.e_ortho_i_y[i_face, j]
                np.testing.assert_allclose(
                    grad_ortho, expected, rtol=1e-5,
                    err_msg=f"Linear field derivative wrong at I-face ({i_face}, {j})"
                )
        
        # Check interior J-faces
        for i in range(1, NI - 1):  # Avoid boundary stencil issues
            for j_face in range(1, NJ):
                stencil_i = [i, i + 1, i + 2, i, i + 1, i + 2]
                stencil_j = [j_face, j_face, j_face, j_face + 1, j_face + 1, j_face + 1]
                phi_stencil = np.array([phi_padded[si, sj] for si, sj in zip(stencil_i, stencil_j)])
                
                grad_ortho = np.dot(ls_weights.weights_j[i, j_face, :], phi_stencil)
                
                expected = a * face_geom.e_ortho_j_x[i, j_face] + b * face_geom.e_ortho_j_y[i, j_face]
                np.testing.assert_allclose(
                    grad_ortho, expected, rtol=1e-5,
                    err_msg=f"Linear field derivative wrong at J-face ({i}, {j_face})"
                )
    
    def test_qr_helper_function(self):
        """Test the QR helper directly with non-uniform spacing."""
        # Non-uniform spacing simulating skewed grid
        s = np.array([-1.2, 0.1, 0.9, -1.1, -0.05, 1.1])  # ortho positions
        t = np.array([-0.6, -0.55, -0.5, 0.45, 0.5, 0.6])  # coord positions
        
        w = _compute_ls_weights_qr(s, t)
        
        # Check constraints
        np.testing.assert_allclose(np.sum(w), 0.0, atol=1e-10)  # sum = 0
        np.testing.assert_allclose(np.dot(w, s), 1.0, atol=1e-10)  # ortho deriv = 1
        np.testing.assert_allclose(np.dot(w, t), 0.0, atol=1e-10)  # coord deriv = 0
    
    def test_ls_weights_shapes_skewed(self):
        """Verify correct shapes of weight arrays on skewed grid."""
        NI, NJ = 6, 5
        X, Y = create_skewed_grid(NI, NJ, skew_factor=0.15)
        
        computer = MetricComputer(X, Y)
        ls_weights = computer.compute_ls_weights()
        
        assert ls_weights.weights_i.shape == (NI + 1, NJ, 6)
        assert ls_weights.weights_j.shape == (NI, NJ + 1, 6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
