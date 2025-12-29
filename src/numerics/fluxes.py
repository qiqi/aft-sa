"""
JST (Jameson-Schmidt-Turkel) Flux Scheme for 2D Incompressible RANS.

This module implements the central-difference flux scheme with scalar
artificial dissipation, as described by Jameson, Schmidt, and Turkel (1981).

Physics: Artificial Compressibility formulation for incompressible flow.
    - State vector: Q = [p, u, v, ν̃]  (pressure, velocities, SA variable)
    - Pseudo-compressibility: ∂p/∂τ + β∇·u = 0
    - β is the artificial compressibility parameter

References:
    [1] Jameson, A., Schmidt, W., & Turkel, E. (1981). "Numerical solution of
        the Euler equations by finite volume methods using Runge-Kutta time
        stepping schemes." AIAA Paper 81-1259.
    [2] Turkel, E. (1987). "Preconditioned methods for solving the 
        incompressible and low speed compressible equations." JCP 72.
"""

import numpy as np
from typing import Tuple, NamedTuple, Optional
from dataclasses import dataclass


@dataclass
class FluxConfig:
    """Configuration for JST flux computation."""
    
    # JST dissipation coefficients
    # ε(2) = k2 * ν   where ν is pressure sensor (0 to 1)
    # ε(4) = max(0, k4 - ε(2))
    k2: float = 0.5      # 2nd-order dissipation coefficient (typically 0.25-1.0)
    k4: float = 0.016    # 4th-order dissipation coefficient (typically 1/64-1/32)
    
    # Limiter for pressure sensor
    eps_p: float = 1e-10  # Small number to avoid division by zero


class GridMetrics(NamedTuple):
    """Grid metrics for flux computation.
    
    For a structured grid with indices (i, j):
    - I-faces are between cells (i, j) and (i+1, j)
    - J-faces are between cells (i, j) and (i, j+1)
    
    All arrays include ghost cells: shape (NI+2, NJ+2) for cell-centered,
    (NI+1, NJ+2) for I-faces, (NI+2, NJ+1) for J-faces.
    """
    # I-face normals and areas (faces at constant i)
    # Shape: (NI+1, NJ) - one more face than cells in i-direction
    Si_x: np.ndarray    # x-component of I-face normal (scaled by area)
    Si_y: np.ndarray    # y-component of I-face normal (scaled by area)
    
    # J-face normals and areas (faces at constant j)  
    # Shape: (NI, NJ+1) - one more face than cells in j-direction
    Sj_x: np.ndarray    # x-component of J-face normal (scaled by area)
    Sj_y: np.ndarray    # y-component of J-face normal (scaled by area)
    
    # Cell volumes
    # Shape: (NI, NJ)
    volume: np.ndarray


def compute_convective_flux(Q: np.ndarray, nx: np.ndarray, ny: np.ndarray, 
                            beta: float) -> np.ndarray:
    """
    Compute the projected convective flux F·n for given states.
    
    For artificial compressibility, the flux vector is:
        F_c = [β(u·n), u(u·n) + p*nx, v(u·n) + p*ny, ν̃(u·n)]
    
    where u·n = u*nx + v*ny is the normal velocity.
    
    Parameters
    ----------
    Q : ndarray, shape (..., 4)
        State vector [p, u, v, nu_tilde].
    nx, ny : ndarray, shape (...)
        Face normal components (already scaled by face area).
    beta : float
        Artificial compressibility parameter.
        
    Returns
    -------
    F : ndarray, shape (..., 4)
        Projected convective flux.
    """
    p = Q[..., 0]
    u = Q[..., 1]
    v = Q[..., 2]
    nu_t = Q[..., 3]
    
    # Contravariant velocity (normal velocity * area)
    U_n = u * nx + v * ny
    
    # Flux components
    F = np.zeros_like(Q)
    F[..., 0] = beta * U_n                  # Continuity (pseudo-compressibility)
    F[..., 1] = u * U_n + p * nx            # x-momentum
    F[..., 2] = v * U_n + p * ny            # y-momentum
    F[..., 3] = nu_t * U_n                  # SA transport
    
    return F


def compute_pressure_sensor(p: np.ndarray, eps: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the pressure-based switching function for JST dissipation.
    
    The sensor detects shocks/discontinuities using the normalized
    second difference of pressure:
    
        ν_i = |p_{i+1} - 2p_i + p_{i-1}| / (p_{i+1} + 2p_i + p_{i-1} + eps)
    
    This sensor is ~0 in smooth regions and ~1 near discontinuities.
    
    Parameters
    ----------
    p : ndarray, shape (NI+2, NJ+2)
        Pressure field including ghost cells.
    eps : float
        Small number to prevent division by zero.
        
    Returns
    -------
    nu_i : ndarray, shape (NI, NJ+2)
        Sensor values at I-faces (between cells i and i+1).
    nu_j : ndarray, shape (NI+2, NJ)
        Sensor values at J-faces (between cells j and j+1).
    """
    # Second difference in i-direction: d2p_i = p[i+1] - 2*p[i] + p[i-1]
    # Computed at cell centers, then we take max of neighbors at faces
    d2p_i = np.abs(p[2:, :] - 2*p[1:-1, :] + p[:-2, :])
    sum_p_i = p[2:, :] + 2*p[1:-1, :] + p[:-2, :] + eps
    nu_cells_i = d2p_i / sum_p_i  # Shape: (NI, NJ+2)
    
    # At face (i+1/2), take max of sensors at cells i and i+1
    # nu_cells_i[i] is sensor at cell i+1 (since we start from index 1)
    # We need sensor at faces, so we take max of left and right cell sensors
    # Pad with zeros at boundaries
    nu_i = np.maximum(
        np.pad(nu_cells_i[:-1, :], ((1, 0), (0, 0)), mode='constant'),
        np.pad(nu_cells_i[1:, :], ((0, 1), (0, 0)), mode='constant')
    )
    # Actually, let's be more careful. nu_cells_i has shape (NI, NJ+2)
    # where nu_cells_i[i, j] is the sensor at interior cell (i+1, j) 
    # (since p[1:-1] corresponds to interior cells 1 to NI)
    
    # For I-faces: face index f goes from 0 to NI (NI+1 faces)
    # Face f is between cells f and f+1 in the original (with ghosts) indexing
    # The sensor at face f should be max(nu[f], nu[f+1]) of cell-centered values
    # But nu_cells_i only exists for interior cells...
    
    # Simpler approach: compute sensor directly at faces
    # Use 4-point stencil centered at face
    d2p_face_i = np.abs(p[2:, :] - 2*p[1:-1, :] + p[:-2, :])  # at cells
    # This gives us sensors at cells 1 to NI (interior cells)
    # For faces, we average or take max of neighbors
    # Face i is between cell i and cell i+1
    # So face 1 uses cells 1 and 2, etc.
    
    # Let's just compute a simple sensor that works for the interior
    # and handle boundaries separately
    
    # Second difference in j-direction
    d2p_j = np.abs(p[:, 2:] - 2*p[:, 1:-1] + p[:, :-2])
    sum_p_j = p[:, 2:] + 2*p[:, 1:-1] + p[:, :-2] + eps
    nu_cells_j = d2p_j / sum_p_j  # Shape: (NI+2, NJ)
    
    nu_j = np.maximum(
        np.pad(nu_cells_j[:, :-1], ((0, 0), (1, 0)), mode='constant'),
        np.pad(nu_cells_j[:, 1:], ((0, 0), (0, 1)), mode='constant')
    )
    
    return nu_cells_i, nu_cells_j


def compute_spectral_radius(Q: np.ndarray, nx: np.ndarray, ny: np.ndarray,
                           beta: float) -> np.ndarray:
    """
    Compute the spectral radius of the flux Jacobian.
    
    For artificial compressibility:
        λ = |U_n| + c_art * |S|
        
    where:
        U_n = u*nx + v*ny (normal velocity, already scaled by area)
        c_art = sqrt(u² + v² + β) (artificial speed of sound)
        |S| = sqrt(nx² + ny²) (face area)
    
    Parameters
    ----------
    Q : ndarray, shape (..., 4)
        State vector.
    nx, ny : ndarray
        Face normal components (scaled by area).
    beta : float
        Artificial compressibility parameter.
        
    Returns
    -------
    lambda_ : ndarray
        Spectral radius at each face.
    """
    u = Q[..., 1]
    v = Q[..., 2]
    
    # Face area
    S = np.sqrt(nx**2 + ny**2)
    
    # Unit normal
    nx_hat = nx / (S + 1e-12)
    ny_hat = ny / (S + 1e-12)
    
    # Normal velocity (not scaled by area)
    U_n = u * nx_hat + v * ny_hat
    
    # Artificial speed of sound
    c_art = np.sqrt(u**2 + v**2 + beta)
    
    # Spectral radius (scaled by face area)
    return (np.abs(U_n) + c_art) * S


def compute_jst_dissipation_i(Q: np.ndarray, lambda_i: np.ndarray, 
                               nu_i: np.ndarray, cfg: FluxConfig) -> np.ndarray:
    """
    Compute JST artificial dissipation for I-faces.
    
    The dissipation flux is:
        D = ε(2) * (Q_R - Q_L) - ε(4) * (Q_{R+1} - 3*Q_R + 3*Q_L - Q_{L-1})
        
    where:
        ε(2) = k2 * max(ν_L, ν_R) * λ  (2nd-order, shock-capturing)
        ε(4) = max(0, k4 - ε(2)/λ) * λ  (4th-order, background)
        
    Parameters
    ----------
    Q : ndarray, shape (NI+2, NJ+2, 4)
        State vector with ghost cells.
    lambda_i : ndarray, shape (NI+1, NJ)
        Spectral radius at I-faces.
    nu_i : ndarray, shape (NI, NJ)
        Pressure sensor at interior cells.
    cfg : FluxConfig
        Dissipation coefficients.
        
    Returns
    -------
    D_i : ndarray, shape (NI+1, NJ, 4)
        Dissipation flux at I-faces.
    """
    # Interior faces only (face index 1 to NI-1 maps to cells 1:NI and 2:NI+1)
    # With ghosts: cells 0 to NI+1, faces 0 to NI
    
    # For face at index f (between cells f and f+1):
    # Q_L = Q[f], Q_R = Q[f+1]
    # Q_{L-1} = Q[f-1], Q_{R+1} = Q[f+2]
    
    NI_ghost, NJ_ghost, _ = Q.shape
    NI = NI_ghost - 2
    NJ = NJ_ghost - 2
    
    # Extract states for all faces (interior faces: 1 to NI-1)
    # Face f has L=f, R=f+1, L-1=f-1, R+1=f+2
    # For faces 1 to NI-1: need cells 0 to NI+1 (which we have with ghosts)
    
    # Slicing for interior faces (excluding boundary faces 0 and NI)
    Q_Lm1 = Q[:-3, 1:-1, :]   # Q[0:NI-1] -> cells 0 to NI-2
    Q_L   = Q[1:-2, 1:-1, :]  # Q[1:NI]   -> cells 1 to NI-1
    Q_R   = Q[2:-1, 1:-1, :]  # Q[2:NI+1] -> cells 2 to NI
    Q_Rp1 = Q[3:, 1:-1, :]    # Q[3:NI+2] -> cells 3 to NI+1
    
    # This gives us NI-2 interior faces (faces 1 to NI-2)
    # Actually let me reconsider the indexing...
    
    # With ghost cells: indices 0, 1, ..., NI, NI+1
    # Interior cells: 1, ..., NI
    # Interior faces in i-direction: between cell i and i+1 for i=1,...,NI-1
    # So NI-1 interior faces
    # We need 4 cells for each face: i-1, i, i+1, i+2
    # For face between cells 1 and 2: need cells 0, 1, 2, 3
    # For face between cells NI-1 and NI: need cells NI-2, NI-1, NI, NI+1
    
    # All faces including boundaries (NI+1 faces from 0 to NI)
    # Face 0: between ghost cell 0 and cell 1
    # Face NI: between cell NI and ghost cell NI+1
    
    # Let's compute for interior faces where we have full stencil
    # Faces 1 to NI-1 (NI-1 faces)
    
    # Pad the sensor to match face locations
    # nu_i has shape (NI, NJ) at cell centers (interior cells 1 to NI in original indexing)
    # For face f, we need max(nu[f], nu[f+1])
    # Actually nu_i was computed on cells, let me recompute properly
    
    # Simpler: compute dissipation on interior domain, zero at boundaries
    
    # Spectral radius at faces (need to compute or receive as input)
    # lambda_i shape should be (NI+1, NJ) for all I-faces
    
    # For now, let's work with a simpler approach:
    # Compute dissipation on a padded result array
    
    n_faces_i = NI + 1  # Number of I-faces
    D_i = np.zeros((n_faces_i, NJ, 4))
    
    # Interior faces: 1 to NI-1 (need 4-point stencil which requires cells 0 to NI+1)
    # All cells available with ghosts
    
    # States at faces 1 to NI-1 (indices in the interior cell array)
    # Face f (1-based) uses cells f-1, f, f+1, f+2 in ghost-included array
    # In 0-based face indexing for interior: face index 1 to NI-1
    
    for f in range(1, NI):  # This is a loop but only NI iterations, not NI*NJ
        # Actually, we should vectorize this. Let me restructure.
        pass
    
    # Vectorized version:
    # Interior faces: indices 1 to NI-1 (NI-1 faces)
    # For face[f], f in [1, NI-1]: 
    #   L-1 = cell[f-1], L = cell[f], R = cell[f+1], R+1 = cell[f+2]
    # In ghost array (0 to NI+1):
    #   f=1: cells 0,1,2,3
    #   f=NI-1: cells NI-2, NI-1, NI, NI+1
    
    # Slices for interior faces:
    f_start, f_end = 1, NI  # Face indices
    # Convert to array slices (face f maps to Q[f-1:f+3] for the stencil)
    
    # Cells for faces 1 to NI-1:
    Q_Lm1 = Q[0:NI-1, 1:-1, :]    # L-1: cells 0 to NI-2
    Q_L   = Q[1:NI, 1:-1, :]      # L:   cells 1 to NI-1
    Q_R   = Q[2:NI+1, 1:-1, :]    # R:   cells 2 to NI
    Q_Rp1 = Q[3:NI+2, 1:-1, :]    # R+1: cells 3 to NI+1
    
    # Spectral radius at these faces
    lambda_faces = lambda_i[1:NI, :]  # Interior faces only
    
    # Pressure sensor: max of left and right cells
    # nu_i has shape (NI, NJ) - sensor at interior cells
    # For face between cells i and i+1: max(nu[i-1], nu[i]) 
    # where nu is indexed from 0 (cell 1) to NI-1 (cell NI)
    # Face 1 (between cells 1,2): max(nu[0], nu[1])
    # Face NI-1 (between cells NI-1, NI): max(nu[NI-2], nu[NI-1])
    
    nu_L = nu_i[:-1, :]  # nu at left cells: indices 0 to NI-2
    nu_R = nu_i[1:, :]   # nu at right cells: indices 1 to NI-1
    nu_face = np.maximum(nu_L, nu_R)  # Shape: (NI-1, NJ)
    
    # JST coefficients
    # ε(2) = k2 * ν * λ
    eps2 = cfg.k2 * nu_face * lambda_faces
    
    # ε(4) = max(0, k4 - k2*ν) * λ
    # Note: we use (k4 - k2*nu) to reduce 4th order where 2nd order is active
    eps4 = np.maximum(0.0, cfg.k4 - cfg.k2 * nu_face) * lambda_faces
    
    # Dissipation terms
    # D2 = ε(2) * (Q_R - Q_L)
    D2 = eps2[..., np.newaxis] * (Q_R - Q_L)
    
    # D4 = ε(4) * (Q_{R+1} - 3*Q_R + 3*Q_L - Q_{L-1})
    # Note: sign convention - this removes oscillations
    D4 = eps4[..., np.newaxis] * (Q_Rp1 - 3*Q_R + 3*Q_L - Q_Lm1)
    
    # Total dissipation (added to flux from L to R)
    D_interior = D2 - D4
    
    # Store in output array
    D_i[1:NI, :, :] = D_interior
    
    # Boundary faces (0 and NI): use reduced stencil or zero
    # For face 0: between ghost cell 0 and cell 1
    # Use 2nd order only: D = ε(2) * (Q[1] - Q[0])
    lambda_0 = lambda_i[0, :]
    nu_0 = nu_i[0, :]  # Sensor at cell 1
    eps2_0 = cfg.k2 * nu_0 * lambda_0
    D_i[0, :, :] = eps2_0[..., np.newaxis] * (Q[1, 1:-1, :] - Q[0, 1:-1, :])
    
    # Face NI: between cell NI and ghost cell NI+1
    lambda_N = lambda_i[NI, :]
    nu_N = nu_i[-1, :]  # Sensor at cell NI
    eps2_N = cfg.k2 * nu_N * lambda_N
    D_i[NI, :, :] = eps2_N[..., np.newaxis] * (Q[NI+1, 1:-1, :] - Q[NI, 1:-1, :])
    
    return D_i


def compute_jst_dissipation_j(Q: np.ndarray, lambda_j: np.ndarray,
                               nu_j: np.ndarray, cfg: FluxConfig) -> np.ndarray:
    """
    Compute JST artificial dissipation for J-faces.
    
    Same algorithm as I-faces but in the j-direction.
    
    Parameters
    ----------
    Q : ndarray, shape (NI+2, NJ+2, 4)
        State vector with ghost cells.
    lambda_j : ndarray, shape (NI, NJ+1)
        Spectral radius at J-faces.
    nu_j : ndarray, shape (NI, NJ)
        Pressure sensor at interior cells (j-direction).
    cfg : FluxConfig
        Dissipation coefficients.
        
    Returns
    -------
    D_j : ndarray, shape (NI, NJ+1, 4)
        Dissipation flux at J-faces.
    """
    NI_ghost, NJ_ghost, _ = Q.shape
    NI = NI_ghost - 2
    NJ = NJ_ghost - 2
    
    n_faces_j = NJ + 1
    D_j = np.zeros((NI, n_faces_j, 4))
    
    # Interior faces: indices 1 to NJ-1
    # Face f uses cells f-1, f, f+1, f+2 in j-direction
    
    Q_Lm1 = Q[1:-1, 0:NJ-1, :]
    Q_L   = Q[1:-1, 1:NJ, :]
    Q_R   = Q[1:-1, 2:NJ+1, :]
    Q_Rp1 = Q[1:-1, 3:NJ+2, :]
    
    lambda_faces = lambda_j[:, 1:NJ]
    
    # Pressure sensor at faces
    nu_L = nu_j[:, :-1]
    nu_R = nu_j[:, 1:]
    nu_face = np.maximum(nu_L, nu_R)
    
    # JST coefficients
    eps2 = cfg.k2 * nu_face * lambda_faces
    eps4 = np.maximum(0.0, cfg.k4 - cfg.k2 * nu_face) * lambda_faces
    
    # Dissipation
    D2 = eps2[..., np.newaxis] * (Q_R - Q_L)
    D4 = eps4[..., np.newaxis] * (Q_Rp1 - 3*Q_R + 3*Q_L - Q_Lm1)
    
    D_interior = D2 - D4
    D_j[:, 1:NJ, :] = D_interior
    
    # Boundary faces
    lambda_0 = lambda_j[:, 0]
    nu_0 = nu_j[:, 0]
    eps2_0 = cfg.k2 * nu_0 * lambda_0
    D_j[:, 0, :] = eps2_0[..., np.newaxis] * (Q[1:-1, 1, :] - Q[1:-1, 0, :])
    
    lambda_N = lambda_j[:, NJ]
    nu_N = nu_j[:, -1]
    eps2_N = cfg.k2 * nu_N * lambda_N
    D_j[:, NJ, :] = eps2_N[..., np.newaxis] * (Q[1:-1, NJ+1, :] - Q[1:-1, NJ, :])
    
    return D_j


def compute_fluxes(Q: np.ndarray, metrics: GridMetrics, beta: float,
                   cfg: FluxConfig) -> np.ndarray:
    """
    Compute the flux residual using the JST scheme.
    
    The residual R for each cell is the sum of fluxes through all faces:
        R = -( F_{i+1/2} - F_{i-1/2} + F_{j+1/2} - F_{j-1/2} )
        
    where each flux consists of a central convective part and dissipation:
        F = F_conv - D
        
    Parameters
    ----------
    Q : ndarray, shape (NI+2, NJ+2, 4)
        State vector including ghost cells.
        Q[0, :] and Q[NI+1, :] are i-direction ghost cells.
        Q[:, 0] and Q[:, NJ+1] are j-direction ghost cells.
    metrics : GridMetrics
        Grid metrics containing face normals and cell volumes.
    beta : float
        Artificial compressibility parameter.
    cfg : FluxConfig
        Configuration with JST coefficients.
        
    Returns
    -------
    residual : ndarray, shape (NI, NJ, 4)
        Flux residual at interior cells (without ghost cells).
        
    Notes
    -----
    Sign convention: Positive residual means net flux INTO the cell.
    For steady state, we want R → 0.
    
    The JST scheme uses:
    - Central (averaged) convective flux
    - 2nd-order dissipation with pressure sensor (shock capturing)
    - 4th-order background dissipation (removes odd-even decoupling)
    """
    NI_ghost, NJ_ghost, n_vars = Q.shape
    NI = NI_ghost - 2
    NJ = NJ_ghost - 2
    
    # Extract normals (already scaled by face area)
    Si_x, Si_y = metrics.Si_x, metrics.Si_y  # I-face normals, shape (NI+1, NJ)
    Sj_x, Sj_y = metrics.Sj_x, metrics.Sj_y  # J-face normals, shape (NI, NJ+1)
    
    # =========================================================================
    # I-direction fluxes (faces between cells i and i+1)
    # =========================================================================
    
    # States at left and right of each I-face
    # Face f is between cell f and cell f+1 (in ghost-included indexing)
    # Interior cells are 1 to NI, so interior faces are 1 to NI-1
    # But we compute for all faces 0 to NI
    
    Q_L_i = Q[:-1, 1:-1, :]   # Left states: cells 0 to NI, shape (NI+1, NJ, 4)
    Q_R_i = Q[1:, 1:-1, :]    # Right states: cells 1 to NI+1, shape (NI+1, NJ, 4)
    
    # Average state at faces for flux evaluation
    Q_avg_i = 0.5 * (Q_L_i + Q_R_i)
    
    # Convective flux at I-faces
    F_conv_i = compute_convective_flux(Q_avg_i, Si_x, Si_y, beta)
    
    # Spectral radius at I-faces
    lambda_i = compute_spectral_radius(Q_avg_i, Si_x, Si_y, beta)
    
    # Pressure sensor for I-direction
    p = Q[:, 1:-1, 0]  # Pressure at all i-cells, interior j-cells
    d2p_i = np.abs(p[2:, :] - 2*p[1:-1, :] + p[:-2, :])
    sum_p_i = np.abs(p[2:, :]) + 2*np.abs(p[1:-1, :]) + np.abs(p[:-2, :]) + cfg.eps_p
    nu_i = d2p_i / sum_p_i  # Shape: (NI, NJ) - at interior cells
    
    # JST dissipation
    D_i = compute_jst_dissipation_i(Q, lambda_i, nu_i, cfg)
    
    # Total flux at I-faces (positive = from L to R)
    F_i = F_conv_i - D_i
    
    # =========================================================================
    # J-direction fluxes (faces between cells j and j+1)
    # =========================================================================
    
    Q_L_j = Q[1:-1, :-1, :]   # Left states: cells 0 to NJ
    Q_R_j = Q[1:-1, 1:, :]    # Right states: cells 1 to NJ+1
    
    Q_avg_j = 0.5 * (Q_L_j + Q_R_j)
    
    # Convective flux at J-faces
    F_conv_j = compute_convective_flux(Q_avg_j, Sj_x, Sj_y, beta)
    
    # Spectral radius at J-faces
    lambda_j = compute_spectral_radius(Q_avg_j, Sj_x, Sj_y, beta)
    
    # Pressure sensor for J-direction
    p = Q[1:-1, :, 0]  # Pressure at interior i-cells, all j-cells
    d2p_j = np.abs(p[:, 2:] - 2*p[:, 1:-1] + p[:, :-2])
    sum_p_j = np.abs(p[:, 2:]) + 2*np.abs(p[:, 1:-1]) + np.abs(p[:, :-2]) + cfg.eps_p
    nu_j = d2p_j / sum_p_j  # Shape: (NI, NJ) - at interior cells
    
    # JST dissipation
    D_j = compute_jst_dissipation_j(Q, lambda_j, nu_j, cfg)
    
    # Total flux at J-faces
    F_j = F_conv_j - D_j
    
    # =========================================================================
    # Compute residual (flux balance for each cell)
    # =========================================================================
    
    # Residual = -(flux_out - flux_in) = flux_in - flux_out
    # For cell (i, j), faces are:
    #   I-face i-1/2 (flux in if positive normal points into cell)
    #   I-face i+1/2 (flux out if positive normal points out of cell)
    #   J-face j-1/2
    #   J-face j+1/2
    
    # With our convention (normal points from L to R = increasing index):
    # Flux INTO cell from i-1/2 face = +F_i[i-1]
    # Flux OUT of cell through i+1/2 face = +F_i[i]
    # Net = F_i[i-1] - F_i[i] for I-direction
    
    # For interior cells (indices 1 to NI in ghost array, or 0 to NI-1 in output):
    # Face indices: cell i (in ghost array) has faces i-1 and i in face array
    # Actually, let's be careful:
    # - Cell with ghost index i has left I-face at face index i-1 and right at i
    # - But we want output for interior cells only
    
    # F_i has shape (NI+1, NJ, 4) - faces 0 to NI
    # Interior cells are at ghost indices 1 to NI
    # Cell at ghost index g has:
    #   left I-face at face index g-1
    #   right I-face at face index g
    # For g = 1: faces 0 and 1
    # For g = NI: faces NI-1 and NI
    
    # Residual from I-faces:
    # R_i = F_i[0:NI] - F_i[1:NI+1]  # flux in from left - flux out through right
    R_i = F_i[:-1, :, :] - F_i[1:, :, :]  # Shape: (NI, NJ, 4)
    
    # F_j has shape (NI, NJ+1, 4) - faces 0 to NJ
    # Interior cells in j are at ghost indices 1 to NJ
    # Cell at ghost index g has faces g-1 and g
    R_j = F_j[:, :-1, :] - F_j[:, 1:, :]  # Shape: (NI, NJ, 4)
    
    # Total residual
    residual = R_i + R_j
    
    return residual


def compute_time_step(Q: np.ndarray, metrics: GridMetrics, beta: float,
                      cfl: float = 0.8) -> np.ndarray:
    """
    Compute the local time step based on CFL condition.
    
    For artificial compressibility:
        Δt = CFL * Volume / (λ_i + λ_j)
        
    where λ is the spectral radius in each direction.
    
    Parameters
    ----------
    Q : ndarray, shape (NI+2, NJ+2, 4)
        State vector with ghost cells.
    metrics : GridMetrics
        Grid metrics.
    beta : float
        Artificial compressibility parameter.
    cfl : float
        CFL number (default 0.8).
        
    Returns
    -------
    dt : ndarray, shape (NI, NJ)
        Local time step for each cell.
    """
    NI = Q.shape[0] - 2
    NJ = Q.shape[1] - 2
    
    # Interior cell states
    Q_int = Q[1:-1, 1:-1, :]
    u = Q_int[..., 1]
    v = Q_int[..., 2]
    
    # Artificial speed of sound
    c_art = np.sqrt(u**2 + v**2 + beta)
    
    # Spectral radius in each direction (approximate using cell-averaged metrics)
    # I-direction: average of left and right face areas
    Si_mag = np.sqrt(metrics.Si_x**2 + metrics.Si_y**2)
    Si_avg = 0.5 * (Si_mag[:-1, :] + Si_mag[1:, :])  # Shape: (NI, NJ)
    
    Sj_mag = np.sqrt(metrics.Sj_x**2 + metrics.Sj_y**2)
    Sj_avg = 0.5 * (Sj_mag[:, :-1] + Sj_mag[:, 1:])  # Shape: (NI, NJ)
    
    # Spectral radius (simplified - assumes roughly aligned grid)
    lambda_i = (np.abs(u) + c_art) * Si_avg / metrics.volume
    lambda_j = (np.abs(v) + c_art) * Sj_avg / metrics.volume
    
    # Time step
    dt = cfl * metrics.volume / (lambda_i + lambda_j + 1e-12)
    
    return dt

