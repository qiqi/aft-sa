"""
Gradient Reconstruction for Finite Volume Method.

This module implements gradient computation using the Green-Gauss theorem,
optimized with Numba JIT compilation.

The Green-Gauss theorem states:
    ∇φ ≈ (1/V) ∮ φ n̂ dA = (1/V) Σ φ_face · S_face

where S_face is the area-scaled normal vector (nx*A, ny*A).

Boundary Handling:
    Ghost cells are assumed to be set correctly by boundary conditions.
    For example, no-slip walls have u_ghost = -u_interior.
    The simple face average 0.5*(L+R) then automatically gives the correct
    boundary gradient (e.g., du/dn = (u_interior - u_ghost)/(2*dn) = u_interior/dn).
"""

import numpy as np
from numba import njit
from typing import NamedTuple


class GradientMetrics(NamedTuple):
    """
    Grid metrics required for gradient computation.
    
    All arrays are for the interior grid (no ghost cells in metrics).
    
    Attributes
    ----------
    Si_x : ndarray, shape (NI+1, NJ)
        x-component of I-face normal (scaled by area).
    Si_y : ndarray, shape (NI+1, NJ)
        y-component of I-face normal (scaled by area).
    Sj_x : ndarray, shape (NI, NJ+1)
        x-component of J-face normal (scaled by area).
    Sj_y : ndarray, shape (NI, NJ+1)
        y-component of J-face normal (scaled by area).
    volume : ndarray, shape (NI, NJ)
        Cell volumes.
    """
    Si_x: np.ndarray
    Si_y: np.ndarray
    Sj_x: np.ndarray
    Sj_y: np.ndarray
    volume: np.ndarray


@njit(cache=True)
def _gradient_kernel(Q: np.ndarray,
                     Si_x: np.ndarray, Si_y: np.ndarray,
                     Sj_x: np.ndarray, Sj_y: np.ndarray,
                     volume: np.ndarray,
                     grad: np.ndarray) -> None:
    """
    Numba-optimized kernel for Green-Gauss gradient computation.
    
    Computes gradients of all 4 state variables using the Green-Gauss theorem:
        ∇φ = (1/V) Σ_faces (φ_face · S_face)
    
    Parameters
    ----------
    Q : ndarray, shape (NI+2, NJ+2, 4)
        State vector with ghost cells.
        Q[i, j, :] = [p, u, v, nu_tilde]
    Si_x, Si_y : ndarray, shape (NI+1, NJ)
        I-face normals (area-scaled).
    Sj_x, Sj_y : ndarray, shape (NI, NJ+1)
        J-face normals (area-scaled).
    volume : ndarray, shape (NI, NJ)
        Cell volumes.
    grad : ndarray, shape (NI, NJ, 4, 2)
        Output gradient array.
        grad[i, j, k, 0] = dQ[k]/dx at cell (i,j)
        grad[i, j, k, 1] = dQ[k]/dy at cell (i,j)
    """
    NI = volume.shape[0]
    NJ = volume.shape[1]
    
    # Initialize gradients to zero
    for i in range(NI):
        for j in range(NJ):
            for k in range(4):
                grad[i, j, k, 0] = 0.0
                grad[i, j, k, 1] = 0.0
    
    # =========================================================================
    # I-face contributions (faces between i and i+1)
    # =========================================================================
    # Face (i, j) is between cells (i-1, j) and (i, j) in Q indexing
    # In interior indexing: between cells (i-1, j) and (i, j)
    # Q uses ghost cells: Q[i_cell+1, j_cell+1, :]
    
    for i in range(NI + 1):
        for j in range(NJ):
            # Face normal (area-scaled, pointing in +i direction)
            nx = Si_x[i, j]
            ny = Si_y[i, j]
            
            # Left and right cells in Q indexing (with ghost cell offset)
            # Cell (i-1) in interior = Q[i, j+1] (ghost offset +1 in j)
            # Cell (i) in interior = Q[i+1, j+1]
            i_L = i      # Left cell in Q
            i_R = i + 1  # Right cell in Q
            j_Q = j + 1  # j index in Q (ghost offset)
            
            # Loop over all state variables
            for k in range(4):
                phi_L = Q[i_L, j_Q, k]
                phi_R = Q[i_R, j_Q, k]
                
                # Face-averaged value
                phi_face = 0.5 * (phi_L + phi_R)
                
                # Contribution to gradient: φ_face * n_face
                contrib_x = phi_face * nx
                contrib_y = phi_face * ny
                
                # Add to left cell (subtract because normal points outward from left cell)
                if i > 0:
                    grad[i - 1, j, k, 0] += contrib_x
                    grad[i - 1, j, k, 1] += contrib_y
                
                # Add to right cell (add because normal points inward to right cell)
                if i < NI:
                    grad[i, j, k, 0] -= contrib_x
                    grad[i, j, k, 1] -= contrib_y
    
    # =========================================================================
    # J-face contributions (faces between j and j+1)
    # =========================================================================
    for i in range(NI):
        for j in range(NJ + 1):
            # Face normal (area-scaled, pointing in +j direction)
            nx = Sj_x[i, j]
            ny = Sj_y[i, j]
            
            # Left and right cells in Q indexing
            i_Q = i + 1  # i index in Q (ghost offset)
            j_L = j      # Left cell (lower j) in Q
            j_R = j + 1  # Right cell (upper j) in Q
            
            # Loop over all state variables
            for k in range(4):
                phi_L = Q[i_Q, j_L, k]
                phi_R = Q[i_Q, j_R, k]
                
                # Face-averaged value
                phi_face = 0.5 * (phi_L + phi_R)
                
                # Contribution to gradient
                contrib_x = phi_face * nx
                contrib_y = phi_face * ny
                
                # Add to lower cell (j-1 in interior indexing)
                if j > 0:
                    grad[i, j - 1, k, 0] += contrib_x
                    grad[i, j - 1, k, 1] += contrib_y
                
                # Add to upper cell (j in interior indexing)
                if j < NJ:
                    grad[i, j, k, 0] -= contrib_x
                    grad[i, j, k, 1] -= contrib_y
    
    # =========================================================================
    # Divide by cell volume to get gradient
    # =========================================================================
    for i in range(NI):
        for j in range(NJ):
            inv_vol = 1.0 / volume[i, j]
            for k in range(4):
                grad[i, j, k, 0] *= inv_vol
                grad[i, j, k, 1] *= inv_vol


def compute_gradients(Q: np.ndarray, metrics: GradientMetrics) -> np.ndarray:
    """
    Compute gradients of state variables using Green-Gauss theorem.
    
    This function computes cell-centered gradients for all state variables
    using the Green-Gauss (divergence theorem) approach:
    
        ∇φ = (1/V) ∮ φ n̂ dA ≈ (1/V) Σ_faces (φ_face · S_face)
    
    where φ_face = 0.5 * (φ_L + φ_R) is the face-averaged value.
    
    Parameters
    ----------
    Q : ndarray, shape (NI+2, NJ+2, 4)
        State vector with ghost cells.
        Q[:, :, 0] = pressure
        Q[:, :, 1] = u-velocity
        Q[:, :, 2] = v-velocity
        Q[:, :, 3] = nu_tilde (SA variable)
    metrics : GradientMetrics
        Grid metrics containing face normals and cell volumes.
        
    Returns
    -------
    grad : ndarray, shape (NI, NJ, 4, 2)
        Gradients at cell centers.
        grad[i, j, k, 0] = ∂Q[k]/∂x at cell (i, j)
        grad[i, j, k, 1] = ∂Q[k]/∂y at cell (i, j)
        
    Notes
    -----
    Boundary handling relies on properly set ghost cells. For example:
    - No-slip wall: u_ghost = -u_interior → correct wall gradient
    - Freestream: u_ghost = u_inf → zero gradient at boundary
    
    Example
    -------
    >>> from src.numerics.gradients import compute_gradients, GradientMetrics
    >>> 
    >>> # Create metrics
    >>> metrics = GradientMetrics(
    ...     Si_x=Si_x, Si_y=Si_y,
    ...     Sj_x=Sj_x, Sj_y=Sj_y,
    ...     volume=volume
    ... )
    >>> 
    >>> # Compute gradients
    >>> grad = compute_gradients(Q, metrics)
    >>> 
    >>> # Extract velocity gradients for viscous terms
    >>> dudx = grad[:, :, 1, 0]  # ∂u/∂x
    >>> dudy = grad[:, :, 1, 1]  # ∂u/∂y
    >>> dvdx = grad[:, :, 2, 0]  # ∂v/∂x
    >>> dvdy = grad[:, :, 2, 1]  # ∂v/∂y
    """
    # Get dimensions from volume (interior cells only)
    NI, NJ = metrics.volume.shape
    
    # Allocate output array
    grad = np.zeros((NI, NJ, 4, 2), dtype=np.float64)
    
    # Call Numba kernel
    _gradient_kernel(
        Q.astype(np.float64),
        metrics.Si_x.astype(np.float64),
        metrics.Si_y.astype(np.float64),
        metrics.Sj_x.astype(np.float64),
        metrics.Sj_y.astype(np.float64),
        metrics.volume.astype(np.float64),
        grad
    )
    
    return grad


@njit(cache=True)
def _compute_vorticity_kernel(dudx: np.ndarray, dudy: np.ndarray,
                               dvdx: np.ndarray, dvdy: np.ndarray,
                               omega: np.ndarray) -> None:
    """Compute vorticity magnitude from velocity gradients."""
    NI, NJ = dudx.shape
    for i in range(NI):
        for j in range(NJ):
            # Vorticity = ∂v/∂x - ∂u/∂y (2D)
            omega[i, j] = abs(dvdx[i, j] - dudy[i, j])


def compute_vorticity(grad: np.ndarray) -> np.ndarray:
    """
    Compute vorticity magnitude from gradient array.
    
    For 2D flow, vorticity is:
        ω = |∂v/∂x - ∂u/∂y|
    
    Parameters
    ----------
    grad : ndarray, shape (NI, NJ, 4, 2)
        Gradient array from compute_gradients().
        
    Returns
    -------
    omega : ndarray, shape (NI, NJ)
        Vorticity magnitude at cell centers.
    """
    NI, NJ = grad.shape[:2]
    
    # Extract velocity gradients
    dudy = grad[:, :, 1, 1]  # ∂u/∂y
    dvdx = grad[:, :, 2, 0]  # ∂v/∂x
    
    # Compute vorticity
    omega = np.zeros((NI, NJ), dtype=np.float64)
    _compute_vorticity_kernel(
        grad[:, :, 1, 0].astype(np.float64),  # dudx (unused but needed for signature)
        dudy.astype(np.float64),
        dvdx.astype(np.float64),
        grad[:, :, 2, 1].astype(np.float64),  # dvdy (unused but needed for signature)
        omega
    )
    
    return omega


def compute_strain_rate(grad: np.ndarray) -> np.ndarray:
    """
    Compute strain rate magnitude from gradient array.
    
    The strain rate tensor is:
        S_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
    
    The magnitude is:
        |S| = sqrt(2 * S_ij * S_ij)
    
    For 2D:
        |S| = sqrt(2*(S_xx² + S_yy² + 2*S_xy²))
            = sqrt(2*((∂u/∂x)² + (∂v/∂y)² + 0.5*(∂u/∂y + ∂v/∂x)²))
    
    Parameters
    ----------
    grad : ndarray, shape (NI, NJ, 4, 2)
        Gradient array from compute_gradients().
        
    Returns
    -------
    S_mag : ndarray, shape (NI, NJ)
        Strain rate magnitude at cell centers.
    """
    # Extract velocity gradients
    dudx = grad[:, :, 1, 0]
    dudy = grad[:, :, 1, 1]
    dvdx = grad[:, :, 2, 0]
    dvdy = grad[:, :, 2, 1]
    
    # Strain rate tensor components
    S_xx = dudx
    S_yy = dvdy
    S_xy = 0.5 * (dudy + dvdx)
    
    # Magnitude: sqrt(2 * S_ij * S_ij)
    S_mag = np.sqrt(2.0 * (S_xx**2 + S_yy**2 + 2.0 * S_xy**2))
    
    return S_mag


