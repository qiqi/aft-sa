"""
Grid Coarsening for Multigrid Methods.

This module provides Numba-optimized kernels for coarsening structured grid
metrics from fine to coarse levels. The coarsening strategy combines 2x2 fine
cells into 1 coarse cell while preserving the Geometric Conservation Law (GCL).

Coarsening Rules:
- Volume: Sum of 4 fine cell volumes
- Face normals: Sum of 2 fine face normals (external faces only)
- Wall distance: Minimum of 4 fine cell values
- Cell centers: Volume-weighted average of 4 fine centers

Design: GPU-ready with flat arrays and @njit kernels.
"""

import numpy as np
from numba import njit, prange
from typing import Tuple

from .metrics import FVMMetrics


@njit(cache=True, parallel=True)
def coarsen_volumes(vol_f: np.ndarray) -> np.ndarray:
    """
    Coarsen cell volumes by summing 2x2 fine cells.
    
    Parameters
    ----------
    vol_f : ndarray, shape (NI_f, NJ_f)
        Fine grid cell volumes.
        
    Returns
    -------
    vol_c : ndarray, shape (NI_f//2, NJ_f//2)
        Coarse grid cell volumes.
    """
    NI_f, NJ_f = vol_f.shape
    NI_c = NI_f // 2
    NJ_c = NJ_f // 2
    
    vol_c = np.zeros((NI_c, NJ_c), dtype=vol_f.dtype)
    
    for i_c in prange(NI_c):
        for j_c in range(NJ_c):
            i_f = 2 * i_c
            j_f = 2 * j_c
            # Sum 4 fine volumes
            vol_c[i_c, j_c] = (vol_f[i_f, j_f] + vol_f[i_f+1, j_f] +
                               vol_f[i_f, j_f+1] + vol_f[i_f+1, j_f+1])
    
    return vol_c


@njit(cache=True, parallel=True)
def coarsen_cell_centers(xc_f: np.ndarray, yc_f: np.ndarray, 
                          vol_f: np.ndarray, vol_c: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coarsen cell centers using volume-weighted average.
    
    Parameters
    ----------
    xc_f, yc_f : ndarray, shape (NI_f, NJ_f)
        Fine grid cell centers.
    vol_f : ndarray, shape (NI_f, NJ_f)
        Fine grid cell volumes.
    vol_c : ndarray, shape (NI_c, NJ_c)
        Coarse grid cell volumes.
        
    Returns
    -------
    xc_c, yc_c : ndarray, shape (NI_c, NJ_c)
        Coarse grid cell centers.
    """
    NI_c, NJ_c = vol_c.shape
    
    xc_c = np.zeros((NI_c, NJ_c), dtype=xc_f.dtype)
    yc_c = np.zeros((NI_c, NJ_c), dtype=yc_f.dtype)
    
    for i_c in prange(NI_c):
        for j_c in range(NJ_c):
            i_f = 2 * i_c
            j_f = 2 * j_c
            
            # Volume-weighted average of 4 fine centers
            v00 = vol_f[i_f, j_f]
            v10 = vol_f[i_f+1, j_f]
            v01 = vol_f[i_f, j_f+1]
            v11 = vol_f[i_f+1, j_f+1]
            
            total_vol = vol_c[i_c, j_c]
            
            xc_c[i_c, j_c] = (xc_f[i_f, j_f] * v00 + xc_f[i_f+1, j_f] * v10 +
                              xc_f[i_f, j_f+1] * v01 + xc_f[i_f+1, j_f+1] * v11) / total_vol
            yc_c[i_c, j_c] = (yc_f[i_f, j_f] * v00 + yc_f[i_f+1, j_f] * v10 +
                              yc_f[i_f, j_f+1] * v01 + yc_f[i_f+1, j_f+1] * v11) / total_vol
    
    return xc_c, yc_c


@njit(cache=True, parallel=True)
def coarsen_i_face_normals(Si_x_f: np.ndarray, Si_y_f: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coarsen I-face normals by summing 2 fine faces per coarse face.
    
    For I-faces (constant i), each coarse face spans 2 fine j-cells.
    The coarse face normal is the sum of the 2 fine face normals.
    
    Fine I-face array: shape (NI_f+1, NJ_f)
    Coarse I-face array: shape (NI_c+1, NJ_c)
    
    Coarse face (i_c, j_c) corresponds to fine faces at i_f=2*i_c, j_f=2*j_c and 2*j_c+1.
    
    Parameters
    ----------
    Si_x_f, Si_y_f : ndarray, shape (NI_f+1, NJ_f)
        Fine grid I-face normal components.
        
    Returns
    -------
    Si_x_c, Si_y_c : ndarray, shape (NI_c+1, NJ_c)
        Coarse grid I-face normal components.
    """
    NI_f_plus1, NJ_f = Si_x_f.shape
    NI_c_plus1 = NI_f_plus1 // 2 + NI_f_plus1 % 2  # Ceiling division
    NJ_c = NJ_f // 2
    
    # Actually, for NI_f cells, there are NI_f+1 I-faces
    # For NI_c = NI_f//2 cells, there are NI_c+1 I-faces
    # Coarse face at i_c corresponds to fine face at i_f = 2*i_c
    NI_c_plus1 = (NI_f_plus1 - 1) // 2 + 1  # NI_f//2 + 1
    
    Si_x_c = np.zeros((NI_c_plus1, NJ_c), dtype=Si_x_f.dtype)
    Si_y_c = np.zeros((NI_c_plus1, NJ_c), dtype=Si_y_f.dtype)
    
    for i_c in prange(NI_c_plus1):
        for j_c in range(NJ_c):
            i_f = 2 * i_c
            j_f = 2 * j_c
            
            # Sum the 2 fine faces in j-direction that form this coarse face
            # Clamp to valid indices
            if i_f < NI_f_plus1:
                Si_x_c[i_c, j_c] = Si_x_f[i_f, j_f] + Si_x_f[i_f, j_f+1]
                Si_y_c[i_c, j_c] = Si_y_f[i_f, j_f] + Si_y_f[i_f, j_f+1]
    
    return Si_x_c, Si_y_c


@njit(cache=True, parallel=True)
def coarsen_j_face_normals(Sj_x_f: np.ndarray, Sj_y_f: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coarsen J-face normals by summing 2 fine faces per coarse face.
    
    For J-faces (constant j), each coarse face spans 2 fine i-cells.
    The coarse face normal is the sum of the 2 fine face normals.
    
    Fine J-face array: shape (NI_f, NJ_f+1)
    Coarse J-face array: shape (NI_c, NJ_c+1)
    
    Coarse face (i_c, j_c) corresponds to fine faces at i_f=2*i_c and 2*i_c+1, j_f=2*j_c.
    
    Parameters
    ----------
    Sj_x_f, Sj_y_f : ndarray, shape (NI_f, NJ_f+1)
        Fine grid J-face normal components.
        
    Returns
    -------
    Sj_x_c, Sj_y_c : ndarray, shape (NI_c, NJ_c+1)
        Coarse grid J-face normal components.
    """
    NI_f, NJ_f_plus1 = Sj_x_f.shape
    NI_c = NI_f // 2
    NJ_c_plus1 = (NJ_f_plus1 - 1) // 2 + 1  # NJ_f//2 + 1
    
    Sj_x_c = np.zeros((NI_c, NJ_c_plus1), dtype=Sj_x_f.dtype)
    Sj_y_c = np.zeros((NI_c, NJ_c_plus1), dtype=Sj_y_f.dtype)
    
    for i_c in prange(NI_c):
        for j_c in range(NJ_c_plus1):
            i_f = 2 * i_c
            j_f = 2 * j_c
            
            # Sum the 2 fine faces in i-direction that form this coarse face
            if j_f < NJ_f_plus1:
                Sj_x_c[i_c, j_c] = Sj_x_f[i_f, j_f] + Sj_x_f[i_f+1, j_f]
                Sj_y_c[i_c, j_c] = Sj_y_f[i_f, j_f] + Sj_y_f[i_f+1, j_f]
    
    return Sj_x_c, Sj_y_c


@njit(cache=True, parallel=True)
def coarsen_wall_distance(wall_dist_f: np.ndarray) -> np.ndarray:
    """
    Coarsen wall distance by taking minimum of 2x2 fine cells.
    
    This ensures the coarse cell has a conservative (smaller) wall distance,
    which is important for turbulence model damping functions.
    
    Parameters
    ----------
    wall_dist_f : ndarray, shape (NI_f, NJ_f)
        Fine grid wall distances.
        
    Returns
    -------
    wall_dist_c : ndarray, shape (NI_c, NJ_c)
        Coarse grid wall distances.
    """
    NI_f, NJ_f = wall_dist_f.shape
    NI_c = NI_f // 2
    NJ_c = NJ_f // 2
    
    wall_dist_c = np.zeros((NI_c, NJ_c), dtype=wall_dist_f.dtype)
    
    for i_c in prange(NI_c):
        for j_c in range(NJ_c):
            i_f = 2 * i_c
            j_f = 2 * j_c
            
            # Take minimum of 4 fine values
            d00 = wall_dist_f[i_f, j_f]
            d10 = wall_dist_f[i_f+1, j_f]
            d01 = wall_dist_f[i_f, j_f+1]
            d11 = wall_dist_f[i_f+1, j_f+1]
            
            wall_dist_c[i_c, j_c] = min(min(d00, d10), min(d01, d11))
    
    return wall_dist_c


@njit(cache=True)
def validate_gcl_coarse(vol_c: np.ndarray, 
                         Si_x_c: np.ndarray, Si_y_c: np.ndarray,
                         Sj_x_c: np.ndarray, Sj_y_c: np.ndarray
                         ) -> Tuple[float, float]:
    """
    Validate GCL on coarse grid: sum of face normals should be zero.
    
    Parameters
    ----------
    vol_c : ndarray, shape (NI_c, NJ_c)
        Coarse cell volumes.
    Si_x_c, Si_y_c : ndarray, shape (NI_c+1, NJ_c)
        Coarse I-face normals.
    Sj_x_c, Sj_y_c : ndarray, shape (NI_c, NJ_c+1)
        Coarse J-face normals.
        
    Returns
    -------
    max_res_x, max_res_y : float
        Maximum absolute GCL residual in x and y.
    """
    NI_c, NJ_c = vol_c.shape
    
    max_res_x = 0.0
    max_res_y = 0.0
    
    for i in range(NI_c):
        for j in range(NJ_c):
            # Sum of outward-pointing face normals
            res_x = (Si_x_c[i+1, j] - Si_x_c[i, j] + 
                     Sj_x_c[i, j+1] - Sj_x_c[i, j])
            res_y = (Si_y_c[i+1, j] - Si_y_c[i, j] + 
                     Sj_y_c[i, j+1] - Sj_y_c[i, j])
            
            max_res_x = max(max_res_x, abs(res_x))
            max_res_y = max(max_res_y, abs(res_y))
    
    return max_res_x, max_res_y


class Coarsener:
    """
    Grid coarsening utility for multigrid methods.
    
    Combines 2x2 fine cells into 1 coarse cell while preserving
    the Geometric Conservation Law (GCL).
    
    Example
    -------
    >>> coarsener = Coarsener()
    >>> coarse_metrics = coarsener.coarsen(fine_metrics)
    >>> # Verify GCL
    >>> max_res_x, max_res_y = coarsener.validate_gcl(coarse_metrics)
    >>> assert max_res_x < 1e-10 and max_res_y < 1e-10
    """
    
    @staticmethod
    def coarsen(fine_metrics: FVMMetrics) -> FVMMetrics:
        """
        Coarsen grid metrics from fine to coarse level.
        
        Parameters
        ----------
        fine_metrics : FVMMetrics
            Fine grid metrics.
            
        Returns
        -------
        coarse_metrics : FVMMetrics
            Coarse grid metrics with dimensions halved.
        """
        # Coarsen volumes first (needed for weighted averaging)
        vol_c = coarsen_volumes(fine_metrics.volume)
        
        # Coarsen cell centers with volume weighting
        xc_c, yc_c = coarsen_cell_centers(
            fine_metrics.xc, fine_metrics.yc,
            fine_metrics.volume, vol_c
        )
        
        # Coarsen face normals (sum to preserve GCL)
        Si_x_c, Si_y_c = coarsen_i_face_normals(
            fine_metrics.Si_x, fine_metrics.Si_y
        )
        Sj_x_c, Sj_y_c = coarsen_j_face_normals(
            fine_metrics.Sj_x, fine_metrics.Sj_y
        )
        
        # Coarsen wall distance (minimum for conservative SA damping)
        wall_dist_c = coarsen_wall_distance(fine_metrics.wall_distance)
        
        return FVMMetrics(
            volume=vol_c,
            xc=xc_c, yc=yc_c,
            Si_x=Si_x_c, Si_y=Si_y_c,
            Sj_x=Sj_x_c, Sj_y=Sj_y_c,
            wall_distance=wall_dist_c
        )
    
    @staticmethod
    def validate_gcl(metrics: FVMMetrics) -> Tuple[float, float]:
        """
        Validate Geometric Conservation Law on given metrics.
        
        Parameters
        ----------
        metrics : FVMMetrics
            Grid metrics to validate.
            
        Returns
        -------
        max_res_x, max_res_y : float
            Maximum absolute GCL residual in x and y directions.
        """
        return validate_gcl_coarse(
            metrics.volume,
            metrics.Si_x, metrics.Si_y,
            metrics.Sj_x, metrics.Sj_y
        )
    
    @staticmethod
    def can_coarsen(NI: int, NJ: int, min_size: int = 4) -> bool:
        """
        Check if grid can be coarsened further.
        
        Parameters
        ----------
        NI, NJ : int
            Current grid dimensions.
        min_size : int
            Minimum cells in each direction after coarsening.
            
        Returns
        -------
        bool
            True if grid can be coarsened, False otherwise.
        """
        return NI >= 2 * min_size and NJ >= 2 * min_size
    
    @staticmethod
    def max_levels(NI: int, NJ: int, min_size: int = 4) -> int:
        """
        Compute maximum number of multigrid levels.
        
        Parameters
        ----------
        NI, NJ : int
            Fine grid dimensions.
        min_size : int
            Minimum cells in each direction on coarsest level.
            
        Returns
        -------
        int
            Maximum number of levels (including finest).
        """
        levels = 1
        ni, nj = NI, NJ
        
        while ni >= 2 * min_size and nj >= 2 * min_size:
            ni = ni // 2
            nj = nj // 2
            levels += 1
        
        return levels

