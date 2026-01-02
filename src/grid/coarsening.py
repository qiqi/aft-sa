"""
Grid coarsening for multigrid methods.

Coarsening rules:
- Volume: sum of 4 fine cells
- Face normals: sum of 2 fine faces (external faces only)
- Wall distance: minimum of 4 fine cells
- Cell centers: volume-weighted average
"""

import numpy as np
from numba import njit, prange
from typing import Tuple

from .metrics import FVMMetrics


@njit(cache=True, parallel=True)
def coarsen_volumes(vol_f: np.ndarray) -> np.ndarray:
    """Coarsen volumes by summing 2x2 fine cells."""
    NI_f, NJ_f = vol_f.shape
    NI_c = NI_f // 2
    NJ_c = NJ_f // 2
    
    vol_c = np.zeros((NI_c, NJ_c), dtype=vol_f.dtype)
    
    for i_c in prange(NI_c):
        for j_c in range(NJ_c):
            i_f = 2 * i_c
            j_f = 2 * j_c
            vol_c[i_c, j_c] = (vol_f[i_f, j_f] + vol_f[i_f+1, j_f] +
                               vol_f[i_f, j_f+1] + vol_f[i_f+1, j_f+1])
    
    return vol_c


@njit(cache=True, parallel=True)
def coarsen_cell_centers(xc_f: np.ndarray, yc_f: np.ndarray, 
                          vol_f: np.ndarray, vol_c: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Coarsen cell centers using volume-weighted average."""
    NI_c, NJ_c = vol_c.shape
    
    xc_c = np.zeros((NI_c, NJ_c), dtype=xc_f.dtype)
    yc_c = np.zeros((NI_c, NJ_c), dtype=yc_f.dtype)
    
    for i_c in prange(NI_c):
        for j_c in range(NJ_c):
            i_f = 2 * i_c
            j_f = 2 * j_c
            
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
    """Coarsen I-face normals by summing 2 fine faces per coarse face."""
    NI_f_plus1, NJ_f = Si_x_f.shape
    NI_c_plus1 = (NI_f_plus1 - 1) // 2 + 1
    NJ_c = NJ_f // 2
    
    Si_x_c = np.zeros((NI_c_plus1, NJ_c), dtype=Si_x_f.dtype)
    Si_y_c = np.zeros((NI_c_plus1, NJ_c), dtype=Si_y_f.dtype)
    
    for i_c in prange(NI_c_plus1):
        for j_c in range(NJ_c):
            i_f = 2 * i_c
            j_f = 2 * j_c
            
            if i_f < NI_f_plus1:
                Si_x_c[i_c, j_c] = Si_x_f[i_f, j_f] + Si_x_f[i_f, j_f+1]
                Si_y_c[i_c, j_c] = Si_y_f[i_f, j_f] + Si_y_f[i_f, j_f+1]
    
    return Si_x_c, Si_y_c


@njit(cache=True, parallel=True)
def coarsen_j_face_normals(Sj_x_f: np.ndarray, Sj_y_f: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Coarsen J-face normals by summing 2 fine faces per coarse face."""
    NI_f, NJ_f_plus1 = Sj_x_f.shape
    NI_c = NI_f // 2
    NJ_c_plus1 = (NJ_f_plus1 - 1) // 2 + 1
    
    Sj_x_c = np.zeros((NI_c, NJ_c_plus1), dtype=Sj_x_f.dtype)
    Sj_y_c = np.zeros((NI_c, NJ_c_plus1), dtype=Sj_y_f.dtype)
    
    for i_c in prange(NI_c):
        for j_c in range(NJ_c_plus1):
            i_f = 2 * i_c
            j_f = 2 * j_c
            
            if j_f < NJ_f_plus1:
                Sj_x_c[i_c, j_c] = Sj_x_f[i_f, j_f] + Sj_x_f[i_f+1, j_f]
                Sj_y_c[i_c, j_c] = Sj_y_f[i_f, j_f] + Sj_y_f[i_f+1, j_f]
    
    return Sj_x_c, Sj_y_c


@njit(cache=True, parallel=True)
def coarsen_wall_distance(wall_dist_f: np.ndarray) -> np.ndarray:
    """Coarsen wall distance by taking minimum of 2x2 fine cells."""
    NI_f, NJ_f = wall_dist_f.shape
    NI_c = NI_f // 2
    NJ_c = NJ_f // 2
    
    wall_dist_c = np.zeros((NI_c, NJ_c), dtype=wall_dist_f.dtype)
    
    for i_c in prange(NI_c):
        for j_c in range(NJ_c):
            i_f = 2 * i_c
            j_f = 2 * j_c
            
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
    """Validate GCL: sum of face normals should be zero."""
    NI_c, NJ_c = vol_c.shape
    
    max_res_x = 0.0
    max_res_y = 0.0
    
    for i in range(NI_c):
        for j in range(NJ_c):
            res_x = (Si_x_c[i+1, j] - Si_x_c[i, j] + 
                     Sj_x_c[i, j+1] - Sj_x_c[i, j])
            res_y = (Si_y_c[i+1, j] - Si_y_c[i, j] + 
                     Sj_y_c[i, j+1] - Sj_y_c[i, j])
            
            max_res_x = max(max_res_x, abs(res_x))
            max_res_y = max(max_res_y, abs(res_y))
    
    return max_res_x, max_res_y


class Coarsener:
    """Grid coarsening utility for multigrid methods."""
    
    @staticmethod
    def coarsen(fine_metrics: FVMMetrics) -> FVMMetrics:
        """Coarsen grid metrics from fine to coarse level."""
        vol_c = coarsen_volumes(fine_metrics.volume)
        
        xc_c, yc_c = coarsen_cell_centers(
            fine_metrics.xc, fine_metrics.yc,
            fine_metrics.volume, vol_c
        )
        
        Si_x_c, Si_y_c = coarsen_i_face_normals(
            fine_metrics.Si_x, fine_metrics.Si_y
        )
        Sj_x_c, Sj_y_c = coarsen_j_face_normals(
            fine_metrics.Sj_x, fine_metrics.Sj_y
        )
        
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
        """Validate Geometric Conservation Law."""
        return validate_gcl_coarse(
            metrics.volume,
            metrics.Si_x, metrics.Si_y,
            metrics.Sj_x, metrics.Sj_y
        )
    
    @staticmethod
    def can_coarsen(NI: int, NJ: int, min_size: int = 4) -> bool:
        """Check if grid can be coarsened further."""
        return NI >= 2 * min_size and NJ >= 2 * min_size
    
    @staticmethod
    def max_levels(NI: int, NJ: int, min_size: int = 4) -> int:
        """Compute maximum number of multigrid levels."""
        levels = 1
        ni, nj = NI, NJ
        
        while ni >= 2 * min_size and nj >= 2 * min_size:
            ni = ni // 2
            nj = nj // 2
            levels += 1
        
        return levels
