"""
Mfoil (XFOIL-like) runner utilities for validation.

This module provides convenience functions for running mfoil simulations
and extracting results for comparison with RANS solutions.
"""

import numpy as np
from typing import Optional, Dict, Any


def run_laminar(reynolds: float, alpha: float = 0.0,
                naca: str = '0012', npanel: int = 199,
                airfoil_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Run mfoil for fully laminar flow.
    
    Parameters
    ----------
    reynolds : float
        Reynolds number based on chord.
    alpha : float, optional
        Angle of attack in degrees (default 0).
    naca : str, optional
        NACA 4-digit airfoil code (default '0012').
        Ignored if airfoil_file is provided.
    npanel : int, optional
        Number of panels (default 199).
    airfoil_file : str, optional
        Path to airfoil coordinate file. If provided, uses this
        instead of NACA code.
        
    Returns
    -------
    result : dict
        Results containing:
        - cl: Lift coefficient
        - cd: Total drag coefficient
        - cdf: Skin friction drag coefficient
        - cdp: Pressure drag coefficient
        - converged: Whether solution converged
        - x_upper, x_lower: Surface x-coordinates
        - cp_upper, cp_lower: Surface Cp
        - cf_upper, cf_lower: Surface Cf
    """
    from .mfoil import mfoil
    
    # Load airfoil
    if airfoil_file is not None:
        coords = _load_airfoil_coords(airfoil_file)
        M = mfoil(coords=coords, npanel=npanel)
    else:
        M = mfoil(naca=naca, npanel=npanel)
    
    # Force fully laminar by setting ncrit extremely high
    M.param.ncrit = 1000.0
    M.param.doplot = False
    M.param.verb = 0
    
    # Set operating conditions
    M.setoper(alpha=alpha, Re=reynolds)
    
    # Solve
    try:
        M.solve()
        converged = True
    except Exception as e:
        print(f"mfoil solve failed: {e}")
        return {
            'cl': np.nan, 'cd': np.nan, 'cdf': np.nan, 'cdp': np.nan,
            'converged': False
        }
    
    # Extract surface data
    result = _extract_surface_data(M)
    result.update({
        'cl': M.post.cl,
        'cd': M.post.cd,
        'cdf': M.post.cdf,
        'cdp': M.post.cdp,
        'converged': converged,
    })
    
    return result


def _load_airfoil_coords(airfoil_file: str) -> np.ndarray:
    """
    Load airfoil coordinates from a .dat file.
    
    Returns coordinates in mfoil format: (2, N) array with x in row 0, y in row 1.
    """
    coords = []
    with open(airfoil_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x, y = float(parts[0]), float(parts[1])
                    coords.append([x, y])
                except ValueError:
                    continue
    
    return np.array(coords).T  # Shape (2, N)


def _extract_surface_data(M) -> Dict[str, np.ndarray]:
    """
    Extract surface Cp and Cf distributions from mfoil solution.
    
    Returns data split into upper and lower surfaces.
    """
    # Get coordinates and data from mfoil
    x_coords = M.foil.x[0, :].copy()
    y_coords = M.foil.x[1, :].copy()
    
    cp_all = M.post.cp.copy() if hasattr(M.post, 'cp') and M.post.cp is not None else None
    cf_all = M.post.cf.copy() if hasattr(M.post, 'cf') and M.post.cf is not None else None
    
    N = M.foil.N
    n_half = N // 2
    
    # Upper surface: first half, reversed to go LE to TE
    x_upper = x_coords[:n_half+1][::-1]
    y_upper = y_coords[:n_half+1][::-1]
    
    # Lower surface: second half
    x_lower = x_coords[n_half:]
    y_lower = y_coords[n_half:]
    
    result = {
        'x_upper': x_upper,
        'y_upper': y_upper,
        'x_lower': x_lower,
        'y_lower': y_lower,
    }
    
    if cp_all is not None:
        result['cp_upper'] = cp_all[:n_half+1][::-1]
        result['cp_lower'] = cp_all[n_half:N]
    else:
        result['cp_upper'] = np.zeros(n_half+1)
        result['cp_lower'] = np.zeros(N - n_half)
    
    if cf_all is not None:
        result['cf_upper'] = cf_all[:n_half+1][::-1]
        result['cf_lower'] = cf_all[n_half:N]
    else:
        result['cf_upper'] = np.zeros(n_half+1)
        result['cf_lower'] = np.zeros(N - n_half)
    
    return result


def run_turbulent(reynolds: float, alpha: float = 0.0,
                  naca: str = '0012', npanel: int = 199,
                  ncrit: float = 9.0) -> Dict[str, Any]:
    """
    Run mfoil with natural transition.
    
    Parameters
    ----------
    reynolds : float
        Reynolds number.
    alpha : float
        Angle of attack in degrees.
    naca : str
        NACA 4-digit code.
    npanel : int
        Number of panels.
    ncrit : float
        Critical amplification factor for transition (default 9.0).
        
    Returns
    -------
    result : dict
        Same format as run_laminar.
    """
    from .mfoil import mfoil
    
    M = mfoil(naca=naca, npanel=npanel)
    M.param.ncrit = ncrit
    M.param.doplot = False
    M.param.verb = 0
    
    M.setoper(alpha=alpha, Re=reynolds)
    
    try:
        M.solve()
        converged = True
    except Exception:
        return {
            'cl': np.nan, 'cd': np.nan, 'cdf': np.nan, 'cdp': np.nan,
            'converged': False
        }
    
    result = _extract_surface_data(M)
    result.update({
        'cl': M.post.cl,
        'cd': M.post.cd,
        'cdf': M.post.cdf,
        'cdp': M.post.cdp,
        'converged': converged,
    })
    
    return result

