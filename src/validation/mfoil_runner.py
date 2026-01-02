"""
Mfoil (XFOIL-like) runner utilities for RANS validation.
"""

import numpy as np
from typing import Optional, Dict, Any


def run_laminar(reynolds: float, alpha: float = 0.0,
                naca: str = '0012', npanel: int = 199,
                airfoil_file: Optional[str] = None) -> Dict[str, Any]:
    """Run mfoil for fully laminar flow."""
    from .mfoil import mfoil
    
    if airfoil_file is not None:
        coords = _load_airfoil_coords(airfoil_file)
        M = mfoil(coords=coords, npanel=npanel)
    else:
        M = mfoil(naca=naca, npanel=npanel)
    
    M.param.ncrit = 1000.0
    M.param.doplot = False
    M.param.verb = 0
    M.setoper(alpha=alpha, Re=reynolds)
    
    try:
        M.solve()
        converged = True
    except Exception as e:
        print(f"mfoil solve failed: {e}")
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


def _load_airfoil_coords(airfoil_file: str) -> np.ndarray:
    """Load airfoil coordinates from a .dat file. Returns (2, N) array."""
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
    """Extract surface Cp and Cf distributions, split into upper/lower surfaces."""
    x_coords = M.foil.x[0, :].copy()
    y_coords = M.foil.x[1, :].copy()
    
    cp_all = M.post.cp.copy() if hasattr(M.post, 'cp') and M.post.cp is not None else None
    cf_all = M.post.cf.copy() if hasattr(M.post, 'cf') and M.post.cf is not None else None
    
    N = M.foil.N
    n_half = N // 2
    
    x_lower = x_coords[:n_half+1][::-1]
    y_lower = y_coords[:n_half+1][::-1]
    x_upper = x_coords[n_half:]
    y_upper = y_coords[n_half:]
    
    result = {
        'x_upper': x_upper,
        'y_upper': y_upper,
        'x_lower': x_lower,
        'y_lower': y_lower,
    }
    
    if cp_all is not None:
        result['cp_lower'] = cp_all[:n_half+1][::-1]  # LE→TE
        result['cp_upper'] = cp_all[n_half:N]          # Already LE→TE
    else:
        result['cp_lower'] = np.zeros(n_half+1)
        result['cp_upper'] = np.zeros(N - n_half)
    
    if cf_all is not None:
        result['cf_lower'] = cf_all[:n_half+1][::-1]
        result['cf_upper'] = cf_all[n_half:N]
    else:
        result['cf_lower'] = np.zeros(n_half+1)
        result['cf_upper'] = np.zeros(N - n_half)
    
    return result


def run_turbulent(reynolds: float, alpha: float = 0.0,
                  naca: str = '0012', npanel: int = 199,
                  ncrit: float = 9.0) -> Dict[str, Any]:
    """Run mfoil with natural transition (e^N method)."""
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

