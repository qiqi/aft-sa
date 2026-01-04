"""
Grid loading utilities.
"""

import math
import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Tuple, Optional, List

from .mesher import Construct2DWrapper, GridOptions
from .plot3d import read_plot3d

NDArrayFloat = npt.NDArray[np.floating]


def compute_n_normal_from_gradation(
    y_plus: float,
    reynolds: float,
    farfield_radius: float,
    gradation: float,
    lref: float = 0.5,
) -> int:
    """
    Compute number of wall-normal cells needed for given gradation.
    
    Uses geometric series: y0 * (g^N - 1) / (g - 1) = L
    Solving for N: N = ln(L*(g-1)/y0 + 1) / ln(g)
    
    Parameters
    ----------
    y_plus : float
        Target y+ for first cell
    reynolds : float
        Reynolds number
    farfield_radius : float
        Farfield distance in chords
    gradation : float
        Cell-to-cell growth ratio (e.g., 1.2)
    lref : float
        Reference length for skin friction (chord fraction, default 0.5)
    
    Returns
    -------
    int
        Number of wall-normal nodes (n_normal)
    """
    # First cell size using Construct2D's formula from surface_util.f90:
    # Cf = (2*log10(Re) - 0.65)^(-2.3)
    # y0 = y+ * Lref / (Re * sqrt(0.5 * Cf))
    cf = (2.0 * math.log10(reynolds) - 0.65) ** (-2.3)
    y0 = y_plus * lref / (reynolds * math.sqrt(0.5 * cf))
    
    # Total distance to span (Construct2D uses 1.2x multiplier internally)
    L = farfield_radius * 1.2
    
    # Geometric series: y0 * (g^N - 1) / (g - 1) = L
    # Solve for N
    g = gradation
    if g <= 1.0:
        raise ValueError(f"gradation must be > 1.0, got {g}")
    
    ratio = L * (g - 1) / y0 + 1
    N = math.log(ratio) / math.log(g)
    
    # n_normal = N + 1 (cells to nodes), round up
    n_normal = int(math.ceil(N)) + 1
    
    return n_normal


def find_construct2d_binary(project_root: Optional[Path] = None) -> Optional[Path]:
    """Find the construct2d binary in common locations."""
    search_paths: List[Path] = []
    
    if project_root is not None:
        search_paths.extend([
            project_root / "external" / "construct2d" / "construct2d",
            project_root / "bin" / "construct2d",
        ])
    
    search_paths.extend([
        Path("./bin/construct2d"),
        Path("./construct2d"),
        Path("/usr/local/bin/construct2d"),
    ])
    
    for p in search_paths:
        if p.exists():
            return p
    
    return None


def load_or_generate_grid(
    grid_file: str,
    n_surface: int = 100,
    n_wake: int = 30,
    y_plus: float = 1.0,
    gradation: float = 1.2,
    reynolds: float = 1e6,
    farfield_radius: float = 15.0,
    max_first_cell: float = 0.001,
    project_root: Optional[Path] = None,
    verbose: bool = True
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """Load a grid from file or generate from airfoil coordinates.
    
    Parameters
    ----------
    grid_file : str
        Path to airfoil .dat file or pre-generated .p3d grid
    n_surface : int
        Number of surface nodes
    n_wake : int
        Number of wake region nodes
    y_plus : float
        Target y+ for first cell
    gradation : float
        Wall-normal cell growth ratio (e.g., 1.2)
    reynolds : float
        Reynolds number (used to compute first cell size)
    farfield_radius : float
        Farfield distance in chords
    max_first_cell : float
        Maximum first cell size (safety limit)
    project_root : Path, optional
        Project root for finding Construct2D binary
    verbose : bool
        Print grid generation details
    
    Returns
    -------
    X, Y : ndarray
        Grid node coordinates
    """
    grid_path: Path = Path(grid_file)
    
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")
    
    suffix: str = grid_path.suffix.lower()
    
    X: NDArrayFloat
    Y: NDArrayFloat
    
    if suffix in ['.p3d', '.x', '.xyz']:
        if verbose:
            print(f"Loading grid from: {grid_path}")
        X, Y = read_plot3d(str(grid_path))
        
    elif suffix == '.dat':
        # Compute n_normal from gradation
        n_normal = compute_n_normal_from_gradation(
            y_plus=y_plus,
            reynolds=reynolds,
            farfield_radius=farfield_radius,
            gradation=gradation,
        )
        
        if verbose:
            print(f"Generating grid from airfoil: {grid_path}")
            print(f"  Surface points:  {n_surface}")
            print(f"  Wake points:     {n_wake}")
            print(f"  Target y+:       {y_plus}")
            print(f"  Gradation:       {gradation:.2f}")
            print(f"  → Normal points: {n_normal} (computed)")
        
        binary_path: Optional[Path] = find_construct2d_binary(project_root)
        
        if binary_path is None:
            raise FileNotFoundError(
                "Construct2D binary not found. Please provide a .p3d grid file "
                "or install Construct2D."
            )
        
        wrapper: Construct2DWrapper = Construct2DWrapper(str(binary_path))
        grid_opts: GridOptions = GridOptions(
            n_surface=n_surface,
            n_normal=n_normal,
            n_wake=n_wake,
            y_plus=y_plus,
            reynolds=reynolds,
            topology='CGRD',
            farfield_radius=farfield_radius,
            max_first_cell=max_first_cell,
        )
        X, Y = wrapper.generate(str(grid_path), grid_opts, verbose=verbose)
    else:
        raise ValueError(f"Unsupported grid file format: {suffix}")
    
    if verbose:
        print(f"Grid size: {X.shape[0]} x {X.shape[1]} nodes ({X.shape[0]-1} x {X.shape[1]-1} cells)")
    
    return X, Y
