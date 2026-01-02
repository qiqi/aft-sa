"""
Grid loading utilities.

This module provides common utilities for loading or generating grids,
used by multiple scripts to avoid code duplication.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional

from .mesher import Construct2DWrapper, GridOptions
from .plot3d import read_plot3d


def find_construct2d_binary(project_root: Optional[Path] = None) -> Optional[Path]:
    """
    Find the construct2d binary.
    
    Parameters
    ----------
    project_root : Path, optional
        Project root directory. If not provided, searches common locations.
        
    Returns
    -------
    binary_path : Path or None
        Path to construct2d binary, or None if not found.
    """
    search_paths = []
    
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
    n_normal: int = 40,
    n_wake: int = 30,
    y_plus: float = 1.0,
    reynolds: float = 1e6,
    farfield_radius: float = 15.0,
    max_first_cell: float = 0.001,
    project_root: Optional[Path] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a grid from file or generate from airfoil coordinates.
    
    Parameters
    ----------
    grid_file : str
        Path to grid file (.p3d, .x, .xyz) or airfoil file (.dat).
    n_surface : int
        Number of surface points (for grid generation).
    n_normal : int
        Number of normal points (for grid generation).
    n_wake : int
        Number of wake points (for grid generation).
    y_plus : float
        Target y+ value (for grid generation).
    reynolds : float
        Reynolds number (for grid generation).
    farfield_radius : float
        Farfield radius in chord lengths.
    max_first_cell : float
        Maximum first cell height in chord lengths.
    project_root : Path, optional
        Project root for finding construct2d binary.
    verbose : bool
        Print progress messages.
        
    Returns
    -------
    X, Y : ndarray
        Grid node coordinates, shape (NI+1, NJ+1).
    """
    grid_path = Path(grid_file)
    
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")
    
    suffix = grid_path.suffix.lower()
    
    if suffix in ['.p3d', '.x', '.xyz']:
        # Load Plot3D grid
        if verbose:
            print(f"Loading grid from: {grid_path}")
        X, Y = read_plot3d(str(grid_path))
        
    elif suffix == '.dat':
        # Airfoil file - generate grid with Construct2D
        if verbose:
            print(f"Generating grid from airfoil: {grid_path}")
            print(f"  Surface points: {n_surface}")
            print(f"  Normal points:  {n_normal}")
            print(f"  Wake points:    {n_wake}")
        
        binary_path = find_construct2d_binary(project_root)
        
        if binary_path is None:
            raise FileNotFoundError(
                "Construct2D binary not found. Please provide a .p3d grid file "
                "or install Construct2D."
            )
        
        wrapper = Construct2DWrapper(str(binary_path))
        grid_opts = GridOptions(
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
        print(f"Grid loaded: {X.shape[0]} x {X.shape[1]} nodes")
        print(f"            {X.shape[0]-1} x {X.shape[1]-1} cells")
    
    return X, Y


