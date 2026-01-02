"""
Grid loading utilities.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional

from .mesher import Construct2DWrapper, GridOptions
from .plot3d import read_plot3d


def find_construct2d_binary(project_root: Optional[Path] = None) -> Optional[Path]:
    """Find the construct2d binary in common locations."""
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
    """Load a grid from file or generate from airfoil coordinates."""
    grid_path = Path(grid_file)
    
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")
    
    suffix = grid_path.suffix.lower()
    
    if suffix in ['.p3d', '.x', '.xyz']:
        if verbose:
            print(f"Loading grid from: {grid_path}")
        X, Y = read_plot3d(str(grid_path))
        
    elif suffix == '.dat':
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
