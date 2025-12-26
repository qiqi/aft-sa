"""
Plot3D Grid Reader and Metrics Calculator.

This module provides functionality for reading Plot3D structured grid files
and computing finite volume method (FVM) grid metrics.
"""

import struct
from pathlib import Path
from typing import Tuple, Optional, NamedTuple
from dataclasses import dataclass
import numpy as np


class GridMetrics(NamedTuple):
    """Container for computed grid metrics."""
    
    # Node coordinates
    x: np.ndarray           # (ni, nj) node x-coordinates
    y: np.ndarray           # (ni, nj) node y-coordinates
    
    # Cell centers
    xc: np.ndarray          # (ni-1, nj-1) cell center x
    yc: np.ndarray          # (ni-1, nj-1) cell center y
    
    # Cell volumes (areas in 2D)
    volume: np.ndarray      # (ni-1, nj-1) cell areas
    
    # Face normals (i-faces: between cells in i-direction)
    # Normal points from cell (i,j) toward cell (i+1,j)
    Si_x: np.ndarray        # (ni, nj-1) i-face normal x-component
    Si_y: np.ndarray        # (ni, nj-1) i-face normal y-component
    Si_mag: np.ndarray      # (ni, nj-1) i-face area (length in 2D)
    
    # Face normals (j-faces: between cells in j-direction)
    # Normal points from cell (i,j) toward cell (i,j+1)
    Sj_x: np.ndarray        # (ni-1, nj) j-face normal x-component
    Sj_y: np.ndarray        # (ni-1, nj) j-face normal y-component
    Sj_mag: np.ndarray      # (ni-1, nj) j-face area (length in 2D)
    
    # Wall distance (from each node to nearest wall)
    wall_distance: np.ndarray   # (ni, nj) distance to wall


def read_plot3d_ascii(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a 2D Plot3D structured grid file in ASCII format.
    
    Supports both single-block and multi-block formats.
    For multi-block, only the first block is returned.
    
    Parameters
    ----------
    filename : str
        Path to the .p3d file.
        
    Returns
    -------
    X, Y : ndarray
        Grid coordinates with shape (ni, nj).
    """
    with open(filename, 'r') as f:
        tokens = f.read().split()
    
    ptr = 0
    
    # Try to detect format
    # Single block 2D: "ni nj" on first line
    # Multi-block 2D: "nblocks" then "ni nj" for each block
    # Single block 3D: "ni nj nk"
    
    first_val = int(tokens[0])
    second_val = int(tokens[1])
    
    # Heuristic: if first value is 1 or small and second is large, likely multi-block
    # If both are moderate (50-500), likely single-block dimensions
    
    # Construct2D outputs: "ni nj" directly (2D single block, no block count)
    # Let's check if there are enough tokens for the declared dimensions
    
    # Assume 2D single block first
    ni, nj = first_val, second_val
    n_points = ni * nj
    expected_tokens = 2 + 2 * n_points  # header + x + y
    
    if len(tokens) >= expected_tokens:
        ptr = 2
    else:
        # Maybe 3D with nk=1?
        if len(tokens) >= 3:
            nk = int(tokens[2])
            n_points_3d = ni * nj * nk
            expected_3d = 3 + 3 * n_points_3d
            if len(tokens) >= expected_3d:
                ptr = 3
                n_points = n_points_3d
    
    # Read X coordinates (Fortran column-major order)
    x_flat = np.array([float(tokens[ptr + i]) for i in range(n_points)])
    ptr += n_points
    
    # Read Y coordinates
    y_flat = np.array([float(tokens[ptr + i]) for i in range(n_points)])
    
    # Reshape to (ni, nj) - Fortran order
    X = x_flat.reshape((ni, nj), order='F')
    Y = y_flat.reshape((ni, nj), order='F')
    
    return X, Y


def read_plot3d_binary(filename: str, single_precision: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a 2D Plot3D structured grid file in binary format.
    
    Supports Fortran unformatted binary with record markers.
    
    Parameters
    ----------
    filename : str
        Path to the .p3d file.
    single_precision : bool
        If True, coordinates are 4-byte floats; otherwise 8-byte doubles.
        
    Returns
    -------
    X, Y : ndarray
        Grid coordinates with shape (ni, nj).
    """
    dtype = np.float32 if single_precision else np.float64
    int_size = 4
    float_size = 4 if single_precision else 8
    
    with open(filename, 'rb') as f:
        # Read first record marker
        rec1_start = struct.unpack('i', f.read(int_size))[0]
        
        # Determine if this is dimensions or block count
        # For single block: record contains ni, nj (8 or 16 bytes)
        # For multi-block: record contains nblocks (4 bytes)
        
        if rec1_start == 4:
            # Single integer - block count
            nblocks = struct.unpack('i', f.read(int_size))[0]
            f.read(int_size)  # end marker
            
            # Next record: dimensions
            f.read(int_size)  # start marker
            ni = struct.unpack('i', f.read(int_size))[0]
            nj = struct.unpack('i', f.read(int_size))[0]
            f.read(int_size)  # end marker
        else:
            # Dimensions directly
            ni = struct.unpack('i', f.read(int_size))[0]
            nj = struct.unpack('i', f.read(int_size))[0]
            # Check for nk
            remaining = rec1_start - 2 * int_size
            if remaining >= int_size:
                nk = struct.unpack('i', f.read(int_size))[0]
            f.read(int_size)  # end marker
        
        n_points = ni * nj
        
        # Read coordinate record
        f.read(int_size)  # start marker
        x_flat = np.frombuffer(f.read(n_points * float_size), dtype=dtype)
        y_flat = np.frombuffer(f.read(n_points * float_size), dtype=dtype)
        # f.read(int_size)  # end marker
    
    X = x_flat.reshape((ni, nj), order='F')
    Y = y_flat.reshape((ni, nj), order='F')
    
    return X, Y


def read_plot3d(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a Plot3D structured grid file (auto-detect ASCII/binary).
    
    Parameters
    ----------
    filename : str
        Path to the .p3d or .x file.
        
    Returns
    -------
    X, Y : ndarray
        Grid coordinates with shape (ni, nj).
    """
    path = Path(filename)
    
    # Try to detect format by reading first few bytes
    with open(path, 'rb') as f:
        header = f.read(32)
    
    # ASCII files typically start with digits and whitespace
    # Binary files have record markers (often small integers like 4, 8, 12...)
    
    try:
        # Check if it looks like ASCII
        header_str = header.decode('ascii')
        if any(c.isdigit() for c in header_str[:10]):
            return read_plot3d_ascii(filename)
    except UnicodeDecodeError:
        pass
    
    # Try binary
    return read_plot3d_binary(filename)


def compute_cell_centers(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cell center coordinates.
    
    For a structured grid, cell (i,j) is bounded by nodes:
    (i,j), (i+1,j), (i+1,j+1), (i,j+1)
    
    Parameters
    ----------
    X, Y : ndarray
        Node coordinates, shape (ni, nj).
        
    Returns
    -------
    Xc, Yc : ndarray
        Cell center coordinates, shape (ni-1, nj-1).
    """
    Xc = 0.25 * (X[:-1, :-1] + X[1:, :-1] + X[1:, 1:] + X[:-1, 1:])
    Yc = 0.25 * (Y[:-1, :-1] + Y[1:, :-1] + Y[1:, 1:] + Y[:-1, 1:])
    return Xc, Yc


def compute_cell_volumes(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute cell volumes (areas in 2D).
    
    Uses the cross-product of diagonals formula:
    Area = 0.5 * |AC × BD|
    
    where A,B,C,D are the four vertices of the quadrilateral cell.
    
    Parameters
    ----------
    X, Y : ndarray
        Node coordinates, shape (ni, nj).
        
    Returns
    -------
    volume : ndarray
        Cell areas, shape (ni-1, nj-1).
    """
    # Vertices: A=(i,j), B=(i+1,j), C=(i+1,j+1), D=(i,j+1)
    # Diagonal AC: C - A
    dx_ac = X[1:, 1:] - X[:-1, :-1]
    dy_ac = Y[1:, 1:] - Y[:-1, :-1]
    
    # Diagonal BD: D - B
    dx_bd = X[:-1, 1:] - X[1:, :-1]
    dy_bd = Y[:-1, 1:] - Y[1:, :-1]
    
    # Cross product (z-component): AC × BD
    volume = 0.5 * np.abs(dx_ac * dy_bd - dy_ac * dx_bd)
    
    return volume


def compute_face_normals_i(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute i-face normals (faces between cells in i-direction).
    
    Face at (i, j) connects nodes (i, j) and (i, j+1).
    Normal points in the positive i-direction (from cell i-1 to cell i).
    
    Parameters
    ----------
    X, Y : ndarray
        Node coordinates, shape (ni, nj).
        
    Returns
    -------
    Sx, Sy : ndarray
        Face normal components, shape (ni, nj-1).
    S_mag : ndarray
        Face area (length in 2D), shape (ni, nj-1).
    """
    # Face vector: from node (i,j) to node (i,j+1)
    dx = X[:, 1:] - X[:, :-1]
    dy = Y[:, 1:] - Y[:, :-1]
    
    # Normal is perpendicular (rotate 90 degrees)
    # For CCW rotation: (dx, dy) -> (-dy, dx)
    # But we want normal pointing in +i direction
    # If j increases upward and i increases rightward, normal should point right
    Sx = dy   # Normal x-component
    Sy = -dx  # Normal y-component
    
    S_mag = np.sqrt(Sx**2 + Sy**2)
    
    return Sx, Sy, S_mag


def compute_face_normals_j(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute j-face normals (faces between cells in j-direction).
    
    Face at (i, j) connects nodes (i, j) and (i+1, j).
    Normal points in the positive j-direction (from cell j-1 to cell j).
    
    Parameters
    ----------
    X, Y : ndarray
        Node coordinates, shape (ni, nj).
        
    Returns
    -------
    Sx, Sy : ndarray
        Face normal components, shape (ni-1, nj).
    S_mag : ndarray
        Face area (length in 2D), shape (ni-1, nj).
    """
    # Face vector: from node (i,j) to node (i+1,j)
    dx = X[1:, :] - X[:-1, :]
    dy = Y[1:, :] - Y[:-1, :]
    
    # Normal perpendicular, pointing in +j direction
    # Rotate 90 degrees CW: (dx, dy) -> (dy, -dx)
    Sx = -dy  # Normal x-component
    Sy = dx   # Normal y-component
    
    S_mag = np.sqrt(Sx**2 + Sy**2)
    
    return Sx, Sy, S_mag


def compute_wall_distance(X: np.ndarray, Y: np.ndarray, wall_j: int = 0) -> np.ndarray:
    """
    Compute wall distance for each node.
    
    For C-grids, the wall (airfoil surface) is typically at j=0.
    This computes the minimum distance from each node to any point on the wall.
    
    Parameters
    ----------
    X, Y : ndarray
        Node coordinates, shape (ni, nj).
    wall_j : int
        j-index of the wall boundary.
        
    Returns
    -------
    d : ndarray
        Wall distance for each node, shape (ni, nj).
    """
    ni, nj = X.shape
    
    # Wall coordinates
    x_wall = X[:, wall_j]
    y_wall = Y[:, wall_j]
    
    # For each node, find minimum distance to wall
    d = np.zeros((ni, nj))
    
    for j in range(nj):
        # Distance from each node at this j-level to all wall points
        dx = X[:, j:j+1] - x_wall[np.newaxis, :]
        dy = Y[:, j:j+1] - y_wall[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        d[:, j] = np.min(dist, axis=1)
    
    return d


def compute_wall_distance_fast(X: np.ndarray, Y: np.ndarray, wall_j: int = 0) -> np.ndarray:
    """
    Compute wall distance using approximate normal distance.
    
    For structured grids with wall-normal lines, this is faster and
    often sufficient. It assumes grid lines are roughly perpendicular
    to the wall.
    
    Parameters
    ----------
    X, Y : ndarray
        Node coordinates, shape (ni, nj).
    wall_j : int
        j-index of the wall boundary.
        
    Returns
    -------
    d : ndarray
        Wall distance for each node, shape (ni, nj).
    """
    ni, nj = X.shape
    d = np.zeros((ni, nj))
    
    # For each i-line, compute cumulative distance from wall
    for i in range(ni):
        x_line = X[i, :]
        y_line = Y[i, :]
        
        # Cumulative distance along the line
        dx = np.diff(x_line)
        dy = np.diff(y_line)
        ds = np.sqrt(dx**2 + dy**2)
        
        if wall_j == 0:
            d[i, 0] = 0.0
            d[i, 1:] = np.cumsum(ds)
        else:
            d[i, wall_j] = 0.0
            d[i, :wall_j] = np.cumsum(ds[:wall_j][::-1])[::-1]
            d[i, wall_j+1:] = np.cumsum(ds[wall_j:])
    
    return d


def compute_metrics(X: np.ndarray, Y: np.ndarray, wall_j: int = 0) -> GridMetrics:
    """
    Compute all grid metrics for FVM discretization.
    
    Parameters
    ----------
    X, Y : ndarray
        Node coordinates, shape (ni, nj).
    wall_j : int
        j-index of the wall boundary.
        
    Returns
    -------
    metrics : GridMetrics
        Named tuple containing all computed metrics.
    """
    # Cell centers
    Xc, Yc = compute_cell_centers(X, Y)
    
    # Cell volumes
    volume = compute_cell_volumes(X, Y)
    
    # Face normals
    Si_x, Si_y, Si_mag = compute_face_normals_i(X, Y)
    Sj_x, Sj_y, Sj_mag = compute_face_normals_j(X, Y)
    
    # Wall distance
    wall_distance = compute_wall_distance_fast(X, Y, wall_j)
    
    return GridMetrics(
        x=X, y=Y,
        xc=Xc, yc=Yc,
        volume=volume,
        Si_x=Si_x, Si_y=Si_y, Si_mag=Si_mag,
        Sj_x=Sj_x, Sj_y=Sj_y, Sj_mag=Sj_mag,
        wall_distance=wall_distance
    )


@dataclass
class StructuredGrid:
    """
    Structured grid class for 2D CFD simulations.
    
    Provides convenient access to grid coordinates and computed metrics.
    """
    
    X: np.ndarray
    Y: np.ndarray
    _metrics: Optional[GridMetrics] = None
    
    @classmethod
    def from_plot3d(cls, filename: str) -> 'StructuredGrid':
        """Load a grid from a Plot3D file."""
        X, Y = read_plot3d(filename)
        return cls(X=X, Y=Y)
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Grid dimensions (ni, nj)."""
        return self.X.shape
    
    @property
    def ni(self) -> int:
        """Number of points in i-direction."""
        return self.X.shape[0]
    
    @property
    def nj(self) -> int:
        """Number of points in j-direction."""
        return self.X.shape[1]
    
    @property
    def n_cells(self) -> int:
        """Total number of cells."""
        return (self.ni - 1) * (self.nj - 1)
    
    @property
    def metrics(self) -> GridMetrics:
        """Computed grid metrics (lazy evaluation)."""
        if self._metrics is None:
            self._metrics = compute_metrics(self.X, self.Y)
        return self._metrics
    
    def compute_metrics(self, wall_j: int = 0) -> GridMetrics:
        """Compute and cache grid metrics."""
        self._metrics = compute_metrics(self.X, self.Y, wall_j)
        return self._metrics
    
    def get_surface_coordinates(self, j: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Get coordinates along a constant-j line (e.g., airfoil surface)."""
        return self.X[:, j], self.Y[:, j]
    
    def get_wake_cut(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get coordinates along a constant-i line (e.g., wake cut)."""
        return self.X[i, :], self.Y[i, :]
    
    def write_plot3d(self, filename: str, ascii: bool = True):
        """
        Write grid to Plot3D format.
        
        Parameters
        ----------
        filename : str
            Output file path.
        ascii : bool
            If True, write ASCII format; otherwise binary.
        """
        if ascii:
            self._write_plot3d_ascii(filename)
        else:
            self._write_plot3d_binary(filename)
    
    def _write_plot3d_ascii(self, filename: str):
        """Write grid in ASCII Plot3D format."""
        ni, nj = self.shape
        
        with open(filename, 'w') as f:
            f.write(f"{ni:10d}{nj:10d}\n")
            
            # Write X coordinates
            for val in self.X.flatten(order='F'):
                f.write(f"  {val:18.10E}\n")
            
            # Write Y coordinates
            for val in self.Y.flatten(order='F'):
                f.write(f"  {val:18.10E}\n")
    
    def _write_plot3d_binary(self, filename: str):
        """Write grid in binary Plot3D format."""
        ni, nj = self.shape
        
        with open(filename, 'wb') as f:
            # Write dimensions record
            rec_size = 8  # two 4-byte integers
            f.write(struct.pack('i', rec_size))
            f.write(struct.pack('ii', ni, nj))
            f.write(struct.pack('i', rec_size))
            
            # Write coordinates record
            n_points = ni * nj
            rec_size = 2 * n_points * 4  # two arrays of floats
            f.write(struct.pack('i', rec_size))
            f.write(self.X.astype(np.float32).flatten(order='F').tobytes())
            f.write(self.Y.astype(np.float32).flatten(order='F').tobytes())
            f.write(struct.pack('i', rec_size))


def check_grid_quality(X: np.ndarray, Y: np.ndarray) -> dict:
    """
    Compute grid quality metrics.
    
    Parameters
    ----------
    X, Y : ndarray
        Node coordinates, shape (ni, nj).
        
    Returns
    -------
    quality : dict
        Dictionary containing:
        - 'orthogonality': sin(angle) at each interior node (ideal: 1.0)
        - 'jacobian': Jacobian (area element) at each node
        - 'min_jacobian': Minimum Jacobian (should be positive)
        - 'max_aspect_ratio': Maximum cell aspect ratio
        - 'max_skew_angle': Maximum skew angle in degrees
    """
    ni, nj = X.shape
    
    # Grid vectors using central differences at interior nodes
    dx_i = X[2:, 1:-1] - X[:-2, 1:-1]
    dy_i = Y[2:, 1:-1] - Y[:-2, 1:-1]
    
    dx_j = X[1:-1, 2:] - X[1:-1, :-2]
    dy_j = Y[1:-1, 2:] - Y[1:-1, :-2]
    
    # Magnitudes
    mag_i = np.sqrt(dx_i**2 + dy_i**2)
    mag_j = np.sqrt(dx_j**2 + dy_j**2)
    
    # Cross product (Jacobian-like)
    cross = dx_i * dy_j - dy_i * dx_j
    
    # Orthogonality: sin(angle) = cross / (mag_i * mag_j)
    ortho = cross / (mag_i * mag_j + 1e-12)
    
    # Dot product for angle
    dot = dx_i * dx_j + dy_i * dy_j
    cos_angle = dot / (mag_i * mag_j + 1e-12)
    
    # Skew angle (deviation from 90 degrees)
    skew_angle = np.abs(np.arccos(np.clip(cos_angle, -1, 1)) - np.pi/2) * 180 / np.pi
    
    # Cell volumes
    volumes = compute_cell_volumes(X, Y)
    
    # Aspect ratio (approximate)
    ds_i = np.sqrt((X[1:, :-1] - X[:-1, :-1])**2 + (Y[1:, :-1] - Y[:-1, :-1])**2)
    ds_j = np.sqrt((X[:-1, 1:] - X[:-1, :-1])**2 + (Y[:-1, 1:] - Y[:-1, :-1])**2)
    aspect = np.maximum(ds_i, ds_j) / (np.minimum(ds_i, ds_j) + 1e-12)
    
    return {
        'orthogonality': ortho,
        'jacobian': cross,
        'min_jacobian': np.min(cross),
        'max_aspect_ratio': np.max(aspect),
        'max_skew_angle': np.max(skew_angle),
        'min_volume': np.min(volumes),
        'max_volume': np.max(volumes)
    }

