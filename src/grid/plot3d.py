"""
Plot3D grid reader and metrics calculator.
"""

import struct
from pathlib import Path
from typing import Tuple, Optional, NamedTuple
from dataclasses import dataclass
import numpy as np


class GridMetrics(NamedTuple):
    """Container for computed grid metrics."""
    x: np.ndarray           # (ni, nj) node x
    y: np.ndarray           # (ni, nj) node y
    xc: np.ndarray          # (ni-1, nj-1) cell center x
    yc: np.ndarray          # (ni-1, nj-1) cell center y
    volume: np.ndarray      # (ni-1, nj-1) cell areas
    Si_x: np.ndarray        # (ni, nj-1) i-face normal x
    Si_y: np.ndarray        # (ni, nj-1) i-face normal y
    Si_mag: np.ndarray      # (ni, nj-1) i-face area
    Sj_x: np.ndarray        # (ni-1, nj) j-face normal x
    Sj_y: np.ndarray        # (ni-1, nj) j-face normal y
    Sj_mag: np.ndarray      # (ni-1, nj) j-face area
    wall_distance: np.ndarray


def read_plot3d_ascii(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read 2D Plot3D grid in ASCII format."""
    with open(filename, 'r') as f:
        tokens = f.read().split()
    
    ptr = 0
    first_val = int(tokens[0])
    second_val = int(tokens[1])
    
    ni, nj = first_val, second_val
    n_points = ni * nj
    expected_tokens = 2 + 2 * n_points
    
    if len(tokens) >= expected_tokens:
        ptr = 2
    else:
        if len(tokens) >= 3:
            nk = int(tokens[2])
            n_points_3d = ni * nj * nk
            expected_3d = 3 + 3 * n_points_3d
            if len(tokens) >= expected_3d:
                ptr = 3
                n_points = n_points_3d
    
    x_flat = np.array([float(tokens[ptr + i]) for i in range(n_points)])
    ptr += n_points
    y_flat = np.array([float(tokens[ptr + i]) for i in range(n_points)])
    
    X = x_flat.reshape((ni, nj), order='F')
    Y = y_flat.reshape((ni, nj), order='F')
    
    return X, Y


def read_plot3d_binary(filename: str, single_precision: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Read 2D Plot3D grid in binary format."""
    dtype = np.float32 if single_precision else np.float64
    int_size = 4
    float_size = 4 if single_precision else 8
    
    with open(filename, 'rb') as f:
        rec1_start = struct.unpack('i', f.read(int_size))[0]
        
        if rec1_start == 4:
            nblocks = struct.unpack('i', f.read(int_size))[0]
            f.read(int_size)
            f.read(int_size)
            ni = struct.unpack('i', f.read(int_size))[0]
            nj = struct.unpack('i', f.read(int_size))[0]
            f.read(int_size)
        else:
            ni = struct.unpack('i', f.read(int_size))[0]
            nj = struct.unpack('i', f.read(int_size))[0]
            remaining = rec1_start - 2 * int_size
            if remaining >= int_size:
                nk = struct.unpack('i', f.read(int_size))[0]
            f.read(int_size)
        
        n_points = ni * nj
        
        f.read(int_size)
        x_flat = np.frombuffer(f.read(n_points * float_size), dtype=dtype)
        y_flat = np.frombuffer(f.read(n_points * float_size), dtype=dtype)
    
    X = x_flat.reshape((ni, nj), order='F')
    Y = y_flat.reshape((ni, nj), order='F')
    
    return X, Y


def read_plot3d(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read Plot3D file (auto-detect ASCII/binary)."""
    path = Path(filename)
    
    with open(path, 'rb') as f:
        header = f.read(32)
    
    try:
        header_str = header.decode('ascii')
        if any(c.isdigit() for c in header_str[:10]):
            return read_plot3d_ascii(filename)
    except UnicodeDecodeError:
        pass
    
    return read_plot3d_binary(filename)


def compute_cell_centers(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute cell center coordinates."""
    Xc = 0.25 * (X[:-1, :-1] + X[1:, :-1] + X[1:, 1:] + X[:-1, 1:])
    Yc = 0.25 * (Y[:-1, :-1] + Y[1:, :-1] + Y[1:, 1:] + Y[:-1, 1:])
    return Xc, Yc


def compute_cell_volumes(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute cell areas using cross-product of diagonals."""
    dx_ac = X[1:, 1:] - X[:-1, :-1]
    dy_ac = Y[1:, 1:] - Y[:-1, :-1]
    dx_bd = X[:-1, 1:] - X[1:, :-1]
    dy_bd = Y[:-1, 1:] - Y[1:, :-1]
    return 0.5 * np.abs(dx_ac * dy_bd - dy_ac * dx_bd)


def compute_face_normals_i(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute i-face normals (perpendicular to i-direction)."""
    dx = X[:, 1:] - X[:, :-1]
    dy = Y[:, 1:] - Y[:, :-1]
    Sx = dy
    Sy = -dx
    S_mag = np.sqrt(Sx**2 + Sy**2)
    return Sx, Sy, S_mag


def compute_face_normals_j(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute j-face normals (perpendicular to j-direction)."""
    dx = X[1:, :] - X[:-1, :]
    dy = Y[1:, :] - Y[:-1, :]
    Sx = -dy
    Sy = dx
    S_mag = np.sqrt(Sx**2 + Sy**2)
    return Sx, Sy, S_mag


def _point_to_segment_distance(px: float, py: float, 
                                ax: float, ay: float, 
                                bx: float, by: float) -> float:
    """Minimum distance from point P to line segment AB."""
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab_sq = abx * abx + aby * aby
    
    if ab_sq < 1e-30:
        return np.sqrt(apx * apx + apy * apy)
    
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_sq))
    closest_x = ax + t * abx
    closest_y = ay + t * aby
    dx = px - closest_x
    dy = py - closest_y
    
    return np.sqrt(dx * dx + dy * dy)


def compute_wall_distance(X: np.ndarray, Y: np.ndarray, wall_j: int = 0,
                          search_radius: int = 20) -> np.ndarray:
    """Compute wall distance using point-to-segment distance."""
    ni, nj = X.shape
    x_wall = X[:, wall_j]
    y_wall = Y[:, wall_j]
    n_wall = len(x_wall)
    
    d = np.zeros((ni, nj))
    
    for i in range(ni):
        for j in range(nj):
            px = X[i, j]
            py = Y[i, j]
            start_idx = i
            min_dist = float('inf')
            idx_min = max(0, start_idx - search_radius)
            idx_max = min(n_wall - 1, start_idx + search_radius)
            
            for k in range(idx_min, idx_max):
                ax, ay = x_wall[k], y_wall[k]
                bx, by = x_wall[k + 1], y_wall[k + 1]
                dist = _point_to_segment_distance(px, py, ax, ay, bx, by)
                min_dist = min(min_dist, dist)
            
            d[i, j] = min_dist
    
    return d


def compute_wall_distance_fast(X: np.ndarray, Y: np.ndarray, wall_j: int = 0) -> np.ndarray:
    """Compute wall distance using cumulative distance along grid lines."""
    ni, nj = X.shape
    d = np.zeros((ni, nj))
    
    for i in range(ni):
        x_line = X[i, :]
        y_line = Y[i, :]
        
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
    """Compute all grid metrics for FVM discretization."""
    Xc, Yc = compute_cell_centers(X, Y)
    volume = compute_cell_volumes(X, Y)
    Si_x, Si_y, Si_mag = compute_face_normals_i(X, Y)
    Sj_x, Sj_y, Sj_mag = compute_face_normals_j(X, Y)
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
    """Structured grid class for 2D CFD simulations."""
    
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
        return self.X.shape
    
    @property
    def ni(self) -> int:
        return self.X.shape[0]
    
    @property
    def nj(self) -> int:
        return self.X.shape[1]
    
    @property
    def n_cells(self) -> int:
        return (self.ni - 1) * (self.nj - 1)
    
    @property
    def metrics(self) -> GridMetrics:
        if self._metrics is None:
            self._metrics = compute_metrics(self.X, self.Y)
        return self._metrics
    
    def compute_metrics(self, wall_j: int = 0) -> GridMetrics:
        """Compute and cache grid metrics."""
        self._metrics = compute_metrics(self.X, self.Y, wall_j)
        return self._metrics
    
    def get_surface_coordinates(self, j: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Get coordinates along a constant-j line."""
        return self.X[:, j], self.Y[:, j]
    
    def get_wake_cut(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get coordinates along a constant-i line."""
        return self.X[i, :], self.Y[i, :]
    
    def write_plot3d(self, filename: str, ascii: bool = True):
        """Write grid to Plot3D format."""
        if ascii:
            self._write_plot3d_ascii(filename)
        else:
            self._write_plot3d_binary(filename)
    
    def _write_plot3d_ascii(self, filename: str):
        ni, nj = self.shape
        with open(filename, 'w') as f:
            f.write(f"{ni:10d}{nj:10d}\n")
            for val in self.X.flatten(order='F'):
                f.write(f"  {val:18.10E}\n")
            for val in self.Y.flatten(order='F'):
                f.write(f"  {val:18.10E}\n")
    
    def _write_plot3d_binary(self, filename: str):
        ni, nj = self.shape
        with open(filename, 'wb') as f:
            rec_size = 8
            f.write(struct.pack('i', rec_size))
            f.write(struct.pack('ii', ni, nj))
            f.write(struct.pack('i', rec_size))
            
            n_points = ni * nj
            rec_size = 2 * n_points * 4
            f.write(struct.pack('i', rec_size))
            f.write(self.X.astype(np.float32).flatten(order='F').tobytes())
            f.write(self.Y.astype(np.float32).flatten(order='F').tobytes())
            f.write(struct.pack('i', rec_size))


def check_grid_quality(X: np.ndarray, Y: np.ndarray) -> dict:
    """Compute grid quality metrics."""
    ni, nj = X.shape
    
    dx_i = X[2:, 1:-1] - X[:-2, 1:-1]
    dy_i = Y[2:, 1:-1] - Y[:-2, 1:-1]
    dx_j = X[1:-1, 2:] - X[1:-1, :-2]
    dy_j = Y[1:-1, 2:] - Y[1:-1, :-2]
    
    mag_i = np.sqrt(dx_i**2 + dy_i**2)
    mag_j = np.sqrt(dx_j**2 + dy_j**2)
    
    cross = dx_i * dy_j - dy_i * dx_j
    ortho = cross / (mag_i * mag_j + 1e-12)
    
    dot = dx_i * dx_j + dy_i * dy_j
    cos_angle = dot / (mag_i * mag_j + 1e-12)
    skew_angle = np.abs(np.arccos(np.clip(cos_angle, -1, 1)) - np.pi/2) * 180 / np.pi
    
    volumes = compute_cell_volumes(X, Y)
    
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
