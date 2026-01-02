"""
Plot3D grid reader and metrics calculator.
"""

import struct
from pathlib import Path
from typing import Tuple, Optional, NamedTuple, Dict, Any
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.floating]


class GridMetrics(NamedTuple):
    """Container for computed grid metrics."""
    x: NDArrayFloat
    y: NDArrayFloat
    xc: NDArrayFloat
    yc: NDArrayFloat
    volume: NDArrayFloat
    Si_x: NDArrayFloat
    Si_y: NDArrayFloat
    Si_mag: NDArrayFloat
    Sj_x: NDArrayFloat
    Sj_y: NDArrayFloat
    Sj_mag: NDArrayFloat
    wall_distance: NDArrayFloat


def read_plot3d_ascii(filename: str) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """Read 2D Plot3D grid in ASCII format."""
    with open(filename, 'r') as f:
        tokens: list[str] = f.read().split()
    
    ptr: int = 0
    first_val: int = int(tokens[0])
    second_val: int = int(tokens[1])
    
    ni: int = first_val
    nj: int = second_val
    n_points: int = ni * nj
    expected_tokens: int = 2 + 2 * n_points
    
    if len(tokens) >= expected_tokens:
        ptr = 2
    else:
        if len(tokens) >= 3:
            nk: int = int(tokens[2])
            n_points_3d: int = ni * nj * nk
            expected_3d: int = 3 + 3 * n_points_3d
            if len(tokens) >= expected_3d:
                ptr = 3
                n_points = n_points_3d
    
    x_flat: NDArrayFloat = np.array([float(tokens[ptr + i]) for i in range(n_points)])
    ptr += n_points
    y_flat: NDArrayFloat = np.array([float(tokens[ptr + i]) for i in range(n_points)])
    
    X: NDArrayFloat = x_flat.reshape((ni, nj), order='F')
    Y: NDArrayFloat = y_flat.reshape((ni, nj), order='F')
    
    return X, Y


def read_plot3d_binary(filename: str, single_precision: bool = True) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """Read 2D Plot3D grid in binary format."""
    dtype = np.float32 if single_precision else np.float64
    int_size: int = 4
    float_size: int = 4 if single_precision else 8
    
    ni: int
    nj: int
    with open(filename, 'rb') as f:
        rec1_start: int = struct.unpack('i', f.read(int_size))[0]
        
        if rec1_start == 4:
            nblocks: int = struct.unpack('i', f.read(int_size))[0]
            f.read(int_size)
            f.read(int_size)
            ni = struct.unpack('i', f.read(int_size))[0]
            nj = struct.unpack('i', f.read(int_size))[0]
            f.read(int_size)
        else:
            ni = struct.unpack('i', f.read(int_size))[0]
            nj = struct.unpack('i', f.read(int_size))[0]
            remaining: int = rec1_start - 2 * int_size
            if remaining >= int_size:
                nk: int = struct.unpack('i', f.read(int_size))[0]
            f.read(int_size)
        
        n_points: int = ni * nj
        
        f.read(int_size)
        x_flat: NDArrayFloat = np.frombuffer(f.read(n_points * float_size), dtype=dtype)
        y_flat: NDArrayFloat = np.frombuffer(f.read(n_points * float_size), dtype=dtype)
    
    X: NDArrayFloat = x_flat.reshape((ni, nj), order='F')
    Y: NDArrayFloat = y_flat.reshape((ni, nj), order='F')
    
    return X, Y


def read_plot3d(filename: str) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """Read Plot3D file (auto-detect ASCII/binary)."""
    path: Path = Path(filename)
    
    with open(path, 'rb') as f:
        header: bytes = f.read(32)
    
    try:
        header_str: str = header.decode('ascii')
        if any(c.isdigit() for c in header_str[:10]):
            return read_plot3d_ascii(filename)
    except UnicodeDecodeError:
        pass
    
    return read_plot3d_binary(filename)


def compute_cell_centers(X: NDArrayFloat, Y: NDArrayFloat) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """Compute cell center coordinates."""
    Xc: NDArrayFloat = 0.25 * (X[:-1, :-1] + X[1:, :-1] + X[1:, 1:] + X[:-1, 1:])
    Yc: NDArrayFloat = 0.25 * (Y[:-1, :-1] + Y[1:, :-1] + Y[1:, 1:] + Y[:-1, 1:])
    return Xc, Yc


def compute_cell_volumes(X: NDArrayFloat, Y: NDArrayFloat) -> NDArrayFloat:
    """Compute cell areas using cross-product of diagonals."""
    dx_ac: NDArrayFloat = X[1:, 1:] - X[:-1, :-1]
    dy_ac: NDArrayFloat = Y[1:, 1:] - Y[:-1, :-1]
    dx_bd: NDArrayFloat = X[:-1, 1:] - X[1:, :-1]
    dy_bd: NDArrayFloat = Y[:-1, 1:] - Y[1:, :-1]
    return 0.5 * np.abs(dx_ac * dy_bd - dy_ac * dx_bd)


def compute_face_normals_i(X: NDArrayFloat, Y: NDArrayFloat) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Compute i-face normals (perpendicular to i-direction)."""
    dx: NDArrayFloat = X[:, 1:] - X[:, :-1]
    dy: NDArrayFloat = Y[:, 1:] - Y[:, :-1]
    Sx: NDArrayFloat = dy
    Sy: NDArrayFloat = -dx
    S_mag: NDArrayFloat = np.sqrt(Sx**2 + Sy**2)
    return Sx, Sy, S_mag


def compute_face_normals_j(X: NDArrayFloat, Y: NDArrayFloat) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Compute j-face normals (perpendicular to j-direction)."""
    dx: NDArrayFloat = X[1:, :] - X[:-1, :]
    dy: NDArrayFloat = Y[1:, :] - Y[:-1, :]
    Sx: NDArrayFloat = -dy
    Sy: NDArrayFloat = dx
    S_mag: NDArrayFloat = np.sqrt(Sx**2 + Sy**2)
    return Sx, Sy, S_mag


def _point_to_segment_distance(px: float, py: float, 
                                ax: float, ay: float, 
                                bx: float, by: float) -> float:
    """Minimum distance from point P to line segment AB."""
    abx: float = bx - ax
    aby: float = by - ay
    apx: float = px - ax
    apy: float = py - ay
    ab_sq: float = abx * abx + aby * aby
    
    if ab_sq < 1e-30:
        return float(np.sqrt(apx * apx + apy * apy))
    
    t: float = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_sq))
    closest_x: float = ax + t * abx
    closest_y: float = ay + t * aby
    dx: float = px - closest_x
    dy: float = py - closest_y
    
    return float(np.sqrt(dx * dx + dy * dy))


def compute_wall_distance(X: NDArrayFloat, Y: NDArrayFloat, wall_j: int = 0,
                          search_radius: int = 20) -> NDArrayFloat:
    """Compute wall distance using point-to-segment distance."""
    ni: int
    nj: int
    ni, nj = X.shape
    x_wall: NDArrayFloat = X[:, wall_j]
    y_wall: NDArrayFloat = Y[:, wall_j]
    n_wall: int = len(x_wall)
    
    d: NDArrayFloat = np.zeros((ni, nj))
    
    for i in range(ni):
        for j in range(nj):
            px: float = float(X[i, j])
            py: float = float(Y[i, j])
            start_idx: int = i
            min_dist: float = float('inf')
            idx_min: int = max(0, start_idx - search_radius)
            idx_max: int = min(n_wall - 1, start_idx + search_radius)
            
            for k in range(idx_min, idx_max):
                ax: float = float(x_wall[k])
                ay: float = float(y_wall[k])
                bx: float = float(x_wall[k + 1])
                by: float = float(y_wall[k + 1])
                dist: float = _point_to_segment_distance(px, py, ax, ay, bx, by)
                min_dist = min(min_dist, dist)
            
            d[i, j] = min_dist
    
    return d


def compute_wall_distance_fast(X: NDArrayFloat, Y: NDArrayFloat, wall_j: int = 0) -> NDArrayFloat:
    """Compute wall distance using cumulative distance along grid lines."""
    ni: int
    nj: int
    ni, nj = X.shape
    d: NDArrayFloat = np.zeros((ni, nj))
    
    for i in range(ni):
        x_line: NDArrayFloat = X[i, :]
        y_line: NDArrayFloat = Y[i, :]
        
        dx: NDArrayFloat = np.diff(x_line)
        dy: NDArrayFloat = np.diff(y_line)
        ds: NDArrayFloat = np.sqrt(dx**2 + dy**2)
        
        if wall_j == 0:
            d[i, 0] = 0.0
            d[i, 1:] = np.cumsum(ds)
        else:
            d[i, wall_j] = 0.0
            d[i, :wall_j] = np.cumsum(ds[:wall_j][::-1])[::-1]
            d[i, wall_j+1:] = np.cumsum(ds[wall_j:])
    
    return d


def compute_metrics(X: NDArrayFloat, Y: NDArrayFloat, wall_j: int = 0) -> GridMetrics:
    """Compute all grid metrics for FVM discretization."""
    Xc: NDArrayFloat
    Yc: NDArrayFloat
    Xc, Yc = compute_cell_centers(X, Y)
    volume: NDArrayFloat = compute_cell_volumes(X, Y)
    Si_x: NDArrayFloat
    Si_y: NDArrayFloat
    Si_mag: NDArrayFloat
    Si_x, Si_y, Si_mag = compute_face_normals_i(X, Y)
    Sj_x: NDArrayFloat
    Sj_y: NDArrayFloat
    Sj_mag: NDArrayFloat
    Sj_x, Sj_y, Sj_mag = compute_face_normals_j(X, Y)
    wall_distance: NDArrayFloat = compute_wall_distance_fast(X, Y, wall_j)
    
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
    
    X: NDArrayFloat
    Y: NDArrayFloat
    _metrics: Optional[GridMetrics] = None
    
    @classmethod
    def from_plot3d(cls, filename: str) -> 'StructuredGrid':
        """Load a grid from a Plot3D file."""
        X: NDArrayFloat
        Y: NDArrayFloat
        X, Y = read_plot3d(filename)
        return cls(X=X, Y=Y)
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.X.shape  # type: ignore[return-value]
    
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
    
    def get_surface_coordinates(self, j: int = 0) -> Tuple[NDArrayFloat, NDArrayFloat]:
        """Get coordinates along a constant-j line."""
        return self.X[:, j], self.Y[:, j]
    
    def get_wake_cut(self, i: int) -> Tuple[NDArrayFloat, NDArrayFloat]:
        """Get coordinates along a constant-i line."""
        return self.X[i, :], self.Y[i, :]
    
    def write_plot3d(self, filename: str, ascii: bool = True) -> None:
        """Write grid to Plot3D format."""
        if ascii:
            self._write_plot3d_ascii(filename)
        else:
            self._write_plot3d_binary(filename)
    
    def _write_plot3d_ascii(self, filename: str) -> None:
        ni: int
        nj: int
        ni, nj = self.shape
        with open(filename, 'w') as f:
            f.write(f"{ni:10d}{nj:10d}\n")
            for val in self.X.flatten(order='F'):
                f.write(f"  {val:18.10E}\n")
            for val in self.Y.flatten(order='F'):
                f.write(f"  {val:18.10E}\n")
    
    def _write_plot3d_binary(self, filename: str) -> None:
        ni: int
        nj: int
        ni, nj = self.shape
        with open(filename, 'wb') as f:
            rec_size: int = 8
            f.write(struct.pack('i', rec_size))
            f.write(struct.pack('ii', ni, nj))
            f.write(struct.pack('i', rec_size))
            
            n_points: int = ni * nj
            rec_size = 2 * n_points * 4
            f.write(struct.pack('i', rec_size))
            f.write(self.X.astype(np.float32).flatten(order='F').tobytes())
            f.write(self.Y.astype(np.float32).flatten(order='F').tobytes())
            f.write(struct.pack('i', rec_size))


def check_grid_quality(X: NDArrayFloat, Y: NDArrayFloat) -> Dict[str, Any]:
    """Compute grid quality metrics."""
    ni: int
    nj: int
    ni, nj = X.shape
    
    dx_i: NDArrayFloat = X[2:, 1:-1] - X[:-2, 1:-1]
    dy_i: NDArrayFloat = Y[2:, 1:-1] - Y[:-2, 1:-1]
    dx_j: NDArrayFloat = X[1:-1, 2:] - X[1:-1, :-2]
    dy_j: NDArrayFloat = Y[1:-1, 2:] - Y[1:-1, :-2]
    
    mag_i: NDArrayFloat = np.sqrt(dx_i**2 + dy_i**2)
    mag_j: NDArrayFloat = np.sqrt(dx_j**2 + dy_j**2)
    
    cross: NDArrayFloat = dx_i * dy_j - dy_i * dx_j
    ortho: NDArrayFloat = cross / (mag_i * mag_j + 1e-12)
    
    dot: NDArrayFloat = dx_i * dx_j + dy_i * dy_j
    cos_angle: NDArrayFloat = dot / (mag_i * mag_j + 1e-12)
    skew_angle: NDArrayFloat = np.abs(np.arccos(np.clip(cos_angle, -1, 1)) - np.pi/2) * 180 / np.pi
    
    volumes: NDArrayFloat = compute_cell_volumes(X, Y)
    
    ds_i: NDArrayFloat = np.sqrt((X[1:, :-1] - X[:-1, :-1])**2 + (Y[1:, :-1] - Y[:-1, :-1])**2)
    ds_j: NDArrayFloat = np.sqrt((X[:-1, 1:] - X[:-1, :-1])**2 + (Y[:-1, 1:] - Y[:-1, :-1])**2)
    aspect: NDArrayFloat = np.maximum(ds_i, ds_j) / (np.minimum(ds_i, ds_j) + 1e-12)
    
    return {
        'orthogonality': ortho,
        'jacobian': cross,
        'min_jacobian': float(np.min(cross)),
        'max_aspect_ratio': float(np.max(aspect)),
        'max_skew_angle': float(np.max(skew_angle)),
        'min_volume': float(np.min(volumes)),
        'max_volume': float(np.max(volumes))
    }
