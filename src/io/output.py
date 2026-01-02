"""
VTK Output Writer for CFD Solutions.

This module provides functionality to write CFD solutions to VTK files
for visualization in ParaView or other VTK-compatible viewers.

Supports:
- Legacy VTK ASCII format (structured grid)
- Scalar fields: pressure, turbulent viscosity, Mach number
- Vector fields: velocity

File Format:
    Uses VTK Legacy ASCII format (.vtk) which is widely supported
    and easy to debug. For larger grids, consider using pyevtk for
    binary output.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
import os

from src.constants import NGHOST


def write_vtk(filename: str, 
              X: np.ndarray, 
              Y: np.ndarray, 
              Q: np.ndarray,
              beta: float = 1.0,
              additional_scalars: Optional[Dict[str, np.ndarray]] = None,
              additional_vectors: Optional[Dict[str, np.ndarray]] = None) -> str:
    """
    Write solution to VTK file for visualization.
    
    Parameters
    ----------
    filename : str
        Output filename (will add .vtk extension if not present).
    X, Y : ndarray, shape (NI+1, NJ+1)
        Node coordinates.
    Q : ndarray, shape (NI, NJ, 4) or (NI+2, NJ+2, 4)
        State vector [p, u, v, ν̃]. Can include ghost cells (will be stripped).
    beta : float
        Artificial compressibility parameter (for Mach calculation).
    additional_scalars : dict, optional
        Additional scalar fields to write. Keys are field names, values are
        arrays of shape (NI, NJ).
    additional_vectors : dict, optional
        Additional vector fields to write. Keys are field names, values are
        arrays of shape (NI, NJ, 2) or (NI, NJ, 3).
        
    Returns
    -------
    str
        Path to the written file.
        
    Notes
    -----
    The VTK file uses STRUCTURED_GRID format with CELL_DATA for the solution
    variables (since FVM stores cell-averaged values).
    """
    # Ensure .vtk extension
    if not filename.endswith('.vtk'):
        filename = filename + '.vtk'
    
    # Create output directory if needed
    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get grid dimensions
    NI_nodes, NJ_nodes = X.shape
    NI = NI_nodes - 1  # Number of cells in i
    NJ = NJ_nodes - 1  # Number of cells in j
    
    # Handle ghost cells in Q (NGHOST ghost layers on each side)
    if Q.shape[0] == NI + 2*NGHOST and Q.shape[1] == NJ + 2*NGHOST:
        # Strip ghost cells (NGHOST ghosts on each side)
        Q = Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
    elif Q.shape[0] != NI or Q.shape[1] != NJ:
        raise ValueError(f"Q shape {Q.shape} incompatible with grid ({NI}, {NJ}), expected interior ({NI}, {NJ}) or with ghosts ({NI + 2*NGHOST}, {NJ + 2*NGHOST})")
    
    # Extract solution components
    p = Q[:, :, 0]
    u = Q[:, :, 1]
    v = Q[:, :, 2]
    nu_t = Q[:, :, 3]
    
    # Compute derived quantities
    velocity_mag = np.sqrt(u**2 + v**2)
    acoustic_speed = np.sqrt(u**2 + v**2 + beta)
    mach = velocity_mag / acoustic_speed
    
    # Write VTK file
    with open(filename, 'w') as f:
        # Header
        f.write("# vtk DataFile Version 3.0\n")
        f.write("CFD Solution - 2D RANS Solver\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_GRID\n")
        
        # Dimensions (VTK uses i, j, k ordering)
        # For 2D, we use k=1
        f.write(f"DIMENSIONS {NI_nodes} {NJ_nodes} 1\n")
        
        # Points (node coordinates)
        n_points = NI_nodes * NJ_nodes
        f.write(f"POINTS {n_points} float\n")
        
        # VTK expects points in Fortran order (j varies fastest for our grid)
        # But for structured grid, we iterate i first, then j
        for j in range(NJ_nodes):
            for i in range(NI_nodes):
                f.write(f"{X[i, j]:.10e} {Y[i, j]:.10e} 0.0\n")
        
        # Cell data
        n_cells = NI * NJ
        f.write(f"\nCELL_DATA {n_cells}\n")
        
        # Pressure
        _write_scalar_field(f, "Pressure", p)
        
        # Velocity (vector)
        _write_vector_field(f, "Velocity", u, v)
        
        # Turbulent viscosity
        _write_scalar_field(f, "TurbulentViscosity", nu_t)
        
        # Mach number
        _write_scalar_field(f, "MachNumber", mach)
        
        # Velocity magnitude
        _write_scalar_field(f, "VelocityMagnitude", velocity_mag)
        
        # Additional scalars
        if additional_scalars:
            for name, data in additional_scalars.items():
                if data.shape != (NI, NJ):
                    raise ValueError(f"Scalar '{name}' has wrong shape: {data.shape}")
                _write_scalar_field(f, name, data)
        
        # Additional vectors
        if additional_vectors:
            for name, data in additional_vectors.items():
                if data.shape[:2] != (NI, NJ):
                    raise ValueError(f"Vector '{name}' has wrong shape: {data.shape}")
                if data.ndim == 2:
                    raise ValueError(f"Vector '{name}' must have shape (NI, NJ, 2) or (NI, NJ, 3)")
                vx = data[:, :, 0]
                vy = data[:, :, 1]
                _write_vector_field(f, name, vx, vy)
    
    return filename


def _write_scalar_field(f, name: str, data: np.ndarray):
    """Write a scalar field to VTK file."""
    f.write(f"SCALARS {name} float 1\n")
    f.write("LOOKUP_TABLE default\n")
    
    NI, NJ = data.shape
    for j in range(NJ):
        for i in range(NI):
            f.write(f"{data[i, j]:.10e}\n")


def _write_vector_field(f, name: str, vx: np.ndarray, vy: np.ndarray, vz: Optional[np.ndarray] = None):
    """Write a vector field to VTK file."""
    f.write(f"VECTORS {name} float\n")
    
    NI, NJ = vx.shape
    for j in range(NJ):
        for i in range(NI):
            z_val = vz[i, j] if vz is not None else 0.0
            f.write(f"{vx[i, j]:.10e} {vy[i, j]:.10e} {z_val:.10e}\n")


def write_vtk_series(base_filename: str,
                     X: np.ndarray,
                     Y: np.ndarray,
                     solutions: Dict[int, np.ndarray],
                     beta: float = 1.0) -> str:
    """
    Write a time series of solutions to VTK files.
    
    Creates a .vtk.series file for ParaView to load the time series.
    
    Parameters
    ----------
    base_filename : str
        Base filename (without extension). Files will be named
        base_filename_XXXX.vtk where XXXX is the iteration number.
    X, Y : ndarray
        Node coordinates.
    solutions : dict
        Dictionary mapping iteration number to Q array.
    beta : float
        Artificial compressibility parameter.
        
    Returns
    -------
    str
        Path to the .vtk.series file.
    """
    output_dir = os.path.dirname(base_filename) or '.'
    base_name = os.path.basename(base_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Write individual VTK files
    files_written = []
    for iter_num, Q in sorted(solutions.items()):
        vtk_filename = f"{base_filename}_{iter_num:06d}.vtk"
        write_vtk(vtk_filename, X, Y, Q, beta)
        files_written.append((iter_num, os.path.basename(vtk_filename)))
    
    # Write .vtk.series file for ParaView
    series_filename = f"{base_filename}.vtk.series"
    with open(series_filename, 'w') as f:
        f.write('{\n')
        f.write('  "file-series-version" : "1.0",\n')
        f.write('  "files" : [\n')
        
        for idx, (iter_num, vtk_file) in enumerate(files_written):
            comma = "," if idx < len(files_written) - 1 else ""
            f.write(f'    {{ "name" : "{vtk_file}", "time" : {iter_num} }}{comma}\n')
        
        f.write('  ]\n')
        f.write('}\n')
    
    return series_filename


class VTKWriter:
    """
    Class-based VTK writer for managing output during simulation.
    
    Example
    -------
    >>> writer = VTKWriter("output/solution", X, Y, beta=10.0)
    >>> for n in range(n_steps):
    >>>     Q = solver.step()
    >>>     if n % save_interval == 0:
    >>>         writer.write(Q, iteration=n)
    >>> writer.finalize()  # Writes .vtk.series file
    """
    
    def __init__(self, 
                 base_filename: str,
                 X: np.ndarray,
                 Y: np.ndarray,
                 beta: float = 1.0):
        """
        Initialize VTK writer.
        
        Parameters
        ----------
        base_filename : str
            Base path for output files.
        X, Y : ndarray
            Node coordinates.
        beta : float
            Artificial compressibility parameter.
        """
        self.base_filename = base_filename
        self.X = X
        self.Y = Y
        self.beta = beta
        self.solutions: Dict[int, str] = {}  # Maps iteration to filename
    
    def write(self, 
              Q: np.ndarray, 
              iteration: int = 0,
              additional_scalars: Optional[Dict[str, np.ndarray]] = None,
              additional_vectors: Optional[Dict[str, np.ndarray]] = None) -> str:
        """
        Write solution for a single iteration.
        
        Parameters
        ----------
        Q : ndarray
            State vector.
        iteration : int
            Iteration/time step number.
        additional_scalars : dict, optional
            Extra scalar fields.
        additional_vectors : dict, optional
            Extra vector fields.
            
        Returns
        -------
        str
            Path to written file.
        """
        filename = f"{self.base_filename}_{iteration:06d}.vtk"
        write_vtk(filename, self.X, self.Y, Q, self.beta,
                  additional_scalars, additional_vectors)
        self.solutions[iteration] = filename
        return filename
    
    def finalize(self) -> str:
        """
        Write .vtk.series file for ParaView time series loading.
        
        Returns
        -------
        str
            Path to the .vtk.series file.
        """
        if not self.solutions:
            return ""
        
        output_dir = os.path.dirname(self.base_filename) or '.'
        
        series_filename = f"{self.base_filename}.vtk.series"
        with open(series_filename, 'w') as f:
            f.write('{\n')
            f.write('  "file-series-version" : "1.0",\n')
            f.write('  "files" : [\n')
            
            sorted_items = sorted(self.solutions.items())
            for idx, (iter_num, vtk_file) in enumerate(sorted_items):
                comma = "," if idx < len(sorted_items) - 1 else ""
                vtk_basename = os.path.basename(vtk_file)
                f.write(f'    {{ "name" : "{vtk_basename}", "time" : {iter_num} }}{comma}\n')
            
            f.write('  ]\n')
            f.write('}\n')
        
        return series_filename

