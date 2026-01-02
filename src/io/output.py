"""
VTK output writer for CFD solutions.
"""

import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Dict, Optional, Any, List, TextIO
import os

from src.constants import NGHOST

NDArrayFloat = npt.NDArray[np.floating]


def write_vtk(filename: str, 
              X: NDArrayFloat, 
              Y: NDArrayFloat, 
              Q: NDArrayFloat,
              beta: float = 1.0,
              additional_scalars: Optional[Dict[str, NDArrayFloat]] = None,
              additional_vectors: Optional[Dict[str, NDArrayFloat]] = None) -> str:
    """Write solution to VTK file for visualization."""
    if not filename.endswith('.vtk'):
        filename = filename + '.vtk'
    
    output_dir: str = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    NI_nodes: int
    NJ_nodes: int
    NI_nodes, NJ_nodes = X.shape
    NI: int = NI_nodes - 1
    NJ: int = NJ_nodes - 1
    
    if Q.shape[0] == NI + 2*NGHOST and Q.shape[1] == NJ + 2*NGHOST:
        Q = Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
    elif Q.shape[0] != NI or Q.shape[1] != NJ:
        raise ValueError(f"Q shape {Q.shape} incompatible with grid ({NI}, {NJ})")
    
    p: NDArrayFloat = Q[:, :, 0]
    u: NDArrayFloat = Q[:, :, 1]
    v: NDArrayFloat = Q[:, :, 2]
    nu_t: NDArrayFloat = Q[:, :, 3]
    
    velocity_mag: NDArrayFloat = np.sqrt(u**2 + v**2)
    acoustic_speed: NDArrayFloat = np.sqrt(u**2 + v**2 + beta)
    mach: NDArrayFloat = velocity_mag / acoustic_speed
    
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("CFD Solution - 2D RANS Solver\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_GRID\n")
        f.write(f"DIMENSIONS {NI_nodes} {NJ_nodes} 1\n")
        
        n_points = NI_nodes * NJ_nodes
        f.write(f"POINTS {n_points} float\n")
        
        for j in range(NJ_nodes):
            for i in range(NI_nodes):
                f.write(f"{X[i, j]:.10e} {Y[i, j]:.10e} 0.0\n")
        
        n_cells = NI * NJ
        f.write(f"\nCELL_DATA {n_cells}\n")
        
        _write_scalar_field(f, "Pressure", p)
        _write_vector_field(f, "Velocity", u, v)
        _write_scalar_field(f, "TurbulentViscosity", nu_t)
        _write_scalar_field(f, "MachNumber", mach)
        _write_scalar_field(f, "VelocityMagnitude", velocity_mag)
        
        if additional_scalars:
            for name, data in additional_scalars.items():
                if data.shape != (NI, NJ):
                    raise ValueError(f"Scalar '{name}' has wrong shape: {data.shape}")
                _write_scalar_field(f, name, data)
        
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


def _write_scalar_field(f: TextIO, name: str, data: NDArrayFloat) -> None:
    """Write a scalar field to VTK file."""
    f.write(f"SCALARS {name} float 1\n")
    f.write("LOOKUP_TABLE default\n")
    
    NI: int
    NJ: int
    NI, NJ = data.shape
    for j in range(NJ):
        for i in range(NI):
            f.write(f"{data[i, j]:.10e}\n")


def _write_vector_field(f: TextIO, name: str, vx: NDArrayFloat, vy: NDArrayFloat, 
                        vz: Optional[NDArrayFloat] = None) -> None:
    """Write a vector field to VTK file."""
    f.write(f"VECTORS {name} float\n")
    
    NI: int
    NJ: int
    NI, NJ = vx.shape
    for j in range(NJ):
        for i in range(NI):
            z_val: float = float(vz[i, j]) if vz is not None else 0.0
            f.write(f"{vx[i, j]:.10e} {vy[i, j]:.10e} {z_val:.10e}\n")


def write_vtk_series(base_filename: str,
                     X: NDArrayFloat,
                     Y: NDArrayFloat,
                     solutions: Dict[int, NDArrayFloat],
                     beta: float = 1.0) -> str:
    """Write a time series of solutions to VTK files."""
    output_dir: str = os.path.dirname(base_filename) or '.'
    base_name: str = os.path.basename(base_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    
    files_written: List[tuple[int, str]] = []
    for iter_num, Q in sorted(solutions.items()):
        vtk_filename: str = f"{base_filename}_{iter_num:06d}.vtk"
        write_vtk(vtk_filename, X, Y, Q, beta)
        files_written.append((iter_num, os.path.basename(vtk_filename)))
    
    series_filename: str = f"{base_filename}.vtk.series"
    with open(series_filename, 'w') as f:
        f.write('{\n')
        f.write('  "file-series-version" : "1.0",\n')
        f.write('  "files" : [\n')
        
        for idx, (iter_num, vtk_file) in enumerate(files_written):
            comma: str = "," if idx < len(files_written) - 1 else ""
            f.write(f'    {{ "name" : "{vtk_file}", "time" : {iter_num} }}{comma}\n')
        
        f.write('  ]\n')
        f.write('}\n')
    
    return series_filename


class VTKWriter:
    """Class-based VTK writer for managing output during simulation."""
    
    base_filename: str
    X: NDArrayFloat
    Y: NDArrayFloat
    beta: float
    solutions: Dict[int, str]
    
    def __init__(self, 
                 base_filename: str,
                 X: NDArrayFloat,
                 Y: NDArrayFloat,
                 beta: float = 1.0) -> None:
        self.base_filename = base_filename
        self.X = X
        self.Y = Y
        self.beta = beta
        self.solutions = {}
    
    def write(self, 
              Q: NDArrayFloat, 
              iteration: int = 0,
              additional_scalars: Optional[Dict[str, NDArrayFloat]] = None,
              additional_vectors: Optional[Dict[str, NDArrayFloat]] = None) -> str:
        """Write solution for a single iteration."""
        filename: str = f"{self.base_filename}_{iteration:06d}.vtk"
        write_vtk(filename, self.X, self.Y, Q, self.beta,
                  additional_scalars, additional_vectors)
        self.solutions[iteration] = filename
        return filename
    
    def finalize(self) -> str:
        """Write .vtk.series file for ParaView."""
        if not self.solutions:
            return ""
        
        output_dir: str = os.path.dirname(self.base_filename) or '.'
        
        series_filename: str = f"{self.base_filename}.vtk.series"
        with open(series_filename, 'w') as f:
            f.write('{\n')
            f.write('  "file-series-version" : "1.0",\n')
            f.write('  "files" : [\n')
            
            sorted_items: List[tuple[int, str]] = sorted(self.solutions.items())
            for idx, (iter_num, vtk_file) in enumerate(sorted_items):
                comma: str = "," if idx < len(sorted_items) - 1 else ""
                vtk_basename: str = os.path.basename(vtk_file)
                f.write(f'    {{ "name" : "{vtk_basename}", "time" : {iter_num} }}{comma}\n')
            
            f.write('  ]\n')
            f.write('}\n')
        
        return series_filename
