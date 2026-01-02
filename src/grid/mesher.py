"""
Construct2D wrapper for automated C-grid generation.
"""

import os
import subprocess
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


@dataclass
class GridOptions:
    """Configuration for Construct2D grid generation."""
    n_surface: int = 250
    le_spacing: float = 0.001
    te_spacing: float = 0.001
    farfield_radius: float = 15.0
    n_wake: int = 50
    farfield_dist: float = 2.0
    wake_length_ratio: float = 1.0
    wake_init_ratio: float = 0.1
    
    n_normal: int = 100
    solver: str = 'HYPR'
    topology: str = 'CGRD'
    y_plus: float = 1.0
    reynolds: float = 1e6
    chord_fraction: float = 0.5
    max_first_cell: float = 0.001
    
    alfa: float = 1.0
    epsi: float = 15.0
    epse: float = 0.0
    funi: float = 0.01
    asmt: int = 1
    
    max_steps: int = 100
    final_steps: int = 50
    nrmt: int = 5
    nrmb: int = 5
    
    grid_dim: int = 2
    n_planes: int = 1
    plane_delta: float = 1.0


class Construct2DError(Exception):
    """Exception for Construct2D-related errors."""
    pass


class Construct2DWrapper:
    """Python wrapper for Construct2D grid generator."""
    
    def __init__(self, binary_path: str):
        self.binary_path = Path(binary_path).resolve()
        if not self.binary_path.exists():
            raise Construct2DError(f"Construct2D binary not found: {self.binary_path}")
        if not os.access(self.binary_path, os.X_OK):
            raise Construct2DError(f"Construct2D binary is not executable: {self.binary_path}")
    
    def _write_grid_options(self, name: str, opts: GridOptions) -> str:
        """Generate the grid_options.in namelist file content."""
        funi = opts.funi if opts.topology == 'CGRD' else 0.2
        
        import math
        y_plus_eff = opts.y_plus
        if opts.max_first_cell > 0 and opts.reynolds > 0:
            y_plus_from_max = opts.max_first_cell * math.sqrt(opts.reynolds) / 5.0
            if y_plus_from_max < opts.y_plus:
                y_plus_eff = y_plus_from_max
        
        return f"""&SOPT
  nsrf = {opts.n_surface}
  lesp = {opts.le_spacing:.6e}
  tesp = {opts.te_spacing:.6e}
  radi = {opts.farfield_radius:.6f}
  nwke = {opts.n_wake}
  fdst = {opts.farfield_dist:.6f}
  fwkl = {opts.wake_length_ratio:.6f}
  fwki = {opts.wake_init_ratio:.6f}
/
&VOPT
  name = '{name}'
  jmax = {opts.n_normal}
  slvr = '{opts.solver}'
  topo = '{opts.topology}'
  ypls = {y_plus_eff:.6f}
  recd = {opts.reynolds:.6e}
  cfrc = {opts.chord_fraction:.6f}
  stp1 = {opts.max_steps}
  stp2 = {opts.final_steps}
  nrmt = {opts.nrmt}
  nrmb = {opts.nrmb}
  alfa = {opts.alfa:.6f}
  epsi = {opts.epsi:.6f}
  epse = {opts.epse:.6f}
  funi = {funi:.6f}
  asmt = {opts.asmt}
/
&OOPT
  gdim = {opts.grid_dim}
  npln = {opts.n_planes}
  dpln = {opts.plane_delta:.6f}
/
"""
    
    def generate(
        self, 
        airfoil_file: str, 
        options: Optional[GridOptions] = None,
        working_dir: Optional[str] = None,
        keep_files: bool = False,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a C-grid around an airfoil."""
        if options is None:
            options = GridOptions()
        
        airfoil_path = Path(airfoil_file).resolve()
        if not airfoil_path.exists():
            raise Construct2DError(f"Airfoil file not found: {airfoil_path}")
        
        name = airfoil_path.stem
        
        if working_dir is None:
            work_dir = Path(tempfile.mkdtemp(prefix="construct2d_"))
            cleanup = not keep_files
        else:
            work_dir = Path(working_dir).resolve()
            work_dir.mkdir(parents=True, exist_ok=True)
            cleanup = False
        
        try:
            local_airfoil = work_dir / airfoil_path.name
            if airfoil_path.resolve() != local_airfoil.resolve():
                shutil.copy2(airfoil_path, local_airfoil)
            
            options_content = self._write_grid_options(name, options)
            options_file = work_dir / "grid_options.in"
            options_file.write_text(options_content)
            
            if verbose:
                print(f"Generating grid for {name}...")
                print(f"  Surface points: {options.n_surface}")
                print(f"  Normal points:  {options.n_normal}")
                print(f"  Target y+:      {options.y_plus}")
                print(f"  Reynolds:       {options.reynolds:.2e}")
                print(f"  Topology:       {options.topology}")
            
            commands = "\nGRID\nSMTH\nQUIT\n"
            
            result = subprocess.run(
                [str(self.binary_path), str(local_airfoil)],
                input=commands,
                capture_output=True,
                text=True,
                cwd=work_dir,
                timeout=120
            )
            
            output = result.stdout
            
            output_file = work_dir / f"{name}.p3d"
            if not output_file.exists():
                raise Construct2DError(
                    f"Expected output file not found: {output_file}\n"
                    f"Construct2D output:\n{output}"
                )
            
            if verbose:
                for line in output.split('\n'):
                    if 'First layer wall distance' in line:
                        print(f"  {line.strip()}")
                    elif 'Max skew angle' in line:
                        print(f"  {line.strip()}")
            
            X, Y = self._read_plot3d(output_file)
            
            if verbose:
                print(f"  Grid size: {X.shape[0]} x {X.shape[1]}")
            
            return X, Y
            
        finally:
            if cleanup:
                shutil.rmtree(work_dir, ignore_errors=True)
    
    def _read_plot3d(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Read a 2D Plot3D structured grid file (ASCII format)."""
        with open(filename, 'r') as f:
            tokens = f.read().split()
        
        ptr = 0
        ni = int(tokens[ptr]); ptr += 1
        nj = int(tokens[ptr]); ptr += 1
        
        n_points = ni * nj
        
        x_flat = np.array([float(tokens[ptr + i]) for i in range(n_points)])
        ptr += n_points
        y_flat = np.array([float(tokens[ptr + i]) for i in range(n_points)])
        
        X = x_flat.reshape((ni, nj), order='F')
        Y = y_flat.reshape((ni, nj), order='F')
        
        return X, Y


def estimate_first_cell_height(reynolds: float, y_plus: float = 1.0, chord: float = 1.0) -> float:
    """Estimate first cell height for target y+ using flat-plate correlation."""
    cf = 0.058 * reynolds**(-0.2)
    tau_w = 0.5 * cf
    u_tau = np.sqrt(tau_w)
    dy = y_plus / (reynolds * u_tau) * chord
    return dy
