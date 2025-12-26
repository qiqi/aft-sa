"""
Construct2D Wrapper for automated C-grid generation.

This module provides a Python interface to the Construct2D Fortran tool
for generating high-quality hyperbolic C-grids around airfoils.
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
    """Configuration options for Construct2D grid generation."""
    
    # Surface grid options (SOPT)
    n_surface: int = 250            # Number of points on airfoil surface
    le_spacing: float = 0.004       # Leading edge point spacing
    te_spacing: float = 0.004       # Trailing edge point spacing  
    farfield_radius: float = 15.0   # Farfield radius in chord lengths
    n_wake: int = 50                # Points along wake for C-grid
    farfield_dist: float = 1.0      # O-grid farfield spacing parameter
    wake_length_ratio: float = 1.0  # C-grid farfield wake length ratio
    wake_init_ratio: float = 10.0   # C-grid farfield wake initial length ratio
    
    # Volume grid options (VOPT)
    n_normal: int = 100             # Points in wall-normal direction
    solver: str = 'HYPR'            # Solver type: 'HYPR' (hyperbolic) or 'ELLP' (elliptic)
    topology: str = 'CGRD'          # Grid topology: 'CGRD' (C-grid) or 'OGRD' (O-grid)
    y_plus: float = 1.0             # Target y+ at wall
    reynolds: float = 1e6           # Chord Reynolds number for y+ calculation
    chord_fraction: float = 0.5     # Chord fraction for y+ reference length
    
    # Hyperbolic solver options
    alfa: float = 1.0               # Implicitness parameter (>=0.5)
    epsi: float = 15.0              # Implicit smoothing parameter
    epse: float = 0.0               # Explicit smoothing parameter
    funi: float = 0.01              # Farfield uniformness (0-1, 0.01 for C-grid)
    asmt: int = 20                  # Cell area smoothing steps
    
    # Elliptic solver options
    max_steps: int = 1000           # Initial grid smoothing steps
    final_steps: int = 20           # Final grid smoothing steps
    nrmt: int = 1                   # First top point to enforce surface-normal grid
    nrmb: int = 1                   # First bottom point to enforce surface-normal grid
    
    # Output options (OOPT)
    grid_dim: int = 2               # Output grid dimension (2 or 3)
    n_planes: int = 1               # Number of planes for 3D output
    plane_delta: float = 1.0        # Plane spacing for 3D output


class Construct2DError(Exception):
    """Exception raised for Construct2D-related errors."""
    pass


class Construct2DWrapper:
    """
    Python wrapper for the Construct2D structured grid generator.
    
    Construct2D generates high-quality hyperbolic or elliptic grids
    for 2D airfoil simulations.
    
    Example
    -------
    >>> wrapper = Construct2DWrapper("bin/construct2d")
    >>> opts = GridOptions(n_surface=300, n_normal=120, y_plus=0.5, reynolds=3e6)
    >>> X, Y = wrapper.generate("data/naca0012.dat", opts)
    """
    
    def __init__(self, binary_path: str):
        """
        Initialize the Construct2D wrapper.
        
        Parameters
        ----------
        binary_path : str
            Path to the construct2d executable.
        """
        self.binary_path = Path(binary_path).resolve()
        if not self.binary_path.exists():
            raise Construct2DError(f"Construct2D binary not found: {self.binary_path}")
        if not os.access(self.binary_path, os.X_OK):
            raise Construct2DError(f"Construct2D binary is not executable: {self.binary_path}")
    
    def _write_grid_options(self, name: str, opts: GridOptions) -> str:
        """Generate the grid_options.in namelist file content."""
        
        # Adjust funi for topology
        funi = opts.funi if opts.topology == 'CGRD' else 0.2
        
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
  ypls = {opts.y_plus:.6f}
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
        output_dir: Optional[str] = None,
        keep_files: bool = False,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a C-grid around an airfoil.
        
        Parameters
        ----------
        airfoil_file : str
            Path to the airfoil coordinate file (XFOIL Selig format).
        options : GridOptions, optional
            Grid generation options. Uses defaults if not provided.
        output_dir : str, optional
            Directory for output files. Uses temp dir if not provided.
        keep_files : bool
            If True, keep intermediate files (namelist, stats, etc.).
        verbose : bool
            If True, print progress information.
            
        Returns
        -------
        X : ndarray
            X-coordinates of grid nodes, shape (ni, nj).
        Y : ndarray
            Y-coordinates of grid nodes, shape (ni, nj).
        """
        if options is None:
            options = GridOptions()
        
        airfoil_path = Path(airfoil_file).resolve()
        if not airfoil_path.exists():
            raise Construct2DError(f"Airfoil file not found: {airfoil_path}")
        
        # Extract project name from airfoil filename
        name = airfoil_path.stem
        
        # Determine working directory
        if output_dir is None:
            work_dir = Path(tempfile.mkdtemp(prefix="construct2d_"))
            cleanup = not keep_files
        else:
            work_dir = Path(output_dir).resolve()
            work_dir.mkdir(parents=True, exist_ok=True)
            cleanup = False
        
        try:
            # Copy airfoil file to working directory
            local_airfoil = work_dir / airfoil_path.name
            shutil.copy2(airfoil_path, local_airfoil)
            
            # Write grid_options.in
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
            
            # Prepare input commands for interactive mode
            # The tool may ask for confirmation if topology doesn't match TE type
            # We answer 'y' to any such prompts, then: GRID -> SMTH -> QUIT
            # Add extra 'y' answers to handle potential topology confirmation
            commands = "y\nGRID\nSMTH\nQUIT\n"
            
            # Run Construct2D
            result = subprocess.run(
                [str(self.binary_path), str(local_airfoil)],
                input=commands,
                capture_output=True,
                text=True,
                cwd=work_dir
            )
            
            if result.returncode != 0:
                raise Construct2DError(
                    f"Construct2D failed with return code {result.returncode}\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )
            
            # Check for output file
            output_file = work_dir / f"{name}.p3d"
            if not output_file.exists():
                raise Construct2DError(
                    f"Expected output file not found: {output_file}\n"
                    f"Construct2D output:\n{result.stdout}"
                )
            
            # Parse wall distance from output
            if verbose:
                for line in result.stdout.split('\n'):
                    if 'First layer wall distance' in line:
                        print(f"  {line.strip()}")
                    elif 'Max skew angle' in line:
                        print(f"  {line.strip()}")
            
            # Read the grid
            X, Y = self._read_plot3d(output_file)
            
            if verbose:
                print(f"  Grid size: {X.shape[0]} x {X.shape[1]}")
            
            # Copy output files if keeping
            if keep_files and output_dir is not None:
                # Files are already in output_dir
                pass
            
            return X, Y
            
        finally:
            if cleanup:
                shutil.rmtree(work_dir, ignore_errors=True)
    
    def _read_plot3d(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read a 2D Plot3D structured grid file (ASCII format).
        
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
        
        # First line: ni nj (for 2D grid without block count)
        # Or could be: nblocks on first line, then ni nj [nk] on second
        # Construct2D 2D output: "ni nj" on first line
        
        ni = int(tokens[ptr]); ptr += 1
        nj = int(tokens[ptr]); ptr += 1
        
        n_points = ni * nj
        
        # Read X coordinates (Fortran column-major order)
        x_flat = np.array([float(tokens[ptr + i]) for i in range(n_points)])
        ptr += n_points
        
        # Read Y coordinates
        y_flat = np.array([float(tokens[ptr + i]) for i in range(n_points)])
        
        # Reshape to (ni, nj) - Fortran order
        X = x_flat.reshape((ni, nj), order='F')
        Y = y_flat.reshape((ni, nj), order='F')
        
        return X, Y


def estimate_first_cell_height(reynolds: float, y_plus: float = 1.0, chord: float = 1.0) -> float:
    """
    Estimate the first cell height for a given y+ target.
    
    Uses flat-plate skin friction correlation (Schlichting).
    
    Parameters
    ----------
    reynolds : float
        Chord-based Reynolds number.
    y_plus : float
        Target y+ value.
    chord : float
        Chord length.
        
    Returns
    -------
    dy : float
        First cell height.
    """
    # Skin friction estimate: Cf = 0.058 * Re^(-0.2)
    cf = 0.058 * reynolds**(-0.2)
    
    # Wall shear stress (non-dimensional): tau_w = 0.5 * Cf
    tau_w = 0.5 * cf
    
    # Friction velocity: u_tau = sqrt(tau_w / rho) = sqrt(tau_w) for rho=1
    u_tau = np.sqrt(tau_w)
    
    # y+ = y * u_tau / nu, nu = 1/Re
    # y = y+ * nu / u_tau = y+ / (Re * u_tau)
    dy = y_plus / (reynolds * u_tau) * chord
    
    return dy

