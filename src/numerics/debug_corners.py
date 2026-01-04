"""
Debug instrumentation for farfield corner cells.

Tracks state values and fluxes at (I=0, J=Jmax) and (I=Imax, J=Jmax) corners
to diagnose divergence originating from farfield boundaries.
"""

import numpy as np
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..constants import NGHOST


@dataclass
class CornerDebugData:
    """Debug data for a single iteration."""
    iteration: int
    cfl: float
    residual: float
    
    # Corner 1: I=0, J=Jmax (lower-left farfield corner in C-grid)
    c1_i: int
    c1_j: int
    c1_p: float
    c1_u: float
    c1_v: float
    c1_nu: float
    c1_R_p: float  # Residual components
    c1_R_u: float
    c1_R_v: float
    c1_R_nu: float
    
    # Corner 2: I=Imax, J=Jmax (lower-right farfield corner in C-grid)
    c2_i: int
    c2_j: int
    c2_p: float
    c2_u: float
    c2_v: float
    c2_nu: float
    c2_R_p: float
    c2_R_u: float
    c2_R_v: float
    c2_R_nu: float
    
    # Flux info for corner 1
    c1_Fi_p: float = 0.0  # I-direction flux
    c1_Fi_u: float = 0.0
    c1_Fi_v: float = 0.0
    c1_Fi_nu: float = 0.0
    c1_Fj_p: float = 0.0  # J-direction flux
    c1_Fj_u: float = 0.0
    c1_Fj_v: float = 0.0
    c1_Fj_nu: float = 0.0
    
    # Flux info for corner 2
    c2_Fi_p: float = 0.0
    c2_Fi_u: float = 0.0
    c2_Fi_v: float = 0.0
    c2_Fi_nu: float = 0.0
    c2_Fj_p: float = 0.0
    c2_Fj_u: float = 0.0
    c2_Fj_v: float = 0.0
    c2_Fj_nu: float = 0.0
    
    # Neighbor values for corner 1
    c1_neighbor_im1_p: float = 0.0  # Q[i-1, j]
    c1_neighbor_ip1_p: float = 0.0  # Q[i+1, j]
    c1_neighbor_jm1_p: float = 0.0  # Q[i, j-1]
    c1_neighbor_jp1_p: float = 0.0  # Q[i, j+1] (ghost)
    
    # Neighbor values for corner 2
    c2_neighbor_im1_p: float = 0.0
    c2_neighbor_ip1_p: float = 0.0
    c2_neighbor_jm1_p: float = 0.0
    c2_neighbor_jp1_p: float = 0.0


class CornerDebugger:
    """Debugger for tracking farfield corner cell behavior."""
    
    def __init__(self, NI: int, NJ: int, output_dir: str = "output/debug"):
        """
        Initialize corner debugger.
        
        Args:
            NI: Number of cells in I direction
            NJ: Number of cells in J direction
            output_dir: Directory for output files
        """
        self.NI = NI
        self.NJ = NJ
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Corner cell indices (interior cell indices, not including ghosts)
        # For Q array with ghosts: Q[NGHOST:-NGHOST, NGHOST:-NGHOST] is interior
        # Corner 1: I=0 (first interior), J=NJ-1 (last interior before farfield)
        # Corner 2: I=NI-1 (last interior), J=NJ-1
        self.c1_i = 0
        self.c1_j = NJ - 1
        self.c2_i = NI - 1
        self.c2_j = NJ - 1
        
        # With ghosts:
        # Q array indices for corners
        self.c1_ig = NGHOST + self.c1_i
        self.c1_jg = NGHOST + self.c1_j
        self.c2_ig = NGHOST + self.c2_i
        self.c2_jg = NGHOST + self.c2_j
        
        self.history: List[CornerDebugData] = []
        
        print("Corner Debugger initialized:")
        print(f"  Grid size: {NI} x {NJ} cells")
        print(f"  Corner 1 (I=0, J=Jmax): cell ({self.c1_i}, {self.c1_j}), "
              f"array idx ({self.c1_ig}, {self.c1_jg})")
        print(f"  Corner 2 (I=Imax, J=Jmax): cell ({self.c2_i}, {self.c2_j}), "
              f"array idx ({self.c2_ig}, {self.c2_jg})")
    
    def record(self, Q: np.ndarray, R: np.ndarray, iteration: int, 
               cfl: float, residual: float,
               flux_i: Optional[np.ndarray] = None,
               flux_j: Optional[np.ndarray] = None) -> CornerDebugData:
        """
        Record debug data for current iteration.
        
        Args:
            Q: State array with ghost cells [NI+2*NGHOST, NJ+2*NGHOST, 4]
            R: Residual array (interior only) [NI, NJ, 4]
            iteration: Current iteration number
            cfl: Current CFL number
            residual: Current residual RMS
            flux_i: Optional I-direction flux array
            flux_j: Optional J-direction flux array
        """
        # Extract corner 1 values
        c1_Q = Q[self.c1_ig, self.c1_jg, :]
        c1_R = R[self.c1_i, self.c1_j, :]
        
        # Extract corner 2 values
        c2_Q = Q[self.c2_ig, self.c2_jg, :]
        c2_R = R[self.c2_i, self.c2_j, :]
        
        # Get neighbor values for corner 1
        c1_Qim1 = Q[self.c1_ig - 1, self.c1_jg, :] if self.c1_ig > 0 else np.zeros(4)
        c1_Qip1 = Q[self.c1_ig + 1, self.c1_jg, :]
        c1_Qjm1 = Q[self.c1_ig, self.c1_jg - 1, :]
        c1_Qjp1 = Q[self.c1_ig, self.c1_jg + 1, :]  # Ghost cell
        
        # Get neighbor values for corner 2
        c2_Qim1 = Q[self.c2_ig - 1, self.c2_jg, :]
        c2_Qip1 = Q[self.c2_ig + 1, self.c2_jg, :] if self.c2_ig + 1 < Q.shape[0] else np.zeros(4)
        c2_Qjm1 = Q[self.c2_ig, self.c2_jg - 1, :]
        c2_Qjp1 = Q[self.c2_ig, self.c2_jg + 1, :]  # Ghost cell
        
        data = CornerDebugData(
            iteration=iteration,
            cfl=cfl,
            residual=residual,
            
            # Corner 1
            c1_i=self.c1_i, c1_j=self.c1_j,
            c1_p=float(c1_Q[0]), c1_u=float(c1_Q[1]), 
            c1_v=float(c1_Q[2]), c1_nu=float(c1_Q[3]),
            c1_R_p=float(c1_R[0]), c1_R_u=float(c1_R[1]),
            c1_R_v=float(c1_R[2]), c1_R_nu=float(c1_R[3]),
            
            # Corner 2
            c2_i=self.c2_i, c2_j=self.c2_j,
            c2_p=float(c2_Q[0]), c2_u=float(c2_Q[1]),
            c2_v=float(c2_Q[2]), c2_nu=float(c2_Q[3]),
            c2_R_p=float(c2_R[0]), c2_R_u=float(c2_R[1]),
            c2_R_v=float(c2_R[2]), c2_R_nu=float(c2_R[3]),
            
            # Neighbor values
            c1_neighbor_im1_p=float(c1_Qim1[0]),
            c1_neighbor_ip1_p=float(c1_Qip1[0]),
            c1_neighbor_jm1_p=float(c1_Qjm1[0]),
            c1_neighbor_jp1_p=float(c1_Qjp1[0]),
            
            c2_neighbor_im1_p=float(c2_Qim1[0]),
            c2_neighbor_ip1_p=float(c2_Qip1[0]),
            c2_neighbor_jm1_p=float(c2_Qjm1[0]),
            c2_neighbor_jp1_p=float(c2_Qjp1[0]),
        )
        
        self.history.append(data)
        return data
    
    def print_current(self, data: CornerDebugData) -> None:
        """Print current corner values to console."""
        print(f"\n--- Iteration {data.iteration} (CFL={data.cfl:.3f}, Res={data.residual:.2e}) ---")
        print(f"Corner 1 (I={data.c1_i}, J={data.c1_j}):")
        print(f"  State: p={data.c1_p:+.6e} u={data.c1_u:+.6e} v={data.c1_v:+.6e} nu={data.c1_nu:+.6e}")
        print(f"  Resid: R_p={data.c1_R_p:+.6e} R_u={data.c1_R_u:+.6e} R_v={data.c1_R_v:+.6e} R_nu={data.c1_R_nu:+.6e}")
        print(f"  Neighbors (p): im1={data.c1_neighbor_im1_p:+.6e} ip1={data.c1_neighbor_ip1_p:+.6e} "
              f"jm1={data.c1_neighbor_jm1_p:+.6e} jp1(ghost)={data.c1_neighbor_jp1_p:+.6e}")
        
        print(f"Corner 2 (I={data.c2_i}, J={data.c2_j}):")
        print(f"  State: p={data.c2_p:+.6e} u={data.c2_u:+.6e} v={data.c2_v:+.6e} nu={data.c2_nu:+.6e}")
        print(f"  Resid: R_p={data.c2_R_p:+.6e} R_u={data.c2_R_u:+.6e} R_v={data.c2_R_v:+.6e} R_nu={data.c2_R_nu:+.6e}")
        print(f"  Neighbors (p): im1={data.c2_neighbor_im1_p:+.6e} ip1={data.c2_neighbor_ip1_p:+.6e} "
              f"jm1={data.c2_neighbor_jm1_p:+.6e} jp1(ghost)={data.c2_neighbor_jp1_p:+.6e}")
    
    def save_csv(self, filename: str = "corner_debug.csv") -> str:
        """Save history to CSV file."""
        output_path = self.output_dir / filename
        
        if not self.history:
            print("No debug data to save.")
            return ""
        
        # Get field names from dataclass
        fieldnames = [f.name for f in self.history[0].__dataclass_fields__.values()]
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for data in self.history:
                writer.writerow({k: getattr(data, k) for k in fieldnames})
        
        print(f"Saved corner debug data to: {output_path}")
        return str(output_path)
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze the recorded data for divergence patterns."""
        if len(self.history) < 2:
            return {}
        
        # Find when values start to grow
        c1_p_values = [d.c1_p for d in self.history]
        c2_p_values = [d.c2_p for d in self.history]
        c1_R_p_values = [abs(d.c1_R_p) for d in self.history]
        c2_R_p_values = [abs(d.c2_R_p) for d in self.history]
        
        # Find first iteration where residual exceeds threshold
        threshold = 0.01
        c1_first_large = None
        c2_first_large = None
        
        for i, d in enumerate(self.history):
            if c1_first_large is None and abs(d.c1_R_p) > threshold:
                c1_first_large = d.iteration
            if c2_first_large is None and abs(d.c2_R_p) > threshold:
                c2_first_large = d.iteration
        
        analysis = {
            'c1_max_p': max(c1_p_values),
            'c1_min_p': min(c1_p_values),
            'c1_max_R_p': max(c1_R_p_values),
            'c1_first_large_residual': c1_first_large,
            'c2_max_p': max(c2_p_values),
            'c2_min_p': min(c2_p_values),
            'c2_max_R_p': max(c2_R_p_values),
            'c2_first_large_residual': c2_first_large,
        }
        
        print("\n=== Corner Analysis ===")
        print("Corner 1 (I=0, J=Jmax):")
        print(f"  Pressure range: [{analysis['c1_min_p']:.6e}, {analysis['c1_max_p']:.6e}]")
        print(f"  Max |R_p|: {analysis['c1_max_R_p']:.6e}")
        print(f"  First large residual at iter: {analysis['c1_first_large_residual']}")
        print("Corner 2 (I=Imax, J=Jmax):")
        print(f"  Pressure range: [{analysis['c2_min_p']:.6e}, {analysis['c2_max_p']:.6e}]")
        print(f"  Max |R_p|: {analysis['c2_max_R_p']:.6e}")
        print(f"  First large residual at iter: {analysis['c2_first_large_residual']}")
        
        return analysis
