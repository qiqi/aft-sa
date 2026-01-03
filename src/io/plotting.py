"""
Visualization utilities for CFD solutions.
"""

import os
import numpy as np

from src.constants import NGHOST
from typing import Optional, List
from pathlib import Path

_plt = None
_matplotlib = None


def _ensure_matplotlib():
    """Ensure matplotlib is available and configured."""
    global _plt, _matplotlib
    if _plt is None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        _matplotlib = matplotlib
        _plt = plt
    return _plt


def plot_flow_field(X: np.ndarray, Y: np.ndarray, Q: np.ndarray,
                    iteration: int, residual: float, cfl: float,
                    output_dir: str, case_name: str = "flow",
                    C_pt: Optional[np.ndarray] = None,
                    residual_field: Optional[np.ndarray] = None) -> str:
    """Plot flow field with global and zoomed views."""
    plt = _ensure_matplotlib()
    
    NI = X.shape[0] - 1
    NJ = X.shape[1] - 1
    
    if Q.shape[0] == NI + 2*NGHOST and Q.shape[1] == NJ + 2*NGHOST:
        Q_int = Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
    elif Q.shape[0] == NI and Q.shape[1] == NJ:
        Q_int = Q
    else:
        raise ValueError(f"Q shape {Q.shape} not compatible with grid ({NI}, {NJ})")
    
    p = Q_int[:, :, 0]
    u = Q_int[:, :, 1]
    v = Q_int[:, :, 2]
    vel_mag = np.sqrt(u**2 + v**2)
    
    n_rows = 2
    if C_pt is not None:
        n_rows += 1
    if residual_field is not None:
        n_rows += 1
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
    if n_rows == 2:
        axes = axes.reshape(2, 2)
    
    ax = axes[0, 0]
    p_clip = np.clip(p, np.percentile(p, 1), np.percentile(p, 99))
    pc = ax.pcolormesh(X.T, Y.T, p_clip.T, cmap='RdBu_r', shading='flat', rasterized=True)
    ax.set_aspect('equal')
    ax.set_title(f'Pressure (Global) - Iter {iteration}')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    plt.colorbar(pc, ax=ax, shrink=0.8, label='p')
    
    ax = axes[0, 1]
    pc = ax.pcolormesh(X.T, Y.T, p_clip.T, cmap='RdBu_r', shading='flat', rasterized=True)
    ax.set_aspect('equal')
    ax.set_xlim(-0.2, 1.4)
    ax.set_ylim(-0.4, 0.4)
    ax.set_title(f'Pressure (Near Airfoil)')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    plt.colorbar(pc, ax=ax, shrink=0.8, label='p')
    
    ax = axes[1, 0]
    vel_clip = np.clip(vel_mag, 0, np.percentile(vel_mag, 99))
    pc = ax.pcolormesh(X.T, Y.T, vel_clip.T, cmap='viridis', shading='flat', rasterized=True)
    ax.set_aspect('equal')
    ax.set_title(f'Velocity Magnitude (Global) - CFL={cfl:.2f}')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    plt.colorbar(pc, ax=ax, shrink=0.8, label='|V|')
    
    ax = axes[1, 1]
    pc = ax.pcolormesh(X.T, Y.T, vel_clip.T, cmap='viridis', shading='flat', rasterized=True)
    ax.set_aspect('equal')
    ax.set_xlim(-0.2, 1.4)
    ax.set_ylim(-0.4, 0.4)
    ax.set_title(f'Velocity Magnitude (Near Airfoil) - Res={residual:.2e}')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    plt.colorbar(pc, ax=ax, shrink=0.8, label='|V|')
    
    row_idx = 2
    
    if C_pt is not None:
        C_pt_abs_max = max(np.abs(np.percentile(C_pt, 1)), np.abs(np.percentile(C_pt, 99)))
        C_pt_abs_max = max(C_pt_abs_max, 0.01)
        
        ax = axes[row_idx, 0]
        pc = ax.pcolormesh(X.T, Y.T, C_pt.T, cmap='RdBu_r', shading='flat',
                           rasterized=True, vmin=-C_pt_abs_max, vmax=C_pt_abs_max)
        ax.set_aspect('equal')
        ax.set_title(f'Total Pressure Loss C_pt (Global)')
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        plt.colorbar(pc, ax=ax, shrink=0.8, label='C_pt')
        
        ax = axes[row_idx, 1]
        pc = ax.pcolormesh(X.T, Y.T, C_pt.T, cmap='RdBu_r', shading='flat',
                           rasterized=True, vmin=-C_pt_abs_max, vmax=C_pt_abs_max)
        ax.set_aspect('equal')
        ax.set_xlim(-0.2, 1.4)
        ax.set_ylim(-0.4, 0.4)
        ax.set_title(f'Total Pressure Loss C_pt (Near Airfoil)')
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        plt.colorbar(pc, ax=ax, shrink=0.8, label='C_pt')
        
        row_idx += 1
    
    if residual_field is not None:
        R_p = np.abs(residual_field[:, :, 0])
        R_log = np.log10(R_p + 1e-12)
        R_min = np.percentile(R_log, 1)
        R_max = np.percentile(R_log, 99)
        
        ax = axes[row_idx, 0]
        pc = ax.pcolormesh(X.T, Y.T, R_log.T, cmap='hot_r', shading='flat',
                           rasterized=True, vmin=R_min, vmax=R_max)
        ax.set_aspect('equal')
        ax.set_title(f'log₁₀|Residual| (Global)')
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        plt.colorbar(pc, ax=ax, shrink=0.8, label='log₁₀|R_p|')
        
        ax = axes[row_idx, 1]
        pc = ax.pcolormesh(X.T, Y.T, R_log.T, cmap='hot_r', shading='flat',
                           rasterized=True, vmin=R_min, vmax=R_max)
        ax.set_aspect('equal')
        ax.set_xlim(-0.2, 1.4)
        ax.set_ylim(-0.4, 0.4)
        ax.set_title(f'log₁₀|Residual| (Near Airfoil)')
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        
        max_idx = np.unravel_index(np.argmax(R_p), R_p.shape)
        x_max = 0.5 * (X[max_idx[0], max_idx[1]] + X[max_idx[0]+1, max_idx[1]])
        y_max = 0.5 * (Y[max_idx[0], max_idx[1]] + Y[max_idx[0]+1, max_idx[1]])
        ax.plot(x_max, y_max, 'g*', markersize=15, markeredgecolor='white', markeredgewidth=1)
        plt.colorbar(pc, ax=ax, shrink=0.8, label='log₁₀|R_p|')
    
    plt.tight_layout()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, f'{case_name}_iter{iteration:06d}.pdf')
    plt.savefig(output_path, dpi=100)
    plt.close()
    
    return output_path


def plot_residual_history(residuals: List[float], output_dir: str,
                          case_name: str = "residual") -> Optional[str]:
    """Plot residual convergence history."""
    if len(residuals) < 2:
        return None
    
    plt = _ensure_matplotlib()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = np.arange(1, len(residuals) + 1)
    ax.semilogy(iterations, residuals, 'b-', lw=1.5)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual RMS')
    ax.set_title('Convergence History')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(residuals))
    
    plt.tight_layout()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, f'{case_name}_convergence.pdf')
    plt.savefig(output_path, dpi=100)
    plt.close()
    
    return output_path


def plot_surface_distributions(x: np.ndarray, Cp: np.ndarray, Cf: np.ndarray,
                                output_dir: str, case_name: str = "surface",
                                reference_data: Optional[dict] = None) -> str:
    """Plot surface Cp and Cf distributions."""
    plt = _ensure_matplotlib()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    ax.plot(x, Cp, 'b.', markersize=3, label='CFD')
    if reference_data is not None:
        if 'cp_upper' in reference_data:
            ax.plot(reference_data['x_upper'], reference_data['cp_upper'],
                    'r-', lw=1.5, alpha=0.7, label='Reference')
        if 'cp_lower' in reference_data:
            ax.plot(reference_data['x_lower'], reference_data['cp_lower'],
                    'r-', lw=1.5, alpha=0.7)
    ax.set_xlabel('x/c')
    ax.set_ylabel('Cp')
    ax.set_title('Pressure Coefficient')
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    
    ax = axes[1]
    ax.plot(x, Cf, 'b.', markersize=3, label='CFD')
    if reference_data is not None:
        if 'cf_upper' in reference_data:
            ax.plot(reference_data['x_upper'], reference_data['cf_upper'],
                    'r-', lw=1.5, alpha=0.7, label='Reference')
        if 'cf_lower' in reference_data:
            ax.plot(reference_data['x_lower'], reference_data['cf_lower'],
                    'r-', lw=1.5, alpha=0.7)
    ax.set_xlabel('x/c')
    ax.set_ylabel('Cf')
    ax.set_title('Skin Friction Coefficient')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    
    plt.tight_layout()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, f'{case_name}_surface.pdf')
    plt.savefig(output_path, dpi=100)
    plt.close()
    
    return output_path
