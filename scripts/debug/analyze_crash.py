
import jax
import jax.numpy as jnp
import numpy as np
import sys
import os
import yaml

# Add project root to path
sys.path.append(os.getcwd())

from src.solvers.params import PhysicsParams
from src.solvers.rans_solver import RANSSolver
from src.config.schema import SimulationConfig
from src.config.loader import from_dict as load_config_from_dict

def main():
    # Load checkpoint
    ckpt_path = "/home/qiqi/flexcompute/sa-ai/output/naca0012_newton/final_q.npy"
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        return

    print(f"Loading {ckpt_path}...")
    Q_np = np.load(ckpt_path)
    Q_final = jnp.asarray(Q_np)
    
    # Load Config
    config_path = "/home/qiqi/flexcompute/sa-ai/config/examples/naca0012_newton.yaml"
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Apply CLI overrides used in run
    config_dict['flow']['chi_inf'] = 1.0
    config_dict['flow']['alpha'] = 8.0
    
    # Load Config Object
    config = load_config_from_dict(config_dict)
    
    # Load Grid
    x_path = "/home/qiqi/flexcompute/sa-ai/output/naca0012_newton/final_x.npy"
    y_path = "/home/qiqi/flexcompute/sa-ai/output/naca0012_newton/final_y.npy"
    if not os.path.exists(x_path):
        print("X not found")
        return
    X = jnp.load(x_path)
    Y = jnp.load(y_path)
    
    # Convert to legacy SolverConfig (as per run_airfoil.py)
    solver_config = config.to_solver_config()
    
    # Initialize Solver with Grid Data
    print("Initializing solver with loaded grid...")
    solver = RANSSolver(grid_data=(np.array(X), np.array(Y)), config=solver_config)
    
    # Verify shape
    NI, NJ = solver.NI, solver.NJ
    print(f"Solver Grid: {NI} x {NJ}")
    print(f"Checkpoint Shape: {Q_final.shape}")
    
    # Q_final is cell centered? Format in solver is (NI+2*nghost, NJ+2*nghost, 4)
    # The checkpoint saved by save_diagnostics is likely the INTERIOR solution?
    # rans_solver.py:1455: `self.save_checkpoint(Q_n)`
    # Q_n is the full array including ghosts.
    
    # Compute Residual
    print("Computing Residual...")
    # metrics are in solver.Si_x, etc.
    # We call solver._jit_residual(Q_final, solver.params)
    
    # Note: jit_residual expects Q with ghosts.
    params = solver._build_params()
    R = solver._jit_residual(Q_final, params)
    
    # R is (NI, NJ, 4)
    R_mag = jnp.max(jnp.abs(R), axis=-1)
    
    max_res = jnp.max(R_mag)
    flat_idx = jnp.argmax(R_mag)
    idx_i, idx_j = jnp.unravel_index(flat_idx, R_mag.shape)
    
    print(f"==================================================")
    print(f"Max Residual: {max_res:.4e} at [{idx_i}, {idx_j}]")
    print(f"==================================================")
    
    # Get State at that location
    q_cell = Q_final[solver.nghost + idx_i, solver.nghost + idx_j]
    nu_tilde = q_cell[3]
    nu_lam = solver.mu_laminar
    chi = nu_tilde / (nu_lam + 1e-30)
    
    # Get SA/AFT params
    from src.numerics.sa_sources import compute_turbulent_fraction
    is_turb = compute_turbulent_fraction(chi)
    
    print(f"State at Failure:")
    print(f"  rho  = {q_cell[0]:.4e}")
    print(f"  u    = {q_cell[1]/q_cell[0]:.4e}")
    print(f"  v    = {q_cell[2]/q_cell[0]:.4e}")
    print(f"  nu~  = {nu_tilde:.4e}")
    print(f"  chi  = {chi:.4e}")
    print(f"  mask = {is_turb:.4e} (0=Lam, 1=Turb)")
    
    # Print Residual Components
    r_cell = R[idx_i, idx_j]
    print(f"Residual Components:")
    print(f"  R_rho = {r_cell[0]:.4e}")
    print(f"  R_ru  = {r_cell[1]:.4e}")
    print(f"  R_rv  = {r_cell[2]:.4e}")
    print(f"  R_nu  = {r_cell[3]:.4e}")

    # Check neighbors
    print(f"Neighborhood nu~:")
    start_i, end_i = max(0, idx_i-2), min(NI, idx_i+3)
    start_j, end_j = max(0, idx_j-2), min(NJ, idx_j+3)
    
    # Indices in Q (shifted by nghost)
    s_qi, e_qi = start_i + solver.nghost, end_i + solver.nghost
    s_qj, e_qj = start_j + solver.nghost, end_j + solver.nghost
    
    curr_slice = Q_final[s_qi:e_qi, s_qj:e_qj, 3]
    print(curr_slice)

if __name__ == "__main__":
    main()
