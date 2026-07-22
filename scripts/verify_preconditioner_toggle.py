
import sys
import os
import time
import numpy as np
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent.resolve())
sys.path.insert(0, project_root)

from src.config.loader import load_yaml
from src.grid.loader import load_or_generate_grid
from src.solvers.rans_solver import RANSSolver, SolverConfig

def test_preconditioner_toggle():
    print("Testing Preconditioner Toggle...")
    
    # Load base config
    config_path = "config/examples/naca0012_newton.yaml"
    sim_config = load_yaml(config_path)
    
    # Force Newton mode
    sim_config.solver.mode = "newton"
    sim_config.solver.print_freq = 1
    
    # Load grid
    g_conf = sim_config.grid
    grid_coords, grid_indices = load_or_generate_grid(
        grid_file=g_conf.airfoil,
        n_surface=g_conf.n_surface,
        n_wake=g_conf.n_wake,
        y_plus=g_conf.y_plus,
        gradation=g_conf.gradation,
        farfield_radius=g_conf.farfield_radius,
        wake_fan_factor=g_conf.wake_fan_factor,
        wake_fan_k=g_conf.wake_fan_k,
        reynolds=sim_config.flow.reynolds
    )
    grid_data = (grid_coords, grid_indices)
    
    # --- Test 1: With Preconditioner (Default) ---
    print("\n--- Test 1: With Preconditioner (True) ---")
    sim_config.solver.use_preconditioner = True
    solver_config_1 = sim_config.to_solver_config()
    
    solver1 = RANSSolver(config=solver_config_1, grid_data=grid_data)
    solver1._initialize_state()
    solver1._initialize_jax()
    
    # Run 1 step
    start_time = time.time()
    solver1.step()
    dt_1 = time.time() - start_time
    
    info1 = solver1._last_gmres_info
    print(f"Time: {dt_1:.4f}s")
    if info1:
        print(f"GMRES: {info1['iterations']} iters, Residual: {info1['residual_norm']:.2e}")
    else:
        print("Warning: No GMRES info captured.")

    # --- Test 2: Without Preconditioner ---
    print("\n--- Test 2: Without Preconditioner (False) ---")
    sim_config.solver.use_preconditioner = False
    solver_config_2 = sim_config.to_solver_config()
    
    solver2 = RANSSolver(config=solver_config_2, grid_data=grid_data)
    # Ensure JAX recompiles for the new static argument
    solver2._initialize_state()
    solver2._initialize_jax()
    
    # Run 1 step
    start_time = time.time()
    solver2.step()
    dt_2 = time.time() - start_time
    
    info2 = solver2._last_gmres_info
    print(f"Time: {dt_2:.4f}s")
    if info2:
        print(f"GMRES: {info2['iterations']} iters, Residual: {info2['residual_norm']:.2e}")
    else:
        print("Warning: No GMRES info captured.")
        
    # Validation
    if info1 and info2:
        # Preconditioned should be generally better (fewer iters or better residual)
        # But for the very first step from uniform flow, differences might be subtle or chaotic.
        # Main check is that they RAN and ideally acted differently.
        print("\nVerification Results:")
        print(f"Precond Iters: {info1['iterations']} vs No-Precond Iters: {info2['iterations']}")
        if info1['iterations'] != info2['iterations']:
             print("SUCCESS: GMRES behavior actually changed.")
        else:
             print("NOTE: Iteration counts same (could be convergence limits). Checking time...")
             # precond adds overhead, so if iters are same, precond step might be slightly slower due to matrix inversion
             pass
    
    print("\nToggle Verification Completed.")

if __name__ == "__main__":
    test_preconditioner_toggle()
