
import sys
import os
import time
import jax
import numpy as np
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent.resolve())
sys.path.insert(0, project_root)

from src.config.loader import load_yaml
from src.grid.loader import load_or_generate_grid
from src.solvers.rans_solver import RANSSolver

def test_aft_toggle():
    print("Testing Dynamic AFT Toggle...")
    
    # Load base config
    config_path = "config/examples/naca0012_newton.yaml"
    sim_config = load_yaml(config_path)
    
    # Force single step, simple setup
    sim_config.solver.mode = "newton"
    sim_config.solver.print_freq = 1
    # Use coarse grid for speed
    sim_config.grid.n_surface = 129
    sim_config.grid.n_wake = 32
    
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
    
    # Initial Config: AFT ON
    sim_config.aft.enable = True
    solver_config = sim_config.to_solver_config()
    
    solver = RANSSolver(config=solver_config, grid_data=grid_data)
    solver._initialize_state()
    solver._initialize_jax()
    
    print("\n--- Step 1: AFT ON (Warmup) ---")
    start_time = time.time()
    solver.step()
    dt_warmup = time.time() - start_time
    print(f"Warmup Step: {dt_warmup:.4f}s")
    
    # Step 2: AFT ON (Measured)
    print("\n--- Step 2: AFT ON (Measured) ---")
    start_time = time.time()
    solver.step()
    dt_on = time.time() - start_time
    # Capture residual with AFT on logic
    res_on = solver._last_gmres_info['residual_norm'] if solver._last_gmres_info else 0.0
    print(f"Time: {dt_on:.4f}s | Residual: {res_on:.4e}")

    # Step 3: AFT OFF (Hot Switch)
    print("\n--- Step 3: AFT OFF (Hot Switch) ---")
    # Toggle flag in CONFIG - this should propagate via _build_params()
    solver.config.aft_enable = False 
    
    start_time = time.time()
    solver.step()
    dt_off = time.time() - start_time
    res_off = solver._last_gmres_info['residual_norm'] if solver._last_gmres_info else 0.0
    print(f"Time: {dt_off:.4f}s | Residual: {res_off:.4e}")
    
    # Check for recompilation:
    # If dt_off is huge (>0.5s) compared to dt_on (<0.1s), it recompiled.
    # Note: The first step is warmup, so dt_on is the fast baseline.
    if dt_off > dt_on * 5.0 and dt_off > 1.0: # Heuristic
        print("FAILURE: Massive slowdown detected, likely Recompilation triggered.")
    else:
        print("SUCCESS: No recompilation detected (Fast Switch).")
        
    # Check for physics change:
    # Residuals should likely differ slightly or significantly depending on flow state.
    # Since we are just 2 steps in, differences might be small but numeric.
    if abs(res_on - res_off) < 1e-12:
         print("WARNING: Residuals identical. Did the physics actually change?")
         print(f"Res ON: {res_on}, Res OFF: {res_off}")
         print("Possibility: Initial flow (uniform) might produce zero production anyway?")
    else:
         print("SUCCESS: Physics change detected (Residuals differ).")

    print("\nAFT Toggle Verification Completed.")

if __name__ == "__main__":
    test_aft_toggle()
