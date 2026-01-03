"""JAX configuration for physics modules: 64-bit precision and device selection."""

import os
import subprocess
from typing import Optional, List, Tuple

# Set device BEFORE importing JAX if CUDA_VISIBLE_DEVICES not already set
_device_configured = False


def _get_gpu_utilization() -> List[Tuple[int, float]]:
    """Get GPU utilization for all available GPUs using nvidia-smi.
    
    Returns list of (gpu_id, utilization_percent) tuples.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return []
        
        utilizations = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(',')
                if len(parts) >= 2:
                    gpu_id = int(parts[0].strip())
                    util = float(parts[1].strip())
                    utilizations.append((gpu_id, util))
        return utilizations
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return []


def _get_gpu_memory_free() -> List[Tuple[int, int]]:
    """Get free GPU memory for all available GPUs.
    
    Returns list of (gpu_id, free_memory_mb) tuples.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return []
        
        memory = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(',')
                if len(parts) >= 2:
                    gpu_id = int(parts[0].strip())
                    free_mb = int(parts[1].strip())
                    memory.append((gpu_id, free_mb))
        return memory
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return []


def select_device(device: Optional[str] = None, verbose: bool = True) -> Optional[int]:
    """Select CUDA device before JAX initialization.
    
    Parameters
    ----------
    device : str, optional
        Device specification:
        - None or "auto": Auto-select least utilized GPU
        - "cpu": Force CPU
        - "0", "1", etc.: Specific GPU index
        - "cuda:0", "gpu:1", etc.: Specific GPU with prefix
    verbose : bool
        Print device selection info.
        
    Returns
    -------
    gpu_id : int or None
        Selected GPU ID, or None if CPU selected.
        
    Notes
    -----
    This must be called BEFORE any JAX imports or operations.
    Sets CUDA_VISIBLE_DEVICES environment variable.
    """
    global _device_configured
    
    if _device_configured:
        return None
    
    # Parse device specification
    if device is None or device == "auto":
        # Auto-select least utilized GPU
        utilizations = _get_gpu_utilization()
        memory = _get_gpu_memory_free()
        
        if not utilizations and not memory:
            if verbose:
                print("  No GPUs detected or nvidia-smi not available, using default device")
            _device_configured = True
            return None
        
        # Prefer GPU with lowest utilization, break ties with most free memory
        if utilizations:
            # Sort by utilization (ascending), then by free memory (descending)
            memory_dict = {gpu_id: free for gpu_id, free in memory}
            scored = [(gpu_id, util, memory_dict.get(gpu_id, 0)) 
                      for gpu_id, util in utilizations]
            scored.sort(key=lambda x: (x[1], -x[2]))  # Low util first, high memory first
            best_gpu = scored[0][0]
            
            if verbose:
                print(f"  GPU utilization: {[(g, f'{u:.0f}%') for g, u, _ in scored]}")
                print(f"  Auto-selected GPU {best_gpu} (lowest utilization)")
            
            os.environ['CUDA_VISIBLE_DEVICES'] = str(best_gpu)
            _device_configured = True
            return best_gpu
        else:
            _device_configured = True
            return None
            
    elif device.lower() == "cpu":
        if verbose:
            print("  Forcing CPU device")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        _device_configured = True
        return None
        
    else:
        # Parse specific GPU ID
        device_str = device.lower()
        for prefix in ['cuda:', 'gpu:']:
            if device_str.startswith(prefix):
                device_str = device_str[len(prefix):]
                break
        
        try:
            gpu_id = int(device_str)
            if verbose:
                print(f"  Using specified GPU {gpu_id}")
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            _device_configured = True
            return gpu_id
        except ValueError:
            raise ValueError(f"Invalid device specification: {device}. "
                           f"Use 'auto', 'cpu', or GPU index (e.g., '0', 'cuda:1')")


# Now import JAX (will use CUDA_VISIBLE_DEVICES if set)
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def get_device_info() -> str:
    """Get available JAX devices as string."""
    devices = jax.devices()
    device_strs = [f"{d.platform}:{d.id}" for d in devices]
    return f"JAX devices: {device_strs}"


def is_gpu_available() -> bool:
    """Check if GPU is available for JAX."""
    devices = jax.devices()
    return any(d.platform == 'gpu' for d in devices)


def get_current_device() -> str:
    """Get the current default JAX device."""
    device = jax.devices()[0]
    return f"{device.platform}:{device.id} ({getattr(device, 'device_kind', 'unknown')})"


__all__ = ['jax', 'jnp', 'get_device_info', 'is_gpu_available', 'select_device', 'get_current_device']
