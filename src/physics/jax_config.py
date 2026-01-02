"""JAX configuration for physics modules: 64-bit precision and device detection."""

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


__all__ = ['jax', 'jnp', 'get_device_info', 'is_gpu_available']
