"""
Laminar Flow Amplification Model.

This module provides functions for computing amplification rates in
laminar/transitional boundary layers, based on local flow properties.

Dimension Agnostic:
    All functions work with any array shape - scalars, 1D arrays, 2D fields, etc.
    Supports both NumPy arrays and PyTorch tensors transparently.

Notes:
    The amplification model is primarily designed for boundary layer flows
    where the shape factor Gamma can be meaningfully defined. For general
    2D flows, consider using full turbulence models (SA, k-ω, etc.) instead.
"""

import numpy as np
import torch


def _get_backend(x):
    """Get the appropriate math backend (numpy or torch) for input x."""
    if isinstance(x, torch.Tensor):
        return torch
    return np


def sigmoid(x):
    """Sigmoid function σ(x) = 1 / (1 + exp(-x))."""
    backend = _get_backend(x)
    return 1 / (1 + backend.exp(-x))


def Re_Omega(dudy, y):
    """
    Compute vorticity Reynolds number Re_Ω = y² |ω|.
    
    Dimension-agnostic: works with any array shape and both numpy/torch.
    
    Parameters
    ----------
    dudy : array_like
        Vorticity magnitude |ω| (any shape).
        For boundary layers: |du/dy|
    y : array_like
        Wall distance (same shape as dudy).
        
    Returns
    -------
    Re_Omega : array_like
        Vorticity Reynolds number (same shape and type as inputs).
    """
    backend = _get_backend(dudy)
    return y**2 * backend.abs(dudy)


def compute_nondimensional_amplification_rate(Re_Omega_val, Gamma, A0=-16, A_Omega=2, A_Gamma=5):
    """
    Compute non-dimensional amplification rate from Re_Ω and shape factor.
    
    Dimension-agnostic: works with any array shape and both numpy/torch.
    
    Parameters
    ----------
    Re_Omega_val : array_like
        Vorticity Reynolds number (any shape).
    Gamma : array_like
        Shape factor parameter (same shape as Re_Omega_val).
    A0, A_Omega, A_Gamma : float
        Model constants (unused in current formulation, kept for API).
        
    Returns
    -------
    amp_rate : array_like
        Non-dimensional amplification rate (same shape and type as inputs).
    """
    backend = _get_backend(Re_Omega_val)
    a = backend.log10(backend.abs(Re_Omega_val) / 1000) / 50 + Gamma
    return 0.2 / (1 + backend.exp(-35 * (a - 1.04)))


def amplification(u, dudy, y):
    """
    Compute amplification rate for boundary layer transition.
    
    This function computes the local amplification rate based on the
    velocity profile shape. It is primarily intended for boundary layer
    flows where the shape factor Gamma is meaningful.
    
    Dimension-agnostic: works with any array shape and both numpy/torch.
    
    Parameters
    ----------
    u : array_like
        Streamwise velocity (any shape).
    dudy : array_like
        Velocity gradient |du/dy| (same shape as u).
    y : array_like
        Wall distance (same shape as u).
        
    Returns
    -------
    amp : array_like
        Amplification rate (same shape and type as inputs).
        
    Notes
    -----
    Gamma = 2(du/dy · y)² / (u² + (du/dy · y)²) is a shape factor that
    characterizes the velocity profile. For Blasius boundary layer, 
    Gamma varies from 0 at the wall to ~0.5 in the freestream.
    """
    Gamma = 2 * (dudy * y)**2 / (u**2 + (dudy * y)**2)
    return compute_nondimensional_amplification_rate(Re_Omega(dudy, y), Gamma)
