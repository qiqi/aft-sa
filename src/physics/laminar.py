import numpy as np
import torch

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Re_Omega(dudy, y):
    if isinstance(dudy, np.ndarray):
        return y**2 * np.absolute(dudy)
    elif isinstance(dudy, torch.Tensor):
        return y**2 * torch.absolute(dudy)

def compute_nondimensional_amplification_rate(Re_Omega_val, Gamma, A0=-16, A_Omega=2, A_Gamma=5):
    '''
    a = np.log10(Re_Omega(shear, eta) / 1000) / 50 + gamma
    0.2 / (1 + np.exp(-32 * (amax - 1.16)))
    '''
    if isinstance(Re_Omega_val, np.ndarray):
        a = np.log10(np.abs(Re_Omega_val) / 1000) / 50 + Gamma
        return 0.2 / (1 + np.exp(-35 * (a - 1.04)))
    elif isinstance(Re_Omega_val, torch.Tensor):
        a = torch.log10(torch.abs(Re_Omega_val) / 1000) / 50 + Gamma
        return 0.2 / (1 + torch.exp(-35 * (a - 1.04)))

def amplification(u, dudy, y):
    Gamma = 2 * (dudy * y)**2 / (u**2 + (dudy * y)**2)
    return compute_nondimensional_amplification_rate(Re_Omega(dudy, y), Gamma)
