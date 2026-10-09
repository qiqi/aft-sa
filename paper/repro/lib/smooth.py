"""NaN-aware Gaussian smoothing for probe-sampled fields.

Used by the suite-figure row-2 diagnostic (probe-max Omega_hat*I_hat): the
transported chi integrates production along streamlines, so isolated
single-point rate values (unstructured-mesh noise, near-singular points)
carry no effect on the solution -- a raw pointwise max over-reports them.
The field is neighbor-smoothed over ~0.5% chord streamwise and ~5% of the
probe length wall-normal before the max is taken (presentation only; the
solver's production is untouched).
"""
import numpy as np
from scipy.ndimage import gaussian_filter


def nan_gaussian(a, sigma):
    """Gaussian-filter `a` treating NaNs as missing data (normalized
    convolution); cells with (almost) no finite support return NaN."""
    a = np.asarray(a, float)
    w = np.isfinite(a).astype(float)
    b = np.where(np.isfinite(a), a, 0.0)
    num = gaussian_filter(b, sigma, mode='nearest')
    den = gaussian_filter(w, sigma, mode='nearest')
    out = num / np.maximum(den, 1e-12)
    out[den < 1e-6] = np.nan
    return out
