"""
Array utility functions for safe handling of NaN/Inf values.
"""

import numpy as np


def sanitize_array(arr: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Replace NaN and Inf values with a finite fill value.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array that may contain NaN/Inf values.
    fill_value : float
        Value to replace NaN/Inf with (default: 0.0).
        
    Returns
    -------
    np.ndarray
        Array with all NaN/Inf values replaced.
    """
    result = np.array(arr, dtype=np.float64)
    mask = ~np.isfinite(result)
    if np.any(mask):
        result[mask] = fill_value
    return result


def to_json_safe_list(arr: np.ndarray) -> list:
    """Convert numpy array to a list that's safe for JSON serialization.
    
    Converts NaN and Inf values to None (which becomes null in JSON).
    This is needed because Plotly's HTML serialization doesn't always
    handle numpy NaN values correctly.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array that may contain NaN/Inf values.
        
    Returns
    -------
    list
        List with NaN/Inf values replaced by None.
    """
    flat = arr.flatten()
    result = []
    for val in flat:
        if np.isfinite(val):
            result.append(float(val))
        else:
            result.append(None)
    return result


def safe_minmax(arr: np.ndarray, default_min: float = -1.0, default_max: float = 1.0) -> tuple:
    """Get min/max of array, handling NaN/Inf gracefully.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array.
    default_min, default_max : float
        Default values if array is all NaN/Inf.
        
    Returns
    -------
    tuple
        (min_val, max_val) with finite values.
    """
    finite_vals = arr[np.isfinite(arr)]
    if len(finite_vals) == 0:
        return default_min, default_max
    return float(finite_vals.min()), float(finite_vals.max())


def safe_absmax(arr: np.ndarray, default: float = 1.0) -> float:
    """Get max absolute value from array, handling NaN/Inf gracefully.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array.
    default : float
        Default value if array is all NaN/Inf.
        
    Returns
    -------
    float
        Maximum absolute finite value in array.
    """
    finite_vals = arr[np.isfinite(arr)]
    if len(finite_vals) == 0:
        return default
    return float(np.abs(finite_vals).max())
