"""Module for computing columnwise correlations using Einstein summation."""

import numpy as np
import torch
import opt_einsum as oe


def optcorr(x, y=None, min_cols_for_gpu=2500):
    """
    Compute columnwise correlations using Einstein summation notation.
    
    Parameters
    ----------
    x : np.ndarray
        First matrix of shape (n_samples, n_features_x)
    y : np.ndarray, optional
        Second matrix of shape (n_samples, n_features_y)
        If None, computes correlation of x with itself
    min_cols_for_gpu : int, optional
        Minimum number of total columns to use GPU acceleration.
        Below this threshold, falls back to numpy.corrcoef for better performance.
        Default is 2500 based on empirical performance analysis.
    
    Returns
    -------
    np.ndarray
        Correlation matrix of shape (n_features_x, n_features_y) if y is provided,
        or (n_features_x, n_features_x) if y is None
    
    Raises
    ------
    ValueError
        If input matrices have incompatible shapes or contain NaN values
    """
    # Input validation
    if not isinstance(x, np.ndarray):
        raise TypeError("x must be a numpy array")
    
    if x.size == 0:
        raise ValueError("Empty arrays are not supported")
    
    if np.any(np.isnan(x)):
        raise ValueError("Input contains NaN values")
    
    # Handle single matrix case
    if y is None:
        y = x
    else:
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy array")
        
        if np.any(np.isnan(y)):
            raise ValueError("Input contains NaN values")
        
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Incompatible shapes: x has {x.shape[0]} rows, y has {y.shape[0]} rows")
    
    # Ensure 2D arrays
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    n_samples = x.shape[0]
    
    # Check if we should use numpy for small matrices
    total_cols = x.shape[1] + (0 if y is x else y.shape[1])
    if total_cols < min_cols_for_gpu:
        return _numpy_corrcoef(x, y)
    
    # Detect available device
    device = _get_device()
    
    # Use float32 for MPS compatibility, float64 for CPU/CUDA
    np_dtype = np.float32 if device.type == 'mps' else np.float64
    
    # Convert to PyTorch tensors
    x_tensor = torch.from_numpy(x.astype(np_dtype)).to(device)
    y_tensor = torch.from_numpy(y.astype(np_dtype)).to(device)
    
    # Center the matrices (subtract column means)
    x_mean = oe.contract("ij->j", x_tensor, backend='torch') / n_samples
    y_mean = oe.contract("ij->j", y_tensor, backend='torch') / n_samples
    
    x_centered = x_tensor - x_mean
    y_centered = y_tensor - y_mean
    
    # Compute covariance matrix
    cov = oe.contract("ij,ik->jk", x_centered, y_centered, backend='torch') / (n_samples - 1)
    
    # Compute standard deviations
    x_var = oe.contract("ij,ij->j", x_centered, x_centered, backend='torch') / (n_samples - 1)
    y_var = oe.contract("ij,ij->j", y_centered, y_centered, backend='torch') / (n_samples - 1)
    
    x_std = torch.sqrt(x_var)
    y_std = torch.sqrt(y_var)
    
    # Compute correlation matrix
    std_product = oe.contract("i,j->ij", x_std, y_std, backend='torch')
    
    # Handle division by zero for constant columns
    std_product = torch.where(std_product == 0, torch.ones_like(std_product), std_product)
    
    corr = cov / std_product
    
    # Ensure correlation values are in [-1, 1] range (numerical stability)
    corr = torch.clamp(corr, -1.0, 1.0)
    
    # Convert back to numpy
    result = corr.cpu().numpy()
    
    return result


def _numpy_corrcoef(x, y=None):
    """
    Compute correlation using numpy.corrcoef for small matrices.
    
    Parameters
    ----------
    x : np.ndarray
        First matrix
    y : np.ndarray, optional
        Second matrix for cross-correlation
        
    Returns
    -------
    np.ndarray
        Correlation matrix
    """
    if y is None or y is x:
        # Self-correlation
        result = np.corrcoef(x.T)
        # Handle single column case where corrcoef returns scalar
        if result.ndim == 0:
            result = np.array([[result]])
        return result
    else:
        # Cross-correlation
        full_corr = np.corrcoef(x.T, y.T)
        if full_corr.ndim == 0:
            # Single feature case
            return np.array([[full_corr]])
        n_x = x.shape[1]
        return full_corr[:n_x, n_x:]


def _get_device():
    """Detect and return the best available device (GPU or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')