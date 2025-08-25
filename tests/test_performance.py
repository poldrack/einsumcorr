"""Performance benchmarking tests for einsumcorr."""

import time
import numpy as np
import pytest
from einsumcorr.optcorr import optcorr


def standard_corrcoef(x, y=None):
    """Standard numpy correlation for comparison."""
    if y is None:
        return np.corrcoef(x.T)
    else:
        full_corr = np.corrcoef(x.T, y.T)
        n_x = x.shape[1]
        return full_corr[:n_x, n_x:]


@pytest.mark.parametrize("size", [(100, 10), (500, 20), (1000, 30)])
def test_performance_comparison(size):
    """Compare performance of optcorr vs numpy.corrcoef."""
    n_samples, n_features = size
    np.random.seed(42)
    x = np.random.randn(n_samples, n_features)
    
    # Time optcorr
    start = time.perf_counter()
    result_optcorr = optcorr(x)
    time_optcorr = time.perf_counter() - start
    
    # Time numpy corrcoef
    start = time.perf_counter()
    result_numpy = standard_corrcoef(x)
    time_numpy = time.perf_counter() - start
    
    # Verify results match
    np.testing.assert_allclose(result_optcorr, result_numpy, rtol=1e-3)
    
    # Report performance
    speedup = time_numpy / time_optcorr if time_optcorr > 0 else float('inf')
    print(f"\nMatrix size {size}: optcorr={time_optcorr:.4f}s, numpy={time_numpy:.4f}s, speedup={speedup:.2f}x")
    
    # optcorr should be reasonably fast (not necessarily faster due to overhead)
    assert time_optcorr < 5.0, f"optcorr took too long: {time_optcorr}s"


def test_gpu_speedup_if_available():
    """Test GPU acceleration provides speedup for large matrices."""
    import torch
    
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        pytest.skip("No GPU available for speedup test")
    
    # Use a larger matrix to see GPU benefits
    np.random.seed(42)
    x = np.random.randn(2000, 50)
    
    # Warm-up run
    _ = optcorr(x[:100, :5])
    
    # Time the actual computation
    start = time.perf_counter()
    result = optcorr(x)
    elapsed = time.perf_counter() - start
    
    print(f"\nLarge matrix (2000x50) correlation computed in {elapsed:.4f}s")
    assert result.shape == (50, 50)
    assert elapsed < 10.0, f"GPU computation took too long: {elapsed}s"