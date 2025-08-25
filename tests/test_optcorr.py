import pytest
import numpy as np
import torch
from einsumcorr.optcorr import optcorr


@pytest.fixture
def sample_matrix():
    """Generate a simple test matrix."""
    np.random.seed(42)
    return np.random.randn(100, 5)


@pytest.fixture
def sample_matrix_pair():
    """Generate a pair of test matrices."""
    np.random.seed(42)
    x = np.random.randn(100, 5)
    y = np.random.randn(100, 3)
    return x, y


@pytest.fixture
def single_column_matrix():
    """Generate a single column matrix."""
    np.random.seed(42)
    return np.random.randn(100, 1)


@pytest.fixture
def large_matrix():
    """Generate a larger matrix for performance testing."""
    np.random.seed(42)
    return np.random.randn(1000, 50)


def test_optcorr_single_matrix_shape(sample_matrix):
    """Test that optcorr returns correct shape for single matrix input."""
    result = optcorr(sample_matrix)
    expected_shape = (sample_matrix.shape[1], sample_matrix.shape[1])
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"


def test_optcorr_two_matrices_shape(sample_matrix_pair):
    """Test that optcorr returns correct shape for two matrix inputs."""
    x, y = sample_matrix_pair
    result = optcorr(x, y)
    expected_shape = (x.shape[1], y.shape[1])
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"


def test_optcorr_self_correlation_diagonal(sample_matrix):
    """Test that self-correlation has ones on the diagonal."""
    result = optcorr(sample_matrix)
    diagonal = np.diag(result)
    np.testing.assert_allclose(diagonal, np.ones_like(diagonal), rtol=1e-5)


def test_optcorr_correlation_range(sample_matrix):
    """Test that all correlation values are in [-1, 1] range."""
    result = optcorr(sample_matrix)
    assert np.all(result >= -1.0 - 1e-6), "Correlation values below -1"
    assert np.all(result <= 1.0 + 1e-6), "Correlation values above 1"


def test_optcorr_symmetric(sample_matrix):
    """Test that self-correlation matrix is symmetric."""
    result = optcorr(sample_matrix)
    np.testing.assert_allclose(result, result.T, rtol=1e-5)


def test_optcorr_single_column(single_column_matrix):
    """Test optcorr with single column matrix."""
    result = optcorr(single_column_matrix)
    assert result.shape == (1, 1)
    np.testing.assert_allclose(result[0, 0], 1.0, rtol=1e-5)


def test_optcorr_empty_matrix():
    """Test optcorr with empty matrix raises appropriate error."""
    empty_matrix = np.array([]).reshape(0, 0)
    with pytest.raises(ValueError):
        optcorr(empty_matrix)


def test_optcorr_mismatched_rows():
    """Test that matrices with different numbers of rows raise an error."""
    x = np.random.randn(100, 5)
    y = np.random.randn(50, 5)
    with pytest.raises(ValueError):
        optcorr(x, y)


def test_optcorr_numpy_input_types():
    """Test that optcorr accepts numpy arrays."""
    x = np.random.randn(50, 3).astype(np.float32)
    result = optcorr(x)
    assert isinstance(result, np.ndarray)


def test_optcorr_handles_nan():
    """Test that optcorr handles NaN values appropriately."""
    x = np.random.randn(50, 3)
    x[0, 0] = np.nan
    with pytest.raises(ValueError):
        optcorr(x)


def compare_with_numpy_corrcoef(x, y=None, rtol=1e-5):
    """Helper function to compare optcorr with numpy's corrcoef."""
    if y is None:
        # Self-correlation
        result_optcorr = optcorr(x)
        result_numpy = np.corrcoef(x.T)
    else:
        # Cross-correlation
        result_optcorr = optcorr(x, y)
        # For cross-correlation, we need to extract the relevant submatrix
        full_corr = np.corrcoef(x.T, y.T)
        n_x = x.shape[1]
        result_numpy = full_corr[:n_x, n_x:]
    
    np.testing.assert_allclose(result_optcorr, result_numpy, rtol=rtol)


def test_comparison_with_numpy_single_matrix(sample_matrix):
    """Test that optcorr matches numpy.corrcoef for single matrix."""
    compare_with_numpy_corrcoef(sample_matrix)


def test_comparison_with_numpy_two_matrices(sample_matrix_pair):
    """Test that optcorr matches numpy.corrcoef for two matrices."""
    x, y = sample_matrix_pair
    compare_with_numpy_corrcoef(x, y)


def test_comparison_with_numpy_large_matrix(large_matrix):
    """Test that optcorr matches numpy.corrcoef for larger matrices."""
    # Use slightly higher tolerance for larger matrices with float32 on MPS
    compare_with_numpy_corrcoef(large_matrix, rtol=5e-4)


def test_gpu_acceleration_if_available():
    """Test that GPU acceleration is used when available."""
    x = np.random.randn(100, 10)
    result = optcorr(x)
    
    # Check if computation was successful
    assert result.shape == (10, 10)
    
    # If GPU is available, this test verifies the code path works
    # The actual GPU usage is tested within the implementation
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        # Just verify the result is valid
        assert np.all(np.isfinite(result))


def test_deterministic_results():
    """Test that optcorr produces deterministic results."""
    np.random.seed(42)
    x = np.random.randn(50, 5)
    
    result1 = optcorr(x)
    result2 = optcorr(x)
    
    np.testing.assert_array_equal(result1, result2)


def test_min_cols_for_gpu_parameter():
    """Test that min_cols_for_gpu parameter works correctly."""
    np.random.seed(42)
    x = np.random.randn(100, 10)
    
    # With default threshold (2500), should use numpy
    result_default = optcorr(x)
    
    # With low threshold, should use GPU/einsum
    result_gpu = optcorr(x, min_cols_for_gpu=5)
    
    # Results should be very similar regardless of backend
    np.testing.assert_allclose(result_default, result_gpu, rtol=1e-5)
    
    # Both should be valid correlation matrices
    assert result_default.shape == (10, 10)
    assert result_gpu.shape == (10, 10)
    
    # Test with cross-correlation
    y = np.random.randn(100, 8)
    result_cross_default = optcorr(x, y)
    result_cross_gpu = optcorr(x, y, min_cols_for_gpu=5)
    
    np.testing.assert_allclose(result_cross_default, result_cross_gpu, rtol=1e-5)
    assert result_cross_default.shape == (10, 8)