# einsumcorr

Efficient columnwise correlation computation using Einstein summation notation with GPU acceleration support.

This module was primarily written as an example for by [Better Code, Better Science](https://poldrack.github.io/BetterCodeBetterScience/frontmatter.html) living textbook. 

All code in this project was generated using Claude Code.  See [CLAUDE.md] for details.


## Features

- Fast columnwise correlation using Einstein summation (einsum) notation
- Automatic GPU acceleration (CUDA/MPS) when available
- Compatible with NumPy arrays
- Command-line interface for easy usage
- Optimized using opt_einsum and PyTorch backends

## Installation

```bash
# Install in development mode
uv pip install -e .

# Or using pip
pip install -e .
```

## Usage

### Python API

```python
import numpy as np
from einsumcorr import optcorr

# Single matrix correlation (self-correlation)
X = np.random.randn(100, 5)
corr_matrix = optcorr(X)
print(corr_matrix.shape)  # (5, 5)

# Two matrix correlation (cross-correlation)
X = np.random.randn(100, 5)
Y = np.random.randn(100, 3)
corr_matrix = optcorr(X, Y)
print(corr_matrix.shape)  # (5, 3)
```

### Command Line Interface

```bash
# Single matrix correlation
einsumcorr data.csv

# Save output to file
einsumcorr data.csv --output correlations.csv

# Cross-correlation between two matrices
einsumcorr matrix1.csv matrix2.csv

# Custom delimiter
einsumcorr data.tsv --delimiter '\t'

# Show help
einsumcorr --help
```

## Performance

The package automatically detects and uses available GPU acceleration:
- CUDA for NVIDIA GPUs
- Metal Performance Shaders (MPS) for Apple Silicon
- Falls back to CPU if no GPU is available

Einstein summation notation provides efficient computation for large matrices, especially when combined with GPU acceleration.

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=einsumcorr

# Run performance benchmarks
uv run pytest tests/test_performance.py -v -s
```

## Requirements

- Python >= 3.12
- NumPy >= 1.24.0
- PyTorch >= 2.0.0
- opt-einsum >= 3.3.0

