# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

einsumcorr is a Python package (version 0.1.0) that defines a module for computing columnwise correlations between two matrices (or a matrix and itself) using the Einstein summation notation.  An example of how to compute a correlation using this method can be found at https://github.com/ikizhvatov/efficient-columnwise-correlation/blob/master/columnwise_corrcoef_perf.py.  This should be implemented using the opt_einsum package (https://github.com/dgasmith/opt_einsum) with PyTorch.  If a GPU is available the code should offload to it to accelerate the computations; the code should automativally detect either cuda or mps backends and use them if available.

The module should define a single main function called `optcorr` that takes in one or two numpy matrices as arguments.  If two matrices are provided, then the function computes the correlation between columns across the two matrices.  The shape of the resulting correlation coefficient for two matrices shaped N_x1 X N_y1 and N_x2 X N_y2 would be N_y1 X N_y2.  If a single matrix is provided, then the correlation coefficient should be computed between all columns within that matrix.

In addition to unit tests, the results should be compared to the results of a columnwise correlation using the standard computation without Einstein summation notation.  

## Project Structure

```
einsumcorr/
├── src/
│   └── einsumcorr/
│       └── __init__.py    # Main package module
├── pyproject.toml          # Project configuration and dependencies
└── README.md               # Project documentation (currently empty)
```

## Development Commands

### Package Management

The project uses `uv` for package management.  The virtual environment can be activated using `source .venv/bin/activate`.

- **Install package in development mode**: `uv pip install -e .`
- **Build package**: `python -m build`
- **Install dependencies**: `uv add <package name>`(to add a single package) or `uv sync` (to install any dependencies listed in pyproject.toml)

## Entry Point

The package defines a console script entry point `einsumcorr` that calls `einsumcorr:main`, though the main function is not yet implemented.

## Development strategy

- Use a test-driven development strategy, developing tests prior to generating solutions to the tests.
  - Run the tests and ensure that they fail prior to generating any solutions.
  - Write code that passes the tests. 
  - IMPORTANT: Do not modify the tests simply so that the code passes. Only modify the tests if you identify a specific error in the test.


## Notes for Development

- Think about the problem before generating code.
- Always add a smoke test for the main() function.
- Prefer reliance on widely used packages (such as numpy, pandas, and scikit-learn); avoid unknown packages from Github.
- Do not include code in __init__.py files.
- Use pytest for testing.
- Write code that is clean and modular.  Prefer shorter functions/methods over longer ones.
- Use functions rather than classes for tests.  Use pytest fixtures to share resources between tests.
- Create code that will work in Python 3.12 or later
