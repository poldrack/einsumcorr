"""einsumcorr: Efficient columnwise correlation using Einstein summation."""

from .optcorr import optcorr
from .cli import main

__version__ = "0.1.0"
__all__ = ["optcorr", "main"]