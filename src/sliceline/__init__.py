"""
SliceLine package for slice discovery in machine learning models.
"""

from .core import SliceLineR, SliceLineRplus, NaivePPIestimator
from .utils import (
    run_three_approaches,
    run_two_approaches,
    clean_slice_results
)

__version__ = "1.0.0"
__all__ = [
    "SliceLineR",
    "SliceLineRplus", 
    "NaivePPIestimator",
    "run_three_approaches",
    "run_two_approaches", 
    "clean_slice_results"
]