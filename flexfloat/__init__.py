"""FlexFloat - A library for arbitrary precision floating point arithmetic.

This package provides the FlexFloat class for handling floating-point numbers
with growable exponents and fixed-size fractions.
"""

from .bitarray import BitArray
from .core import FlexFloat

__version__ = "0.1.2"
__author__ = "Ferran Sanchez Llado"

__all__ = ["FlexFloat", "BitArray"]
