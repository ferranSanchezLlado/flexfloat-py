"""BigFloat - A library for arbitrary precision floating point arithmetic.

This package provides the BigFloat class for handling floating-point numbers
with growable exponents and fixed-size fractions.
"""

from .bitarray import BitArray
from .core import BigFloat

__version__ = "0.1.0"
__author__ = "Ferran Sanchez Llado"

__all__ = ["BigFloat", "BitArray"]
