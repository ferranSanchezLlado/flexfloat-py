"""BigFloat - A library for arbitrary precision floating point arithmetic.

This package provides the BigFloat class for handling floating-point numbers
with growable exponents and fixed-size fractions.
"""

from .core import BigFloat
from .types import BitArray, Number
from .utils import (
    bitarray_to_float,
    bitarray_to_int,
    bitarray_to_signed_int,
    float_to_bitarray,
    shift_bitarray,
    signed_int_to_bitarray,
)

__version__ = "0.1.0"
__author__ = "BigFloat Contributors"

__all__ = [
    "BigFloat",
    "Number",
    "BitArray",
    "float_to_bitarray",
    "bitarray_to_float",
    "bitarray_to_int",
    "bitarray_to_signed_int",
    "signed_int_to_bitarray",
    "shift_bitarray",
]
