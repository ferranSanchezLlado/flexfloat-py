"""FlexFloat - A library for arbitrary precision floating point arithmetic.

This package provides the FlexFloat class for handling floating-point numbers
with growable exponents and fixed-size fractions.
"""

from .bitarray import (
    BitArray,
    BitArrayType,
    Int64BitArray,
    ListBitArray,
    create_bitarray,
    get_available_implementations,
    parse_bitarray,
    set_default_implementation,
)
from .core import FlexFloat

__version__ = "0.1.3"
__author__ = "Ferran Sanchez Llado"

__all__ = [
    "FlexFloat",
    "BitArrayType",
    "BitArray",
    "ListBitArray",
    "Int64BitArray",
    "create_bitarray",
    "set_default_implementation",
    "get_available_implementations",
    "parse_bitarray",
]
