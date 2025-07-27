"""FlexFloat - A library for arbitrary precision floating point arithmetic.

This package provides the FlexFloat class for handling floating-point numbers with
growable exponents and fixed-size fractions. It also provides several BitArray
implementations for efficient bit manipulation.

Example:
    from flexfloat import FlexFloat
    x = FlexFloat(1.5, exponent_length=8, fraction_length=23)
    print(x)
    # Output: FlexFloat(sign=False, exponent=..., fraction=...)

Modules:
    core: Main FlexFloat class implementation
    bitarray: BitArray implementations (bool, int64, bigint)
    types: Type definitions
"""

from .bitarray import (
    BigIntBitArray,
    BitArray,
    ListBoolBitArray,
    ListInt64BitArray,
)
from .core import FlexFloat
from . import math

__version__ = "0.3.0"
__author__ = "Ferran Sanchez Llado"

__all__ = [
    "FlexFloat",
    "BitArray",
    "ListBoolBitArray",
    "ListInt64BitArray",
    "BigIntBitArray",
    "math",
]
