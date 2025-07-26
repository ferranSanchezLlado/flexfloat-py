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
    BitArrayType,
    ListBoolBitArray,
    ListInt64BitArray,
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
    "ListBoolBitArray",
    "ListInt64BitArray",
    "BigIntBitArray",
    "create_bitarray",
    "set_default_implementation",
    "get_available_implementations",
    "parse_bitarray",
]
