"""BitArray implementation for the flexfloat package.

This module provides a factory and utilities for working with different BitArray
implementations, including:
    - ListBoolBitArray: List of booleans (default, flexible, easy to use)
    - ListInt64BitArray: List of int64 chunks (memory efficient for large arrays)
    - BigIntBitArray: Single Python int (arbitrary size, efficient for very large
        arrays)

Example:
    from flexfloat.bitarray import create_bitarray
    ba = create_bitarray('int64', [True, False, True])
    print(type(ba).__name__)
    # Output: ListInt64BitArray
"""

from __future__ import annotations

from typing import Dict, Type

from .bitarray import BitArray
from .bitarray_bigint import BigIntBitArray
from .bitarray_bool import ListBoolBitArray
from .bitarray_int64 import ListInt64BitArray
from .bitarray_mixins import BitArrayCommonMixin

# Type alias for the default BitArray implementation
BitArrayType: Type[BitArray] = ListBoolBitArray

# Available implementations
IMPLEMENTATIONS: Dict[str, Type[BitArray]] = {
    "bool": ListBoolBitArray,
    "int64": ListInt64BitArray,
    "bigint": BigIntBitArray,
}


def create_bitarray(
    implementation: str = "bool", bits: list[bool] | None = None
) -> BitArray:
    """Factory function to create a BitArray with the specified implementation.

    Args:
        implementation (str, optional): The implementation to use ("bool", "int64", or
            "bigint"). Defaults to "bool".
        bits (list[bool] | None, optional): Initial list of boolean values. Defaults to
            None.

    Returns:
        BitArray: A BitArray instance using the specified implementation.

    Raises:
        ValueError: If the implementation is not supported.
    """
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(
            f"Unknown implementation '{implementation}'. "
            f"Available: {list(IMPLEMENTATIONS.keys())}"
        )

    return IMPLEMENTATIONS[implementation].from_bits(bits)


def set_default_implementation(implementation: str) -> None:
    """Set the default BitArray implementation.

    Args:
        implementation (str): The implementation to use as default ("bool", "int64", or
            "bigint").

    Raises:
        ValueError: If the implementation is not supported.
    """
    global BitArrayType

    if implementation not in IMPLEMENTATIONS:
        raise ValueError(
            f"Unknown implementation '{implementation}'. "
            f"Available: {list(IMPLEMENTATIONS.keys())}"
        )

    BitArrayType = IMPLEMENTATIONS[implementation]


def get_available_implementations() -> list[str]:
    """Get the list of available BitArray implementations.

    Returns:
        list[str]: List of available implementation names.
    """
    return list(IMPLEMENTATIONS.keys())


# Maintain backward compatibility by exposing the methods as module-level functions
def parse_bitarray(bitstring: str) -> BitArray:
    """Parse a string of bits (with optional spaces) into a BitArray instance.

    Args:
        bitstring (str): A string of bits, e.g., "1010 1100".

    Returns:
        BitArray: A BitArray instance created from the bit string.
    """
    return BitArrayType.parse_bitarray(bitstring)


__all__ = [
    "BitArray",
    "ListBoolBitArray",
    "ListInt64BitArray",
    "BigIntBitArray",
    "BitArrayCommonMixin",
    "create_bitarray",
    "set_default_implementation",
    "get_available_implementations",
    "parse_bitarray",
]
