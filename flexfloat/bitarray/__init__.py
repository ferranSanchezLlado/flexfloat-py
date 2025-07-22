"""BitArray implementation for the flexfloat package."""

from __future__ import annotations

from typing import Dict, Type

from .bitarray import BitArray
from .bitarray_int64 import Int64BitArray
from .bitarray_list import ListBitArray
from .bitarray_mixins import BitArrayCommonMixin

# Type alias for the default BitArray implementation
BitArrayType: Type[BitArray] = ListBitArray

# Available implementations
IMPLEMENTATIONS: Dict[str, Type[BitArray]] = {
    "list": ListBitArray,
    "int64": Int64BitArray,
}


def create_bitarray(
    implementation: str = "list", bits: list[bool] | None = None
) -> BitArray:
    """Factory function to create a BitArray with the specified implementation.

    Args:
        implementation: The implementation to use ("list" or "int64")
        bits: Initial list of boolean values

    Returns:
        BitArray: A BitArray instance using the specified implementation

    Raises:
        ValueError: If the implementation is not supported
    """
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(
            f"Unknown implementation '{implementation}'. "
            f"Available: {list(IMPLEMENTATIONS.keys())}"
        )

    return IMPLEMENTATIONS[implementation](bits)


def set_default_implementation(implementation: str) -> None:
    """Set the default BitArray implementation.

    Args:
        implementation: The implementation to use as default ("list" or "int64")

    Raises:
        ValueError: If the implementation is not supported
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
        list[str]: List of available implementation names
    """
    return list(IMPLEMENTATIONS.keys())


# Maintain backward compatibility by exposing the methods as module-level functions
def parse_bitarray(bitstring: str) -> BitArray:
    """Parse a string of bits (with optional spaces) into a BitArray instance."""
    return BitArrayType.parse_bitarray(bitstring)


__all__ = [
    "BitArray",
    "ListBitArray",
    "Int64BitArray",
    "BitArrayCommonMixin",
    "create_bitarray",
    "set_default_implementation",
    "get_available_implementations",
    "parse_bitarray",
]
