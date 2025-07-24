"""Mixin classes providing common BitArray functionality."""

from __future__ import annotations

import struct
from typing import Any

from .bitarray import BitArray


class BitArrayCommonMixin(BitArray):
    """Mixin providing common methods that can be implemented using the BitArray
    protocol.

    This mixin provides default implementations for methods that can be expressed
    in terms of the core BitArray protocol methods (__iter__, __len__, etc.).

    Classes using this mixin must implement the BitArray protocol.
    """

    @classmethod
    def from_float(cls, value: float) -> Any:
        """Convert a floating-point number to a bit array.

        Args:
            value (float): The floating-point number to convert.
        Returns:
            BitArray: A BitArray representing the bits of the floating-point number.
        """
        # Pack as double precision (64 bits)
        packed = struct.pack("!d", value)
        # Convert to boolean list
        bits = [bool((byte >> bit) & 1) for byte in packed for bit in range(7, -1, -1)]
        return cls(bits)

    @classmethod
    def from_signed_int(cls, value: int, length: int) -> Any:
        """Convert a signed integer to a bit array using off-set binary representation.

        Args:
            value (int): The signed integer to convert.
            length (int): The length of the resulting bit array.
        Returns:
            BitArray: A BitArray representing the bits of the signed integer.
        Raises:
            AssertionError: If the value is out of range for the specified length.
        """
        half = 1 << (length - 1)
        max_value = half - 1
        min_value = -half

        assert (
            min_value <= value <= max_value
        ), "Value out of range for specified length."

        # Convert to unsigned integer representation
        unsigned_value = value - half

        bits = [(unsigned_value >> i) & 1 == 1 for i in range(length - 1, -1, -1)]
        return cls(bits)

    def __str__(self) -> str:
        """Return a string representation of the bits."""
        # This assumes self implements __iter__ as per the BitArray protocol
        return "".join("1" if bit else "0" for bit in self)  # type: ignore

    def __eq__(self, other: object) -> bool:
        """Check equality with another BitArray or list."""
        if hasattr(other, "__iter__") and hasattr(other, "__len__"):
            if len(self) != len(other):  # type: ignore
                return False
            return all(a == b for a, b in zip(self, other))  # type: ignore
        return False

    def __bool__(self) -> bool:
        """Return True if any bit is set."""
        return self.any()

    def any(self) -> bool:
        """Return True if any bit is set to True."""
        return any(self)  # type: ignore

    def all(self) -> bool:
        """Return True if all bits are set to True."""
        return all(self)  # type: ignore

    def count(self, value: bool = True) -> int:
        """Count the number of bits set to the specified value."""
        return sum(1 for bit in self if bit == value)  # type: ignore

    def reverse(self) -> Any:
        """Return a new BitArray with the bits in reverse order."""
        return self.__class__(list(self)[::-1])  # type: ignore
