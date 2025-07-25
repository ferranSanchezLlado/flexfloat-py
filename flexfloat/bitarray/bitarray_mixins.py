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
        return cls.from_bits(bits)

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
        unsigned_value = value + half

        bits = [(unsigned_value >> i) & 1 == 1 for i in range(length - 1, -1, -1)]
        return cls.from_bits(bits)

    @classmethod
    def parse_bitarray(cls, bitstring: str) -> BitArray:
        """Parse a string of bits (with optional spaces) into a BitArray instance.
        Non-valid characters are ignored.


        Args:
            bitstring: A string of bits, e.g., "1010 1100".
        Returns:
            BitArray: A BitArray instance created from the bit string.
        """
        return cls.from_bits([c == "1" for c in bitstring if c in "01"])

    def __str__(self) -> str:
        """Return a string representation of the bits."""
        # This assumes self implements __iter__ as per the BitArray protocol
        return "".join("1" if bit else "0" for bit in self)

    def __eq__(self, other: Any) -> bool:
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
        return any(self)

    def all(self) -> bool:
        """Return True if all bits are set to True."""
        return all(self)

    def count(self, value: bool = True) -> int:
        """Count the number of bits set to the specified value."""
        return sum(1 for bit in self if bit == value)

    def reverse(self) -> Any:
        """Return a new BitArray with the bits in reverse order."""
        return self.from_bits(list(reversed(self)))

    def to_signed_int(self) -> int:
        """Convert a bit array into a signed integer using off-set binary
        representation.

        Returns:
            int: The signed integer represented by the bit array.
        Raises:
            AssertionError: If the bit array is empty.
        """
        assert len(self) > 0, "Bit array must not be empty."

        int_value = self.to_int()
        # Half of the maximum value
        bias = 1 << (len(self) - 1)
        # Subtract the bias to get the signed value
        return int_value - bias

    def shift(self, shift_amount: int, fill: bool = False) -> Any:
        """Shift the bit array left or right by a specified number of bits.

        Args:
            shift_amount (int): The number of bits to shift. Positive for left shift,
                negative for right shift.
            fill (bool): The value to fill in the new bits created by the shift.
                Defaults to False.
        Returns:
            BitArray: A new BitArray with the bits shifted and filled.
        """
        if shift_amount == 0:
            return self.copy()

        # Convert to bit list for consistent behavior with other implementations
        bits = list(self)

        if abs(shift_amount) > len(bits):
            new_bits = [fill] * len(bits)
        elif shift_amount > 0:  # Left shift
            new_bits = [fill] * shift_amount + bits[:-shift_amount]
        else:  # Right shift
            new_bits = bits[-shift_amount:] + [fill] * (-shift_amount)

        return self.from_bits(new_bits)
