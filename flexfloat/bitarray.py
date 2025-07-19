"""BitArray implementation for the flexfloat package."""

from __future__ import annotations

import struct
from typing import Iterator, overload


class BitArray:
    """A bit array class that encapsulates a list of booleans with utility methods.

    This class provides all the functionality previously available through utility
    functions, now encapsulated as methods for better object-oriented design.
    """

    def __init__(self, bits: list[bool] | None = None):
        """Initialize a BitArray.

        Args:
            bits: Initial list of boolean values. Defaults to empty list.
        """
        self._bits = bits if bits is not None else []

    @classmethod
    def from_float(cls, value: float) -> BitArray:
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
    def from_signed_int(cls, value: int, length: int) -> BitArray:
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

    @classmethod
    def zeros(cls, length: int) -> BitArray:
        """Create a BitArray filled with zeros.

        Args:
            length: The length of the bit array.
        Returns:
            BitArray: A BitArray filled with False values.
        """
        return cls([False] * length)

    @classmethod
    def ones(cls, length: int) -> BitArray:
        """Create a BitArray filled with ones.

        Args:
            length: The length of the bit array.
        Returns:
            BitArray: A BitArray filled with True values.
        """
        return cls([True] * length)

    @staticmethod
    def parse_bitarray(bitstring: str) -> BitArray:
        """Parse a string of bits (with optional spaces) into a BitArray instance."""
        bits = [c == "1" for c in bitstring if c in "01"]
        return BitArray(bits)

    def to_float(self) -> float:
        """Convert a 64-bit array to a floating-point number.

        Returns:
            float: The floating-point number represented by the bit array.
        Raises:
            AssertionError: If the bit array is not 64 bits long.
        """
        assert len(self._bits) == 64, "Bit array must be 64 bits long."

        byte_values = bytearray()
        for i in range(0, 64, 8):
            byte = 0
            for j in range(8):
                if self._bits[i + j]:
                    byte |= 1 << (7 - j)
            byte_values.append(byte)
        # Unpack as double precision (64 bits)
        return struct.unpack("!d", bytes(byte_values))[0]  # type: ignore

    def to_int(self) -> int:
        """Convert the bit array to an unsigned integer.

        Returns:
            int: The integer represented by the bit array.
        """
        return sum((1 << i) for i, bit in enumerate(reversed(self._bits)) if bit)

    def to_signed_int(self) -> int:
        """Convert a bit array into a signed integer using off-set binary
        representation.

        Returns:
            int: The signed integer represented by the bit array.
        Raises:
            AssertionError: If the bit array is empty.
        """
        assert len(self._bits) > 0, "Bit array must not be empty."

        int_value = self.to_int()
        # Half of the maximum value
        bias = 1 << (len(self._bits) - 1)
        # If the sign bit is set, subtract the bias
        return int_value - bias

    def shift(self, shift_amount: int, fill: bool = False) -> BitArray:
        """Shift the bit array left or right by a specified number of bits.

        This function shifts the bits in the array, filling in new bits with the
        specified fill value.
        If the value is positive, it shifts left; if negative, it shifts right.
        Fills the new bits with the specified fill value (default is False).

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
        if abs(shift_amount) > len(self._bits):
            new_bits = [fill] * len(self._bits)
        elif shift_amount > 0:
            new_bits = [fill] * shift_amount + self._bits[:-shift_amount]
        else:
            new_bits = self._bits[-shift_amount:] + [fill] * (-shift_amount)
        return BitArray(new_bits)

    def copy(self) -> BitArray:
        """Create a copy of the bit array.

        Returns:
            BitArray: A new BitArray with the same bits.
        """
        return BitArray(self._bits.copy())

    def __len__(self) -> int:
        """Return the length of the bit array."""
        return len(self._bits)

    @overload
    def __getitem__(self, index: int) -> bool: ...
    @overload
    def __getitem__(self, index: slice) -> BitArray: ...

    def __getitem__(self, index: int | slice) -> bool | BitArray:
        """Get an item or slice from the bit array."""
        if isinstance(index, slice):
            return BitArray(self._bits[index])
        return self._bits[index]

    @overload
    def __setitem__(self, index: int, value: bool) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: BitArray | list[bool]) -> None: ...

    def __setitem__(
        self, index: int | slice, value: bool | list[bool] | BitArray
    ) -> None:
        """Set an item or slice in the bit array."""
        if isinstance(index, slice):
            if isinstance(value, BitArray):
                self._bits[index] = value._bits
            elif isinstance(value, list):
                self._bits[index] = value
            else:
                raise TypeError("Cannot assign a single bool to a slice")
            return
        if isinstance(value, bool):
            self._bits[index] = value
        else:
            raise TypeError("Cannot assign a list or BitArray to a single index")

    def __iter__(self) -> Iterator[bool]:
        """Iterate over the bits in the array."""
        return iter(self._bits)

    def __add__(self, other: BitArray | list[bool]) -> BitArray:
        """Concatenate two bit arrays."""
        if isinstance(other, BitArray):
            return BitArray(self._bits + other._bits)
        return BitArray(self._bits + other)

    def __radd__(self, other: list[bool]) -> BitArray:
        """Reverse concatenation with a list."""
        return BitArray(other + self._bits)

    def __eq__(self, other: object) -> bool:
        """Check equality with another BitArray or list."""
        if isinstance(other, BitArray):
            return self._bits == other._bits
        if isinstance(other, list):
            return self._bits == other
        return False

    def __bool__(self) -> bool:
        """Return True if any bit is set."""
        return any(self._bits)

    def __repr__(self) -> str:
        """Return a string representation of the BitArray."""
        return f"BitArray({self._bits})"

    def __str__(self) -> str:
        """Return a string representation of the bits."""
        return "".join("1" if bit else "0" for bit in self._bits)

    def any(self) -> bool:
        """Return True if any bit is set to True."""
        return any(self._bits)

    def all(self) -> bool:
        """Return True if all bits are set to True."""
        return all(self._bits)

    def count(self, value: bool = True) -> int:
        """Count the number of bits set to the specified value."""
        return self._bits.count(value)

    def reverse(self) -> BitArray:
        """Return a new BitArray with the bits in reverse order."""
        return BitArray(self._bits[::-1])
