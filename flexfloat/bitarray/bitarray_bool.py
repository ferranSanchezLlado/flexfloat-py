"""List-based BitArray implementation for the flexfloat package."""

from __future__ import annotations

import struct
from typing import Iterator, overload

from .bitarray import BitArray
from .bitarray_mixins import BitArrayCommonMixin


class ListBoolBitArray(BitArrayCommonMixin):
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
    def zeros(cls, length: int) -> ListBoolBitArray:
        """Create a BitArray filled with zeros.

        Args:
            length: The length of the bit array.
        Returns:
            ListBitArray: A BitArray filled with False values.
        """
        return cls([False] * length)

    @classmethod
    def ones(cls, length: int) -> ListBoolBitArray:
        """Create a BitArray filled with ones.

        Args:
            length: The length of the bit array.
        Returns:
            ListBitArray: A BitArray filled with True values.
        """
        return cls([True] * length)

    @staticmethod
    def parse_bitarray(bitstring: str) -> ListBoolBitArray:
        """Parse a string of bits (with optional spaces) into a BitArray instance."""
        bits = [c == "1" for c in bitstring if c in "01"]
        return ListBoolBitArray(bits)

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
        float_value = struct.unpack("!d", bytes(byte_values))[0]
        return float_value  # type: ignore

    def to_int(self) -> int:
        """Convert the bit array to an unsigned integer.

        Returns:
            int: The integer represented by the bit array.
        """
        return sum((1 << i) for i, bit in enumerate(reversed(self._bits)) if bit)

    def copy(self) -> ListBoolBitArray:
        """Create a copy of the bit array.

        Returns:
            ListBitArray: A new BitArray with the same bits.
        """
        return ListBoolBitArray(self._bits.copy())

    def __len__(self) -> int:
        """Return the length of the bit array."""
        return len(self._bits)

    @overload
    def __getitem__(self, index: int) -> bool: ...
    @overload
    def __getitem__(self, index: slice) -> ListBoolBitArray: ...

    def __getitem__(self, index: int | slice) -> bool | ListBoolBitArray:
        """Get an item or slice from the bit array."""
        if isinstance(index, slice):
            return ListBoolBitArray(self._bits[index])
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
                self._bits[index] = list(value)
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

    def __add__(self, other: BitArray | list[bool]) -> ListBoolBitArray:
        """Concatenate two bit arrays."""
        if isinstance(other, BitArray):
            return ListBoolBitArray(self._bits + list(other))
        return ListBoolBitArray(self._bits + other)

    def __radd__(self, other: list[bool]) -> ListBoolBitArray:
        """Reverse concatenation with a list."""
        return ListBoolBitArray(other + self._bits)

    def __repr__(self) -> str:
        """Return a string representation of the BitArray."""
        return f"ListBoolBitArray({self._bits})"
