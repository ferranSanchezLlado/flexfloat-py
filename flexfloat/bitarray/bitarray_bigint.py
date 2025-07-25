"""Infinite-size int-based BitArray implementation for the flexfloat package."""

from __future__ import annotations

import struct
from typing import Iterator, overload

from .bitarray import BitArray
from .bitarray_mixins import BitArrayCommonMixin


class BigIntBitArray(BitArrayCommonMixin):
    """A memory-efficient bit array class using Python's infinite-size int.

    This implementation stores all bits as a single Python integer, leveraging
    Python's arbitrary precision arithmetic for potentially unlimited size.
    Since Python integers are arbitrary precision, this can handle bit arrays
    of any size limited only by available memory.
    """

    def __init__(self, bits: list[bool] | None = None):
        """Initialize a BitArray.

        Args:
            bits: Initial list of boolean values. Defaults to empty list.
        """
        if bits is None:
            bits = []

        self._length = len(bits)
        self._value = 0

        # Pack bits into a single integer
        # Most significant bit is at index 0
        for i, bit in enumerate(bits):
            if bit:
                self._value |= 1 << (self._length - 1 - i)

    @classmethod
    def zeros(cls, length: int) -> BigIntBitArray:
        """Create a BitArray filled with zeros.

        Args:
            length: The length of the bit array.
        Returns:
            BigIntBitArray: A BitArray filled with False values.
        """
        instance = cls.__new__(cls)
        instance._length = length
        instance._value = 0
        return instance

    @classmethod
    def ones(cls, length: int) -> BigIntBitArray:
        """Create a BitArray filled with ones.

        Args:
            length: The length of the bit array.
        Returns:
            BigIntBitArray: A BitArray filled with True values.
        """
        instance = cls.__new__(cls)
        instance._length = length
        # Create a mask with all bits set
        instance._value = (1 << length) - 1 if length > 0 else 0
        return instance

    @staticmethod
    def parse_bitarray(bitstring: str) -> BigIntBitArray:
        """Parse a string of bits (with optional spaces) into a BitArray instance."""
        bits = [c == "1" for c in bitstring if c in "01"]
        return BigIntBitArray(bits)

    def _get_bit(self, index: int) -> bool:
        """Get a single bit at the specified index."""
        if index < 0 or index >= self._length:
            raise IndexError("Bit index out of range")

        bit_position = self._length - 1 - index
        return bool(self._value & (1 << bit_position))

    def _set_bit(self, index: int, value: bool) -> None:
        """Set a single bit at the specified index."""
        if index < 0 or index >= self._length:
            raise IndexError("Bit index out of range")

        bit_position = self._length - 1 - index

        if value:
            self._value |= 1 << bit_position
        else:
            self._value &= ~(1 << bit_position)

    def to_float(self) -> float:
        """Convert a 64-bit array to a floating-point number.

        Returns:
            float: The floating-point number represented by the bit array.
        Raises:
            AssertionError: If the bit array is not 64 bits long.
        """
        assert self._length == 64, "Bit array must be 64 bits long."

        # Convert integer to bytes (big-endian)
        byte_values = bytearray()
        value = self._value
        for i in range(8):
            byte = (value >> (56 - i * 8)) & 0xFF
            byte_values.append(byte)

        # Unpack as double precision (64 bits)
        float_value = struct.unpack("!d", bytes(byte_values))[0]
        return float_value  # type: ignore

    def to_int(self) -> int:
        """Convert the bit array to an unsigned integer.

        Returns:
            int: The integer represented by the bit array.
        """
        return self._value

    def copy(self) -> BigIntBitArray:
        """Create a copy of the bit array.

        Returns:
            BigIntBitArray: A new BitArray with the same bits.
        """
        result = BigIntBitArray.__new__(BigIntBitArray)
        result._length = self._length
        result._value = self._value
        return result

    def __len__(self) -> int:
        """Return the length of the bit array."""
        return self._length

    @overload
    def __getitem__(self, index: int) -> bool: ...
    @overload
    def __getitem__(self, index: slice) -> BigIntBitArray: ...

    def __getitem__(self, index: int | slice) -> bool | BigIntBitArray:
        """Get an item or slice from the bit array."""
        if isinstance(index, int):
            return self._get_bit(index)
        else:  # slice
            start, stop, step = index.indices(self._length)
            if step != 1:
                # Handle step != 1 by extracting individual bits
                bits: list[bool] = [self._get_bit(i) for i in range(start, stop, step)]
            else:
                # Efficient slice extraction for step == 1
                bits = []
                for i in range(start, stop):
                    bits.append(self._get_bit(i))
            return BigIntBitArray(bits)

    @overload
    def __setitem__(self, index: int, value: bool) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: BitArray | list[bool]) -> None: ...

    def __setitem__(
        self, index: int | slice, value: bool | list[bool] | BitArray
    ) -> None:
        """Set an item or slice in the bit array."""
        if isinstance(index, int):
            if not isinstance(value, bool):
                raise TypeError("Value must be bool for single index")
            self._set_bit(index, value)
        else:  # slice
            start, stop, step = index.indices(self._length)

            # Convert value to list of bools
            if isinstance(value, bool):
                raise TypeError("Cannot assign bool to slice")
            elif hasattr(value, "__iter__"):
                value_list = list(value)
            else:
                raise TypeError("Value must be iterable for slice assignment")

            if step != 1:
                # Handle step != 1
                indices = list(range(start, stop, step))
                if len(value_list) != len(indices):
                    raise ValueError("Value length doesn't match slice length")
                for i, v in zip(indices, value_list):
                    self._set_bit(i, bool(v))
            else:
                # Handle step == 1
                if len(value_list) != (stop - start):
                    raise ValueError("Value length doesn't match slice length")
                for i, v in enumerate(value_list):
                    self._set_bit(start + i, bool(v))

    def __iter__(self) -> Iterator[bool]:
        """Iterate over the bits in the array."""
        for i in range(self._length):
            yield self._get_bit(i)

    def __add__(self, other: BitArray | list[bool]) -> BigIntBitArray:
        """Concatenate two bit arrays."""
        if hasattr(other, "__iter__"):
            other_bits = list(other)
        else:
            raise TypeError("Can only concatenate with iterable")

        all_bits = list(self) + other_bits
        return BigIntBitArray(all_bits)

    def __radd__(self, other: list[bool]) -> BigIntBitArray:
        """Reverse concatenation with a list."""
        if hasattr(other, "__iter__"):
            other_bits = list(other)
        else:
            raise TypeError("Can only concatenate with iterable")

        all_bits = other_bits + list(self)
        return BigIntBitArray(all_bits)

    def __repr__(self) -> str:
        """Return a string representation of the BitArray."""
        return f"BigIntBitArray({list(self)})"
