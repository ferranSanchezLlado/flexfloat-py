"""Memory-efficient int64-based BitArray implementation for the flexfloat package."""

from __future__ import annotations

import struct
from typing import Iterator, overload

from .bitarray import BitArray
from .bitarray_mixins import BitArrayCommonMixin


class Int64BitArray(BitArrayCommonMixin):
    """A memory-efficient bit array class using a list of int64 values.

    This implementation packs 64 bits per integer, making it more memory efficient
    for large bit arrays compared to the boolean list implementation.
    """

    def __init__(self, bits: list[bool] | None = None):
        """Initialize a BitArray.

        Args:
            bits: Initial list of boolean values. Defaults to empty list.
        """
        if bits is None:
            bits = []

        self._length = len(bits)
        # Pack bits into int64 chunks (64 bits per int)
        self._chunks: list[int] = []

        for i in range(0, len(bits), 64):
            chunk = 0
            chunk_end = min(i + 64, len(bits))
            for j in range(i, chunk_end):
                if bits[j]:
                    chunk |= 1 << (63 - (j - i))
            self._chunks.append(chunk)

    @classmethod
    def zeros(cls, length: int) -> Int64BitArray:
        """Create a BitArray filled with zeros.

        Args:
            length: The length of the bit array.
        Returns:
            Int64BitArray: A BitArray filled with False values.
        """
        instance = cls.__new__(cls)
        instance._length = length
        instance._chunks = [0] * ((length + 63) // 64)
        return instance

    @classmethod
    def ones(cls, length: int) -> Int64BitArray:
        """Create a BitArray filled with ones.

        Args:
            length: The length of the bit array.
        Returns:
            Int64BitArray: A BitArray filled with True values.
        """
        instance = cls.__new__(cls)
        instance._length = length
        num_full_chunks = length // 64
        remaining_bits = length % 64

        instance._chunks = []

        # Add full chunks of all 1s
        for _ in range(num_full_chunks):
            instance._chunks.append(0xFFFFFFFFFFFFFFFF)  # All 64 bits set

        # Add partial chunk if needed
        if remaining_bits > 0:
            partial_chunk = (1 << remaining_bits) - 1
            partial_chunk <<= 64 - remaining_bits  # Left-align the bits
            instance._chunks.append(partial_chunk)

        return instance

    @staticmethod
    def parse_bitarray(bitstring: str) -> Int64BitArray:
        """Parse a string of bits (with optional spaces) into a BitArray instance."""
        bits = [c == "1" for c in bitstring if c in "01"]
        return Int64BitArray(bits)

    def _get_bit(self, index: int) -> bool:
        """Get a single bit at the specified index."""
        if index < 0 or index >= self._length:
            raise IndexError("Bit index out of range")

        chunk_index = index // 64
        bit_index = index % 64
        bit_position = 63 - bit_index  # Left-aligned

        return bool(self._chunks[chunk_index] & (1 << bit_position))

    def _set_bit(self, index: int, value: bool) -> None:
        """Set a single bit at the specified index."""
        if index < 0 or index >= self._length:
            raise IndexError("Bit index out of range")

        chunk_index = index // 64
        bit_index = index % 64
        bit_position = 63 - bit_index  # Left-aligned

        if value:
            self._chunks[chunk_index] |= 1 << bit_position
        else:
            self._chunks[chunk_index] &= ~(1 << bit_position)

    def to_float(self) -> float:
        """Convert a 64-bit array to a floating-point number.

        Returns:
            float: The floating-point number represented by the bit array.
        Raises:
            AssertionError: If the bit array is not 64 bits long.
        """
        assert self._length == 64, "Bit array must be 64 bits long."

        # Convert first chunk directly to bytes
        chunk = self._chunks[0]
        byte_values = bytearray()
        for i in range(8):
            byte = (chunk >> (56 - i * 8)) & 0xFF
            byte_values.append(byte)

        # Unpack as double precision (64 bits)
        return struct.unpack("!d", bytes(byte_values))[0]  # type: ignore

    def to_int(self) -> int:
        """Convert the bit array to an unsigned integer.

        Returns:
            int: The integer represented by the bit array.
        """
        result = 0
        for i in range(self._length):
            if self._get_bit(i):
                result |= 1 << (self._length - 1 - i)
        return result

    def to_signed_int(self) -> int:
        """Convert a bit array into a signed integer using off-set binary
        representation.

        Returns:
            int: The signed integer represented by the bit array.
        Raises:
            AssertionError: If the bit array is empty.
        """
        assert self._length > 0, "Bit array must not be empty."

        int_value = self.to_int()
        # Half of the maximum value
        bias = 1 << (self._length - 1)
        # If the sign bit is set, subtract the bias
        return int_value - bias

    def shift(self, shift_amount: int, fill: bool = False) -> Int64BitArray:
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
            Int64BitArray: A new BitArray with the bits shifted and filled.
        """
        if shift_amount == 0:
            return self.copy()

        # Convert to bit list for simplicity (can be optimized later)
        bits = list(self)

        if abs(shift_amount) > len(bits):
            new_bits = [fill] * len(bits)
        elif shift_amount > 0:
            new_bits = [fill] * shift_amount + bits[:-shift_amount]
        else:
            new_bits = bits[-shift_amount:] + [fill] * (-shift_amount)

        return Int64BitArray(new_bits)

    def copy(self) -> Int64BitArray:
        """Create a copy of the bit array.

        Returns:
            Int64BitArray: A new BitArray with the same bits.
        """
        instance = Int64BitArray.__new__(Int64BitArray)
        instance._length = self._length
        instance._chunks = self._chunks.copy()  # This creates a shallow copy
        return instance

    def __len__(self) -> int:
        """Return the length of the bit array."""
        return self._length

    @overload
    def __getitem__(self, index: int) -> bool: ...
    @overload
    def __getitem__(self, index: slice) -> Int64BitArray: ...

    def __getitem__(self, index: int | slice) -> bool | Int64BitArray:
        """Get an item or slice from the bit array."""
        if isinstance(index, slice):
            start, stop, step = index.indices(self._length)
            bits = [self._get_bit(i) for i in range(start, stop, step)]
            return Int64BitArray(bits)
        return self._get_bit(index)

    @overload
    def __setitem__(self, index: int, value: bool) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: BitArray | list[bool]) -> None: ...

    def __setitem__(
        self, index: int | slice, value: bool | list[bool] | BitArray
    ) -> None:
        """Set an item or slice in the bit array."""
        if isinstance(index, slice):
            start, stop, step = index.indices(self._length)
            indices = list(range(start, stop, step))

            if isinstance(value, BitArray):
                values = list(value)
            elif isinstance(value, list):
                values = value
            else:
                raise TypeError("Cannot assign a single bool to a slice")

            if len(indices) != len(values):
                raise ValueError("Length mismatch in slice assignment")

            for i, v in zip(indices, values):
                self._set_bit(i, v)
            return

        if isinstance(value, bool):
            self._set_bit(index, value)
        else:
            raise TypeError("Cannot assign a list or BitArray to a single index")

    def __iter__(self) -> Iterator[bool]:
        """Iterate over the bits in the array."""
        for i in range(self._length):
            yield self._get_bit(i)

    def __add__(self, other: BitArray | list[bool]) -> Int64BitArray:
        """Concatenate two bit arrays."""
        if isinstance(other, BitArray):
            return Int64BitArray(list(self) + list(other))
        return Int64BitArray(list(self) + other)

    def __radd__(self, other: list[bool]) -> Int64BitArray:
        """Reverse concatenation with a list."""
        return Int64BitArray(other + list(self))

    def __eq__(self, other: object) -> bool:
        """Check equality with another BitArray or list."""
        if isinstance(other, BitArray):
            if len(self) != len(other):
                return False
            return all(a == b for a, b in zip(self, other))
        if isinstance(other, list):
            return list(self) == other
        return False

    def __bool__(self) -> bool:
        """Return True if any bit is set."""
        return any(chunk != 0 for chunk in self._chunks)

    def __repr__(self) -> str:
        """Return a string representation of the BitArray."""
        return f"Int64BitArray({list(self)})"

    def any(self) -> bool:
        """Return True if any bit is set to True."""
        return any(chunk != 0 for chunk in self._chunks)

    def all(self) -> bool:
        """Return True if all bits are set to True."""
        if self._length == 0:
            return True

        # Check full chunks
        num_full_chunks = self._length // 64
        for i in range(num_full_chunks):
            if self._chunks[i] != 0xFFFFFFFFFFFFFFFF:
                return False

        # Check partial chunk if exists
        remaining_bits = self._length % 64
        if remaining_bits > 0:
            expected_pattern = ((1 << remaining_bits) - 1) << (64 - remaining_bits)
            if self._chunks[-1] != expected_pattern:
                return False

        return True
