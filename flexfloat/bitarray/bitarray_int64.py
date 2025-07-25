"""Memory-efficient int64-based BitArray implementation for the flexfloat package."""

from __future__ import annotations

import struct
from typing import Iterator, overload

from .bitarray import BitArray
from .bitarray_mixins import BitArrayCommonMixin


class ListInt64BitArray(BitArrayCommonMixin):
    """A memory-efficient bit array class using a list of int64 values.

    This implementation packs 64 bits per integer, making it more memory efficient
    for large bit arrays compared to the boolean list implementation.
    """

    def __init__(self, chunks: list[int] | None = None, length: int = 0):
        """Initialize a ListInt64BitArray.

        Args:
            bits: Initial list of int64 chunks. Defaults to empty list.
            length: The amount of bits in the array. Defaults to 0.
        Raises:
            ValueError: If length is negative.
        """
        chunks = chunks or []
        if length < 0:
            raise ValueError("Length must be non-negative")
        self._length: int = length
        self._chunks: list[int] = chunks

    @classmethod
    def from_bits(cls, bits: list[bool] | None = None) -> ListInt64BitArray:
        """Create a BitArray from a list of boolean values.

        Args:
            bits: List of boolean values.
                (Defaults to None, which creates an empty BitArray.)
        Returns:
            ListInt64BitArray: A BitArray created from the bits.
        """
        if bits is None:
            return cls()
        chunks: list[int] = []

        for i in range(0, len(bits), 64):
            chunk = 0
            chunk_end = min(i + 64, len(bits))
            for j in range(i, chunk_end):
                if bits[j]:
                    chunk |= 1 << (63 - (j - i))
            chunks.append(chunk)

        return cls(chunks, len(bits))

    @classmethod
    def zeros(cls, length: int) -> ListInt64BitArray:
        """Create a BitArray filled with zeros.

        Args:
            length: The length of the bit array.
        Returns:
            Int64BitArray: A BitArray filled with False values.
        """
        return cls([0] * ((length + 63) // 64), length)

    @classmethod
    def ones(cls, length: int) -> ListInt64BitArray:
        """Create a BitArray filled with ones.

        Args:
            length: The length of the bit array.
        Returns:
            Int64BitArray: A BitArray filled with True values.
        """
        chunks = [0xFFFFFFFFFFFFFFFF] * (length // 64)
        if length % 64 > 0:
            partial_chunk = (1 << (length % 64)) - 1
            partial_chunk <<= 64 - (length % 64)
            chunks.append(partial_chunk)
        return cls(chunks, length)

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

        # Branchless version
        mask = 1 << bit_position
        self._chunks[chunk_index] ^= (-value ^ self._chunks[chunk_index]) & mask

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
        float_value = struct.unpack("!d", bytes(byte_values))[0]
        return float_value  # type: ignore

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

    def copy(self) -> ListInt64BitArray:
        """Create a copy of the bit array.

        Returns:
            Int64BitArray: A new BitArray with the same bits.
        """
        return ListInt64BitArray(self._chunks.copy(), self._length)

    def __len__(self) -> int:
        """Return the length of the bit array."""
        return self._length

    @overload
    def __getitem__(self, index: int) -> bool: ...
    @overload
    def __getitem__(self, index: slice) -> ListInt64BitArray: ...

    def __getitem__(self, index: int | slice) -> bool | ListInt64BitArray:
        """Get an item or slice from the bit array."""
        if isinstance(index, slice):
            start, stop, step = index.indices(self._length)
            bits = [self._get_bit(i) for i in range(start, stop, step)]
            return ListInt64BitArray.from_bits(bits)
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

    def __add__(self, other: BitArray | list[bool]) -> ListInt64BitArray:
        """Concatenate two bit arrays."""
        if isinstance(other, BitArray):
            return ListInt64BitArray.from_bits(list(self) + list(other))
        return ListInt64BitArray.from_bits(list(self) + other)

    def __radd__(self, other: list[bool]) -> ListInt64BitArray:
        """Reverse concatenation with a list."""
        return ListInt64BitArray.from_bits(other + list(self))

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
        return f"ListInt64BitArray({list(self)})"

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
