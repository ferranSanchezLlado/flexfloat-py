"""BitArray protocol definition for the flexfloat package."""

from __future__ import annotations

from typing import Iterator, Protocol, overload, runtime_checkable


@runtime_checkable
class BitArray(Protocol):
    """Protocol defining the interface for BitArray implementations.

    This protocol defines all the methods and properties that a BitArray
    implementation must provide.
    """

    def __init__(self, bits: list[bool] | None = None) -> None:
        """Initialize a BitArray.

        Args:
            bits: Initial list of boolean values. Defaults to empty list.
        """
        ...

    @classmethod
    def from_float(cls, value: float) -> BitArray:
        """Convert a floating-point number to a bit array.

        Args:
            value (float): The floating-point number to convert.
        Returns:
            BitArrayProtocol: A BitArray representing the bits of the floating-point
                number.
        """
        ...

    @classmethod
    def from_signed_int(cls, value: int, length: int) -> BitArray:
        """Convert a signed integer to a bit array using off-set binary representation.

        Args:
            value (int): The signed integer to convert.
            length (int): The length of the resulting bit array.
        Returns:
            BitArrayProtocol: A BitArray representing the bits of the signed integer.
        Raises:
            AssertionError: If the value is out of range for the specified length.
        """
        ...

    @classmethod
    def zeros(cls, length: int) -> BitArray:
        """Create a BitArray filled with zeros.

        Args:
            length: The length of the bit array.
        Returns:
            BitArrayProtocol: A BitArray filled with False values.
        """
        ...

    @classmethod
    def ones(cls, length: int) -> BitArray:
        """Create a BitArray filled with ones.

        Args:
            length: The length of the bit array.
        Returns:
            BitArrayProtocol: A BitArray filled with True values.
        """
        ...

    @staticmethod
    def parse_bitarray(bitstring: str) -> BitArray:
        """Parse a string of bits (with optional spaces) into a BitArray instance."""
        ...

    def to_float(self) -> float:
        """Convert a 64-bit array to a floating-point number.

        Returns:
            float: The floating-point number represented by the bit array.
        Raises:
            AssertionError: If the bit array is not 64 bits long.
        """
        ...

    def to_int(self) -> int:
        """Convert the bit array to an unsigned integer.

        Returns:
            int: The integer represented by the bit array.
        """
        ...

    def to_signed_int(self) -> int:
        """Convert a bit array into a signed integer using off-set binary
        representation.

        Returns:
            int: The signed integer represented by the bit array.
        Raises:
            AssertionError: If the bit array is empty.
        """
        ...

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
            BitArrayProtocol: A new BitArray with the bits shifted and filled.
        """
        ...

    def copy(self) -> BitArray:
        """Create a copy of the bit array.

        Returns:
            BitArrayProtocol: A new BitArray with the same bits.
        """
        ...

    def __len__(self) -> int:
        """Return the length of the bit array."""
        ...

    @overload
    def __getitem__(self, index: int) -> bool: ...
    @overload
    def __getitem__(self, index: slice) -> BitArray: ...

    def __getitem__(self, index: int | slice) -> bool | BitArray:
        """Get an item or slice from the bit array."""
        ...

    @overload
    def __setitem__(self, index: int, value: bool) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: BitArray | list[bool]) -> None: ...

    def __setitem__(
        self, index: int | slice, value: bool | list[bool] | BitArray
    ) -> None:
        """Set an item or slice in the bit array."""
        ...

    def __iter__(self) -> Iterator[bool]:
        """Iterate over the bits in the array."""
        ...

    def __add__(self, other: BitArray | list[bool]) -> BitArray:
        """Concatenate two bit arrays."""
        ...

    def __radd__(self, other: list[bool]) -> BitArray:
        """Reverse concatenation with a list."""
        ...

    def __eq__(self, other: object) -> bool:
        """Check equality with another BitArray or list."""
        ...

    def __bool__(self) -> bool:
        """Return True if any bit is set."""
        ...

    def __repr__(self) -> str:
        """Return a string representation of the BitArray."""
        ...

    def __str__(self) -> str:
        """Return a string representation of the bits."""
        ...

    def any(self) -> bool:
        """Return True if any bit is set to True."""
        ...

    def all(self) -> bool:
        """Return True if all bits are set to True."""
        ...

    def count(self, value: bool = True) -> int:
        """Count the number of bits set to the specified value."""
        ...

    def reverse(self) -> BitArray:
        """Return a new BitArray with the bits in reverse order."""
        ...
