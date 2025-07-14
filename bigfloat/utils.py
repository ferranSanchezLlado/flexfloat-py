"""Utility functions for bit array operations and conversions."""

from __future__ import annotations

import struct

from .types import BitArray


def float_to_bitarray(value: float) -> BitArray:
    """Convert a floating-point number to a bit array.

    Args:
        value (float): The floating-point number to convert.
    Returns:
        BitArray: A list of booleans representing the bits of the floating-point number.
    """
    # Pack as double precision (64 bits)
    packed = struct.pack("!d", value)
    # Convert to boolean list
    return [bool((byte >> bit) & 1) for byte in packed for bit in range(7, -1, -1)]


def bitarray_to_float(bit_array: BitArray) -> float:
    """Convert a 64-bit array to a floating-point number.

    Args:
        bit_array (BitArray): A list of booleans representing the bits of the floating-point number.
    Returns:
        float: The floating-point number represented by the bit array.
    """
    assert len(bit_array) == 64, "Bit array must be 64 bits long."

    byte_values = bytearray()
    for i in range(0, 64, 8):
        byte = 0
        for j in range(8):
            if bit_array[i + j]:
                byte |= 1 << (7 - j)
        byte_values.append(byte)
    # Unpack as double precision (64 bits)
    return struct.unpack("!d", bytes(byte_values))[0]  # type: ignore


def bitarray_to_int(bit_array: BitArray) -> int:
    """Convert an infinite bit array to an python integer (also unbounded).

    Args:
        bit_array (BitArray): A list of booleans representing the bits of the number.
    Returns:
        int: The integer represented by the bit array.
    """
    return sum((1 << i) for i, bit in enumerate(reversed(bit_array)) if bit)


def bitarray_to_signed_int(bit_array: BitArray) -> int:
    """Convert a bit array into a signed integer using off-set binary representation.

    Args:
        bit_array (BitArray): A list of booleans representing the bits of the signed integer.
    Returns:
        int: The signed integer represented by the bit array.
    Raises:
        AssertionError: If the bit array is empty.
    """
    assert len(bit_array) > 0, "Bit array must not be empty."

    int_value = bitarray_to_int(bit_array)
    # Half of the maximum value
    bias = 1 << (len(bit_array) - 1)
    # If the sign bit is set, subtract the bias
    return int_value - bias


def signed_int_to_bitarray(value: int, length: int) -> BitArray:
    """Convert a signed integer to a bit array using off-set binary representation.

    Args:
        value (int): The signed integer to convert.
        length (int): The length of the resulting bit array.
    Returns:
        BitArray: A list of booleans representing the bits of the signed integer.
    Raises:
        AssertionError: If the value is out of range for the specified length.
    """
    max_value = (1 << length) - 1
    min_value = -(1 << (length - 1))

    assert min_value <= value <= max_value, "Value out of range for specified length."

    # Calculate the bias
    bias = 1 << (length - 1)
    # Convert to unsigned integer representation
    unsigned_value = value + bias

    return [(unsigned_value >> i) & 1 == 1 for i in range(length - 1, -1, -1)]


def shift_bitarray(bit_array: BitArray, shift: int, fill: bool = False) -> BitArray:
    """Shift a bit array left or right by a specified number of bits.

    This function shifts the bits in the array, filling in new bits with the specified fill value.
    If the shift is positive, it shifts left and fills with the fill value at the end.
    If the shift is negative, it shifts right and fills with the fill value at the start

    Args:
        bit_array (BitArray): The bit array to shift.
        shift (int): The number of bits to shift. Positive for left shift, negative for right shift.
        fill (bool): The value to fill in the new bits created by the shift. Defaults to False.
    Returns:
        BitArray: A new bit array with the bits shifted and filled.
    """
    if shift == 0:
        return bit_array
    if abs(shift) > len(bit_array):
        return [fill] * len(bit_array)
    elif shift > 0:
        return [fill] * shift + bit_array[:-shift]
    else:
        return bit_array[-shift:] + [fill] * (-shift)
