from __future__ import annotations
import math
import struct
from unittest import TestCase

Number = int | float
BitArray = list[bool]


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
    return struct.unpack("!d", bytes(byte_values))[0]


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


class BigFloat:
    """A class to represent a floating-point number with growable exponent and fixed-size fraction.
    This class is designed to handle very large or very small numbers by adjusting the exponent dynamically.
    While keeping the mantissa (fraction) fixed in size.

    This class follows the IEEE 754 double-precision floating-point format,
    but extends it to allow for a growable exponent and a fixed-size fraction.

    Attributes:
        sign (bool): The sign of the number (True for negative, False for positive).
        exponent (BitArray): A growable bit array representing the exponent (uses off-set binary representation).
        fraction (BitArray): A fixed-size bit array representing the fraction (mantissa) of the number.
    """

    def __init__(
        self,
        sign: bool = False,
        exponent: BitArray | None = None,
        fraction: BitArray | None = None,
    ):
        """Initialize a BigFloat instance.

        Args:
            sign (bool): The sign of the number (True for negative, False for positive).
            exponent (BitArray | None): The exponent bit array. If None, defaults to a zero exponent.
            fraction (BitArray | None): The fraction bit array. If None, defaults to a zero fraction.
        """
        self.sign = sign
        self.exponent = exponent if exponent is not None else [False] * 11
        self.fraction = fraction if fraction is not None else [False] * 52

    @classmethod
    def from_float(cls, value: Number) -> BigFloat:
        """Create a BigFloat instance from a number.

        Args:
            value (Number): The number to convert to BigFloat.
        Returns:
            BigFloat: A new BigFloat instance representing the number.
        """
        value = float(value)
        bits = float_to_bitarray(value)

        return cls(sign=bits[0], exponent=bits[1:12], fraction=bits[12:64])

    def to_float(self) -> float:
        """Convert the BigFloat instance back to a floating-point number.

        Returns:
            float: The floating-point number represented by the BigFloat instance.
        Raises:
            ValueError: If the exponent or fraction lengths are not as expected.
        """
        if len(self.exponent) != 11 or len(self.fraction) != 52:
            raise ValueError("Must be a standard 64-bit BigFloat")

        bits = [self.sign] + self.exponent + self.fraction
        return bitarray_to_float(bits)

    def __repr__(self) -> str:
        """Return a string representation of the BigFloat instance.

        Returns:
            str: A string representation of the BigFloat instance.
        """
        return f"BigFloat(sign={self.sign}, exponent={self.exponent}, fraction={self.fraction})"

    def pretty(self) -> str:
        """Return an easier to read string representation of the BigFloat instance.
        Mainly converts the exponent and fraction to integers for readability.

        Returns:
            str: A pretty string representation of the BigFloat instance.
        """
        sign = "-" if self.sign else ""
        exponent_value = bitarray_to_signed_int(self.exponent) + 1
        fraction_value = bitarray_to_int(self.fraction)
        return f"{sign}BigFloat(exponent={exponent_value}, fraction={fraction_value})"

    def _is_special_exponent(self) -> bool:
        """Check if the exponent represents a special value (NaN or Infinity).

        Returns:
            bool: True if the exponent is at its maximum value, False otherwise.
        """
        # In IEEE 754, special values have all exponent bits set to 1
        # This corresponds to the maximum value in the unsigned representation
        # For signed offset binary, the maximum value is 2^(n-1) - 1 where n is the number of bits
        max_signed_value = (1 << (len(self.exponent) - 1)) - 1
        return bitarray_to_signed_int(self.exponent) == max_signed_value

    def is_nan(self) -> bool:
        """Check if the BigFloat instance represents NaN (Not a Number).

        Returns:
            bool: True if the BigFloat instance is NaN, False otherwise.
        """
        return self._is_special_exponent() and any(self.fraction)

    def is_infinity(self) -> bool:
        """Check if the BigFloat instance represents Infinity.

        Returns:
            bool: True if the BigFloat instance is Infinity, False otherwise.
        """
        return self._is_special_exponent() and not any(self.fraction)

    def is_zero(self) -> bool:
        """Check if the BigFloat instance represents zero.

        Returns:
            bool: True if the BigFloat instance is zero, False otherwise.
        """
        return not any(self.exponent)

    def copy(self) -> BigFloat:
        """Create a copy of the BigFloat instance.

        Returns:
            BigFloat: A new BigFloat instance with the same sign, exponent, and fraction.
        """
        return BigFloat(
            sign=self.sign, exponent=self.exponent.copy(), fraction=self.fraction.copy()
        )

    def __str__(self) -> str:
        """Float representation of the BigFloat."""
        sign = "-" if self.sign else ""

        exponent_value = bitarray_to_signed_int(self.exponent)
        if exponent_value == 0:
            return f"{sign}0.0"
        max_exponent = 2 ** len(self.exponent) - 1
        # Check NaN or Infinity
        if exponent_value == max_exponent:
            if any(self.fraction):
                return f"{sign}NaN"
            return f"{sign}Infinity"

        fraction_value: float = 1
        for i, bit in enumerate(self.fraction):
            if bit:
                fraction_value += 2 ** -(i + 1)

        if exponent_value == 0:
            return f"{sign}{fraction_value}.0"

        # raise NotImplementedError("String representation for non-zero exponent not implemented yet.")
        return ""

    def __neg__(self) -> BigFloat:
        """Negate the BigFloat instance."""
        return BigFloat(
            sign=not self.sign,
            exponent=self.exponent.copy(),
            fraction=self.fraction.copy(),
        )

    def __add__(self, other: BigFloat) -> BigFloat:
        """Add two BigFloat instances together.

        Args:
            other (BigFloat): The other BigFloat instance to add.
        Returns:
            BigFloat: A new BigFloat instance representing the sum.
        """
        if not isinstance(other, BigFloat):
            raise TypeError("Can only add BigFloat instances.")

        if self.sign != other.sign:
            return self - (-other)

        # OBJECTIVE: Add two BigFloat instances together.
        # Based on: https://www.sciencedirect.com/topics/computer-science/floating-point-addition
        # and: https://cse.hkust.edu.hk/~cktang/cs180/notes/lec21.pdf
        #
        # Steps:
        # 0. Handle special cases (NaN, Infinity).
        # 1. Extract exponent and fraction bits.
        # 2. Prepend leading 1 to form the mantissa.
        # 3. Compare exponents.
        # 4. Shift smaller mantissa if necessary.
        # 5. Add mantissas.
        # 6. Normalize mantissa and adjust exponent if necessary.
        # 7. Grow exponent if necessary.
        # 8. Round result.
        # 9. Return new BigFloat instance.

        # Step 0: Handle special cases
        if self.is_nan() or other.is_nan():
            return self.copy() if self.is_nan() else other.copy()
        if self.is_infinity() or other.is_infinity():
            if self.is_infinity() and other.is_infinity():
                if self.sign == other.sign:
                    return self.copy()
                return BigFloat(sign=True)
            return self.copy() if self.is_infinity() else other.copy()
        if self.is_zero() or other.is_zero():
            return other.copy() if self.is_zero() else self.copy()

        # Step 1: Extract exponent and fraction bits
        exponent_self = bitarray_to_signed_int(self.exponent) + 1
        exponent_other = bitarray_to_signed_int(other.exponent) + 1

        # Step 2: Prepend leading 1 to form the mantissa
        mantissa_self = [True] + self.fraction
        mantissa_other = [True] + other.fraction

        # Step 3: Compare exponents (self is always larger or equal)
        if exponent_self < exponent_other:
            exponent_self, exponent_other = exponent_other, exponent_self
            mantissa_self, mantissa_other = mantissa_other, mantissa_self

        # Step 4: Shift smaller mantissa if necessary
        if exponent_self > exponent_other:
            shift_amount = exponent_self - exponent_other
            mantissa_other = shift_bitarray(mantissa_other, shift_amount)

        # Step 5: Add mantissas
        assert (
            len(mantissa_self) == 53
        ), "Fraction must be 53 bits long. (1 leading bit + 52 fraction bits)"
        assert len(mantissa_self) == len(
            mantissa_other
        ), f"Mantissas must be the same length. Expected 53 bits, got {len(mantissa_other)} bits."

        mantissa_result = [False] * 53  # 1 leading bit + 52 fraction bits
        carry = False
        for i in range(52, -1, -1):
            total = mantissa_self[i] + mantissa_other[i] + carry
            mantissa_result[i] = total % 2 == 1
            carry = total > 1

        # Step 6: Normalize mantissa and adjust exponent if necessary
        # Only need to normalize if there is a carry
        if carry:
            # Insert the carry bit and shift right
            mantissa_result = shift_bitarray(mantissa_result, 1, fill=True)
            exponent_self += 1

        # Step 7: Grow exponent if necessary
        exponent_result_length = len(self.exponent)
        if exponent_self >= (1 << (len(self.exponent) - 1)) - 1:
            # If the exponent is too large, we need to grow it
            # NOTE: Growth should never be bigger than 1 extra bit.
            exponent_result_length = exponent_result_length + 1
            assert (
                exponent_self - (2**exponent_result_length - 1) < 2
            ), "Exponent growth should not exceed 1 bit."

        exponent_result = signed_int_to_bitarray(
            exponent_self - 1, length=exponent_result_length
        )
        return BigFloat(
            sign=self.sign,
            exponent=exponent_result,
            fraction=mantissa_result[1:],  # Exclude leading bit
        )


class BigFloatUnitTest(TestCase):
    @staticmethod
    def _parse_bitarray(data: str) -> BitArray:
        """Parse a string of bits into a BitArray."""
        data = data.replace(" ", "")
        return [bit == "1" for bit in data]

    # === Float to BitArray Conversion Tests ===
    # https://binaryconvert.com/result_double.html
    def test_float_to_bitarray_converts_zero_correctly(self):
        """Test that zero is converted to all False bits."""
        value = 0
        expected = [False] * 64
        result = float_to_bitarray(value)
        self.assertEqual(result, expected)

    def test_float_to_bitarray_converts_positive_one_correctly(self):
        """Test that 1.0 is converted to correct IEEE 754 representation."""
        value = 1.0
        expected = self._parse_bitarray(
            "00111111 11110000 00000000 00000000 00000000 00000000 00000000 00000000"
        )
        result = float_to_bitarray(value)
        self.assertEqual(result, expected)

    def test_float_to_bitarray_converts_negative_integer_correctly(self):
        """Test that large negative integer is converted correctly."""
        value = -15789123456789
        expected = self._parse_bitarray(
            "11000010 10101100 10111000 01100010 00110000 10011110 00101010 00000000"
        )
        result = float_to_bitarray(value)
        self.assertEqual(result, expected)

    def test_float_to_bitarray_converts_fractional_number(self):
        """Test conversion of fractional numbers."""
        value = 0.5
        result = float_to_bitarray(value)
        # 0.5 in IEEE 754: sign=0, exponent=01111111110, mantissa=0...0
        self.assertFalse(result[0])  # Sign bit should be False (positive)
        self.assertEqual(len(result), 64)

    def test_float_to_bitarray_converts_infinity(self):
        """Test conversion of positive infinity."""
        value = float("inf")
        result = float_to_bitarray(value)
        # Infinity has all exponent bits set to 1 and mantissa all 0
        self.assertFalse(result[0])  # Sign bit False for positive infinity
        # Exponent bits (1-11) should all be True
        self.assertTrue(all(result[1:12]))
        # Mantissa bits (12-63) should all be False
        self.assertFalse(any(result[12:64]))

    def test_float_to_bitarray_converts_negative_infinity(self):
        """Test conversion of negative infinity."""
        value = float("-inf")
        result = float_to_bitarray(value)
        self.assertTrue(result[0])  # Sign bit True for negative infinity
        self.assertTrue(all(result[1:12]))  # All exponent bits True
        self.assertFalse(any(result[12:64]))  # All mantissa bits False

    def test_float_to_bitarray_converts_nan(self):
        """Test conversion of NaN (Not a Number)."""
        value = float("nan")
        result = float_to_bitarray(value)
        # NaN has all exponent bits set to 1 and at least one mantissa bit set
        self.assertTrue(all(result[1:12]))  # All exponent bits True
        self.assertTrue(any(result[12:64]))  # At least one mantissa bit True

    # === BitArray to Float Conversion Tests ===
    def test_bitarray_to_float_converts_zero_correctly(self):
        """Test that all False bits convert to zero."""
        bit_array = [False] * 64
        expected = 0.0
        result = bitarray_to_float(bit_array)
        self.assertEqual(result, expected)

    def test_bitarray_to_float_converts_positive_one_correctly(self):
        """Test that IEEE 754 representation of 1.0 converts correctly."""
        bit_array = self._parse_bitarray(
            "00111111 11110000 00000000 00000000 00000000 00000000 00000000 00000000"
        )
        expected = 1.0
        result = bitarray_to_float(bit_array)
        self.assertEqual(result, expected)

    def test_bitarray_to_float_converts_negative_number_correctly(self):
        """Test that negative number bit array converts correctly."""
        bit_array = self._parse_bitarray(
            "11000010 10101100 10111000 01100010 00110000 10011110 00101010 00000000"
        )
        expected = -15789123456789.0
        result = bitarray_to_float(bit_array)
        self.assertEqual(result, expected)

    def test_bitarray_to_float_raises_error_on_wrong_length(self):
        """Test that assertion error is raised for non-64-bit arrays."""
        with self.assertRaises(AssertionError):
            bitarray_to_float([True] * 32)  # Wrong length
        with self.assertRaises(AssertionError):
            bitarray_to_float([])  # Empty array

    def test_bitarray_to_float_roundtrip_preserves_value(self):
        """Test that converting float->bitarray->float preserves the original value."""
        original_values = [0.0, 1.0, -1.0, 3.14159, -2.71828, 1e100, 1e-100]
        for value in original_values:
            if not (math.isnan(value) or math.isinf(value)):
                bit_array = float_to_bitarray(value)
                result = bitarray_to_float(bit_array)
                self.assertEqual(result, value, f"Roundtrip failed for {value}")

    # === BitArray to Integer Conversion Tests ===
    def test_bitarray_to_int_converts_empty_array_to_zero(self):
        """Test that empty bit array converts to zero."""
        bit_array = []
        expected = 0
        result = bitarray_to_int(bit_array)
        self.assertEqual(result, expected)

    def test_bitarray_to_int_converts_all_false_to_zero(self):
        """Test that all False bits convert to zero."""
        bit_array = [False] * 64
        expected = 0
        result = bitarray_to_int(bit_array)
        self.assertEqual(result, expected)

    def test_bitarray_to_int_converts_single_bit_correctly(self):
        """Test that single True bit converts to 1."""
        bit_array = [False, True]
        expected = 1
        result = bitarray_to_int(bit_array)
        self.assertEqual(result, expected)

    def test_bitarray_to_int_converts_multiple_bits_correctly(self):
        """Test conversion of multiple bit patterns."""
        # Test binary 101 (decimal 5)
        bit_array = [True, False, True]
        expected = 5
        result = bitarray_to_int(bit_array)
        self.assertEqual(result, expected)

    def test_bitarray_to_int_converts_large_number_correctly(self):
        """Test conversion of large bit array to integer."""
        bit_array = self._parse_bitarray(
            "11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111001"
        )
        expected = 18446744073709551609
        result = bitarray_to_int(bit_array)
        self.assertEqual(result, expected)

    def test_bitarray_to_int_handles_leading_zeros(self):
        """Test that leading zeros don't affect the result."""
        bit_array1 = [True, False, True]  # 101 = 5
        bit_array2 = [False, False, True, False, True]  # 00101 = 5
        result1 = bitarray_to_int(bit_array1)
        result2 = bitarray_to_int(bit_array2)
        self.assertEqual(result1, result2)

    # === Signed Integer Conversion Tests ===
    def test_bitarray_to_signed_int_converts_zero_bias_correctly(self):
        """Test signed integer conversion with zero as negative bias."""
        bitarray = self._parse_bitarray("00000000001")  # 11-bit array
        expected = -1023  # -2^(11-1) + 1 = -1024 + 1
        result = bitarray_to_signed_int(bitarray)
        self.assertEqual(result, expected)

    def test_bitarray_to_signed_int_converts_near_zero_correctly(self):
        """Test signed integer conversion near zero."""
        bitarray = self._parse_bitarray("01111111111")  # 11-bit array
        expected = -1  # -2^(11-1) + 1023 = -1024 + 1023
        result = bitarray_to_signed_int(bitarray)
        self.assertEqual(result, expected)

    def test_bitarray_to_signed_int_converts_maximum_value_correctly(self):
        """Test signed integer conversion at maximum value."""
        bitarray = self._parse_bitarray("11111111111")  # 11-bit array
        expected = 1023  # -2^(11-1) + 2047 = -1024 + 2047
        result = bitarray_to_signed_int(bitarray)
        self.assertEqual(result, expected)

    def test_bitarray_to_signed_int_raises_error_on_empty_array(self):
        """Test that assertion error is raised for empty bit array."""
        with self.assertRaises(AssertionError):
            bitarray_to_signed_int([])

    def test_bitarray_to_signed_int_handles_different_lengths(self):
        """Test signed integer conversion with different bit array lengths."""
        # 8-bit test: bias = 2^7 = 128
        bitarray_8bit = self._parse_bitarray("10000000")  # 128 in unsigned
        expected_8bit = 0  # 128 - 128 = 0
        result_8bit = bitarray_to_signed_int(bitarray_8bit)
        self.assertEqual(result_8bit, expected_8bit)

        # 4-bit test: bias = 2^3 = 8
        bitarray_4bit = self._parse_bitarray("1100")  # 12 in unsigned
        expected_4bit = 4  # 12 - 8 = 4
        result_4bit = bitarray_to_signed_int(bitarray_4bit)
        self.assertEqual(result_4bit, expected_4bit)

    def test_signed_int_to_bitarray_converts_zero_correctly(self):
        """Test conversion of zero to signed bit array."""
        value = 0
        length = 8
        result = signed_int_to_bitarray(value, length)
        expected = self._parse_bitarray("10000000")  # bias = 128, so 0 + 128 = 128
        self.assertEqual(result, expected)

    def test_signed_int_to_bitarray_converts_positive_value_correctly(self):
        """Test conversion of positive value to signed bit array."""
        value = 5
        length = 8
        result = signed_int_to_bitarray(value, length)
        expected = self._parse_bitarray("10000101")  # bias = 128, so 5 + 128 = 133
        self.assertEqual(result, expected)

    def test_signed_int_to_bitarray_converts_negative_value_correctly(self):
        """Test conversion of negative value to signed bit array."""
        value = -5
        length = 8
        result = signed_int_to_bitarray(value, length)
        expected = self._parse_bitarray("01111011")  # bias = 128, so -5 + 128 = 123
        self.assertEqual(result, expected)

    def test_signed_int_to_bitarray_raises_error_on_overflow(self):
        """Test that assertion error is raised when value exceeds range."""
        # For 8-bit: max_value = (1 << 8) - 1 = 255, min_value = -(1 << 7) = -128
        # So range is -128 to 255

        # Test values within range should work
        try:
            result = signed_int_to_bitarray(255, 8)  # Should work (max)
            result = signed_int_to_bitarray(-128, 8)  # Should work (min)
        except AssertionError:
            self.fail("Valid values should not raise AssertionError")

        # Test values that should definitely fail
        with self.assertRaises(AssertionError):
            signed_int_to_bitarray(256, 8)  # Beyond max range
        with self.assertRaises(AssertionError):
            signed_int_to_bitarray(-129, 8)  # Beyond min range

    def test_signed_int_to_bitarray_roundtrip_preserves_value(self):
        """Test that signed int->bitarray->signed int preserves the original value."""
        length = 8
        test_values = [0, 1, -1, 127, -128, 50, -75]
        for value in test_values:
            bit_array = signed_int_to_bitarray(value, length)
            result = bitarray_to_signed_int(bit_array)
            self.assertEqual(result, value, f"Roundtrip failed for {value}")

    # === Bit Array Shifting Tests ===
    def test_shift_bitarray_no_shift_returns_original(self):
        """Test that zero shift returns the original array."""
        bit_array = [True, False, True, False]
        result = shift_bitarray(bit_array, 0)
        self.assertEqual(result, bit_array)

    def test_shift_bitarray_left_shift_with_default_fill(self):
        """Test left shift with default False fill."""
        bit_array = [True, False, True, False]
        result = shift_bitarray(bit_array, 2)
        expected = [False, False, True, False]
        self.assertEqual(result, expected)

    def test_shift_bitarray_left_shift_with_true_fill(self):
        """Test left shift with True fill value."""
        bit_array = [True, False, True, False]
        result = shift_bitarray(bit_array, 2, fill=True)
        expected = [True, True, True, False]
        self.assertEqual(result, expected)

    def test_shift_bitarray_right_shift_with_default_fill(self):
        """Test right shift with default False fill."""
        bit_array = [True, False, True, False]
        result = shift_bitarray(bit_array, -2)
        expected = [True, False, False, False]
        self.assertEqual(result, expected)

    def test_shift_bitarray_right_shift_with_true_fill(self):
        """Test right shift with True fill value."""
        bit_array = [True, False, True, False]
        result = shift_bitarray(bit_array, -2, fill=True)
        expected = [True, False, True, True]
        self.assertEqual(result, expected)

    def test_shift_bitarray_shift_entire_length(self):
        """Test shifting by the entire length of the array."""
        bit_array = [True, False, True, False]
        # Left shift by entire length
        result_left = shift_bitarray(bit_array, 4)
        expected_left = [False, False, False, False]
        self.assertEqual(result_left, expected_left)

        # Right shift by entire length
        result_right = shift_bitarray(bit_array, -4)
        expected_right = [False, False, False, False]
        self.assertEqual(result_right, expected_right)

    def test_shift_bitarray_shift_beyond_length(self):
        """Test shifting beyond the array length."""
        bit_array = [True, False, True, False]

        self.assertEquals(shift_bitarray(bit_array, 5), [False] * len(bit_array))

    def test_shift_bitarray_preserves_array_length(self):
        """Test that shifting can change array length depending on shift amount."""
        bit_array = [True, False, True, False, True]
        # Note: The shift_bitarray function doesn't preserve length when shift is larger than array
        # Let's test with shifts that should preserve or reasonably modify length
        shifts = [0, 1, -1, 3, -3]
        for shift in shifts:
            result = shift_bitarray(bit_array, shift)
            # The result length depends on the shift implementation
            self.assertIsInstance(
                result, list, f"Result should be a list for shift {shift}"
            )
            # For shifts within reasonable bounds, length should be preserved
            if abs(shift) <= len(bit_array):
                self.assertEqual(
                    len(result),
                    len(bit_array),
                    f"Length not preserved for shift {shift}",
                )

    # === BigFloat Construction and Basic Properties Tests ===
    def test_bigfloat_default_constructor_creates_zero(self):
        """Test that default constructor creates a zero BigFloat."""
        bf = BigFloat()
        self.assertFalse(bf.sign)
        self.assertEqual(bf.exponent, [False] * 11)
        self.assertEqual(bf.fraction, [False] * 52)

    def test_bigfloat_constructor_with_custom_values(self):
        """Test BigFloat constructor with custom sign, exponent, and fraction."""
        sign = True
        exponent = [True, False] * 5 + [True]  # 11 bits
        fraction = [False, True] * 26  # 52 bits
        bf = BigFloat(sign=sign, exponent=exponent, fraction=fraction)
        self.assertEqual(bf.sign, sign)
        self.assertEqual(bf.exponent, exponent)
        self.assertEqual(bf.fraction, fraction)

    def test_bigfloat_from_float_creates_correct_representation(self):
        """Test that from_float creates correct BigFloat representation."""
        value = 1.0
        bf = BigFloat.from_float(value)
        self.assertEqual(len(bf.exponent), 11)
        self.assertEqual(len(bf.fraction), 52)
        # Verify roundtrip
        self.assertEqual(bf.to_float(), value)

    def test_bigfloat_from_float_handles_special_values(self):
        """Test from_float with special IEEE 754 values."""
        # Test infinity
        bf_inf = BigFloat.from_float(float("inf"))
        self.assertTrue(bf_inf.is_infinity())
        self.assertFalse(bf_inf.sign)

        # Test negative infinity
        bf_neg_inf = BigFloat.from_float(float("-inf"))
        self.assertTrue(bf_neg_inf.is_infinity())
        self.assertTrue(bf_neg_inf.sign)

        # Test NaN
        bf_nan = BigFloat.from_float(float("nan"))
        self.assertTrue(bf_nan.is_nan())

    def test_bigfloat_to_float_raises_error_on_wrong_dimensions(self):
        """Test that to_float raises error when exponent or fraction have wrong length."""
        # Wrong exponent length
        bf_wrong_exp = BigFloat(exponent=[False] * 10, fraction=[False] * 52)
        with self.assertRaises(ValueError):
            bf_wrong_exp.to_float()

        # Wrong fraction length
        bf_wrong_frac = BigFloat(exponent=[False] * 11, fraction=[False] * 51)
        with self.assertRaises(ValueError):
            bf_wrong_frac.to_float()

    def test_bigfloat_copy_creates_independent_copy(self):
        """Test that copy creates an independent copy of BigFloat."""
        original = BigFloat.from_float(3.14)
        copy = original.copy()

        # Verify values are equal
        self.assertEqual(copy.sign, original.sign)
        self.assertEqual(copy.exponent, original.exponent)
        self.assertEqual(copy.fraction, original.fraction)

        # Verify independence
        copy.sign = not copy.sign
        copy.exponent[0] = not copy.exponent[0]
        copy.fraction[0] = not copy.fraction[0]

        self.assertNotEqual(copy.sign, original.sign)
        self.assertNotEqual(copy.exponent[0], original.exponent[0])
        self.assertNotEqual(copy.fraction[0], original.fraction[0])

    # === BigFloat Special Value Detection Tests ===
    def test_bigfloat_is_zero_detects_zero_correctly(self):
        """Test that is_zero correctly identifies zero values."""
        # Standard zero
        bf_zero = BigFloat.from_float(0.0)
        self.assertTrue(bf_zero.is_zero())

        # Negative zero
        bf_neg_zero = BigFloat.from_float(-0.0)
        self.assertTrue(bf_neg_zero.is_zero())

        # Non-zero
        bf_nonzero = BigFloat.from_float(1e-100)
        self.assertFalse(bf_nonzero.is_zero())

    def test_bigfloat_is_infinity_detects_infinity_correctly(self):
        """Test that is_infinity correctly identifies infinity values."""
        bf_inf = BigFloat.from_float(float("inf"))
        bf_neg_inf = BigFloat.from_float(float("-inf"))
        bf_finite = BigFloat.from_float(1e100)
        bf_nan = BigFloat.from_float(float("nan"))

        self.assertTrue(bf_inf.is_infinity())
        self.assertTrue(bf_neg_inf.is_infinity())
        self.assertFalse(bf_finite.is_infinity())
        self.assertFalse(bf_nan.is_infinity())

    def test_bigfloat_is_nan_detects_nan_correctly(self):
        """Test that is_nan correctly identifies NaN values."""
        bf_nan = BigFloat.from_float(float("nan"))
        bf_inf = BigFloat.from_float(float("inf"))
        bf_finite = BigFloat.from_float(42.0)

        self.assertTrue(bf_nan.is_nan())
        self.assertFalse(bf_inf.is_nan())
        self.assertFalse(bf_finite.is_nan())

    # === BigFloat Negation Tests ===
    def test_bigfloat_negation_flips_sign_correctly(self):
        """Test that negation correctly flips the sign bit."""
        bf_positive = BigFloat.from_float(42.0)
        bf_negative = -bf_positive

        self.assertFalse(bf_positive.sign)
        self.assertTrue(bf_negative.sign)
        self.assertEqual(bf_negative.exponent, bf_positive.exponent)
        self.assertEqual(bf_negative.fraction, bf_positive.fraction)

    def test_bigfloat_double_negation_preserves_original(self):
        """Test that double negation returns to original value."""
        original = BigFloat.from_float(-123.456)
        double_negated = -(-original)

        self.assertEqual(double_negated.sign, original.sign)
        self.assertEqual(double_negated.exponent, original.exponent)
        self.assertEqual(double_negated.fraction, original.fraction)

    # === BigFloat Addition Tests ===
    def test_bigfloat_addition_with_zero_returns_original(self):
        """Test that adding zero returns the original value."""
        value = 0.0
        bf = BigFloat.from_float(value)
        result = bf + bf
        self.assertEqual(result.to_float(), 0.0)

    def test_bigfloat_addition_simple_case_works_correctly(self):
        """Test simple addition case."""
        value = 1.0
        bf = BigFloat.from_float(value)
        result = bf + bf
        self.assertEqual(result.to_float(), 2.0)

    def test_bigfloat_addition_different_values_works_correctly(self):
        """Test addition of two different positive values."""
        bf1 = BigFloat.from_float(1.0)
        bf2 = BigFloat.from_float(2.0)
        result = bf1 + bf2
        self.assertEqual(result.to_float(), 3.0)

    def test_bigfloat_addition_large_numbers_works_correctly(self):
        """Test addition of large numbers."""
        bf1 = BigFloat.from_float(1.57e17)
        bf2 = BigFloat.from_float(2.34e18)
        result = bf1 + bf2
        self.assertEqual(result.to_float(), 2.497e18)

    def test_bigfloat_addition_rejects_non_bigfloat_operands(self):
        """Test that addition with non-BigFloat operands raises TypeError."""
        bf = BigFloat.from_float(1.0)
        with self.assertRaises(TypeError):
            bf + 1.0
        with self.assertRaises(TypeError):
            bf + "not a number"

    def test_bigfloat_addition_handles_nan_operands(self):
        """Test addition behavior with NaN operands."""
        bf_normal = BigFloat.from_float(1.0)
        bf_nan = BigFloat.from_float(float("nan"))

        result = bf_normal + bf_nan
        self.assertTrue(result.is_nan())

    def test_bigfloat_addition_handles_infinity_operands(self):
        """Test addition behavior with infinity operands."""
        bf_normal = BigFloat.from_float(1.0)
        bf_inf = BigFloat.from_float(float("inf"))
        bf_neg_inf = BigFloat.from_float(float("-inf"))

        result_inf = bf_normal + bf_inf
        self.assertTrue(result_inf.is_infinity())
        self.assertFalse(result_inf.sign)

        # result_neg_inf = bf_normal + bf_neg_inf
        # self.assertTrue(result_neg_inf.is_infinity())
        # self.assertTrue(result_neg_inf.sign)

        # result_zero = bf_inf + bf_neg_inf
        # self.assertTrue(result_zero.is_zero())

    def test_bigfloat_addition_with_mixed_signs_uses_subtraction(self):
        """Test that addition with different signs delegates to subtraction."""
        bf_pos = BigFloat.from_float(5.0)
        bf_neg = BigFloat.from_float(-3.0)

        # TODO: Implement subtraction in BigFloat

    # === BigFloat Subtraction Tests ===
    def test_bigfloat_subtraction_with_zero_returns_original(self):
        """Test that subtracting zero returns the original value."""
        bf = BigFloat.from_float(42.0)
        bf_zero = BigFloat.from_float(0.0)
        result = bf - bf_zero
        self.assertEqual(result.to_float(), 42.0)

    def test_bigfloat_subtraction_zero_minus_value_returns_negated(self):
        """Test that zero minus a value returns the negated value."""
        bf_zero = BigFloat.from_float(0.0)
        bf = BigFloat.from_float(42.0)
        result = bf_zero - bf
        self.assertEqual(result.to_float(), -42.0)

    def test_bigfloat_subtraction_same_value_returns_zero(self):
        """Test that subtracting a value from itself returns zero."""
        bf = BigFloat.from_float(123.456)
        result = bf - bf
        self.assertTrue(result.is_zero())

    def test_bigfloat_subtraction_simple_case_works_correctly(self):
        """Test simple subtraction case."""
        bf1 = BigFloat.from_float(5.0)
        bf2 = BigFloat.from_float(3.0)
        result = bf1 - bf2
        self.assertEqual(result.to_float(), 2.0)

    def test_bigfloat_subtraction_negative_result_works_correctly(self):
        """Test subtraction that results in a negative value."""
        bf1 = BigFloat.from_float(3.0)
        bf2 = BigFloat.from_float(5.0)
        result = bf1 - bf2
        self.assertEqual(result.to_float(), -2.0)

    def test_bigfloat_subtraction_large_numbers_works_correctly(self):
        """Test subtraction of large numbers."""
        bf1 = BigFloat.from_float(2.34e18)
        bf2 = BigFloat.from_float(1.57e17)
        result = bf1 - bf2
        self.assertAlmostEqual(result.to_float(), 2.183e18, places=10)

    def test_bigfloat_subtraction_small_numbers_works_correctly(self):
        """Test subtraction of very small numbers."""
        bf1 = BigFloat.from_float(1e-15)
        bf2 = BigFloat.from_float(5e-16)
        result = bf1 - bf2
        self.assertAlmostEqual(result.to_float(), 5e-16, places=20)

    def test_bigfloat_subtraction_different_exponents_works_correctly(self):
        """Test subtraction with numbers having different exponents."""
        bf1 = BigFloat.from_float(1000.0)
        bf2 = BigFloat.from_float(0.001)
        result = bf1 - bf2
        self.assertAlmostEqual(result.to_float(), 999.999, places=10)

    def test_bigfloat_subtraction_rejects_non_bigfloat_operands(self):
        """Test that subtraction with non-BigFloat operands raises TypeError."""
        bf = BigFloat.from_float(1.0)
        with self.assertRaises(TypeError):
            bf - 1.0
        with self.assertRaises(TypeError):
            bf - "not a number"

    def test_bigfloat_subtraction_handles_nan_operands(self):
        """Test subtraction behavior with NaN operands."""
        bf_normal = BigFloat.from_float(1.0)
        bf_nan = BigFloat.from_float(float("nan"))

        result1 = bf_normal - bf_nan
        self.assertTrue(result1.is_nan())

        result2 = bf_nan - bf_normal
        self.assertTrue(result2.is_nan())

        result3 = bf_nan - bf_nan
        self.assertTrue(result3.is_nan())

    def test_bigfloat_subtraction_handles_infinity_operands(self):
        """Test subtraction behavior with infinity operands."""
        bf_normal = BigFloat.from_float(1.0)
        bf_inf = BigFloat.from_float(float("inf"))
        bf_neg_inf = BigFloat.from_float(float("-inf"))

        # Normal - Infinity = -Infinity
        result1 = bf_normal - bf_inf
        self.assertTrue(result1.is_infinity())
        self.assertTrue(result1.sign)

        # Normal - (-Infinity) = Infinity
        result2 = bf_normal - bf_neg_inf
        self.assertTrue(result2.is_infinity())
        self.assertFalse(result2.sign)

        # Infinity - Normal = Infinity
        result3 = bf_inf - bf_normal
        self.assertTrue(result3.is_infinity())
        self.assertFalse(result3.sign)

        # (-Infinity) - Normal = -Infinity
        result4 = bf_neg_inf - bf_normal
        self.assertTrue(result4.is_infinity())
        self.assertTrue(result4.sign)

        # Infinity - Infinity = NaN
        result5 = bf_inf - bf_inf
        self.assertTrue(result5.is_nan())

        # (-Infinity) - (-Infinity) = NaN
        result6 = bf_neg_inf - bf_neg_inf
        self.assertTrue(result6.is_nan())

        # Infinity - (-Infinity) = Infinity
        result7 = bf_inf - bf_neg_inf
        self.assertTrue(result7.is_infinity())
        self.assertFalse(result7.sign)

        # (-Infinity) - Infinity = -Infinity
        result8 = bf_neg_inf - bf_inf
        self.assertTrue(result8.is_infinity())
        self.assertTrue(result8.sign)

    def test_bigfloat_subtraction_with_mixed_signs_becomes_addition(self):
        """Test that subtraction with different signs becomes addition."""
        bf_pos = BigFloat.from_float(5.0)
        bf_neg = BigFloat.from_float(-3.0)

        # 5 - (-3) = 5 + 3 = 8
        result1 = bf_pos - bf_neg
        self.assertEqual(result1.to_float(), 8.0)

        # (-3) - 5 = (-3) + (-5) = -8
        result2 = bf_neg - bf_pos
        self.assertEqual(result2.to_float(), -8.0)

    def test_bigfloat_subtraction_precision_loss_edge_cases(self):
        """Test subtraction edge cases that might cause precision loss."""
        # Subtracting very close numbers
        bf1 = BigFloat.from_float(1.0000000000000002)
        bf2 = BigFloat.from_float(1.0)
        result = bf1 - bf2
        # Should get approximately 2.220446049250313e-16
        self.assertGreater(result.to_float(), 0)
        self.assertLess(result.to_float(), 1e-15)

    def test_bigfloat_subtraction_mantissa_borrowing(self):
        """Test subtraction that requires mantissa borrowing."""
        # Test case where the first mantissa is smaller than the second
        bf1 = BigFloat.from_float(1.25)  # 1.01 in binary fraction
        bf2 = BigFloat.from_float(1.75)  # 1.11 in binary fraction
        result = bf1 - bf2
        self.assertEqual(result.to_float(), -0.5)

    def test_bigfloat_subtraction_denormalized_results(self):
        """Test subtraction that might result in denormalized numbers."""
        # Create numbers that when subtracted might underflow
        bf1 = BigFloat.from_float(1e-100)
        bf2 = BigFloat.from_float(9e-101)
        result = bf1 - bf2
        # Check if result has grown exponent or is standard format
        if len(result.exponent) == 11 and len(result.fraction) == 52:
            self.assertGreater(result.to_float(), 0)
            self.assertLess(result.to_float(), 1e-100)
        else:
            # Extended precision case - check basic properties
            self.assertFalse(result.is_zero())
            self.assertFalse(result.is_nan())
            self.assertFalse(result.is_infinity())

    def test_bigfloat_subtraction_exponent_underflow(self):
        """Test subtraction that causes exponent underflow."""
        # This test might need adjustment based on actual implementation
        # The idea is to test cases where the result might have a very small exponent
        bf1 = BigFloat.from_float(1e-300)
        bf2 = BigFloat.from_float(9.9e-301)
        result = bf1 - bf2
        self.assertGreater(result.to_float(), 0)
        self.assertFalse(result.is_zero())

    # === Additional BigFloat Addition Tests ===
    def test_bigfloat_addition_comprehensive_basic_cases(self):
        """Test comprehensive basic addition cases."""
        test_cases = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, 2.0, 3.0),
            (2.0, 4.0, 6.0),
            (4.0, 8.0, 12.0),
        ]

        for a, b, expected in test_cases:
            with self.subTest(a=a, b=b, expected=expected):
                bf_a = BigFloat.from_float(a)
                bf_b = BigFloat.from_float(b)
                result = bf_a + bf_b
                actual = result.to_float()
                self.assertEqual(
                    actual, expected, f"{a} + {b} should equal {expected}, got {actual}"
                )

    def test_bigfloat_addition_fractional_cases(self):
        """Test addition with fractional numbers."""
        test_cases = [
            (0.5, 0.5, 1.0),
            (0.125, 0.375, 0.5),
            (0.25, 0.25, 0.5),
            (0.75, 0.25, 1.0),
            (1.25, 2.75, 4.0),
            (3.5, 4.5, 8.0),
            (7.25, 0.75, 8.0),
        ]

        for a, b, expected in test_cases:
            with self.subTest(a=a, b=b, expected=expected):
                bf_a = BigFloat.from_float(a)
                bf_b = BigFloat.from_float(b)
                result = bf_a + bf_b
                actual = result.to_float()
                self.assertEqual(
                    actual, expected, f"{a} + {b} should equal {expected}, got {actual}"
                )

    def test_bigfloat_addition_original_bug_case(self):
        """Test the specific case that was originally failing: 7.5 + 2.5 = 10.0."""
        bf1 = BigFloat.from_float(7.5)
        bf2 = BigFloat.from_float(2.5)
        result = bf1 + bf2
        self.assertEqual(result.to_float(), 10.0, "7.5 + 2.5 should equal 10.0")

        # Test reverse order
        result_reverse = bf2 + bf1
        self.assertEqual(result_reverse.to_float(), 10.0, "2.5 + 7.5 should equal 10.0")

    def test_bigfloat_addition_different_exponents(self):
        """Test addition with numbers having different exponents."""
        test_cases = [
            (1.0, 0.001, 1.001),
            (1000.0, 0.1, 1000.1),
            (0.001, 1000.0, 1000.001),
            (1.5, 0.0625, 1.5625),  # 1.5 + 1/16
            (8.0, 0.125, 8.125),  # 8 + 1/8
        ]

        for a, b, expected in test_cases:
            with self.subTest(a=a, b=b, expected=expected):
                bf_a = BigFloat.from_float(a)
                bf_b = BigFloat.from_float(b)
                result = bf_a + bf_b
                actual = result.to_float()
                # Use relative tolerance for precision
                tolerance = max(1e-14, abs(expected) * 1e-15)
                self.assertLess(
                    abs(actual - expected),
                    tolerance,
                    f"{a} + {b} should equal {expected}, got {actual}",
                )

    def test_bigfloat_addition_large_numbers(self):
        """Test addition with large numbers."""
        test_cases = [
            (1000000.0, 2000000.0, 3000000.0),
            (1e10, 2e10, 3e10),
            (1.23e15, 4.56e15, 5.79e15),
        ]

        for a, b, expected in test_cases:
            with self.subTest(a=a, b=b, expected=expected):
                bf_a = BigFloat.from_float(a)
                bf_b = BigFloat.from_float(b)
                result = bf_a + bf_b
                actual = result.to_float()
                # Use relative tolerance for large numbers
                tolerance = abs(expected) * 1e-15
                self.assertLess(
                    abs(actual - expected),
                    tolerance,
                    f"{a} + {b} should equal {expected}, got {actual}",
                )

    def test_bigfloat_addition_small_numbers(self):
        """Test addition with very small numbers."""
        test_cases = [
            (1e-10, 2e-10, 3e-10),
            (1e-100, 2e-100, 3e-100),
            (5e-16, 5e-16, 1e-15),
        ]

        for a, b, expected in test_cases:
            with self.subTest(a=a, b=b, expected=expected):
                bf_a = BigFloat.from_float(a)
                bf_b = BigFloat.from_float(b)
                result = bf_a + bf_b
                actual = result.to_float()
                # Use relative tolerance
                tolerance = max(1e-14, abs(expected) * 1e-15)
                self.assertLess(
                    abs(actual - expected),
                    tolerance,
                    f"{a} + {b} should equal {expected}, got {actual}",
                )

    def test_bigfloat_addition_mantissa_carry_cases(self):
        """Test addition cases that require mantissa carry (overflow)."""
        # These cases specifically test the carry handling in mantissa addition
        test_cases = [
            (1.5, 1.5, 3.0),  # 1.1 + 1.1 = 11.0 (binary) -> needs carry
            (3.75, 4.25, 8.0),  # Should cause mantissa overflow
            (7.5, 8.5, 16.0),  # Multiple carries
            (15.5, 16.5, 32.0),  # Larger carry case
        ]

        for a, b, expected in test_cases:
            with self.subTest(a=a, b=b, expected=expected):
                bf_a = BigFloat.from_float(a)
                bf_b = BigFloat.from_float(b)
                result = bf_a + bf_b
                actual = result.to_float()
                self.assertEqual(
                    actual, expected, f"{a} + {b} should equal {expected}, got {actual}"
                )

    def test_bigfloat_addition_edge_precision_cases(self):
        """Test addition edge cases for precision."""
        # Test cases that might reveal precision issues
        bf1 = BigFloat.from_float(1.0000000000000002)
        bf2 = BigFloat.from_float(1.0000000000000002)
        result = bf1 + bf2
        expected = 2.0000000000000004
        actual = result.to_float()
        self.assertAlmostEqual(
            actual,
            expected,
            places=15,
            msg=f"High precision addition failed: got {actual}, expected {expected}",
        )

    def test_bigfloat_addition_commutative_property(self):
        """Test that addition is commutative (a + b = b + a)."""
        test_values = [1.0, 2.5, 7.5, 0.125, 1000.0, 1e-10]

        for i, a in enumerate(test_values):
            for b in test_values[i + 1 :]:
                with self.subTest(a=a, b=b):
                    bf_a = BigFloat.from_float(a)
                    bf_b = BigFloat.from_float(b)
                    result1 = bf_a + bf_b
                    result2 = bf_b + bf_a
                    self.assertEqual(
                        result1.to_float(),
                        result2.to_float(),
                        f"Addition should be commutative: {a} + {b} != {b} + {a}",
                    )

    def test_bigfloat_addition_associative_property(self):
        """Test that addition is associative ((a + b) + c = a + (b + c))."""
        test_cases = [
            (1.0, 2.0, 3.0),
            (0.5, 1.5, 2.5),
            (7.5, 2.5, 5.0),
            (0.125, 0.25, 0.375),
        ]

        for a, b, c in test_cases:
            with self.subTest(a=a, b=b, c=c):
                bf_a = BigFloat.from_float(a)
                bf_b = BigFloat.from_float(b)
                bf_c = BigFloat.from_float(c)

                # (a + b) + c
                result1 = (bf_a + bf_b) + bf_c
                # a + (b + c)
                result2 = bf_a + (bf_b + bf_c)

                self.assertAlmostEqual(
                    result1.to_float(),
                    result2.to_float(),
                    places=14,
                    msg=f"Addition should be associative: ({a} + {b}) + {c} != {a} + ({b} + {c})",
                )

    def test_bigfloat_addition_identity_element(self):
        """Test that zero is the additive identity (a + 0 = a)."""
        test_values = [0.0, 1.0, -1.0, 0.5, 7.5, 2.5, 1000.0, 1e-10, 1e10]

        for value in test_values:
            with self.subTest(value=value):
                bf_value = BigFloat.from_float(value)
                bf_zero = BigFloat.from_float(0.0)

                result1 = bf_value + bf_zero
                result2 = bf_zero + bf_value

                self.assertEqual(
                    result1.to_float(), value, f"{value} + 0 should equal {value}"
                )
                self.assertEqual(
                    result2.to_float(), value, f"0 + {value} should equal {value}"
                )
