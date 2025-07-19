"""Tests for conversion functions between floats and bit arrays."""

import math
import unittest

from flexfloat import BitArray
from tests import FlexFloatTestCase


class TestConversions(FlexFloatTestCase):
    """Test conversion functions for floats and bit arrays."""

    # === Float to BitArray Conversion Tests ===
    # https://binaryconvert.com/result_double.html
    def test_float_to_bitarray_converts_zero_correctly(self):
        """Test that zero is converted to all False bits."""
        value = 0
        expected = [False] * 64
        result = BitArray.from_float(value)
        self.assertEqual(result, BitArray(expected))

    def test_float_to_bitarray_converts_positive_one_correctly(self):
        """Test that 1.0 is converted to correct IEEE 754 representation."""
        value = 1.0
        expected = BitArray.parse_bitarray(
            "00111111 11110000 00000000 00000000 00000000 00000000 00000000 00000000"
        )
        result = BitArray.from_float(value)
        self.assertEqual(result, expected)

    def test_float_to_bitarray_converts_negative_integer_correctly(self):
        """Test that large negative integer is converted correctly."""
        value = -15789123456789
        expected = BitArray.parse_bitarray(
            "11000010 10101100 10111000 01100010 00110000 10011110 00101010 00000000"
        )
        result = BitArray.from_float(value)
        self.assertEqual(result, expected)

    def test_float_to_bitarray_converts_fractional_number(self):
        """Test conversion of fractional numbers."""
        value = 0.5
        result = BitArray.from_float(value)
        # 0.5 in IEEE 754: sign=0, exponent=01111111110, mantissa=0...0
        self.assertFalse(result[0])  # Sign bit should be False (positive)
        self.assertEqual(len(result), 64)

    def test_float_to_bitarray_converts_infinity(self):
        """Test conversion of positive infinity."""
        value = float("inf")
        result = BitArray.from_float(value)
        # Infinity has all exponent bits set to 1 and mantissa all 0
        self.assertFalse(result[0])  # Sign bit False for positive infinity
        # Exponent bits (1-11) should all be True
        self.assertTrue(all(result[1:12]))
        # Mantissa bits (12-63) should all be False
        self.assertFalse(any(result[12:64]))

    def test_float_to_bitarray_converts_negative_infinity(self):
        """Test conversion of negative infinity."""
        value = float("-inf")
        result = BitArray.from_float(value)
        self.assertTrue(result[0])  # Sign bit True for negative infinity
        self.assertTrue(all(result[1:12]))  # All exponent bits True
        self.assertFalse(any(result[12:64]))  # All mantissa bits False

    def test_float_to_bitarray_converts_nan(self):
        """Test conversion of NaN (Not a Number)."""
        value = float("nan")
        result = BitArray.from_float(value)
        # NaN has all exponent bits set to 1 and at least one mantissa bit set
        self.assertTrue(all(result[1:12]))  # All exponent bits True
        self.assertTrue(any(result[12:64]))  # At least one mantissa bit True

    # === BitArray to Float Conversion Tests ===
    def test_bitarray_to_float_converts_zero_correctly(self):
        """Test that all False bits convert to zero."""
        bit_array = [False] * 64
        expected = 0.0
        result = BitArray(bit_array).to_float()
        self.assertEqual(result, expected)

    def test_bitarray_to_float_converts_positive_one_correctly(self):
        """Test that IEEE 754 representation of 1.0 converts correctly."""
        bit_array = BitArray.parse_bitarray(
            "00111111 11110000 00000000 00000000 00000000 00000000 00000000 00000000"
        )
        expected = 1.0
        result = bit_array.to_float()
        self.assertEqual(result, expected)

    def test_bitarray_to_float_converts_negative_number_correctly(self):
        """Test that negative number bit array converts correctly."""
        bit_array = BitArray.parse_bitarray(
            "11000010 10101100 10111000 01100010 00110000 10011110 00101010 00000000"
        )
        expected = -15789123456789.0
        result = bit_array.to_float()
        self.assertEqual(result, expected)

    def test_bitarray_to_float_raises_error_on_wrong_length(self):
        """Test that assertion error is raised for non-64-bit arrays."""
        with self.assertRaises(AssertionError):
            BitArray([True] * 32).to_float()  # Wrong length
        with self.assertRaises(AssertionError):
            BitArray().to_float()  # Empty array

    def test_bitarray_to_float_roundtrip_preserves_value(self):
        """Test that converting float->bitarray->float preserves the original value."""
        original_values = [0.0, 1.0, -1.0, 3.14159, -2.71828, 1e100, 1e-100]
        for value in original_values:
            if not (math.isnan(value) or math.isinf(value)):
                bit_array = BitArray.from_float(value)
                result = bit_array.to_float()
                self.assertEqual(result, value, f"Roundtrip failed for {value}")


if __name__ == "__main__":
    unittest.main()
