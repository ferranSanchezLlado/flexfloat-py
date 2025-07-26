"""Tests for conversion functions between floats and bit arrays."""

import math
import unittest

from flexfloat import BitArrayType
from tests import FlexFloatTestCase


class TestConversions(FlexFloatTestCase):
    """Test conversion functions for floats and bit arrays."""

    # === Float to BitArray Conversion Tests ===
    # https://binaryconvert.com/result_double.html
    def test_float_to_bitarray_converts_zero_correctly(self):
        """Test that zero is converted to all False bits."""
        value = 0
        expected = [False] * 64
        result = BitArrayType.from_float(value)
        self.assertEqual(result, BitArrayType.from_bits(expected))

    def test_float_to_bitarray_converts_positive_one_correctly(self):
        """Test that 1.0 is converted to correct IEEE 754 representation."""
        value = 1.0
        expected = BitArrayType.parse_bitarray(
            "00000000 00000000 00000000 00000000 00000000 00000000 00001111 11111100"
        )
        result = BitArrayType.from_float(value)
        self.assertEqual(result, expected)

    def test_float_to_bitarray_converts_negative_integer_correctly(self):
        """Test that large negative integer is converted correctly."""
        value = -15789123456789
        expected = BitArrayType.parse_bitarray(
            "00000000 01010100 01111001 00001100 01000110 00011101 00110101 01000011"
        )
        result = BitArrayType.from_float(value)
        self.assertEqual(result, expected)

    def test_float_to_bitarray_converts_fractional_number(self):
        """Test conversion of fractional numbers."""
        value = 0.5
        result = BitArrayType.from_float(value)
        # 0.5 in IEEE 754: sign=0, exponent=01111111110, mantissa=0...0
        self.assertFalse(result[0])  # Sign bit should be False (positive)
        self.assertEqual(len(result), 64)

    def test_float_to_bitarray_converts_infinity(self):
        """Test conversion of positive infinity."""
        value = float("inf")
        result = BitArrayType.from_float(value)
        # Infinity has all exponent bits set to 1 and mantissa all 0
        self.assertFalse(result[63])  # Sign bit False for positive infinity
        # Exponent bits (52-62) should all be True
        self.assertTrue(all(result[52:63]))
        # Mantissa bits (0-51) should all be False
        self.assertFalse(any(result[0:52]))

    def test_float_to_bitarray_converts_negative_infinity(self):
        """Test conversion of negative infinity."""
        value = float("-inf")
        result = BitArrayType.from_float(value)
        self.assertTrue(result[63])  # Sign bit True for negative infinity
        self.assertTrue(all(result[52:63]))  # All exponent bits True
        self.assertFalse(any(result[0:52]))  # All mantissa bits False

    def test_float_to_bitarray_converts_nan(self):
        """Test conversion of NaN (Not a Number)."""
        value = float("nan")
        result = BitArrayType.from_float(value)
        # NaN has all exponent bits set to 1 and at least one mantissa bit set
        self.assertTrue(all(result[52:63]))  # All exponent bits True
        self.assertTrue(any(result[0:52]))  # At least one mantissa bit True

    # === BitArray to Float Conversion Tests ===
    def test_bitarray_to_float_converts_zero_correctly(self):
        """Test that all False bits convert to zero."""
        bit_array = [False] * 64
        expected = 0.0
        result = BitArrayType.from_bits(bit_array).to_float()
        self.assertEqual(result, expected)

    def test_bitarray_to_float_converts_positive_one_correctly(self):
        """Test that IEEE 754 representation of 1.0 converts correctly."""
        bit_array = BitArrayType.parse_bitarray(
            "00000000 00000000 00000000 00000000 00000000 00000000 00001111 11111100"
        )
        expected = 1.0
        result = bit_array.to_float()
        self.assertEqual(result, expected)

    def test_bitarray_to_float_converts_negative_number_correctly(self):
        """Test that negative number bit array converts correctly."""
        bit_array = BitArrayType.parse_bitarray(
            "00000000 01010100 01111001 00001100 01000110 00011101 00110101 01000011"
        )
        expected = -15789123456789.0
        result = bit_array.to_float()
        self.assertEqual(result, expected)

    def test_bitarray_to_float_raises_error_on_wrong_length(self):
        """Test that assertion error is raised for non-64-bit arrays."""
        with self.assertRaises(AssertionError):
            BitArrayType.from_bits([True] * 32).to_float()  # Wrong length
        with self.assertRaises(AssertionError):
            BitArrayType.from_bits().to_float()  # Empty array

    def test_bitarray_to_float_roundtrip_preserves_value(self):
        """Test that converting float->bitarray->float preserves the original value."""
        original_values = [0.0, 1.0, -1.0, 3.14159, -2.71828, 1e100, 1e-100]
        for value in original_values:
            if not (math.isnan(value) or math.isinf(value)):
                bit_array = BitArrayType.from_float(value)
                result = bit_array.to_float()
                self.assertEqual(result, value, f"Roundtrip failed for {value}")


if __name__ == "__main__":
    unittest.main()
