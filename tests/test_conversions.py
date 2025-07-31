"""Tests for conversion functions between floats and bit arrays."""

import math
import unittest

from flexfloat import ListBoolBitArray
from tests import FlexFloatTestCase


class TestConversions(FlexFloatTestCase):
    """Test conversion functions for floats and bit arrays."""

    # === Float to BitArray Conversion Tests ===
    # https://binaryconvert.com/result_double.html
    def test_float_to_bitarray_converts_zero_correctly(self):
        """Test that zero is converted to all False bits."""
        value = 0
        expected = [False] * 64
        result = ListBoolBitArray.from_float(value)
        self.assertEqual(result, ListBoolBitArray.from_bits(expected))

    def test_float_to_bitarray_converts_positive_one_correctly(self):
        """Test that 1.0 is converted to correct IEEE 754 representation."""
        value = 1.0
        expected = ListBoolBitArray.parse_bitarray(
            "00000000 00000000 00000000 00000000 00000000 00000000 00001111 11111100"
        )
        result = ListBoolBitArray.from_float(value)
        self.assertEqual(result, expected)

    def test_float_to_bitarray_converts_negative_integer_correctly(self):
        """Test that large negative integer is converted correctly."""
        value = -15789123456789
        expected = ListBoolBitArray.parse_bitarray(
            "00000000 01010100 01111001 00001100 01000110 00011101 00110101 01000011"
        )
        result = ListBoolBitArray.from_float(value)
        self.assertEqual(result, expected)

    def test_float_to_bitarray_converts_fractional_number(self):
        """Test conversion of fractional numbers."""
        value = 0.5
        result = ListBoolBitArray.from_float(value)
        # 0.5 in IEEE 754: sign=0, exponent=01111111110, mantissa=0...0
        self.assertFalse(result[0])  # Sign bit should be False (positive)
        self.assertEqual(len(result), 64)

    def test_float_to_bitarray_converts_infinity(self):
        """Test conversion of positive infinity."""
        value = float("inf")
        result = ListBoolBitArray.from_float(value)
        # Infinity has all exponent bits set to 1 and mantissa all 0
        self.assertFalse(result[63])  # Sign bit False for positive infinity
        # Exponent bits (52-62) should all be True
        self.assertTrue(all(result[52:63]))
        # Mantissa bits (0-51) should all be False
        self.assertFalse(any(result[0:52]))

    def test_float_to_bitarray_converts_negative_infinity(self):
        """Test conversion of negative infinity."""
        value = float("-inf")
        result = ListBoolBitArray.from_float(value)
        self.assertTrue(result[63])  # Sign bit True for negative infinity
        self.assertTrue(all(result[52:63]))  # All exponent bits True
        self.assertFalse(any(result[0:52]))  # All mantissa bits False

    def test_float_to_bitarray_converts_nan(self):
        """Test conversion of NaN (Not a Number)."""
        value = float("nan")
        result = ListBoolBitArray.from_float(value)
        # NaN has all exponent bits set to 1 and at least one mantissa bit set
        self.assertTrue(all(result[52:63]))  # All exponent bits True
        self.assertTrue(any(result[0:52]))  # At least one mantissa bit True

    # === BitArray to Float Conversion Tests ===
    def test_bitarray_to_float_converts_zero_correctly(self):
        """Test that all False bits convert to zero."""
        bit_array = [False] * 64
        expected = 0.0
        result = ListBoolBitArray.from_bits(bit_array).to_float()
        self.assertEqual(result, expected)

    def test_bitarray_to_float_converts_positive_one_correctly(self):
        """Test that IEEE 754 representation of 1.0 converts correctly."""
        bit_array = ListBoolBitArray.parse_bitarray(
            "00000000 00000000 00000000 00000000 00000000 00000000 00001111 11111100"
        )
        expected = 1.0
        result = bit_array.to_float()
        self.assertEqual(result, expected)

    def test_bitarray_to_float_converts_negative_number_correctly(self):
        """Test that negative number bit array converts correctly."""
        bit_array = ListBoolBitArray.parse_bitarray(
            "00000000 01010100 01111001 00001100 01000110 00011101 00110101 01000011"
        )
        expected = -15789123456789.0
        result = bit_array.to_float()
        self.assertEqual(result, expected)

    def test_bitarray_to_float_raises_error_on_wrong_length(self):
        """Test that assertion error is raised for non-64-bit arrays."""
        with self.assertRaises(AssertionError):
            ListBoolBitArray.from_bits([True] * 32).to_float()  # Wrong length
        with self.assertRaises(AssertionError):
            ListBoolBitArray.from_bits().to_float()  # Empty array

    def test_bitarray_to_float_roundtrip_preserves_value(self):
        """Test that converting float->bitarray->float preserves the original value."""
        original_values = [0.0, 1.0, -1.0, 3.14159, -2.71828, 1e100, 1e-100]
        for value in original_values:
            if not (math.isnan(value) or math.isinf(value)):
                bit_array = ListBoolBitArray.from_float(value)
                result = bit_array.to_float()
                self.assertEqual(result, value, f"Roundtrip failed for {value}")

    # === FlexFloat.from_int Conversion Tests ===
    def test_from_int_converts_zero(self):
        """Test that zero integer converts to zero FlexFloat."""
        from flexfloat import FlexFloat

        result = FlexFloat.from_int(0)
        expected = FlexFloat.from_float(0.0)
        self.assertEqual(result, expected)
        self.assertEqual(result.to_float(), 0.0)

    def test_from_int_converts_positive_integers(self):
        """Test conversion of positive integers."""
        from flexfloat import FlexFloat

        test_values = [1, 2, 3, 5, 7, 10, 15, 31, 42, 100, 255, 256, 1000, 1023, 1024]

        for value in test_values:
            with self.subTest(value=value):
                result = FlexFloat.from_int(value)
                self.assertEqual(result.to_float(), float(value))
                self.assertFalse(result.sign)  # Should be positive

    def test_from_int_converts_negative_integers(self):
        """Test conversion of negative integers."""
        from flexfloat import FlexFloat

        test_values = [
            -1,
            -2,
            -3,
            -5,
            -7,
            -10,
            -15,
            -31,
            -42,
            -100,
            -255,
            -256,
            -1000,
            -1023,
            -1024,
        ]

        for value in test_values:
            with self.subTest(value=value):
                result = FlexFloat.from_int(value)
                self.assertEqual(result.to_float(), float(value))
                self.assertTrue(result.sign)  # Should be negative

    def test_from_int_converts_powers_of_two(self):
        """Test conversion of powers of two (exact representations)."""
        from flexfloat import FlexFloat

        # Test powers of 2 from 2^0 to 2^60
        for i in range(61):
            value = 2**i
            with self.subTest(power=i, value=value):
                result = FlexFloat.from_int(value)
                self.assertEqual(result.to_float(), float(value))

                # Also test negative powers of two
                neg_result = FlexFloat.from_int(-value)
                self.assertEqual(neg_result.to_float(), float(-value))

    def test_from_int_converts_large_integers(self):
        """Test conversion of integers larger than standard float precision."""
        from flexfloat import FlexFloat

        large_values = [
            123456789012345678901234567890,
            2**100,
            2**200,
            10**50,
            -123456789012345678901234567890,
            -(2**100),
            -(2**200),
            -(10**50),
        ]

        for value in large_values:
            with self.subTest(value=value):
                result = FlexFloat.from_int(value)
                # For very large numbers, we check that the conversion preserves
                # the magnitude and sign correctly
                self.assertEqual(result.sign, value < 0)

                # For large numbers, we expect some precision loss due to limited
                # fraction bits, but the magnitude should be approximately correct
                back_to_int = result.to_int()

                # Check that the order of magnitude is approximately preserved
                # Allow for one digit difference due to precision loss
                orig_digits = len(str(abs(value)))
                back_digits = len(str(abs(back_to_int)))
                self.assertLessEqual(abs(orig_digits - back_digits), 1)

                # For powers of 2, which should be exactly representable, check exactness
                if value in [2**100, -(2**100), 2**200, -(2**200)]:
                    self.assertEqual(back_to_int, value)

    def test_from_int_converts_fibonacci_numbers(self):
        """Test conversion of Fibonacci numbers (covers various bit patterns)."""
        from flexfloat import FlexFloat

        # Generate first 50 Fibonacci numbers
        fib = [0, 1]
        for i in range(2, 50):
            fib.append(fib[i - 1] + fib[i - 2])

        for i, value in enumerate(fib):
            with self.subTest(fib_index=i, value=value):
                result = FlexFloat.from_int(value)
                self.assertEqual(result.to_float(), float(value))

    def test_from_int_converts_mersenne_numbers(self):
        """Test conversion of Mersenne numbers (2^n - 1)."""
        from flexfloat import FlexFloat

        # Test Mersenne numbers for various powers
        for n in [3, 5, 7, 13, 17, 19, 31, 61, 89, 107]:
            value = (2**n) - 1
            with self.subTest(mersenne_exp=n, value=value):
                result = FlexFloat.from_int(value)
                back_to_int = result.to_int()

                # For small Mersenne numbers (n <= 52), expect exact conversion
                if n <= 52:
                    self.assertEqual(back_to_int, value)
                else:
                    # For larger numbers, just check that we get a reasonable approximation
                    # The relative error should be small
                    if value > 0:
                        relative_error = abs(back_to_int - value) / value
                        self.assertLess(
                            relative_error, 1e-15
                        )  # Very small relative error

    def test_from_int_handles_edge_cases(self):
        """Test edge cases for from_int conversion."""
        from flexfloat import FlexFloat

        # Test maximum values that fit in standard types
        edge_cases = [
            2**31 - 1,  # Max 32-bit signed int
            2**31,  # Min 32-bit signed int magnitude
            2**63 - 1,  # Max 64-bit signed int
            2**64 - 1,  # Max 64-bit unsigned int
            -(2**31),  # Min 32-bit signed int
            -(2**63),  # Approximate min 64-bit signed int
        ]

        for value in edge_cases:
            with self.subTest(value=value):
                result = FlexFloat.from_int(value)
                back_to_int = result.to_int()

                # For values that fit in 52 bits of precision, expect exact conversion
                if abs(value).bit_length() <= 53:  # 52 fraction bits + 1 implicit bit
                    self.assertEqual(back_to_int, value)
                else:
                    # For larger values, check that the sign is preserved and magnitude is close
                    self.assertEqual(back_to_int < 0, value < 0)  # Same sign
                    if value != 0:
                        relative_error = abs(back_to_int - value) / abs(value)
                        self.assertLess(relative_error, 1e-15)  # Small relative error

    def test_from_int_precision_preservation(self):
        """Test that from_int preserves precision for integers within reasonable range."""
        from flexfloat import FlexFloat

        # Test integers that should be exactly representable
        for value in range(-1000, 1001):
            with self.subTest(value=value):
                result = FlexFloat.from_int(value)
                self.assertEqual(result.to_int(), value)
                self.assertEqual(result.to_float(), float(value))

    def test_from_int_very_large_numbers(self):
        """Test conversion of extremely large integers."""
        from flexfloat import FlexFloat

        # Test extremely large numbers
        very_large = [
            10**100,
            10**200,
            10**300,
            2**1000,
            2**2000,
            123456789 * (10**100),
        ]

        for value in very_large:
            with self.subTest(value=str(value)[:50] + "..."):
                result = FlexFloat.from_int(value)
                # Verify the sign is correct
                self.assertFalse(result.sign)

                # For extremely large numbers, we mainly check that
                # the conversion doesn't raise an exception and produces
                # a reasonable result
                self.assertGreater(result.to_float(), 0)

                # Test negative version too
                neg_result = FlexFloat.from_int(-value)
                self.assertTrue(neg_result.sign)

    def test_from_int_roundtrip_consistency(self):
        """Test that int->FlexFloat->int roundtrip preserves values when possible."""
        from flexfloat import FlexFloat

        # Test values that should roundtrip exactly (within 53-bit precision)
        test_values = [
            0,
            1,
            -1,
            2,
            -2,
            10,
            -10,
            100,
            -100,
            2**10,
            2**20,
            2**30,
            2**40,
            2**50,
            1234567890,
            -1234567890,
            # Note: 2**60 - 1 has more than 53 bits, so it won't roundtrip exactly
            2**52 - 1,
            -(2**52 - 1),  # These should be exact
            (1 << 53) - 1,
            -((1 << 53) - 1),  # Largest integers that roundtrip exactly
        ]

        for value in test_values:
            with self.subTest(value=value):
                flex_float = FlexFloat.from_int(value)
                roundtrip_value = flex_float.to_int()

                # Values with <= 53 significant bits should roundtrip exactly
                if abs(value).bit_length() <= 53:
                    self.assertEqual(roundtrip_value, value)
                else:
                    # For larger values, check approximate equality
                    if value != 0:
                        relative_error = abs(roundtrip_value - value) / abs(value)
                        self.assertLess(relative_error, 1e-15)


if __name__ == "__main__":
    unittest.main()
