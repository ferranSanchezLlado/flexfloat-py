"""Tests for FlexFloat class construction, properties, and basic operations."""

import unittest

from flexfloat import FlexFloat
from flexfloat.bitarray import BitArray
from tests import FlexFloatTestCase


class TestFlexFloat(FlexFloatTestCase):
    """Test FlexFloat class construction and basic properties."""

    # === FlexFloat Construction and Basic Properties Tests ===
    def test_flexfloat_default_constructor_creates_zero(self):
        """Test that default constructor creates a zero FlexFloat."""
        bf = FlexFloat()
        self.assertFalse(bf.sign)
        self.assertEqual(bf.exponent, [False] * 11)
        self.assertEqual(bf.fraction, [False] * 52)

    def test_flexfloat_constructor_with_custom_values(self):
        """Test FlexFloat constructor with custom sign, exponent, and fraction."""
        sign = True
        exponent = BitArray([True, False] * 5 + [True])  # 11 bits
        fraction = BitArray([False, True] * 26)  # 52 bits
        bf = FlexFloat(sign=sign, exponent=exponent, fraction=fraction)
        self.assertEqual(bf.sign, sign)
        self.assertEqual(bf.exponent, exponent)
        self.assertEqual(bf.fraction, fraction)

    def test_flexfloat_from_float_creates_correct_representation(self):
        """Test that from_float creates correct FlexFloat representation."""
        value = 1.0
        bf = FlexFloat.from_float(value)
        self.assertEqual(len(bf.exponent), 11)
        self.assertEqual(len(bf.fraction), 52)
        # Verify roundtrip
        self.assertEqual(bf.to_float(), value)

    def test_flexfloat_from_float_handles_special_values(self):
        """Test from_float with special IEEE 754 values."""
        # Test infinity
        bf_inf = FlexFloat.from_float(float("inf"))
        self.assertTrue(bf_inf.is_infinity())
        self.assertFalse(bf_inf.sign)

        # Test negative infinity
        bf_neg_inf = FlexFloat.from_float(float("-inf"))
        self.assertTrue(bf_neg_inf.is_infinity())
        self.assertTrue(bf_neg_inf.sign)

        # Test NaN
        bf_nan = FlexFloat.from_float(float("nan"))
        self.assertTrue(bf_nan.is_nan())

    def test_flexfloat_to_float_raises_error_on_wrong_dimensions(self):
        "Test that to_float raises error when exponent or fraction have wrong length."
        # Wrong exponent length
        bf_wrong_exp = FlexFloat(
            exponent=BitArray([False] * 10), fraction=BitArray([False] * 52)
        )
        with self.assertRaises(ValueError):
            bf_wrong_exp.to_float()

        # Wrong fraction length
        bf_wrong_frac = FlexFloat(
            exponent=BitArray([False] * 11), fraction=BitArray([False] * 51)
        )
        with self.assertRaises(ValueError):
            bf_wrong_frac.to_float()

    def test_flexfloat_copy_creates_independent_copy(self):
        """Test that copy creates an independent copy of FlexFloat."""
        original = FlexFloat.from_float(3.14)
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

    # === FlexFloat Special Value Detection Tests ===
    def test_flexfloat_is_zero_detects_zero_correctly(self):
        """Test that is_zero correctly identifies zero values."""
        # Standard zero
        bf_zero = FlexFloat.from_float(0.0)
        self.assertTrue(bf_zero.is_zero())

        # Negative zero
        bf_neg_zero = FlexFloat.from_float(-0.0)
        self.assertTrue(bf_neg_zero.is_zero())

        # Non-zero
        bf_nonzero = FlexFloat.from_float(1e-100)
        self.assertFalse(bf_nonzero.is_zero())

    def test_flexfloat_is_infinity_detects_infinity_correctly(self):
        """Test that is_infinity correctly identifies infinity values."""
        bf_inf = FlexFloat.from_float(float("inf"))
        bf_neg_inf = FlexFloat.from_float(float("-inf"))
        bf_finite = FlexFloat.from_float(1e100)
        bf_nan = FlexFloat.from_float(float("nan"))

        self.assertTrue(bf_inf.is_infinity())
        self.assertTrue(bf_neg_inf.is_infinity())
        self.assertFalse(bf_finite.is_infinity())
        self.assertFalse(bf_nan.is_infinity())

    def test_flexfloat_is_nan_detects_nan_correctly(self):
        """Test that is_nan correctly identifies NaN values."""
        bf_nan = FlexFloat.from_float(float("nan"))
        bf_inf = FlexFloat.from_float(float("inf"))
        bf_finite = FlexFloat.from_float(42.0)

        self.assertTrue(bf_nan.is_nan())
        self.assertFalse(bf_inf.is_nan())
        self.assertFalse(bf_finite.is_nan())

    # === FlexFloat Negation Tests ===
    def test_flexfloat_negation_flips_sign_correctly(self):
        """Test that negation correctly flips the sign bit."""
        bf_positive = FlexFloat.from_float(42.0)
        bf_negative = -bf_positive

        self.assertFalse(bf_positive.sign)
        self.assertTrue(bf_negative.sign)
        self.assertEqual(bf_negative.exponent, bf_positive.exponent)
        self.assertEqual(bf_negative.fraction, bf_positive.fraction)

    def test_flexfloat_double_negation_preserves_original(self):
        """Test that double negation returns to original value."""
        original = FlexFloat.from_float(-123.456)
        double_negated = -(-original)

        self.assertEqual(double_negated.sign, original.sign)
        self.assertEqual(double_negated.exponent, original.exponent)
        self.assertEqual(double_negated.fraction, original.fraction)


if __name__ == "__main__":
    unittest.main()
