"""Tests for BigFloat arithmetic operations (addition and subtraction)."""

import unittest
from bigfloat import BigFloat
from tests import BigFloatTestCase


class TestArithmetic(BigFloatTestCase):
    """Test BigFloat arithmetic operations."""

    # === BigFloat Addition Tests ===
    def test_bigfloat_addition_with_zero_returns_original(self):
        """Test that adding zero returns the original value."""
        f1 = 0.0
        f2 = 0.0
        bf1 = BigFloat.from_float(f1)
        bf2 = BigFloat.from_float(f2)
        result = bf1 + bf2
        expected = f1 + f2
        self.assertEqual(result.to_float(), expected)

    def test_bigfloat_addition_simple_case_works_correctly(self):
        """Test simple addition case."""
        f1 = 1.0
        f2 = 1.0
        bf1 = BigFloat.from_float(f1)
        bf2 = BigFloat.from_float(f2)
        result = bf1 + bf2
        expected = f1 + f2
        self.assertEqual(result.to_float(), expected)

    def test_bigfloat_addition_different_values_works_correctly(self):
        """Test addition of two different positive values."""
        f1 = 1.0
        f2 = 2.0
        bf1 = BigFloat.from_float(f1)
        bf2 = BigFloat.from_float(f2)
        result = bf1 + bf2
        expected = f1 + f2
        self.assertEqual(result.to_float(), expected)

    def test_bigfloat_addition_large_numbers_works_correctly(self):
        """Test addition of large numbers."""
        f1 = 1.57e17
        f2 = 2.34e18
        bf1 = BigFloat.from_float(f1)
        bf2 = BigFloat.from_float(f2)
        result = bf1 + bf2
        expected = f1 + f2
        self.assertEqual(result.to_float(), expected)

    def test_bigfloat_addition_overflow_works_correctly(self):
        """Test addition that results in overflow."""
        f1 = 1e308
        f2 = 1e308
        bf1 = BigFloat.from_float(f1)
        bf2 = BigFloat.from_float(f2)
        result = bf1 + bf2
        self.assertFalse(result.is_infinity())
        self.assertGreater(len(result.exponent), 11)

    def test_bigfloat_addition_rejects_non_bigfloat_operands(self):
        """Test that addition with non-BigFloat operands raises TypeError."""
        bf = BigFloat.from_float(1.0)
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

        result_neg_inf = bf_normal + bf_neg_inf
        self.assertTrue(result_neg_inf.is_infinity())
        self.assertTrue(result_neg_inf.sign)

        result_zero = bf_inf + bf_neg_inf
        self.assertTrue(result_zero.is_nan())

    def test_bigfloat_addition_with_mixed_signs_uses_subtraction(self):
        """Test that addition with different signs delegates to subtraction."""
        f1 = 5.0
        f2 = -3.0
        bf_pos = BigFloat.from_float(f1)
        bf_neg = BigFloat.from_float(f2)

        result = bf_pos + bf_neg
        expected = f1 + f2
        self.assertEqual(result.to_float(), expected)

    def test_bigfloat_addition_comprehensive_basic_cases(self):
        """Test comprehensive basic addition cases."""
        test_cases = [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 2.0),
            (2.0, 4.0),
            (4.0, 8.0),
        ]

        for a, b in test_cases:
            with self.subTest(a=a, b=b):
                bf_a = BigFloat.from_float(a)
                bf_b = BigFloat.from_float(b)
                result = bf_a + bf_b
                expected = a + b
                actual = result.to_float()
                self.assertEqual(
                    actual, expected, f"{a} + {b} should equal {expected}, got {actual}"
                )

    def test_bigfloat_addition_fractional_cases(self):
        """Test addition with fractional numbers."""
        test_cases = [
            (0.5, 0.5),
            (0.125, 0.375),
            (0.25, 0.25),
            (0.75, 0.25),
            (1.25, 2.75),
            (3.5, 4.5),
            (7.25, 0.75),
        ]

        for a, b in test_cases:
            with self.subTest(a=a, b=b):
                bf_a = BigFloat.from_float(a)
                bf_b = BigFloat.from_float(b)
                result = bf_a + bf_b
                expected = a + b
                actual = result.to_float()
                self.assertEqual(
                    actual, expected, f"{a} + {b} should equal {expected}, got {actual}"
                )

    def test_bigfloat_addition_original_bug_case(self):
        """Test the specific case that was originally failing: 7.5 + 2.5 = 10.0."""
        f1 = 7.5
        f2 = 2.5
        bf1 = BigFloat.from_float(f1)
        bf2 = BigFloat.from_float(f2)
        result = bf1 + bf2
        expected = f1 + f2
        self.assertEqual(
            result.to_float(), expected, f"{f1} + {f2} should equal {expected}"
        )

        # Test reverse order
        result_reverse = bf2 + bf1
        expected_reverse = f2 + f1
        self.assertEqual(
            result_reverse.to_float(),
            expected_reverse,
            f"{f2} + {f1} should equal {expected_reverse}",
        )

    def test_bigfloat_addition_different_exponents(self):
        """Test addition with numbers having different exponents."""
        test_cases = [
            (1.0, 0.001),
            (1000.0, 0.1),
            (0.001, 1000.0),
            (1.5, 0.0625),  # 1.5 + 1/16
            (8.0, 0.125),  # 8 + 1/8
        ]

        for a, b in test_cases:
            with self.subTest(a=a, b=b):
                bf_a = BigFloat.from_float(a)
                bf_b = BigFloat.from_float(b)
                result = bf_a + bf_b
                actual = result.to_float()
                expected = a + b
                self.assertAlmostEqual(
                    actual,
                    expected,
                    places=12,
                    msg=f"{a} + {b} should equal {expected}, got {actual}",
                )

    def test_bigfloat_addition_large_numbers(self):
        """Test addition with large numbers."""
        test_cases = [
            (1000000.0, 2000000.0),
            (1e10, 2e10),
            (1.23e15, 4.56e15),
        ]

        for a, b in test_cases:
            with self.subTest(a=a, b=b):
                bf_a = BigFloat.from_float(a)
                bf_b = BigFloat.from_float(b)
                result = bf_a + bf_b
                expected = a + b
                actual = result.to_float()
                self.assertEqual(
                    actual, expected, f"{a} + {b} should equal {expected}, got {actual}"
                )

    def test_bigfloat_addition_small_numbers(self):
        """Test addition with very small numbers."""
        test_cases = [
            (1e-10, 2e-10),
            (1e-100, 2e-100),
            (5e-16, 5e-16),
        ]

        for a, b in test_cases:
            with self.subTest(a=a, b=b):
                bf_a = BigFloat.from_float(a)
                bf_b = BigFloat.from_float(b)
                result = bf_a + bf_b
                actual = result.to_float()
                expected = a + b
                self.assertEqual(
                    actual, expected, f"{a} + {b} should equal {expected}, got {actual}"
                )

    def test_bigfloat_addition_mantissa_carry_cases(self):
        """Test addition cases that require mantissa carry (overflow)."""
        # These cases specifically test the carry handling in mantissa addition
        test_cases = [
            (1.5, 1.5),  # 1.1 + 1.1 = 11.0 (binary) -> needs carry
            (3.75, 4.25),  # Should cause mantissa overflow
            (7.5, 8.5),  # Multiple carries
            (15.5, 16.5),  # Larger carry case
        ]

        for a, b in test_cases:
            with self.subTest(a=a, b=b):
                bf_a = BigFloat.from_float(a)
                bf_b = BigFloat.from_float(b)
                result = bf_a + bf_b
                actual = result.to_float()
                expected = a + b
                self.assertEqual(
                    actual, expected, f"{a} + {b} should equal {expected}, got {actual}"
                )

    def test_bigfloat_addition_edge_precision_cases(self):
        """Test addition edge cases for precision."""
        # Test cases that might reveal precision issues
        f1 = 1.0000000000000002
        f2 = 1.0000000000000002
        bf1 = BigFloat.from_float(f1)
        bf2 = BigFloat.from_float(f2)
        result = bf1 + bf2
        expected = f1 + f2
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

    # === BigFloat Subtraction Tests ===
    def test_bigfloat_subtraction_with_zero_returns_original(self):
        """Test that subtracting zero returns the original value."""
        f1 = 42.0
        f2 = 0.0
        bf = BigFloat.from_float(f1)
        bf_zero = BigFloat.from_float(f2)
        result = bf - bf_zero
        expected = f1 - f2
        self.assertEqual(result.to_float(), expected)

    def test_bigfloat_subtraction_zero_minus_value_returns_negated(self):
        """Test that zero minus a value returns the negated value."""
        f1 = 0.0
        f2 = 42.0
        bf_zero = BigFloat.from_float(f1)
        bf = BigFloat.from_float(f2)
        result = bf_zero - bf
        expected = f1 - f2
        self.assertEqual(result.to_float(), expected)

    def test_bigfloat_subtraction_same_value_returns_zero(self):
        """Test that subtracting a value from itself returns zero."""
        bf = BigFloat.from_float(123.456)
        result = bf - bf
        self.assertTrue(result.is_zero())

    def test_bigfloat_subtraction_simple_case_works_correctly(self):
        """Test simple subtraction case."""
        f1 = 5.0
        f2 = 3.0
        bf1 = BigFloat.from_float(f1)
        bf2 = BigFloat.from_float(f2)
        result = bf1 - bf2
        expected = f1 - f2
        self.assertEqual(result.to_float(), expected)

    def test_bigfloat_subtraction_negative_result_works_correctly(self):
        """Test subtraction that results in a negative value."""
        f1 = 3.0
        f2 = 5.0
        bf1 = BigFloat.from_float(f1)
        bf2 = BigFloat.from_float(f2)
        result = bf1 - bf2
        expected = f1 - f2
        self.assertEqual(result.to_float(), expected)

    def test_bigfloat_subtraction_large_numbers_works_correctly(self):
        """Test subtraction of large numbers."""
        f1 = 2.34e18
        f2 = 1.57e17
        bf1 = BigFloat.from_float(f1)
        bf2 = BigFloat.from_float(f2)
        result = bf1 - bf2
        expected = f1 - f2
        self.assertEqual(result.to_float(), expected)

    def test_bigfloat_subtraction_small_numbers_works_correctly(self):
        """Test subtraction of very small numbers."""
        f1 = 1e-15
        f2 = 5e-16
        bf1 = BigFloat.from_float(f1)
        bf2 = BigFloat.from_float(f2)
        result = bf1 - bf2
        expected = f1 - f2
        self.assertEqual(result.to_float(), expected)

    def test_bigfloat_subtraction_different_exponents_works_correctly(self):
        """Test subtraction with numbers having different exponents."""
        f1 = 1000.0
        f2 = 0.001
        bf1 = BigFloat.from_float(f1)
        bf2 = BigFloat.from_float(f2)
        result = bf1 - bf2
        expected = f1 - f2
        self.assertEqual(result.to_float(), expected)

    def test_bigfloat_subtraction_rejects_non_bigfloat_operands(self):
        """Test that subtraction with non-BigFloat operands raises TypeError."""
        bf = BigFloat.from_float(1.0)
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
        f1 = 5.0
        f2 = -3.0
        bf_pos = BigFloat.from_float(f1)
        bf_neg = BigFloat.from_float(f2)

        # 5 - (-3) = 5 + 3 = 8
        result1 = bf_pos - bf_neg
        expected1 = f1 - f2
        self.assertEqual(result1.to_float(), expected1)

        # (-3) - 5 = (-3) + (-5) = -8
        result2 = bf_neg - bf_pos
        expected2 = f2 - f1
        self.assertEqual(result2.to_float(), expected2)

    def test_bigfloat_subtraction_precision_loss_edge_cases(self):
        """Test subtraction edge cases that might cause precision loss."""
        # Subtracting very close numbers
        f1 = 1.0000000000000002
        f2 = 1.0
        bf1 = BigFloat.from_float(f1)
        bf2 = BigFloat.from_float(f2)
        result = bf1 - bf2
        expected = f1 - f2
        self.assertEqual(result.to_float(), expected)

    def test_bigfloat_subtraction_mantissa_borrowing(self):
        """Test subtraction that requires mantissa borrowing."""
        # Test case where the first mantissa is smaller than the second
        f1 = 1.25
        f2 = 1.75
        bf1 = BigFloat.from_float(f1)  # 1.01 in binary fraction
        bf2 = BigFloat.from_float(f2)  # 1.11 in binary fraction
        result = bf1 - bf2
        expected = f1 - f2
        self.assertEqual(result.to_float(), expected)

    def test_bigfloat_subtraction_denormalized_results(self):
        """Test subtraction that might result in denormalized numbers."""
        # Create numbers that when subtracted might underflow
        f1 = 1e-100
        f2 = 9e-101
        bf1 = BigFloat.from_float(f1)
        bf2 = BigFloat.from_float(f2)
        result = bf1 - bf2
        expected = f1 - f2

        # Verify the result is correct and not zero
        self.assertEqual(result.to_float(), expected)
        self.assertFalse(result.is_zero())
        self.assertGreater(result.to_float(), 0)

        # Test another case with very small differences
        f3 = 1.0000000000000002
        f4 = 1.0
        bf3 = BigFloat.from_float(f3)
        bf4 = BigFloat.from_float(f4)
        result2 = bf3 - bf4
        expected2 = f3 - f4

        self.assertEqual(result2.to_float(), expected2)
        self.assertGreater(result2.to_float(), 0)

    @unittest.skip("This test is not implemented yet.")
    def test_bigfloat_subtraction_exponent_underflow(self):
        """Test subtraction that causes exponent underflow."""
        # Test with numbers that when subtracted create a result requiring a smaller exponent
        # than can be represented in 11 bits (IEEE 754 standard)
        self.fail("This test is not implemented yet.")


if __name__ == "__main__":
    unittest.main()
