"""Tests for BigFloat subtraction operations."""

import unittest
from bigfloat import BigFloat
from tests import BigFloatTestCase


class TestSubtraction(BigFloatTestCase):
    """Test BigFloat subtraction operations."""

    def test_bigfloat_subtraction_with_zero_returns_original(self):
        f1 = 42.0
        f2 = 0.0
        bf = BigFloat.from_float(f1)
        bf_zero = BigFloat.from_float(f2)
        result = bf - bf_zero
        expected = f1 - f2
        self.assertEqual(result.to_float(), expected)

    def test_bigfloat_subtraction_zero_minus_value_returns_negated(self):
        f1 = 0.0
        f2 = 42.0
        bf_zero = BigFloat.from_float(f1)
        bf = BigFloat.from_float(f2)
        result = bf_zero - bf
        expected = f1 - f2
        self.assertEqual(result.to_float(), expected)

    def test_bigfloat_subtraction_same_value_returns_zero(self):
        bf = BigFloat.from_float(123.456)
        result = bf - bf
        self.assertTrue(result.is_zero())

    def test_bigfloat_subtraction_simple_case_works_correctly(self):
        f1 = 5.0
        f2 = 3.0
        bf1 = BigFloat.from_float(f1)
        bf2 = BigFloat.from_float(f2)
        result = bf1 - bf2
        expected = f1 - f2
        self.assertEqual(result.to_float(), expected)

    def test_bigfloat_subtraction_negative_result_works_correctly(self):
        f1 = 3.0
        f2 = 5.0
        bf1 = BigFloat.from_float(f1)
        bf2 = BigFloat.from_float(f2)
        result = bf1 - bf2
        expected = f1 - f2
        self.assertEqual(result.to_float(), expected)

    def test_bigfloat_subtraction_large_numbers_works_correctly(self):
        f1 = 2.34e18
        f2 = 1.57e17
        bf1 = BigFloat.from_float(f1)
        bf2 = BigFloat.from_float(f2)
        result = bf1 - bf2
        expected = f1 - f2
        self.assertEqual(result.to_float(), expected)

    def test_bigfloat_subtraction_small_numbers_works_correctly(self):
        f1 = 1e-15
        f2 = 5e-16
        bf1 = BigFloat.from_float(f1)
        bf2 = BigFloat.from_float(f2)
        result = bf1 - bf2
        expected = f1 - f2
        self.assertEqual(result.to_float(), expected)

    def test_bigfloat_subtraction_different_exponents_works_correctly(self):
        f1 = 1000.0
        f2 = 0.001
        bf1 = BigFloat.from_float(f1)
        bf2 = BigFloat.from_float(f2)
        result = bf1 - bf2
        expected = f1 - f2
        self.assertEqual(result.to_float(), expected)

    def test_bigfloat_subtraction_rejects_non_bigfloat_operands(self):
        bf = BigFloat.from_float(1.0)
        with self.assertRaises(TypeError):
            bf - "not a number"

    def test_bigfloat_subtraction_handles_nan_operands(self):
        bf_normal = BigFloat.from_float(1.0)
        bf_nan = BigFloat.from_float(float("nan"))
        result1 = bf_normal - bf_nan
        self.assertTrue(result1.is_nan())
        result2 = bf_nan - bf_normal
        self.assertTrue(result2.is_nan())
        result3 = bf_nan - bf_nan
        self.assertTrue(result3.is_nan())

    def test_bigfloat_subtraction_handles_infinity_operands(self):
        bf_normal = BigFloat.from_float(1.0)
        bf_inf = BigFloat.from_float(float("inf"))
        bf_neg_inf = BigFloat.from_float(float("-inf"))
        result1 = bf_normal - bf_inf
        self.assertTrue(result1.is_infinity())
        self.assertTrue(result1.sign)
        result2 = bf_normal - bf_neg_inf
        self.assertTrue(result2.is_infinity())
        self.assertFalse(result2.sign)
        result3 = bf_inf - bf_normal
        self.assertTrue(result3.is_infinity())
        self.assertFalse(result3.sign)
        result4 = bf_neg_inf - bf_normal
        self.assertTrue(result4.is_infinity())
        self.assertTrue(result4.sign)
        result5 = bf_inf - bf_inf
        self.assertTrue(result5.is_nan())
        result6 = bf_neg_inf - bf_neg_inf
        self.assertTrue(result6.is_nan())
        result7 = bf_inf - bf_neg_inf
        self.assertTrue(result7.is_infinity())
        self.assertFalse(result7.sign)
        result8 = bf_neg_inf - bf_inf
        self.assertTrue(result8.is_infinity())
        self.assertTrue(result8.sign)

    def test_bigfloat_subtraction_with_mixed_signs_becomes_addition(self):
        f1 = 5.0
        f2 = -3.0
        bf_pos = BigFloat.from_float(f1)
        bf_neg = BigFloat.from_float(f2)
        result1 = bf_pos - bf_neg
        expected1 = f1 - f2
        self.assertEqual(result1.to_float(), expected1)
        result2 = bf_neg - bf_pos
        expected2 = f2 - f1
        self.assertEqual(result2.to_float(), expected2)

    def test_bigfloat_subtraction_precision_loss_edge_cases(self):
        f1 = 1.0000000000000002
        f2 = 1.0
        bf1 = BigFloat.from_float(f1)
        bf2 = BigFloat.from_float(f2)
        result = bf1 - bf2
        expected = f1 - f2
        self.assertEqual(result.to_float(), expected)

    def test_bigfloat_subtraction_mantissa_borrowing(self):
        f1 = 1.25
        f2 = 1.75
        bf1 = BigFloat.from_float(f1)
        bf2 = BigFloat.from_float(f2)
        result = bf1 - bf2
        expected = f1 - f2
        self.assertEqual(result.to_float(), expected)

    def test_bigfloat_subtraction_denormalized_results(self):
        f1 = 1e-100
        f2 = 9e-101
        bf1 = BigFloat.from_float(f1)
        bf2 = BigFloat.from_float(f2)
        result = bf1 - bf2
        expected = f1 - f2
        self.assertEqual(result.to_float(), expected)
        self.assertFalse(result.is_zero())
        self.assertGreater(result.to_float(), 0)
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
        self.fail("This test is not implemented yet.")


if __name__ == "__main__":
    unittest.main()
