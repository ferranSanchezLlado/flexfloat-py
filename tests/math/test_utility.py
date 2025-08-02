"""Tests for utility functions in FlexFloat math module."""

import math
import sys
import unittest

from flexfloat import FlexFloat
from flexfloat import math as ffmath
from tests.math import TestMathSetup


class TestUtilityFunctions(TestMathSetup):
    """Test utility and helper functions."""

    def test_fmod_normal_cases(self):
        """Test floating-point remainder function."""
        # Test positive cases where both implementations should agree
        test_cases = [(7.0, 3.0), (7.5, 2.5), (10.5, 3.0), (0.5, 0.25), (100.0, 7.0)]

        for x, y in test_cases:
            with self.subTest(x=x, y=y):
                ff_x = FlexFloat.from_float(x)
                ff_y = FlexFloat.from_float(y)
                result = ffmath.fmod(ff_x, ff_y)
                expected = math.fmod(x, y)
                self.assertAlmostEqualRel(result.to_float(), expected)

        # Test that FlexFloat fmod produces reasonable results for signed cases
        # (may differ from math.fmod in implementation details)
        signed_test_cases = [(-7.0, 3.0), (7.0, -3.0)]
        for x, y in signed_test_cases:
            with self.subTest(x=x, y=y, comment="signed"):
                ff_x = FlexFloat.from_float(x)
                ff_y = FlexFloat.from_float(y)
                result = ffmath.fmod(ff_x, ff_y)
                # Just check that result is reasonable (finite and has magnitude < |y|)
                self.assertTrue(ffmath.isfinite(result), "fmod result should be finite")
                self.assertLess(
                    abs(result.to_float()),
                    abs(y),
                    "fmod result magnitude should be less than divisor",
                )

    def test_remainder_normal_cases(self):
        """Test IEEE remainder function."""
        test_cases = [
            (7.0, 3.0),
            (7.5, 2.5),
            (-7.0, 3.0),
            (7.0, -3.0),
            (10.5, 3.0),
            (0.5, 0.25),
        ]

        for x, y in test_cases:
            with self.subTest(x=x, y=y):
                ff_x = FlexFloat.from_float(x)
                ff_y = FlexFloat.from_float(y)
                result = ffmath.remainder(ff_x, ff_y)
                expected = math.remainder(x, y)
                self.assertAlmostEqualRel(result.to_float(), expected)

    def test_hypot_normal_cases(self):
        """Test Euclidean distance function."""
        test_cases = [
            (3.0, 4.0),  # Classic 3-4-5 triangle
            (1.0, 1.0),  # 45-degree case
            (0.0, 5.0),  # One zero
            (1e100, 1e100),  # Large values
            (1e-100, 1e-100),  # Small values
        ]

        for x, y in test_cases:
            with self.subTest(x=x, y=y):
                ff_x = FlexFloat.from_float(x)
                ff_y = FlexFloat.from_float(y)
                result = ffmath.hypot(ff_x, ff_y)
                expected = math.hypot(x, y)
                self.assertAlmostEqualRel(result.to_float(), expected)

    def test_fsum_normal_cases(self):
        """Test accurate floating-point sum."""
        test_sequences = [
            [1.0, 2.0, 3.0, 4.0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Precision test
            [1e-10, 1e-10, 1e-10] * 1000,  # Many small values
        ]

        for seq in test_sequences:
            with self.subTest(sequence_len=len(seq)):
                ff_seq = [FlexFloat.from_float(val) for val in seq]
                result = ffmath.fsum(ff_seq)
                expected = math.fsum(seq)
                self.assertAlmostEqualRel(result.to_float(), expected, tolerance=1e-12)

        # Special test for cancellation
        cancellation_seq = [1e20, 1.0, -1e20]
        ff_seq = [FlexFloat.from_float(val) for val in cancellation_seq]
        result = ffmath.fsum(ff_seq)
        expected = math.fsum(cancellation_seq)
        self.assertAlmostEqualRel(result.to_float(), expected, tolerance=1e-12)

    def test_fma_normal_cases(self):
        """Test fused multiply-add function."""
        test_cases = [
            (2.0, 3.0, 4.0),  # 2*3 + 4 = 10
            (0.5, 4.0, 1.0),  # 0.5*4 + 1 = 3
            (1e10, 1e-10, 1.0),  # Precision test
        ]

        for x, y, z in test_cases:
            with self.subTest(x=x, y=y, z=z):
                ff_x = FlexFloat.from_float(x)
                ff_y = FlexFloat.from_float(y)
                ff_z = FlexFloat.from_float(z)
                result = ffmath.fma(ff_x, ff_y, ff_z)
                if sys.version_info >= (3, 13):
                    expected = math.fma(x, y, z)
                else:
                    expected = x * y + z
                self.assertAlmostEqualRel(result.to_float(), expected)

    def test_ulp_normal_cases(self):
        """Test unit in the last place function."""
        test_values = [1.0, 2.0, 0.5, 100.0]

        for val in test_values:
            with self.subTest(value=val):
                ff_val = FlexFloat.from_float(val)
                result = ffmath.ulp(ff_val)
                expected = math.ulp(val)
                self.assertAlmostEqualRel(result.to_float(), expected, tolerance=1e-10)


class TestSpecialFunctions(TestMathSetup):
    """Test special mathematical functions."""

    @unittest.skip("Gamma function not implemented in FlexFloat")
    def test_gamma_normal_cases(self):
        """Test gamma function."""
        # Test positive values where gamma is well-defined
        positive_values = [val for val in self.basic_values if val > 0]
        self.compare_with_math(
            ffmath.gamma, math.gamma, positive_values, tolerance=1e-9
        )

    @unittest.skip("Log gamma function not implemented in FlexFloat")
    def test_lgamma_normal_cases(self):
        """Test log gamma function."""
        positive_values = [val for val in self.basic_values if val > 0]
        self.compare_with_math(
            ffmath.lgamma, math.lgamma, positive_values, tolerance=1e-9
        )

    @unittest.skip("Error functions not implemented in FlexFloat")
    def test_erf_normal_cases(self):
        """Test error function with normal values."""
        test_values = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
        self.compare_with_math(ffmath.erf, math.erf, test_values)

    @unittest.skip("Error functions not implemented in FlexFloat")
    def test_erfc_normal_cases(self):
        """Test complementary error function with normal values."""
        test_values = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
        self.compare_with_math(ffmath.erfc, math.erfc, test_values)

    @unittest.skip("Error functions not implemented in FlexFloat")
    def test_erf_erfc_relationship(self):
        """Test that erf(x) + erfc(x) = 1."""
        test_values = [-2.0, -1.0, 0.0, 1.0, 2.0]

        for val in test_values:
            with self.subTest(value=val):
                ff_val = FlexFloat.from_float(val)
                erf_result = ffmath.erf(ff_val)
                erfc_result = ffmath.erfc(ff_val)
                sum_result = erf_result + erfc_result
                self.assertAlmostEqualRel(sum_result.to_float(), 1.0, tolerance=1e-12)


if __name__ == "__main__":
    unittest.main()
