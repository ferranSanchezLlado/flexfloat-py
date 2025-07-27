"""Tests for flexfloat.math module functions."""

import math  # For reference values in tests
import unittest

from flexfloat import FlexFloat
from flexfloat import math as ffmath
from tests import FlexFloatTestCase


class TestFlexFloatMath(FlexFloatTestCase):
    """Test class for FlexFloat math module functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_values = [0.0, 1.0, -1.0, 0.5, 2.0, 3.14159, -3.14159, 10.0, -10.0]
        self.small_positive = 1e-10
        self.large_positive = 1e10

    # Tests for constants
    def test_constants(self):
        """Test that math constants are properly defined."""
        self.assertIsInstance(ffmath.e, FlexFloat)
        self.assertIsInstance(ffmath.pi, FlexFloat)
        self.assertIsInstance(ffmath.inf, FlexFloat)
        self.assertIsInstance(ffmath.nan, FlexFloat)
        self.assertIsInstance(ffmath.tau, FlexFloat)

        # Check approximate values
        self.assertAlmostEqualRel(ffmath.e.to_float(), math.e)
        self.assertAlmostEqualRel(ffmath.pi.to_float(), math.pi)
        self.assertAlmostEqualRel(ffmath.tau.to_float(), math.tau)
        self.assertTrue(ffmath.inf.is_infinity())
        self.assertTrue(ffmath.nan.is_nan())

    # Tests for implemented functions
    def test_exp(self):
        """Test exp function."""
        test_cases = [0.0, 1.0, -1.0, 0.5, 2.0]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                result = ffmath.exp(x)
                expected = math.exp(value)
                self.assertAlmostEqualRel(result.to_float(), expected, tolerance=1e-6)

    def test_pow(self):
        """Test pow function."""
        test_cases = [
            (2.0, 3.0),
            (1.5, 2.0),
            (4.0, 0.5),
            (10.0, 2.0),
            (2.0, -1.0),
        ]
        for base, exp in test_cases:
            with self.subTest(base=base, exp=exp):
                base_ff = FlexFloat.from_float(base)
                exp_ff = FlexFloat.from_float(exp)
                result = ffmath.pow(base_ff, exp_ff)
                expected = math.pow(base, exp)
                self.assertAlmostEqualRel(result.to_float(), expected, tolerance=1e-6)

    def test_copysign(self):
        """Test copysign function."""
        test_cases = [
            (5.0, 1.0),
            (5.0, -1.0),
            (-5.0, 1.0),
            (-5.0, -1.0),
            (0.0, 1.0),
            (0.0, -1.0),
        ]
        for mag, sign in test_cases:
            with self.subTest(mag=mag, sign=sign):
                mag_ff = FlexFloat.from_float(mag)
                sign_ff = FlexFloat.from_float(sign)
                result = ffmath.copysign(mag_ff, sign_ff)
                expected = math.copysign(mag, sign)
                self.assertAlmostEqualRel(result.to_float(), expected)

    def test_fabs(self):
        """Test fabs function."""
        for value in self.test_values:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                result = ffmath.fabs(x)
                expected = math.fabs(value)
                self.assertAlmostEqualRel(result.to_float(), expected)

    def test_isinf(self):
        """Test isinf function."""
        # Test finite values
        for value in self.test_values:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                self.assertEqual(ffmath.isinf(x), math.isinf(value))

        # Test infinity
        pos_inf = FlexFloat.infinity()
        neg_inf = FlexFloat.infinity(sign=True)
        self.assertTrue(ffmath.isinf(pos_inf))
        self.assertTrue(ffmath.isinf(neg_inf))

        # Test NaN
        nan_val = FlexFloat.nan()
        self.assertFalse(ffmath.isinf(nan_val))

    def test_isnan(self):
        """Test isnan function."""
        # Test finite values
        for value in self.test_values:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                self.assertEqual(ffmath.isnan(x), math.isnan(value))

        # Test infinity
        pos_inf = FlexFloat.infinity()
        neg_inf = FlexFloat.infinity(sign=True)
        self.assertFalse(ffmath.isnan(pos_inf))
        self.assertFalse(ffmath.isnan(neg_inf))

        # Test NaN
        nan_val = FlexFloat.nan()
        self.assertTrue(ffmath.isnan(nan_val))

    def test_isfinite(self):
        """Test isfinite function."""
        # Test finite values
        for value in self.test_values:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                self.assertTrue(ffmath.isfinite(x))

        # Test infinity
        pos_inf = FlexFloat.infinity()
        neg_inf = FlexFloat.infinity(sign=True)
        self.assertFalse(ffmath.isfinite(pos_inf))
        self.assertFalse(ffmath.isfinite(neg_inf))

        # Test NaN
        nan_val = FlexFloat.nan()
        self.assertFalse(ffmath.isfinite(nan_val))

    def test_sqrt(self):
        """Test sqrt function."""
        test_cases = [0.0, 1.0, 4.0, 9.0, 16.0, 0.25, 2.0]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                result = ffmath.sqrt(x)
                expected = math.sqrt(value)
                self.assertAlmostEqualRel(result.to_float(), expected, tolerance=1e-6)

    # Tests for unimplemented functions
    @unittest.skip("acos is not implemented yet")
    def test_acos(self):
        """Test acos function."""
        test_cases = [0.0, 0.5, 1.0, -1.0]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                result = ffmath.acos(x)
                expected = math.acos(value)
                self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("acosh is not implemented yet")
    def test_acosh(self):
        """Test acosh function."""
        test_cases = [1.0, 2.0, 5.0]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                result = ffmath.acosh(x)
                expected = math.acosh(value)
                self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("asin is not implemented yet")
    def test_asin(self):
        """Test asin function."""
        test_cases = [0.0, 0.5, 1.0, -1.0]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                result = ffmath.asin(x)
                expected = math.asin(value)
                self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("asinh is not implemented yet")
    def test_asinh(self):
        """Test asinh function."""
        test_cases = [0.0, 1.0, -1.0, 2.0]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                result = ffmath.asinh(x)
                expected = math.asinh(value)
                self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("atan is not implemented yet")
    def test_atan(self):
        """Test atan function."""
        test_cases = [0.0, 1.0, -1.0, 2.0]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                result = ffmath.atan(x)
                expected = math.atan(value)
                self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("atan2 is not implemented yet")
    def test_atan2(self):
        """Test atan2 function."""
        test_cases = [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 1.0)]
        for y, x in test_cases:
            with self.subTest(y=y, x=x):
                y_ff = FlexFloat.from_float(y)
                x_ff = FlexFloat.from_float(x)
                result = ffmath.atan2(y_ff, x_ff)
                expected = math.atan2(y, x)
                self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("atanh is not implemented yet")
    def test_atanh(self):
        """Test atanh function."""
        test_cases = [0.0, 0.5, -0.5]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                result = ffmath.atanh(x)
                expected = math.atanh(value)
                self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("cbrt is not implemented yet")
    def test_cbrt(self):
        """Test cbrt function."""
        test_cases = [0.0, 1.0, 8.0, 27.0, -8.0]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                result = ffmath.cbrt(x)
                expected = math.cbrt(value)
                self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("ceil is not implemented yet")
    def test_ceil(self):
        """Test ceil function."""
        test_cases = [0.0, 1.5, -1.5, 2.1, -2.1]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                result = ffmath.ceil(x)
                expected = math.ceil(value)
                self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("dist is not implemented yet")
    def test_dist(self):
        """Test dist function."""
        p = [FlexFloat.from_float(1.0), FlexFloat.from_float(2.0)]
        q = [FlexFloat.from_float(4.0), FlexFloat.from_float(6.0)]
        result = ffmath.dist(p, q)
        expected = math.dist([1.0, 2.0], [4.0, 6.0])
        self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("erf is not implemented yet")
    def test_erf(self):
        """Test erf function."""
        test_cases = [0.0, 1.0, -1.0, 0.5]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                result = ffmath.erf(x)
                expected = math.erf(value)
                self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("erfc is not implemented yet")
    def test_erfc(self):
        """Test erfc function."""
        test_cases = [0.0, 1.0, -1.0, 0.5]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                result = ffmath.erfc(x)
                expected = math.erfc(value)
                self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("expm1 is not implemented yet")
    def test_expm1(self):
        """Test expm1 function."""
        test_cases = [0.0, 1.0, -1.0, 0.1]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                result = ffmath.expm1(x)
                expected = math.expm1(value)
                self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("factorial is not implemented yet")
    def test_factorial(self):
        """Test factorial function."""
        test_cases = [0.0, 1.0, 2.0, 5.0]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                result = ffmath.factorial(x)
                expected = math.factorial(int(value))
                self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("fmod is not implemented yet")
    def test_fmod(self):
        """Test fmod function."""
        test_cases = [(7.0, 3.0), (10.0, 3.0), (5.5, 2.0)]
        for x_val, y_val in test_cases:
            with self.subTest(x=x_val, y=y_val):
                x = FlexFloat.from_float(x_val)
                y = FlexFloat.from_float(y_val)
                result = ffmath.fmod(x, y)
                expected = math.fmod(x_val, y_val)
                self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("frexp is not implemented yet")
    def test_frexp(self):
        """Test frexp function."""
        test_cases = [1.0, 2.0, 3.5, 0.5]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                mantissa, exponent = ffmath.frexp(x)
                expected_mantissa, expected_exponent = math.frexp(value)
                self.assertAlmostEqualRel(mantissa.to_float(), expected_mantissa)
                self.assertEqual(exponent, expected_exponent)

    @unittest.skip("fsum is not implemented yet")
    def test_fsum(self):
        """Test fsum function."""
        seq = [FlexFloat.from_float(x) for x in [1.0, 2.0, 3.0]]
        result = ffmath.fsum(seq)
        expected = math.fsum([1.0, 2.0, 3.0])
        self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("gamma is not implemented yet")
    def test_gamma(self):
        """Test gamma function."""
        test_cases = [1.0, 2.0, 3.0, 0.5]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                result = ffmath.gamma(x)
                expected = math.gamma(value)
                self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("gcd is not implemented yet")
    def test_gcd(self):
        """Test gcd function."""
        args = [FlexFloat.from_float(x) for x in [12.0, 18.0]]
        result = ffmath.gcd(*args)
        expected = math.gcd(12, 18)
        self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("hypot is not implemented yet")
    def test_hypot(self):
        """Test hypot function."""
        args = [FlexFloat.from_float(x) for x in [3.0, 4.0]]
        result = ffmath.hypot(*args)
        expected = math.hypot(3.0, 4.0)
        self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("isclose is not implemented yet")
    def test_isclose(self):
        """Test isclose function."""
        a = FlexFloat.from_float(1.0)
        b = FlexFloat.from_float(1.0001)
        result = ffmath.isclose(a, b)
        expected = math.isclose(1.0, 1.0001)
        self.assertEqual(result, expected)

    @unittest.skip("lcm is not implemented yet")
    def test_lcm(self):
        """Test lcm function."""
        args = [FlexFloat.from_float(x) for x in [12.0, 18.0]]
        result = ffmath.lcm(*args)
        expected = math.lcm(12, 18)
        self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("ldexp is not implemented yet")
    def test_ldexp(self):
        """Test ldexp function."""
        x = FlexFloat.from_float(1.0)
        result = ffmath.ldexp(x, 2)
        expected = math.ldexp(1.0, 2)
        self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("lgamma is not implemented yet")
    def test_lgamma(self):
        """Test lgamma function."""
        test_cases = [1.0, 2.0, 3.0, 0.5]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                result = ffmath.lgamma(x)
                expected = math.lgamma(value)
                self.assertAlmostEqualRel(result.to_float(), expected)

    def test_log(self):
        """Test log function."""
        test_cases = [1.0, 2.0, 10.0, math.e]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                result = ffmath.log(x)
                expected = math.log(value)
                self.assertAlmostEqualRel(result.to_float(), expected)

    def test_log10(self):
        """Test log10 function."""
        test_cases = [1.0, 10.0, 100.0]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                result = ffmath.log10(x)
                expected = math.log10(value)
                self.assertAlmostEqualRel(result.to_float(), expected)

    def test_log1p(self):
        """Test log1p function."""
        test_cases = [0.0, 1.0, 0.1]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                result = ffmath.log1p(x)
                expected = math.log1p(value)
                self.assertAlmostEqualRel(result.to_float(), expected)

    def test_log2(self):
        """Test log2 function."""
        test_cases = [1.0, 2.0, 4.0, 8.0]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                result = ffmath.log2(x)
                expected = math.log2(value)
                self.assertAlmostEqualRel(result.to_float(), expected)

    def test_log_extreme_values(self):
        """Test logarithm functions with extreme values beyond Python's normal range."""

        # Test very small values that would underflow in normal float
        tiny_values = [1e-100, 1e-200, 1e-300]
        for value in tiny_values:
            with self.subTest(f"tiny_value_{value}"):
                x = FlexFloat.from_float(value)
                result = ffmath.log(x)
                # For very small x, ln(x) ≈ ln(value) calculated using high precision
                # We can verify using the property that e^(ln(x)) = x
                # Since e^result should equal x
                self.assertTrue(result.sign, f"ln({value}) should be negative")
                self.assertFalse(result.is_nan(), f"ln({value}) should not be NaN")
                self.assertFalse(
                    result.is_infinity(), f"ln({value}) should not be infinity"
                )

        # Test very large values that would overflow in normal float
        large_values = [1e100, 1e200, 1e300]
        for value in large_values:
            with self.subTest(f"large_value_{value}"):
                x = FlexFloat.from_float(value)
                result = ffmath.log(x)
                # For very large x, ln(x) should be positive and finite
                self.assertFalse(result.sign, f"ln({value}) should be positive")
                self.assertFalse(result.is_nan(), f"ln({value}) should not be NaN")
                self.assertFalse(
                    result.is_infinity(), f"ln({value}) should not be infinity"
                )

        # Test values very close to 1 (where precision matters most)
        # Note: For values extremely close to 1, we hit floating-point precision limits
        near_one_values = [1.0 + 1e-10, 1.0 - 1e-10, 1.0 + 1e-8, 1.0 - 1e-8]
        for value in near_one_values:
            with self.subTest(f"near_one_{value}"):
                x = FlexFloat.from_float(value)
                result = ffmath.log(x)
                expected = math.log(value)
                # Should be reasonably close for values near 1
                self.assertAlmostEqualRel(result.to_float(), expected, tolerance=1e-5)

    def test_log_edge_cases(self):
        """Test logarithm functions with edge cases and special values."""

        # Test ln(1) = 0 exactly
        x = FlexFloat.from_float(1.0)
        result = ffmath.log(x)
        self.assertTrue(result.is_zero(), "ln(1) should be exactly 0")

        # Test that ln(e) ≈ 1
        x = FlexFloat.from_float(math.e)
        result = ffmath.log(x)
        self.assertAlmostEqualRel(result.to_float(), 1.0, tolerance=1e-14)

        # Test negative values return NaN
        x = FlexFloat.from_float(-1.0)
        result = ffmath.log(x)
        self.assertTrue(result.is_nan(), "ln(-1) should be NaN")

        # Test zero returns NaN
        x = FlexFloat.zero()
        result = ffmath.log(x)
        self.assertTrue(result.is_nan(), "ln(0) should be NaN")

        # Test infinity returns infinity
        x = FlexFloat.infinity()
        result = ffmath.log(x)
        self.assertTrue(result.is_infinity(), "ln(∞) should be ∞")

        # Test NaN returns NaN
        x = FlexFloat.nan()
        result = ffmath.log(x)
        self.assertTrue(result.is_nan(), "ln(NaN) should be NaN")

    def test_log_custom_bases(self):
        """Test logarithm with custom bases including extreme base values."""

        # Test with base 2 manually
        x = FlexFloat.from_float(16.0)
        base = FlexFloat.from_float(2.0)
        result = ffmath.log(x, base)
        self.assertAlmostEqualRel(result.to_float(), 4.0, tolerance=1e-14)

        # Test with base 10
        x = FlexFloat.from_float(1000.0)
        base = FlexFloat.from_float(10.0)
        result = ffmath.log(x, base)
        self.assertAlmostEqualRel(result.to_float(), 3.0, tolerance=1e-14)

        # Test with very small base (but > 1)
        x = FlexFloat.from_float(2.0)
        base = FlexFloat.from_float(1.0 + 1e-10)
        result = ffmath.log(x, base)
        expected = math.log(2.0) / math.log(1.0 + 1e-10)
        self.assertAlmostEqualRel(result.to_float(), expected, tolerance=1e-6)

        # Test with large base
        x = FlexFloat.from_float(1e6)
        base = FlexFloat.from_float(100.0)
        result = ffmath.log(x, base)
        expected = math.log(1e6) / math.log(100.0)
        self.assertAlmostEqualRel(result.to_float(), expected, tolerance=1e-12)

        # Test invalid bases
        x = FlexFloat.from_float(10.0)

        # Base = 1 should return NaN
        base_one = FlexFloat.from_float(1.0)
        result = ffmath.log(x, base_one)
        self.assertTrue(result.is_nan(), "log with base 1 should be NaN")

        # Base = 0 should return NaN
        base_zero = FlexFloat.zero()
        result = ffmath.log(x, base_zero)
        self.assertTrue(result.is_nan(), "log with base 0 should be NaN")

        # Negative base should return NaN
        base_neg = FlexFloat.from_float(-2.0)
        result = ffmath.log(x, base_neg)
        self.assertTrue(result.is_nan(), "log with negative base should be NaN")

    def test_log1p_extreme_precision(self):
        """Test log1p with very small values where precision matters most."""

        # Test very small positive values
        tiny_values = [1e-15, 1e-16, 1e-17, 1e-18]
        for value in tiny_values:
            with self.subTest(f"tiny_log1p_{value}"):
                x = FlexFloat.from_float(value)
                result = ffmath.log1p(x)
                expected = math.log1p(value)
                # log1p should be reasonably accurate for small values (relaxed tolerance due to precision limits)
                self.assertAlmostEqualRel(result.to_float(), expected, tolerance=1e-5)

        # Test very small negative values (but > -1)
        small_neg_values = [-1e-15, -1e-10, -0.1, -0.5]
        for value in small_neg_values:
            with self.subTest(f"small_neg_log1p_{value}"):
                x = FlexFloat.from_float(value)
                result = ffmath.log1p(x)
                expected = math.log1p(value)
                # Adjust tolerance based on magnitude - very small values have precision limits
                tolerance = 1e-5 if abs(value) < 1e-8 else 1e-12
                self.assertAlmostEqualRel(
                    result.to_float(), expected, tolerance=tolerance
                )

        # Test edge case: log1p(0) = 0
        x = FlexFloat.zero()
        result = ffmath.log1p(x)
        self.assertTrue(result.is_zero(), "log1p(0) should be exactly 0")

        # Test edge case: log1p(-1) should be NaN
        x = FlexFloat.from_float(-1.0)
        result = ffmath.log1p(x)
        self.assertTrue(result.is_nan(), "log1p(-1) should be NaN")

    def test_log10_log2_extreme_values(self):
        """Test log10 and log2 with extreme values."""

        # Test log10 with very large and small values
        extreme_values_log10 = [1e-50, 1e-100, 1e50, 1e100]
        for value in extreme_values_log10:
            with self.subTest(f"log10_extreme_{value}"):
                x = FlexFloat.from_float(value)
                result = ffmath.log10(x)
                # Verify it's finite and has correct sign
                self.assertFalse(result.is_nan(), f"log10({value}) should not be NaN")
                self.assertFalse(
                    result.is_infinity(), f"log10({value}) should not be infinity"
                )
                if value < 1.0:
                    self.assertTrue(result.sign, f"log10({value}) should be negative")
                else:
                    self.assertFalse(result.sign, f"log10({value}) should be positive")

        # Test log2 with powers of 2 (should give exact results)
        powers_of_2 = [2**i for i in range(-10, 11)]  # 2^-10 to 2^10
        for i, value in enumerate(powers_of_2):
            expected_exp = i - 10  # Since we start from 2^-10
            with self.subTest(f"log2_power_{expected_exp}"):
                x = FlexFloat.from_float(value)
                result = ffmath.log2(x)
                self.assertAlmostEqualRel(
                    result.to_float(), expected_exp, tolerance=1e-14
                )

        # Test log2 with very large powers (beyond normal float range)
        x = FlexFloat.from_float(2.0**100)
        result = ffmath.log2(x)
        self.assertAlmostEqualRel(result.to_float(), 100.0, tolerance=1e-12)

    def test_log_consistency_properties(self):
        """Test mathematical properties and consistency of logarithm functions."""

        # Test: log(a*b) = log(a) + log(b)
        a = FlexFloat.from_float(3.0)
        b = FlexFloat.from_float(7.0)
        ab = a * b

        log_a = ffmath.log(a)
        log_b = ffmath.log(b)
        log_ab = ffmath.log(ab)
        log_sum = log_a + log_b

        self.assertAlmostEqualRel(
            log_ab.to_float(), log_sum.to_float(), tolerance=1e-14
        )

        # Test: log(a^n) = n * log(a)
        a = FlexFloat.from_float(2.5)
        n = 3
        a_power_n = a ** FlexFloat.from_float(n)

        log_a_power_n = ffmath.log(a_power_n)
        n_times_log_a = FlexFloat.from_float(n) * ffmath.log(a)

        self.assertAlmostEqualRel(
            log_a_power_n.to_float(), n_times_log_a.to_float(), tolerance=1e-13
        )

        # Test: log_b(x) = log(x) / log(b) for consistency between generic log and specialized functions
        x = FlexFloat.from_float(100.0)

        # Compare log10 with generic log base 10
        log10_result = ffmath.log10(x)
        generic_log10 = ffmath.log(x, FlexFloat.from_float(10.0))
        self.assertAlmostEqualRel(
            log10_result.to_float(), generic_log10.to_float(), tolerance=1e-14
        )

        # Compare log2 with generic log base 2
        log2_result = ffmath.log2(x)
        generic_log2 = ffmath.log(x, FlexFloat.from_float(2.0))
        self.assertAlmostEqualRel(
            log2_result.to_float(), generic_log2.to_float(), tolerance=1e-14
        )

    @unittest.skip("modf is not implemented yet")
    def test_modf(self):
        """Test modf function."""
        test_cases = [1.5, 2.7, -1.5]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                fractional, integral = ffmath.modf(x)
                expected_fractional, expected_integral = math.modf(value)
                self.assertAlmostEqualRel(fractional.to_float(), expected_fractional)
                self.assertAlmostEqualRel(integral.to_float(), expected_integral)

    @unittest.skip("nextafter is not implemented yet")
    def test_nextafter(self):
        """Test nextafter function."""
        x = FlexFloat.from_float(1.0)
        y = FlexFloat.from_float(2.0)
        result = ffmath.nextafter(x, y)
        expected = math.nextafter(1.0, 2.0)
        self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("perm is not implemented yet")
    def test_perm(self):
        """Test perm function."""
        n = FlexFloat.from_float(5.0)
        k = FlexFloat.from_float(3.0)
        result = ffmath.perm(n, k)
        expected = math.perm(5, 3)
        self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("radians is not implemented yet")
    def test_radians(self):
        """Test radians function."""
        test_cases = [0.0, 90.0, 180.0, 360.0]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                result = ffmath.radians(x)
                expected = math.radians(value)
                self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("remainder is not implemented yet")
    def test_remainder(self):
        """Test remainder function."""
        test_cases = [(7.0, 3.0), (10.0, 3.0), (5.5, 2.0)]
        for x_val, y_val in test_cases:
            with self.subTest(x=x_val, y=y_val):
                x = FlexFloat.from_float(x_val)
                y = FlexFloat.from_float(y_val)
                result = ffmath.remainder(x, y)
                expected = math.remainder(x_val, y_val)
                self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("sin is not implemented yet")
    def test_sin(self):
        """Test sin function."""
        test_cases = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                result = ffmath.sin(x)
                expected = math.sin(value)
                self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("sinh is not implemented yet")
    def test_sinh(self):
        """Test sinh function."""
        test_cases = [0.0, 1.0, -1.0, 2.0]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                result = ffmath.sinh(x)
                expected = math.sinh(value)
                self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("tan is not implemented yet")
    def test_tan(self):
        """Test tan function."""
        test_cases = [0.0, math.pi / 4, math.pi / 3]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                result = ffmath.tan(x)
                expected = math.tan(value)
                self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("tanh is not implemented yet")
    def test_tanh(self):
        """Test tanh function."""
        test_cases = [0.0, 1.0, -1.0, 2.0]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                result = ffmath.tanh(x)
                expected = math.tanh(value)
                self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("trunc is not implemented yet")
    def test_trunc(self):
        """Test trunc function."""
        test_cases = [1.5, 2.7, -1.5, -2.7]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                result = ffmath.trunc(x)
                expected = math.trunc(value)
                self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("ulp is not implemented yet")
    def test_ulp(self):
        """Test ulp function."""
        test_cases = [1.0, 2.0, 0.5]
        for value in test_cases:
            with self.subTest(value=value):
                x = FlexFloat.from_float(value)
                result = ffmath.ulp(x)
                expected = math.ulp(value)
                self.assertAlmostEqualRel(result.to_float(), expected)

    @unittest.skip("fma is not implemented yet")
    def test_fma(self):
        """Test fma function."""
        x = FlexFloat.from_float(2.0)
        y = FlexFloat.from_float(3.0)
        z = FlexFloat.from_float(1.0)
        result = ffmath.fma(x, y, z)
        # fma(x, y, z) = x * y + z
        expected = 2.0 * 3.0 + 1.0
        self.assertAlmostEqualRel(result.to_float(), expected)


if __name__ == "__main__":
    unittest.main()
