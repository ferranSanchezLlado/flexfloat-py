from __future__ import annotations
import math
from enum import IntEnum
from typing import ClassVar
from unittest import TestCase

Number = int | float

class SignValue(IntEnum):
    NEGATIVE = -1
    ZERO = 0
    POSITIVE = 1

    @classmethod
    def from_value(cls, value: Number) -> SignValue:
        """Returns the sign of a number as a SignValue enum."""
        if value < 0:
            return cls.NEGATIVE
        elif value > 0:
            return cls.POSITIVE
        else:
            return cls.ZERO

# BigFloat is a class that approximates very large or very small floating-point numbers
# This is done by represeting each digit in a huge base (e.g., base 10^100)
class BigFloat:
    max_number: ClassVar[float] = 1e100
    min_number: ClassVar[float] = 1e-100

    def __init__(self, value: Number | None = None, level: int = 0):
        if value is None:
            value = 0.0
        elif not isinstance(value, (int, float)):
            raise TypeError("Value must be an int, float, or None.")
        
        self.value = value
        self.level = level

    def __repr__(self) -> str:
        return f"BigFloat({self.value=}, {self.level=})"
    
    def partial_eq(self, other: BigFloat, precision: float = 1e-10) -> bool:
        if not isinstance(other, BigFloat):
            raise TypeError("Comparison must be with another BigFloat instance.")
        if self is other:
            return True
        # TODO: fix bug where the difference between levels is not considered and could be precision away
        return self.level == other.level and math.isclose(self.value, other.value, abs_tol=precision)
    
    def __add__(self, other: BigFloat | Number) -> BigFloat:
        if isinstance(other, BigFloat):
            return self._add_bigfloat(other)
        elif isinstance(other, (int, float)):
            return self._add_number(other)
        else:
            raise TypeError("Unsupported type for addition. Must be BigFloat, int, or float.")
        
    def _check_overflow(self, new_value: Number) -> BigFloat:
        new_value_sign = SignValue.from_value(new_value)

        if new_value_sign == SignValue.ZERO:
            if self.level == 0:
                return BigFloat(0.0, level=0)
            return BigFloat(BigFloat.min_number, level=self.level - 1)
           
        abs_value = math.fabs(new_value)
        
        if math.isinf(new_value) or math.isnan(new_value):
            return BigFloat(int(new_value_sign), level=self.level + new_value_sign)
        if abs_value >= BigFloat.max_number:
            return BigFloat(int(new_value_sign) * math.log(abs_value, BigFloat.max_number), level=self.level + new_value_sign)
        if abs_value < BigFloat.min_number:
            return BigFloat(int(new_value_sign) * math.pow(abs_value, BigFloat.max_number), level=self.level - new_value_sign)
        return BigFloat(new_value, level=self.level)

                
    def _add_bigfloat(self, other: BigFloat) -> BigFloat:
        if self.level == other.level:
            new_value = self.value + other.value
            return self._check_overflow(new_value)
        
        if self.level > other.level:
            return other._add_bigfloat(self)
        number = math.fabs(self.value)
        for _ in range(other.level - self.level):
            number = math.log(number, BigFloat.max_number)
            if number < 0:
                number = 0.0
                break
        new_value = other.value + SignValue.from_value(self.value) * number
        return other._check_overflow(new_value)

        
    def _add_number(self, number: Number) -> BigFloat:
        new_number = math.fabs(number)
        for _ in range(self.level):
            new_number = math.log(new_number, BigFloat.max_number)
            if new_number < 0:
                new_number = 0.0
                break

        new_value = self.value + SignValue.from_value(number) * new_number
        return self._check_overflow(new_value)
        
    def __neg__(self) -> BigFloat:
        return BigFloat(-self.value, self.level)
    
    def __sub__(self, other: BigFloat | Number) -> BigFloat:
        if isinstance(other, BigFloat):
            return self + -other
        elif isinstance(other, (int, float)):
            return self + -other
        else:
            raise TypeError("Unsupported type for subtraction. Must be BigFloat, int, or float." )
    
    
class BigFloatUnitTest(TestCase):
    @staticmethod
    def assert_almost_equal(a: BigFloat, b: BigFloat):
        assert a.partial_eq(b), f"Expected {b}, got {a}"

    def test_addition_number_normal(self):
        bf1 = BigFloat(5.0)
        result = bf1 + 2.0

        expected = BigFloat(7.0)
        BigFloatUnitTest.assert_almost_equal(result, expected)

    def test_addition_number_1_level(self):
        bf2 = BigFloat(3.0, level=1)
        result = bf2 + BigFloat.max_number

        expected = BigFloat(4.0, level=1)
        BigFloatUnitTest.assert_almost_equal(result, expected)

    def test_addition_number_overflow(self):
        bf1 = BigFloat(5.0)
        result = bf1 + BigFloat.max_number

        expected = BigFloat(1.0, level=1)
        BigFloatUnitTest.assert_almost_equal(result, expected)

    def test_addition_bigfloat_normal(self):
        bf1 = BigFloat(5.0)
        bf2 = BigFloat(3.0)
        result = bf1 + bf2
        
        expected = BigFloat(8.0)
        BigFloatUnitTest.assert_almost_equal(result, expected)

    def test_addition_bigfloat_same_level(self):
        bf1 = BigFloat(5.0, level=1)
        bf2 = BigFloat(3.0, level=1)
        result = bf1 + bf2
        
        expected = BigFloat(8.0, level=1)
        BigFloatUnitTest.assert_almost_equal(result, expected)

    def test_addition_bigfloat_different_levels(self):
        bf1 = BigFloat(5.0, level=1)
        bf2 = BigFloat(3.0, level=0)
        result = bf1 + bf2

        expected = BigFloat(5.0047712125471966, level=1)
        BigFloatUnitTest.assert_almost_equal(result, expected)

    # --- Testing addition with negative numbers ---
    def test_addition_negative_number_normal(self):
        bf1 = BigFloat(5.0)
        result = bf1 + -2.0

        expected = BigFloat(3.0)
        BigFloatUnitTest.assert_almost_equal(result, expected)

    def test_addition_negative_number_1_level(self):
        bf2 = BigFloat(3.0, level=1)
        result = bf2 + -BigFloat.max_number

        expected = BigFloat(2.0, level=1)
        BigFloatUnitTest.assert_almost_equal(result, expected)

    def test_addition_negative_bigfloat_normal(self):
        bf1 = BigFloat(5.0)
        bf2 = BigFloat(-3.0)
        result = bf1 + bf2
        
        expected = BigFloat(2.0)
        BigFloatUnitTest.assert_almost_equal(result, expected)

    def test_addition_negative_bigfloat_same_level(self):
        bf1 = BigFloat(5.0, level=1)
        bf2 = BigFloat(-6.0, level=1)
        result = bf1 + bf2
        
        expected = BigFloat(-1.0, level=1)
        BigFloatUnitTest.assert_almost_equal(result, expected)

    def test_addition_negative_bigfloat_zero(self):
        bf1 = BigFloat(5.0, level=1)
        bf2 = BigFloat(-5.0, level=1)
        result = bf1 + bf2
        
        expected = BigFloat(0.0, level=0)
        BigFloatUnitTest.assert_almost_equal(result, expected)

    def test_addition_negative_bigfloat_different_levels(self):
        bf1 = BigFloat(5.0, level=1)
        bf2 = BigFloat(-3.0, level=0)
        result = bf1 + bf2

        expected = BigFloat(4.9952287874528034, level=1)
        BigFloatUnitTest.assert_almost_equal(result, expected)

    # --- Testing subtraction ---
    def test_subtraction_number_normal(self):
        bf1 = BigFloat(5.0)
        result = bf1 - 2.0

        expected = BigFloat(3.0)
        BigFloatUnitTest.assert_almost_equal(result, expected)

    def test_subtraction_number_1_level(self):
        bf2 = BigFloat(3.0, level=1)
        result = bf2 - BigFloat.max_number

        expected = BigFloat(2.0, level=1)
        BigFloatUnitTest.assert_almost_equal(result, expected)

    def test_subtraction_number_overflow(self):
        bf1 = BigFloat(5.0)
        result = bf1 - BigFloat.max_number

        expected = BigFloat(-1, level=-1)
        BigFloatUnitTest.assert_almost_equal(result, expected)

    def test_subtraction_bigfloat_normal(self):
        bf1 = BigFloat(5.0)
        bf2 = BigFloat(3.0)
        result = bf1 - bf2
        
        expected = BigFloat(2.0)
        BigFloatUnitTest.assert_almost_equal(result, expected)

    def test_subtraction_bigfloat_same_level(self):
        bf1 = BigFloat(5.0, level=1)
        bf2 = BigFloat(3.0, level=1)
        result = bf1 - bf2
        
        expected = BigFloat(2.0, level=1)
        BigFloatUnitTest.assert_almost_equal(result, expected)

    def test_subtraction_bigfloat_different_levels(self):
        bf1 = BigFloat(5.0, level=1)
        bf2 = BigFloat(3.0, level=0)
        result = bf1 - bf2

        expected = BigFloat(4.995228787452803, level=1)
        BigFloatUnitTest.assert_almost_equal(result, expected)

    # --- Testing edge cases and error handling ---
    def test_init_with_none(self):
        bf = BigFloat(None)
        expected = BigFloat(0.0)
        BigFloatUnitTest.assert_almost_equal(bf, expected)

    def test_init_with_invalid_type(self):
        with self.assertRaises(TypeError):
            BigFloat("invalid")

    def test_init_with_custom_level(self):
        bf = BigFloat(5.0, level=2)
        self.assertEqual(bf.value, 5.0)
        self.assertEqual(bf.level, 2)

    def test_repr(self):
        bf = BigFloat(3.14, level=1)
        expected_repr = "BigFloat(self.value=3.14, self.level=1)"
        self.assertEqual(repr(bf), expected_repr)

    # --- Testing partial_eq method ---
    def test_partial_eq_same_instance(self):
        bf = BigFloat(5.0)
        self.assertTrue(bf.partial_eq(bf))

    def test_partial_eq_different_levels(self):
        bf1 = BigFloat(5.0, level=1)
        bf2 = BigFloat(5.0, level=2)
        self.assertFalse(bf1.partial_eq(bf2))

    def test_partial_eq_within_precision(self):
        bf1 = BigFloat(5.0000000001)
        bf2 = BigFloat(5.0)
        self.assertTrue(bf1.partial_eq(bf2))

    def test_partial_eq_outside_precision(self):
        bf1 = BigFloat(5.1)
        bf2 = BigFloat(5.0)
        self.assertFalse(bf1.partial_eq(bf2))

    def test_partial_eq_invalid_type(self):
        bf = BigFloat(5.0)
        with self.assertRaises(TypeError):
            bf.partial_eq(5.0)

    # --- Testing SignValue enum ---
    def test_sign_value_positive(self):
        sign = SignValue.from_value(5.5)
        self.assertEqual(sign, SignValue.POSITIVE)

    def test_sign_value_negative(self):
        sign = SignValue.from_value(-3.2)
        self.assertEqual(sign, SignValue.NEGATIVE)

    def test_sign_value_zero(self):
        sign = SignValue.from_value(0)
        self.assertEqual(sign, SignValue.ZERO)

    # --- Testing overflow conditions ---
    def test_overflow_with_infinity(self):
        bf = BigFloat(5.0)
        result = bf._check_overflow(float('inf'))
        expected = BigFloat(1, level=1)
        BigFloatUnitTest.assert_almost_equal(result, expected)

    def test_overflow_with_negative_infinity(self):
        bf = BigFloat(5.0)
        result = bf._check_overflow(float('-inf'))
        expected = BigFloat(-1, level=1)
        BigFloatUnitTest.assert_almost_equal(result, expected)

    def test_overflow_with_nan(self):
        bf = BigFloat(5.0)
        result = bf._check_overflow(float('nan'))
        # NaN handling might need adjustment based on expected behavior
        self.assertTrue(math.isnan(result.value) or result.value in [-1, 0, 1])

    def test_underflow_condition(self):
        bf = BigFloat(level=5.0)
        small_value = 1e-150  # Smaller than min_number
        result = bf._check_overflow(small_value)
        self.assertEqual(result.level, 4)  # level should decrease

    # --- Testing addition with zero ---
    def test_addition_with_zero_bigfloat(self):
        bf1 = BigFloat(5.0)
        bf2 = BigFloat(0.0)
        result = bf1 + bf2
        expected = BigFloat(5.0)
        BigFloatUnitTest.assert_almost_equal(result, expected)

    def test_addition_with_zero_number(self):
        bf = BigFloat(5.0)
        result = bf + 0
        expected = BigFloat(5.0)
        BigFloatUnitTest.assert_almost_equal(result, expected)

    # --- Testing negation ---
    def test_negation_positive(self):
        bf = BigFloat(5.0, level=1)
        result = -bf
        expected = BigFloat(-5.0, level=1)
        BigFloatUnitTest.assert_almost_equal(result, expected)

    def test_negation_negative(self):
        bf = BigFloat(-3.0, level=2)
        result = -bf
        expected = BigFloat(3.0, level=2)
        BigFloatUnitTest.assert_almost_equal(result, expected)

    def test_negation_zero(self):
        bf = BigFloat(0.0)
        result = -bf
        expected = BigFloat(0.0)
        BigFloatUnitTest.assert_almost_equal(result, expected)

    # --- Testing type errors ---
    def test_addition_invalid_type(self):
        bf = BigFloat(5.0)
        with self.assertRaises(TypeError):
            bf + "invalid"

    def test_subtraction_invalid_type(self):
        bf = BigFloat(5.0)
        with self.assertRaises(TypeError):
            bf - "invalid"

    # --- Testing extreme values ---
    def test_addition_very_large_numbers(self):
        bf1 = BigFloat(9.99e99)
        bf2 = BigFloat(1.1e99)
        result = bf1 + bf2
        # Should handle large numbers without overflow to next level
        self.assertLess(result.value, 2)
        self.assertEqual(result.level, 1)

    def test_addition_causes_level_increase(self):
        bf1 = BigFloat(9.9e99)
        bf2 = BigFloat(9.9e99)
        result = bf1 + bf2
        # Should overflow to next level
        self.assertEqual(result.level, 1)

    def test_subtraction_to_zero_different_levels(self):
        bf1 = BigFloat(5.0, level=2)
        bf2 = BigFloat(5.0, level=2)
        result = bf1 - bf2
        expected = BigFloat(0.0, level=0)
        BigFloatUnitTest.assert_almost_equal(result, expected)

    # --- Testing multiple level differences ---
    def test_addition_large_level_difference(self):
        bf1 = BigFloat(5.0, level=5)
        bf2 = BigFloat(1000.0, level=0)
        result = bf1 + bf2
        # The smaller level number should have minimal impact
        expected = BigFloat(5.0, level=5)  # Approximately
        self.assertEqual(result.level, 5)

    def test_class_variables(self):
        self.assertEqual(BigFloat.max_number, 1e100)
        self.assertEqual(BigFloat.min_number, 1e-100)
