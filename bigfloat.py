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
    max_number: ClassVar[float] = 1e100  # Default maximum number for BigFloat
    def __init__(self, value: Number | None = None, level: int = 0, sign: SignValue | None = None):
        if value is None:
            value = 0.0
        elif not isinstance(value, (int, float)):
            raise TypeError("Value must be an int, float, or None.")
        
        self.value = value
        self.level = level
        self.sign = SignValue.from_value(value) if sign is None else sign

    def __repr__(self) -> str:
        return f"BigFloat({self.value=}, {self.level=}, {self.sign=})"
    
    def partial_eq(self, other: BigFloat, precision: float = 1e-10) -> bool:
        if not isinstance(other, BigFloat):
            raise TypeError("Comparison must be with another BigFloat instance.")
        if self is other:
            return True
        return self.level == other.level and self.sign == other.sign and math.isclose(self.value, other.value, abs_tol=precision)
    
    def __add__(self, other: BigFloat | Number) -> BigFloat:
        if isinstance(other, BigFloat):
            return self._add_bigfloat(other)
        elif isinstance(other, (int, float)):
            return self._add_number(other)
        else:
            raise TypeError("Unsupported type for addition. Must be BigFloat, int, or float.")
        
    def _check_overflow(self, new_value: Number) -> BigFloat:
        new_value_sign = SignValue.from_value(new_value)
        if self.sign == new_value_sign:
            match new_value_sign:
                case SignValue.ZERO:
                    return BigFloat(0.0, level=0, sign=SignValue.ZERO)
                case SignValue.POSITIVE:
                    if 0 < new_value < BigFloat.max_number:
                        return BigFloat(new_value, level=self.level, sign=SignValue.POSITIVE)
                    if new_value >= BigFloat.max_number:
                        if math.isinf(new_value):
                            return BigFloat(1, level=self.level + 1, sign=SignValue.POSITIVE)

                        return BigFloat(math.log(new_value, BigFloat.max_number), level=self.level + 1, sign=SignValue.POSITIVE)
                    if math.isinf(new_value):
                        return BigFloat(BigFloat.max_number, level=self.level - 1, sign=SignValue.POSITIVE)
                    raise NotImplementedError("Not implemented yet unederflow for positive values.")
                case SignValue.NEGATIVE:
                    raise NotImplementedError("Not implemented yet for negative values.")
        raise NotImplementedError("Not implemented yet subtraction for different signs.")
                
    def _add_bigfloat(self, other: BigFloat) -> BigFloat:
        if self.level == other.level:
            new_value = self.value + other.value
            return self._check_overflow(new_value)
        
        if self.level > other.level:
            return other._add_bigfloat(self)
        number = self.value
        for _ in range(other.level - self.level):
            number = math.log(number, BigFloat.max_number)
            if number < 0:
                number = 0.0
                break
        new_value = other.value + number
        return other._check_overflow(new_value)

        
    def _add_number(self, number: Number) -> BigFloat:
        for _ in range(self.level):
            number = math.log(number, BigFloat.max_number)
            if number < 0:
                number = 0.0
                break

        new_value = self.value + number
        return self._check_overflow(new_value)
        
    
    
class BigFloatUnitTest(TestCase):
    @staticmethod
    def assert_almost_equal(a: BigFloat, b: BigFloat):
        assert a.partial_eq(b), f"Expected {b}, got {a}"

    def test_addition_number_multiplicative(self):
        bf1 = BigFloat(5.0)
        result = bf1 + 2.0

        expected = BigFloat(7.0)
        BigFloatUnitTest.assert_almost_equal(result, expected)

    def test_addition_number_exponential(self):
        bf2 = BigFloat(3.0, level=1)
        result = bf2 + BigFloat.max_number

        expected = BigFloat(4.0, level=1)
        BigFloatUnitTest.assert_almost_equal(result, expected)

    def test_addition_bigfloat_multiplicative(self):
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