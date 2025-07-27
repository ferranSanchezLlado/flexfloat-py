"""Core FlexFloat class implementation.

This module defines the FlexFloat class, which represents a floating-point number with a
growable exponent and a fixed-size fraction. The class is designed for arbitrary
precision floating-point arithmetic, supporting very large or small values by
dynamically adjusting the exponent size.

Example:
    from flexfloat import FlexFloat
    a = FlexFloat(3.14, exponent_length=10, fraction_length=20)
    b = FlexFloat(2.71, exponent_length=10, fraction_length=20)
    c = a + b
    print(c)
    # Output: FlexFloat(...)
"""

from __future__ import annotations

import math
from typing import ClassVar, Final, Type

from .bitarray import BitArray, ListBoolBitArray
from .types import Number

LOG10_2: Final[float] = math.log10(2)


class FlexFloat:
    """A class to represent a floating-point number with growable exponent and
    fixed-size fraction. This class is designed to handle very large or very
    small numbers by adjusting the exponent dynamically. While keeping the
    mantissa (fraction) fixed in size.

    This class follows the IEEE 754 double-precision floating-point format,
    but extends it to allow for a growable exponent and a fixed-size fraction.

    Attributes:
        sign (bool): The sign of the number (True for negative, False for positive).
        exponent (BitArray): A growable bit array representing the exponent
            (uses off-set binary representation).
        fraction (BitArray): A fixed-size bit array representing the fraction
            (mantissa) of the number.
    """

    _bitarray_implementation: ClassVar[Type[BitArray]] = ListBoolBitArray
    """The BitArray implementation class used for all FlexFloat instances."""

    sign: bool
    """The sign of the number (True for negative, False for positive)."""
    exponent: BitArray
    """A growable bit array representing the exponent (uses off-set binary
    representation)."""
    fraction: BitArray
    """A fixed-size bit array representing the fraction (mantissa) of the number."""

    @classmethod
    def set_bitarray_implementation(cls, implementation: Type[BitArray]) -> None:
        """Set the BitArray implementation to use for all FlexFloat instances.

        Args:
            implementation (Type[BitArray]): The BitArray implementation class to use.
        """
        cls._bitarray_implementation = implementation

    def __init__(
        self,
        sign: bool = False,
        exponent: BitArray | None = None,
        fraction: BitArray | None = None,
    ):
        """Initializes a FlexFloat instance.

        BitArrays are expected to be LSB-first (least significant bit at index 0,
        increasing to MSB).

        Args:
            sign (bool, optional): The sign of the number (True for negative, False for
                positive). Defaults to False.
            exponent (BitArray | None, optional): The exponent bit array. If None,
                represents 0. Defaults to None.
            fraction (BitArray | None, optional): The fraction bit array. If None,
                represents 0. Defaults to None.
        """
        self.sign = sign
        self.exponent = (
            exponent
            if exponent is not None
            else self._bitarray_implementation.zeros(11)
        )
        self.fraction = (
            fraction
            if fraction is not None
            else self._bitarray_implementation.zeros(52)
        )

    @classmethod
    def from_float(cls, value: Number) -> FlexFloat:
        """Creates a FlexFloat instance from a number.

        Args:
            value (Number): The number to convert to FlexFloat.

        Returns:
            FlexFloat: A new FlexFloat instance representing the number.
        """
        value = float(value)
        bits = cls._bitarray_implementation.from_float(value)

        return cls(sign=bits[63], exponent=bits[52:63], fraction=bits[:52])

    def to_float(self) -> float:
        """Converts the FlexFloat instance back to a 64-bit float.

        If float is bigger than 64 bits, it will truncate the value to fit.

        Returns:
            float: The floating-point number represented by the FlexFloat instance.

        Raises:
            ValueError: If the FlexFloat does not have standard 64-bit exponent and
                fraction.
        """
        if len(self.exponent) < 11 or len(self.fraction) < 52:
            raise ValueError("Must be a standard 64-bit FlexFloat")

        bits = (
            self.fraction[:52]
            + self.exponent[:11]
            + self._bitarray_implementation.from_bits([self.sign])
        )
        return bits.to_float()

    def __repr__(self) -> str:
        """Returns a string representation of the FlexFloat instance.

        Returns:
            str: A string representation of the FlexFloat instance.
        """
        return (
            "FlexFloat("
            f"sign={self.sign}, "
            f"exponent={self.exponent}, "
            f"fraction={self.fraction})"
        )

    def pretty(self) -> str:
        """Returns an easier to read string representation of the FlexFloat instance.
        Mainly converts the exponent and fraction to integers for readability.

        Returns:
            str: A pretty string representation of the FlexFloat instance.
        """
        sign = "-" if self.sign else ""
        exponent_value = self.exponent.to_signed_int() + 1
        fraction_value = self.fraction.to_int()
        return f"{sign}FlexFloat(exponent={exponent_value}, fraction={fraction_value})"

    @classmethod
    def nan(cls) -> FlexFloat:
        """Creates a FlexFloat instance representing NaN (Not a Number).

        Returns:
            FlexFloat: A new FlexFloat instance representing NaN.
        """
        exponent = cls._bitarray_implementation.ones(11)
        fraction = cls._bitarray_implementation.ones(52)
        return cls(sign=True, exponent=exponent, fraction=fraction)

    @classmethod
    def infinity(cls, sign: bool = False) -> FlexFloat:
        """Creates a FlexFloat instance representing Infinity.

        Args:
            sign (bool, optional): Indicates if the infinity is negative. Defaults to
                False.

        Returns:
            FlexFloat: A new FlexFloat instance representing Infinity.
        """
        exponent = cls._bitarray_implementation.ones(11)
        fraction = cls._bitarray_implementation.zeros(52)
        return cls(sign=sign, exponent=exponent, fraction=fraction)

    @classmethod
    def zero(cls) -> FlexFloat:
        """Creates a FlexFloat instance representing zero.

        Returns:
            FlexFloat: A new FlexFloat instance representing zero.
        """
        exponent = cls._bitarray_implementation.zeros(11)
        fraction = cls._bitarray_implementation.zeros(52)
        return cls(sign=False, exponent=exponent, fraction=fraction)

    def _is_special_exponent(self) -> bool:
        """Checks if the exponent represents a special value (NaN or Infinity).

        Returns:
            bool: True if the exponent is at its maximum value, False otherwise.
        """
        max_signed_value = (1 << (len(self.exponent) - 1)) - 1
        return self.exponent.to_signed_int() == max_signed_value

    def is_nan(self) -> bool:
        """Checks if the FlexFloat instance represents NaN (Not a Number).

        Returns:
            bool: True if the FlexFloat instance is NaN, False otherwise.
        """
        return self._is_special_exponent() and any(self.fraction)

    def is_infinity(self) -> bool:
        """Checks if the FlexFloat instance represents Infinity.

        Returns:
            bool: True if the FlexFloat instance is Infinity, False otherwise.
        """
        return self._is_special_exponent() and not any(self.fraction)

    def is_zero(self) -> bool:
        """Checks if the FlexFloat instance represents zero.

        Returns:
            bool: True if the FlexFloat instance is zero, False otherwise.
        """
        return not any(self.exponent) and not any(self.fraction)

    def copy(self) -> FlexFloat:
        """Creates a copy of the FlexFloat instance.

        Returns:
            FlexFloat: A new FlexFloat instance with the same data as the original.
        """
        return FlexFloat(
            sign=self.sign, exponent=self.exponent.copy(), fraction=self.fraction.copy()
        )

    def __str__(self) -> str:
        """Returns a float representation of the FlexFloat using a generic algorithm.

        Currently, it only operates in one format: scientific notation with 5 decimal
        places.

        Returns:
            str: The string representation in scientific notation.
        """
        sign_str = "-" if self.sign else ""
        # Handle special cases first
        if self.is_nan():
            return "nan"

        if self.is_infinity():
            return f"{sign_str}inf"

        if self.is_zero():
            return f"{sign_str}0.00000e+00"

        exponent = self.exponent.to_signed_int() + 1

        # Convert fraction to decimal value between 1 and 2
        # (starting with 1.0 for the implicit leading bit)
        mantissa = 1.0
        for i, bit in enumerate(reversed(self.fraction)):
            if bit:
                mantissa += 1.0 / (1 << (i + 1))

        # To avoid overflow with very large exponents, work in log space
        # log10(mantissa * 2^exponent) = log10(mantissa) + exponent * log10(2)
        log10_mantissa = math.log10(mantissa)
        log10_total = log10_mantissa + exponent * LOG10_2

        decimal_exponent = int(log10_total)

        log10_normalized = log10_total - decimal_exponent
        normalized_mantissa = math.pow(10, log10_normalized)

        # Ensure the mantissa is properly normalized (between 1.0 and 10.0)
        while normalized_mantissa >= 10.0:
            normalized_mantissa /= 10.0
            decimal_exponent += 1

        while normalized_mantissa < 1.0:
            normalized_mantissa *= 10.0
            decimal_exponent -= 1

        # Format with 5 decimal places
        return f"{sign_str}{normalized_mantissa:.5f}e{decimal_exponent:+03d}"

    def __neg__(self) -> FlexFloat:
        """Negates the FlexFloat instance.

        Returns:
            FlexFloat: A new FlexFloat instance with the sign flipped.
        """
        return FlexFloat(
            sign=not self.sign,
            exponent=self.exponent.copy(),
            fraction=self.fraction.copy(),
        )

    @staticmethod
    def _grow_exponent(exponent: int, exponent_length: int) -> int:
        """Grows the exponent if it exceeds the maximum value for the current length.

        Args:
            exponent (int): The current exponent value.
            exponent_length (int): The current length of the exponent in bits.

        Returns:
            int: The new exponent length if it needs to be grown, otherwise the same
                length.
        """
        while True:
            half = 1 << (exponent_length - 1)
            min_exponent = -half
            max_exponent = half - 1

            if min_exponent <= exponent <= max_exponent:
                break
            exponent_length += 1

        return exponent_length

    def __add__(self, other: FlexFloat | Number) -> FlexFloat:
        """Adds two FlexFloat instances together.

        Args:
            other (FlexFloat | Number): The other FlexFloat instance to add.

        Returns:
            FlexFloat: A new FlexFloat instance representing the sum.

        Raises:
            TypeError: If other is not a FlexFloat or numeric type.
        """
        if isinstance(other, Number):
            other = FlexFloat.from_float(other)
        if not isinstance(other, FlexFloat):
            raise TypeError("Can only add FlexFloat instances.")

        if self.sign != other.sign:
            return self - (-other)

        # OBJECTIVE: Add two FlexFloat instances together.
        # https://www.sciencedirect.com/topics/computer-science/floating-point-addition
        # and: https://cse.hkust.edu.hk/~cktang/cs180/notes/lec21.pdf
        #
        # Steps:
        # 0. Handle special cases (NaN, Infinity).
        # 1. Extract exponent and fraction bits.
        # 2. Prepend leading 1 to form the mantissa.
        # 3. Compare exponents.
        # 4. Shift smaller mantissa if necessary.
        # 5. Add mantissas.
        # 6. Normalize mantissa and adjust exponent if necessary.
        # 7. Grow exponent if necessary.
        # 8. Round result.
        # 9. Return new FlexFloat instance.

        # Step 0: Handle special cases
        if self.is_zero() or other.is_zero():
            return self.copy() if other.is_zero() else other.copy()

        if self.is_nan() or other.is_nan():
            return FlexFloat.nan()

        if self.is_infinity() and other.is_infinity():
            return self.copy() if self.sign == other.sign else FlexFloat.nan()
        if self.is_infinity() or other.is_infinity():
            return self.copy() if self.is_infinity() else other.copy()

        # Step 1: Extract exponent and fraction bits
        exponent_self = self.exponent.to_signed_int() + 1
        exponent_other = other.exponent.to_signed_int() + 1

        # Step 2: Append the implicit leading 1 to form the mantissa
        mantissa_self = self.fraction + [True]
        mantissa_other = other.fraction + [True]

        # Step 3: Compare exponents (self is always larger or equal)
        if exponent_self < exponent_other:
            exponent_self, exponent_other = exponent_other, exponent_self
            mantissa_self, mantissa_other = mantissa_other, mantissa_self

        # Step 4: Shift smaller mantissa if necessary
        if exponent_self > exponent_other:
            shift_amount = exponent_self - exponent_other
            mantissa_other = mantissa_other.shift(shift_amount)

        # Step 5: Add mantissas
        assert (
            len(mantissa_self) == 53
        ), "Fraction must be 53 bits long. (1 leading bit + 52 fraction bits)"
        assert len(mantissa_self) == len(mantissa_other), (
            f"Mantissas must be the same length. Expected 53 bits, "
            f"got {len(mantissa_other)} bits."
        )

        # 1 leading bit + 52 fraction bits
        mantissa_result = self._bitarray_implementation.zeros(53)
        carry = False
        for i in range(53):
            total = mantissa_self[i] + mantissa_other[i] + carry
            mantissa_result[i] = total % 2 == 1
            carry = total > 1

        # Step 6: Normalize mantissa and adjust exponent if necessary
        # Only need to normalize if there is a carry
        if carry:
            # Insert the carry bit and shift right
            mantissa_result = mantissa_result.shift(1, fill=True)
            exponent_self += 1

        # Step 7: Grow exponent if necessary
        exp_result_length = self._grow_exponent(exponent_self, len(self.exponent))
        assert (
            exponent_self - (1 << (exp_result_length - 1)) < 2
        ), "Exponent growth should not exceed 1 bit."

        exponent_result = self._bitarray_implementation.from_signed_int(
            exponent_self - 1, exp_result_length
        )
        return FlexFloat(
            sign=self.sign,
            exponent=exponent_result,
            fraction=mantissa_result[:-1],  # Exclude leading bit
        )

    def __sub__(self, other: FlexFloat | Number) -> FlexFloat:
        """Subtracts one FlexFloat instance from another.

        Args:
            other (FlexFloat | Number): The FlexFloat instance to subtract.

        Returns:
            FlexFloat: A new FlexFloat instance representing the difference.

        Raises:
            TypeError: If other is not a FlexFloat or numeric type.
        """
        if isinstance(other, Number):
            other = FlexFloat.from_float(other)
        if not isinstance(other, FlexFloat):
            raise TypeError("Can only subtract FlexFloat instances.")

        # If signs are different, subtraction becomes addition
        if self.sign != other.sign:
            return self + (-other)

        # OBJECTIVE: Subtract two FlexFloat instances.
        # Based on floating-point subtraction algorithms
        #
        # Steps:
        # 0. Handle special cases (NaN, Infinity, zero).
        # 1. Extract exponent and fraction bits.
        # 2. Prepend leading 1 to form the mantissa.
        # 3. Compare exponents and align mantissas.
        # 4. Compare magnitudes to determine result sign.
        # 5. Subtract mantissas (larger - smaller).
        # 6. Normalize mantissa and adjust exponent if necessary.
        # 7. Grow exponent if necessary.
        # 8. Return new FlexFloat instance.

        # Step 0: Handle special cases
        if self.is_zero() or other.is_zero():
            return self.copy() if other.is_zero() else -other.copy()

        if self.is_nan() or other.is_nan():
            return FlexFloat.nan()

        if self.is_infinity() and other.is_infinity():
            if self.sign == other.sign:
                return FlexFloat.nan()  # inf - inf = NaN
            return self.copy()  # inf - (-inf) = inf

        if self.is_infinity():
            return self.copy()

        if other.is_infinity():
            return -other

        # Step 1: Extract exponent and fraction bits
        exponent_self = self.exponent.to_signed_int() + 1
        exponent_other = other.exponent.to_signed_int() + 1

        # Step 2: Append the implicit leading 1 to form the mantissa
        mantissa_self = self.fraction + [True]
        mantissa_other = other.fraction + [True]

        # Step 3: Align mantissas by shifting the smaller exponent
        result_sign = self.sign
        shift_amount = abs(exponent_self - exponent_other)
        if exponent_self >= exponent_other:
            mantissa_other = mantissa_other.shift(shift_amount)
            result_exponent = exponent_self
        else:
            mantissa_self = mantissa_self.shift(shift_amount)
            result_exponent = exponent_other

        # Step 4: Compare magnitudes to determine which mantissa is larger
        # Convert mantissas to integers for comparison
        mantissa_self_int = mantissa_self.to_int()
        mantissa_other_int = mantissa_other.to_int()

        if mantissa_self_int >= mantissa_other_int:
            larger_mantissa = mantissa_self
            smaller_mantissa = mantissa_other
            result_sign = self.sign
        else:
            larger_mantissa = mantissa_other
            smaller_mantissa = mantissa_self
            # Flip sign since we're computing -(smaller - larger)
            result_sign = not self.sign

        # Step 5: Subtract mantissas (larger - smaller)
        assert (
            len(larger_mantissa) == 53
        ), "Mantissa must be 53 bits long. (1 leading bit + 52 fraction bits)"
        assert len(larger_mantissa) == len(smaller_mantissa), (
            f"Mantissas must be the same length. Expected 53 bits, "
            f"got {len(smaller_mantissa)} bits."
        )

        mantissa_result = self._bitarray_implementation.zeros(53)
        borrow = False
        for i in range(53):
            diff = int(larger_mantissa[i]) - int(smaller_mantissa[i]) - int(borrow)

            mantissa_result[i] = diff % 2 == 1
            borrow = diff < 0

        assert not borrow, "Subtraction should not result in a negative mantissa."

        # Step 6: Normalize mantissa and adjust exponent if necessary
        # Find the first 1 bit (leading bit might have been canceled out)
        leading_zero_count = next(
            (i for i, bit in enumerate(reversed(mantissa_result)) if bit),
            len(mantissa_result),
        )

        # Handle case where result becomes zero or denormalized
        if leading_zero_count >= 53:
            return FlexFloat.from_float(0.0)

        if leading_zero_count > 0:
            # Shift left to normalize
            mantissa_result = mantissa_result.shift(-leading_zero_count)
            result_exponent -= leading_zero_count

        # Step 7: Grow exponent if necessary (handle underflow)
        exp_result_length = self._grow_exponent(result_exponent, len(self.exponent))

        exp_result = self._bitarray_implementation.from_signed_int(
            result_exponent - 1, exp_result_length
        )

        return FlexFloat(
            sign=result_sign,
            exponent=exp_result,
            fraction=mantissa_result[:-1],  # Exclude leading bit
        )

    def __mul__(self, other: FlexFloat | Number) -> FlexFloat:
        """Multiplies two FlexFloat instances together.

        Args:
            other (FlexFloat | Number): The other FlexFloat instance to multiply.

        Returns:
            FlexFloat: A new FlexFloat instance representing the product.

        Raises:
            TypeError: If other is not a FlexFloat or numeric type.
        """
        if isinstance(other, Number):
            other = FlexFloat.from_float(other)
        if not isinstance(other, FlexFloat):
            raise TypeError("Can only multiply FlexFloat instances.")

        # OBJECTIVE: Multiply two FlexFloat instances together.
        # https://www.rfwireless-world.com/tutorials/ieee-754-floating-point-arithmetic
        #
        # Steps:
        # 0. Handle special cases (NaN, Infinity, zero).
        # 1. Calculate result sign (XOR of operand signs).
        # 2. Extract and add exponents (subtract bias).
        # 3. Multiply mantissas.
        # 4. Normalize mantissa and adjust exponent if necessary.
        # 5. Check for overflow/underflow.
        # 6. Grow exponent if necessary.
        # 7. Return new FlexFloat instance.

        # Step 0: Handle special cases
        if self.is_nan() or other.is_nan():
            return FlexFloat.nan()

        if self.is_zero() or other.is_zero():
            return FlexFloat.zero()

        if self.is_infinity() or other.is_infinity():
            result_sign = self.sign ^ other.sign
            return FlexFloat.infinity(sign=result_sign)

        # Step 1: Calculate result sign (XOR of signs)
        result_sign = self.sign ^ other.sign

        # Step 2: Extract exponent and fraction bits
        # Note: The stored exponent needs +1 to get the actual value (like in addition)
        exponent_self = self.exponent.to_signed_int() + 1
        exponent_other = other.exponent.to_signed_int() + 1

        # Step 3: Add exponents
        # When multiplying, we add the unbiased exponents
        result_exponent = exponent_self + exponent_other

        # Step 4: Multiply mantissas
        # Append the implicit leading 1 to form the mantissa
        mantissa_self = self.fraction + [True]
        mantissa_other = other.fraction + [True]

        # Convert mantissas to integers for multiplication
        mantissa_self_int = mantissa_self.to_int()
        mantissa_other_int = mantissa_other.to_int()

        # Multiply the mantissas
        product = mantissa_self_int * mantissa_other_int

        # Convert back to bit array
        # The product will have up to 106 bits (53 + 53)
        if product == 0:
            return FlexFloat.zero()

        product_bits = self._bitarray_implementation.zeros(106)
        for i in range(106):
            product_bits[i] = product & 1 == 1
            product >>= 1
            if product <= 0:
                break

        # Step 5: Normalize mantissa and adjust exponent if necessary
        # Find the position of the most significant bit
        msb_position = next(
            (i for i, bit in enumerate(reversed(product_bits)) if bit), None
        )

        assert msb_position is not None, "Product should not be zero here."

        # The mantissa multiplication gives us a result with a 2 integer bits
        # We need to normalize to have exactly 1 integer bit
        # If MSB is at position 0, we have a 2-bit integer part (11.xxxxx)
        # If MSB is at position 1, we have a 1-bit integer part (1.xxxxx)
        # Mantissa goes from LSB to MSB, so we need to adjust the exponent accordingly
        if msb_position == 0:
            result_exponent += 1

        # Extract the normalized mantissa
        lsb_position = 53 - msb_position
        normalized_mantissa = product_bits[lsb_position : lsb_position + 53]

        # Pad with zeros if we don't have enough bits
        missing_bits = 53 - len(normalized_mantissa)
        if missing_bits > 0:
            normalized_mantissa = normalized_mantissa.shift(-missing_bits, fill=False)

        # Step 6: Grow exponent if necessary to accommodate the result
        exp_result_length = max(len(self.exponent), len(other.exponent))

        # Check if we need to grow the exponent to accommodate the result
        exp_result_length = self._grow_exponent(result_exponent, exp_result_length)

        exp_result = self._bitarray_implementation.from_signed_int(
            result_exponent - 1, exp_result_length
        )

        return FlexFloat(
            sign=result_sign,
            exponent=exp_result,
            fraction=normalized_mantissa[:-1],  # Exclude leading bit
        )

    def __rmul__(self, other: Number) -> FlexFloat:
        """Right-hand multiplication for Number types.

        Args:
            other (Number): The number to multiply with this FlexFloat.

        Returns:
            FlexFloat: A new FlexFloat instance representing the product.
        """
        return self * FlexFloat.from_float(other)

    def __truediv__(self, other: FlexFloat | Number) -> FlexFloat:
        """Divides this FlexFloat by another FlexFloat or number.

        Args:
            other (FlexFloat | Number): The divisor.

        Returns:
            FlexFloat: A new FlexFloat instance representing the quotient.

        Raises:
            TypeError: If other is not a FlexFloat or numeric type.
        """
        if isinstance(other, Number):
            other = FlexFloat.from_float(other)
        if not isinstance(other, FlexFloat):
            raise TypeError("Can only divide FlexFloat instances.")

        # OBJECTIVE: Divide two FlexFloat instances.
        # https://www.rfwireless-world.com/tutorials/ieee-754-floating-point-arithmetic
        #
        # Steps:
        # 0. Handle special cases (NaN, Infinity, zero).
        # 1. Calculate result sign (XOR of operand signs).
        # 2. Extract and subtract exponents (add bias).
        # 3. Divide mantissas.
        # 4. Normalize mantissa and adjust exponent if necessary.
        # 5. Check for overflow/underflow.
        # 6. Grow exponent if necessary.
        # 7. Return new FlexFloat instance.

        # Step 0: Handle special cases
        if self.is_nan() or other.is_nan():
            return FlexFloat.nan()

        # Zero cases
        if self.is_zero() and other.is_zero():
            return FlexFloat.nan()  # 0 / 0 = NaN
        if self.is_zero() and not other.is_zero():
            return FlexFloat.zero()  # 0 / finite = 0
        if not self.is_zero() and other.is_zero():
            return FlexFloat.infinity(sign=self.sign ^ other.sign)  # finite / 0 = inf

        # Infinity cases
        if self.is_infinity() and other.is_infinity():
            return FlexFloat.nan()  # inf / inf = NaN
        if self.is_infinity():
            return FlexFloat.infinity(sign=self.sign ^ other.sign)  # inf / finite = inf
        if other.is_infinity():
            return FlexFloat.zero()  # finite / inf = 0

        # Step 1: Calculate result sign (XOR of signs)
        result_sign = self.sign ^ other.sign

        # Step 2: Extract exponent and fraction bits
        # Note: The stored exponent needs +1 to get the actual value
        # (like in multiplication)
        exponent_self = self.exponent.to_signed_int() + 1
        exponent_other = other.exponent.to_signed_int() + 1

        # Step 3: Subtract exponents (for division, we subtract the divisor's exponent)
        result_exponent = exponent_self - exponent_other

        # Step 4: Divide mantissas
        # Append the implicit leading 1 to form the mantissa
        mantissa_self = self.fraction + [True]
        mantissa_other = other.fraction + [True]

        # Convert mantissas to integers for division
        mantissa_self_int = mantissa_self.to_int()
        mantissa_other_int = mantissa_other.to_int()

        # Normalize mantissa for division (avoid overflow) -> scale the dividend
        if mantissa_self_int < mantissa_other_int:
            scale_factor = 1 << 53
            result_exponent -= 1  # Adjust exponent since result < 1.0
        else:
            scale_factor = 1 << 52
        scaled_dividend = mantissa_self_int * scale_factor
        quotient = scaled_dividend // mantissa_other_int

        if quotient == 0:
            return FlexFloat.zero()

        # Convert quotient to BitArray for easier bit manipulation
        # Use a fixed size for consistency
        quotient_bitarray = self._bitarray_implementation.zeros(64)
        temp_quotient = quotient
        bit_pos = 0
        while temp_quotient > 0 and bit_pos < 64:
            quotient_bitarray[bit_pos] = (temp_quotient & 1) == 1
            temp_quotient >>= 1
            bit_pos += 1

        # Step 5: Normalize mantissa and adjust exponent if necessary
        # Find the position of the most significant bit (first 1)
        msb_pos = next(
            (i for i, bit in enumerate(reversed(quotient_bitarray)) if bit), None
        )

        if msb_pos is None:
            return FlexFloat.zero()

        # Extract exactly 53 bits starting from the MSB (1 integer + 52 fraction)
        lsb_pos = 11 - msb_pos
        normalized_mantissa = quotient_bitarray[lsb_pos : lsb_pos + 53]

        # If we don't have enough bits, pad with zeros
        missing_bits = 53 - len(normalized_mantissa)
        if missing_bits > 0:
            normalized_mantissa = normalized_mantissa.shift(-missing_bits, fill=False)

        # Step 6: Grow exponent if necessary to accommodate the result
        exp_result_length = max(len(self.exponent), len(other.exponent))

        # Check if we need to grow the exponent to accommodate the result
        exp_result_length = self._grow_exponent(result_exponent, exp_result_length)

        exp_result = self._bitarray_implementation.from_signed_int(
            result_exponent - 1, exp_result_length
        )

        return FlexFloat(
            sign=result_sign,
            exponent=exp_result,
            fraction=normalized_mantissa[:-1],  # Exclude leading bit
        )

    def __rtruediv__(self, other: Number) -> FlexFloat:
        """Right-hand division for Number types.

        Args:
            other (Number): The number to divide by this FlexFloat.

        Returns:
            FlexFloat: A new FlexFloat instance representing the quotient.
        """
        return FlexFloat.from_float(other) / self

    def __abs__(self) -> FlexFloat:
        """Returns the absolute value of the FlexFloat instance.

        Returns:
            FlexFloat: A new FlexFloat instance with the same exponent and fraction, but
                with the sign set to False (positive).
        """
        return FlexFloat(
            sign=False,
            exponent=self.exponent.copy(),
            fraction=self.fraction.copy(),
        )

    def abs(self) -> FlexFloat:
        """Calculates the absolute value of the FlexFloat instance.

        Returns:
            FlexFloat: A new FlexFloat instance with the same exponent and fraction, but
                with the sign set to False (positive).
        """
        return abs(self)

    def __pow__(self, other: FlexFloat | Number) -> FlexFloat:
        """Raises this FlexFloat to the power of another FlexFloat or number.

        Args:
            other (FlexFloat | Number): The exponent.

        Returns:
            FlexFloat: A new FlexFloat instance representing the power.

        Raises:
            TypeError: If other is not a FlexFloat or numeric type.
        """
        if isinstance(other, Number):
            other = FlexFloat.from_float(other)
        if not isinstance(other, FlexFloat):
            raise TypeError("Can only raise FlexFloat instances to a power.")

        # OBJECTIVE: Compute self^other using the identity: a^b = exp(b * ln(a))
        # For most cases, we use logarithmic computation for generality
        # Handle special cases first for performance and correctness
        #
        # Steps:
        # 0. Handle special cases (NaN, infinity, zero, one)
        # 1. Handle negative base with fractional exponent (returns NaN)
        # 2. For general case: compute ln(|base|) * exponent
        # 3. Compute exp(result) to get the final power
        # 4. Handle sign for negative base with integer exponent
        # 5. Return new FlexFloat instance

        # Step 0: Handle special cases
        if self.is_nan() or other.is_nan():
            return FlexFloat.nan()

        # Zero base cases
        if self.is_zero():
            if other.is_zero():
                return FlexFloat.from_float(1.0)  # 0^0 = 1 (by convention)
            if other.sign:  # negative exponent
                return FlexFloat.infinity(sign=False)  # 0^(-n) = +infinity
            return FlexFloat.zero()  # 0^n = 0 for positive n

        # One base: 1^anything = 1
        if (
            not self.sign
            and self.exponent.to_signed_int() == -1
            and not any(self.fraction)
        ):
            # Check if self is exactly 1.0 (exponent = -1, fraction = 0)
            return FlexFloat.from_float(1.0)

        # Zero exponent: anything^0 = 1 (except 0^0 handled above)
        if other.is_zero():
            return FlexFloat.from_float(1.0)

        # One exponent: anything^1 = anything
        if (
            not other.sign
            and other.exponent.to_signed_int() == -1
            and not any(other.fraction)
        ):
            # Check if other is exactly 1.0 (exponent = -1, fraction = 0)
            return self.copy()

        # Infinity cases
        if self.is_infinity():
            if other.is_zero():
                return FlexFloat.from_float(1.0)  # inf^0 = 1
            if other.sign:  # negative exponent
                return FlexFloat.zero()  # inf^(-n) = 0
            # inf^n = inf, but need to handle sign for negative infinity
            if not self.sign:  # positive infinity
                return FlexFloat.infinity(sign=False)  # (+inf)^n = +inf

            # Check if exponent is an integer for sign determination
            try:
                exp_float = other.to_float()
                if exp_float == int(exp_float):  # integer exponent
                    result_sign = int(exp_float) % 2 == 1  # odd = negative
                else:
                    return FlexFloat.nan()  # (-inf)^(non-integer) = NaN
            except (ValueError, OverflowError):
                # For very large exponents, assume positive result
                result_sign = False
            return FlexFloat.infinity(sign=result_sign)

        if other.is_infinity():
            try:
                base_abs = abs(self.to_float())
                if base_abs == 1.0:
                    return FlexFloat.from_float(1.0)  # 1^inf = 1
                if base_abs > 1.0:
                    return (
                        FlexFloat.infinity(sign=False)
                        if not other.sign
                        else FlexFloat.zero()
                    )
                # base_abs < 1.0
                return (
                    FlexFloat.zero()
                    if not other.sign
                    else FlexFloat.infinity(sign=False)
                )
            except (ValueError, OverflowError):
                # For very large/small bases, use log-based approach
                pass

        # Step 1: Handle negative base with fractional exponent
        if self.sign:
            try:
                # Check if exponent is an integer
                exp_float = other.to_float()
                if exp_float != int(exp_float):
                    return FlexFloat.nan()  # (-base)^(non-integer) = NaN
                is_odd_exponent = int(exp_float) % 2 == 1
            except (ValueError, OverflowError):
                # For very large exponents, we can't easily determine if integer
                # Conservative approach: return NaN for negative base
                return FlexFloat.nan()
        else:
            is_odd_exponent = False

        # Step 2-3: General case using a^b = exp(b * ln(a))
        # We need to compute ln(|self|) * other, then exp of that result

        # For the logarithmic approach, we work with the absolute value
        abs_base = self.abs()

        # Use the mathematical identity: a^b = exp(b * ln(a))
        # However, since we don't have ln implemented, we'll use Python's math
        # for the core calculation and then convert back to FlexFloat
        try:
            # Convert to float for the mathematical computation
            base_float = abs_base.to_float()
            exp_float = other.to_float()

            # For very small results that would underflow in float arithmetic,
            # we need to use extended precision
            log_result_estimate = exp_float * math.log(base_float)

            # Check if the result would be too small for standard float
            if log_result_estimate < -700:  # This would underflow to 0 in float
                # Use extended precision approach
                # Estimate the required exponent bits for the very small result
                estimated_exp = int(log_result_estimate / LOG10_2)
                required_exp_bits = max(11, abs(estimated_exp).bit_length() + 2)

                # Create a small result with extended exponent
                small_exp = self._bitarray_implementation.from_signed_int(
                    estimated_exp, required_exp_bits
                )
                # Use a normalized mantissa (leading 1 + some fraction bits)
                result = FlexFloat(
                    sign=False,
                    exponent=small_exp,
                    fraction=self._bitarray_implementation.ones(52),
                )
            else:
                # Compute the power using Python's built-in pow
                result_float = pow(base_float, exp_float)

                # Handle potential overflow/underflow
                if math.isinf(result_float):
                    return FlexFloat.infinity(sign=False)
                if result_float == 0.0:
                    # Even if Python returns 0, we might want extended precision
                    if log_result_estimate < -300:  # Very small but not quite underflow
                        estimated_exp = int(log_result_estimate / LOG10_2)
                        required_exp_bits = max(11, abs(estimated_exp).bit_length() + 2)
                        small_exp = self._bitarray_implementation.from_signed_int(
                            estimated_exp, required_exp_bits
                        )
                        result = FlexFloat(
                            sign=False,
                            exponent=small_exp,
                            fraction=self._bitarray_implementation.ones(52),
                        )
                    else:
                        return FlexFloat.zero()
                if math.isnan(result_float):
                    return FlexFloat.nan()

                # Convert result back to FlexFloat
                result = FlexFloat.from_float(result_float)

        except (ValueError, OverflowError):
            # If float computation overflows, use extended precision approach
            # This is a fallback for when the result is too large for float

            # For very large results, we estimate the exponent growth needed
            try:
                # Estimate log(result) = exponent * log(base)
                log_base = (
                    math.log(abs(self.to_float()))
                    if not self.is_zero()
                    else -float("inf")
                )
                exp_float = other.to_float()
                estimated_log_result = exp_float * log_base

                # Estimate the required exponent bits
                estimated_exp_float = estimated_log_result / LOG10_2
                required_exp_bits = max(
                    11, int(abs(estimated_exp_float)).bit_length() + 2
                )

                # Create a result with extended exponent
                # This is a simplified approach - a full implementation would
                # use arbitrary precision arithmetic
                if estimated_exp_float > 0:
                    # Very large positive result
                    large_exp = self._bitarray_implementation.from_signed_int(
                        int(estimated_exp_float), required_exp_bits
                    )
                    result = FlexFloat(
                        sign=False,
                        exponent=large_exp,
                        fraction=self._bitarray_implementation.ones(52),
                    )
                else:
                    # Very small positive result
                    small_exp = self._bitarray_implementation.from_signed_int(
                        int(estimated_exp_float), required_exp_bits
                    )
                    result = FlexFloat(
                        sign=False,
                        exponent=small_exp,
                        fraction=self._bitarray_implementation.ones(52),
                    )

            except (ValueError, OverflowError, ZeroDivisionError):
                # Ultimate fallback
                return FlexFloat.nan()

        # Step 4: Handle sign for negative base with integer exponent
        if self.sign and is_odd_exponent:
            result.sign = True

        return result

    def __rpow__(self, base: Number) -> FlexFloat:
        """Right-hand power operation for Number types.

        Args:
            base (Number): The base to raise to the power of this FlexFloat.

        Returns:
            FlexFloat: A new FlexFloat instance representing the power.
        """
        return FlexFloat.from_float(base) ** self
