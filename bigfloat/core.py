"""Core BigFloat class implementation."""

from __future__ import annotations

from .types import BitArray, Number
from .utils import (
    bitarray_to_float,
    bitarray_to_int,
    bitarray_to_signed_int,
    float_to_bitarray,
    shift_bitarray,
    signed_int_to_bitarray,
)


class BigFloat:
    """A class to represent a floating-point number with growable exponent and fixed-size fraction.
    This class is designed to handle very large or very small numbers by adjusting the exponent dynamically.
    While keeping the mantissa (fraction) fixed in size.

    This class follows the IEEE 754 double-precision floating-point format,
    but extends it to allow for a growable exponent and a fixed-size fraction.

    Attributes:
        sign (bool): The sign of the number (True for negative, False for positive).
        exponent (BitArray): A growable bit array representing the exponent (uses off-set binary representation).
        fraction (BitArray): A fixed-size bit array representing the fraction (mantissa) of the number.
    """

    def __init__(
        self,
        sign: bool = False,
        exponent: BitArray | None = None,
        fraction: BitArray | None = None,
    ):
        """Initialize a BigFloat instance.

        Args:
            sign (bool): The sign of the number (True for negative, False for positive).
            exponent (BitArray | None): The exponent bit array. If None, defaults to a zero exponent.
            fraction (BitArray | None): The fraction bit array. If None, defaults to a zero fraction.
        """
        self.sign = sign
        self.exponent = exponent if exponent is not None else [False] * 11
        self.fraction = fraction if fraction is not None else [False] * 52

    @classmethod
    def from_float(cls, value: Number) -> BigFloat:
        """Create a BigFloat instance from a number.

        Args:
            value (Number): The number to convert to BigFloat.
        Returns:
            BigFloat: A new BigFloat instance representing the number.
        """
        value = float(value)
        bits = float_to_bitarray(value)

        return cls(sign=bits[0], exponent=bits[1:12], fraction=bits[12:64])

    def to_float(self) -> float:
        """Convert the BigFloat instance back to a floating-point number.

        Returns:
            float: The floating-point number represented by the BigFloat instance.
        Raises:
            ValueError: If the exponent or fraction lengths are not as expected.
        """
        if len(self.exponent) != 11 or len(self.fraction) != 52:
            raise ValueError("Must be a standard 64-bit BigFloat")

        bits = [self.sign] + self.exponent + self.fraction
        return bitarray_to_float(bits)

    def __repr__(self) -> str:
        """Return a string representation of the BigFloat instance.

        Returns:
            str: A string representation of the BigFloat instance.
        """
        return f"BigFloat(sign={self.sign}, exponent={self.exponent}, fraction={self.fraction})"

    def pretty(self) -> str:
        """Return an easier to read string representation of the BigFloat instance.
        Mainly converts the exponent and fraction to integers for readability.

        Returns:
            str: A pretty string representation of the BigFloat instance.
        """
        sign = "-" if self.sign else ""
        exponent_value = bitarray_to_signed_int(self.exponent) + 1
        fraction_value = bitarray_to_int(self.fraction)
        return f"{sign}BigFloat(exponent={exponent_value}, fraction={fraction_value})"

    @classmethod
    def nan(cls) -> BigFloat:
        """Create a BigFloat instance representing NaN (Not a Number).

        Returns:
            BigFloat: A new BigFloat instance representing NaN.
        """
        exponent = [True] * 11  # All exponent bits set to 1
        fraction = [True] * 52  # At least one fraction bit set to 1
        return cls(sign=True, exponent=exponent, fraction=fraction)

    def _is_special_exponent(self) -> bool:
        """Check if the exponent represents a special value (NaN or Infinity).

        Returns:
            bool: True if the exponent is at its maximum value, False otherwise.
        """
        # In IEEE 754, special values have all exponent bits set to 1
        # This corresponds to the maximum value in the unsigned representation
        # For signed offset binary, the maximum value is 2^(n-1) - 1 where n is the number of bits
        max_signed_value = (1 << (len(self.exponent) - 1)) - 1
        return bitarray_to_signed_int(self.exponent) == max_signed_value

    def is_nan(self) -> bool:
        """Check if the BigFloat instance represents NaN (Not a Number).

        Returns:
            bool: True if the BigFloat instance is NaN, False otherwise.
        """
        return self._is_special_exponent() and any(self.fraction)

    def is_infinity(self) -> bool:
        """Check if the BigFloat instance represents Infinity.

        Returns:
            bool: True if the BigFloat instance is Infinity, False otherwise.
        """
        return self._is_special_exponent() and not any(self.fraction)

    def is_zero(self) -> bool:
        """Check if the BigFloat instance represents zero.

        Returns:
            bool: True if the BigFloat instance is zero, False otherwise.
        """
        return not any(self.exponent)

    def copy(self) -> BigFloat:
        """Create a copy of the BigFloat instance.

        Returns:
            BigFloat: A new BigFloat instance with the same sign, exponent, and fraction.
        """
        return BigFloat(
            sign=self.sign, exponent=self.exponent.copy(), fraction=self.fraction.copy()
        )

    def __str__(self) -> str:
        """Float representation of the BigFloat."""
        sign = "-" if self.sign else ""

        exponent_value = bitarray_to_signed_int(self.exponent)
        if exponent_value == 0:
            return f"{sign}0.0"
        max_exponent = 2 ** len(self.exponent) - 1
        # Check NaN or Infinity
        if exponent_value == max_exponent:
            if any(self.fraction):
                return f"{sign}NaN"
            return f"{sign}Infinity"

        fraction_value: float = 1
        for i, bit in enumerate(self.fraction):
            if bit:
                fraction_value += 2 ** -(i + 1)

        if exponent_value == 0:
            return f"{sign}{fraction_value}.0"

        # raise NotImplementedError("String representation for non-zero exponent not implemented yet.")
        return ""

    def __neg__(self) -> BigFloat:
        """Negate the BigFloat instance."""
        return BigFloat(
            sign=not self.sign,
            exponent=self.exponent.copy(),
            fraction=self.fraction.copy(),
        )

    def __add__(self, other: BigFloat) -> BigFloat:
        """Add two BigFloat instances together.

        Args:
            other (BigFloat): The other BigFloat instance to add.
        Returns:
            BigFloat: A new BigFloat instance representing the sum.
        """
        if not isinstance(other, BigFloat):
            raise TypeError("Can only add BigFloat instances.")

        if self.sign != other.sign:
            return self - (-other)

        # OBJECTIVE: Add two BigFloat instances together.
        # Based on: https://www.sciencedirect.com/topics/computer-science/floating-point-addition
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
        # 9. Return new BigFloat instance.

        # Step 0: Handle special cases
        if self.is_zero() or other.is_zero():
            return other.copy() if self.is_zero() else self.copy()

        if self.is_nan() or other.is_nan():
            return self.copy() if self.is_nan() else other.copy()

        if self.is_infinity() and other.is_infinity():
            return self.copy() if self.sign == other.sign else BigFloat.nan()
        if self.is_infinity() or other.is_infinity():
            return self.copy() if self.is_infinity() else other.copy()

        # Step 1: Extract exponent and fraction bits
        exponent_self = bitarray_to_signed_int(self.exponent) + 1
        exponent_other = bitarray_to_signed_int(other.exponent) + 1

        # Step 2: Prepend leading 1 to form the mantissa
        mantissa_self = [True] + self.fraction
        mantissa_other = [True] + other.fraction

        # Step 3: Compare exponents (self is always larger or equal)
        if exponent_self < exponent_other:
            exponent_self, exponent_other = exponent_other, exponent_self
            mantissa_self, mantissa_other = mantissa_other, mantissa_self

        # Step 4: Shift smaller mantissa if necessary
        if exponent_self > exponent_other:
            shift_amount = exponent_self - exponent_other
            mantissa_other = shift_bitarray(mantissa_other, shift_amount)

        # Step 5: Add mantissas
        assert len(mantissa_self) == 53, "Fraction must be 53 bits long. (1 leading bit + 52 fraction bits)"  # fmt: skip
        assert len(mantissa_self) == len(mantissa_other), f"Mantissas must be the same length. Expected 53 bits, got {len(mantissa_other)} bits."  # fmt: skip

        mantissa_result = [False] * 53  # 1 leading bit + 52 fraction bits
        carry = False
        for i in range(52, -1, -1):
            total = mantissa_self[i] + mantissa_other[i] + carry
            mantissa_result[i] = total % 2 == 1
            carry = total > 1

        # Step 6: Normalize mantissa and adjust exponent if necessary
        # Only need to normalize if there is a carry
        if carry:
            # Insert the carry bit and shift right
            mantissa_result = shift_bitarray(mantissa_result, 1, fill=True)
            exponent_self += 1

        # Step 7: Grow exponent if necessary
        exp_result_length = len(self.exponent)
        if exponent_self >= (1 << (len(self.exponent) - 1)) - 1:
            exp_result_length = exp_result_length + 1
            assert (exponent_self - (2**exp_result_length - 1) < 2), "Exponent growth should not exceed 1 bit."  # fmt: skip

        exponent_result = signed_int_to_bitarray(exponent_self - 1, exp_result_length)
        return BigFloat(
            sign=self.sign,
            exponent=exponent_result,
            fraction=mantissa_result[1:],  # Exclude leading bit
        )

    def __sub__(self, other: BigFloat) -> BigFloat:
        """Subtract one BigFloat instance from another.

        Args:
            other (BigFloat): The BigFloat instance to subtract.
        Returns:
            BigFloat: A new BigFloat instance representing the difference.
        """
        if not isinstance(other, BigFloat):
            raise TypeError("Can only subtract BigFloat instances.")

        # If signs are different, subtraction becomes addition
        if self.sign != other.sign:
            return self + (-other)

        # OBJECTIVE: Subtract two BigFloat instances.
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
        # 8. Return new BigFloat instance.

        # Step 0: Handle special cases
        if self.is_zero() and other.is_zero():
            return BigFloat.from_float(0.0)

        if self.is_zero():
            return -other

        if other.is_zero():
            return self.copy()

        if self.is_nan() or other.is_nan():
            return BigFloat.nan()

        if self.is_infinity() and other.is_infinity():
            if self.sign == other.sign:
                return BigFloat.nan()  # inf - inf = NaN
            else:
                return self.copy()  # inf - (-inf) = inf

        if self.is_infinity():
            return self.copy()

        if other.is_infinity():
            return -other

        # Step 1: Extract exponent and fraction bits
        exponent_self = bitarray_to_signed_int(self.exponent) + 1
        exponent_other = bitarray_to_signed_int(other.exponent) + 1

        # Step 2: Prepend leading 1 to form the mantissa
        mantissa_self = [True] + self.fraction
        mantissa_other = [True] + other.fraction

        # Step 3: Align mantissas by shifting the smaller exponent
        shift_amount = abs(exponent_self - exponent_other)
        if exponent_self >= exponent_other:
            mantissa_other = shift_bitarray(mantissa_other, shift_amount)
            result_exponent = exponent_self
        else:
            mantissa_self = shift_bitarray(mantissa_self, shift_amount)
            result_exponent = exponent_other

        # Step 4: Compare magnitudes to determine which mantissa is larger
        # Convert mantissas to integers for comparison
        mantissa_self_int = bitarray_to_int(mantissa_self)
        mantissa_other_int = bitarray_to_int(mantissa_other)

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
        assert len(larger_mantissa) == 53, "Mantissa must be 53 bits long. (1 leading bit + 52 fraction bits)"  # fmt: skip
        assert len(larger_mantissa) == len(smaller_mantissa), f"Mantissas must be the same length. Expected 53 bits, got {len(smaller_mantissa)} bits."  # fmt: skip

        mantissa_result = [False] * 53
        borrow = False

        for i in range(52, -1, -1):
            diff = larger_mantissa[i] - smaller_mantissa[i] - borrow

            mantissa_result[i] = diff % 2 == 1
            borrow = diff < 0

        # Step 6: Normalize mantissa and adjust exponent if necessary
        # Find the first 1 bit (leading bit might have been canceled out)
        leading_zero_count = next(
            (i for i, bit in enumerate(mantissa_result) if bit), len(mantissa_result)
        )

        # Handle case where result becomes zero or denormalized
        if leading_zero_count >= 53:
            return BigFloat.from_float(0.0)

        if leading_zero_count > 0:
            # Shift left to normalize
            mantissa_result = shift_bitarray(mantissa_result, -leading_zero_count)
            result_exponent -= leading_zero_count

        # Step 7: Grow exponent if necessary (handle underflow)
        exp_result_length = len(self.exponent)
        min_exponent = -(1 << (exp_result_length - 1))

        if result_exponent < min_exponent:
            exp_result_length += 1

        exponent_result = signed_int_to_bitarray(result_exponent - 1, exp_result_length)

        return BigFloat(
            sign=result_sign,
            exponent=exponent_result,
            fraction=mantissa_result[1:],  # Exclude leading bit
        )
