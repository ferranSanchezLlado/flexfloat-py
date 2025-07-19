"""Core FlexFloat class implementation."""

from __future__ import annotations

from .bitarray import BitArray
from .types import Number


class FlexFloat:
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
        """Initialize a FlexFloat instance.

        Args:
            sign (bool): The sign of the number (True for negative, False for positive).
            exponent (BitArray | None): The exponent bit array. If None, defaults to a zero exponent.
            fraction (BitArray | None): The fraction bit array. If None, defaults to a zero fraction.
        """
        self.sign = sign
        self.exponent = exponent if exponent is not None else BitArray.zeros(11)
        self.fraction = fraction if fraction is not None else BitArray.zeros(52)

    @classmethod
    def from_float(cls, value: Number) -> FlexFloat:
        """Create a FlexFloat instance from a number.

        Args:
            value (Number): The number to convert to FlexFloat.
        Returns:
            FlexFloat: A new FlexFloat instance representing the number.
        """
        value = float(value)
        bits = BitArray.from_float(value)

        return cls(sign=bits[0], exponent=bits[1:12], fraction=bits[12:64])

    def to_float(self) -> float:
        """Convert the FlexFloat instance back to a 64-bit float.

        If float is bigger than 64 bits, it will truncate the value to fit into a 64-bit float.

        Returns:
            float: The floating-point number represented by the FlexFloat instance.
        Raises:
            ValueError: If the exponent or fraction lengths are not as expected.
        """
        if len(self.exponent) < 11 or len(self.fraction) < 52:
            raise ValueError("Must be a standard 64-bit FlexFloat")

        bits = BitArray([self.sign]) + self.exponent[:11] + self.fraction[:52]
        return bits.to_float()

    def __repr__(self) -> str:
        """Return a string representation of the FlexFloat instance.

        Returns:
            str: A string representation of the FlexFloat instance.
        """
        return f"FlexFloat(sign={self.sign}, exponent={self.exponent}, fraction={self.fraction})"

    def pretty(self) -> str:
        """Return an easier to read string representation of the FlexFloat instance.
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
        """Create a FlexFloat instance representing NaN (Not a Number).

        Returns:
            FlexFloat: A new FlexFloat instance representing NaN.
        """
        exponent = BitArray.ones(11)
        fraction = BitArray.ones(52)
        return cls(sign=True, exponent=exponent, fraction=fraction)

    @classmethod
    def infinity(cls, sign: bool = False) -> FlexFloat:
        """Create a FlexFloat instance representing Infinity.

        Args:
            sign (bool): The sign of the infinity (True for negative, False for positive).
        Returns:
            FlexFloat: A new FlexFloat instance representing Infinity.
        """
        exponent = BitArray.ones(11)
        fraction = BitArray.zeros(52)
        return cls(sign=sign, exponent=exponent, fraction=fraction)

    @classmethod
    def zero(cls) -> FlexFloat:
        """Create a FlexFloat instance representing zero.

        Returns:
            FlexFloat: A new FlexFloat instance representing zero.
        """
        exponent = BitArray.zeros(11)
        fraction = BitArray.zeros(52)
        return cls(sign=False, exponent=exponent, fraction=fraction)

    def _is_special_exponent(self) -> bool:
        """Check if the exponent represents a special value (NaN or Infinity).

        Returns:
            bool: True if the exponent is at its maximum value, False otherwise.
        """
        # In IEEE 754, special values have all exponent bits set to 1
        # This corresponds to the maximum value in the unsigned representation
        # For signed offset binary, the maximum value is 2^(n-1) - 1 where n is the number of bits
        max_signed_value = (1 << (len(self.exponent) - 1)) - 1
        return self.exponent.to_signed_int() == max_signed_value

    def is_nan(self) -> bool:
        """Check if the FlexFloat instance represents NaN (Not a Number).

        Returns:
            bool: True if the FlexFloat instance is NaN, False otherwise.
        """
        return self._is_special_exponent() and any(self.fraction)

    def is_infinity(self) -> bool:
        """Check if the FlexFloat instance represents Infinity.

        Returns:
            bool: True if the FlexFloat instance is Infinity, False otherwise.
        """
        return self._is_special_exponent() and not any(self.fraction)

    def is_zero(self) -> bool:
        """Check if the FlexFloat instance represents zero.

        Returns:
            bool: True if the FlexFloat instance is zero, False otherwise.
        """
        return not any(self.exponent) and not any(self.fraction)

    def copy(self) -> FlexFloat:
        """Create a copy of the FlexFloat instance.

        Returns:
            FlexFloat: A new FlexFloat instance with the same sign, exponent, and fraction.
        """
        return FlexFloat(
            sign=self.sign, exponent=self.exponent.copy(), fraction=self.fraction.copy()
        )

    def __str__(self) -> str:
        """Float representation of the FlexFloat."""
        sign = "-" if self.sign else ""

        exponent_value = self.exponent.to_signed_int()
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

    def __neg__(self) -> FlexFloat:
        """Negate the FlexFloat instance."""
        return FlexFloat(
            sign=not self.sign,
            exponent=self.exponent.copy(),
            fraction=self.fraction.copy(),
        )

    @staticmethod
    def _grow_exponent(exponent: int, exponent_length: int) -> int:
        """Grow the exponent if it exceeds the maximum value for the current length.

        Args:
            exponent (int): The current exponent value.
            exponent_length (int): The current length of the exponent in bits.
        Returns:
            int: The new exponent length if it needs to be grown, otherwise the same length.
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
        """Add two FlexFloat instances together.

        Args:
            other (FlexFloat | float | int): The other FlexFloat instance to add.
        Returns:
            FlexFloat: A new FlexFloat instance representing the sum.
        """
        if isinstance(other, Number):
            other = FlexFloat.from_float(other)
        if not isinstance(other, FlexFloat):
            raise TypeError("Can only add FlexFloat instances.")

        if self.sign != other.sign:
            return self - (-other)

        # OBJECTIVE: Add two FlexFloat instances together.
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
            mantissa_other = mantissa_other.shift(shift_amount)

        # Step 5: Add mantissas
        assert len(mantissa_self) == 53, "Fraction must be 53 bits long. (1 leading bit + 52 fraction bits)"  # fmt: skip
        assert len(mantissa_self) == len(mantissa_other), f"Mantissas must be the same length. Expected 53 bits, got {len(mantissa_other)} bits."  # fmt: skip

        mantissa_result = BitArray.zeros(53)  # 1 leading bit + 52 fraction bits
        carry = False
        for i in range(52, -1, -1):
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

        exponent_result = BitArray.from_signed_int(exponent_self - 1, exp_result_length)
        return FlexFloat(
            sign=self.sign,
            exponent=exponent_result,
            fraction=mantissa_result[1:],  # Exclude leading bit
        )

    def __sub__(self, other: FlexFloat | Number) -> FlexFloat:
        """Subtract one FlexFloat instance from another.

        Args:
            other (FlexFloat | float | int): The FlexFloat instance to subtract.
        Returns:
            FlexFloat: A new FlexFloat instance representing the difference.
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

        # Step 2: Prepend leading 1 to form the mantissa
        mantissa_self = [True] + self.fraction
        mantissa_other = [True] + other.fraction

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
        assert len(larger_mantissa) == 53, "Mantissa must be 53 bits long. (1 leading bit + 52 fraction bits)"  # fmt: skip
        assert len(larger_mantissa) == len(smaller_mantissa), f"Mantissas must be the same length. Expected 53 bits, got {len(smaller_mantissa)} bits."  # fmt: skip

        mantissa_result = BitArray.zeros(53)
        borrow = False

        for i in range(52, -1, -1):
            diff = int(larger_mantissa[i]) - int(smaller_mantissa[i]) - int(borrow)

            mantissa_result[i] = diff % 2 == 1
            borrow = diff < 0

        assert not borrow, "Subtraction should not result in a negative mantissa."

        # Step 6: Normalize mantissa and adjust exponent if necessary
        # Find the first 1 bit (leading bit might have been canceled out)
        leading_zero_count = next(
            (i for i, bit in enumerate(mantissa_result) if bit), len(mantissa_result)
        )

        # Handle case where result becomes zero or denormalized
        if leading_zero_count >= 53:
            return FlexFloat.from_float(0.0)

        if leading_zero_count > 0:
            # Shift left to normalize
            mantissa_result = mantissa_result.shift(-leading_zero_count)
            result_exponent -= leading_zero_count

        # Step 7: Grow exponent if necessary (handle underflow)
        # exp_result_length = len(self.exponent)

        # # Keep growing exponent until it can accommodate the result
        # while True:
        #     min_exponent = -(1 << (exp_result_length - 1))
        #     max_exponent = (1 << (exp_result_length - 1)) - 1

        #     if min_exponent <= result_exponent - 1 <= max_exponent:
        #         break
        #     exp_result_length += 1
        exp_result_length = self._grow_exponent(result_exponent, len(self.exponent))

        exp_result = BitArray.from_signed_int(result_exponent - 1, exp_result_length)

        return FlexFloat(
            sign=result_sign,
            exponent=exp_result,
            fraction=mantissa_result[1:],  # Exclude leading bit
        )

    def __mul__(self, other: FlexFloat | Number) -> FlexFloat:
        """Multiply two FlexFloat instances together.

        Args:
            other (FlexFloat | float | int): The other FlexFloat instance to multiply.
        Returns:
            FlexFloat: A new FlexFloat instance representing the product.
        """
        if isinstance(other, Number):
            other = FlexFloat.from_float(other)
        if not isinstance(other, FlexFloat):
            raise TypeError("Can only multiply FlexFloat instances.")

        # OBJECTIVE: Multiply two FlexFloat instances together.
        # Based on floating-point multiplication algorithms
        #
        # Steps:
        # 0. Handle special cases (NaN, Infinity, zero).
        # 1. Calculate result sign.
        # 2. Extract exponent and fraction bits.
        # 3. Add exponents.
        # 4. Multiply mantissas.
        # 5. Normalize mantissa and adjust exponent if necessary.
        # 6. Grow exponent if necessary.
        # 7. Return new FlexFloat instance.

        # Step 0: Handle special cases
        if self.is_nan() or other.is_nan():
            return FlexFloat.nan()

        if self.is_zero() or other.is_zero():
            return FlexFloat.zero()

        if self.is_infinity() or other.is_infinity():
            result_sign = self.sign != other.sign
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
        # Prepend leading 1 to form the mantissa (1.fraction)
        mantissa_self = [True] + self.fraction
        mantissa_other = [True] + other.fraction

        # Convert mantissas to integers for multiplication
        mantissa_self_int = mantissa_self.to_int()
        mantissa_other_int = mantissa_other.to_int()

        # Multiply the mantissas
        product = mantissa_self_int * mantissa_other_int

        # Convert back to bit array
        # The product will have up to 106 bits (53 + 53)
        if product == 0:
            return FlexFloat.zero()

        product_bits = []
        temp_product = product
        bit_count = 0
        while temp_product > 0 and bit_count < 106:
            product_bits.append(temp_product & 1 == 1)
            temp_product >>= 1
            bit_count += 1

        # Pad with zeros if needed
        while len(product_bits) < 106:
            product_bits.append(False)

        # Reverse to get most significant bit first
        product_bits.reverse()

        # Step 5: Normalize mantissa and adjust exponent if necessary
        # Find the position of the most significant bit
        msb_position = next((i for i, bit in enumerate(product_bits) if bit), None)

        assert msb_position is not None, "Product should not be zero here."

        # The mantissa multiplication gives us a result with 2 integer bits
        # We need to normalize to have exactly 1 integer bit
        # If MSB is at position 0, we have a 2-bit integer part (11.xxxxx)
        # If MSB is at position 1, we have a 1-bit integer part (1.xxxxx)

        if msb_position == 0:
            # We have 11.xxxxx, need to shift right by 1 and increment exponent
            normalized_mantissa = product_bits[0:53]  # Take bits 0-52 (53 bits total)
            result_exponent += 1
        else:
            # We have 1.xxxxx, use as is
            normalized_mantissa = product_bits[msb_position : msb_position + 53]

        # Pad with zeros if we don't have enough bits
        missing_bits = 53 - len(normalized_mantissa)
        if missing_bits > 0:
            normalized_mantissa += [False] * missing_bits

        mantissa_result = BitArray(normalized_mantissa)

        # Step 6: Grow exponent if necessary to accommodate the result
        exp_result_length = max(len(self.exponent), len(other.exponent))

        # Check if we need to grow the exponent to accommodate the result
        exp_result_length = self._grow_exponent(result_exponent, exp_result_length)

        exp_result = BitArray.from_signed_int(result_exponent - 1, exp_result_length)

        return FlexFloat(
            sign=result_sign,
            exponent=exp_result,
            fraction=mantissa_result[1:],  # Exclude leading bit
        )

    def __rmul__(self, other: Number) -> FlexFloat:
        """Right-hand multiplication for Number types.

        Args:
            other (float | int): The number to multiply with this FlexFloat.
        Returns:
            FlexFloat: A new FlexFloat instance representing the product.
        """
        return self * other
