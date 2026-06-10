"""Core FlexFloat class implementation.

This module defines the FlexFloat class, which represents a floating-point number with
growable exponent and fraction fields. The class is designed for arbitrary precision
floating-point arithmetic, supporting very large or small values by dynamically
adjusting the exponent size and growing the fraction with it.

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
from enum import Enum
from typing import ClassVar, Final, Type

from .bitarray import BitArray, ListBoolBitArray
from .types import Number

LOG10_2: Final[float] = math.log10(2)
SIGN_LENGTH: Final[int] = 1
BASE_TOTAL_LENGTH: Final[int] = 64
BASE_EXPONENT_LENGTH: Final[int] = 11
BASE_FRACTION_LENGTH: Final[int] = 52
EXPONENT_LENGTH_OFFSET: Final[int] = 7


class ComparisonResult(Enum):
    """Enum representing the result of comparing two FlexFloat values."""

    LESS_THAN = -1
    EQUAL = 0
    GREATER_THAN = 1
    INCOMPARABLE = None  # For NaN comparisons


class FlexFloat:
    """A class to represent a floating-point number with growable exponent and
    fraction fields. This class is designed to handle very large or very small
    numbers by adjusting the exponent dynamically while growing the mantissa
    (fraction) to preserve precision.

    This class follows the IEEE 754 double-precision floating-point format,
    but extends it to allow for growable exponent and fraction fields.

    Attributes:
        sign (bool): The sign of the number (True for negative, False for positive).
        exponent (BitArray): A growable bit array representing the exponent
            (uses off-set binary representation).
        fraction (BitArray): A growable bit array representing the fraction
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
    """A growable bit array representing the fraction (mantissa) of the number."""

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
            else self._bitarray_implementation.zeros(BASE_EXPONENT_LENGTH)
        )
        self.fraction = (
            fraction
            if fraction is not None
            else self._bitarray_implementation.zeros(BASE_FRACTION_LENGTH)
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

        return cls(
            sign=bits[63],
            exponent=bits[
                BASE_FRACTION_LENGTH : BASE_FRACTION_LENGTH + BASE_EXPONENT_LENGTH
            ],
            fraction=bits[:BASE_FRACTION_LENGTH],
        )

    @classmethod
    def from_int(cls, value: int) -> FlexFloat:
        """Creates a FlexFloat instance from an integer without overflow or underflow.

        This method can handle arbitrarily large integers by dynamically growing
        the exponent size as needed. The fraction length can be customized to
        control precision.

        Args:
            value (int): The integer value to convert to FlexFloat.

        Returns:
            FlexFloat: A new FlexFloat instance representing the integer.
        """
        if not isinstance(value, int):  # type: ignore[unreachable]
            value = int(value)
        # Handle zero
        if value == 0:
            return cls.zero()

        # Determine sign and work with absolute value
        sign = value < 0
        abs_value = abs(value)

        # Find the position of the most significant bit (MSB)
        bit_length = abs_value.bit_length()

        # For FlexFloat representation, we need:
        # - The MSB represents the implicit leading 1 of the mantissa
        # - The remaining bits become the fraction
        # - The exponent indicates the position of the binary point

        # The exponent should be the position of the MSB (0-indexed from the right)
        # Since we want the MSB to be the implicit 1, the exponent is bit_length - 1
        actual_exponent = bit_length - 1

        exponent_length, fraction_length = cls._format_lengths_for_exponent(
            actual_exponent
        )

        # Extract the fraction bits (all bits except the MSB)
        fraction_bits: list[bool] = [False] * fraction_length

        if bit_length > 1:
            # Get all bits except the MSB (which becomes the implicit 1)
            fraction_mask = (1 << (bit_length - 1)) - 1
            fraction_value = abs_value & fraction_mask

            # Place the fraction bits in the correct positions
            # For LSB-first storage in FlexFloat, we need to map the fraction correctly
            # The most significant bit of the fraction goes to the highest index
            for i in range(min(bit_length - 1, fraction_length)):
                # Position in the original fraction (MSB to LSB)
                bit_pos = bit_length - 2 - i
                # Target position (high index to low)
                target_index = fraction_length - 1 - i
                fraction_bits[target_index] = (fraction_value >> bit_pos) & 1 == 1

        # No need to pad or truncate since we pre-allocated the correct size

        # Create the fraction BitArray
        fraction = cls._bitarray_implementation.from_bits(fraction_bits)

        # Create the exponent BitArray (stored as actual_exponent - 1)
        exponent = cls._bitarray_implementation.from_signed_int(
            actual_exponent - 1, exponent_length
        )

        return cls(sign=sign, exponent=exponent, fraction=fraction)

    def to_float(self) -> float:
        """Converts the FlexFloat instance back to a 64-bit float.

        If float is bigger than 64 bits, it will truncate the value to fit.

        Returns:
            float: The floating-point number represented by the FlexFloat instance.

        Raises:
            ValueError: If the FlexFloat does not have standard 64-bit exponent and
                fraction.
        """
        if (
            len(self.exponent) < BASE_EXPONENT_LENGTH
            or len(self.fraction) < BASE_FRACTION_LENGTH
        ):
            raise ValueError("Must be a standard 64-bit FlexFloat")

        if self.is_nan():
            return float("nan")
        if self.is_infinity():
            return float("-inf") if self.sign else float("inf")
        if self.is_zero():
            return -0.0 if self.sign else 0.0

        fraction = self._resize_fraction(self.fraction, BASE_FRACTION_LENGTH)
        sign = self._bitarray_implementation.from_bits([self.sign])

        # Preserve standard IEEE subnormal values produced by from_float().
        if len(self.exponent) == BASE_EXPONENT_LENGTH and not any(self.exponent):
            return (fraction + self.exponent + sign).to_float()

        actual_exponent = self.exponent.to_signed_int() + 1
        max_standard_exponent = (1 << (BASE_EXPONENT_LENGTH - 1)) - 1
        min_standard_exponent = -((1 << (BASE_EXPONENT_LENGTH - 1)) - 2)

        if actual_exponent > max_standard_exponent:
            return float("-inf") if self.sign else float("inf")

        if actual_exponent >= min_standard_exponent:
            exponent = self._bitarray_implementation.from_signed_int(
                actual_exponent - 1, BASE_EXPONENT_LENGTH
            )
            return (fraction + exponent + sign).to_float()

        # Convert underflowed normalized FlexFloat values to IEEE subnormals.
        significand = self._significand_int(self)
        shift = actual_exponent - len(self.fraction) + 1074
        if shift >= 0:
            fraction_value = significand << shift
        else:
            fraction_value = significand >> -shift

        if fraction_value <= 0:
            return -0.0 if self.sign else 0.0

        fraction_value = min(fraction_value, (1 << BASE_FRACTION_LENGTH) - 1)
        subnormal_fraction = self._bitarray_implementation.from_bits(
            [(fraction_value >> i) & 1 == 1 for i in range(BASE_FRACTION_LENGTH)]
        )
        subnormal_exponent = self._bitarray_implementation.zeros(BASE_EXPONENT_LENGTH)
        return (subnormal_fraction + subnormal_exponent + sign).to_float()

    def to_int(self) -> int:
        """Converts the FlexFloat instance to an integer.

        The conversion truncates towards zero, similar to Python's int(float) behavior.
        For example: int(3.7) = 3, int(-3.7) = -3.

        Returns:
            int: The integer representation of the FlexFloat instance.

        Raises:
            ArithmeticError: If the FlexFloat represents infinity or NaN.
        """
        # Handle special cases
        if self.is_nan():
            raise ArithmeticError("Cannot convert NaN to integer")

        if self.is_infinity():
            raise ArithmeticError("Cannot convert infinity to integer")

        if self.is_zero():
            return 0

        # Get the biased exponent value (stored as offset binary)
        exponent_biased = self.exponent.to_signed_int()

        # IEEE 754 uses a bias. For our flexible exponent size, the bias is
        # 2^(exp_bits-1) - 1. But the implementation seems to use exponent + 1 as the
        # actual exponent value
        actual_exponent = exponent_biased + 1

        # Get the fraction as an integer
        fraction_int = self.fraction.to_int()

        # For normalized numbers, add the implicit leading 1
        # The mantissa is 1.fraction_bits in binary
        mantissa_int = fraction_int + (1 << len(self.fraction))

        # Calculate the actual value before applying sign
        # The value is mantissa * 2^(exponent - fraction_length)
        shift_amount = actual_exponent - len(self.fraction)

        if shift_amount >= 0:
            # Positive shift: multiply by 2^shift_amount
            abs_value = mantissa_int << shift_amount
        else:
            # Negative shift: divide by 2^(-shift_amount), truncate towards zero
            abs_value = mantissa_int >> (-shift_amount)

        # Apply sign and return
        return -abs_value if self.sign else abs_value

    def __float__(self) -> float:
        """Converts the FlexFloat instance to a float.

        This method is provided for compatibility with Python's float type.
        It uses the to_float method to perform the conversion.

        Returns:
            float: The floating-point representation of the FlexFloat instance.
        """
        return self.to_float()

    def __int__(self) -> int:
        """Converts the FlexFloat instance to an integer.

        This method is provided for compatibility with Python's int type.
        It uses the to_int method to perform the conversion.

        Returns:
            int: The integer representation of the FlexFloat instance.
        """
        return self.to_int()

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
        exponent = cls._bitarray_implementation.ones(BASE_EXPONENT_LENGTH)
        fraction = cls._bitarray_implementation.ones(BASE_FRACTION_LENGTH)
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
        exponent = cls._bitarray_implementation.ones(BASE_EXPONENT_LENGTH)
        fraction = cls._bitarray_implementation.zeros(BASE_FRACTION_LENGTH)
        return cls(sign=sign, exponent=exponent, fraction=fraction)

    @classmethod
    def zero(cls, sign: bool = False) -> FlexFloat:
        """Creates a FlexFloat instance representing zero.

        Args:
            sign (bool, optional): Indicates if the zero is negative. Defaults to False.

        Returns:
            FlexFloat: A new FlexFloat instance representing zero.
        """
        exponent = cls._bitarray_implementation.zeros(BASE_EXPONENT_LENGTH)
        fraction = cls._bitarray_implementation.zeros(BASE_FRACTION_LENGTH)
        return cls(sign=sign, exponent=exponent, fraction=fraction)

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

    @staticmethod
    def _fraction_length_for_exponent_length(exponent_length: int) -> int:
        """Return the fraction length implied by an exponent field size.

        The layout keeps the IEEE-754 double baseline at 64 bits and grows total
        storage so ``exponent ~= 3 * log2(total_bits) - 7``. The fraction gets the
        remaining bits after sign and exponent.
        """
        if exponent_length <= BASE_EXPONENT_LENGTH:
            return BASE_FRACTION_LENGTH

        try:
            total_length = math.ceil(
                2 ** ((exponent_length + EXPONENT_LENGTH_OFFSET) / 3)
            )
        except OverflowError:
            total_length = 1 << math.ceil(
                (exponent_length + EXPONENT_LENGTH_OFFSET) / 3
            )

        total_length = max(BASE_TOTAL_LENGTH, total_length)
        return max(
            BASE_FRACTION_LENGTH,
            total_length - exponent_length - SIGN_LENGTH,
        )

    @classmethod
    def _format_lengths_for_exponent(
        cls,
        exponent: int,
        min_exponent_length: int = BASE_EXPONENT_LENGTH,
        min_fraction_length: int = BASE_FRACTION_LENGTH,
    ) -> tuple[int, int]:
        """Return exponent and fraction lengths for a finite exponent value."""
        exponent_length = cls._grow_exponent(
            exponent, max(BASE_EXPONENT_LENGTH, min_exponent_length)
        )
        fraction_length = max(
            min_fraction_length,
            cls._fraction_length_for_exponent_length(exponent_length),
        )
        return exponent_length, fraction_length

    @classmethod
    def _resize_fraction(cls, fraction: BitArray, length: int) -> BitArray:
        """Resize fractional bits while preserving the represented value."""
        current_length = len(fraction)
        bits = list(fraction)
        if length == current_length:
            return fraction.copy()
        if length > current_length:
            return cls._bitarray_implementation.from_bits(
                [False] * (length - current_length) + bits
            )
        return cls._bitarray_implementation.from_bits(bits[current_length - length :])

    @staticmethod
    def _significand_int(value: FlexFloat) -> int:
        """Return the integer significand including the implicit leading bit."""
        return value.fraction.to_int() + (1 << len(value.fraction))

    @staticmethod
    def _resize_significand(
        significand: int,
        current_fraction_length: int,
        target_fraction_length: int,
    ) -> int:
        """Resize an integer significand to a new fraction precision."""
        shift = target_fraction_length - current_fraction_length
        if shift > 0:
            return significand << shift
        if shift < 0:
            return significand >> -shift
        return significand

    @classmethod
    def _fraction_from_significand(
        cls, significand: int, fraction_length: int
    ) -> BitArray:
        """Create fraction bits from a normalized integer significand."""
        implicit_bit = 1 << fraction_length
        assert (
            implicit_bit <= significand < (implicit_bit << 1)
        ), "Significand must be normalized before extracting fraction bits."
        fraction_value = significand - implicit_bit
        return cls._bitarray_implementation.from_bits(
            [(fraction_value >> i) & 1 == 1 for i in range(fraction_length)]
        )

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
        if not isinstance(other, FlexFloat):  # type: ignore[unreachable]
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

        # Step 1: Extract exponent and significand bits
        exponent_self = self.exponent.to_signed_int() + 1
        exponent_other = other.exponent.to_signed_int() + 1

        fraction_length = max(len(self.fraction), len(other.fraction))
        mantissa_self = self._resize_significand(
            self._significand_int(self), len(self.fraction), fraction_length
        )
        mantissa_other = self._resize_significand(
            self._significand_int(other), len(other.fraction), fraction_length
        )

        # Step 3: Compare exponents (self is always larger or equal)
        if exponent_self < exponent_other:
            exponent_self, exponent_other = exponent_other, exponent_self
            mantissa_self, mantissa_other = mantissa_other, mantissa_self

        # Step 4: Shift smaller mantissa if necessary
        if exponent_self > exponent_other:
            shift_amount = exponent_self - exponent_other
            mantissa_other >>= shift_amount

        # Step 5: Add mantissas
        mantissa_result = mantissa_self + mantissa_other

        # Step 6: Normalize mantissa and adjust exponent if necessary
        implicit_bit = 1 << fraction_length
        if mantissa_result >= (implicit_bit << 1):
            mantissa_result >>= 1
            exponent_self += 1

        # Step 7: Grow exponent if necessary
        exp_result_length, fraction_result_length = self._format_lengths_for_exponent(
            exponent_self,
            max(len(self.exponent), len(other.exponent)),
            fraction_length,
        )
        mantissa_result = self._resize_significand(
            mantissa_result, fraction_length, fraction_result_length
        )

        exponent_result = self._bitarray_implementation.from_signed_int(
            exponent_self - 1, exp_result_length
        )
        return FlexFloat(
            sign=self.sign,
            exponent=exponent_result,
            fraction=self._fraction_from_significand(
                mantissa_result, fraction_result_length
            ),
        )

    def __radd__(self, other: FlexFloat | Number) -> FlexFloat:
        """Right addition operator for FlexFloat instances.

        This method allows the FlexFloat instance to be added to another FlexFloat
        or numeric type on the right side of the addition operator.

        Args:
            other (FlexFloat | Number): The other FlexFloat instance or numeric type
                to add.

        Returns:
            FlexFloat: A new FlexFloat instance representing the sum.
        """
        return self + other

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
        if not isinstance(other, FlexFloat):  # type: ignore[unreachable]
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

        # Step 1: Extract exponent and significand bits
        exponent_self = self.exponent.to_signed_int() + 1
        exponent_other = other.exponent.to_signed_int() + 1

        fraction_length = max(len(self.fraction), len(other.fraction))
        mantissa_self = self._resize_significand(
            self._significand_int(self), len(self.fraction), fraction_length
        )
        mantissa_other = self._resize_significand(
            self._significand_int(other), len(other.fraction), fraction_length
        )

        # Step 3: Align mantissas by shifting the smaller exponent
        shift_amount = abs(exponent_self - exponent_other)
        if exponent_self >= exponent_other:
            mantissa_other >>= shift_amount
            result_exponent = exponent_self
        else:
            mantissa_self >>= shift_amount
            result_exponent = exponent_other

        # Step 4: Compare magnitudes to determine which mantissa is larger
        if mantissa_self >= mantissa_other:
            mantissa_result = mantissa_self - mantissa_other
            result_sign = self.sign
        else:
            mantissa_result = mantissa_other - mantissa_self
            # Flip sign since we're computing -(smaller - larger)
            result_sign = not self.sign

        # Step 6: Normalize mantissa and adjust exponent if necessary
        # Handle case where result becomes zero or denormalized
        if mantissa_result == 0:
            return FlexFloat.from_float(0.0)

        implicit_bit = 1 << fraction_length
        while mantissa_result < implicit_bit:
            mantissa_result <<= 1
            result_exponent -= 1

        # Step 7: Grow exponent if necessary (handle underflow)
        exp_result_length, fraction_result_length = self._format_lengths_for_exponent(
            result_exponent,
            max(len(self.exponent), len(other.exponent)),
            fraction_length,
        )
        mantissa_result = self._resize_significand(
            mantissa_result, fraction_length, fraction_result_length
        )

        exp_result = self._bitarray_implementation.from_signed_int(
            result_exponent - 1, exp_result_length
        )

        return FlexFloat(
            sign=result_sign,
            exponent=exp_result,
            fraction=self._fraction_from_significand(
                mantissa_result, fraction_result_length
            ),
        )

    def __rsub__(self, other: FlexFloat | Number) -> FlexFloat:
        """Right-hand subtraction for FlexFloat instances.

        This method allows the FlexFloat instance to be subtracted from another
        FlexFloat or numeric type on the right side of the subtraction operator.

        Args:
            other (FlexFloat | Number): The other FlexFloat instance or numeric type
                to subtract from.

        Returns:
            FlexFloat: A new FlexFloat instance representing the difference.
        """
        return -self + other

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
        if not isinstance(other, FlexFloat):  # type: ignore[unreachable]
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
        mantissa_self_int = self._significand_int(self)
        mantissa_other_int = self._significand_int(other)
        fraction_length_self = len(self.fraction)
        fraction_length_other = len(other.fraction)

        # Multiply the mantissas
        product = mantissa_self_int * mantissa_other_int
        if product == 0:
            return FlexFloat.zero()

        # Step 5: Normalize mantissa and adjust exponent if necessary
        # Find the position of the most significant bit
        msb_position = product.bit_length() - 1

        result_exponent += msb_position - (fraction_length_self + fraction_length_other)

        # Step 6: Grow exponent if necessary to accommodate the result
        exp_result_length, fraction_result_length = self._format_lengths_for_exponent(
            result_exponent,
            max(len(self.exponent), len(other.exponent)),
            max(fraction_length_self, fraction_length_other),
        )

        significand_shift = msb_position - fraction_result_length
        if significand_shift >= 0:
            normalized_mantissa = product >> significand_shift
        else:
            normalized_mantissa = product << -significand_shift

        exp_result = self._bitarray_implementation.from_signed_int(
            result_exponent - 1, exp_result_length
        )

        return FlexFloat(
            sign=result_sign,
            exponent=exp_result,
            fraction=self._fraction_from_significand(
                normalized_mantissa, fraction_result_length
            ),
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
        if not isinstance(other, FlexFloat):  # type: ignore[unreachable]
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
        mantissa_self_int = self._significand_int(self)
        mantissa_other_int = self._significand_int(other)
        fraction_length_self = len(self.fraction)
        fraction_length_other = len(other.fraction)

        ratio_less_than_one = (mantissa_self_int << fraction_length_other) < (
            mantissa_other_int << fraction_length_self
        )
        if ratio_less_than_one:
            result_exponent -= 1  # Adjust exponent since result < 1.0

        # Step 6: Grow exponent if necessary to accommodate the result
        exp_result_length, fraction_result_length = self._format_lengths_for_exponent(
            result_exponent,
            max(len(self.exponent), len(other.exponent)),
            max(fraction_length_self, fraction_length_other),
        )

        scale_shift = (
            fraction_length_other
            + fraction_result_length
            + int(ratio_less_than_one)
            - fraction_length_self
        )
        if scale_shift >= 0:
            numerator = mantissa_self_int << scale_shift
            denominator = mantissa_other_int
        else:
            numerator = mantissa_self_int
            denominator = mantissa_other_int << -scale_shift

        normalized_mantissa = numerator // denominator

        if normalized_mantissa == 0:
            return FlexFloat.zero()

        implicit_bit = 1 << fraction_result_length
        while normalized_mantissa < implicit_bit:
            normalized_mantissa <<= 1
            result_exponent -= 1
            (
                exp_result_length,
                fraction_result_length,
            ) = self._format_lengths_for_exponent(
                result_exponent, exp_result_length, fraction_result_length
            )
            implicit_bit = 1 << fraction_result_length

        if normalized_mantissa >= (implicit_bit << 1):
            normalized_mantissa >>= 1
            result_exponent += 1
            (
                exp_result_length,
                fraction_result_length,
            ) = self._format_lengths_for_exponent(
                result_exponent, exp_result_length, fraction_result_length
            )

        exp_result = self._bitarray_implementation.from_signed_int(
            result_exponent - 1, exp_result_length
        )

        return FlexFloat(
            sign=result_sign,
            exponent=exp_result,
            fraction=self._fraction_from_significand(
                normalized_mantissa, fraction_result_length
            ),
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
        if not isinstance(other, FlexFloat):  # type: ignore[unreachable]
            raise TypeError("Can only raise FlexFloat instances to a power.")

        # Handle special cases for power operation
        # (Python float semantics, no overflow/underflow)
        # TODO: Handle integer powers more efficiently

        # 1. If exponent is 0.0 or -0.0: result is 1.0
        if other.is_zero():
            return ONE

        # 2. If base is nan
        if self.is_nan():
            # nan ** 0 = 1, nan ** x = nan
            if other.is_zero():
                return ONE
            return FlexFloat.nan()

        # 3. If exponent is nan
        if other.is_nan():
            # x ** nan = 1 if x is 1 or -1, else nan
            # if self == ONE or self == MINUS_ONE:
            if self in (ONE, MINUS_ONE):
                return ONE
            return FlexFloat.nan()

        # 4. If base is 1.0 or -1.0
        if self == ONE:
            return ONE
        if self == MINUS_ONE:
            # -1.0 ** integer = ±1, -1.0 ** non-integer = nan
            # TODO: Check if exponent is integer
            return FlexFloat.nan()

        # 5. If base is 0.0 or -0.0
        if self.is_zero():
            if other > ZERO:
                # 0 ** positive = 0 (preserve sign)
                return FlexFloat.zero(sign=self.sign)
            if other < ZERO:
                # 0 ** negative = inf (preserve sign)
                return FlexFloat.infinity(sign=self.sign)
            # 0 ** 0 already handled above
            return FlexFloat.nan()

        # 6. If base is inf or -inf
        if self.is_infinity():
            if other.is_zero():
                return ONE
            if other > ZERO:
                return FlexFloat.infinity(sign=self.sign)
            return FlexFloat.zero(sign=self.sign)

        # TODO: 7. If base < 0 and exponent is not integer: nan
        # if self.sign and self.fraction.to_int() != 0:

        from . import math as ffmath  # Import to avoid circular import issues

        # Otherwise, use exp(other * log(self))
        return ffmath.exp(other * ffmath.log(self))

    def __rpow__(self, other: Number) -> FlexFloat:
        """Right-hand power operation for Number types.

        Args:
            other (Number): The base to raise to the power of this FlexFloat.

        Returns:
            FlexFloat: A new FlexFloat instance representing the power.
        """
        return FlexFloat.from_float(other) ** self

    def _compare(self, other: FlexFloat) -> ComparisonResult:
        """Compare this FlexFloat with another FlexFloat.

        This method handles comparison between FlexFloats with potentially different
        exponent sizes. It returns a ComparisonResult indicating the comparison result.

        Args:
            other (FlexFloat): The FlexFloat to compare with.

        Returns:
            ComparisonResult: LESS_THAN if self < other, EQUAL if self == other,
                            GREATER_THAN if self > other, INCOMPARABLE for NaN
                            comparisons.
        """
        # Handle NaN cases - NaN is not equal to anything, including itself
        if self.is_nan() or other.is_nan():
            return ComparisonResult.INCOMPARABLE

        # Handle zero cases
        if self.is_zero() and other.is_zero():
            return ComparisonResult.EQUAL
        if self.is_zero():
            return (
                ComparisonResult.LESS_THAN
                if not other.sign
                else ComparisonResult.GREATER_THAN
            )
        if other.is_zero():
            return (
                ComparisonResult.GREATER_THAN
                if not self.sign
                else ComparisonResult.LESS_THAN
            )

        # Handle infinity cases
        if self.is_infinity() and other.is_infinity():
            if self.sign == other.sign:
                return ComparisonResult.EQUAL
            return (
                ComparisonResult.LESS_THAN
                if self.sign
                else ComparisonResult.GREATER_THAN
            )
        if self.is_infinity():
            return (
                ComparisonResult.LESS_THAN
                if self.sign
                else ComparisonResult.GREATER_THAN
            )
        if other.is_infinity():
            return (
                ComparisonResult.GREATER_THAN
                if other.sign
                else ComparisonResult.LESS_THAN
            )

        # Handle sign differences for finite numbers
        if self.sign != other.sign:
            return (
                ComparisonResult.LESS_THAN
                if self.sign
                else ComparisonResult.GREATER_THAN
            )

        # Both numbers have the same sign and are finite
        # Compare exponents first
        exponent_self = self.exponent.to_signed_int()
        exponent_other = other.exponent.to_signed_int()

        # If exponents are different, the comparison is determined by exponent
        if exponent_self != exponent_other:
            result = (
                ComparisonResult.GREATER_THAN
                if exponent_self > exponent_other
                else ComparisonResult.LESS_THAN
            )
            # If both numbers are negative, reverse the result
            if self.sign:
                return (
                    ComparisonResult.LESS_THAN
                    if result == ComparisonResult.GREATER_THAN
                    else ComparisonResult.GREATER_THAN
                )
            return result

        # Exponents are equal, compare fractions
        # Convert fractions to integers for comparison
        fraction_self = self.fraction.to_int()
        fraction_other = other.fraction.to_int()

        # Pad the shorter fraction with zeros on the right (LSB side)
        len_self = len(self.fraction)
        len_other = len(other.fraction)

        if len_self < len_other:
            # Pad self's fraction
            fraction_self <<= len_other - len_self
        elif len_other < len_self:
            # Pad other's fraction
            fraction_other <<= len_self - len_other

        if fraction_self == fraction_other:
            return ComparisonResult.EQUAL

        result = (
            ComparisonResult.GREATER_THAN
            if fraction_self > fraction_other
            else ComparisonResult.LESS_THAN
        )
        # If both numbers are negative, reverse the result
        if self.sign:
            return (
                ComparisonResult.LESS_THAN
                if result == ComparisonResult.GREATER_THAN
                else ComparisonResult.GREATER_THAN
            )
        return result

    def __eq__(self, other: object) -> bool:
        """Check if this FlexFloat is equal to another value.

        Args:
            other (object): The value to compare with.

        Returns:
            bool: True if the values are equal, False otherwise.
        """
        if not isinstance(other, (FlexFloat, int, float)):
            return False
        if not isinstance(other, FlexFloat):
            other = FlexFloat.from_float(other)

        result = self._compare(other)
        # Handle NaN case - NaN is never equal to anything
        if result == ComparisonResult.INCOMPARABLE:
            return False
        return result == ComparisonResult.EQUAL

    def __ne__(self, other: object) -> bool:
        """Check if this FlexFloat is not equal to another value.

        Args:
            other (object): The value to compare with.

        Returns:
            bool: True if the values are not equal, False otherwise.
        """
        if not isinstance(other, (FlexFloat, int, float)):
            return True
        if not isinstance(other, FlexFloat):
            other = FlexFloat.from_float(other)

        result = self._compare(other)
        # Handle NaN case - NaN is never equal to anything, so != is always True
        if result == ComparisonResult.INCOMPARABLE:
            return True
        return result != ComparisonResult.EQUAL

    def __lt__(self, other: FlexFloat | Number) -> bool:
        """Check if this FlexFloat is less than another value.

        Args:
            other (FlexFloat | Number): The value to compare with.

        Returns:
            bool: True if this FlexFloat is less than other, False otherwise.
        """
        if not isinstance(other, FlexFloat):
            other = FlexFloat.from_float(other)

        result = self._compare(other)
        # Handle NaN case - any comparison with NaN is False
        if result == ComparisonResult.INCOMPARABLE:
            return False
        return result == ComparisonResult.LESS_THAN

    def __le__(self, other: FlexFloat | Number) -> bool:
        """Check if this FlexFloat is less than or equal to another value.

        Args:
            other (FlexFloat | Number): The value to compare with.

        Returns:
            bool: True if this FlexFloat is less than or equal to other, False
                otherwise.
        """
        if not isinstance(other, FlexFloat):
            other = FlexFloat.from_float(other)

        result = self._compare(other)
        # Handle NaN case - any comparison with NaN is False
        if result == ComparisonResult.INCOMPARABLE:
            return False
        return result in (ComparisonResult.LESS_THAN, ComparisonResult.EQUAL)

    def __gt__(self, other: FlexFloat | Number) -> bool:
        """Check if this FlexFloat is greater than another value.

        Args:
            other (FlexFloat | Number): The value to compare with.

        Returns:
            bool: True if this FlexFloat is greater than other, False otherwise.
        """
        if not isinstance(other, FlexFloat):
            other = FlexFloat.from_float(other)

        result = self._compare(other)
        # Handle NaN case - any comparison with NaN is False
        if result == ComparisonResult.INCOMPARABLE:
            return False
        return result == ComparisonResult.GREATER_THAN

    def __ge__(self, other: FlexFloat | Number) -> bool:
        """Check if this FlexFloat is greater than or equal to another value.

        Args:
            other (FlexFloat | Number): The value to compare with.

        Returns:
            bool: True if this FlexFloat is greater than or equal to other, False
                otherwise.
        """
        if not isinstance(other, FlexFloat):
            other = FlexFloat.from_float(other)

        result = self._compare(other)
        # Handle NaN case - any comparison with NaN is False
        if result == ComparisonResult.INCOMPARABLE:
            return False
        return result in (ComparisonResult.EQUAL, ComparisonResult.GREATER_THAN)


ZERO: Final[FlexFloat] = FlexFloat.from_float(0.0)
"""Constant representing the FlexFloat value of 0.0."""
ONE: Final[FlexFloat] = FlexFloat.from_float(1.0)
"""Constant representing the FlexFloat value of 1.0."""
MINUS_ONE: Final[FlexFloat] = FlexFloat.from_float(-1.0)
"""Constant representing the FlexFloat value of -1.0."""
