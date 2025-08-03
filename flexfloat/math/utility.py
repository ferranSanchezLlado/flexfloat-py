"""Utility functions for FlexFloat math operations."""

from typing import Iterable

from ..core import FlexFloat
from .constants import _1, _2  # type: ignore[attr-defined]


def floor(x: FlexFloat) -> FlexFloat:
    """Return the floor of x as a FlexFloat.

    Args:
        x (FlexFloat): The value to compute the floor of.

    Returns:
        FlexFloat: The largest integer less than or equal to x.
    """
    if x.is_nan() or x.is_infinity():
        return x.copy()

    x_int = int(x)
    recasted_x = FlexFloat.from_int(x_int)
    return FlexFloat.from_int(x_int - (1 if x < recasted_x else 0))


def ceil(x: FlexFloat) -> FlexFloat:
    """Return the ceiling of x as a FlexFloat.

    Args:
        x (FlexFloat): The value to compute the ceiling of.

    Returns:
        FlexFloat: The smallest integer greater than or equal to x.
    """
    if x.is_nan() or x.is_infinity():
        return x.copy()

    x_int = int(x)
    recasted_x = FlexFloat.from_int(x_int)
    return FlexFloat.from_int(x_int + (1 if x > recasted_x else 0))


def fmod(x: FlexFloat, y: FlexFloat) -> FlexFloat:
    """Return the remainder of x divided by y (modulo operation).

    Args:
        x (FlexFloat): The dividend value.
        y (FlexFloat): The divisor value.

    Returns:
        FlexFloat: The remainder of x divided by y.
    """
    if y.is_zero():
        return FlexFloat.nan()  # Handle division by zero
    quotient = floor(x / y)
    return x - (y * quotient)


def copysign(x: FlexFloat, y: FlexFloat) -> FlexFloat:
    """Return a FlexFloat with the magnitude of x and the sign of y.

    Args:
        x (FlexFloat): The value to use the magnitude from.
        y (FlexFloat): The value to use the sign from.

    Returns:
        FlexFloat: A FlexFloat with the magnitude of x and the sign of y.
    """
    result = x.abs()
    if y.sign:
        result = -result
    return result


def fabs(x: FlexFloat) -> FlexFloat:
    """Return the absolute value of x as a FlexFloat.

    Args:
        x (FlexFloat): The value to compute the absolute value of.

    Returns:
        FlexFloat: The absolute value of x.
    """
    return abs(x)


def isinf(x: FlexFloat) -> bool:
    """Check if x is positive or negative infinity.

    Args:
        x (FlexFloat): The value to check.

    Returns:
        bool: True if x is infinity, False otherwise.
    """
    return x.is_infinity()


def isnan(x: FlexFloat) -> bool:
    """Check if x is NaN (not a number).

    Args:
        x (FlexFloat): The value to check.

    Returns:
        bool: True if x is NaN, False otherwise.
    """
    return x.is_nan()


def isfinite(x: FlexFloat) -> bool:
    """Check if x is finite (not NaN or infinity).

    Args:
        x (FlexFloat): The value to check.

    Returns:
        bool: True if x is finite, False otherwise.
    """
    return not (x.is_nan() or x.is_infinity())


def trunc(x: FlexFloat) -> FlexFloat:
    """Return the integer part of x, truncated towards zero.

    Args:
        x (FlexFloat): The value to truncate.

    Returns:
        FlexFloat: The integer part of x.
    """
    return ceil(x) if x.sign else floor(x)


def ulp(x: FlexFloat) -> FlexFloat:
    """Return the unit in the last place (ULP) of x.

    Args:
        x (FlexFloat): The value to compute the ULP of.

    Returns:
        FlexFloat: The ULP of x.
    """
    import math

    if x.is_nan() or x.is_infinity() or x.is_zero():
        return FlexFloat.nan()

    # Use Python's math.ulp function as reference and convert result
    x_float = x.to_float()
    ulp_value = math.ulp(x_float)
    return FlexFloat.from_float(ulp_value)


def fma(x: FlexFloat, y: FlexFloat, z: FlexFloat) -> FlexFloat:
    """Return (x * y) + z with extended precision.

    Args:
        x (FlexFloat): The first multiplicand.
        y (FlexFloat): The second multiplicand.
        z (FlexFloat): The value to add.

    Returns:
        FlexFloat: The result of (x * y) + z.
    """
    return (x * y) + z


def dist(p: Iterable[FlexFloat], q: Iterable[FlexFloat]) -> FlexFloat:
    """Return the Euclidean distance between two points p and q.

    Args:
        p (Iterable[FlexFloat]): The first point coordinates.
        q (Iterable[FlexFloat]): The second point coordinates.

    Returns:
        FlexFloat: The Euclidean distance between p and q.
    """
    from .sqrt import sqrt

    return sqrt(sum(((a - b) ** 2 for a, b in zip(p, q)), FlexFloat.zero()))


def hypot(*coordinates: FlexFloat) -> FlexFloat:
    """Return the Euclidean norm (L2 norm) of the given coordinates.

    Args:
        *coordinates (FlexFloat): The coordinates to compute the norm of.

    Returns:
        FlexFloat: The Euclidean norm of the coordinates.
    """
    from .sqrt import sqrt

    return sqrt(sum((coord**2 for coord in coordinates), FlexFloat.zero()))


def isclose(
    a: FlexFloat,
    b: FlexFloat,
    *,
    rel_tol: FlexFloat = FlexFloat.from_float(1e-09),
    abs_tol: FlexFloat = FlexFloat.from_float(0.0),
) -> bool:
    """Check if two FlexFloat values are close to each other within a tolerance.

    Args:
        a (FlexFloat): The first value to compare.
        b (FlexFloat): The second value to compare.
        rel_tol (FlexFloat, optional): Relative tolerance. Defaults to 1e-09.
        abs_tol: Absolute tolerance. Defaults to 0.0.

    Returns:
        bool: True if a and b are close within the given tolerances, False otherwise.
    """
    if a.is_nan() or b.is_nan():
        return False

    if a.is_infinity() or b.is_infinity():
        return a.is_infinity() and b.is_infinity() and a.sign == b.sign

    diff = (a - b).abs()
    return diff <= abs_tol or diff <= rel_tol * max(a.abs(), b.abs(), _1)


def ldexp(x: FlexFloat, i: int) -> FlexFloat:
    """Return x multiplied by 2 raised to the power i.

    Args:
        x (FlexFloat): The value to scale.
        i (int): The exponent value.

    Returns:
        FlexFloat: The result of x * (2**i).
    """
    return x * (_2**i)


def frexp(x: FlexFloat) -> tuple[FlexFloat, int]:
    """Decompose a FlexFloat into its mantissa and exponent.

    Args:
        x (FlexFloat): The value to decompose.

    Returns:
        tuple[FlexFloat, int]: (mantissa, exponent) such that x = mantissa *
            2**exponent.
    """
    bitarray = FlexFloat._bitarray_implementation  # type: ignore[attr-defined]
    return (
        FlexFloat(
            sign=x.sign,
            fraction=x.fraction,
            exponent=bitarray.from_bits([True] * 11),
        ),
        x.exponent.to_signed_int() + 2,
    )


def fsum(seq: Iterable[FlexFloat]) -> FlexFloat:
    """Accurately sum a sequence of FlexFloat values (sorted by exponent).

    Args:
        seq (Iterable[FlexFloat]): The sequence of values to sum.

    Returns:
        FlexFloat: The sum of the sequence.
    """
    return sum(
        sorted(seq, key=lambda x: -abs(x.exponent.to_signed_int())), FlexFloat.zero()
    )


def modf(x: FlexFloat) -> tuple[FlexFloat, FlexFloat]:
    """Split a FlexFloat into its fractional and integer parts.

    Args:
        x (FlexFloat): The value to split.

    Returns:
        tuple[FlexFloat, FlexFloat]: (fractional part, integer part), with the
            fractional part having the same sign as x.
    """
    int_part = floor(x) if x.to_float() >= 0 else ceil(x)
    frac_part = x - int_part
    return (frac_part, int_part)


def remainder(x: FlexFloat, y: FlexFloat) -> FlexFloat:
    """Return the IEEE 754-style remainder of x with respect to y.

    Args:
        x (FlexFloat): The dividend value.
        y (FlexFloat): The divisor value.

    Returns:
        FlexFloat: The IEEE 754-style remainder.
    """
    if y.is_zero():
        return FlexFloat.nan()
    q = (x / y).to_float()
    n = int(round(q))
    # Round ties to even
    if abs(q - n) == 0.5:
        n = int(2 * round(q / 2.0))
    return x - y * FlexFloat.from_int(n)


def nextafter(x: FlexFloat, y: FlexFloat, *, steps: int | None = None) -> FlexFloat:
    """Return the next representable FlexFloat value after x towards y
    (not implemented).

    Args:
        x (FlexFloat): The starting value.
        y (FlexFloat): The target value.
        steps (int | None, optional): The number of steps to take. Defaults to None.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("nextafter is not implemented for FlexFloat.")


# Unimplemented functions that raise NotImplementedError
def erf(x: FlexFloat) -> FlexFloat:
    """Return the error function of x (not implemented).

    Args:
        x (FlexFloat): The value to compute the error function of.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("erf is not implemented for FlexFloat.")


def erfc(x: FlexFloat) -> FlexFloat:
    """Return the complementary error function of x (not implemented).

    Args:
        x (FlexFloat): The value to compute the complementary error function of.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("erfc is not implemented for FlexFloat.")


def gamma(x: FlexFloat) -> FlexFloat:
    """Return the gamma function of x (not implemented).

    Args:
        x (FlexFloat): The value to compute the gamma function of.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("gamma is not implemented for FlexFloat.")


def lgamma(x: FlexFloat) -> FlexFloat:
    """Return the natural logarithm of the absolute value of the gamma function
    (not implemented).

    Args:
        x (FlexFloat): The value to compute the logarithm of the gamma function of.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("lgamma is not implemented for FlexFloat.")
