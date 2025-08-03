"""Exponential and power functions for FlexFloat."""

from typing import Final

from ..core import FlexFloat

_1: Final[FlexFloat] = FlexFloat.from_float(1.0)
"""The FlexFloat representation of 1.0."""

_2: Final[FlexFloat] = FlexFloat.from_float(2.0)
"""The FlexFloat representation of 2.0."""


def _exp_taylor_series(
    x: FlexFloat,
    max_terms: int = 100,
    tolerance: FlexFloat = FlexFloat.from_float(1e-16),
) -> FlexFloat:
    """Compute the exponential of a FlexFloat using the Taylor series expansion.

    This function evaluates e^x using the Taylor series:
        e^x = 1 + x + x²/2! + x³/3! + ...
    The series converges rapidly for |x| < 1. For best accuracy, use this function
    only for small values of x.

    Args:
        x (FlexFloat): The exponent value (should be small for best convergence).
        max_terms (int, optional): Maximum number of terms to evaluate. Defaults to 100.
        tolerance (FlexFloat, optional): Convergence threshold. Defaults to 1e-16.

    Returns:
        FlexFloat: The computed value of e^x.
    """
    tolerance = tolerance.abs()

    # Initialize result with first term: 1
    result = _1.copy()

    if x.is_zero():
        return result

    # Initialize for the series computation
    term = x.copy()  # First term: x
    result += term

    # For subsequent terms, use the recurrence relation:
    # term[n+1] = term[n] * x / (n+1)
    for n in range(1, max_terms):
        # Calculate next term: x^(n+1) / (n+1)!
        term = term * x / (n + 1)
        result += term

        # Check for convergence
        if term.abs() < tolerance:
            break

    return result


def _exp_range_reduction(x: FlexFloat, max_reductions: int = 50) -> FlexFloat:
    """Compute the exponential of a FlexFloat using range reduction and Taylor series.

    Uses the identity e^x = (e^(x/2^k))^(2^k) to reduce large |x| to a small value,
    computes exp using the Taylor series, and then squares the result k times.

    Args:
        x (FlexFloat): The exponent value.
        max_reductions (int, optional): Maximum number of times to halve x.
            Defaults to 50.

    Returns:
        FlexFloat: The computed value of e^x.
    """
    # TODO: Improve for faster convergence and better handling of large x
    abs_x = x.abs()

    # For small values, use Taylor series directly
    if abs_x <= _1:
        return _exp_taylor_series(x)

    # Determine how many times to halve x to get |x/2^k| <= 1
    reduction_count = 0

    # Keep halving until |reduced_x| <= 1
    while x.abs() > _1 and reduction_count < max_reductions:
        x = x / _2
        reduction_count += 1

    # Compute exp(reduced_x) using Taylor series
    x = _exp_taylor_series(x)

    # Square the result reduction_count times: result = result^(2^reduction_count)
    for _ in range(reduction_count):
        x *= x

    return x


def exp(x: FlexFloat) -> FlexFloat:
    """Compute the exponential function e^x for a FlexFloat value.

    This function handles special cases (NaN, infinity, zero) and uses a combination
    of range reduction and Taylor series for accurate computation.

    Args:
        x (FlexFloat): The exponent value.

    Returns:
        FlexFloat: The value of e^x as a FlexFloat.
    """
    # Handle special cases
    if x.is_nan():
        return FlexFloat.nan()

    if x.is_zero():
        return _1.copy()

    if x.is_infinity():
        if x.sign:
            return FlexFloat.zero()
        return FlexFloat.infinity(sign=False)

    return _exp_range_reduction(x)


def pow(base: FlexFloat, exp: FlexFloat) -> FlexFloat:
    """Raise a FlexFloat base to a FlexFloat exponent.

    Args:
        base (FlexFloat): The base value.
        exp (FlexFloat): The exponent value.

    Returns:
        FlexFloat: The value of base**exp as a FlexFloat.
    """
    return base**exp


def expm1(x: FlexFloat) -> FlexFloat:
    """Return e^x minus 1 for a FlexFloat value.

    Args:
        x (FlexFloat): The exponent value.

    Returns:
        FlexFloat: The value of e^x - 1.
    """
    return exp(x) - _1
