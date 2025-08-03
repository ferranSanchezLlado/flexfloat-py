"""Floating point utility functions for FlexFloat."""

from ..core import FlexFloat


def copysign(x: FlexFloat, y: FlexFloat) -> FlexFloat:
    """Return a FlexFloat with the magnitude of x and the sign of y.

    Args:
        x (FlexFloat): Value whose magnitude is used.
        y (FlexFloat): Value whose sign is used.

    Returns:
        FlexFloat: A FlexFloat with the magnitude of x and the sign of y.
    """
    result = x.copy()
    result.sign = y.sign
    return result


def fabs(x: FlexFloat) -> FlexFloat:
    """Return the absolute value of a FlexFloat.

    Args:
        x (FlexFloat): The value to get the absolute value of.

    Returns:
        FlexFloat: The absolute value of x.
    """
    return abs(x)


def isinf(x: FlexFloat) -> bool:
    """Check if a FlexFloat is positive or negative infinity.

    Args:
        x (FlexFloat): The value to check.

    Returns:
        bool: True if x is infinity, False otherwise.
    """
    return x.is_infinity()


def isnan(x: FlexFloat) -> bool:
    """Check if a FlexFloat is NaN (not a number).

    Args:
        x (FlexFloat): The value to check.

    Returns:
        bool: True if x is NaN, False otherwise.
    """
    return x.is_nan()


def isfinite(x: FlexFloat) -> bool:
    """Check if a FlexFloat is finite (not infinity or NaN).

    Args:
        x (FlexFloat): The value to check.

    Returns:
        bool: True if x is finite, False otherwise.
    """
    return not x.is_infinity() and not x.is_nan()
