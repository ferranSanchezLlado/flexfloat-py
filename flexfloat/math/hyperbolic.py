"""Hyperbolic functions for FlexFloat."""

from ..core import FlexFloat
from .constants import _0_5, _1, _2  # type: ignore[attr-defined]
from .exponential import exp
from .logarithmic import log
from .sqrt import sqrt


def sinh(x: FlexFloat) -> FlexFloat:
    """Return the hyperbolic sine of x.

    This function computes sinh(x) = (e^x - e^(-x)) / 2, handling special cases
    appropriately.

    Args:
        x (FlexFloat): The value to compute the hyperbolic sine of.

    Returns:
        FlexFloat: The hyperbolic sine of x.

    Examples:
        >>> sinh(FlexFloat.from_float(0.0))  # Returns 0.0
        >>> sinh(FlexFloat.from_float(1.0))  # Returns ~1.175
    """
    # Handle special cases
    if x.is_nan():
        return FlexFloat.nan()

    if x.is_infinity():
        return x.copy()  # sinh(±∞) = ±∞

    if x.is_zero():
        return FlexFloat.zero()

    # For very small x, use Taylor series: sinh(x) ≈ x for |x| << 1
    if x.abs() < FlexFloat.from_float(1e-10):
        return x.copy()

    # For moderate values, use the definition: sinh(x) = (e^x - e^(-x)) / 2
    exp_x = exp(x)
    exp_neg_x = exp(-x)

    return (exp_x - exp_neg_x) / _2


def cosh(x: FlexFloat) -> FlexFloat:
    """Return the hyperbolic cosine of x.

    This function computes cosh(x) = (e^x + e^(-x)) / 2, handling special cases
    appropriately.

    Args:
        x (FlexFloat): The value to compute the hyperbolic cosine of.

    Returns:
        FlexFloat: The hyperbolic cosine of x.

    Examples:
        >>> cosh(FlexFloat.from_float(0.0))  # Returns 1.0
        >>> cosh(FlexFloat.from_float(1.0))  # Returns ~1.543
    """
    # Handle special cases
    if x.is_nan():
        return FlexFloat.nan()

    if x.is_infinity():
        return FlexFloat.infinity(sign=False)  # cosh(±∞) = +∞

    if x.is_zero():
        return _1.copy()

    # Use the definition: cosh(x) = (e^x + e^(-x)) / 2
    exp_x = exp(x)
    exp_neg_x = exp(-x)

    return (exp_x + exp_neg_x) / _2


def tanh(x: FlexFloat) -> FlexFloat:
    """Return the hyperbolic tangent of x.

    This function computes tanh(x) = sinh(x) / cosh(x) = (e^x - e^(-x)) / (e^x + e^(-x)),
    handling special cases appropriately.

    Args:
        x (FlexFloat): The value to compute the hyperbolic tangent of.

    Returns:
        FlexFloat: The hyperbolic tangent of x.

    Examples:
        >>> tanh(FlexFloat.from_float(0.0))  # Returns 0.0
        >>> tanh(FlexFloat.from_float(1.0))  # Returns ~0.762
    """
    # Handle special cases
    if x.is_nan():
        return FlexFloat.nan()

    if x.is_infinity():
        # tanh(+∞) = 1, tanh(-∞) = -1
        return _1.copy() if not x.sign else -_1

    if x.is_zero():
        return FlexFloat.zero()

    # For very large |x|, tanh(x) approaches ±1
    if x.abs() > FlexFloat.from_float(20.0):
        return _1.copy() if not x.sign else -_1

    # For very small x, use Taylor series: tanh(x) ≈ x for |x| << 1
    if x.abs() < FlexFloat.from_float(1e-10):
        return x.copy()

    # Use the definition: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    exp_x = exp(x)
    exp_neg_x = exp(-x)

    numerator = exp_x - exp_neg_x
    denominator = exp_x + exp_neg_x

    return numerator / denominator


def asinh(x: FlexFloat) -> FlexFloat:
    """Return the hyperbolic arc sine of x.

    Uses the identity asinh(x) = ln(x + sqrt(x² + 1)).

    Args:
        x (FlexFloat): The value to compute the hyperbolic arc sine of.

    Returns:
        FlexFloat: The hyperbolic arc sine of x.

    Examples:
        >>> asinh(FlexFloat.from_float(0.0))  # Returns 0.0
        >>> asinh(FlexFloat.from_float(1.0))  # Returns ~0.881
    """
    # Handle special cases
    if x.is_nan():
        return FlexFloat.nan()

    if x.is_infinity():
        return x.copy()  # asinh(±∞) = ±∞

    if x.is_zero():
        return FlexFloat.zero()

    # For very small x, use Taylor series: asinh(x) ≈ x for |x| << 1
    if x.abs() < FlexFloat.from_float(1e-10):
        return x.copy()

    # Use the identity: asinh(x) = ln(x + sqrt(x² + 1))
    x_squared = x * x
    sqrt_term = sqrt(x_squared + _1)
    return log(x + sqrt_term)


def acosh(x: FlexFloat) -> FlexFloat:
    """Return the hyperbolic arc cosine of x.

    Uses the identity acosh(x) = ln(x + sqrt(x² - 1)) for x >= 1.

    Args:
        x (FlexFloat): The value to compute the hyperbolic arc cosine of, must be >= 1.

    Returns:
        FlexFloat: The hyperbolic arc cosine of x.

    Examples:
        >>> acosh(FlexFloat.from_float(1.0))  # Returns 0.0
        >>> acosh(FlexFloat.from_float(2.0))  # Returns ~1.317
    """
    # Handle special cases
    if x.is_nan():
        return FlexFloat.nan()

    if x.is_infinity():
        if not x.sign:  # positive infinity
            return FlexFloat.infinity(sign=False)
        else:  # negative infinity
            return FlexFloat.nan()

    # Check domain x >= 1
    if x < _1:
        return FlexFloat.nan()

    if x == _1:
        return FlexFloat.zero()

    # Use the identity: acosh(x) = ln(x + sqrt(x² - 1))
    x_squared = x * x
    sqrt_term = sqrt(x_squared - _1)
    return log(x + sqrt_term)


def atanh(x: FlexFloat) -> FlexFloat:
    """Return the hyperbolic arc tangent of x.

    Uses the identity atanh(x) = (1/2) * ln((1+x)/(1-x)) for |x| < 1.

    Args:
        x (FlexFloat): The value to compute the hyperbolic arc tangent of, must be in (-1, 1).

    Returns:
        FlexFloat: The hyperbolic arc tangent of x.

    Examples:
        >>> atanh(FlexFloat.from_float(0.0))  # Returns 0.0
        >>> atanh(FlexFloat.from_float(0.5))  # Returns ~0.549
    """
    # Handle special cases
    if x.is_nan():
        return FlexFloat.nan()

    if x.is_infinity():
        return FlexFloat.nan()

    if x.is_zero():
        return FlexFloat.zero()

    # Check domain (-1, 1)
    if x.abs() >= _1:
        return FlexFloat.nan()

    # For very small x, use Taylor series: atanh(x) ≈ x for |x| << 1
    if x.abs() < FlexFloat.from_float(1e-10):
        return x.copy()

    # Use the identity: atanh(x) = (1/2) * ln((1+x)/(1-x))
    numerator = _1 + x
    denominator = _1 - x

    # Check for division issues
    if denominator.is_zero():
        return FlexFloat.nan()

    ratio = numerator / denominator
    return _0_5 * log(ratio)
