"""Math module for FlexFloat, similar to Python's math module.

This module provides mathematical functions that operate on FlexFloat instances,
mirroring the interface of Python's built-in math module where possible.
Most functions are designed to work with FlexFloat objects, enabling arbitrary-precision
floating-point arithmetic.

Example:
    from flexfloat.math import sqrt, exp, pow
    from flexfloat import FlexFloat
    a = FlexFloat.from_float(2.0)
    b = sqrt(a)
    print(b)
    # Output: 1.41421e+00
"""

import math
from typing import Callable, Final, Iterable, TypeAlias

from .core import FlexFloat
from .types import Number

# Constants
e: Final[FlexFloat] = FlexFloat.from_float(math.e)
"""The mathematical constant e (Euler's number) as a FlexFloat."""
pi: Final[FlexFloat] = FlexFloat.from_float(math.pi)
"""The mathematical constant pi as a FlexFloat."""
inf: Final[FlexFloat] = FlexFloat.infinity()
"""Positive infinity as a FlexFloat."""
nan: Final[FlexFloat] = FlexFloat.nan()
"""Not-a-Number (NaN) as a FlexFloat."""
tau: Final[FlexFloat] = FlexFloat.from_float(math.tau)
"""The mathematical constant tau (2*pi) as a FlexFloat."""

_0_5: Final[FlexFloat] = FlexFloat.from_float(0.5)
"""The FlexFloat representation of 0.5."""
_1: Final[FlexFloat] = FlexFloat.from_float(1.0)
"""The FlexFloat representation of 1.0."""
_2: Final[FlexFloat] = FlexFloat.from_float(2.0)
"""The FlexFloat representation of 2.0."""
_10: Final[FlexFloat] = FlexFloat.from_float(10.0)
"""The FlexFloat representation of 10.0."""

_SCALE_FACTOR_SQRT: Final[FlexFloat] = FlexFloat.from_float(1024.0)
"""Factor used for scaling or reducing values in square root calculations."""
_SCALE_FACTOR_SQRT_RESULT: Final[FlexFloat] = FlexFloat.from_float(32.0)
"""Reducing factor for square root calculations to avoid precision issues."""
_ArithmeticOperation: TypeAlias = Callable[[FlexFloat, FlexFloat | Number], FlexFloat]
"""Type alias for arithmetic operations on FlexFloat instances.
This is used to define operations like addition, subtraction, multiplication, and 
division between FlexFloat and Number types."""


def _exp_taylor_series(
    x: FlexFloat,
    max_terms: int = 100,
    tolerance: FlexFloat = FlexFloat.from_float(1e-16),
) -> FlexFloat:
    """Compute exponential using Taylor series for small values of x.

    Uses the series: e^x = 1 + x + x²/2! + x³/3! + x⁴/4! + ...
    This converges rapidly for |x| < 1.

    Args:
        x (FlexFloat): Input value (should be small for best convergence).
        max_terms (int): Maximum number of terms in the series.
        tolerance (FlexFloat): Convergence tolerance.

    Returns:
        FlexFloat: The exponential of x.
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


def _exp_range_reduction(x: FlexFloat) -> FlexFloat:
    """Compute exponential using range reduction to improve convergence.

    Uses the identity: e^x = (e^(x/2^k))^(2^k)
    Reduces x to a small value, computes exp using Taylor series,
    then squares the result k times.

    Args:
        x (FlexFloat): The input value.

    Returns:
        FlexFloat: The exponential of x.
    """
    abs_x = x.abs()

    # For small values, use Taylor series directly
    if abs_x <= _1:
        return _exp_taylor_series(x)

    # Determine how many times to halve x to get |x/2^k| <= 1
    reduction_count = 0

    # Keep halving until |reduced_x| <= 1
    max_reductions = 50  # Safety limit
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
    """Return e raised to the power of x using Taylor series and range reduction.

    This implementation uses a Taylor series approach with range reduction
    for accurate computation without relying on the power operator.

    The algorithm:
    1. Handle special cases (NaN, infinity, zero)
    2. For large |x|, use range reduction to bring x into convergent range
    3. Apply Taylor series: e^x = 1 + x + x²/2! + x³/3! + ...

    Args:
        x (FlexFloat): The exponent value.

    Returns:
        FlexFloat: The value of e^x as a FlexFloat.

    Examples:
        >>> exp(FlexFloat.from_float(0.0))  # Returns 1.0
        >>> exp(FlexFloat.from_float(1.0))  # Returns e ≈ 2.718281828
        >>> exp(FlexFloat.from_float(-1.0)) # Returns 1/e ≈ 0.367879441
    """
    # Handle special cases
    if x.is_nan():
        return FlexFloat.nan()

    if x.is_zero():
        return _1.copy()

    if x.is_infinity():
        if x.sign:  # negative infinity
            return FlexFloat.zero()
        else:  # positive infinity
            return FlexFloat.infinity(sign=False)

    return _exp_range_reduction(x)


def pow(base: FlexFloat, exp: FlexFloat) -> FlexFloat:
    """Return base raised to the power of exp (both are FlexFloat instances).

    Args:
        base (FlexFloat): The base value.
        exp (FlexFloat): The exponent value.

    Returns:
        FlexFloat: The value of base**exp as a FlexFloat.
    """
    return base**exp


def copysign(x: FlexFloat, y: FlexFloat) -> FlexFloat:
    """Return a FlexFloat with the magnitude of x and the sign of y.

    Args:
        x (FlexFloat): The value whose magnitude is used.
        y (FlexFloat): The value whose sign is used.

    Returns:
        FlexFloat: A FlexFloat with the magnitude of x and the sign of y.
    """
    result = x.copy()
    result.sign = y.sign
    return result


def fabs(x: FlexFloat) -> FlexFloat:
    """Return the absolute value of x.

    Args:
        x (FlexFloat): The value to get the absolute value of.

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
    """Check if x is neither an infinity nor NaN.

    Args:
        x (FlexFloat): The value to check.

    Returns:
        bool: True if x is finite, False otherwise.
    """
    return not x.is_infinity() and not x.is_nan()


def _sqrt_taylor_core(
    x: FlexFloat,
    max_terms: int = 100,
    tolerance: FlexFloat = FlexFloat.from_float(1e-16),
) -> FlexFloat:
    """Compute square root using Taylor series for x in [0.5, 2].

    Uses the series: √(1+u) = 1 + u/2 - u²/8 + u³/16 - 5u⁴/128 + ...
    where u = x - 1.

    For better accuracy, we use the exact coefficients computed using the binomial theorem:
    √(1+u) = Σ(n=0 to ∞) C(1/2, n) * u^n
    where C(1/2, n) = (1/2) * (-1/2) * (-3/2) * ... * ((1/2) - n + 1) / n!

    Args:
        x (FlexFloat): Input value in range [0.5, 2].
        max_terms (int): Maximum number of terms to compute.
        tolerance (FlexFloat): Convergence tolerance.

    Returns:
        FlexFloat: The square root of x.
    """
    # Transform to √(1+u) form where u = x - 1
    u = x - _1
    tolerance = tolerance.abs()

    # Initialize result with first term: 1
    result = _1.copy()

    if u.is_zero():
        return result

    # Initialize for the series computation
    term = u / _2  # First term: u/2
    result += term

    # For subsequent terms, use the recurrence relation:
    # coefficient[n+1] = coefficient[n] * (1/2 - n) / (n + 1)
    coefficient = _0_5  # coefficient for u^1 term
    u_power = u.copy()  # u^1

    for n in range(1, max_terms):
        # Update coefficient: coeff[n+1] = coeff[n] * (1/2 - n) / (n + 1)
        coefficient = coefficient * (_0_5 - n) / (n + 1)

        # Update u power
        u_power = u_power * u  # u^(n+1)

        # Calculate new term
        term = coefficient * u_power
        result += term

        # Check for convergence
        if term.abs() < tolerance:
            break

    return result


def _sqrt_newton_raphson_core(
    x: FlexFloat,
    max_iterations: int = 100,
    tolerance: FlexFloat = FlexFloat.from_float(1e-16),
) -> FlexFloat:
    """Core Newton-Raphson algorithm for computing square roots.

    Args:
        x (FlexFloat): The input value (must be positive and not extremely large).
        max_iterations (int): Maximum number of iterations.
        tolerance (FlexFloat): Convergence tolerance.

    Returns:
        FlexFloat: The square root of x.
    """
    # Better initial guess strategy
    if x >= _1:
        # For x >= 1, use x/2 as initial guess, but ensure it's reasonable
        guess = x / _2
        # If x is very large, use a better approximation
        if x > FlexFloat.from_float(1000.0):
            # Use bit manipulation approach for better initial guess
            # For now, use a simple heuristic
            guess = x / FlexFloat.from_float(10.0)
    else:
        # For 0 < x < 1, start with 1 (since sqrt(x) is between x and 1)
        guess = _1.copy()

    tolerance = tolerance.abs()

    for _ in range(max_iterations):
        # Newton-Raphson iteration: new_guess = (guess + x/guess) / 2
        x_over_guess = x / guess
        new_guess = (guess + x_over_guess) / _2

        # Check for convergence using relative error
        diff = (new_guess - guess).abs()
        relative_error = diff / new_guess.abs() if not new_guess.is_zero() else diff

        if relative_error < tolerance:
            return new_guess

        # Update guess for next iteration
        guess = new_guess

    return guess


def _scale_sqrt(
    x: FlexFloat,
    scale_up: bool,
    lower_bound: FlexFloat = FlexFloat.from_float(1e-20),
    upper_bound: FlexFloat = FlexFloat.from_float(1e20),
) -> FlexFloat:
    operation: _ArithmeticOperation = (
        FlexFloat.__truediv__ if scale_up else FlexFloat.__mul__
    )
    inverse_operation: _ArithmeticOperation = (
        FlexFloat.__mul__ if scale_up else FlexFloat.__truediv__
    )

    scale_count = 0

    while (x < lower_bound or x > upper_bound) and scale_count < 100:
        x = operation(x, _SCALE_FACTOR_SQRT)
        scale_count += 1

    scaled_result = _sqrt_newton_raphson_core(x)

    for _ in range(scale_count):
        scaled_result = inverse_operation(scaled_result, _SCALE_FACTOR_SQRT_RESULT)

    return scaled_result


def sqrt(x: FlexFloat) -> FlexFloat:
    """Return the square root of x using hybrid optimization.

    This implementation uses a hybrid approach that selects the optimal method
    based on the input value:
    - For values very close to 1: Taylor series (extremely fast and accurate)
    - For other values: Newton-Raphson (generally faster with good accuracy)
    - For very small or large values: scaling to avoid precision issues

    Args:
        x (FlexFloat): The value to compute the square root of.

    Returns:
        FlexFloat: The square root of x.

    Raises:
        ValueError: If x is negative (square root of negative numbers
                   is not supported for real numbers).
    """
    # Handle special cases
    if x.is_nan():
        return FlexFloat.nan()

    if x.is_zero():
        return FlexFloat.zero()

    if x.sign:  # x < 0
        return FlexFloat.nan()  # Square root of negative number

    if x.is_infinity():
        return FlexFloat.infinity(sign=False)

    # Hybrid approach: use Taylor series for values close to 1, Newton-Raphson otherwise
    # Taylor series is much faster and equally accurate for values near 1
    if abs(x - _1) < FlexFloat.from_float(0.2):
        return _sqrt_taylor_core(x)

    # For extremely small values, use scaling to avoid precision issues
    if x < FlexFloat.from_float(1e-30):
        return _scale_sqrt(x, scale_up=False)

    # For extremely large values, use scaling to avoid numerical issues
    if x > FlexFloat.from_float(1e40):
        return _scale_sqrt(x, scale_up=True)

    # For normal values, use the core algorithm
    return _sqrt_newton_raphson_core(x)


# Unimplemented functions
def acos(x: FlexFloat) -> FlexFloat:
    """Return the arc cosine of x (not implemented).

    Args:
        x (FlexFloat): The value to compute the arc cosine of.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("acos is not implemented for FlexFloat.")


def acosh(x: FlexFloat) -> FlexFloat:
    """Return the hyperbolic arc cosine of x (not implemented).

    Args:
        x (FlexFloat): The value to compute the hyperbolic arc cosine of.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("acosh is not implemented for FlexFloat.")


def asin(x: FlexFloat) -> FlexFloat:
    """Return the arc sine of x (not implemented).

    Args:
        x (FlexFloat): The value to compute the arc sine of.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("asin is not implemented for FlexFloat.")


def asinh(x: FlexFloat) -> FlexFloat:
    """Return the hyperbolic arc sine of x (not implemented).

    Args:
        x (FlexFloat): The value to compute the hyperbolic arc sine of.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("asinh is not implemented for FlexFloat.")


def atan(x: FlexFloat) -> FlexFloat:
    """Return the arc tangent of x (not implemented).

    Args:
        x (FlexFloat): The value to compute the arc tangent of.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("atan is not implemented for FlexFloat.")


def atan2(y: FlexFloat, x: FlexFloat) -> FlexFloat:
    """Return the arc tangent of y/x (not implemented).

    Args:
        y (FlexFloat): The numerator value.
        x (FlexFloat): The denominator value.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("atan2 is not implemented for FlexFloat.")


def atanh(x: FlexFloat) -> FlexFloat:
    """Return the hyperbolic arc tangent of x (not implemented).

    Args:
        x (FlexFloat): The value to compute the hyperbolic arc tangent of.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("atanh is not implemented for FlexFloat.")


def cbrt(x: FlexFloat) -> FlexFloat:
    """Return the cube root of x (not implemented).

    Args:
        x (FlexFloat): The value to compute the cube root of.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("cbrt is not implemented for FlexFloat.")


def ceil(x: FlexFloat) -> FlexFloat:
    """Return the ceiling of x (not implemented).

    Args:
        x (FlexFloat): The value to compute the ceiling of.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("ceil is not implemented for FlexFloat.")


# Not implemented functions
def dist(p: Iterable[FlexFloat], q: Iterable[FlexFloat]) -> FlexFloat:
    """Return the Euclidean distance between two points p and q (not implemented).

    Args:
        p (Iterable[FlexFloat]): The first point coordinates.
        q (Iterable[FlexFloat]): The second point coordinates.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("dist is not implemented for FlexFloat.")


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


def expm1(x: FlexFloat) -> FlexFloat:
    """Return e raised to the power of x, minus 1 (not implemented).

    Args:
        x (FlexFloat): The exponent value.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("expm1 is not implemented for FlexFloat.")


def factorial(x: FlexFloat) -> FlexFloat:
    """Return the factorial of x (not implemented).

    Args:
        x (FlexFloat): The value to compute the factorial of.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("factorial is not implemented for FlexFloat.")


def fmod(x: FlexFloat, y: FlexFloat) -> FlexFloat:
    """Return the remainder of x divided by y (not implemented).

    Args:
        x (FlexFloat): The dividend value.
        y (FlexFloat): The divisor value.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("fmod is not implemented for FlexFloat.")


def frexp(x: FlexFloat) -> tuple[FlexFloat, int]:
    """Return the mantissa and exponent of x (not implemented).

    Args:
        x (FlexFloat): The value to decompose into mantissa and exponent.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("frexp is not implemented for FlexFloat.")


def fsum(seq: Iterable[FlexFloat]) -> FlexFloat:
    """Return an accurate floating-point sum of the values in seq (not implemented).

    Args:
        seq (Iterable[FlexFloat]): The sequence of values to sum.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("fsum is not implemented for FlexFloat.")


def gamma(x: FlexFloat) -> FlexFloat:
    """Return the gamma function of x (not implemented).

    Args:
        x (FlexFloat): The value to compute the gamma function of.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("gamma is not implemented for FlexFloat.")


def gcd(*integers: FlexFloat) -> FlexFloat:
    """Return the greatest common divisor of the integers (not implemented).

    Args:
        *integers (FlexFloat): The integers to compute the gcd of.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("gcd is not implemented for FlexFloat.")


def hypot(*coordinates: FlexFloat) -> FlexFloat:
    """Return the Euclidean norm, sqrt(x1*x1 + x2*x2 + ... + xn*xn)
    (not implemented).

    Args:
        *coordinates (FlexFloat): The coordinates to compute the norm of.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("hypot is not implemented for FlexFloat.")


def isclose(
    a: FlexFloat,
    b: FlexFloat,
    *,
    rel_tol: FlexFloat = FlexFloat.from_float(1e-09),
    abs_tol: FlexFloat = FlexFloat.from_float(0.0),
) -> bool:
    """Check if two FlexFloat instances are close in value (not implemented).

    Args:
        a (FlexFloat): The first value to compare.
        b (FlexFloat): The second value to compare.
        rel_tol (FlexFloat, optional): The relative tolerance. Defaults to 1e-09.
        abs_tol (FlexFloat, optional): The absolute tolerance. Defaults to 0.0.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("isclose is not implemented for FlexFloat.")


def lcm(*integers: FlexFloat) -> FlexFloat:
    """Return the least common multiple of the integers (not implemented).

    Args:
        *integers (FlexFloat): The integers to compute the lcm of.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("lcm is not implemented for FlexFloat.")


def ldexp(x: FlexFloat, i: int) -> FlexFloat:
    """Return x * (2**i) (not implemented).

    Args:
        x (FlexFloat): The value to scale.
        i (int): The exponent value.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("ldexp is not implemented for FlexFloat.")


def lgamma(x: FlexFloat) -> FlexFloat:
    """Return the natural logarithm of the absolute value of the gamma function of x
    (not implemented).

    Args:
        x (FlexFloat): The value to compute the natural logarithm of the gamma
          function of.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("lgamma is not implemented for FlexFloat.")


def _ln_taylor_series(
    x: FlexFloat,
    max_iterations: int = 100,
    tolerance: FlexFloat = FlexFloat.from_float(1e-16),
) -> FlexFloat:
    """Compute the natural logarithm of x using a fast-converging Taylor series.

    Uses the identity:
        ln(x) = 2 * artanh((x-1)/(x+1))
    where artanh(y) = y + y³/3 + y⁵/5 + ... for |y| < 1.
    This converges rapidly for x near 1 (i.e., 0 < x ≤ 2).

    Args:
        x (FlexFloat): The input value (should be close to 1 for best convergence).
        max_iterations (int): Maximum number of terms in the series.
        tolerance (FlexFloat): Convergence tolerance.

    Returns:
        FlexFloat: The natural logarithm of x.
    """
    x_minus_1 = x - _1
    x_plus_1 = x + _1

    # Check for division by zero
    if x_plus_1.is_zero():
        return FlexFloat.nan()

    y = x_minus_1 / x_plus_1
    tolerance = tolerance.abs()

    # Initialize series: artanh(y) = y + y³/3 + y⁵/5 + ...
    result = y.copy()
    y_squared = y * y
    term = y.copy()

    for n in range(1, max_iterations):
        # Calculate next term: y^(2n+1) / (2n+1)
        term *= y_squared
        term_contribution = term / (2 * n + 1)
        result += term_contribution

        # Check for convergence (compare absolute values)
        if term_contribution.abs() < tolerance:
            break

    # Return 2 * artanh((x-1)/(x+1))
    return _2 * result


def _ln_range_reduction(x: FlexFloat) -> FlexFloat:
    """Compute natural logarithm using range reduction to improve convergence.

    Uses different strategies based on the magnitude of x:
    - For values near 1: direct Taylor series
    - For large values: iterative square roots
    - For small values: use ln(x) = -ln(1/x)

    Args:
        x (FlexFloat): The input value (must be positive).

    Returns:
        FlexFloat: The natural logarithm of x.
    """
    # For very small values, use ln(x) = -ln(1/x)
    if x < 0.1:
        reciprocal = _1 / x
        return -_ln_range_reduction(reciprocal)

    # For values close to 1, use direct Taylor series
    if x <= 2.0:
        return _ln_taylor_series(x)

    # For large values, use iterative square roots
    multiplier = _1

    max_reductions = 30
    for _ in range(max_reductions):
        if x <= 2.0:
            break
        x = sqrt(x)
        multiplier = multiplier * _2

    # Compute ln(current_x) using Taylor series
    ln_result = _ln_taylor_series(x)

    # Apply the multiplier
    return multiplier * ln_result


def log(x: FlexFloat, base: FlexFloat = e) -> FlexFloat:
    """Return the logarithm of x to the given base using Taylor series.

    Args:
        x (FlexFloat): The value to compute the logarithm of.
        base (FlexFloat, optional): The base of the logarithm. Defaults to e.

    Returns:
        FlexFloat: The logarithm of x to the given base.

    Raises:
        ValueError: If x is negative or zero, or if base is invalid.
    """
    # Handle special cases
    if x.is_nan() or base.is_nan():
        return FlexFloat.nan()

    if x.is_zero() or x.sign:  # x <= 0
        return FlexFloat.nan()

    if x.is_infinity():
        return FlexFloat.infinity(sign=False)

    # Handle base special cases
    if base.is_zero() or base.sign or base.is_infinity():
        return FlexFloat.nan()

    # Check if base is 1 (which would make logarithm undefined)
    if abs(base - 1.0) < 1e-15:
        return FlexFloat.nan()

    # If x is 1, log of any valid base is 0
    if abs(x - 1.0) < 1e-15:
        return FlexFloat.zero()

    # Compute natural logarithm using range reduction and Taylor series
    ln_x = _ln_range_reduction(x)

    # If base is e (natural logarithm), return directly
    if abs(base - math.e) < 1e-15:
        return ln_x

    # For other bases, use change of base formula: log_base(x) = ln(x) / ln(base)
    ln_base = _ln_range_reduction(base)
    return ln_x / ln_base


def log10(x: FlexFloat) -> FlexFloat:
    """Return the base-10 logarithm of x using Taylor series.

    Args:
        x (FlexFloat): The value to compute the base-10 logarithm of.

    Returns:
        FlexFloat: The base-10 logarithm of x.

    Raises:
        ValueError: If x is negative or zero.
    """
    return log(x, _10)


def log1p(x: FlexFloat) -> FlexFloat:
    """Return the natural logarithm of 1 + x using Taylor series.

    This function is more accurate than log(1 + x) for values of x close to zero.

    Args:
        x (FlexFloat): The value to compute the natural logarithm of 1 + x.

    Returns:
        FlexFloat: The natural logarithm of 1 + x.

    Raises:
        ValueError: If 1 + x is negative or zero.
    """
    # Handle special cases
    if x.is_nan():
        return FlexFloat.nan()

    # Check if 1 + x would be <= 0
    one_plus_x = _1 + x
    if one_plus_x.is_zero() or one_plus_x.sign:
        return FlexFloat.nan()

    if one_plus_x.is_infinity():
        return FlexFloat.infinity(sign=False)

    # For small x, use Taylor series directly: ln(1+x) = x - x²/2 + x³/3 - ...
    if abs(x) < 0.5:  # Direct Taylor series for better accuracy
        return _ln_taylor_series(one_plus_x)

    # For larger x, use the regular log function
    return log(one_plus_x)


def log2(x: FlexFloat) -> FlexFloat:
    """Return the base-2 logarithm of x using Taylor series.

    Args:
        x (FlexFloat): The value to compute the base-2 logarithm of.

    Returns:
        FlexFloat: The base-2 logarithm of x.

    Raises:
        ValueError: If x is negative or zero.
    """
    return log(x, _2)


def modf(x: FlexFloat) -> tuple[FlexFloat, FlexFloat]:
    """Return the fractional and integer parts of x (not implemented).

    Args:
        x (FlexFloat): The value to split into fractional and integer parts.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("modf is not implemented for FlexFloat.")


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


def perm(n: FlexFloat, k: FlexFloat | None = None) -> FlexFloat:
    """Return the number of ways to choose k items from n items without repetition
    (not implemented).

    Args:
        n (FlexFloat): The total number of items.
        k (FlexFloat | None, optional): The number of items to choose. Defaults to None.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("perm is not implemented for FlexFloat.")


def radians(x: FlexFloat) -> FlexFloat:
    """Convert angle x from degrees to radians (not implemented).

    Args:
        x (FlexFloat): The angle in degrees.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("radians is not implemented for FlexFloat.")


def remainder(x: FlexFloat, y: FlexFloat) -> FlexFloat:
    """Return the remainder of x divided by y (not implemented).

    Args:
        x (FlexFloat): The dividend value.
        y (FlexFloat): The divisor value.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("remainder is not implemented for FlexFloat.")


def sin(x: FlexFloat) -> FlexFloat:
    """Return the sine of x (not implemented).

    Args:
        x (FlexFloat): The value to compute the sine of.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("sin is not implemented for FlexFloat.")


def sinh(x: FlexFloat) -> FlexFloat:
    """Return the hyperbolic sine of x (not implemented).

    Args:
        x (FlexFloat): The value to compute the hyperbolic sine of.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("sinh is not implemented for FlexFloat.")


def tan(x: FlexFloat) -> FlexFloat:
    """Return the tangent of x (not implemented).

    Args:
        x (FlexFloat): The value to compute the tangent of.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("tan is not implemented for FlexFloat.")


def tanh(x: FlexFloat) -> FlexFloat:
    """Return the hyperbolic tangent of x (not implemented).

    Args:
        x (FlexFloat): The value to compute the hyperbolic tangent of.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("tanh is not implemented for FlexFloat.")


def trunc(x: FlexFloat) -> FlexFloat:
    """Return the integer part of x (not implemented).

    Args:
        x (FlexFloat): The value to truncate.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("trunc is not implemented for FlexFloat.")


def ulp(x: FlexFloat) -> FlexFloat:
    """Return the value of the least significant bit of x (not implemented).

    Args:
        x (FlexFloat): The value to compute the least significant bit of.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("ulp is not implemented for FlexFloat.")


def fma(x: FlexFloat, y: FlexFloat, z: FlexFloat) -> FlexFloat:
    """Return x * y + z (not implemented).

    Args:
        x (FlexFloat): The first multiplicand.
        y (FlexFloat): The second multiplicand.
        z (FlexFloat): The value to add to the product.

    Raises:
        NotImplementedError: Always, as this function is not implemented.
    """
    raise NotImplementedError("fma is not implemented for FlexFloat.")
