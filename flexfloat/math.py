"""Math module for FlexFloat, similar to Python's math module.

Provides mathematical functions that operate on FlexFloat instances.
"""

import math
from typing import Final, Iterable
from .core import FlexFloat

# Constants
e: Final[FlexFloat] = FlexFloat.from_float(math.e)
pi: Final[FlexFloat] = FlexFloat.from_float(math.pi)
inf: Final[FlexFloat] = FlexFloat.infinity()
nan: Final[FlexFloat] = FlexFloat.nan()
tau: Final[FlexFloat] = FlexFloat.from_float(math.tau)

_0_5: Final[FlexFloat] = FlexFloat.from_float(0.5)


# Math-like functions for FlexFloat
def exp(x: FlexFloat) -> FlexFloat:
    """Return e raised to the power of x (where x is a FlexFloat)."""
    return e**x


def pow(base: FlexFloat, exp: FlexFloat) -> FlexFloat:
    """Return base raised to the power of exp (both are FlexFloat instances)."""
    return base**exp


def copysign(x: FlexFloat, y: FlexFloat) -> FlexFloat:
    """Return a FlexFloat with the magnitude of x and the sign of y."""
    result = x.copy()
    result.sign = y.sign
    return result


def fabs(x: FlexFloat) -> FlexFloat:
    """Return the absolute value of x."""
    return abs(x)


def isinf(x: FlexFloat) -> bool:
    """Check if x is positive or negative infinity."""
    return x.is_infinity()


def isnan(x: FlexFloat) -> bool:
    """Check if x is NaN (not a number)."""
    return x.is_nan()


def isfinite(x: FlexFloat) -> bool:
    """Check if x is neither an infinity nor NaN."""
    return not x.is_infinity() and not x.is_nan()


def sqrt(x: FlexFloat) -> FlexFloat:
    """Return the square root of x (using power operator)."""
    return x**_0_5


# Unimplemented functions
def acos(x: FlexFloat) -> FlexFloat:
    """Return the arc cosine of x (not implemented)."""
    raise NotImplementedError("acos is not implemented for FlexFloat.")


def acosh(x: FlexFloat) -> FlexFloat:
    """Return the hyperbolic arc cosine of x (not implemented)."""
    raise NotImplementedError("acosh is not implemented for FlexFloat.")


def asin(x: FlexFloat) -> FlexFloat:
    """Return the arc sine of x (not implemented)."""
    raise NotImplementedError("asin is not implemented for FlexFloat.")


def asinh(x: FlexFloat) -> FlexFloat:
    """Return the hyperbolic arc sine of x (not implemented)."""
    raise NotImplementedError("asinh is not implemented for FlexFloat.")


def atan(x: FlexFloat) -> FlexFloat:
    """Return the arc tangent of x (not implemented)."""
    raise NotImplementedError("atan is not implemented for FlexFloat.")


def atan2(y: FlexFloat, x: FlexFloat) -> FlexFloat:
    """Return the arc tangent of y/x (not implemented)."""
    raise NotImplementedError("atan2 is not implemented for FlexFloat.")


def atanh(x: FlexFloat) -> FlexFloat:
    """Return the hyperbolic arc tangent of x (not implemented)."""
    raise NotImplementedError("atanh is not implemented for FlexFloat.")


def cbrt(x: FlexFloat) -> FlexFloat:
    """Return the cube root of x (not implemented)."""
    raise NotImplementedError("cbrt is not implemented for FlexFloat.")


def ceil(x: FlexFloat) -> FlexFloat:
    """Return the ceiling of x (not implemented)."""
    raise NotImplementedError("ceil is not implemented for FlexFloat.")


# Not implemented functions
def dist(p: Iterable[FlexFloat], q: Iterable[FlexFloat]) -> FlexFloat:
    """Return the Euclidean distance between two points p and q (not implemented)."""
    raise NotImplementedError("dist is not implemented for FlexFloat.")


def erf(x: FlexFloat) -> FlexFloat:
    """Return the error function of x (not implemented)."""
    raise NotImplementedError("erf is not implemented for FlexFloat.")


def erfc(x: FlexFloat) -> FlexFloat:
    """Return the complementary error function of x (not implemented)."""
    raise NotImplementedError("erfc is not implemented for FlexFloat.")


def expm1(x: FlexFloat) -> FlexFloat:
    """Return e raised to the power of x, minus 1 (not implemented)."""
    raise NotImplementedError("expm1 is not implemented for FlexFloat.")


def factorial(x: FlexFloat) -> FlexFloat:
    """Return the factorial of x (not implemented)."""
    raise NotImplementedError("factorial is not implemented for FlexFloat.")


def fmod(x: FlexFloat, y: FlexFloat) -> FlexFloat:
    """Return the remainder of x divided by y (not implemented)."""
    raise NotImplementedError("fmod is not implemented for FlexFloat.")


def frexp(x: FlexFloat) -> tuple[FlexFloat, int]:
    """Return the mantissa and exponent of x (not implemented)."""
    raise NotImplementedError("frexp is not implemented for FlexFloat.")


def fsum(seq: Iterable[FlexFloat]) -> FlexFloat:
    """Return an accurate floating-point sum of the values in seq (not implemented)."""
    raise NotImplementedError("fsum is not implemented for FlexFloat.")


def gamma(x: FlexFloat) -> FlexFloat:
    """Return the gamma function of x (not implemented)."""
    raise NotImplementedError("gamma is not implemented for FlexFloat.")


def gcd(*integers: FlexFloat) -> FlexFloat:
    """Return the greatest common divisor of the integers (not implemented)."""
    raise NotImplementedError("gcd is not implemented for FlexFloat.")


def hypot(*coordinates: FlexFloat) -> FlexFloat:
    """Return the Euclidean norm, sqrt(x1*x1 + x2*x2 + ... + xn*xn) (not implemented)."""
    raise NotImplementedError("hypot is not implemented for FlexFloat.")


def isclose(
    a: FlexFloat,
    b: FlexFloat,
    *,
    rel_tol: FlexFloat = FlexFloat.from_float(1e-09),
    abs_tol: FlexFloat = FlexFloat.from_float(0.0),
) -> bool:
    """Check if two FlexFloat instances are close in value (not implemented)."""
    raise NotImplementedError("isclose is not implemented for FlexFloat.")


def lcm(*integers: FlexFloat) -> FlexFloat:
    """Return the least common multiple of the integers (not implemented)."""
    raise NotImplementedError("lcm is not implemented for FlexFloat.")


def ldexp(x: FlexFloat, i: int) -> FlexFloat:
    """Return x * (2**i) (not implemented)."""
    raise NotImplementedError("ldexp is not implemented for FlexFloat.")


def lgamma(x: FlexFloat) -> FlexFloat:
    """Return the natural logarithm of the absolute value of the gamma function of x (not implemented)."""
    raise NotImplementedError("lgamma is not implemented for FlexFloat.")


def log(x: FlexFloat, base: FlexFloat = e) -> FlexFloat:
    """Return the logarithm of x to the given base (not implemented)."""
    raise NotImplementedError("log is not implemented for FlexFloat.")


def log10(x: FlexFloat) -> FlexFloat:
    """Return the base-10 logarithm of x (not implemented)."""
    raise NotImplementedError("log10 is not implemented for FlexFloat.")


def log1p(x: FlexFloat) -> FlexFloat:
    """Return the natural logarithm of 1 + x (not implemented)."""
    raise NotImplementedError("log1p is not implemented for FlexFloat.")


def log2(x: FlexFloat) -> FlexFloat:
    """Return the base-2 logarithm of x (not implemented)."""
    raise NotImplementedError("log2 is not implemented for FlexFloat.")


def modf(x: FlexFloat) -> tuple[FlexFloat, FlexFloat]:
    """Return the fractional and integer parts of x (not implemented)."""
    raise NotImplementedError("modf is not implemented for FlexFloat.")


def nextafter(x: FlexFloat, y: FlexFloat, *, steps: int | None = None) -> FlexFloat:
    """Return the next representable FlexFloat value after x towards y (not implemented)."""
    raise NotImplementedError("nextafter is not implemented for FlexFloat.")


def perm(n: FlexFloat, k: FlexFloat | None = None) -> FlexFloat:
    """Return the number of ways to choose k items from n items without repetition (not implemented)."""
    raise NotImplementedError("perm is not implemented for FlexFloat.")


def radians(x: FlexFloat) -> FlexFloat:
    """Convert angle x from degrees to radians (not implemented)."""
    raise NotImplementedError("radians is not implemented for FlexFloat.")


def remainder(x: FlexFloat, y: FlexFloat) -> FlexFloat:
    """Return the remainder of x divided by y (not implemented)."""
    raise NotImplementedError("remainder is not implemented for FlexFloat.")


def sin(x: FlexFloat) -> FlexFloat:
    """Return the sine of x (not implemented)."""
    raise NotImplementedError("sin is not implemented for FlexFloat.")


def sinh(x: FlexFloat) -> FlexFloat:
    """Return the hyperbolic sine of x (not implemented)."""
    raise NotImplementedError("sinh is not implemented for FlexFloat.")


def tan(x: FlexFloat) -> FlexFloat:
    """Return the tangent of x (not implemented)."""
    raise NotImplementedError("tan is not implemented for FlexFloat.")


def tanh(x: FlexFloat) -> FlexFloat:
    """Return the hyperbolic tangent of x (not implemented)."""
    raise NotImplementedError("tanh is not implemented for FlexFloat.")


def trunc(x: FlexFloat) -> FlexFloat:
    """Return the integer part of x (not implemented)."""
    raise NotImplementedError("trunc is not implemented for FlexFloat.")


def ulp(x: FlexFloat) -> FlexFloat:
    """Return the value of the least significant bit of x (not implemented)."""
    raise NotImplementedError("ulp is not implemented for FlexFloat.")


def fma(x: FlexFloat, y: FlexFloat, z: FlexFloat) -> FlexFloat:
    """Return x * y + z (not implemented)."""
    raise NotImplementedError("fma is not implemented for FlexFloat.")
