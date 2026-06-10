# FlexFloat

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/flexfloat.svg)](https://badge.fury.io/py/flexfloat)

A high-precision Python library for arbitrary precision floating-point arithmetic with **growable exponents** and **growable fractions**. FlexFloat extends IEEE 754 double-precision format to handle numbers beyond the standard range while increasing fraction precision as exponent ranges expand.

## ✨ Key Features

- **🔢 Growable Exponents**: Dynamically expand exponent size to handle extremely large (>10^308) or small (<10^-308) numbers
- **🎯 Growable Fractions**: Increase fraction precision proportionally as exponent ranges expand
- **⚡ Full Arithmetic Support**: Addition, subtraction, multiplication, division, and power operations
- **📐 Complete Math Library**: Comprehensive mathematical functions including trigonometric, logarithmic, exponential, and hyperbolic functions
- **🔧 Multiple BitArray Backends**: Choose between bool-list, int64-list, and big-integer implementations for optimal performance
- **🌟 Special Value Handling**: Complete support for NaN, ±infinity, and zero values
- **🛡️ Overflow Protection**: Automatic exponent and fraction growth prevents overflow/underflow errors
- **📊 IEEE 754 Baseline**: Compatible with standard double-precision values before growth is needed

## 🚀 Quick Start

### Installation

```bash
pip install flexfloat
```

### Basic Usage

```python
from flexfloat import FlexFloat

# Create FlexFloat instances
a = FlexFloat.from_float(1.5)
b = FlexFloat.from_float(2.5)

# Perform arithmetic operations
result = a + b
print(result.to_float())  # 4.0

# Handle very large numbers that would overflow standard floats
large_a = FlexFloat.from_float(1e308)
large_b = FlexFloat.from_float(1e308)
large_result = large_a + large_b

# Result automatically grows exponent and fraction precision to handle overflow
print(f"Exponent grew: {len(large_result.exponent) > len(large_a.exponent)}")
print(f"Fraction grew: {len(large_result.fraction) > len(large_a.fraction)}")
print(f"Can represent: {large_result}")  # No overflow!
```

### Advanced Mathematical Functions

```python
from flexfloat import FlexFloat
from flexfloat.math import sin, cos, log, exp, sqrt, sinh, cosh

# Create FlexFloat instances
x = FlexFloat.from_float(2.0)
y = FlexFloat.from_float(3.0)

# Trigonometric functions
angle = FlexFloat.from_float(1.5708)  # π/2 radians
sin_result = sin(angle)
cos_result = cos(angle)
print(f"sin(π/2) = {sin_result.to_float()}")  # ≈ 1.0
print(f"cos(π/2) = {cos_result.to_float()}")  # ≈ 0.0

# Logarithmic and exponential functions
log_result = log(x)  # Natural logarithm
exp_result = exp(x)  # e^x
sqrt_result = sqrt(x)  # √x
print(f"ln(2) = {log_result.to_float()}")
print(f"e^2 = {exp_result.to_float()}")
print(f"√2 = {sqrt_result.to_float()}")

# Hyperbolic functions
sinh_result = sinh(x)
cosh_result = cosh(x)
print(f"sinh(2) = {sinh_result.to_float()}")
print(f"cosh(2) = {cosh_result.to_float()}")

# Power operations with extreme precision
power_result = x ** y  # 2^3 = 8
print(f"2^3 = {power_result.to_float()}")

# Working with mathematical constants
from flexfloat.math import pi, e
circle_area = pi * (x ** FlexFloat.from_float(2.0))  # π * r²
print(f"Area of circle with radius 2: {circle_area.to_float()}")
```

## 🔧 BitArray Backends

FlexFloat supports multiple BitArray implementations for different performance characteristics. You can use them directly or configure FlexFloat to use a specific implementation:

```python
from flexfloat import (
    FlexFloat, 
    ListBoolBitArray,
    ListInt64BitArray,
    BigIntBitArray
)

# Configure FlexFloat to use a specific BitArray implementation
FlexFloat.set_bitarray_implementation(ListBoolBitArray)  # Default
flex_bool = FlexFloat.from_float(42.0)

FlexFloat.set_bitarray_implementation(ListInt64BitArray)  # For performance
flex_int64 = FlexFloat.from_float(42.0)

FlexFloat.set_bitarray_implementation(BigIntBitArray)  # For very large arrays
flex_bigint = FlexFloat.from_float(42.0)

# Use BitArray implementations directly
bits = [True, False, True, False]
bool_array = ListBoolBitArray.from_bits(bits)
int64_array = ListInt64BitArray.from_bits(bits)
bigint_array = BigIntBitArray.from_bits(bits)
```

### Implementation Comparison

| Implementation | Best For | Pros | Cons |
|---------------|----------|------|------|
| `ListBoolBitArray` | Testing and development | Simple, flexible, easy to debug | Slower for large operations |
| `ListInt64BitArray` | Standard operations | Fast for medium-sized arrays, memory efficient | Some overhead for very small arrays |
| `BigIntBitArray` | Any usescases | Python already optimizes it | Overhead for small arrays |

## 📚 API Reference

### Core Operations

```python
# Construction
FlexFloat.from_float(value: float) -> FlexFloat
FlexFloat.from_int(value: int) -> FlexFloat
FlexFloat(sign: bool, exponent: BitArray, fraction: BitArray)

# Conversion
flexfloat.to_float() -> float
flexfloat.to_int() -> int

# Arithmetic Operations
a + b, a - b, a * b, a / b, a ** b
abs(a), -a

# Comparison Operations  
a == b, a != b, a < b, a <= b, a > b, a >= b
```

### Mathematical Functions

FlexFloat provides a comprehensive math library similar to Python's `math` module:

```python
from flexfloat.math import *

# Exponential and Power Functions
exp(x)      # e^x
expm1(x)    # exp(x) - 1 (accurate for small x)
pow(x, y)   # x^y

# Logarithmic Functions
log(x)       # Natural logarithm (base e)
log10(x)     # Base-10 logarithm
log2(x)      # Base-2 logarithm
log1p(x)     # log(1 + x) (accurate for small x)

# Trigonometric Functions
sin(x), cos(x), tan(x)           # Basic trig functions
asin(x), acos(x), atan(x)        # Inverse trig functions
atan2(y, x)                      # Two-argument arctangent
degrees(x), radians(x)           # Angle conversion

# Hyperbolic Functions
sinh(x), cosh(x), tanh(x)        # Hyperbolic functions
asinh(x), acosh(x), atanh(x)     # Inverse hyperbolic functions

# Square Root Functions
sqrt(x)      # Square root
cbrt(x)      # Cube root

# Mathematical Constants
pi, e, tau   # π, Euler's number, τ (2π)
inf, nan     # Positive infinity, Not a Number

# Utility Functions
ceil(x), floor(x)                # Ceiling and floor
fmod(x, y)                       # Floating-point remainder
fabs(x)                          # Absolute value
copysign(x, y)                   # Copy sign from y to x
```

### BitArray Configuration

```python
from flexfloat import FlexFloat
from flexfloat.math import sin, cos, pi

# Configure FlexFloat to use a specific BitArray implementation
FlexFloat.set_bitarray_implementation(implementation: Type[BitArray])
```

### Special Values

```python
from flexfloat import FlexFloat

# Create special values
nan_val = FlexFloat.nan()
inf_val = FlexFloat.infinity()
neg_inf = FlexFloat.negative_infinity()  
zero_val = FlexFloat.zero()

# Check for special values
if result.is_nan():
    print("Result is Not a Number")
if result.is_infinite():
    print("Result is infinite")
if result.is_zero():
    print("Result is zero")
```

## 🧪 Development & Testing

### Development Installation

```bash
git clone https://github.com/ferranSanchezLlado/flexfloat-py.git
cd flexfloat-py
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=flexfloat --cov-report=html

# Run specific test categories
python -m pytest tests/test_arithmetic.py  # Arithmetic operations
python -m pytest tests/test_conversions.py  # Number conversions
python -m pytest tests/test_bitarray.py  # BitArray implementations
```

### Code Quality

```bash
# Format code
black flexfloat/ tests/

# Sort imports
isort flexfloat/ tests/

# Type checking
mypy flexfloat/

# Linting
pylint flexfloat/
flake8 flexfloat/
```

## 🎯 Use Cases

### Scientific Computing
```python
from flexfloat import FlexFloat
from flexfloat.math import sin, cos, pi, exp, log

# High-precision trigonometric calculations
def calculate_wave_interference(amplitude, frequency, time):
    ff_amp = FlexFloat.from_float(amplitude)
    ff_freq = FlexFloat.from_float(frequency)
    ff_time = FlexFloat.from_float(time)
    
    wave = ff_amp * sin(ff_freq * ff_time * pi)
    return wave

# Handle calculations that would overflow standard floats
def flex_factorial(n):
    result = FlexFloat.from_float(1.0)
    for i in range(1, n + 1):
        result = result * FlexFloat.from_float(i)
    return result

large_factorial = flex_factorial(1000)  # No overflow!
```

### Financial Calculations
```python
from flexfloat import FlexFloat
from flexfloat.math import log, exp

# High-precision compound interest calculations
def compound_interest(principal, rate, years, compounds_per_year):
    p = FlexFloat.from_float(principal)
    r = FlexFloat.from_float(rate)
    n = FlexFloat.from_float(compounds_per_year)
    t = FlexFloat.from_float(years)
    
    # A = P(1 + r/n)^(nt)
    rate_per_period = r / n
    exponent = n * t
    base = FlexFloat.from_float(1.0) + rate_per_period
    
    final_amount = p * (base ** exponent)
    return final_amount

# Calculate compound interest over very long periods with high precision
result = compound_interest(1000000.0, 0.05, 100, 12)
```

### Physics Simulations
```python
from flexfloat import FlexFloat
from flexfloat.math import sqrt, pi, exp

# Handle extreme values in physics calculations
c = FlexFloat.from_float(299792458)  # Speed of light (m/s)
mass = FlexFloat.from_float(1e-30)   # Atomic mass (kg)

# E = mc² with extreme precision
energy = mass * c * c

# Quantum mechanics - wave function calculations
def gaussian_wave_packet(x, x0, sigma, k0):
    ff_x = FlexFloat.from_float(x)
    ff_x0 = FlexFloat.from_float(x0)
    ff_sigma = FlexFloat.from_float(sigma)
    ff_k0 = FlexFloat.from_float(k0)
    
    # ψ(x) = exp(-(x-x0)²/(4σ²)) * exp(ik0x)
    displacement = ff_x - ff_x0
    gaussian = exp(-(displacement * displacement) / (FlexFloat.from_float(4.0) * ff_sigma * ff_sigma))
    phase = ff_k0 * ff_x
    
    return gaussian  # Real part only for this example
```

## 🏗️ Architecture

FlexFloat is built with a modular architecture:

```
flexfloat/
├── core.py              # Main FlexFloat class and arithmetic operations
├── types.py             # Type definitions and protocols
├── math/                # Complete mathematical function library
│   ├── __init__.py          # Math module exports
│   ├── constants.py         # Mathematical constants (π, e, etc.)
│   ├── exponential.py       # exp, expm1, pow functions
│   ├── logarithmic.py       # log, log10, log2, log1p functions
│   ├── trigonometric.py     # sin, cos, tan and inverse functions
│   ├── hyperbolic.py        # sinh, cosh, tanh and inverse functions
│   ├── sqrt.py              # sqrt, cbrt functions
│   ├── floating_point.py    # IEEE 754 utilities
│   └── utility.py           # ceil, floor, fmod and other utilities
├── bitarray/            # BitArray implementations
│   ├── bitarray.py          # Abstract base class
│   ├── bitarray_bool.py     # List[bool] implementation
│   ├── bitarray_int64.py    # List[int64] implementation  
│   ├── bitarray_bigint.py   # Python int implementation
│   └── bitarray_mixins.py   # Common functionality
└── __init__.py          # Public API exports
```

### Design Principles

1. **IEEE 754 Compatibility**: Start with standard double-precision format
2. **Graceful Scaling**: Automatically expand exponent and fraction together when needed
3. **Precision Preservation**: Grow fraction capacity proportionally instead of keeping it fixed
4. **Performance Options**: Multiple backends for different use cases
5. **Pythonic Interface**: Natural syntax for mathematical operations
6. **Comprehensive Math Library**: Complete set of mathematical functions matching Python's math module
7. **Special Value Handling**: Proper IEEE 754 semantics for NaN, infinity, and zero

## 📊 Performance Considerations

### When to Use FlexFloat

✅ **Good for:**
- Calculations requiring numbers > 10^308 or < 10^-308
- Scientific computing with extreme values
- Financial calculations requiring high precision
- Preventing overflow/underflow in long calculations

❌ **Consider alternatives for:**
- Simple arithmetic with standard-range numbers
- Performance-critical tight loops
- Applications where standard `float` precision is sufficient

### Optimization Tips

```python
from flexfloat import FlexFloat, ListInt64BitArray, BigIntBitArray
from flexfloat.math import sin, cos, pi

# Choose the right BitArray implementation for your use case
# For standard operations with moderate precision
FlexFloat.set_bitarray_implementation(ListInt64BitArray)

# For most use cases, Python's int is already optimized
FlexFloat.set_bitarray_implementation(BigIntBitArray)

# Use mathematical constants from the math module
from flexfloat.math import pi, e
circle_area = pi * radius * radius  # More accurate than FlexFloat.from_float(3.14159...)

# Batch operations when possible
values = [FlexFloat.from_float(x) for x in range(1000)]
sum_result = sum(values, FlexFloat.zero())

# Use appropriate precision for your use case
if value_in_standard_range:
    result = float(flexfloat_result.to_float())  # Convert back if needed
```

## 📋 Roadmap

### ✅ Version 1.0.0 - Complete Core Features
- [x] Initial release with basic arithmetic and special values
- [x] Complete mathematical function library (trigonometric, logarithmic, exponential, hyperbolic)
- [x] Square root and power functions  
- [x] Mathematical constants (π, e, τ)
- [x] Comprehensive test suite with high coverage
- [x] Multiple BitArray backend implementations
- [x] IEEE 754 compatibility and special value handling

### 🚧 Future Enhancements
- [ ] Performance optimizations for large arrays
- [ ] Serialization support (JSON, pickle)
- [ ] Decimal mode for exact decimal representation
- [ ] Complex number support (FlexComplex class)
- [ ] Additional utility functions (gamma, erf, etc.)
- [ ] GPU acceleration support
- [ ] Integration with NumPy arrays


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- IEEE 754 standard for floating-point arithmetic foundation
- Python community for inspiration and best practices
- Contributors and users who help improve the library

## 📞 Support

- 📚 **Documentation**: Full API documentation available in docstrings
- 🐛 **Issues**: Report bugs on [GitHub Issues](https://github.com/ferranSanchezLlado/flexfloat-py/issues)
- 💬 **Discussions**: Join conversations on [GitHub Discussions](https://github.com/ferranSanchezLlado/flexfloat-py/discussions)
- 📧 **Contact**: Reach out to the maintainer for questions
