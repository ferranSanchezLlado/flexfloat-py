# FlexFloat

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/flexfloat.svg)](https://badge.fury.io/py/flexfloat)

A high-precision Python library for arbitrary precision floating-point arithmetic with **growable exponents** and **fixed-size fractions**. FlexFloat extends IEEE 754 double-precision format to handle numbers beyond the standard range while maintaining computational efficiency and precision consistency.

## ✨ Key Features

- **🔢 Growable Exponents**: Dynamically expand exponent size to handle extremely large (>10^308) or small (<10^-308) numbers
- **🎯 Fixed-Size Fractions**: Maintain IEEE 754-compatible 52-bit fraction precision for consistent accuracy
- **⚡ Full Arithmetic Support**: Addition, subtraction, multiplication, division, and power operations
- **🔧 Multiple BitArray Backends**: Choose between list-based and int64-based implementations for optimal performance
- **🌟 Special Value Handling**: Complete support for NaN, ±infinity, and zero values
- **🛡️ Overflow Protection**: Automatic exponent growth prevents overflow/underflow errors
- **📊 IEEE 754 Baseline**: Fully compatible with standard double-precision format as the starting point

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

# Result automatically grows exponent to handle the overflow
print(f"Exponent bits: {len(large_result.exponent)}")  # > 11 (grown beyond IEEE 754)
print(f"Can represent: {large_result}")  # No overflow!
```

### Advanced Examples

```python
from flexfloat import FlexFloat

# Mathematical operations
x = FlexFloat.from_float(2.0)
y = FlexFloat.from_float(3.0)

# Power operations
power_result = x ** y  # 2^3 = 8
print(power_result.to_float())  # 8.0

# Exponential using Euler's number
e_result = FlexFloat.e ** x  # e^2
print(f"e^2 ≈ {e_result.to_float()}")

# Working with extreme values
tiny = FlexFloat.from_float(1e-300)
huge = FlexFloat.from_float(1e300)
extreme_product = tiny * huge
print(f"Product: {extreme_product.to_float()}")  # Still computable!

# Precision demonstration
precise_calc = FlexFloat.from_float(1.0) / FlexFloat.from_float(3.0)
print(f"1/3 with 52-bit precision: {precise_calc}")
```

## 🔧 BitArray Backends

FlexFloat supports multiple BitArray implementations for different performance characteristics:

```python
from flexfloat import (
    FlexFloat, 
    set_default_implementation, 
    get_available_implementations
)

# View available implementations
print(get_available_implementations())  # ['list', 'int64']

# Use list-based implementation (default, more flexible)
set_default_implementation('list')
flex_list = FlexFloat.from_float(42.0)

# Use int64-based implementation (faster for small bit arrays)
set_default_implementation('int64')
flex_int64 = FlexFloat.from_float(42.0)

# Both produce the same results with different performance characteristics
```

### Implementation Comparison

| Implementation | Best For | Pros | Cons |
|---------------|----------|------|------|
| `list[bool]` | Smaller exponents and testing | Flexible, easy to understand | Slower for large numbers |
| `list[int64]` | Standard operations | Fast for bigger numbers, efficient memory | Overhead for small numbers |

## 📚 API Reference

### Core Operations

```python
# Construction
FlexFloat.from_float(value: float) -> FlexFloat
FlexFloat(sign: bool, exponent: BitArray, fraction: BitArray)

# Conversion
flexfloat.to_float() -> float

# Arithmetic
a + b, a - b, a * b, a / b, a ** b
abs(a), -a

# Mathematical functions
FlexFloat.e ** x  # Exponential function
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
# Handle calculations that would overflow standard floats
from flexfloat import FlexFloat

# Factorial of large numbers
def flex_factorial(n):
    result = FlexFloat.from_float(1.0)
    for i in range(1, n + 1):
        result = result * FlexFloat.from_float(i)
    return result

large_factorial = flex_factorial(1000)  # No overflow!
```

### Financial Calculations
```python
# High-precision compound interest calculations
principal = FlexFloat.from_float(1000000.0)
rate = FlexFloat.from_float(1.05)  # 5% annual return
years = FlexFloat.from_float(100)

# Calculate compound interest over very long periods
final_amount = principal * (rate ** years)
```

### Physics Simulations
```python
# Handle extreme values in physics calculations
c = FlexFloat.from_float(299792458)  # Speed of light
mass = FlexFloat.from_float(1e-30)   # Atomic mass

# E = mc² with extreme precision
energy = mass * c * c
```

## 🏗️ Architecture

FlexFloat is built with a modular architecture:

```
flexfloat/
├── core.py              # Main FlexFloat class
├── types.py             # Type definitions
├── bitarray/            # BitArray implementations
│   ├── bitarray.py          # Abstract base class
│   ├── bitarray_list.py     # List-based implementation
│   ├── bitarray_int64.py    # Int64-based implementation
│   └── bitarray_mixins.py   # Common functionality
└── __init__.py          # Public API exports
```

### Design Principles

1. **IEEE 754 Compatibility**: Start with standard double-precision format
2. **Graceful Scaling**: Automatically expand exponent when needed
3. **Precision Preservation**: Keep fraction size fixed for consistent accuracy
4. **Performance Options**: Multiple backends for different use cases
5. **Pythonic Interface**: Natural syntax for mathematical operations

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
# Prefer int64 implementation for standard operations
set_default_implementation('int64')

# Batch operations when possible
values = [FlexFloat.from_float(x) for x in range(1000)]
sum_result = sum(values, FlexFloat.zero())

# Use appropriate precision for your use case
if value_in_standard_range:
    result = float(flexfloat_result.to_float())  # Convert back if needed
```

## 📋 Roadmap

- [ ] Additional mathematical functions (sin, cos, tan, log, sqrt)
- [ ] Serialization support (JSON, pickle)
- [ ] Performance optimizations for large arrays
- [ ] Complex number support
- [ ] Decimal mode for exact decimal representation


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