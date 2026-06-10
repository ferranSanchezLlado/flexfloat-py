# Changelog

All notable changes to FlexFloat will be documented in this file.

## [1.1.0] - 2026-06-10

### Added

- Proportional fraction growth: FlexFloat now expands fraction capacity alongside exponent capacity so precision scales with representable range.
- Dynamic format sizing that assigns bits to the exponent and gives the remaining storage to the mantissa after the sign bit.
- Test coverage for fraction growth, extended `to_float()` conversion, and full pytest execution under the project dev environment.

### Changed

- `from_int()` now chooses both exponent and fraction sizes for large integer values.
- Addition, subtraction, multiplication, and division now operate on variable-width significands.
- `to_float()` now re-encodes extended finite values into the standard float layout instead of slicing raw exponent bits.
- Documentation now describes exponent and fraction growth together throughout the user guide and README.

## [1.0.0] - 2025-08-03

### 🎉 First Stable Release

FlexFloat 1.0.0 marks the first stable release with a complete implementation of arbitrary precision floating-point arithmetic and a comprehensive mathematical function library.

### ✨ Features

#### Core Arithmetic
- ✅ Full arithmetic operations: addition, subtraction, multiplication, division, power
- ✅ IEEE 754 compatible special value handling (NaN, ±infinity, zero)
- ✅ Growable exponents for handling extremely large/small numbers
- ✅ Stable fraction precision for consistency
- ✅ Multiple BitArray backend implementations for performance optimization

#### Complete Mathematical Function Library
- ✅ **Trigonometric functions**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`
- ✅ **Exponential functions**: `exp`, `expm1`, `pow`
- ✅ **Logarithmic functions**: `log`, `log10`, `log2`, `log1p`
- ✅ **Hyperbolic functions**: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- ✅ **Square root functions**: `sqrt`, `cbrt`
- ✅ **Mathematical constants**: `pi`, `e`, `tau`, `inf`, `nan`
- ✅ **Utility functions**: `ceil`, `floor`, `fabs`, `fmod`, `copysign`, and more

#### BitArray Implementations
- ✅ `ListBoolBitArray`: Simple bool-list implementation for debugging
- ✅ `ListInt64BitArray`: Optimized int64-list implementation for performance
- ✅ `BigIntBitArray`: Python int-based implementation for maximum flexibility

#### API and Usability
- ✅ Pythonic interface with natural mathematical syntax
- ✅ Comprehensive type hints and documentation
- ✅ Full test coverage with extensive edge case testing
- ✅ Clear error handling and special case behavior

### 📊 Technical Details

- **Precision**: Stable fraction precision for consistency
- **Range**: Dynamically growable exponents prevent overflow/underflow
- **Performance**: Multiple backend implementations for different use cases
- **Compatibility**: Python 3.11+ support

### 📚 Documentation

- ✅ Complete API documentation
- ✅ User guide with examples
- ✅ Mathematical function reference
- ✅ Performance optimization guide

### 🧪 Testing

- ✅ Comprehensive test suite with >95% code coverage
- ✅ Mathematical function accuracy verified against Python's `math` module
- ✅ Edge case testing for special values (NaN, infinity, zero)
- ✅ Extreme value testing for very large/small numbers

### 🏗️ Architecture

The library is built with a modular architecture:
- `flexfloat.core`: Main FlexFloat class and arithmetic operations
- `flexfloat.math`: Complete mathematical function library
- `flexfloat.bitarray`: Multiple BitArray backend implementations
- `flexfloat.types`: Type definitions and protocols

This release represents a production-ready library suitable for scientific computing, financial calculations, and any application requiring high-precision floating-point arithmetic beyond the limitations of standard IEEE 754 double precision.
