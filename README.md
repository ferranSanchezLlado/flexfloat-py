# BigFloat

A Python library for arbitrary precision floating point arithmetic with growable exponents and fixed-size fractions.

## Features

- **Growable Exponents**: Handle very large or very small numbers by dynamically adjusting the exponent size
- **Fixed-Size Fractions**: Maintain precision consistency with IEEE 754-compatible 52-bit fractions  
- **IEEE 754 Compatibility**: Follows IEEE 754 double-precision format as the baseline
- **Special Value Support**: Handles NaN, positive/negative infinity, and zero values
- **Arithmetic Operations**: Addition and subtraction with proper overflow/underflow handling

## Installation

```bash
pip install -e .
```

## Development Installation

```bash
pip install -e ".[dev]"
```

## Usage

```python
from bigfloat import BigFloat

# Create BigFloat instances
a = BigFloat.from_float(1.5)
b = BigFloat.from_float(2.5)

# Perform arithmetic operations
result = a + b
print(result.to_float())  # 4.0

# Handle very large numbers
large_a = BigFloat.from_float(1e308)
large_b = BigFloat.from_float(1e308)
large_result = large_a + large_b
# Result has grown exponent to handle overflow
print(len(large_result.exponent))  # > 11 (grows beyond IEEE 754 standard)

# Work with special values
inf = BigFloat.from_float(float('inf'))
nan = BigFloat.from_float(float('nan'))
print(inf.is_infinity())  # True
print(nan.is_nan())       # True
```

## Running Tests

```bash
# Run all tests
python -m unittest discover tests

# Run specific test files
python -m unittest tests.test_arithmetic
python -m unittest tests.test_conversions
python -m unittest tests.test_bigfloat
python -m unittest tests.test_utils
```

## Project Structure

```
bigfloat/
├── bigfloat/           # Main package
│   ├── __init__.py     # Package initialization and exports
│   ├── core.py         # BigFloat class implementation
│   ├── utils.py        # Utility functions for bit operations
│   └── types.py        # Type definitions
├── tests/              # Test suite
│   ├── __init__.py     # Test utilities and base classes
│   ├── test_arithmetic.py      # Arithmetic operation tests
│   ├── test_bigfloat.py        # BigFloat class tests  
│   ├── test_conversions.py     # Conversion function tests
│   └── test_utils.py           # Utility function tests
├── setup.py            # Package setup configuration
└── README.md           # This file
```

## License

MIT License
