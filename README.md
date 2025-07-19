# FlexFloat

A Python library for arbitrary precision floating point arithmetic with a flexible exponent and fixed-size fraction.

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
from flexfloat import FlexFloat

# Create FlexFloat instances
a = FlexFloat.from_float(1.5)
b = FlexFloat.from_float(2.5)

# Perform arithmetic operations
result = a + b
print(result.to_float())  # 4.0

# Handle very large numbers
large_a = FlexFloat.from_float(1e308)
large_b = FlexFloat.from_float(1e308)
large_result = large_a + large_b
# Result has grown exponent to handle overflow
print(len(large_result.exponent))  # > 11 (grows beyond IEEE 754 standard)

# Work with special values
inf = FlexFloat.from_float(float('inf'))
nan = FlexFloat.from_float(float('nan'))
print(inf.is_infinity())  # True
print(nan.is_nan())       # True
```

## Running Tests

```bash
python -m pytest tests
```


## License

MIT License
