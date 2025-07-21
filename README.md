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
pip install flexfloat
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
```

## Running Tests

```bash
python -m pytest tests
```

## CI/CD and Release Process

This project uses automated CI/CD workflows for building, testing, and releasing:

### Automatic Releases

When a Pull Request is merged to the `main` branch:

1. **Version Detection**: The system analyzes commit messages and PR labels to determine the version bump type:
   - `major`: Breaking changes (e.g., commit messages with "BREAKING CHANGE")
   - `minor`: New features (e.g., commit messages starting with "feat:")
   - `patch`: Bug fixes and other changes (default)

2. **Version Update**: All version references are automatically updated:
   - `pyproject.toml`
   - `flexfloat/__init__.py`

3. **Automated Publishing**:
   - Package is built and tested
   - New Git tag is created (e.g., `v0.2.0`)
   - GitHub Release is created with artifacts
   - Package is published to PyPI

### Manual Releases

You can also trigger releases manually:

1. Go to the "Actions" tab in GitHub
2. Select "Manual Release" workflow
3. Choose the version bump type and add release notes
4. Run the workflow

### Version Bump Labels

Add these labels to your PRs to control version bumping:
- `major` - For breaking changes
- `minor` - For new features  
- `patch` - For bug fixes (default)

### PyPI Setup

To enable automatic PyPI publishing, configure trusted publishing:

1. Go to [PyPI](https://pypi.org/manage/account/publishing/)
2. Add a trusted publisher for your GitHub repository
3. Set the workflow name to `release.yml` or `manual-release.yml`

### Development Workflow

1. Create a feature branch from `main`
2. Make your changes
3. Create a Pull Request
4. The CI will automatically:
   - Run tests and linting
   - Check version consistency
   - Suggest version bump type
   - Build and validate the package
5. When merged, automatic release is triggered

## License

MIT License
