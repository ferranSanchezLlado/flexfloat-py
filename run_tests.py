#!/usr/bin/env python3
"""Test runner script for the bigfloat package."""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all tests for the bigfloat package."""
    print("Running BigFloat tests...")
    print("=" * 50)

    # Change to package directory
    package_dir = Path(__file__).parent

    try:
        # Run unittest discovery
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "tests", "-v"],
            cwd=package_dir,
            capture_output=True,
            text=True,
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode == 0:
            print("\n" + "=" * 50)
            print("✅ All tests passed!")
        else:
            print("\n" + "=" * 50)
            print("❌ Some tests failed!")
            sys.exit(1)

    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)


def run_specific_test(test_module):
    """Run tests for a specific module."""
    print(f"Running tests for {test_module}...")
    print("=" * 50)

    package_dir = Path(__file__).parent

    try:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", f"tests.{test_module}", "-v"],
            cwd=package_dir,
            capture_output=True,
            text=True,
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode != 0:
            sys.exit(1)

    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_module = sys.argv[1]
        run_specific_test(test_module)
    else:
        run_tests()
