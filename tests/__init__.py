"""Base test utilities for the bigfloat package."""

import unittest
from bigfloat.types import BitArray


class BigFloatTestCase(unittest.TestCase):
    """Base test case class with common utilities for BigFloat tests."""

    @staticmethod
    def _parse_bitarray(data: str) -> BitArray:
        """Parse a string of bits into a BitArray."""
        data = data.replace(" ", "")
        return [bit == "1" for bit in data]
