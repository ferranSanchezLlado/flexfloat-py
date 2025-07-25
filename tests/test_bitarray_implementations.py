"""Tests for BitArray implementations and protocol."""

import unittest
from typing import Type

from flexfloat import (
    BigIntBitArray,
    BitArray,
    BitArrayType,
    ListBoolBitArray,
    ListInt64BitArray,
    create_bitarray,
    get_available_implementations,
)
from tests import FlexFloatTestCase


class TestBitArrayImplementations(FlexFloatTestCase):
    """Test all BitArray implementations to ensure consistent behavior."""

    def get_implementations(self) -> list[tuple[str, Type[BitArray]]]:
        """Get all available implementations for testing."""
        return [
            ("bool", ListBoolBitArray),
            ("int64", ListInt64BitArray),
            ("bigint", BigIntBitArray),
        ]

    def test_factory_function(self):
        """Test the factory function creates correct implementations."""
        for impl_name, impl_class in self.get_implementations():
            with self.subTest(implementation=impl_name):
                bits = [True, False, True]
                result = create_bitarray(impl_name, bits)
                self.assertIsInstance(result, impl_class)
                self.assertEqual(list(result), bits)

    def test_factory_function_invalid_implementation(self):
        """Test factory function raises error for invalid implementation."""
        with self.assertRaises(ValueError):
            create_bitarray("invalid_implementation")

    def test_get_available_implementations(self):
        """Test getting available implementations."""
        implementations = get_available_implementations()
        self.assertIsInstance(implementations, list)
        self.assertIn("bool", implementations)
        self.assertIn("int64", implementations)
        self.assertIn("bigint", implementations)

    def test_empty_initialization(self):
        """Test that all implementations handle empty initialization consistently."""
        for impl_name, impl_class in self.get_implementations():
            with self.subTest(implementation=impl_name):
                bit_array = impl_class()
                self.assertEqual(len(bit_array), 0)
                self.assertEqual(list(bit_array), [])

    def test_initialization_with_bits(self):
        """Test initialization with bit list."""
        test_bits = [True, False, True, False, True]
        for impl_name, impl_class in self.get_implementations():
            with self.subTest(implementation=impl_name):
                bit_array = impl_class(test_bits)
                self.assertEqual(len(bit_array), 5)
                self.assertEqual(list(bit_array), test_bits)

    def test_from_float(self):
        """Test from_float class method."""
        test_value = 3.14159
        for impl_name, impl_class in self.get_implementations():
            with self.subTest(implementation=impl_name):
                bit_array = impl_class.from_float(test_value)
                self.assertEqual(len(bit_array), 64)
                # Test that it can be converted back
                result = bit_array.to_float()
                self.assertAlmostEqual(result, test_value, places=10)

    def test_from_signed_int(self):
        """Test from_signed_int class method."""
        test_cases = [
            (5, 8),
            (-3, 8),
            (127, 8),
            (-128, 8),
        ]
        for impl_name, impl_class in self.get_implementations():
            for value, length in test_cases:
                with self.subTest(implementation=impl_name, value=value, length=length):
                    bit_array = impl_class.from_signed_int(value, length)
                    self.assertEqual(len(bit_array), length)
                    result = bit_array.to_signed_int()
                    self.assertEqual(result, value)

    def test_zeros_and_ones(self):
        """Test zeros and ones class methods."""
        for impl_name, impl_class in self.get_implementations():
            with self.subTest(implementation=impl_name):
                # Test zeros
                zeros = impl_class.zeros(10)
                self.assertEqual(len(zeros), 10)
                self.assertEqual(list(zeros), [False] * 10)
                self.assertFalse(zeros.any())

                # Test ones
                ones = impl_class.ones(10)
                self.assertEqual(len(ones), 10)
                self.assertEqual(list(ones), [True] * 10)
                self.assertTrue(ones.all())

    def test_parse_bitarray(self):
        """Test parse_bitarray static method."""
        test_string = "1010 1100"
        expected = [True, False, True, False, True, True, False, False]
        for impl_name, impl_class in self.get_implementations():
            with self.subTest(implementation=impl_name):
                bit_array = impl_class.parse_bitarray(test_string)
                self.assertEqual(list(bit_array), expected)

    def test_to_int(self):
        """Test to_int method."""
        test_cases: list[tuple[list[bool], int]] = [
            ([True, False, True], 5),  # 101 binary = 5 decimal
            ([True, True, True], 7),  # 111 binary = 7 decimal
            ([False, False, False], 0),  # 000 binary = 0 decimal
            ([], 0),  # empty = 0
        ]
        for impl_name, impl_class in self.get_implementations():
            for bits, expected in test_cases:
                with self.subTest(implementation=impl_name, bits=bits):
                    bit_array = impl_class(bits)
                    result = bit_array.to_int()
                    self.assertEqual(result, expected)

    def test_to_signed_int(self):
        """Test to_signed_int method."""
        for impl_name, impl_class in self.get_implementations():
            with self.subTest(implementation=impl_name):
                # Test positive number
                bit_array = impl_class.from_signed_int(5, 8)
                result = bit_array.to_signed_int()
                self.assertEqual(result, 5)

                # Test negative number
                bit_array = impl_class.from_signed_int(-3, 8)
                result = bit_array.to_signed_int()
                self.assertEqual(result, -3)

    def test_indexing(self):
        """Test indexing operations."""
        test_bits = [True, False, True, False, True]
        for impl_name, impl_class in self.get_implementations():
            with self.subTest(implementation=impl_name):
                bit_array = impl_class(test_bits)

                # Test getting individual bits
                self.assertTrue(bit_array[0])
                self.assertFalse(bit_array[1])
                self.assertTrue(bit_array[2])

                # Test slicing
                slice_result = bit_array[1:4]
                self.assertEqual(list(slice_result), [False, True, False])

    def test_setitem(self):
        """Test setting items."""
        for impl_name, impl_class in self.get_implementations():
            with self.subTest(implementation=impl_name):
                bit_array = impl_class([False, False, False])

                # Set individual bit
                bit_array[1] = True
                self.assertTrue(bit_array[1])

                # Set slice
                bit_array[0:2] = [True, False]
                self.assertTrue(bit_array[0])
                self.assertFalse(bit_array[1])

    def test_iteration(self):
        """Test iteration over bits."""
        test_bits = [True, False, True]
        for impl_name, impl_class in self.get_implementations():
            with self.subTest(implementation=impl_name):
                bit_array = impl_class(test_bits)
                result = list(bit_array)
                self.assertEqual(result, test_bits)

    def test_concatenation(self):
        """Test concatenation operations."""
        bits1 = [True, False]
        bits2 = [False, True]
        for impl_name, impl_class in self.get_implementations():
            with self.subTest(implementation=impl_name):
                bit_array1 = impl_class(bits1)
                bit_array2 = impl_class(bits2)

                # Test concatenation with another BitArray
                result = bit_array1 + bit_array2
                self.assertEqual(list(result), bits1 + bits2)

                # Test concatenation with list
                result = bit_array1 + [True, True]
                self.assertEqual(list(result), bits1 + [True, True])

    def test_equality(self):
        """Test equality operations."""
        test_bits = [True, False, True]
        for impl_name, impl_class in self.get_implementations():
            with self.subTest(implementation=impl_name):
                bit_array1 = impl_class(test_bits)
                bit_array2 = impl_class(test_bits)

                self.assertEqual(bit_array1, bit_array2)
                self.assertEqual(bit_array1, test_bits)

    def test_boolean_operations(self):
        """Test boolean operations (any, all, bool)."""
        for impl_name, impl_class in self.get_implementations():
            with self.subTest(implementation=impl_name):
                # Test with all False
                all_false = impl_class([False, False, False])
                self.assertFalse(all_false.any())
                self.assertFalse(all_false.all())
                self.assertFalse(bool(all_false))

                # Test with all True
                all_true = impl_class([True, True, True])
                self.assertTrue(all_true.any())
                self.assertTrue(all_true.all())
                self.assertTrue(bool(all_true))

                # Test with mixed
                mixed = impl_class([True, False, True])
                self.assertTrue(mixed.any())
                self.assertFalse(mixed.all())
                self.assertTrue(bool(mixed))

    def test_count(self):
        """Test count method."""
        test_bits = [True, False, True, False, True]
        for impl_name, impl_class in self.get_implementations():
            with self.subTest(implementation=impl_name):
                bit_array = impl_class(test_bits)
                self.assertEqual(bit_array.count(True), 3)
                self.assertEqual(bit_array.count(False), 2)

    def test_reverse(self):
        """Test reverse method."""
        test_bits = [True, False, True, False]
        expected = [False, True, False, True]
        for impl_name, impl_class in self.get_implementations():
            with self.subTest(implementation=impl_name):
                bit_array = impl_class(test_bits)
                reversed_array = bit_array.reverse()
                self.assertEqual(list(reversed_array), expected)

    def test_shift(self):
        """Test shift method."""
        test_bits = [True, False, True, False]
        for impl_name, impl_class in self.get_implementations():
            with self.subTest(implementation=impl_name):
                bit_array = impl_class(test_bits)

                # Test left shift
                left_shifted = bit_array.shift(1)
                self.assertEqual(list(left_shifted), [False, True, False, True])

                # Test right shift
                right_shifted = bit_array.shift(-1)
                self.assertEqual(list(right_shifted), [False, True, False, False])

                # Test shift with fill
                filled_shift = bit_array.shift(1, fill=True)
                self.assertEqual(list(filled_shift), [True, True, False, True])

    def test_copy(self):
        """Test copy method."""
        # Test each implementation separately to avoid any shared state
        for impl_name, impl_class in self.get_implementations():
            with self.subTest(implementation=impl_name):
                test_bits = [
                    True,
                    False,
                    True,
                ]  # Create fresh test data for each iteration
                bit_array = impl_class(
                    test_bits.copy()
                )  # Use copy to ensure no shared references
                copied = bit_array.copy()

                # Verify initial state
                self.assertEqual(list(copied), test_bits)
                self.assertIsNot(copied, bit_array)  # Different objects

                # Verify data integrity before modification
                self.assertEqual(list(bit_array), test_bits)
                self.assertEqual(list(copied), test_bits)

                # Modify original and ensure copy is unaffected
                bit_array[0] = False

                # Verify that only the original was modified
                self.assertEqual(bit_array[0], False)
                self.assertEqual(
                    copied[0],
                    True,
                    f"Copy was unexpectedly modified for {impl_name} implementation",
                )

    def test_string_representations(self):
        """Test string representations."""
        test_bits = [True, False, True]
        for impl_name, impl_class in self.get_implementations():
            with self.subTest(implementation=impl_name):
                bit_array = impl_class(test_bits)

                # Test __str__
                self.assertEqual(str(bit_array), "101")

                # Test __repr__
                self.assertIn(impl_class.__name__, repr(bit_array))
                self.assertIn("[True, False, True]", repr(bit_array))

    def test_protocol_compliance(self):
        """Test that implementations satisfy the protocol."""
        for impl_name, impl_class in self.get_implementations():
            with self.subTest(implementation=impl_name):
                instance = impl_class([True, False])
                self.assertIsInstance(instance, BitArray)


class TestBitArrayBackwardCompatibility(FlexFloatTestCase):
    """Test that the old BitArray interface still works."""

    def test_old_bitarray_interface(self):
        """Test that old code using BitArray still works."""
        # This should work exactly as before
        bit_array = BitArrayType([True, False, True])
        self.assertEqual(len(bit_array), 3)
        self.assertEqual(list(bit_array), [True, False, True])

        # Test class methods
        zeros = BitArrayType.zeros(5)
        self.assertEqual(list(zeros), [False] * 5)

        ones = BitArrayType.ones(3)
        self.assertEqual(list(ones), [True] * 3)

        from_float = BitArrayType.from_float(1.0)
        self.assertEqual(len(from_float), 64)

    def test_parse_bitarray_module_function(self):
        """Test the module-level parse_bitarray function."""
        from flexfloat.bitarray import parse_bitarray

        result = parse_bitarray("101")
        self.assertEqual(list(result), [True, False, True])


if __name__ == "__main__":
    unittest.main()
