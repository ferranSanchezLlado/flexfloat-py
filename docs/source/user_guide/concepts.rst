Core Concepts
=============

Understanding FlexFloat's Architecture
--------------------------------------

FlexFloat is designed around the concept of **growable exponents** and **growable fractions**. This section explains the fundamental concepts that make FlexFloat unique.

IEEE 754 Foundation
~~~~~~~~~~~~~~~~~~~

FlexFloat builds upon IEEE 754 double-precision semantics, then allows the stored exponent and fraction fields to grow:

- **Sign bit**: 1 bit indicating positive (0) or negative (1)
- **Exponent**: Variable length, expanded when the represented range requires it
- **Fraction**: Variable length, expanded with the exponent so precision scales with range

Traditional IEEE 754 Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standard double-precision floats have limitations:

- **Range**: Approximately ±1.8 × 10^308
- **Overflow**: Numbers beyond this range become infinity
- **Underflow**: Very small numbers become zero

FlexFloat's Innovation
~~~~~~~~~~~~~~~~~~~~~~

FlexFloat overcomes these limitations through:

1. **Growable Exponents**: When a number exceeds the current exponent range, FlexFloat automatically increases the exponent bit length
2. **Growable Fractions**: The fraction grows with the exponent field so precision scales with range
3. **Seamless Transition**: Operations seamlessly handle the transition between different exponent sizes

Number Representation
---------------------

FlexFloat Structure
~~~~~~~~~~~~~~~~~~~

A FlexFloat number consists of:

.. code-block:: python

   FlexFloat(
       sign=False,           # Boolean: False=positive, True=negative
       exponent=BitArray,    # Variable-length exponent
       fraction=BitArray     # Variable-length fraction
   )

Example representations:

.. code-block:: python

   from flexfloat import FlexFloat

   # Standard double precision equivalent
   x = FlexFloat.from_float(1.5)
   print(f"Sign: {x.sign}")  # Sign: False
   print(f"Exponent length: {len(x.exponent)} bits")
   print(f"Fraction length: {len(x.fraction)} bits")

Exponent And Fraction Growth
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When an operation would cause overflow, FlexFloat grows the exponent and fraction together:

.. code-block:: python

   from flexfloat import FlexFloat

   # Start with standard precision
   x = FlexFloat.from_float(10.0)

   # Perform operation that would overflow standard float
   large = x ** 400
   print(len(large.exponent) > len(x.exponent))  # True
   print(len(large.fraction) > len(x.fraction))  # True

Special Values
--------------

FlexFloat supports all IEEE 754 special values with extended range:

Infinity
~~~~~~~~

.. code-block:: python

   from flexfloat import FlexFloat

   # Positive and negative infinity
   pos_inf = FlexFloat.infinity()
   neg_inf = FlexFloat.infinity(sign=True)

   # Infinity arithmetic
   result = pos_inf + FlexFloat.from_float(1000)  # Still positive infinity
   result = pos_inf * neg_inf                     # Negative infinity

NaN (Not a Number)
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from flexfloat import FlexFloat

   # Create NaN
   nan = FlexFloat.nan()

   # NaN propagation
   result = nan + FlexFloat.from_float(42)        # Result is NaN
   result = FlexFloat.zero() / FlexFloat.zero()   # Division by zero gives NaN

Zero Values
~~~~~~~~~~~

.. code-block:: python

   from flexfloat import FlexFloat

   # Zero value
   zero = FlexFloat.zero()

   # Zero arithmetic
   result = zero + zero        # Zero
   result = FlexFloat.from_float(1) * zero    # Zero

Precision and Accuracy
----------------------

Mantissa Precision
~~~~~~~~~~~~~~~~~~

FlexFloat grows the fraction when the exponent field grows. For total storage size ``n``, the layout is approximately:

- **Sign**: 1 bit
- **Exponent**: ``3 * log2(n) - 7`` bits
- **Mantissa**: ``n - exponent - 1`` bits

.. code-block:: python

   from flexfloat import FlexFloat

   standard = FlexFloat.from_float(1.23456789012345)
   large = FlexFloat.from_int(1 << 2047)

   print(len(large.exponent) > len(standard.exponent))  # True
   print(len(large.fraction) > len(standard.fraction))  # True

Rounding Behavior
~~~~~~~~~~~~~~~~~

FlexFloat follows IEEE 754 rounding rules:

- **Round to nearest, ties to even** (default)
- Consistent rounding across all operations
- Preserves mathematical properties

.. code-block:: python

   from flexfloat import FlexFloat

   # Rounding examples
   x = FlexFloat.from_float(1) / FlexFloat.from_float(3)     # 0.333...
   y = x * FlexFloat.from_float(3)                # Close to 1.0, with rounding

Comparison with Standard Floats
-------------------------------

Range Comparison
~~~~~~~~~~~~~~~~

.. list-table:: Range Comparison
   :header-rows: 1
   :widths: 30 35 35

   * - Type
     - Minimum Magnitude
     - Maximum Magnitude
   * - IEEE 754 Double
     - ~2.2 × 10^-308
     - ~1.8 × 10^308
   * - FlexFloat
     - Limited by memory
     - Limited by memory

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Standard range**: FlexFloat performs similarly to double precision
- **Extended range**: Some overhead due to dynamic exponent and fraction management
- **Memory usage**: Scales with exponent and fraction size

Use Cases
---------

FlexFloat is ideal for:

Scientific Computing
~~~~~~~~~~~~~~~~~~~~

- Astronomical calculations (very large distances)
- Quantum mechanics (very small scales)
- Numerical analysis requiring extended range

Financial Modeling
~~~~~~~~~~~~~~~~~~

- Long-term compound interest calculations
- Risk modeling with extreme scenarios
- High-precision currency conversions

Engineering Applications
~~~~~~~~~~~~~~~~~~~~~~~~

- Simulations requiring extended precision
- Control systems with wide dynamic ranges
- Signal processing with extreme values

Mathematical Research
~~~~~~~~~~~~~~~~~~~~~

- Number theory computations
- Iterative algorithms prone to overflow
- Exploration of mathematical constants
