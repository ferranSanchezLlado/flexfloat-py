FlexFloat Documentation
=======================

.. image:: https://img.shields.io/badge/python-3.11+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.11+

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://badge.fury.io/py/flexfloat.svg
   :target: https://badge.fury.io/py/flexfloat
   :alt: PyPI version

Welcome to FlexFloat, a high-precision Python library for arbitrary precision floating-point arithmetic with **growable exponents** and **fixed-size fractions**.

FlexFloat extends IEEE 754 double-precision format to handle numbers beyond the standard range while maintaining computational efficiency and precision consistency.

✨ Key Features
----------------

- **🔢 Growable Exponents**: Dynamically expand exponent size to handle extremely large (>10^308) or small (<10^-308) numbers
- **🎯 Fixed-Size Fractions**: Maintain IEEE 754-compatible 52-bit fraction precision for consistent accuracy
- **⚡ Full Arithmetic Support**: Addition, subtraction, multiplication, division, and power operations
- **🔧 Multiple BitArray Backends**: Choose between bool-list, int64-list, and big-integer implementations for optimal performance
- **🌟 Special Value Handling**: Complete support for NaN, ±infinity, and zero values
- **🛡️ Overflow Protection**: Automatic exponent growth prevents overflow/underflow errors
- **📊 IEEE 754 Baseline**: Fully compatible with standard double-precision format as the starting point

🚀 Quick Start
---------------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install flexfloat

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from flexfloat import FlexFloat

   # Create a FlexFloat number
   x = FlexFloat(1.5)
   y = FlexFloat(2.5)

   # Perform arithmetic operations
   result = x + y
   print(result)  # FlexFloat(4.0)

   # Handle very large numbers
   large_num = FlexFloat(10) ** 400
   print(large_num)  # Handles numbers beyond standard float range

📚 Table of Contents
---------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   user_guide/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/bitarray
   api/types

.. toctree::
   :maxdepth: 1
   :caption: Math API

   api/math

📖 API Documentation
---------------------

Core Classes
~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   flexfloat.FlexFloat

BitArray Implementations
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   flexfloat.BitArray
   flexfloat.ListBoolBitArray
   flexfloat.ListInt64BitArray
   flexfloat.BigIntBitArray

Math Functions
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :template: module.rst

   flexfloat.math

🔗 Indices and Tables
----------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

🤝 Contributing
----------------

We welcome contributions! Please see our `Contributing Guide <contributing.html>`_ for details on how to get started.

📄 License
-----------

This project is licensed under the MIT License - see the `License <license.html>`_ file for details.

💬 Support
-----------

If you encounter any issues or have questions, please:

1. Check the `documentation <https://flexfloat-py.readthedocs.io/>`_
2. Search existing `GitHub issues <https://github.com/ferranSanchezLlado/flexfloat-py/issues>`_
3. Create a new issue if needed

📊 Version Information
-----------------------

This documentation is for FlexFloat version |version|.
