"""Mathematical constants for FlexFloat."""

import math
from typing import Final

from ..core import FlexFloat

# Public constants
e: Final[FlexFloat] = FlexFloat.from_float(math.e)
"""The mathematical constant e (Euler's number) as a FlexFloat."""

pi: Final[FlexFloat] = FlexFloat.from_float(math.pi)
"""The mathematical constant pi as a FlexFloat."""

inf: Final[FlexFloat] = FlexFloat.infinity()
"""Positive infinity as a FlexFloat."""

nan: Final[FlexFloat] = FlexFloat.nan()
"""Not-a-Number (NaN) as a FlexFloat."""

tau: Final[FlexFloat] = FlexFloat.from_float(math.tau)
"""The mathematical constant tau (2*pi) as a FlexFloat."""

# Internal constants for calculations
_0_5: Final[FlexFloat] = FlexFloat.from_float(0.5)
"""The FlexFloat representation of 0.5."""

_1: Final[FlexFloat] = FlexFloat.from_float(1.0)
"""The FlexFloat representation of 1.0."""

_2: Final[FlexFloat] = FlexFloat.from_float(2.0)
"""The FlexFloat representation of 2.0."""

_3: Final[FlexFloat] = FlexFloat.from_float(3.0)
"""The FlexFloat representation of 3.0."""

_6: Final[FlexFloat] = FlexFloat.from_float(6.0)
"""The FlexFloat representation of 6.0."""

_10: Final[FlexFloat] = FlexFloat.from_float(10.0)
"""The FlexFloat representation of 10.0."""

_1__3: Final[FlexFloat] = FlexFloat.from_float(1 / 3)
"""The FlexFloat representation of 1/3."""

_180: Final[FlexFloat] = FlexFloat.from_float(180.0)
"""The FlexFloat representation of 180."""

# Derived constants
_PI_2: Final[FlexFloat] = pi / _2
"""The FlexFloat representation of pi/2."""

_PI_4: Final[FlexFloat] = pi / FlexFloat.from_float(4.0)
"""The FlexFloat representation of pi/4."""

_2_PI: Final[FlexFloat] = _2 * pi
"""The FlexFloat representation of 2*pi."""
