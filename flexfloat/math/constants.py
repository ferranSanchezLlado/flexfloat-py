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
