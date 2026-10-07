"""Public device API.

The rest of the project should import device construction and kernel launch
from this module rather than reaching into ``hw.core`` directly.  That keeps
the execution-core implementation free to evolve behind a stable public seam.
"""

from .core import Device, create_device, launch
from .timing import ProfileCounters, StallCounters, TimingModel

__all__ = [
    "Device",
    "ProfileCounters",
    "StallCounters",
    "TimingModel",
    "create_device",
    "launch",
]
