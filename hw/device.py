"""Public device API.

The rest of the project should import device construction and kernel launch
from this module rather than reaching into ``hw.core`` directly.  That keeps
the execution-core implementation free to evolve behind a stable public seam.
"""

from .core import Device, create_device, launch

__all__ = ["Device", "create_device", "launch"]
