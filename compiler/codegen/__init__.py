"""C++ ISA backend dispatcher."""

from .dispatcher import lower_common_kernel, lower_kernel

__all__ = ["lower_common_kernel", "lower_kernel"]
