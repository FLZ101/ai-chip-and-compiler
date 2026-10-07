"""ISA code generation from the unified Model IR."""

from .common_kernels import lower_common_kernel

__all__ = ["lower_common_kernel"]
