"""Core JAX function transformations.

JAX's programming model is built on composable function transformations:
- jit: Compile functions with XLA for GPU/TPU execution
- grad: Compute gradients via reverse-mode automatic differentiation
- vmap: Auto-vectorize functions over batch dimensions
- pmap: Parallelize functions across multiple devices

Rust equivalents:
    jit → trueno::jit (SIMD/GPU compilation)
    grad → entrenar::autograd (reverse-mode AD)
    vmap → trueno SIMD lane mapping (AVX2/AVX-512/NEON)
    pmap → repartir::Pool (work-stealing parallelism)
"""

from jax_gtc.transforms.core import (
    compose_transforms,
    grad_transform,
    jit_compile,
    pmap_parallel,
    value_and_grad_transform,
    vmap_batch,
)

__all__ = [
    "jit_compile",
    "grad_transform",
    "value_and_grad_transform",
    "vmap_batch",
    "pmap_parallel",
    "compose_transforms",
]
