"""NumPy-compatible array operations on accelerators.

JAX's jax.numpy module provides a NumPy-compatible API that runs on
GPU/TPU. Arrays are immutable — use .at[].set() for updates.

Rust equivalents:
    jnp.array → trueno::Tensor::from_slice
    jnp.matmul → trueno::ops::matmul (SIMD-accelerated)
    jnp.einsum → trueno::ops::einsum
    jnp.reshape → trueno::Tensor::reshape
    jnp.fft.fft → trueno::ops::fft
"""

from jax_gtc.arrays.operations import (
    batch_matmul,
    broadcast_add,
    create_array,
    einsum_contract,
    fft_transform,
    index_update,
    matmul,
    reshape_array,
)

__all__ = [
    "create_array",
    "reshape_array",
    "matmul",
    "batch_matmul",
    "einsum_contract",
    "fft_transform",
    "broadcast_add",
    "index_update",
]
