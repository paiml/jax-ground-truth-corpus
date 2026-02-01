"""Array operation recipes for jax.numpy.

Demonstrates core array operations that map to SIMD/GPU primitives
in the Sovereign AI Stack's trueno crate.

References:
    - JAX NumPy API: https://jax.readthedocs.io/en/latest/jax.numpy.html
    - JAX source: jax/numpy/linalg.py, jax/_src/numpy/

Rust cross-reference:
    All operations map to trueno::Tensor methods with SIMD dispatch
    (AVX2/AVX-512/NEON) and optional wgpu GPU execution.

"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array


def create_array(
    data: Sequence[float] | Sequence[Sequence[float]],
    dtype: jnp.dtype | None = None,
) -> Array:
    """Create a JAX array from Python data.

    JAX arrays are immutable and allocated on the default device
    (GPU if available). Use jnp.float32 for GPU efficiency.

    Args:
        data: Nested sequences of numbers.
        dtype: Data type. Default jnp.float32.

    Returns:
        JAX array on the default device.

    Examples:
        >>> a = create_array([1.0, 2.0, 3.0])
        >>> a.shape
        (3,)
        >>> a.dtype
        dtype('float32')

        >>> m = create_array([[1.0, 2.0], [3.0, 4.0]])
        >>> m.shape
        (2, 2)

    Rust equivalent:
        trueno::Tensor::from_slice(&[1.0, 2.0, 3.0], &[3])

    """
    if dtype is None:
        dtype = jnp.float32
    return jnp.array(data, dtype=dtype)


def reshape_array(x: Array, shape: tuple[int, ...]) -> Array:
    """Reshape array to a new shape without copying data.

    Uses -1 for a single inferred dimension. The total element count
    must remain the same.

    Args:
        x: Input array.
        shape: Target shape. Use -1 for one inferred dimension.

    Returns:
        Reshaped view of the array.

    Examples:
        >>> import jax.numpy as jnp
        >>> x = jnp.arange(12)
        >>> reshape_array(x, (3, 4)).shape
        (3, 4)
        >>> reshape_array(x, (2, -1)).shape
        (2, 6)

    Rust equivalent:
        trueno::Tensor::reshape(&[3, 4]) — zero-copy view change.

    """
    return jnp.reshape(x, shape)


def matmul(a: Array, b: Array) -> Array:
    """Matrix multiplication.

    For 2D arrays, standard matrix product. For higher dimensions,
    batched matmul on the last two axes.

    Args:
        a: Left matrix (..., M, K).
        b: Right matrix (..., K, N).

    Returns:
        Product matrix (..., M, N).

    Examples:
        >>> import jax.numpy as jnp
        >>> a = jnp.ones((3, 4))
        >>> b = jnp.ones((4, 2))
        >>> matmul(a, b).shape
        (3, 2)

    Rust equivalent:
        trueno::ops::matmul dispatches to AVX2 FMA, AVX-512 VNNI,
        or wgpu GemmKernel depending on size and backend.

    """
    return jnp.matmul(a, b)


def batch_matmul(a: Array, b: Array) -> Array:
    """Batched matrix multiplication.

    Multiplies corresponding matrices in a batch. Leading dimensions
    must be broadcastable.

    Args:
        a: Batched left matrices (B, M, K).
        b: Batched right matrices (B, K, N).

    Returns:
        Batched product (B, M, N).

    Examples:
        >>> import jax.numpy as jnp
        >>> a = jnp.ones((8, 3, 4))
        >>> b = jnp.ones((8, 4, 2))
        >>> batch_matmul(a, b).shape
        (8, 3, 2)

    Rust equivalent:
        trueno::ops::batch_matmul or vmap over matmul.
        On GPU, uses realizar::kernels::GemmKernel with batch dispatch.

    """
    return jnp.matmul(a, b)


def einsum_contract(subscripts: str, *operands: Array) -> Array:
    """Einstein summation convention.

    Expressive notation for tensor contractions, traces, outer products,
    transposes, and more. Compiled to optimized XLA operations.

    Args:
        subscripts: Einsum subscript string (e.g., "ij,jk->ik" for matmul).
        operands: Input arrays.

    Returns:
        Result of the einsum contraction.

    Examples:
        >>> import jax.numpy as jnp
        >>> # Matrix multiply
        >>> a = jnp.ones((3, 4))
        >>> b = jnp.ones((4, 5))
        >>> einsum_contract("ij,jk->ik", a, b).shape
        (3, 5)

        >>> # Batch dot product
        >>> x = jnp.ones((8, 3))
        >>> y = jnp.ones((8, 3))
        >>> einsum_contract("bi,bi->b", x, y).shape
        (8,)

        >>> # Trace
        >>> m = jnp.eye(3)
        >>> float(einsum_contract("ii->", m))
        3.0

    Rust equivalent:
        trueno::ops::einsum parses subscript notation and dispatches
        to optimized contraction kernels (BLAS for matmul, custom
        SIMD for general contractions).

    """
    return jnp.einsum(subscripts, *operands)


def fft_transform(x: Array, axis: int = -1) -> Array:
    """Compute the 1D discrete Fourier transform.

    Args:
        x: Input signal array.
        axis: Axis along which to compute the FFT.

    Returns:
        Complex array of Fourier coefficients.

    Examples:
        >>> import jax.numpy as jnp
        >>> signal = jnp.array([1.0, 0.0, -1.0, 0.0])
        >>> freqs = fft_transform(signal)
        >>> freqs.shape
        (4,)

    Rust equivalent:
        trueno::ops::fft uses SIMD-optimized radix-2/4 FFT
        (AVX2 for f32, NEON for ARM).

    """
    return jnp.fft.fft(x, axis=axis)


def broadcast_add(a: Array, b: Array) -> Array:
    """Element-wise addition with NumPy-style broadcasting.

    Args:
        a: First array.
        b: Second array (broadcastable with a).

    Returns:
        Element-wise sum.

    Examples:
        >>> import jax.numpy as jnp
        >>> a = jnp.ones((3, 4))
        >>> b = jnp.ones((4,))
        >>> broadcast_add(a, b).shape
        (3, 4)

    Rust equivalent:
        trueno::broadcast handles shape expansion and dispatches
        to SIMD add (vaddps for AVX2, fadd for NEON).

    """
    return a + b


def index_update(x: Array, index: int, value: float) -> Array:
    """Functional array update (immutable).

    JAX arrays are immutable. Use .at[].set() for functional updates
    that return a new array.

    Args:
        x: Input array.
        index: Position to update.
        value: New value.

    Returns:
        New array with updated value.

    Examples:
        >>> import jax.numpy as jnp
        >>> x = jnp.zeros(5)
        >>> updated = index_update(x, 2, 1.0)
        >>> updated.tolist()
        [0.0, 0.0, 1.0, 0.0, 0.0]

    Rust equivalent:
        trueno::Tensor is Copy-on-Write. Mutations create a new
        allocation only if the reference count > 1.

    """
    return x.at[index].set(value)
