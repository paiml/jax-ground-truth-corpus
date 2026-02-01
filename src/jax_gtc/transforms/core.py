"""Core JAX transformation recipes.

This module provides production-ready patterns for JAX's fundamental transforms:
jit, grad, vmap, and pmap. Each function demonstrates correct usage with type
annotations, error handling, and composability.

Rust cross-reference:
    jit_compile → trueno::jit (planned XLA-to-wgpu compilation)
    grad_transform → entrenar::autograd::backward
    vmap_batch → trueno SIMD vectorization across batch dimension
    pmap_parallel → repartir::Pool::map with work-stealing

References:
    - JAX docs: https://jax.readthedocs.io/en/latest/key-concepts.html
    - JAX source: jax/_src/api.py

"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
from jax import Array


def jit_compile(
    fn: Callable[..., Any],
    *,
    static_argnums: tuple[int, ...] | None = None,
    donate_argnums: tuple[int, ...] | None = None,
) -> Callable[..., Any]:
    """JIT-compile a function using XLA for accelerator execution.

    Traces the function, builds an XLA computation graph, and compiles it
    for the target backend (CPU/GPU/TPU). Subsequent calls with same-shaped
    inputs reuse the compiled kernel.

    Args:
        fn: Pure function to compile. Must have no side effects.
        static_argnums: Argument indices to treat as compile-time constants.
            Use for arguments that change the computation graph shape
            (e.g., axis parameters, boolean flags).
        donate_argnums: Argument indices whose buffers can be reused for output.
            Eliminates copy overhead when input is no longer needed.

    Returns:
        JIT-compiled version of fn with identical semantics.

    Examples:
        >>> import jax.numpy as jnp
        >>> @jit_compile
        ... def add(x, y):
        ...     return x + y
        >>> result = add(jnp.array(1.0), jnp.array(2.0))
        >>> float(result)
        3.0

        >>> # With static arguments
        >>> @jit_compile
        ... def scale(x, factor=2.0):
        ...     return x * factor
        >>> float(scale(jnp.array(3.0)))
        6.0

    Rust equivalent:
        trueno::jit compiles compute graphs to wgpu shaders or SIMD
        instruction sequences. Static args map to const generics.

    """
    kwargs: dict[str, Any] = {}
    if static_argnums is not None:
        kwargs["static_argnums"] = static_argnums
    if donate_argnums is not None:
        kwargs["donate_argnums"] = donate_argnums
    return jax.jit(fn, **kwargs)


def grad_transform(
    fn: Callable[..., Array],
    *,
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
) -> Callable[..., Array | tuple[Array, Any]]:
    """Compute gradients via reverse-mode automatic differentiation.

    Returns a function that computes the gradient of fn with respect to
    the positional arguments specified by argnums. The original function
    must return a scalar (0-d array).

    Args:
        fn: Scalar-valued function to differentiate. Must return a single
            float value (the loss).
        argnums: Which positional arguments to differentiate with respect to.
            Default 0 (first argument).
        has_aux: If True, fn returns (scalar, aux) and the gradient function
            returns (grad, aux). Use for returning metrics alongside loss.

    Returns:
        Function computing gradients of fn.

    Examples:
        >>> import jax.numpy as jnp
        >>> def mse_loss(w, x, y):
        ...     pred = x @ w
        ...     return jnp.mean((pred - y) ** 2)
        >>> w = jnp.ones((3, 1))
        >>> x = jnp.ones((4, 3))
        >>> y = jnp.zeros((4, 1))
        >>> grads = grad_transform(mse_loss)(w, x, y)
        >>> grads.shape
        (3, 1)

    Rust equivalent:
        entrenar::autograd::backward traverses the computation tape
        in reverse to accumulate gradients. Uses the Wengert list
        (reverse-mode AD) pattern.

    """
    return jax.grad(fn, argnums=argnums, has_aux=has_aux)


def value_and_grad_transform(
    fn: Callable[..., Array],
    *,
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
) -> Callable[..., tuple[Array, Array]]:
    """Compute both the function value and its gradient in one pass.

    More efficient than calling fn and grad(fn) separately — the forward
    pass is shared. Essential for training loops where you need both the
    loss value (for logging) and gradients (for parameter updates).

    Args:
        fn: Scalar-valued function to differentiate.
        argnums: Positional arguments to differentiate.
        has_aux: If True, fn returns (scalar, aux).

    Returns:
        Function returning (value, gradient) tuple.

    Examples:
        >>> import jax.numpy as jnp
        >>> def loss_fn(w, x):
        ...     return jnp.sum(w * x)
        >>> w = jnp.array([1.0, 2.0, 3.0])
        >>> x = jnp.array([4.0, 5.0, 6.0])
        >>> loss, grads = value_and_grad_transform(loss_fn)(w, x)
        >>> float(loss)
        32.0
        >>> grads.tolist()
        [4.0, 5.0, 6.0]

    Rust equivalent:
        entrenar::autograd::value_and_grad runs forward and backward
        passes on the same tape, returning (loss, gradients).

    """
    return jax.value_and_grad(fn, argnums=argnums, has_aux=has_aux)


def vmap_batch(
    fn: Callable[..., Any],
    *,
    in_axes: int | tuple[int | None, ...] = 0,
    out_axes: int | tuple[int, ...] = 0,
) -> Callable[..., Any]:
    """Auto-vectorize a function over a batch dimension.

    Transforms a function that operates on single examples into one that
    operates on batches. JAX compiles this into efficient SIMD/GPU
    vectorized operations — no Python loops.

    Args:
        fn: Function operating on unbatched inputs.
        in_axes: Which axis of each input to map over. Use None to broadcast
            an argument (not batched). Default 0 (first axis is batch).
        out_axes: Where to place the mapped axis in outputs. Default 0.

    Returns:
        Batched version of fn.

    Examples:
        >>> import jax.numpy as jnp
        >>> def dot_product(a, b):
        ...     return jnp.sum(a * b)
        >>> batch_dot = vmap_batch(dot_product)
        >>> a = jnp.ones((8, 3))
        >>> b = jnp.ones((8, 3))
        >>> result = batch_dot(a, b)
        >>> result.shape
        (8,)

        >>> # Broadcast: shared weights, batched inputs
        >>> def linear(w, x):
        ...     return x @ w
        >>> batched_linear = vmap_batch(linear, in_axes=(None, 0))
        >>> w = jnp.ones((3, 2))
        >>> x = jnp.ones((16, 3))
        >>> batched_linear(w, x).shape
        (16, 2)

    Rust equivalent:
        trueno maps vmap to SIMD lane operations: AVX2 (8 lanes),
        AVX-512 (16 lanes), or NEON (4 lanes). For GPU, maps to
        wgpu workgroup dispatches.

    """
    return jax.vmap(fn, in_axes=in_axes, out_axes=out_axes)


def pmap_parallel(
    fn: Callable[..., Any],
    *,
    axis_name: str | None = None,
    devices: list[Any] | None = None,
) -> Callable[..., Any]:
    """Parallelize a function across multiple devices.

    Replicates fn across available devices (GPUs/TPUs) and splits the
    input's leading axis across them. Each device executes fn on its
    shard independently. Use jax.lax.psum/pmean inside fn for
    cross-device communication.

    Args:
        fn: Function to parallelize. First axis of inputs is split
            across devices.
        axis_name: Name for the parallel axis, required for collective
            operations (psum, pmean) inside fn.
        devices: Explicit device list. Default: all available devices.

    Returns:
        Parallelized version of fn.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> # Single-device example (pmap with 1 device = identity)
        >>> @pmap_parallel
        ... def add_one(x):
        ...     return x + 1
        >>> x = jnp.ones((1, 4))  # batch=1 for single device
        >>> result = add_one(x)
        >>> result.shape
        (1, 4)

    Rust equivalent:
        repartir::Pool distributes tasks across CPU cores (Rayon)
        or GPU devices (wgpu). Work-stealing ensures load balance.
        Remote executors extend to multi-machine clusters.

    """
    kwargs: dict[str, Any] = {}
    if axis_name is not None:
        kwargs["axis_name"] = axis_name
    if devices is not None:
        kwargs["devices"] = devices
    return jax.pmap(fn, **kwargs)


def compose_transforms(
    fn: Callable[..., Any],
    *,
    jit: bool = True,
    grad_argnums: int | tuple[int, ...] | None = None,
    vmap_in_axes: int | tuple[int | None, ...] | None = None,
) -> Callable[..., Any]:
    """Compose multiple JAX transforms in the canonical order.

    Applies transforms inside-out: grad → vmap → jit. This ordering
    ensures that:
    1. grad sees scalar loss (before vmap adds batch dim)
    2. vmap vectorizes the gradient computation
    3. jit compiles the entire batched-gradient pipeline

    Args:
        fn: Base function to transform.
        jit: Whether to JIT-compile the result. Default True.
        grad_argnums: If set, wrap fn with grad on these arguments.
        vmap_in_axes: If set, wrap with vmap on these input axes.

    Returns:
        Composed transformation of fn.

    Examples:
        >>> import jax.numpy as jnp
        >>> def loss(w, x):
        ...     return jnp.sum(w * x)
        >>> # grad + jit (no vmap)
        >>> grad_fn = compose_transforms(loss, grad_argnums=0)
        >>> w = jnp.array([1.0, 2.0])
        >>> x = jnp.array([3.0, 4.0])
        >>> grads = grad_fn(w, x)
        >>> grads.tolist()
        [3.0, 4.0]

    Rust equivalent:
        Transform composition in Rust uses trait chaining:
        Jit<Vmap<Grad<F>>> where each layer wraps the inner.

    """
    result = fn
    if grad_argnums is not None:
        result = jax.grad(result, argnums=grad_argnums)
    if vmap_in_axes is not None:
        result = jax.vmap(result, in_axes=vmap_in_axes)
    if jit:
        result = jax.jit(result)
    return result
