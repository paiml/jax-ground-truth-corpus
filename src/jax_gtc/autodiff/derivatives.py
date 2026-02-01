"""Automatic differentiation recipes.

Covers reverse-mode (grad), forward-mode (jvp), Jacobians, Hessians,
and custom derivative rules. All functions are pure and composable
with other JAX transforms.

References:
    - JAX autodiff cookbook: https://jax.readthedocs.io/en/latest/advanced-autodiff.html
    - JAX source: jax/_src/api.py (grad, jacobian, hessian)

Rust cross-reference:
    entrenar::autograd implements reverse-mode AD with a Wengert list
    (computation tape). Forward-mode uses dual numbers.

"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array


def gradient(
    fn: Callable[..., Array],
    argnums: int | tuple[int, ...] = 0,
) -> Callable[..., Array]:
    """Compute gradients via reverse-mode AD.

    Args:
        fn: Scalar-valued differentiable function.
        argnums: Which arguments to differentiate.

    Returns:
        Gradient function.

    Examples:
        >>> import jax.numpy as jnp
        >>> def f(x):
        ...     return jnp.sum(x ** 2)
        >>> grad_f = gradient(f)
        >>> grad_f(jnp.array([1.0, 2.0, 3.0])).tolist()
        [2.0, 4.0, 6.0]

    """
    return jax.grad(fn, argnums=argnums)


def value_and_gradient(
    fn: Callable[..., Array],
    argnums: int | tuple[int, ...] = 0,
) -> Callable[..., tuple[Array, Array]]:
    """Compute value and gradient simultaneously.

    Shares the forward pass between value and gradient computation.

    Args:
        fn: Scalar-valued function.
        argnums: Which arguments to differentiate.

    Returns:
        Function returning (value, gradient) tuple.

    Examples:
        >>> import jax.numpy as jnp
        >>> def f(x):
        ...     return jnp.sum(x ** 2)
        >>> val, grad = value_and_gradient(f)(jnp.array([3.0]))
        >>> float(val)
        9.0
        >>> float(grad[0])
        6.0

    """
    return jax.value_and_grad(fn, argnums=argnums)


def jacobian_forward(fn: Callable[..., Array], argnums: int = 0) -> Callable[..., Array]:
    """Compute the Jacobian matrix using forward-mode AD.

    Forward-mode is efficient when the input dimension is small relative
    to the output dimension (one forward pass per input dimension).

    Args:
        fn: Vector-valued function R^n → R^m.
        argnums: Argument to differentiate.

    Returns:
        Function computing the m×n Jacobian matrix.

    Examples:
        >>> import jax.numpy as jnp
        >>> def f(x):
        ...     return jnp.array([x[0] ** 2, x[0] * x[1], x[1] ** 3])
        >>> J = jacobian_forward(f)(jnp.array([2.0, 3.0]))
        >>> J.shape
        (3, 2)

    Rust equivalent:
        entrenar::autograd::jacobian with Mode::Forward uses dual
        numbers for efficient column-wise Jacobian computation.

    """
    return jax.jacfwd(fn, argnums=argnums)


def jacobian_reverse(fn: Callable[..., Array], argnums: int = 0) -> Callable[..., Array]:
    """Compute the Jacobian matrix using reverse-mode AD.

    Reverse-mode is efficient when the output dimension is small relative
    to the input dimension (one backward pass per output dimension).

    Args:
        fn: Vector-valued function R^n → R^m.
        argnums: Argument to differentiate.

    Returns:
        Function computing the m×n Jacobian matrix.

    Examples:
        >>> import jax.numpy as jnp
        >>> def f(x):
        ...     return jnp.array([x[0] + x[1], x[0] * x[1]])
        >>> J = jacobian_reverse(f)(jnp.array([2.0, 3.0]))
        >>> J.shape
        (2, 2)

    Rust equivalent:
        entrenar::autograd::jacobian with Mode::Reverse uses VJPs
        for efficient row-wise Jacobian computation.

    """
    return jax.jacrev(fn, argnums=argnums)


def hessian_matrix(fn: Callable[..., Array], argnums: int = 0) -> Callable[..., Array]:
    """Compute the Hessian (second derivative matrix).

    Composed as jacfwd(jacrev(fn)) — reverse-mode for the inner gradient,
    forward-mode for the outer. This is the most efficient ordering for
    scalar-valued functions.

    Args:
        fn: Scalar-valued function R^n → R.
        argnums: Argument to differentiate.

    Returns:
        Function computing the n×n Hessian matrix.

    Examples:
        >>> import jax.numpy as jnp
        >>> def f(x):
        ...     return x[0] ** 2 + x[0] * x[1] + x[1] ** 2
        >>> H = hessian_matrix(f)(jnp.array([1.0, 1.0]))
        >>> H.tolist()
        [[2.0, 1.0], [1.0, 2.0]]

    Rust equivalent:
        entrenar::autograd::hessian composes forward and reverse
        passes. For large n, uses Hessian-vector products instead
        of materializing the full matrix.

    """
    return jax.hessian(fn, argnums=argnums)


def custom_vjp_rule(
    fwd_fn: Callable[..., tuple[Array, Any]],
    bwd_fn: Callable[..., tuple[Array, ...]],
    fn: Callable[..., Array],
) -> Callable[..., Array]:
    """Define a custom vector-Jacobian product (backward pass) rule.

    Use when:
    - The automatic derivative is numerically unstable
    - You want to checkpoint (recompute vs store) intermediates
    - The function calls non-JAX code (FFI, custom CUDA kernels)

    Args:
        fwd_fn: Forward pass returning (output, residuals). Residuals
            are saved for the backward pass.
        bwd_fn: Backward pass taking (residuals, grad_output) and
            returning gradients for each input.
        fn: Original function to wrap.

    Returns:
        Function with custom backward pass.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> @jax.custom_vjp
        ... def safe_log(x):
        ...     return jnp.log(x)
        >>> def safe_log_fwd(x):
        ...     return safe_log(x), (x,)
        >>> def safe_log_bwd(res, g):
        ...     (x,) = res
        ...     return (g / jnp.maximum(x, 1e-8),)
        >>> safe_log.defvjp(safe_log_fwd, safe_log_bwd)
        >>> grad_safe_log = jax.grad(safe_log)
        >>> float(grad_safe_log(jnp.array(2.0)))
        0.5

    Rust equivalent:
        In entrenar, implement the Backward trait for custom ops:
        fn backward(&self, grad: &Tensor) -> Vec<Tensor>

    """
    wrapped = jax.custom_vjp(fn)
    wrapped.defvjp(fwd_fn, bwd_fn)
    return wrapped


def stop_gradient(x: Array) -> Array:
    """Stop gradient propagation through a value.

    The value is used in the forward pass but treated as a constant
    in the backward pass (gradient is zero). Useful for:
    - Target networks (DQN, actor-critic)
    - Straight-through estimators
    - Detaching computed values from the graph

    Args:
        x: Array to detach from gradient computation.

    Returns:
        Same value as x, but with zero gradient.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> def f(x):
        ...     return x * jax.lax.stop_gradient(x)
        >>> # d/dx [x * stop_grad(x)] = stop_grad(x) = x (as constant)
        >>> grad_f = jax.grad(f)
        >>> float(grad_f(jnp.array(3.0)))
        3.0

    Rust equivalent:
        entrenar::autograd::detach() removes a tensor from the
        computation graph, returning a leaf tensor.

    """
    return jax.lax.stop_gradient(x)


def gradient_check(
    fn: Callable[..., Array],
    args: tuple[Array, ...],
    *,
    eps: float = 1e-4,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> bool:
    """Verify gradients using finite differences.

    Compares analytical gradients (from jax.grad) against numerical
    gradients (central differences). Use in tests to validate custom
    derivative rules.

    Args:
        fn: Scalar-valued function to check.
        args: Input arguments to test at.
        eps: Finite difference step size.
        atol: Absolute tolerance for comparison.
        rtol: Relative tolerance for comparison.

    Returns:
        True if analytical and numerical gradients match within tolerance.

    Examples:
        >>> import jax.numpy as jnp
        >>> def f(x):
        ...     return jnp.sum(jnp.sin(x))
        >>> gradient_check(f, (jnp.array([1.0, 2.0]),))
        True

    """
    analytical_grad = jax.grad(fn)(*args)

    x = args[0]
    numerical_grad = jnp.zeros_like(x)
    for i in range(x.size):
        x_plus = x.at[i].add(eps)
        x_minus = x.at[i].add(-eps)
        numerical_grad = numerical_grad.at[i].set(
            (fn(x_plus, *args[1:]) - fn(x_minus, *args[1:])) / (2 * eps)
        )

    return bool(jnp.allclose(analytical_grad, numerical_grad, atol=atol, rtol=rtol))
