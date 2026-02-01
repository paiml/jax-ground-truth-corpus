"""Optimizer and training loop recipes.

Functional optimizer implementations compatible with JAX's transform model.
All state is explicit — no hidden mutable state.

References:
    - Optax library: https://optax.readthedocs.io/
    - JAX training patterns: https://jax.readthedocs.io/en/latest/notebooks/neural_network_with_tfds.html

Rust cross-reference:
    entrenar::optimizers provides Sgd, Adam, AdamW with identical
    update rules. State is stored in aprender::apr format.

"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class SgdState(NamedTuple):
    """SGD optimizer state (stateless — just learning rate)."""

    learning_rate: float


class AdamState(NamedTuple):
    """Adam optimizer state tracking first/second moments."""

    learning_rate: float
    beta1: float
    beta2: float
    eps: float
    step: int
    m: dict[str, Array]  # First moment estimates
    v: dict[str, Array]  # Second moment estimates


def sgd_optimizer(
    learning_rate: float = 0.01,
) -> tuple[SgdState, Callable[..., tuple[dict[str, Array], SgdState]]]:
    """Create a Stochastic Gradient Descent optimizer.

    Args:
        learning_rate: Step size for parameter updates.

    Returns:
        Tuple of (initial_state, update_fn).
        update_fn(params, grads, state) -> (new_params, new_state)

    Examples:
        >>> import jax.numpy as jnp
        >>> state, update_fn = sgd_optimizer(learning_rate=0.1)
        >>> params = {"w": jnp.array([1.0, 2.0])}
        >>> grads = {"w": jnp.array([0.5, 0.5])}
        >>> new_params, new_state = update_fn(params, grads, state)
        >>> new_params["w"].tolist()
        [0.95, 1.95]

    Rust equivalent:
        entrenar::optimizers::Sgd::new(0.01).step(&mut params, &grads)

    """
    state = SgdState(learning_rate=learning_rate)

    def update(
        params: dict[str, Array],
        grads: dict[str, Array],
        state: SgdState,
    ) -> tuple[dict[str, Array], SgdState]:
        new_params = {k: p - state.learning_rate * grads[k] for k, p in params.items()}
        return new_params, state

    return state, update


def adam_optimizer(
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    param_shapes: dict[str, tuple[int, ...]] | None = None,
) -> tuple[AdamState, Callable[..., tuple[dict[str, Array], AdamState]]]:
    """Create an Adam optimizer (Kingma & Ba, 2014).

    Adaptive learning rates per parameter using first and second moment
    estimates with bias correction.

    Args:
        learning_rate: Base learning rate.
        beta1: Exponential decay rate for first moment (mean).
        beta2: Exponential decay rate for second moment (variance).
        eps: Numerical stability constant.
        param_shapes: Parameter shapes for state initialization.
            If None, state is initialized on first call.

    Returns:
        Tuple of (initial_state, update_fn).

    Examples:
        >>> import jax.numpy as jnp
        >>> shapes = {"w": (3,)}
        >>> state, update_fn = adam_optimizer(
        ...     learning_rate=0.001, param_shapes=shapes
        ... )
        >>> params = {"w": jnp.array([1.0, 2.0, 3.0])}
        >>> grads = {"w": jnp.array([0.1, 0.2, 0.3])}
        >>> new_params, new_state = update_fn(params, grads, state)
        >>> new_params["w"].shape
        (3,)

    Rust equivalent:
        entrenar::optimizers::Adam stores moment vectors in trueno
        tensors. Bias correction is fused into the update step
        for efficiency.

    """
    if param_shapes is None:
        param_shapes = {}

    m = {k: jnp.zeros(s) for k, s in param_shapes.items()}
    v = {k: jnp.zeros(s) for k, s in param_shapes.items()}

    state = AdamState(
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        step=0,
        m=m,
        v=v,
    )

    def update(
        params: dict[str, Array],
        grads: dict[str, Array],
        state: AdamState,
    ) -> tuple[dict[str, Array], AdamState]:
        step = state.step + 1
        new_m = {}
        new_v = {}
        new_params = {}

        for k in params:
            g = grads[k]
            m_prev = state.m.get(k, jnp.zeros_like(g))
            v_prev = state.v.get(k, jnp.zeros_like(g))

            new_m[k] = state.beta1 * m_prev + (1 - state.beta1) * g
            new_v[k] = state.beta2 * v_prev + (1 - state.beta2) * g**2

            m_hat = new_m[k] / (1 - state.beta1**step)
            v_hat = new_v[k] / (1 - state.beta2**step)

            new_params[k] = params[k] - state.learning_rate * m_hat / (jnp.sqrt(v_hat) + state.eps)

        new_state = AdamState(
            learning_rate=state.learning_rate,
            beta1=state.beta1,
            beta2=state.beta2,
            eps=state.eps,
            step=step,
            m=new_m,
            v=new_v,
        )
        return new_params, new_state

    return state, update


def training_step(
    params: dict[str, Array],
    x: Array,
    y: Array,
    loss_fn: Callable[..., Array],
    optimizer_update: Callable[..., tuple[dict[str, Array], Any]],
    optimizer_state: Any,
) -> tuple[dict[str, Array], Any, float]:
    """Execute one training step: forward → backward → update.

    A pure function suitable for jit compilation.

    Args:
        params: Model parameters.
        x: Input batch.
        y: Target batch.
        loss_fn: Loss function(params, x, y) -> scalar.
        optimizer_update: Optimizer update function.
        optimizer_state: Current optimizer state.

    Returns:
        Tuple of (new_params, new_optimizer_state, loss_value).

    Examples:
        >>> import jax.numpy as jnp
        >>> def mse(params, x, y):
        ...     pred = x @ params["w"]
        ...     return jnp.mean((pred - y) ** 2)
        >>> state, update_fn = sgd_optimizer(0.01)
        >>> params = {"w": jnp.ones((3, 1))}
        >>> x = jnp.ones((4, 3))
        >>> y = jnp.zeros((4, 1))
        >>> new_params, new_state, loss = training_step(
        ...     params, x, y, mse, update_fn, state
        ... )
        >>> loss > 0
        True

    Rust equivalent:
        entrenar::Trainer::step runs forward (realizar inference),
        backward (entrenar autograd), and optimizer update in a
        single fused operation.

    """
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    new_params, new_state = optimizer_update(params, grads, optimizer_state)
    return new_params, new_state, float(loss)


def cosine_schedule(
    base_lr: float,
    step: int,
    total_steps: int,
    min_lr: float = 0.0,
) -> float:
    """Cosine annealing learning rate schedule.

    Smoothly decays learning rate from base_lr to min_lr over
    total_steps using a cosine curve.

    Args:
        base_lr: Initial learning rate.
        step: Current training step.
        total_steps: Total number of training steps.
        min_lr: Minimum learning rate at end.

    Returns:
        Learning rate for the current step.

    Examples:
        >>> cosine_schedule(0.1, 0, 100)
        0.1
        >>> cosine_schedule(0.1, 100, 100)
        0.0
        >>> 0.04 < cosine_schedule(0.1, 50, 100) < 0.06
        True

    Rust equivalent:
        entrenar::schedulers::CosineAnnealing::get_lr(step)

    """
    import math

    if step >= total_steps:
        return min_lr
    progress = step / total_steps
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
