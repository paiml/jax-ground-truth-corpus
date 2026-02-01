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


def gradient_accumulation(
    loss_fn: Callable[..., Array],
    params: dict[str, Array],
    batches: list[tuple[Array, Array]],
) -> dict[str, Array]:
    """Accumulate gradients over multiple micro-batches.

    Useful for simulating larger batch sizes when memory-constrained.
    Computes gradient for each micro-batch and averages them.

    Args:
        loss_fn: Loss function(params, x, y) -> scalar.
        params: Model parameters.
        batches: List of (x, y) micro-batch tuples.

    Returns:
        Averaged gradients across all micro-batches.

    Raises:
        ValueError: If batches is empty.

    Examples:
        >>> import jax.numpy as jnp
        >>> def mse(params, x, y):
        ...     return jnp.mean((x @ params["w"] - y) ** 2)
        >>> params = {"w": jnp.ones((2, 1))}
        >>> batches = [
        ...     (jnp.ones((2, 2)), jnp.zeros((2, 1))),
        ...     (jnp.ones((2, 2)), jnp.zeros((2, 1))),
        ... ]
        >>> grads = gradient_accumulation(mse, params, batches)
        >>> grads["w"].shape
        (2, 1)

    Rust equivalent:
        entrenar::GradAccumulator stores running gradient sums and
        divides by count after all micro-batches are processed.

    """
    if not batches:
        msg = "batches must contain at least one batch"
        raise ValueError(msg)

    accumulated: dict[str, Array] = {}
    num_batches = len(batches)

    for x, y in batches:
        grads = jax.grad(loss_fn)(params, x, y)
        for k, g in grads.items():
            if k in accumulated:
                accumulated[k] = accumulated[k] + g
            else:
                accumulated[k] = g

    # Average the gradients
    return {k: g / num_batches for k, g in accumulated.items()}


def checkpoint_save(
    params: dict[str, Array],
    path: str | Any,
    optimizer_state: Any | None = None,
) -> None:
    """Save model checkpoint to disk.

    Serializes parameters (and optionally optimizer state) to a
    NumPy .npz file for persistence.

    Args:
        params: Model parameters to save.
        path: File path for the checkpoint.
        optimizer_state: Optional optimizer state to include.

    Examples:
        >>> import jax.numpy as jnp
        >>> import tempfile
        >>> params = {"w": jnp.array([1.0, 2.0])}
        >>> with tempfile.NamedTemporaryFile(suffix=".npz") as f:
        ...     checkpoint_save(params, f.name)

    Rust equivalent:
        pacha::ModelRegistry::save persists models in APR format
        with Ed25519 signatures for integrity verification.

    """
    from pathlib import Path

    import numpy as np

    save_dict = {f"param_{k}": np.asarray(v) for k, v in params.items()}

    if optimizer_state is not None:
        # Handle NamedTuple optimizer states
        save_dict["__has_optimizer__"] = np.array([1])
        if hasattr(optimizer_state, "_asdict"):
            state_dict = optimizer_state._asdict()
            for k, v in state_dict.items():
                if isinstance(v, dict):
                    for sk, sv in v.items():
                        save_dict[f"opt_{k}_{sk}"] = np.asarray(sv)
                elif isinstance(v, (int, float)):
                    save_dict[f"opt_{k}"] = np.array([v])
                else:
                    save_dict[f"opt_{k}"] = np.asarray(v)

    np.savez(str(Path(path)), **save_dict)  # type: ignore[arg-type]


def checkpoint_load(
    path: str | Any,
    load_optimizer: bool = False,
) -> dict[str, Array] | tuple[dict[str, Array], Any]:
    """Load model checkpoint from disk.

    Deserializes parameters (and optionally optimizer state) from
    a NumPy .npz file.

    Args:
        path: File path to the checkpoint.
        load_optimizer: Whether to also load optimizer state.

    Returns:
        If load_optimizer is False: dict of parameters.
        If load_optimizer is True: tuple of (params, optimizer_state).

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.

    Examples:
        >>> import jax.numpy as jnp
        >>> import tempfile
        >>> params = {"w": jnp.array([1.0, 2.0])}
        >>> with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        ...     checkpoint_save(params, f.name)
        ...     loaded = checkpoint_load(f.name)
        >>> loaded["w"].tolist()
        [1.0, 2.0]

    Rust equivalent:
        pacha::ModelRegistry::load verifies signatures and loads
        APR format models into trueno tensors.

    """
    from pathlib import Path

    import numpy as np

    file_path = Path(path)
    if not file_path.exists():
        msg = f"Checkpoint not found: {path}"
        raise FileNotFoundError(msg)

    data = np.load(file_path)

    # Extract parameters
    params = {}
    for k in data.files:
        if k.startswith("param_"):
            param_name = k[6:]  # Remove "param_" prefix
            params[param_name] = jnp.array(data[k])

    if not load_optimizer:
        return params

    # Reconstruct optimizer state
    has_optimizer = "__has_optimizer__" in data.files

    if not has_optimizer:
        return params, None

    # Extract optimizer state fields
    opt_fields: dict[str, Any] = {}
    m_dict: dict[str, Array] = {}
    v_dict: dict[str, Array] = {}

    for k in data.files:
        if k.startswith("opt_"):
            field_name = k[4:]  # Remove "opt_" prefix
            if field_name.startswith("m_"):
                m_dict[field_name[2:]] = jnp.array(data[k])
            elif field_name.startswith("v_"):
                v_dict[field_name[2:]] = jnp.array(data[k])
            else:
                val = data[k]
                if val.shape == (1,):
                    opt_fields[field_name] = float(val[0]) if "." in str(val[0]) else int(val[0])
                else:
                    opt_fields[field_name] = jnp.array(val)

    # Reconstruct AdamState
    state = AdamState(
        learning_rate=opt_fields.get("learning_rate", 0.001),
        beta1=opt_fields.get("beta1", 0.9),
        beta2=opt_fields.get("beta2", 0.999),
        eps=opt_fields.get("eps", 1e-8),
        step=int(opt_fields.get("step", 0)),
        m=m_dict,
        v=v_dict,
    )

    return params, state
