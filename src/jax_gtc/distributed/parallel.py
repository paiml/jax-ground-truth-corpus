"""Distributed computing recipes.

Patterns for multi-device training and inference with JAX. Covers
data parallelism (pmap), parameter replication, and collective ops.

References:
    - JAX multi-host: https://jax.readthedocs.io/en/latest/multi_process.html
    - JAX sharding: https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html

Rust cross-reference:
    repartir provides CPU (Rayon work-stealing), GPU (wgpu), and
    Remote (TCP/TLS) executors. Topology describes device mesh.

"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array


def get_device_count() -> int:
    """Get the number of available JAX devices.

    Returns:
        Number of devices (GPUs/TPUs/CPUs).

    Examples:
        >>> count = get_device_count()
        >>> count >= 1
        True

    Rust equivalent:
        repartir::Pool::device_count() returns available executors.

    """
    return jax.device_count()


def replicate_params(params: dict[str, Array], num_devices: int | None = None) -> dict[str, Array]:
    """Replicate parameters across devices for data parallelism.

    Creates copies of each parameter on every device. Required
    before using pmap for data-parallel training.

    Args:
        params: Model parameters to replicate.
        num_devices: Number of devices. Default: all available.

    Returns:
        Replicated parameters with leading device dimension.

    Examples:
        >>> import jax.numpy as jnp
        >>> params = {"w": jnp.ones((3, 2))}
        >>> replicated = replicate_params(params, num_devices=1)
        >>> replicated["w"].shape
        (1, 3, 2)

    Rust equivalent:
        repartir broadcasts parameters to all workers via
        repartir::broadcast(&params, &topology).

    """
    if num_devices is None:
        num_devices = jax.device_count()
    return {k: jnp.stack([v] * num_devices) for k, v in params.items()}


def shard_array(x: Array, num_shards: int | None = None) -> Array:
    """Shard an array's leading dimension across devices.

    Splits the batch dimension evenly across available devices
    for data-parallel processing.

    Args:
        x: Array to shard. Leading dimension must be divisible by num_shards.
        num_shards: Number of shards. Default: device_count.

    Returns:
        Reshaped array (num_shards, batch_per_shard, ...).

    Examples:
        >>> import jax.numpy as jnp
        >>> x = jnp.ones((8, 4))
        >>> sharded = shard_array(x, num_shards=2)
        >>> sharded.shape
        (2, 4, 4)

    Rust equivalent:
        repartir::shard(&array, num_workers) splits data for
        distribution across executors.

    """
    if num_shards is None:
        num_shards = jax.device_count()
    batch_size = x.shape[0]
    per_shard = batch_size // num_shards
    return x.reshape(num_shards, per_shard, *x.shape[1:])


def data_parallel_step(
    loss_fn: Callable[..., Array],
    params: dict[str, Array],
    x: Array,
    y: Array,
    learning_rate: float = 0.01,
) -> tuple[dict[str, Array], float]:
    """Execute a data-parallel training step using pmap.

    Splits input batch across devices, computes gradients on each,
    averages gradients with pmean, and applies synchronized updates.

    Args:
        loss_fn: Loss function(params, x, y) -> scalar.
        params: Replicated model parameters (device, ...).
        x: Sharded input batch (device, batch_per_device, ...).
        y: Sharded target batch (device, batch_per_device, ...).
        learning_rate: SGD learning rate.

    Returns:
        Tuple of (updated_params, mean_loss).

    Examples:
        >>> import jax.numpy as jnp
        >>> def mse(params, x, y):
        ...     pred = x @ params["w"]
        ...     return jnp.mean((pred - y) ** 2)
        >>> # Single device example
        >>> params = {"w": jnp.ones((1, 3, 1))}  # 1 device
        >>> x = jnp.ones((1, 4, 3))
        >>> y = jnp.zeros((1, 4, 1))
        >>> new_params, loss = data_parallel_step(mse, params, x, y)
        >>> loss > 0
        True

    Rust equivalent:
        repartir::Pool::map_reduce distributes mini-batches to workers,
        each computes local gradients, then AllReduce averages them.

    """

    def step(
        params: dict[str, Array], x: Array, y: Array,
    ) -> tuple[dict[str, Array], Array]:
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        grads = jax.lax.pmean(grads, axis_name="devices")
        new_params = {k: p - learning_rate * grads[k] for k, p in params.items()}
        return new_params, loss

    step_named = jax.pmap(step, axis_name="devices")

    new_params, losses = step_named(params, x, y)
    mean_loss = float(jnp.mean(losses))
    return new_params, mean_loss


def all_reduce_sum(x: Array, axis_name: str = "devices") -> Array:
    """Sum across all devices (collective operation).

    Must be called inside a pmap with matching axis_name.

    Args:
        x: Local array on each device.
        axis_name: Name of the pmap axis.

    Returns:
        Sum of x across all devices (replicated on each device).

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> @jax.pmap(axis_name="i")
        ... def f(x):
        ...     from jax_gtc.distributed import all_reduce_sum
        ...     return all_reduce_sum(x, axis_name="i")
        >>> x = jnp.array([[1.0, 2.0]])
        >>> result = f(x)
        >>> result.shape
        (1, 2)

    Rust equivalent:
        repartir::collective::AllReduce::sum reduces across workers
        using tree-reduction pattern for O(log n) latency.

    """
    return jax.lax.psum(x, axis_name=axis_name)
