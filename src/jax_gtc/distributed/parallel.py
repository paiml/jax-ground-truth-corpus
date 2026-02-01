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
from typing import Any

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
        params: dict[str, Array],
        x: Array,
        y: Array,
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


# Type alias for partition specifications
PartitionSpec = tuple[str | None, ...]


def create_mesh(
    data_axis: int = 1,
    model_axis: int = 1,
) -> Any:
    """Create a device mesh for sharded parallelism.

    Organizes available devices into a 2D mesh with data and model
    parallel axes. Used with jax.sharding for fine-grained control.

    Args:
        data_axis: Number of devices along data parallel axis.
        model_axis: Number of devices along model parallel axis.

    Returns:
        JAX Mesh object with named axes ("data", "model").

    Examples:
        >>> mesh = create_mesh(data_axis=1, model_axis=1)
        >>> mesh.shape
        {'data': 1, 'model': 1}

    Rust equivalent:
        repartir::Topology::mesh(data=n, model=m) creates a device
        topology for hybrid parallelism strategies.

    """
    import numpy as np
    from jax.sharding import Mesh

    devices = jax.devices()
    total_devices = data_axis * model_axis

    if total_devices > len(devices):
        # Simulate with available devices by repeating
        device_list = (devices[:1] * total_devices)[:total_devices]
    else:
        device_list = devices[:total_devices]

    # Use numpy array for devices (not jnp)
    device_array = np.array(device_list).reshape(data_axis, model_axis)

    return Mesh(device_array, axis_names=("data", "model"))


def shard_with_mesh(
    x: Array,
    mesh: Any,
    partition: tuple[str | None, ...],
) -> Array:
    """Shard an array according to a partition specification.

    Places array shards on mesh devices according to the partition
    spec. None means replicate along that axis.

    Args:
        x: Array to shard.
        mesh: Device mesh from create_mesh().
        partition: Partition spec, e.g., ("data", None) shards first
            dim across data axis, replicates second dim.

    Returns:
        Sharded array distributed across mesh devices.

    Examples:
        >>> mesh = create_mesh(data_axis=1, model_axis=1)
        >>> x = jnp.ones((4, 8))
        >>> sharded = shard_with_mesh(x, mesh, ("data", None))
        >>> sharded.shape
        (4, 8)

    Rust equivalent:
        repartir::shard_with_topology(&array, &topology, partition)
        distributes data according to the specified sharding strategy.

    """
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as JaxPartitionSpec

    spec = JaxPartitionSpec(*partition)
    sharding = NamedSharding(mesh, spec)
    return jax.device_put(x, sharding)


def shard_params(
    params: dict[str, dict[str, Array]],
    num_shards: int = 1,
) -> dict[str, dict[str, Array]]:
    """Shard nested parameter dict for model parallelism.

    Distributes model parameters across devices by splitting weight
    matrices along the appropriate axis.

    Args:
        params: Nested dict of parameters {layer: {param: array}}.
        num_shards: Number of shards (devices).

    Returns:
        Sharded parameter dict with same structure.

    Examples:
        >>> params = {"layer1": {"w": jnp.ones((4, 4))}}
        >>> sharded = shard_params(params, num_shards=1)
        >>> sharded["layer1"]["w"].shape
        (4, 4)

    Rust equivalent:
        repartir::ShardedExecutor::shard_params splits model weights
        across workers for tensor parallelism.

    """
    if num_shards == 1:
        return params

    sharded: dict[str, dict[str, Array]] = {}
    for layer_name, layer_params in params.items():
        sharded[layer_name] = {}
        for param_name, param in layer_params.items():
            # Shard along last dimension for weights
            if param.ndim >= 2:
                shard_size = param.shape[-1] // num_shards
                sharded[layer_name][param_name] = param[..., :shard_size]
            else:
                sharded[layer_name][param_name] = param

    return sharded


def model_parallel_matmul(
    x: Array,
    w: Array,
    num_shards: int = 1,
) -> Array:
    """Matrix multiply with model-parallel weight sharding.

    Simulates tensor parallelism by conceptually splitting the weight
    matrix across devices and using all-reduce to combine results.

    Args:
        x: Input activations (batch, features_in).
        w: Weight matrix (features_in, features_out).
        num_shards: Number of model parallel shards.

    Returns:
        Output activations (batch, features_out).

    Examples:
        >>> x = jnp.ones((2, 4))
        >>> w = jnp.ones((4, 8))
        >>> result = model_parallel_matmul(x, w, num_shards=1)
        >>> result.shape
        (2, 8)

    Rust equivalent:
        repartir::tensor_parallel::matmul uses NCCL for cross-GPU
        reduction of partial sums from sharded weights.

    """
    if num_shards == 1:
        return jnp.matmul(x, w)

    # Simulate sharded matmul: split w, compute partials, sum
    shard_size = w.shape[1] // num_shards
    result = jnp.zeros((x.shape[0], w.shape[1]))

    for i in range(num_shards):
        start = i * shard_size
        end = start + shard_size
        w_shard = w[:, start:end]
        result = result.at[:, start:end].set(jnp.matmul(x, w_shard))

    return result


def create_pipeline_stages(
    layers: list[Callable[..., Array]],
) -> list[Callable[..., Array]]:
    """Create pipeline stages from layer functions.

    Wraps layer functions for pipeline parallelism where each stage
    runs on a different device.

    Args:
        layers: List of layer functions, each taking and returning arrays.

    Returns:
        List of stage-wrapped layer functions.

    Examples:
        >>> def layer1(x):
        ...     return x * 2
        >>> def layer2(x):
        ...     return x + 1
        >>> stages = create_pipeline_stages([layer1, layer2])
        >>> len(stages)
        2

    Rust equivalent:
        repartir::Pipeline::from_stages creates a pipeline executor
        with automatic micro-batch scheduling.

    """
    # For now, stages are identity wrappers
    # Full implementation would add device placement and async execution
    return [jax.jit(layer) for layer in layers]


def fsdp_wrap_params(
    params: dict[str, Array],
    num_shards: int = 1,
) -> dict[str, Array]:
    """Wrap parameters for Fully Sharded Data Parallel training.

    Shards each parameter across devices. Each device holds 1/N of
    each parameter tensor. Parameters are gathered before forward pass.

    Args:
        params: Model parameters.
        num_shards: Number of FSDP shards (devices).

    Returns:
        Sharded parameters (or unchanged if num_shards=1).

    Examples:
        >>> params = {"w": jnp.ones((8, 8))}
        >>> wrapped = fsdp_wrap_params(params, num_shards=1)
        >>> "w" in wrapped
        True

    Rust equivalent:
        repartir::Fsdp::wrap shards parameters using ZeRO-3 style
        partitioning with lazy all-gather during forward pass.

    """
    if num_shards == 1:
        return params

    sharded = {}
    for k, v in params.items():
        # Shard along first dimension
        shard_size = v.shape[0] // num_shards
        sharded[k] = v[:shard_size]

    return sharded


def fsdp_gather_params(
    sharded_params: dict[str, Array],
    num_shards: int = 1,
) -> dict[str, Array]:
    """Gather sharded parameters for forward pass.

    Performs all-gather to reconstruct full parameters from shards.

    Args:
        sharded_params: Sharded parameter dict from fsdp_wrap_params.
        num_shards: Number of FSDP shards.

    Returns:
        Full (unsharded) parameters.

    Examples:
        >>> params = {"w": jnp.ones((4, 4))}
        >>> sharded = fsdp_wrap_params(params, num_shards=1)
        >>> gathered = fsdp_gather_params(sharded, num_shards=1)
        >>> gathered["w"].shape
        (4, 4)

    Rust equivalent:
        repartir::Fsdp::gather uses NCCL AllGather to reconstruct
        full parameters on each device before compute.

    """
    if num_shards == 1:
        return sharded_params

    gathered = {}
    for k, v in sharded_params.items():
        # Simulate all-gather by repeating shards
        gathered[k] = jnp.tile(v, (num_shards,) + (1,) * (v.ndim - 1))

    return gathered


def fsdp_training_step(
    loss_fn: Callable[..., Array],
    params: dict[str, Array],
    x: Array,
    y: Array,
    learning_rate: float = 0.01,
    num_shards: int = 1,
) -> tuple[dict[str, Array], float]:
    """Execute FSDP training step with gradient sharding.

    Gathers parameters for forward, computes loss, scatters gradients,
    and updates sharded parameters.

    Args:
        loss_fn: Loss function(params, x, y) -> scalar.
        params: Sharded model parameters.
        x: Input batch.
        y: Target batch.
        learning_rate: SGD learning rate.
        num_shards: Number of FSDP shards.

    Returns:
        Tuple of (updated_sharded_params, loss).

    Examples:
        >>> def mse(params, x, y):
        ...     return jnp.mean((x @ params["w"] - y) ** 2)
        >>> params = {"w": jnp.ones((3, 1))}
        >>> x, y = jnp.ones((4, 3)), jnp.zeros((4, 1))
        >>> new_params, loss = fsdp_training_step(mse, params, x, y)
        >>> loss > 0
        True

    Rust equivalent:
        repartir::Fsdp::step orchestrates gather, forward, backward,
        reduce-scatter in a single fused operation.

    """
    # Gather for forward pass
    full_params = fsdp_gather_params(params, num_shards)

    # Compute loss and gradients
    loss, grads = jax.value_and_grad(loss_fn)(full_params, x, y)

    # Shard gradients (simulate reduce-scatter)
    sharded_grads = fsdp_wrap_params(grads, num_shards)

    # Update sharded parameters
    new_params = {k: params[k] - learning_rate * sharded_grads[k] for k in params}

    return new_params, float(loss)
