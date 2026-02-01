"""Multi-device and multi-host parallelism with JAX.

Patterns for data parallelism (pmap), model parallelism (shard_map),
and collective communication.

Rust equivalents:
    pmap → repartir::Pool::map (work-stealing)
    shard_map → repartir::ShardedExecutor
    Mesh → repartir::Topology
    psum → repartir::collective::AllReduce
"""

from jax_gtc.distributed.parallel import (
    all_reduce_sum,
    data_parallel_step,
    get_device_count,
    replicate_params,
    shard_array,
)

__all__ = [
    "get_device_count",
    "replicate_params",
    "shard_array",
    "data_parallel_step",
    "all_reduce_sum",
]
