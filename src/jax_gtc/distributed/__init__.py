"""Multi-device and multi-host parallelism with JAX.

Patterns for data parallelism (pmap), model parallelism (shard_map),
and collective communication.

Rust equivalents:
    pmap → repartir::Pool::map (work-stealing)
    shard_map → repartir::ShardedExecutor
    Mesh → repartir::Topology
    psum → repartir::collective::AllReduce
    fsdp → repartir::Fsdp (ZeRO-3 style sharding)
"""

from jax_gtc.distributed.parallel import (
    PartitionSpec,
    all_reduce_sum,
    create_mesh,
    create_pipeline_stages,
    data_parallel_step,
    fsdp_gather_params,
    fsdp_training_step,
    fsdp_wrap_params,
    get_device_count,
    model_parallel_matmul,
    replicate_params,
    shard_array,
    shard_params,
    shard_with_mesh,
)

__all__ = [
    "get_device_count",
    "replicate_params",
    "shard_array",
    "data_parallel_step",
    "all_reduce_sum",
    "create_mesh",
    "shard_with_mesh",
    "PartitionSpec",
    "shard_params",
    "model_parallel_matmul",
    "create_pipeline_stages",
    "fsdp_wrap_params",
    "fsdp_gather_params",
    "fsdp_training_step",
]
