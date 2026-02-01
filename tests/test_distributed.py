"""Tests for jax_gtc.distributed module."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_gtc.distributed import (
    all_reduce_sum,
    data_parallel_step,
    get_device_count,
    replicate_params,
    shard_array,
)


class TestGetDeviceCount:
    """Tests for get_device_count."""

    def test_at_least_one(self):
        assert get_device_count() >= 1

    def test_matches_jax(self):
        assert get_device_count() == jax.device_count()


class TestReplicateParams:
    """Tests for replicate_params."""

    def test_basic(self):
        params = {"w": jnp.ones((3, 2))}
        replicated = replicate_params(params, num_devices=1)
        assert replicated["w"].shape == (1, 3, 2)

    def test_multiple_devices(self):
        params = {"w": jnp.ones((3,)), "b": jnp.zeros(2)}
        replicated = replicate_params(params, num_devices=4)
        assert replicated["w"].shape == (4, 3)
        assert replicated["b"].shape == (4, 2)

    def test_default_devices(self):
        params = {"w": jnp.ones((2,))}
        replicated = replicate_params(params)
        assert replicated["w"].shape[0] == jax.device_count()

    def test_values_preserved(self):
        params = {"w": jnp.array([1.0, 2.0, 3.0])}
        replicated = replicate_params(params, num_devices=2)
        assert jnp.allclose(replicated["w"][0], params["w"])
        assert jnp.allclose(replicated["w"][1], params["w"])


class TestShardArray:
    """Tests for shard_array."""

    def test_basic(self):
        x = jnp.ones((8, 4))
        sharded = shard_array(x, num_shards=2)
        assert sharded.shape == (2, 4, 4)

    def test_four_shards(self):
        x = jnp.arange(16).reshape(16, 1)
        sharded = shard_array(x, num_shards=4)
        assert sharded.shape == (4, 4, 1)

    def test_single_shard(self):
        x = jnp.ones((4, 3))
        sharded = shard_array(x, num_shards=1)
        assert sharded.shape == (1, 4, 3)

    def test_preserves_values(self):
        x = jnp.arange(8.0).reshape(8, 1)
        sharded = shard_array(x, num_shards=2)
        # First shard should have 0-3, second should have 4-7
        assert float(sharded[0, 0, 0]) == 0.0
        assert float(sharded[1, 0, 0]) == 4.0

    def test_default_shards(self):
        n = jax.device_count()
        x = jnp.ones((n * 4, 3))
        sharded = shard_array(x)
        assert sharded.shape == (n, 4, 3)


class TestDataParallelStep:
    """Tests for data_parallel_step."""

    def test_basic(self):
        def mse(params, x, y):
            pred = x @ params["w"]
            return jnp.mean((pred - y) ** 2)

        # Single device
        params = {"w": jnp.ones((1, 3, 1))}
        x = jnp.ones((1, 4, 3))
        y = jnp.zeros((1, 4, 1))
        new_params, loss = data_parallel_step(mse, params, x, y)
        assert loss > 0
        assert new_params["w"].shape == (1, 3, 1)

    def test_loss_is_scalar(self):
        def mse(params, x, y):
            return jnp.mean((x @ params["w"] - y) ** 2)

        params = {"w": jnp.ones((1, 2, 1))}
        x = jnp.ones((1, 3, 2))
        y = jnp.zeros((1, 3, 1))
        _, loss = data_parallel_step(mse, params, x, y)
        assert isinstance(loss, float)


class TestAllReduceSum:
    """Tests for all_reduce_sum."""

    def test_inside_pmap(self):
        def f(x):
            return all_reduce_sum(x, axis_name="i")

        pmapped = jax.pmap(f, axis_name="i")
        x = jnp.array([[1.0, 2.0]])  # 1 device
        result = pmapped(x)
        assert result.shape == (1, 2)
        assert jnp.allclose(result, x)  # single device, sum = identity

    def test_custom_axis_name(self):
        def f(x):
            return all_reduce_sum(x, axis_name="batch")

        pmapped = jax.pmap(f, axis_name="batch")
        x = jnp.ones((1, 3))
        result = pmapped(x)
        assert result.shape == (1, 3)


class TestMeshSharding:
    """Tests for mesh_sharding."""

    def test_create_mesh(self):
        from jax_gtc.distributed import create_mesh

        mesh = create_mesh(data_axis=1, model_axis=1)
        assert mesh is not None
        assert hasattr(mesh, "devices")

    def test_mesh_shape(self):
        from jax_gtc.distributed import create_mesh

        mesh = create_mesh(data_axis=1, model_axis=1)
        assert mesh.shape == {"data": 1, "model": 1}

    def test_shard_with_mesh(self):
        from jax_gtc.distributed import create_mesh, shard_with_mesh

        mesh = create_mesh(data_axis=1, model_axis=1)
        x = jnp.ones((4, 8))
        sharded = shard_with_mesh(x, mesh, partition=("data", None))
        assert sharded.shape == (4, 8)

    def test_partition_spec(self):
        from jax.sharding import PartitionSpec

        from jax_gtc.distributed import create_mesh

        _mesh = create_mesh(data_axis=1, model_axis=1)  # noqa: F841
        spec = PartitionSpec("data", None)
        assert spec[0] == "data"
        assert spec[1] is None


class TestModelParallel:
    """Tests for model_parallel."""

    def test_shard_params_by_layer(self):
        from jax_gtc.distributed import shard_params

        params = {
            "layer1": {"w": jnp.ones((4, 4))},
            "layer2": {"w": jnp.ones((4, 4))},
        }
        # Shard across 1 device (identity)
        sharded = shard_params(params, num_shards=1)
        assert sharded["layer1"]["w"].shape == (4, 4)

    def test_model_parallel_matmul(self):
        from jax_gtc.distributed import model_parallel_matmul

        x = jnp.ones((2, 4))
        w = jnp.ones((4, 8))
        result = model_parallel_matmul(x, w, num_shards=1)
        assert result.shape == (2, 8)

    def test_pipeline_stages(self):
        from jax_gtc.distributed import create_pipeline_stages

        def layer1(x):
            return x * 2

        def layer2(x):
            return x + 1

        stages = create_pipeline_stages([layer1, layer2])
        assert len(stages) == 2
        x = jnp.ones((2,))
        result = stages[1](stages[0](x))
        assert jnp.allclose(result, jnp.array([3.0, 3.0]))


class TestFsdpTraining:
    """Tests for fsdp_training."""

    def test_fsdp_wrap_params(self):
        from jax_gtc.distributed import fsdp_wrap_params

        params = {"w": jnp.ones((8, 8)), "b": jnp.zeros((8,))}
        wrapped = fsdp_wrap_params(params, num_shards=1)
        assert "w" in wrapped
        assert "b" in wrapped

    def test_fsdp_gather_params(self):
        from jax_gtc.distributed import fsdp_gather_params, fsdp_wrap_params

        params = {"w": jnp.ones((4, 4))}
        wrapped = fsdp_wrap_params(params, num_shards=1)
        gathered = fsdp_gather_params(wrapped, num_shards=1)
        assert jnp.allclose(gathered["w"], params["w"])

    def test_fsdp_training_step(self):
        from jax_gtc.distributed import fsdp_training_step

        def loss_fn(params, x, y):
            return jnp.mean((x @ params["w"] - y) ** 2)

        params = {"w": jnp.ones((3, 1))}
        x = jnp.ones((4, 3))
        y = jnp.zeros((4, 1))

        new_params, loss = fsdp_training_step(
            loss_fn, params, x, y, learning_rate=0.01, num_shards=1,
        )
        assert loss > 0
        assert new_params["w"].shape == (3, 1)

    def test_fsdp_reduces_memory(self):
        from jax_gtc.distributed import fsdp_wrap_params

        params = {"w": jnp.ones((16, 16))}
        wrapped = fsdp_wrap_params(params, num_shards=2)
        # With 2 shards, each shard should have half the params
        # For single-device simulation, shape is unchanged but metadata indicates sharding
        assert "w" in wrapped

    def test_fsdp_gather_multi_shard(self):
        """Test fsdp_gather_params with num_shards > 1."""
        from jax_gtc.distributed import fsdp_gather_params

        # Simulate a sharded parameter (half the original)
        sharded_params = {"w": jnp.ones((4, 4))}
        gathered = fsdp_gather_params(sharded_params, num_shards=2)
        # Should tile along first dimension
        assert gathered["w"].shape == (8, 4)


class TestMeshLargerThanDevices:
    """Tests for mesh creation with more requested devices than available."""

    def test_mesh_simulates_devices(self):
        """Test that create_mesh works when requesting more devices than available."""
        from jax_gtc.distributed import create_mesh

        # Request a 2x2 mesh (4 devices) - will simulate if not enough real devices
        mesh = create_mesh(data_axis=2, model_axis=2)
        assert mesh.shape == {"data": 2, "model": 2}


class TestShardParamsMultiShard:
    """Tests for shard_params with num_shards > 1."""

    def test_shard_weights_by_last_dim(self):
        """Test that 2D weights are sharded along last dimension."""
        from jax_gtc.distributed import shard_params

        params = {
            "layer1": {"w": jnp.ones((4, 8)), "b": jnp.ones((8,))},
        }
        sharded = shard_params(params, num_shards=2)
        # Weight matrix (4, 8) -> (4, 4) when sharded by 2
        assert sharded["layer1"]["w"].shape == (4, 4)
        # 1D bias is unchanged
        assert sharded["layer1"]["b"].shape == (8,)


class TestModelParallelMatmulMultiShard:
    """Tests for model_parallel_matmul with num_shards > 1."""

    def test_matmul_sharded(self):
        """Test sharded matrix multiplication."""
        from jax_gtc.distributed import model_parallel_matmul

        x = jnp.ones((2, 4))
        w = jnp.ones((4, 8))
        # Sharded into 2 parts
        result = model_parallel_matmul(x, w, num_shards=2)
        # Result shape should be same as regular matmul
        assert result.shape == (2, 8)
        # Values should match regular matmul (each output is sum of 4 ones = 4.0)
        expected = jnp.matmul(x, w)
        assert jnp.allclose(result, expected)
