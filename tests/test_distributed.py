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
