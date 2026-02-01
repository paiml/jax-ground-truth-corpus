"""Tests for jax_gtc.random module."""

from __future__ import annotations

import jax.numpy as jnp
from hypothesis import given, settings
from hypothesis import strategies as st

from jax_gtc.random import (
    categorical_sample,
    create_key,
    normal_sample,
    split_key,
    uniform_sample,
)


class TestCreateKey:
    """Tests for create_key."""

    def test_basic(self):
        key = create_key(42)
        assert key.shape == ()

    def test_deterministic(self):
        k1 = create_key(0)
        k2 = create_key(0)
        assert jnp.array_equal(k1, k2)

    def test_different_seeds(self):
        k1 = create_key(0)
        k2 = create_key(1)
        assert not jnp.array_equal(k1, k2)


class TestSplitKey:
    """Tests for split_key."""

    def test_basic_split(self):
        key = create_key(0)
        subkeys = split_key(key, 3)
        assert subkeys.shape == (3,)

    def test_default_two(self):
        key = create_key(0)
        subkeys = split_key(key)
        assert subkeys.shape == (2,)

    def test_keys_are_different(self):
        key = create_key(42)
        subkeys = split_key(key, 4)
        # Each subkey should be unique
        for i in range(4):
            for j in range(i + 1, 4):
                assert not jnp.array_equal(subkeys[i], subkeys[j])

    def test_unpacking_pattern(self):
        key = create_key(0)
        key, subkey = split_key(key)
        assert key.shape == ()
        assert subkey.shape == ()


class TestNormalSample:
    """Tests for normal_sample."""

    def test_shape(self):
        key = create_key(42)
        samples = normal_sample(key, (1000,))
        assert samples.shape == (1000,)

    def test_approximate_mean(self):
        key = create_key(0)
        samples = normal_sample(key, (10000,))
        assert abs(float(jnp.mean(samples))) < 0.1

    def test_approximate_std(self):
        key = create_key(0)
        samples = normal_sample(key, (10000,))
        assert abs(float(jnp.std(samples)) - 1.0) < 0.1

    def test_dtype(self):
        key = create_key(0)
        samples = normal_sample(key, (10,), dtype=jnp.float32)
        assert samples.dtype == jnp.float32

    def test_multidim_shape(self):
        key = create_key(0)
        samples = normal_sample(key, (3, 4, 5))
        assert samples.shape == (3, 4, 5)

    def test_deterministic(self):
        s1 = normal_sample(create_key(42), (100,))
        s2 = normal_sample(create_key(42), (100,))
        assert jnp.allclose(s1, s2)


class TestUniformSample:
    """Tests for uniform_sample."""

    def test_shape(self):
        key = create_key(42)
        samples = uniform_sample(key, (100,))
        assert samples.shape == (100,)

    def test_default_range(self):
        key = create_key(0)
        samples = uniform_sample(key, (10000,))
        assert float(jnp.min(samples)) >= 0.0
        assert float(jnp.max(samples)) < 1.0

    def test_custom_range(self):
        key = create_key(0)
        samples = uniform_sample(key, (10000,), minval=-1.0, maxval=1.0)
        assert float(jnp.min(samples)) >= -1.0
        assert float(jnp.max(samples)) < 1.0

    def test_dtype(self):
        key = create_key(0)
        samples = uniform_sample(key, (10,), dtype=jnp.float32)
        assert samples.dtype == jnp.float32

    @given(st.integers(min_value=0, max_value=1000))
    @settings(max_examples=10)
    def test_in_range_property(self, seed):
        """Property: all samples in [0, 1)."""
        key = create_key(seed)
        samples = uniform_sample(key, (100,))
        assert float(jnp.min(samples)) >= 0.0
        assert float(jnp.max(samples)) < 1.0


class TestCategoricalSample:
    """Tests for categorical_sample."""

    def test_peaked_distribution(self):
        key = create_key(42)
        logits = jnp.array([0.0, 0.0, 100.0])
        sample = categorical_sample(key, logits)
        assert int(sample) == 2

    def test_multiple_samples(self):
        key = create_key(0)
        logits = jnp.array([0.0, 0.0, 100.0])
        samples = categorical_sample(key, logits, num_samples=10)
        assert samples.shape == (10,)
        # All should be 2 given the peaked distribution
        assert jnp.all(samples == 2)

    def test_single_sample(self):
        key = create_key(0)
        logits = jnp.array([10.0, 0.0])
        sample = categorical_sample(key, logits)
        assert sample.shape == ()

    def test_valid_indices(self):
        key = create_key(42)
        logits = jnp.ones(5)
        samples = categorical_sample(key, logits, num_samples=100)
        assert float(jnp.min(samples)) >= 0
        assert float(jnp.max(samples)) < 5
