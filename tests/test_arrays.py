"""Tests for jax_gtc.arrays module."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jax_gtc.arrays import (
    batch_matmul,
    broadcast_add,
    create_array,
    einsum_contract,
    fft_transform,
    index_update,
    matmul,
    reshape_array,
)


class TestCreateArray:
    """Tests for create_array."""

    def test_1d(self):
        a = create_array([1.0, 2.0, 3.0])
        assert a.shape == (3,)
        assert a.dtype == jnp.float32

    def test_2d(self):
        a = create_array([[1.0, 2.0], [3.0, 4.0]])
        assert a.shape == (2, 2)

    def test_explicit_dtype(self):
        a = create_array([1.0, 2.0], dtype=jnp.float32)
        assert a.dtype == jnp.float32

    def test_default_dtype(self):
        a = create_array([1, 2, 3])
        assert a.dtype == jnp.float32


class TestReshapeArray:
    """Tests for reshape_array."""

    def test_basic(self):
        x = jnp.arange(12)
        assert reshape_array(x, (3, 4)).shape == (3, 4)

    def test_infer_dim(self):
        x = jnp.arange(12)
        assert reshape_array(x, (2, -1)).shape == (2, 6)

    def test_preserves_values(self):
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        reshaped = reshape_array(x, (2, 2))
        assert float(reshaped[0, 0]) == 1.0
        assert float(reshaped[1, 1]) == 4.0


class TestMatmul:
    """Tests for matmul."""

    def test_basic(self):
        a = jnp.ones((3, 4))
        b = jnp.ones((4, 2))
        result = matmul(a, b)
        assert result.shape == (3, 2)
        assert jnp.allclose(result, jnp.full((3, 2), 4.0))

    def test_identity(self):
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = matmul(x, jnp.eye(2))
        assert jnp.allclose(result, x)

    @given(
        st.integers(min_value=1, max_value=8),
        st.integers(min_value=1, max_value=8),
        st.integers(min_value=1, max_value=8),
    )
    @settings(max_examples=10)
    def test_shape_property(self, m, k, n):
        """Property: (M,K) @ (K,N) = (M,N)."""
        a = jnp.ones((m, k))
        b = jnp.ones((k, n))
        assert matmul(a, b).shape == (m, n)


class TestBatchMatmul:
    """Tests for batch_matmul."""

    def test_basic(self):
        a = jnp.ones((8, 3, 4))
        b = jnp.ones((8, 4, 2))
        result = batch_matmul(a, b)
        assert result.shape == (8, 3, 2)

    def test_single_batch(self):
        a = jnp.eye(3).reshape(1, 3, 3)
        b = jnp.ones((1, 3, 2))
        result = batch_matmul(a, b)
        assert jnp.allclose(result, b)


class TestEinsumContract:
    """Tests for einsum_contract."""

    def test_matmul(self):
        a = jnp.ones((3, 4))
        b = jnp.ones((4, 5))
        result = einsum_contract("ij,jk->ik", a, b)
        assert result.shape == (3, 5)

    def test_batch_dot(self):
        x = jnp.ones((8, 3))
        y = jnp.ones((8, 3))
        result = einsum_contract("bi,bi->b", x, y)
        assert result.shape == (8,)
        assert jnp.allclose(result, jnp.full(8, 3.0))

    def test_trace(self):
        m = jnp.eye(3)
        result = einsum_contract("ii->", m)
        assert float(result) == 3.0

    def test_outer_product(self):
        a = jnp.ones(3)
        b = jnp.ones(4)
        result = einsum_contract("i,j->ij", a, b)
        assert result.shape == (3, 4)


class TestFftTransform:
    """Tests for fft_transform."""

    def test_basic_shape(self):
        signal = jnp.array([1.0, 0.0, -1.0, 0.0])
        result = fft_transform(signal)
        assert result.shape == (4,)

    def test_dc_component(self):
        signal = jnp.ones(4)
        result = fft_transform(signal)
        assert abs(float(jnp.abs(result[0])) - 4.0) < 1e-5

    def test_custom_axis(self):
        signal = jnp.ones((3, 8))
        result = fft_transform(signal, axis=1)
        assert result.shape == (3, 8)


class TestBroadcastAdd:
    """Tests for broadcast_add."""

    def test_vector_broadcast(self):
        a = jnp.ones((3, 4))
        b = jnp.ones((4,))
        result = broadcast_add(a, b)
        assert result.shape == (3, 4)
        assert jnp.allclose(result, jnp.full((3, 4), 2.0))

    def test_scalar_broadcast(self):
        a = jnp.ones((3, 4))
        b = jnp.array(5.0)
        result = broadcast_add(a, b)
        assert jnp.allclose(result, jnp.full((3, 4), 6.0))


class TestIndexUpdate:
    """Tests for index_update."""

    def test_basic(self):
        x = jnp.zeros(5)
        result = index_update(x, 2, 1.0)
        assert result.tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]

    def test_preserves_others(self):
        x = jnp.array([1.0, 2.0, 3.0])
        result = index_update(x, 1, 99.0)
        assert float(result[0]) == 1.0
        assert float(result[1]) == 99.0
        assert float(result[2]) == 3.0

    def test_original_unchanged(self):
        x = jnp.zeros(3)
        _ = index_update(x, 0, 1.0)
        assert float(x[0]) == 0.0  # immutable
