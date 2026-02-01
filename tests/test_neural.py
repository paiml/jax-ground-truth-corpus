"""Tests for jax_gtc.neural module."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jax_gtc.neural import (
    attention_scores,
    dense_layer,
    dropout_layer,
    gelu_activation,
    layer_norm,
    multi_head_attention,
    relu_activation,
    silu_activation,
    softmax_output,
)


class TestReluActivation:
    """Tests for relu_activation."""

    def test_basic(self):
        x = jnp.array([-1.0, 0.0, 1.0, 2.0])
        assert relu_activation(x).tolist() == [0.0, 0.0, 1.0, 2.0]

    def test_all_negative(self):
        x = jnp.array([-3.0, -2.0, -1.0])
        assert jnp.allclose(relu_activation(x), jnp.zeros(3))

    @given(st.floats(min_value=0.0, max_value=100.0, allow_nan=False))
    @settings(max_examples=20)
    def test_positive_identity(self, x_val):
        """Property: relu(x) = x for x >= 0."""
        result = float(relu_activation(jnp.array(x_val)))
        assert abs(result - x_val) < 1e-5


class TestGeluActivation:
    """Tests for gelu_activation."""

    def test_zero(self):
        result = gelu_activation(jnp.array(0.0))
        assert abs(float(result)) < 1e-6

    def test_positive(self):
        result = gelu_activation(jnp.array(1.0))
        assert float(result) > 0.8

    def test_approximate_flag(self):
        x = jnp.array([1.0, 2.0])
        approx = gelu_activation(x, approximate=True)
        exact = gelu_activation(x, approximate=False)
        assert jnp.allclose(approx, exact, atol=0.02)


class TestSiluActivation:
    """Tests for silu_activation."""

    def test_zero(self):
        result = silu_activation(jnp.array(0.0))
        assert abs(float(result)) < 1e-6

    def test_large_positive(self):
        result = silu_activation(jnp.array(10.0))
        assert float(result) > 9.9  # silu(x) ≈ x for large x


class TestSoftmaxOutput:
    """Tests for softmax_output."""

    def test_sums_to_one(self):
        logits = jnp.array([1.0, 2.0, 3.0])
        probs = softmax_output(logits)
        assert abs(float(jnp.sum(probs)) - 1.0) < 1e-6

    def test_ordering_preserved(self):
        probs = softmax_output(jnp.array([1.0, 2.0, 3.0]))
        assert float(probs[2]) > float(probs[1]) > float(probs[0])

    def test_custom_axis(self):
        logits = jnp.ones((3, 4))
        probs = softmax_output(logits, axis=0)
        assert jnp.allclose(jnp.sum(probs, axis=0), jnp.ones(4), atol=1e-6)

    @given(st.integers(min_value=2, max_value=10))
    @settings(max_examples=10, deadline=None)
    def test_sum_to_one_property(self, n):
        """Property: softmax outputs always sum to 1."""
        logits = jnp.ones(n)
        probs = softmax_output(logits)
        assert abs(float(jnp.sum(probs)) - 1.0) < 1e-5


class TestLayerNorm:
    """Tests for layer_norm."""

    def test_zero_mean(self):
        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        normed = layer_norm(x)
        for i in range(2):
            assert abs(float(jnp.mean(normed[i]))) < 1e-5

    def test_with_weight_and_bias(self):
        x = jnp.array([[1.0, 2.0, 3.0]])
        weight = jnp.array([2.0, 2.0, 2.0])
        bias = jnp.array([1.0, 1.0, 1.0])
        result = layer_norm(x, weight=weight, bias=bias)
        assert result.shape == (1, 3)

    def test_without_affine(self):
        x = jnp.array([[10.0, 20.0, 30.0]])
        normed = layer_norm(x)
        assert abs(float(jnp.std(normed[0])) - 1.0) < 0.1

    def test_weight_only(self):
        x = jnp.array([[1.0, 2.0, 3.0]])
        weight = jnp.ones(3) * 3.0
        result = layer_norm(x, weight=weight)
        assert result.shape == (1, 3)

    def test_bias_only(self):
        x = jnp.array([[1.0, 2.0, 3.0]])
        bias = jnp.ones(3) * 5.0
        result = layer_norm(x, bias=bias)
        assert result.shape == (1, 3)


class TestDenseLayer:
    """Tests for dense_layer."""

    def test_basic(self):
        x = jnp.ones((4, 3))
        w = jnp.ones((2, 3))
        b = jnp.zeros(2)
        result = dense_layer(x, w, b)
        assert result.shape == (4, 2)
        assert jnp.allclose(result, jnp.full((4, 2), 3.0))

    def test_no_bias(self):
        x = jnp.ones((2, 3))
        w = jnp.eye(3)
        result = dense_layer(x, w)
        assert jnp.allclose(result, x)

    def test_with_bias(self):
        x = jnp.zeros((2, 3))
        w = jnp.eye(3)
        b = jnp.array([1.0, 2.0, 3.0])
        result = dense_layer(x, w, b)
        assert jnp.allclose(result[0], b)


class TestDropoutLayer:
    """Tests for dropout_layer."""

    def test_inference_mode(self):
        key = jax.random.key(42)
        x = jnp.ones((4, 8))
        result = dropout_layer(x, key, rate=0.5, training=False)
        assert jnp.allclose(result, x)

    def test_training_mode_scales(self):
        key = jax.random.key(0)
        x = jnp.ones((1000,))
        result = dropout_layer(x, key, rate=0.5, training=True)
        # Non-zero elements should be scaled by 1/(1-rate) = 2
        nonzero = result[result > 0]
        assert jnp.allclose(nonzero, jnp.full_like(nonzero, 2.0))

    def test_training_drops_elements(self):
        key = jax.random.key(42)
        x = jnp.ones((10000,))
        result = dropout_layer(x, key, rate=0.5, training=True)
        zero_frac = float(jnp.mean(result == 0.0))
        assert 0.3 < zero_frac < 0.7


class TestAttentionScores:
    """Tests for attention_scores."""

    def test_basic_shape(self):
        q = jnp.ones((2, 4, 8))
        k = jnp.ones((2, 4, 8))
        v = jnp.ones((2, 4, 16))
        result = attention_scores(q, k, v)
        assert result.shape == (2, 4, 16)

    def test_with_mask(self):
        q = jnp.ones((1, 3, 4))
        k = jnp.ones((1, 3, 4))
        v = jnp.ones((1, 3, 4))
        mask = jnp.tril(jnp.ones((1, 3, 3), dtype=bool))
        result = attention_scores(q, k, v, mask=mask)
        assert result.shape == (1, 3, 4)

    def test_without_mask(self):
        q = jnp.ones((1, 2, 4))
        k = jnp.ones((1, 2, 4))
        v = jnp.ones((1, 2, 4))
        result = attention_scores(q, k, v)
        assert result.shape == (1, 2, 4)


class TestMultiHeadAttention:
    """Tests for multi_head_attention."""

    def test_basic_shape(self):
        batch, seq, d_model, heads = 2, 8, 16, 4
        q = jnp.ones((batch, seq, d_model))
        k = jnp.ones((batch, seq, d_model))
        v = jnp.ones((batch, seq, d_model))
        w_q = jnp.ones((d_model, d_model)) * 0.1
        w_k = jnp.ones((d_model, d_model)) * 0.1
        w_v = jnp.ones((d_model, d_model)) * 0.1
        w_o = jnp.ones((d_model, d_model)) * 0.1
        result = multi_head_attention(q, k, v, w_q, w_k, w_v, w_o, heads)
        assert result.shape == (batch, seq, d_model)

    def test_single_head(self):
        batch, seq, d_model = 1, 4, 8
        q = jnp.ones((batch, seq, d_model))
        k = jnp.ones((batch, seq, d_model))
        v = jnp.ones((batch, seq, d_model))
        w = jnp.eye(d_model) * 0.1
        result = multi_head_attention(q, k, v, w, w, w, w, num_heads=1)
        assert result.shape == (batch, seq, d_model)
