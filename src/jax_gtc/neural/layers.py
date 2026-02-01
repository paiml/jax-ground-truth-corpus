"""Neural network layer recipes.

Production patterns for building neural networks with JAX primitives.
Each function maps to optimized GPU kernels in realizar.

References:
    - JAX nn API: https://jax.readthedocs.io/en/latest/jax.nn.html
    - JAX source: jax/_src/nn/functions.py

Rust cross-reference:
    realizar provides fused GPU kernels for inference:
    GemmKernel, SoftmaxKernel, LayerNormKernel, AttentionKernel.
    aprender::nn provides training-compatible layer implementations.

"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def relu_activation(x: Array) -> Array:
    """Rectified Linear Unit: max(0, x).

    Args:
        x: Input array.

    Returns:
        Element-wise ReLU.

    Examples:
        >>> import jax.numpy as jnp
        >>> x = jnp.array([-1.0, 0.0, 1.0, 2.0])
        >>> relu_activation(x).tolist()
        [0.0, 0.0, 1.0, 2.0]

    Rust equivalent:
        aprender::nn::relu — SIMD max(0, x) via vmaxps (AVX2).

    """
    return jax.nn.relu(x)


def gelu_activation(x: Array, approximate: bool = True) -> Array:
    """Gaussian Error Linear Unit.

    GELU(x) = x * Phi(x) where Phi is the Gaussian CDF.
    Used in BERT, GPT, and most modern transformers.

    Args:
        x: Input array.
        approximate: Use tanh approximation (faster, default True).

    Returns:
        Element-wise GELU.

    Examples:
        >>> import jax.numpy as jnp
        >>> x = jnp.array([-1.0, 0.0, 1.0])
        >>> result = gelu_activation(x)
        >>> abs(float(result[1])) < 1e-6
        True
        >>> float(result[2]) > 0.8
        True

    Rust equivalent:
        aprender::nn::gelu — fused SIMD implementation using
        the tanh approximation for throughput.

    """
    return jax.nn.gelu(x, approximate=approximate)


def silu_activation(x: Array) -> Array:
    """Sigmoid Linear Unit (SiLU / Swish): x * sigmoid(x).

    Used in LLaMA, Mistral, and modern architectures.

    Args:
        x: Input array.

    Returns:
        Element-wise SiLU.

    Examples:
        >>> import jax.numpy as jnp
        >>> x = jnp.array([0.0, 1.0, -1.0])
        >>> result = silu_activation(x)
        >>> abs(float(result[0])) < 1e-6
        True

    Rust equivalent:
        aprender::nn::silu — fused x * sigmoid(x) in single SIMD pass.

    """
    return jax.nn.silu(x)


def softmax_output(x: Array, axis: int = -1) -> Array:
    """Numerically stable softmax.

    Subtracts max for numerical stability before exp.
    Output sums to 1 along the specified axis.

    Args:
        x: Logits array.
        axis: Axis to normalize over. Default -1 (last axis).

    Returns:
        Probability distribution (sums to 1 along axis).

    Examples:
        >>> import jax.numpy as jnp
        >>> logits = jnp.array([1.0, 2.0, 3.0])
        >>> probs = softmax_output(logits)
        >>> abs(float(jnp.sum(probs)) - 1.0) < 1e-6
        True
        >>> float(probs[2]) > float(probs[0])
        True

    Rust equivalent:
        realizar::kernels::SoftmaxKernel — wgpu compute shader with
        warp-shuffle reduction for max and sum.

    """
    return jax.nn.softmax(x, axis=axis)


def layer_norm(
    x: Array,
    weight: Array | None = None,
    bias: Array | None = None,
    eps: float = 1e-5,
) -> Array:
    """Layer normalization.

    Normalizes the last dimension to zero mean and unit variance,
    then applies optional affine transform.

    Args:
        x: Input array (..., D).
        weight: Scale parameter (D,). Optional.
        bias: Shift parameter (D,). Optional.
        eps: Epsilon for numerical stability.

    Returns:
        Normalized array, same shape as input.

    Examples:
        >>> import jax.numpy as jnp
        >>> x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> normed = layer_norm(x)
        >>> abs(float(jnp.mean(normed[0]))) < 1e-5
        True

    Rust equivalent:
        realizar::kernels::LayerNormKernel — fused mean/variance
        computation in a single wgpu pass.

    """
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    normalized = (x - mean) / jnp.sqrt(var + eps)
    if weight is not None:
        normalized = normalized * weight
    if bias is not None:
        normalized = normalized + bias
    return normalized


def dense_layer(x: Array, weight: Array, bias: Array | None = None) -> Array:
    """Fully-connected (dense) linear layer: y = xW^T + b.

    Args:
        x: Input (..., in_features).
        weight: Weight matrix (out_features, in_features).
        bias: Optional bias (out_features,).

    Returns:
        Output (..., out_features).

    Examples:
        >>> import jax.numpy as jnp
        >>> x = jnp.ones((4, 3))
        >>> w = jnp.ones((2, 3))
        >>> b = jnp.zeros(2)
        >>> dense_layer(x, w, b).shape
        (4, 2)

    Rust equivalent:
        aprender::nn::Linear wraps trueno::ops::matmul for the
        forward pass. On GPU, uses realizar::kernels::GemmKernel.

    """
    y = x @ weight.T
    if bias is not None:
        y = y + bias
    return y


def dropout_layer(x: Array, key: Array, rate: float = 0.1, training: bool = True) -> Array:
    """Dropout regularization (functional, requires PRNG key).

    JAX requires explicit randomness — pass a PRNG key for reproducibility.
    During inference (training=False), returns x unchanged.

    Args:
        x: Input array.
        key: JAX PRNG key for random mask generation.
        rate: Dropout probability (fraction of elements zeroed).
        training: If False, skip dropout (inference mode).

    Returns:
        Masked and scaled array (or unchanged if not training).

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> key = jax.random.key(42)
        >>> x = jnp.ones((4, 8))
        >>> # Inference mode — no dropout
        >>> out = dropout_layer(x, key, rate=0.5, training=False)
        >>> float(jnp.sum(out)) == float(jnp.sum(x))
        True

    Rust equivalent:
        aprender::nn::Dropout uses trueno::random for mask generation,
        scales by 1/(1-rate) during training.

    """
    if not training:
        return x
    mask = jax.random.bernoulli(key, 1.0 - rate, shape=x.shape)
    return jnp.where(mask, x / (1.0 - rate), 0.0)


def attention_scores(
    query: Array,
    key: Array,
    value: Array,
    mask: Array | None = None,
) -> Array:
    """Scaled dot-product attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        query: Query tensor (..., seq_q, d_k).
        key: Key tensor (..., seq_k, d_k).
        value: Value tensor (..., seq_k, d_v).
        mask: Optional boolean mask (..., seq_q, seq_k). True = attend.

    Returns:
        Attention output (..., seq_q, d_v).

    Examples:
        >>> import jax.numpy as jnp
        >>> q = jnp.ones((2, 4, 8))   # batch=2, seq=4, d_k=8
        >>> k = jnp.ones((2, 4, 8))
        >>> v = jnp.ones((2, 4, 16))  # d_v=16
        >>> attention_scores(q, k, v).shape
        (2, 4, 16)

    Rust equivalent:
        realizar::kernels::AttentionKernel implements FlashAttention-style
        tiled computation on wgpu, avoiding O(n^2) memory for the
        attention matrix.

    """
    d_k = query.shape[-1]
    scores = jnp.matmul(query, jnp.swapaxes(key, -2, -1)) / jnp.sqrt(d_k)
    if mask is not None:
        scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.matmul(weights, value)


def multi_head_attention(
    query: Array,
    key: Array,
    value: Array,
    w_q: Array,
    w_k: Array,
    w_v: Array,
    w_o: Array,
    num_heads: int,
    mask: Array | None = None,
) -> Array:
    """Multi-head attention mechanism.

    Splits Q, K, V into num_heads parallel attention heads, computes
    scaled dot-product attention on each, then concatenates and projects.

    Args:
        query: Input queries (batch, seq_q, d_model).
        key: Input keys (batch, seq_k, d_model).
        value: Input values (batch, seq_k, d_model).
        w_q: Query projection (d_model, d_model).
        w_k: Key projection (d_model, d_model).
        w_v: Value projection (d_model, d_model).
        w_o: Output projection (d_model, d_model).
        num_heads: Number of attention heads.
        mask: Optional attention mask.

    Returns:
        Output (batch, seq_q, d_model).

    Examples:
        >>> import jax.numpy as jnp
        >>> batch, seq, d_model, heads = 2, 8, 16, 4
        >>> q = jnp.ones((batch, seq, d_model))
        >>> k = jnp.ones((batch, seq, d_model))
        >>> v = jnp.ones((batch, seq, d_model))
        >>> w_q = jnp.ones((d_model, d_model)) * 0.1
        >>> w_k = jnp.ones((d_model, d_model)) * 0.1
        >>> w_v = jnp.ones((d_model, d_model)) * 0.1
        >>> w_o = jnp.ones((d_model, d_model)) * 0.1
        >>> out = multi_head_attention(q, k, v, w_q, w_k, w_v, w_o, heads)
        >>> out.shape
        (2, 8, 16)

    Rust equivalent:
        realizar fuses MHA into a single GPU kernel dispatch:
        QKV projection → split heads → FlashAttention → concat → output proj.

    """
    batch, seq_q, d_model = query.shape
    d_head = d_model // num_heads

    q = (query @ w_q).reshape(batch, seq_q, num_heads, d_head).transpose(0, 2, 1, 3)
    k = (key @ w_k).reshape(batch, -1, num_heads, d_head).transpose(0, 2, 1, 3)
    v = (value @ w_v).reshape(batch, -1, num_heads, d_head).transpose(0, 2, 1, 3)

    attn = attention_scores(q, k, v, mask)
    attn = attn.transpose(0, 2, 1, 3).reshape(batch, seq_q, d_model)
    return attn @ w_o
