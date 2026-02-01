"""Neural network building blocks using jax.nn.

Activation functions, normalization layers, and attention mechanisms.

Rust equivalents:
    relu → aprender::nn::relu
    gelu → aprender::nn::gelu
    softmax → realizar::kernels::SoftmaxKernel
    layer_norm → realizar::kernels::LayerNormKernel
    attention → realizar::kernels::AttentionKernel
"""

from jax_gtc.neural.layers import (
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

__all__ = [
    "relu_activation",
    "gelu_activation",
    "silu_activation",
    "softmax_output",
    "layer_norm",
    "dense_layer",
    "dropout_layer",
    "attention_scores",
    "multi_head_attention",
]
