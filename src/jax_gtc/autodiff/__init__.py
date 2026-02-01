"""Automatic differentiation patterns for JAX.

Forward-mode (JVP) and reverse-mode (VJP) differentiation, Jacobians,
Hessians, and custom derivative rules.

Rust equivalents:
    grad → entrenar::autograd::backward (Wengert list)
    jacfwd → entrenar::autograd::jacobian (forward-mode)
    jacrev → entrenar::autograd::jacobian (reverse-mode)
    hessian → entrenar::autograd::hessian
    custom_vjp → Custom Backward trait implementation
"""

from jax_gtc.autodiff.derivatives import (
    custom_vjp_rule,
    gradient,
    gradient_check,
    hessian_matrix,
    jacobian_forward,
    jacobian_reverse,
    stop_gradient,
    value_and_gradient,
)

__all__ = [
    "gradient",
    "value_and_gradient",
    "jacobian_forward",
    "jacobian_reverse",
    "hessian_matrix",
    "custom_vjp_rule",
    "stop_gradient",
    "gradient_check",
]
