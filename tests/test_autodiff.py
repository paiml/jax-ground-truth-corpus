"""Tests for jax_gtc.autodiff module."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from hypothesis import given, settings
from hypothesis import strategies as st

from jax_gtc.autodiff import (
    custom_vjp_rule,
    gradient,
    gradient_check,
    hessian_matrix,
    jacobian_forward,
    jacobian_reverse,
    stop_gradient,
    value_and_gradient,
)


class TestGradient:
    """Tests for gradient."""

    def test_x_squared(self):
        grad_f = gradient(lambda x: jnp.sum(x**2))
        result = grad_f(jnp.array([1.0, 2.0, 3.0]))
        assert jnp.allclose(result, jnp.array([2.0, 4.0, 6.0]))

    def test_multi_argnums(self):
        grad_f = gradient(lambda x, y: jnp.sum(x * y), argnums=(0, 1))
        gx, gy = grad_f(jnp.ones(3), jnp.full(3, 2.0))
        assert jnp.allclose(gx, jnp.full(3, 2.0))
        assert jnp.allclose(gy, jnp.ones(3))

    @given(st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=20)
    def test_sin_derivative(self, x_val):
        """Property: d/dx[sin(x)] = cos(x)."""
        grad_f = gradient(lambda x: jnp.sin(x))
        result = float(grad_f(jnp.array(x_val)))
        expected = float(jnp.cos(jnp.array(x_val)))
        assert abs(result - expected) < 1e-5


class TestValueAndGradient:
    """Tests for value_and_gradient."""

    def test_basic(self):
        val, grad = value_and_gradient(lambda x: jnp.sum(x**2))(jnp.array([3.0]))
        assert float(val) == 9.0
        assert float(grad[0]) == 6.0

    def test_multi_arg(self):
        def f(x, y):
            return jnp.sum(x + y)

        val, grad = value_and_gradient(f)(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
        assert float(val) == 10.0


class TestJacobianForward:
    """Tests for jacobian_forward."""

    def test_shape(self):
        def f(x):
            return jnp.array([x[0] ** 2, x[0] * x[1], x[1] ** 3])

        J = jacobian_forward(f)(jnp.array([2.0, 3.0]))
        assert J.shape == (3, 2)

    def test_identity(self):
        J = jacobian_forward(lambda x: x)(jnp.array([1.0, 2.0]))
        assert jnp.allclose(J, jnp.eye(2))

    def test_linear(self):
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        J = jacobian_forward(lambda x: A @ x)(jnp.ones(2))
        assert jnp.allclose(J, A)


class TestJacobianReverse:
    """Tests for jacobian_reverse."""

    def test_shape(self):
        def f(x):
            return jnp.array([x[0] + x[1], x[0] * x[1]])

        J = jacobian_reverse(f)(jnp.array([2.0, 3.0]))
        assert J.shape == (2, 2)

    def test_matches_forward(self):
        def f(x):
            return jnp.array([x[0] ** 2, x[1] ** 2])

        x = jnp.array([2.0, 3.0])
        J_fwd = jacobian_forward(f)(x)
        J_rev = jacobian_reverse(f)(x)
        assert jnp.allclose(J_fwd, J_rev)


class TestHessianMatrix:
    """Tests for hessian_matrix."""

    def test_quadratic(self):
        def f(x):
            return x[0] ** 2 + x[0] * x[1] + x[1] ** 2

        H = hessian_matrix(f)(jnp.array([1.0, 1.0]))
        expected = jnp.array([[2.0, 1.0], [1.0, 2.0]])
        assert jnp.allclose(H, expected)

    def test_pure_quadratic(self):
        def f(x):
            return jnp.sum(x**2)

        H = hessian_matrix(f)(jnp.ones(3))
        assert jnp.allclose(H, 2.0 * jnp.eye(3))


class TestCustomVjpRule:
    """Tests for custom_vjp_rule."""

    def test_custom_backward(self):
        @jax.custom_vjp
        def safe_log(x):
            return jnp.log(x)

        def fwd(x):
            return safe_log(x), (x,)

        def bwd(res, g):
            (x,) = res
            return (g / jnp.maximum(x, 1e-8),)

        safe_log.defvjp(fwd, bwd)
        grad_f = jax.grad(safe_log)
        assert abs(float(grad_f(jnp.array(2.0))) - 0.5) < 1e-5

    def test_custom_vjp_rule_wrapper(self):
        """Test our custom_vjp_rule wrapper function."""

        def my_fn(x):
            return jnp.log(x)

        def my_fwd(x):
            return jnp.log(x), (x,)

        def my_bwd(res, g):
            (x,) = res
            return (g / jnp.maximum(x, 1e-8),)

        wrapped = custom_vjp_rule(my_fwd, my_bwd, my_fn)
        grad_f = jax.grad(wrapped)
        result = float(grad_f(jnp.array(4.0)))
        assert abs(result - 0.25) < 1e-5


class TestStopGradient:
    """Tests for stop_gradient."""

    def test_forward_pass(self):
        x = jnp.array([1.0, 2.0, 3.0])
        result = stop_gradient(x)
        assert jnp.allclose(result, x)

    def test_gradient_stops(self):
        def f(x):
            return x * jax.lax.stop_gradient(x)

        grad_f = jax.grad(f)
        assert float(grad_f(jnp.array(3.0))) == 3.0

    def test_stop_gradient_wrapper(self):
        """Test our stop_gradient wrapper."""

        def f(x):
            return x * stop_gradient(x)

        grad_f = jax.grad(f)
        assert float(grad_f(jnp.array(5.0))) == 5.0


class TestGradientCheck:
    """Tests for gradient_check."""

    def test_sin_passes(self):
        def f(x):
            return jnp.sum(jnp.sin(x))

        assert gradient_check(f, (jnp.array([1.0, 2.0]),))

    def test_polynomial_passes(self):
        def f(x):
            return jnp.sum(x**3 + 2 * x**2 + x)

        assert gradient_check(f, (jnp.array([1.0, -1.0, 0.5]),))

    def test_quadratic_passes(self):
        def f(x):
            return jnp.sum(x**2)

        assert gradient_check(f, (jnp.array([1.0, 2.0, 3.0]),))
