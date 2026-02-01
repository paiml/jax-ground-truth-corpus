"""Tests for jax_gtc.transforms module."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jax_gtc.transforms import (
    compose_transforms,
    grad_transform,
    jit_compile,
    pmap_parallel,
    value_and_grad_transform,
    vmap_batch,
)


class TestJitCompile:
    """Tests for jit_compile."""

    def test_basic_jit(self):
        @jit_compile
        def add(x, y):
            return x + y

        result = add(jnp.array(1.0), jnp.array(2.0))
        assert float(result) == 3.0

    def test_jit_preserves_shape(self):
        @jit_compile
        def matmul(a, b):
            return a @ b

        a = jnp.ones((3, 4))
        b = jnp.ones((4, 2))
        result = matmul(a, b)
        assert result.shape == (3, 2)

    def test_jit_with_static_argnums(self):
        compiled = jit_compile(
            lambda x, axis: jnp.sum(x, axis=axis),
            static_argnums=(1,),
        )
        x = jnp.ones((3, 4))
        result = compiled(x, 0)
        assert result.shape == (4,)

    def test_jit_with_donate_argnums(self):
        compiled = jit_compile(lambda x: x + 1, donate_argnums=(0,))
        result = compiled(jnp.array(5.0))
        assert float(result) == 6.0

    def test_jit_no_kwargs(self):
        compiled = jit_compile(lambda x: x * 2)
        assert float(compiled(jnp.array(3.0))) == 6.0


class TestGradTransform:
    """Tests for grad_transform."""

    def test_simple_gradient(self):
        def f(x):
            return jnp.sum(x**2)

        grad_f = grad_transform(f)
        result = grad_f(jnp.array([1.0, 2.0, 3.0]))
        assert jnp.allclose(result, jnp.array([2.0, 4.0, 6.0]))

    def test_gradient_multi_argnums(self):
        def f(x, y):
            return jnp.sum(x * y)

        grad_f = grad_transform(f, argnums=(0, 1))
        gx, gy = grad_f(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
        assert jnp.allclose(gx, jnp.array([3.0, 4.0]))
        assert jnp.allclose(gy, jnp.array([1.0, 2.0]))

    def test_gradient_has_aux(self):
        def f(x):
            return jnp.sum(x**2), x * 2

        grad_f = grad_transform(f, has_aux=True)
        grads, aux = grad_f(jnp.array([1.0, 2.0]))
        assert jnp.allclose(grads, jnp.array([2.0, 4.0]))
        assert jnp.allclose(aux, jnp.array([2.0, 4.0]))

    @given(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=20)
    def test_gradient_x_squared(self, x_val):
        """Property: d/dx[x^2] = 2x."""
        grad_f = grad_transform(lambda x: x**2)
        result = float(grad_f(jnp.array(x_val)))
        expected = 2.0 * x_val
        assert abs(result - expected) < 1e-4


class TestValueAndGradTransform:
    """Tests for value_and_grad_transform."""

    def test_basic(self):
        def f(x):
            return jnp.sum(x**2)

        vg = value_and_grad_transform(f)
        val, grad = vg(jnp.array([3.0]))
        assert float(val) == 9.0
        assert float(grad[0]) == 6.0

    def test_with_has_aux(self):
        def f(x):
            return jnp.sum(x), x * 2

        vg = value_and_grad_transform(f, has_aux=True)
        (val, aux), grad = vg(jnp.array([1.0, 2.0]))
        assert float(val) == 3.0
        assert jnp.allclose(aux, jnp.array([2.0, 4.0]))


class TestVmapBatch:
    """Tests for vmap_batch."""

    def test_basic_vmap(self):
        def dot(a, b):
            return jnp.sum(a * b)

        batch_dot = vmap_batch(dot)
        a = jnp.ones((8, 3))
        b = jnp.ones((8, 3))
        result = batch_dot(a, b)
        assert result.shape == (8,)
        assert jnp.allclose(result, jnp.full(8, 3.0))

    def test_vmap_broadcast(self):
        def linear(w, x):
            return x @ w

        batched = vmap_batch(linear, in_axes=(None, 0))
        w = jnp.eye(3)
        x = jnp.ones((16, 3))
        result = batched(w, x)
        assert result.shape == (16, 3)

    def test_vmap_out_axes(self):
        batched = vmap_batch(lambda x: x * 2, out_axes=0)
        result = batched(jnp.ones((4, 3)))
        assert result.shape == (4, 3)


class TestPmapParallel:
    """Tests for pmap_parallel."""

    def test_basic_pmap(self):
        @pmap_parallel
        def add_one(x):
            return x + 1

        x = jnp.ones((1, 4))
        result = add_one(x)
        assert result.shape == (1, 4)
        assert jnp.allclose(result, jnp.full((1, 4), 2.0))

    def test_pmap_with_axis_name(self):
        fn = pmap_parallel(lambda x: x * 2, axis_name="i")
        result = fn(jnp.ones((1, 3)))
        assert result.shape == (1, 3)

    def test_pmap_with_devices(self):
        devices = jax.devices()[:1]
        fn = pmap_parallel(lambda x: x + 1, devices=devices)
        result = fn(jnp.ones((1, 2)))
        assert result.shape == (1, 2)


class TestComposeTransforms:
    """Tests for compose_transforms."""

    def test_grad_and_jit(self):
        def f(w, x):
            return jnp.sum(w * x)

        grad_fn = compose_transforms(f, grad_argnums=0)
        w = jnp.array([1.0, 2.0])
        x = jnp.array([3.0, 4.0])
        result = grad_fn(w, x)
        assert jnp.allclose(result, jnp.array([3.0, 4.0]))

    def test_vmap_and_jit(self):
        fn = compose_transforms(lambda x: x * 2, vmap_in_axes=0)
        result = fn(jnp.ones((4, 3)))
        assert result.shape == (4, 3)

    def test_no_jit(self):
        fn = compose_transforms(lambda x: x + 1, jit=False)
        result = fn(jnp.array(1.0))
        assert float(result) == 2.0

    def test_grad_vmap_jit(self):
        def loss(w, x):
            return jnp.sum(w * x)

        fn = compose_transforms(loss, grad_argnums=0, vmap_in_axes=(None, 0))
        w = jnp.array([1.0, 2.0])
        x = jnp.ones((4, 2))
        result = fn(w, x)
        assert result.shape == (4, 2)
