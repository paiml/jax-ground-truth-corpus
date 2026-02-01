"""Tests for jax_gtc.training module."""

from __future__ import annotations

import jax.numpy as jnp

from jax_gtc.training import (
    adam_optimizer,
    cosine_schedule,
    sgd_optimizer,
    training_step,
)


class TestSgdOptimizer:
    """Tests for sgd_optimizer."""

    def test_basic_update(self):
        state, update_fn = sgd_optimizer(learning_rate=0.1)
        params = {"w": jnp.array([1.0, 2.0])}
        grads = {"w": jnp.array([0.5, 0.5])}
        new_params, new_state = update_fn(params, grads, state)
        assert jnp.allclose(new_params["w"], jnp.array([0.95, 1.95]))

    def test_zero_gradient(self):
        state, update_fn = sgd_optimizer(learning_rate=0.1)
        params = {"w": jnp.array([1.0, 2.0])}
        grads = {"w": jnp.zeros(2)}
        new_params, _ = update_fn(params, grads, state)
        assert jnp.allclose(new_params["w"], params["w"])

    def test_multiple_params(self):
        state, update_fn = sgd_optimizer(learning_rate=0.01)
        params = {"w": jnp.ones(3), "b": jnp.zeros(2)}
        grads = {"w": jnp.ones(3), "b": jnp.ones(2)}
        new_params, _ = update_fn(params, grads, state)
        assert new_params["w"].shape == (3,)
        assert new_params["b"].shape == (2,)

    def test_state_preserved(self):
        state, update_fn = sgd_optimizer(learning_rate=0.05)
        params = {"w": jnp.ones(2)}
        grads = {"w": jnp.ones(2)}
        _, new_state = update_fn(params, grads, state)
        assert new_state.learning_rate == 0.05


class TestAdamOptimizer:
    """Tests for adam_optimizer."""

    def test_basic_update(self):
        shapes = {"w": (3,)}
        state, update_fn = adam_optimizer(learning_rate=0.001, param_shapes=shapes)
        params = {"w": jnp.array([1.0, 2.0, 3.0])}
        grads = {"w": jnp.array([0.1, 0.2, 0.3])}
        new_params, new_state = update_fn(params, grads, state)
        assert new_params["w"].shape == (3,)
        assert new_state.step == 1

    def test_decreases_loss_direction(self):
        shapes = {"w": (2,)}
        state, update_fn = adam_optimizer(learning_rate=0.01, param_shapes=shapes)
        params = {"w": jnp.array([5.0, 5.0])}
        grads = {"w": jnp.array([1.0, 1.0])}  # positive gradient
        new_params, _ = update_fn(params, grads, state)
        # Adam should move params in the negative gradient direction
        assert float(new_params["w"][0]) < 5.0

    def test_step_counter(self):
        shapes = {"w": (2,)}
        state, update_fn = adam_optimizer(param_shapes=shapes)
        params = {"w": jnp.ones(2)}
        grads = {"w": jnp.ones(2)}
        _, state = update_fn(params, grads, state)
        assert state.step == 1
        _, state = update_fn(params, grads, state)
        assert state.step == 2

    def test_no_initial_shapes(self):
        state, update_fn = adam_optimizer()
        params = {"w": jnp.ones(3)}
        grads = {"w": jnp.ones(3)}
        new_params, new_state = update_fn(params, grads, state)
        assert new_params["w"].shape == (3,)

    def test_custom_betas(self):
        shapes = {"w": (2,)}
        state, update_fn = adam_optimizer(
            beta1=0.8, beta2=0.99, eps=1e-7, param_shapes=shapes,
        )
        assert state.beta1 == 0.8
        assert state.beta2 == 0.99
        assert state.eps == 1e-7


class TestTrainingStep:
    """Tests for training_step."""

    def test_basic(self):
        def mse(params, x, y):
            pred = x @ params["w"]
            return jnp.mean((pred - y) ** 2)

        state, update_fn = sgd_optimizer(0.01)
        params = {"w": jnp.ones((3, 1))}
        x = jnp.ones((4, 3))
        y = jnp.zeros((4, 1))
        new_params, new_state, loss = training_step(
            params, x, y, mse, update_fn, state,
        )
        assert loss > 0
        assert new_params["w"].shape == (3, 1)

    def test_loss_decreases(self):
        def mse(params, x, y):
            pred = x @ params["w"]
            return jnp.mean((pred - y) ** 2)

        state, update_fn = sgd_optimizer(0.01)
        params = {"w": jnp.ones((2, 1))}
        x = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = jnp.zeros((3, 1))

        _, _, loss1 = training_step(params, x, y, mse, update_fn, state)
        new_params, new_state, _ = training_step(params, x, y, mse, update_fn, state)
        _, _, loss2 = training_step(new_params, x, y, mse, update_fn, new_state)
        assert loss2 < loss1


class TestCosineSchedule:
    """Tests for cosine_schedule."""

    def test_start(self):
        assert cosine_schedule(0.1, 0, 100) == 0.1

    def test_end(self):
        assert cosine_schedule(0.1, 100, 100) == 0.0

    def test_midpoint(self):
        mid = cosine_schedule(0.1, 50, 100)
        assert 0.04 < mid < 0.06

    def test_min_lr(self):
        result = cosine_schedule(0.1, 100, 100, min_lr=0.01)
        assert result == 0.01

    def test_beyond_total_steps(self):
        result = cosine_schedule(0.1, 200, 100)
        assert result == 0.0

    def test_monotonic_decrease(self):
        lrs = [cosine_schedule(0.1, s, 100) for s in range(101)]
        for i in range(len(lrs) - 1):
            assert lrs[i] >= lrs[i + 1] - 1e-10
