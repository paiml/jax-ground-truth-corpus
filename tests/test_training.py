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
            beta1=0.8,
            beta2=0.99,
            eps=1e-7,
            param_shapes=shapes,
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
            params,
            x,
            y,
            mse,
            update_fn,
            state,
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


class TestGradientAccumulation:
    """Tests for gradient_accumulation."""

    def test_accumulate_two_steps(self):
        from jax_gtc.training import gradient_accumulation

        def loss_fn(params, x, y):
            return jnp.mean((x @ params["w"] - y) ** 2)

        params = {"w": jnp.ones((3, 1))}
        batches = [
            (jnp.ones((2, 3)), jnp.zeros((2, 1))),
            (jnp.ones((2, 3)) * 2, jnp.zeros((2, 1))),
        ]
        accumulated_grads = gradient_accumulation(loss_fn, params, batches)
        assert "w" in accumulated_grads
        assert accumulated_grads["w"].shape == (3, 1)

    def test_single_batch_equals_direct_grad(self):
        import jax

        from jax_gtc.training import gradient_accumulation

        def loss_fn(params, x, y):
            return jnp.mean((x @ params["w"] - y) ** 2)

        params = {"w": jnp.array([[1.0], [2.0], [3.0]])}
        x = jnp.ones((4, 3))
        y = jnp.zeros((4, 1))

        # Direct gradient
        direct_grads = jax.grad(loss_fn)(params, x, y)

        # Accumulated with single batch
        accumulated_grads = gradient_accumulation(loss_fn, params, [(x, y)])

        assert jnp.allclose(accumulated_grads["w"], direct_grads["w"])

    def test_accumulation_averages(self):
        from jax_gtc.training import gradient_accumulation

        def loss_fn(params, x, _y):
            return jnp.sum(params["w"] * x)

        params = {"w": jnp.ones((2,))}
        # Two batches with known gradients
        batches = [
            (jnp.array([1.0, 0.0]), jnp.array(0.0)),
            (jnp.array([0.0, 1.0]), jnp.array(0.0)),
        ]
        grads = gradient_accumulation(loss_fn, params, batches)
        # Average of [1,0] and [0,1] = [0.5, 0.5]
        assert jnp.allclose(grads["w"], jnp.array([0.5, 0.5]))

    def test_empty_batches_raises(self):
        import pytest

        from jax_gtc.training import gradient_accumulation

        def loss_fn(params, x, y):
            return jnp.sum(params["w"])

        params = {"w": jnp.ones((2,))}
        with pytest.raises(ValueError, match="at least one batch"):
            gradient_accumulation(loss_fn, params, [])


class TestCheckpoint:
    """Tests for checkpoint_save and checkpoint_load."""

    def test_save_and_load_roundtrip(self, tmp_path):
        from jax_gtc.training import checkpoint_load, checkpoint_save

        params = {"w": jnp.array([1.0, 2.0, 3.0]), "b": jnp.array([0.5])}
        path = tmp_path / "checkpoint.npz"

        checkpoint_save(params, path)
        loaded = checkpoint_load(path)

        assert set(loaded.keys()) == set(params.keys())
        assert jnp.allclose(loaded["w"], params["w"])
        assert jnp.allclose(loaded["b"], params["b"])

    def test_save_creates_file(self, tmp_path):
        from jax_gtc.training import checkpoint_save

        params = {"w": jnp.ones((2, 2))}
        path = tmp_path / "model.npz"

        checkpoint_save(params, path)
        assert path.exists()

    def test_load_nonexistent_raises(self, tmp_path):
        import pytest

        from jax_gtc.training import checkpoint_load

        path = tmp_path / "nonexistent.npz"
        with pytest.raises(FileNotFoundError):
            checkpoint_load(path)

    def test_load_optimizer_from_checkpoint_without_optimizer(self, tmp_path):
        """Test loading optimizer state from checkpoint that has no optimizer."""
        from jax_gtc.training import checkpoint_load, checkpoint_save

        params = {"w": jnp.ones((3,))}
        path = tmp_path / "params_only.npz"

        # Save without optimizer state
        checkpoint_save(params, path, optimizer_state=None)

        # Load with load_optimizer=True should return None for state
        loaded_params, loaded_state = checkpoint_load(path, load_optimizer=True)
        assert jnp.allclose(loaded_params["w"], params["w"])
        assert loaded_state is None

    def test_save_with_optimizer_state(self, tmp_path):
        from jax_gtc.training import adam_optimizer, checkpoint_load, checkpoint_save

        params = {"w": jnp.ones((3,))}
        state, _ = adam_optimizer(param_shapes={"w": (3,)})

        path = tmp_path / "full_checkpoint.npz"
        checkpoint_save(params, path, optimizer_state=state)

        loaded_params, loaded_state = checkpoint_load(path, load_optimizer=True)
        assert jnp.allclose(loaded_params["w"], params["w"])
        assert loaded_state.step == state.step

    def test_save_with_trained_optimizer_state(self, tmp_path):
        """Test checkpoint with optimizer state after training steps."""
        from jax_gtc.training import adam_optimizer, checkpoint_load, checkpoint_save

        params = {"w": jnp.ones((3,))}
        state, update_fn = adam_optimizer(param_shapes={"w": (3,)})
        grads = {"w": jnp.array([0.1, 0.2, 0.3])}

        # Do a few training steps to populate m and v dictionaries
        new_params, state = update_fn(params, grads, state)
        new_params, state = update_fn(new_params, grads, state)

        path = tmp_path / "trained_checkpoint.npz"
        checkpoint_save(new_params, path, optimizer_state=state)

        loaded_params, loaded_state = checkpoint_load(path, load_optimizer=True)
        assert jnp.allclose(loaded_params["w"], new_params["w"])
        assert loaded_state.step == 2
        assert "w" in loaded_state.m
        assert "w" in loaded_state.v

    def test_load_optimizer_false_with_optimizer_saved(self, tmp_path):
        """Test loading only params when optimizer was also saved."""
        from jax_gtc.training import adam_optimizer, checkpoint_load, checkpoint_save

        params = {"w": jnp.ones((3,))}
        state, _ = adam_optimizer(param_shapes={"w": (3,)})

        path = tmp_path / "with_opt.npz"
        checkpoint_save(params, path, optimizer_state=state)

        # Load without optimizer
        loaded = checkpoint_load(path, load_optimizer=False)
        assert isinstance(loaded, dict)
        assert "w" in loaded

    def test_save_custom_optimizer_state_with_arrays(self, tmp_path):
        """Test checkpoint with custom optimizer state containing array fields."""
        from typing import NamedTuple

        from jax_gtc.training import checkpoint_save

        class CustomOptimizerState(NamedTuple):
            learning_rate: float
            step: int
            momentum: jnp.ndarray  # Array field (not in a dict)

        params = {"w": jnp.ones((3,))}
        state = CustomOptimizerState(
            learning_rate=0.01,
            step=5,
            momentum=jnp.array([0.1, 0.2, 0.3]),
        )

        path = tmp_path / "custom_opt.npz"
        checkpoint_save(params, path, optimizer_state=state)
        assert path.exists()

        # Verify file contains the momentum array
        import numpy as np

        data = np.load(path)
        assert "opt_momentum" in data.files
        assert np.allclose(data["opt_momentum"], [0.1, 0.2, 0.3])

    def test_load_checkpoint_with_custom_array_field(self, tmp_path):
        """Test loading checkpoint that has custom array optimizer fields."""
        from typing import NamedTuple

        from jax_gtc.training import checkpoint_load, checkpoint_save

        class CustomOptimizerState(NamedTuple):
            learning_rate: float
            step: int
            momentum: jnp.ndarray  # Array field that triggers line 452

        params = {"w": jnp.ones((3,))}
        state = CustomOptimizerState(
            learning_rate=0.01,
            step=5,
            momentum=jnp.array([0.1, 0.2, 0.3]),
        )

        path = tmp_path / "custom_opt_load.npz"
        checkpoint_save(params, path, optimizer_state=state)

        # Loading with load_optimizer=True will parse the custom field
        # even though it constructs an AdamState (the parsing still happens)
        loaded_params, loaded_state = checkpoint_load(path, load_optimizer=True)
        assert jnp.allclose(loaded_params["w"], params["w"])
        # The loaded state will be AdamState with defaults for unrecognized fields
        assert loaded_state is not None
