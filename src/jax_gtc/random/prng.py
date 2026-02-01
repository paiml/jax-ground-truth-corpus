"""Functional PRNG recipes.

JAX's PRNG is deterministic and splittable. Every random operation
consumes a key, and keys must be explicitly split to generate new
independent streams. This eliminates hidden state and enables
reproducibility across devices.

References:
    - JAX random: https://jax.readthedocs.io/en/latest/random-numbers.html
    - JAX source: jax/_src/random.py

Rust cross-reference:
    trueno::random provides a Threefry-based splittable PRNG with
    SIMD-accelerated sampling (AVX2 Box-Muller for normals).

"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def create_key(seed: int) -> Array:
    """Create a PRNG key from an integer seed.

    This is the entry point for all JAX randomness. The key encodes
    the PRNG state and must be split before reuse.

    Args:
        seed: Integer seed for reproducibility.

    Returns:
        PRNG key array.

    Examples:
        >>> key = create_key(42)
        >>> key.shape
        ()

    Rust equivalent:
        trueno::random::Rng::new(42) creates a Threefry PRNG.

    """
    return jax.random.key(seed)


def split_key(key: Array, num: int = 2) -> Array:
    """Split a PRNG key into multiple independent sub-keys.

    Keys must NEVER be reused — always split before use. Each split
    produces statistically independent streams.

    Args:
        key: Parent PRNG key (consumed — do not reuse).
        num: Number of sub-keys to generate.

    Returns:
        Array of sub-keys with shape (num, ...).

    Examples:
        >>> key = create_key(0)
        >>> subkeys = split_key(key, 3)
        >>> subkeys.shape
        (3,)

        >>> # Common pattern: split off one key for use, keep the rest
        >>> key = create_key(0)
        >>> key, subkey = split_key(key)

    Rust equivalent:
        trueno::random::Rng::split() returns two independent RNGs.
        For n splits: rng.split_n(n) -> Vec<Rng>.

    """
    return jax.random.split(key, num)


def normal_sample(key: Array, shape: tuple[int, ...], dtype: jnp.dtype = jnp.float32) -> Array:
    """Sample from a standard normal distribution N(0, 1).

    Args:
        key: PRNG key (consumed).
        shape: Output shape.
        dtype: Output dtype.

    Returns:
        Array of normal samples.

    Examples:
        >>> key = create_key(42)
        >>> samples = normal_sample(key, (1000,))
        >>> samples.shape
        (1000,)
        >>> abs(float(jnp.mean(samples))) < 0.1
        True
        >>> abs(float(jnp.std(samples)) - 1.0) < 0.1
        True

    Rust equivalent:
        trueno::random::normal(rng, shape) uses SIMD Box-Muller
        transform (AVX2: 8 samples per cycle).

    """
    return jax.random.normal(key, shape=shape, dtype=dtype)


def uniform_sample(
    key: Array,
    shape: tuple[int, ...],
    minval: float = 0.0,
    maxval: float = 1.0,
    dtype: jnp.dtype = jnp.float32,
) -> Array:
    """Sample from a uniform distribution U(minval, maxval).

    Args:
        key: PRNG key (consumed).
        shape: Output shape.
        minval: Lower bound (inclusive).
        maxval: Upper bound (exclusive).
        dtype: Output dtype.

    Returns:
        Array of uniform samples in [minval, maxval).

    Examples:
        >>> key = create_key(42)
        >>> samples = uniform_sample(key, (1000,), minval=-1.0, maxval=1.0)
        >>> float(jnp.min(samples)) >= -1.0
        True
        >>> float(jnp.max(samples)) < 1.0
        True

    Rust equivalent:
        trueno::random::uniform(rng, shape, min, max) uses SIMD
        scaled integer conversion for uniform floats.

    """
    return jax.random.uniform(key, shape=shape, minval=minval, maxval=maxval, dtype=dtype)


def categorical_sample(key: Array, logits: Array, num_samples: int = 1) -> Array:
    """Sample from a categorical distribution defined by logits.

    Args:
        key: PRNG key (consumed).
        logits: Unnormalized log-probabilities (last axis is categories).
        num_samples: Number of independent samples per set of logits.

    Returns:
        Integer indices of sampled categories.

    Examples:
        >>> import jax.numpy as jnp
        >>> key = create_key(42)
        >>> # Strongly peaked distribution — should almost always pick index 2
        >>> logits = jnp.array([0.0, 0.0, 100.0])
        >>> sample = categorical_sample(key, logits)
        >>> int(sample) == 2
        True

    Rust equivalent:
        trueno::random::categorical uses Gumbel-max trick for
        efficient parallel sampling.

    """
    if num_samples == 1:
        return jax.random.categorical(key, logits)
    keys = jax.random.split(key, num_samples)
    return jax.vmap(lambda k: jax.random.categorical(k, logits))(keys)
