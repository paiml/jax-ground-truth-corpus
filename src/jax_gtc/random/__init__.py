"""JAX's functional PRNG system.

JAX uses an explicit, splittable PRNG — no hidden global state.
Every random operation requires a key, which must be split to avoid
reuse.

Rust equivalents:
    jax.random.key → trueno::random::Rng::new(seed)
    jax.random.split → trueno::random::Rng::split()
    jax.random.normal → trueno::random::normal()
    jax.random.uniform → trueno::random::uniform()
"""

from jax_gtc.random.prng import (
    categorical_sample,
    create_key,
    normal_sample,
    split_key,
    uniform_sample,
)

__all__ = [
    "create_key",
    "split_key",
    "normal_sample",
    "uniform_sample",
    "categorical_sample",
]
