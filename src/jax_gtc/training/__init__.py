"""Training loops, optimizers, and learning rate schedules.

Functional training patterns — no mutable state, pure update functions.

Rust equivalents:
    sgd → entrenar::optimizers::Sgd
    adam → entrenar::optimizers::Adam
    training_loop → entrenar::Trainer::fit
    checkpoint → pacha::ModelRegistry
"""

from jax_gtc.training.optimizers import (
    adam_optimizer,
    cosine_schedule,
    sgd_optimizer,
    training_step,
)

__all__ = [
    "sgd_optimizer",
    "adam_optimizer",
    "training_step",
    "cosine_schedule",
]
