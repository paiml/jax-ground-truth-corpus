"""Training loops, optimizers, and learning rate schedules.

Functional training patterns — no mutable state, pure update functions.

Rust equivalents:
    sgd → entrenar::optimizers::Sgd
    adam → entrenar::optimizers::Adam
    training_loop → entrenar::Trainer::fit
    checkpoint → pacha::ModelRegistry
    gradient_accumulation → entrenar::GradAccumulator
"""

from jax_gtc.training.optimizers import (
    adam_optimizer,
    checkpoint_load,
    checkpoint_save,
    cosine_schedule,
    gradient_accumulation,
    sgd_optimizer,
    training_step,
)

__all__ = [
    "sgd_optimizer",
    "adam_optimizer",
    "training_step",
    "cosine_schedule",
    "gradient_accumulation",
    "checkpoint_save",
    "checkpoint_load",
]
