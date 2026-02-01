# JAX Ground Truth Corpus - Development Guidelines

## Project Overview

Production-ready Python recipes for JAX ML workflows, organized as a ground truth corpus for the PAIML Sovereign AI Stack Oracle RAG system. Maps JAX patterns to Rust equivalents (trueno SIMD/GPU primitives).

## Critical Rules

- 95%+ test coverage with property-based testing (Hypothesis)
- Zero linting violations (ruff)
- Type annotations on all public functions (mypy strict)
- Every public function must have a docstring with Examples section
- All recipes must be self-contained and runnable
- Cross-reference Rust equivalents in docstrings where applicable

## Commands (uv-only)

```bash
uv sync            # Install dependencies
make lint          # ruff check
make test          # pytest with coverage
make fmt           # ruff format
make typecheck     # mypy strict
make coverage      # pytest-cov HTML report
make all           # lint + typecheck + test
```

All make targets use `uv run` internally.

## Module Structure

### jax_gtc.transforms

Core JAX function transformations — the fundamental building blocks.

| Function | JAX API | Rust Equivalent |
|----------|---------|-----------------|
| `jit_compile` | `jax.jit` | `trueno::jit` (planned) |
| `grad_transform` | `jax.grad` | `entrenar::autograd` |
| `vmap_batch` | `jax.vmap` | `trueno::vmap` (SIMD batch) |
| `pmap_parallel` | `jax.pmap` | `repartir::Pool` |
| `compose_transforms` | Chained transforms | Trait composition |

### jax_gtc.autodiff

Automatic differentiation patterns — forward and reverse mode.

| Function | JAX API | Rust Equivalent |
|----------|---------|-----------------|
| `gradient` | `jax.grad` | `entrenar::autograd::backward` |
| `jacobian_forward` | `jax.jacfwd` | `entrenar::autograd::jacobian` |
| `jacobian_reverse` | `jax.jacrev` | `entrenar::autograd::jacobian` |
| `hessian_matrix` | `jax.hessian` | `entrenar::autograd::hessian` |
| `value_and_gradient` | `jax.value_and_grad` | `entrenar::autograd::value_and_grad` |
| `custom_vjp_rule` | `jax.custom_vjp` | Custom `Backward` trait impl |

### jax_gtc.arrays

NumPy-compatible array operations on accelerators.

| Function | JAX API | Rust Equivalent |
|----------|---------|-----------------|
| `create_array` | `jnp.array` | `trueno::Tensor::from_slice` |
| `reshape_array` | `jnp.reshape` | `trueno::Tensor::reshape` |
| `matmul` | `jnp.matmul` | `trueno::ops::matmul` (SIMD) |
| `einsum_contract` | `jnp.einsum` | `trueno::ops::einsum` |
| `fft_transform` | `jnp.fft.fft` | `trueno::ops::fft` |
| `broadcast_ops` | Broadcasting | `trueno::broadcast` |

### jax_gtc.neural

Neural network building blocks using jax.nn.

| Function | JAX API | Rust Equivalent |
|----------|---------|-----------------|
| `relu_activation` | `jax.nn.relu` | `aprender::nn::relu` |
| `gelu_activation` | `jax.nn.gelu` | `aprender::nn::gelu` |
| `softmax_output` | `jax.nn.softmax` | `realizar::kernels::SoftmaxKernel` |
| `layer_norm` | `jax.nn.normalize` | `realizar::kernels::LayerNormKernel` |
| `attention_mechanism` | Custom | `realizar::kernels::AttentionKernel` |
| `dense_layer` | `stax.Dense` | `aprender::nn::Linear` |

### jax_gtc.training

Training loops, optimizers, and learning rate schedules.

| Function | JAX API | Rust Equivalent |
|----------|---------|-----------------|
| `sgd_optimizer` | `optax.sgd` | `entrenar::optimizers::Sgd` |
| `adam_optimizer` | `optax.adam` | `entrenar::optimizers::Adam` |
| `training_loop` | Custom | `entrenar::Trainer::fit` |
| `gradient_accumulation` | Manual | `entrenar::GradAccumulator` |
| `learning_rate_schedule` | `optax.schedule` | `entrenar::schedulers` |
| `checkpoint_save` | `orbax` | `pacha::ModelRegistry` |

### jax_gtc.distributed

Multi-device and multi-host parallelism.

| Function | JAX API | Rust Equivalent |
|----------|---------|-----------------|
| `data_parallel` | `jax.pmap` | `repartir::Pool::map` |
| `model_parallel` | `jax.shard_map` | `repartir::ShardedExecutor` |
| `mesh_sharding` | `jax.sharding.Mesh` | `repartir::Topology` |
| `all_reduce_sum` | `jax.lax.psum` | `repartir::collective::AllReduce` |
| `fsdp_training` | Sharded params | `repartir + entrenar` |

### jax_gtc.random

JAX's functional PRNG system.

| Function | JAX API | Rust Equivalent |
|----------|---------|-----------------|
| `create_key` | `jax.random.key` | `trueno::random::Rng::new` |
| `split_key` | `jax.random.split` | `trueno::random::Rng::split` |
| `normal_sample` | `jax.random.normal` | `trueno::random::normal` |
| `uniform_sample` | `jax.random.uniform` | `trueno::random::uniform` |
| `categorical_sample` | `jax.random.categorical` | `trueno::random::categorical` |

## Rust Cross-References

JAX's core transforms map to the Sovereign AI Stack:

| JAX Concept | Stack Crate | Notes |
|-------------|-------------|-------|
| `jax.jit` (XLA compile) | `trueno` (SIMD/GPU) | XLA → wgpu compute shaders |
| `jax.grad` (autodiff) | `entrenar` (autograd) | Reverse-mode AD with tape |
| `jax.vmap` (vectorize) | `trueno` (SIMD lanes) | AVX2/AVX-512/NEON mapping |
| `jax.pmap` (parallelize) | `repartir` (distributed) | Work-stealing + GPU |
| `jax.numpy` (arrays) | `trueno::Tensor` | Zero-copy, SIMD-aligned |
| `jax.nn` (neural ops) | `aprender::nn` + `realizar` | GPU kernels for inference |
| `jax.random` (PRNG) | `trueno::random` | Splittable PRNG |
| PyTrees | `serde` traits | Recursive serialization |

## Quality Standards

- Property-based tests for all pure functions (Hypothesis)
- Numerical stability tests (gradient checking via finite differences)
- Cross-validation against JAX reference outputs
- Doctest coverage for all public APIs
- No mutable global state — all functions must be pure
