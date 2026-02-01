# JAX Ground Truth Corpus

<p align="center">
  <img src="assets/hero.svg" alt="JAX Ground Truth Corpus" width="100%">
</p>

<p align="center">
  <strong>Production-ready JAX recipes for the PAIML Sovereign AI Stack</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/coverage-100%25-brightgreen" alt="Coverage 100%">
  <img src="https://img.shields.io/badge/tests-147_passed-brightgreen" alt="Tests 147">
  <img src="https://img.shields.io/badge/ruff-passing-blue" alt="Ruff passing">
  <img src="https://img.shields.io/badge/ty-passing-purple" alt="ty passing">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
</p>

---

## Overview

This corpus provides curated, tested, and documented JAX patterns that serve three purposes:

1. **Ground truth for Batuta Oracle RAG** -- enabling natural language queries about JAX workflows across the Sovereign AI Stack documentation
2. **Cross-language reference** -- every function documents its Rust equivalent in the stack (trueno, entrenar, realizar, repartir)
3. **Production-grade recipes** -- 100% test coverage, full type checking (ty), zero lint violations (ruff), property-based testing (Hypothesis)

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

```python
import jax.numpy as jnp
from jax_gtc.transforms import jit_compile, grad_transform, vmap_batch
from jax_gtc.arrays import matmul
from jax_gtc.neural import relu_activation

# JIT-compiled matrix multiply
x = jnp.ones((32, 64))
w = jnp.ones((64, 10))
result = jit_compile(matmul)(x, w)

# Automatic differentiation
def loss_fn(params, x, y):
    pred = matmul(x, params)
    return jnp.mean((pred - y) ** 2)

grads = grad_transform(loss_fn)(w, x, jnp.zeros((32, 10)))

# Vectorized batch processing
batched_relu = vmap_batch(relu_activation)
activations = batched_relu(result)
```

## Module Directory

| Category | Module | Functions | Rust Equivalent |
|----------|--------|-----------|-----------------|
| **Transforms** | `jax_gtc.transforms` | `jit_compile`, `grad_transform`, `vmap_batch`, `pmap_parallel`, `compose_transforms` | `trueno::jit`, `entrenar::autograd`, `repartir::Pool` |
| **Autodiff** | `jax_gtc.autodiff` | `gradient`, `jacobian_forward`, `jacobian_reverse`, `hessian_matrix`, `custom_vjp_rule`, `gradient_check` | `entrenar::autograd` (Wengert list, dual numbers) |
| **Arrays** | `jax_gtc.arrays` | `create_array`, `matmul`, `batch_matmul`, `einsum_contract`, `fft_transform`, `index_update` | `trueno::Tensor`, `trueno::ops` (SIMD/GPU) |
| **Neural** | `jax_gtc.neural` | `relu`, `gelu`, `silu`, `softmax`, `layer_norm`, `dense_layer`, `dropout`, `attention`, `multi_head_attention` | `aprender::nn`, `realizar::kernels` |
| **Training** | `jax_gtc.training` | `sgd_optimizer`, `adam_optimizer`, `training_step`, `cosine_schedule` | `entrenar::optimizers`, `entrenar::schedulers` |
| **Distributed** | `jax_gtc.distributed` | `data_parallel_step`, `replicate_params`, `shard_array`, `all_reduce_sum` | `repartir::Pool`, `repartir::collective` |
| **Random** | `jax_gtc.random` | `create_key`, `split_key`, `normal_sample`, `uniform_sample`, `categorical_sample` | `trueno::random` (Threefry PRNG) |

## Architecture

```
jax-ground-truth-corpus/
├── CLAUDE.md                         # P0: Dev guidelines & API reference
├── README.md                         # P1: This file
├── pyproject.toml                    # Python project config
├── ty.toml                           # ty type checker config
├── Makefile                          # Quality gate commands
├── assets/
│   └── hero.svg                      # Hero image
├── src/
│   └── jax_gtc/                      # Main package (250 stmts, 100% covered)
│       ├── __init__.py
│       ├── transforms/               # jit, grad, vmap, pmap
│       │   ├── __init__.py
│       │   └── core.py
│       ├── autodiff/                 # Jacobians, Hessians, custom VJPs
│       │   ├── __init__.py
│       │   └── derivatives.py
│       ├── arrays/                   # NumPy-compatible ops
│       │   ├── __init__.py
│       │   └── operations.py
│       ├── neural/                   # Activations, attention, MHA
│       │   ├── __init__.py
│       │   └── layers.py
│       ├── training/                 # SGD, Adam, training loops
│       │   ├── __init__.py
│       │   └── optimizers.py
│       ├── distributed/              # pmap, sharding, collectives
│       │   ├── __init__.py
│       │   └── parallel.py
│       └── random/                   # Functional PRNG
│           ├── __init__.py
│           └── prng.py
└── tests/                            # 147 tests, Hypothesis property-based
    ├── test_transforms.py
    ├── test_autodiff.py
    ├── test_arrays.py
    ├── test_neural.py
    ├── test_training.py
    ├── test_distributed.py
    └── test_random.py
```

## Rust Cross-Reference Map

Every JAX concept maps to a crate in the Sovereign AI Stack:

```
JAX (Python)                          Sovereign AI Stack (Rust)
─────────────────────────────────     ─────────────────────────────────
jax.jit        (XLA compile)     ──►  trueno         (SIMD/GPU compute)
jax.grad       (reverse-mode AD) ──►  entrenar       (autograd engine)
jax.vmap       (vectorization)   ──►  trueno         (AVX2/512/NEON lanes)
jax.pmap       (parallelism)     ──►  repartir       (work-stealing pool)
jax.numpy      (array ops)       ──►  trueno::Tensor (zero-copy, aligned)
jax.nn         (activations)     ──►  aprender::nn   (training layers)
                                 ──►  realizar        (GPU inference kernels)
jax.random     (PRNG)            ──►  trueno::random (Threefry, SIMD)
jax.lax        (primitives)      ──►  trueno::ops    (fused SIMD kernels)
PyTrees        (nested structs)  ──►  serde traits   (recursive serialization)
```

## Quality Gates

| Gate | Tool | Status | Standard |
|------|------|--------|----------|
| **Lint** | `ruff` | All checks passed | Zero violations (E, F, W, I, N, D, UP) |
| **Typing** | `ty` | All checks passed | Full type inference, no unresolved imports |
| **Tests** | `pytest` | 147 passed | Property-based (Hypothesis), numerical gradient checks |
| **Coverage** | `pytest-cov` | **100%** | Every file at 100%, 250/250 statements covered |

```
Name                                  Stmts   Miss  Cover
──────────────────────────────────────────────────────────
src/jax_gtc/__init__.py                   1      0   100%
src/jax_gtc/arrays/operations.py         22      0   100%
src/jax_gtc/autodiff/derivatives.py      31      0   100%
src/jax_gtc/distributed/parallel.py      29      0   100%
src/jax_gtc/neural/layers.py             47      0   100%
src/jax_gtc/random/prng.py               17      0   100%
src/jax_gtc/training/optimizers.py       55      0   100%
src/jax_gtc/transforms/core.py           34      0   100%
──────────────────────────────────────────────────────────
TOTAL                                   250      0   100%
```

## Development

```bash
# Quality gates
make lint          # ruff check
make test          # pytest with coverage
make fmt           # ruff format
make typecheck     # mypy strict
make all           # lint + typecheck + test

# Type checking with ty
ty check src/
```

## Oracle RAG Integration

This corpus is automatically indexed by `batuta oracle --rag-index` with the following priority tiers:

| Priority | File | Purpose |
|----------|------|---------|
| **P0** | `CLAUDE.md` | Development guidelines, API cross-reference |
| **P1** | `README.md` | Overview, module directory, architecture |
| **P2** | `src/**/*.py` | Source code with docstrings and examples |

Query the indexed corpus:

```bash
batuta oracle --rag "How do I use vmap in JAX?"
batuta oracle --rag "JAX automatic differentiation patterns"
batuta oracle --rag "distributed training with pmap"
batuta oracle --rag "attention mechanism implementation"
batuta oracle --rag "Adam optimizer functional pattern"
```

## RAG Indexing Stats

```
Documents: 17 Python files indexed
Chunks:    240 semantic chunks
Source:    jax-ground-truth-corpus
```

## Design Principles

This corpus follows the Toyota Production System principles used across the Sovereign AI Stack:

- **Jidoka (autonomation)**: Every function validates its invariants via property-based tests. Gradient checks verify numerical correctness automatically.
- **Poka-Yoke (mistake-proofing)**: Full type annotations catch errors at analysis time. Pure functions prevent side-effect bugs.
- **Kaizen (continuous improvement)**: BLAKE3-hashed incremental reindexing ensures the RAG index stays current with zero wasted compute.
- **Genchi Genbutsu (go and see)**: Cross-references point directly to Rust implementations, not abstract descriptions.

## License

MIT
