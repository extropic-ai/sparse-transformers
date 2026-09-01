# Attention-Free Transformers in JAX

z1t is a [JAX](https://github.com/jax-ml/jax)-based library for building and training decoder models with attention-free (AFT) attention, Dynamic Tanh norms, and fixed-sparsity linear layers.

## Installation

Requires Python 3.12+.

```bash
cd research/z1t
pip install -e .
```

To run the training experiment, from the repo root:

```bash
python -m z1t.train --config configs/z1t_smoke.yaml
```
