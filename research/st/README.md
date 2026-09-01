# Sparse Transformers in JAX

Sparse Transformers is a [JAX](https://github.com/jax-ml/jax)-based library for building and training decoder models with local attention and fixed-sparsity linear layers.

## Installation

Requires Python 3.12+.

```bash
cd research/st
pip install -e .
```

To run the training experiment:

```bash
pip install -e ".[experiments]"
python experiments/text_scaling.py --config experiments/example.yaml
```
