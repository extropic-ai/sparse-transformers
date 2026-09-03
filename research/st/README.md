# Sparse Transformers in JAX

Sparse Transformers is a [JAX](https://github.com/jax-ml/jax)-based library for building and training decoder models with local attention and fixed-sparsity linear layers.

## Installation

Requires Python 3.12+.

```bash
git clone https://github.com/extropic-ai/sparse-transformers.git
cd sparse-transformers
pip install -e "research/st[experiments]"
```

For the core model without experiment dependencies, use `pip install -e research/st`.

## Training

The OpenWebText experiment expects byte-tokenized `train.npy` and `validation.npy` files under the configured `data_path`. From the repository root, run:

```bash
pip install -e ".[experiments]"
python experiments/sparse_gpt_scaling.py --config experiments/example.yaml
```
