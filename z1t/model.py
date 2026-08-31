"""Model construction: a base `Z1T` plus the one swap that is easier to apply
after the fact than to thread through every constructor.

`create_model` is the single entry point and takes the same `(config, key)` pair
`st.Transformer` does. Attention, normalisation and per-site linear sparsity are
all chosen inside `Z1T.__init__` from the `Config`; only the tanh wrap is a
post-hoc `jax.tree.map`, because it has to catch the qkv/out projections the
attention variant introduced.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from z1t.components import Config, SparseLinear, TanhLinear, Z1T


def create_model(config: Config, key: Key[Array, ""]) -> Z1T:
    """Build a model from a `Config`. No disk access, no argv."""
    model = Z1T(config, key)

    # Tanh-linearity: wrap every linear as tanh(Wx + b) so the model is in the
    # tanh-linear form the z1nn Z1 compiler ingests.
    #
    # The logit head (`clf`) is EXCLUDED: a tanh on it bounds logits to (-1, 1)
    # and wrecks softmax cross-entropy. We skip it by object identity, not by
    # shape — capture the actual head node and skip that one leaf — so it's
    # skipped regardless of width/type and no internal (·, vocab) linear is
    # skipped by accident.
    if config.tanh_linear:
        head = model.clf  # the one linear that must stay un-squashed

        def is_linear(x):
            return isinstance(x, (eqx.nn.Linear, SparseLinear))

        def wrap(x):
            if not is_linear(x) or x is head:
                return x
            return TanhLinear(x)

        model = jax.tree.map(wrap, model, is_leaf=is_linear)

    return model


def _test_z1t():
    key = jax.random.PRNGKey(0)
    config = Config(vocab=100, sequence=256, n_layers=4, n_embed=8,
                    aft_kind="recurrence")
    model = create_model(config, key)

    print(eqx.tree_pformat(model))

    # single sequence
    toks = jnp.array([3, 1, 4, 1, 5])
    print("logits:", model(toks).shape)                    # (5, 100)


if __name__ == "__main__":
    _test_z1t()
