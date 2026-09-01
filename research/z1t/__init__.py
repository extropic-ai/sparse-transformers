"""z1t — the Z1 transformer training loop.

A small Equinox AFT transformer — attention-free by construction, no softmax
attention anywhere — in the Z1-compilable form: one of three AFT attention
variants, DyT as the only norm, optional per-site sparse linears, and an
optional tanh wrap on every linear.

A run is one nested yaml, in the same shape `st` uses:

    python -m z1t.train --config configs/addition_aft_conv.yaml

Modules:
    train       the one entrypoint: config loading, run naming, the loop
    model       create_model + the tanh-linear swap
    components  Config, Z1T, AFT variants, SparseLinear, DyT, ...
    dataset     shakespeare / wikitext / openwebtext / addition loaders
    tokenizer   CharTokenizer (char corpora) / GPT2Tokenizer (openwebtext)
"""

from z1t.components import (
    AFT_KINDS,
    AFTConv,
    AFTRecurrence,
    Config,
    DyT,
    GatedAttention,
    MLP,
    PE,
    SparseLinear,
    TanhLinear,
    Z1T,
    active_params,
    attention_factory,
    linear_factory,
    sparse_count,
)
from z1t.dataset import (
    Dataset,
    make_addition,
    make_openwebtext,
    make_shakespeare,
    make_wikitext,
)
from z1t.model import create_model
from z1t.tokenizer import CharTokenizer, GPT2Tokenizer

__all__ = [
    # model
    "Config", "Z1T", "MLP", "PE", "DyT", "SparseLinear", "TanhLinear",
    "GatedAttention", "AFTRecurrence", "AFTConv", "AFT_KINDS",
    "attention_factory", "linear_factory", "sparse_count", "active_params",
    "create_model",
    # data
    "Dataset", "make_shakespeare", "make_wikitext", "make_openwebtext",
    "make_addition", "CharTokenizer", "GPT2Tokenizer",
]
