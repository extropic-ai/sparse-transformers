from dataclasses import dataclass

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Key


def sparse_count(count, dim: int) -> int:
    if count is None:
        return dim
    return max(1, min(count, dim))


class SparseLinear(eqx.Module):
    linear: eqx.nn.Linear
    mask: Bool[Array, "outs ins"]

    def __init__(self, ins, outs, fan_in, key, use_bias=True):
        subkey1, subkey2 = jax.random.split(key)
        self.linear = eqx.nn.Linear(ins, outs, use_bias=use_bias, key=subkey1)

        def _mask_row(key):
            vals = jax.random.choice(key, jnp.arange(ins), (fan_in,), replace=False)
            return jnp.zeros(ins, dtype=bool).at[vals].set(True)

        keys = jax.random.split(subkey2, outs)
        self.mask = jax.vmap(_mask_row)(keys)

    def __call__(self, x: Float[Array, " dim2"]):
        layer = eqx.tree_at(
            lambda x: x.weight,
            self.linear,
            jax.lax.stop_gradient(self.mask) * self.linear.weight,
        )
        return layer(x)


class CausalSelfAttention(eqx.Module):
    q_proj: eqx.nn.Linear | SparseLinear
    k_proj: eqx.nn.Linear | SparseLinear
    v_proj: eqx.nn.Linear | SparseLinear
    out_proj: eqx.nn.Linear | SparseLinear
    n_head: int
    window: int | None

    def __init__(
        self,
        n_embed: int,
        n_head: int,
        seq_len: int,
        attn_window,
        linear_fan_in,
        key: Key[Array, ""],
    ):
        keys = jax.random.split(key, 4)
        fan_in = sparse_count(linear_fan_in, n_embed)
        if fan_in >= n_embed:
            self.q_proj = eqx.nn.Linear(n_embed, n_embed, key=keys[0])
            self.k_proj = eqx.nn.Linear(n_embed, n_embed, key=keys[1])
            self.v_proj = eqx.nn.Linear(n_embed, n_embed, key=keys[2])
            self.out_proj = eqx.nn.Linear(n_embed, n_embed, key=keys[3])
        else:
            self.q_proj = SparseLinear(n_embed, n_embed, fan_in, key=keys[0])
            self.k_proj = SparseLinear(n_embed, n_embed, fan_in, key=keys[1])
            self.v_proj = SparseLinear(n_embed, n_embed, fan_in, key=keys[2])
            self.out_proj = SparseLinear(n_embed, n_embed, fan_in, key=keys[3])

        self.n_head = n_head
        window = sparse_count(attn_window, seq_len)
        self.window = window if window < seq_len else None

    def __call__(
        self, x: Float[Array, "seq_len embed"], mask: Bool[Array, " seq_len"] | None
    ):
        T, H = x.shape
        q = eqx.filter_vmap(self.q_proj)(x)
        k = eqx.filter_vmap(self.k_proj)(x)
        v = eqx.filter_vmap(self.v_proj)(x)

        q = jnp.reshape(q, (T, self.n_head, H // self.n_head))
        k = jnp.reshape(k, (T, self.n_head, H // self.n_head))
        v = jnp.reshape(v, (T, self.n_head, H // self.n_head))
        if mask is not None:
            mask = mask[None, None, :]

        # (T, n_head, H)
        att = jax.nn.dot_product_attention(
            q, k, v, mask=mask, is_causal=True, local_window_size=self.window
        )
        att = jnp.reshape(att, (T, att.shape[-1] * att.shape[-2]))

        y = eqx.filter_vmap(self.out_proj)(att)

        return y


class FFN(eqx.Module):
    lin1: eqx.nn.Linear | SparseLinear
    lin2: eqx.nn.Linear | SparseLinear

    def __init__(self, n_embed: int, linear_fan_in, key: Key[Array, ""]):
        subkey1, subkey2 = jax.random.split(key)
        ffw = 4 * n_embed
        fan1 = sparse_count(linear_fan_in, n_embed)
        fan2 = sparse_count(linear_fan_in, ffw)
        if fan1 >= n_embed:
            self.lin1 = eqx.nn.Linear(n_embed, ffw, key=subkey1)
            self.lin2 = eqx.nn.Linear(ffw, n_embed, key=subkey2)
        else:
            self.lin1 = SparseLinear(n_embed, ffw, fan1, key=subkey1)
            self.lin2 = SparseLinear(ffw, n_embed, fan2, key=subkey2)

    def __call__(
        self, x: Float[Array, "seq_len n_embed"]
    ) -> Float[Array, "seq_len n_embed"]:
        x = eqx.filter_vmap(self.lin1)(x)
        x = jax.nn.gelu(x)
        x = eqx.filter_vmap(self.lin2)(x)
        return x


class Block(eqx.Module):
    attn: CausalSelfAttention
    ffn: FFN
    norm1: eqx.nn.RMSNorm
    norm2: eqx.nn.RMSNorm

    def __init__(
        self,
        n_embed: int,
        n_head: int,
        seq_len: int,
        attn_window,
        linear_fan_in,
        key: Key[Array, ""],
    ):
        keys = jax.random.split(key)
        self.attn = CausalSelfAttention(
            n_embed, n_head, seq_len, attn_window, linear_fan_in, keys[0]
        )
        self.ffn = FFN(n_embed, linear_fan_in, keys[1])
        self.norm1 = eqx.nn.RMSNorm(n_embed)
        self.norm2 = eqx.nn.RMSNorm(n_embed)

    def __call__(
        self, x: Float[Array, "seq_len n_embed"], mask: Bool[Array, " seq_len"] | None
    ):
        y = eqx.filter_vmap(self.norm1)(x)
        x = x + self.attn(y, mask)
        y = eqx.filter_vmap(self.norm2)(x)
        x = x + self.ffn(y)
        return x


@dataclass
class Config:
    vocab: int
    sequence: int
    n_layers: int
    n_embed: int
    n_head: int
    attn_window: int | None = None
    linear_fan_in: int | None = None
    remat: bool = False


class Transformer(eqx.Module):
    wte: eqx.nn.Embedding
    wpe: eqx.nn.Embedding
    norm: eqx.nn.RMSNorm
    h: list[Block]
    output_linear: eqx.nn.Linear
    seq_len: int
    remat: bool = eqx.field(static=True)

    def __init__(self, config: Config, key: Key[Array, ""]):
        keys = jax.random.split(key, 3 + config.n_layers)
        self.wte = eqx.nn.Embedding(config.vocab, config.n_embed, key=keys[0])
        self.wpe = eqx.nn.Embedding(config.sequence, config.n_embed, key=keys[1])
        self.h = [
            Block(
                config.n_embed,
                config.n_head,
                config.sequence,
                config.attn_window,
                config.linear_fan_in,
                keys[i + 2],
            )
            for i in range(config.n_layers)
        ]
        self.norm = eqx.nn.RMSNorm(config.n_embed)
        self.output_linear = eqx.nn.Linear(
            config.n_embed, config.vocab, use_bias=False, key=keys[-1]
        )
        self.seq_len = config.sequence
        self.remat = config.remat

    def __call__(
        self, x: Int[Array, " seq_len"], mask: Bool[Array, " seq_len"] | None = None
    ) -> Float[Array, "seq_len vocab"]:
        T = x.shape[0]
        pos = jnp.arange(0, T)

        wte = eqx.filter_vmap(self.wte)(x)
        wpe = eqx.filter_vmap(self.wpe)(pos)

        x = wte + wpe

        def run(blk, h, m):
            return blk(h, m)

        apply = eqx.filter_checkpoint(run) if self.remat else run
        for block in self.h:
            x = apply(block, x, mask)
        x = eqx.filter_vmap(self.norm)(x)
        logits = eqx.filter_vmap(self.output_linear)(x)
        return logits

    def generate(
        self,
        x: Int[Array, "seq_len"],
        start: int,
        key: Key[Array, ""],
        temperature: float = 1.0,
    ):
        assert start > 0, "start cannot be 0, since first token is the starting flag"

        def _loop(i, carry):
            x, mask, key = carry
            logits = self(x, mask)
            logits = logits[i - 1] / temperature
            key, sample_key = jax.random.split(key)
            idx = jax.random.categorical(sample_key, logits)
            x = x.at[i].set(idx)
            mask = mask.at[i].set(True)
            return x, mask, key

        mask = jnp.full(self.seq_len, False)
        mask = mask.at[:start].set(True)
        x, mask, _ = jax.lax.fori_loop(start, self.seq_len, _loop, (x, mask, key))
        return x


def active_params(model) -> int:
    total = 0
    for leaf in jax.tree.leaves(model, is_leaf=lambda x: isinstance(x, SparseLinear)):
        if isinstance(leaf, SparseLinear):
            total += int(leaf.mask.sum())
            if leaf.linear.bias is not None:
                total += int(leaf.linear.bias.size)
        elif eqx.is_inexact_array(leaf):
            total += int(leaf.size)
    return total


# from https://github.com/karpathy/nanoGPT/blob/master/scaling_laws.ipynb
def flops(config: Config):
    """
    Calculate total number of FLOPs, see Chinchilla
    paper Appendix F as reference: https://arxiv.org/pdf/2203.15556.pdf
    """
    seq_len = config.sequence
    vocab_size = config.vocab
    d_model = config.n_embed
    num_heads = config.n_head
    num_layers = config.n_layers
    ffw_size = 4 * config.n_embed
    key_size = d_model // num_heads
    window = sparse_count(config.attn_window, seq_len)
    proj_fan = sparse_count(config.linear_fan_in, d_model)
    lin2_fan = sparse_count(config.linear_fan_in, ffw_size)
    # embeddings
    embeddings = 2 * seq_len * vocab_size * d_model
    # attention
    # key, query, value projections (SparseLinear: fan-in proj_fan)
    attention = 2 * 3 * seq_len * (key_size * num_heads) * proj_fan
    # key @ query logits (over `window` keys)
    attlogits = 2 * seq_len * window * (key_size * num_heads)
    # softmax
    attsoftmax = 3 * num_heads * seq_len * window
    # softmax @ value reductions
    attvalue = 2 * seq_len * window * (key_size * num_heads)
    # final linear (fan-in proj_fan)
    attlinear = 2 * seq_len * (key_size * num_heads) * proj_fan
    att = attention + attlogits + attsoftmax + attvalue + attlinear
    # feed forward
    dense = 2 * seq_len * (ffw_size * proj_fan + d_model * lin2_fan)

    # logits
    logits = 2 * seq_len * d_model * vocab_size

    forward_flops = embeddings + num_layers * (att + dense) + logits
    backward_flops = 2 * forward_flops  # as in Kaplan et al. 2020
    total_flops = forward_flops + backward_flops

    return total_flops
