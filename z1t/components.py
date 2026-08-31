from __future__ import annotations

import math
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float, Int, Key


class SparseLinear(eqx.Module):
    weight: Float[Array, "out k"]
    bias: Float[Array, "out"]
    indices: Int[Array, "out k"]                  # fixed fan-in, not trained
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    k: int = eqx.field(static=True)

    def __init__(self, in_features, out_features, k, *, key):
        if k > in_features:
            raise ValueError(f"k={k} cannot exceed in_features={in_features}")
        self.in_features, self.out_features, self.k = in_features, out_features, k

        idx_key, w_key = jax.random.split(key)
        rows = jax.random.split(idx_key, out_features)
        choose = lambda kk: jax.random.choice(kk, in_features, (k,), replace=False)
        self.indices = jax.vmap(choose)(rows)                 # (out, k) int

        scale = 1.0 / math.sqrt(k)
        self.weight = jax.random.uniform(w_key, (out_features, k),
                                         minval=-scale, maxval=scale)
        self.bias = jnp.zeros(out_features)

    def __call__(self, x: Float[Array, "in"]) -> Float[Array, "out"]:
        # k-sparse gather: O(out·k) work, fuses (gather→mul→reduce) into one kernel,
        # never materializes a dense matrix. Benchmarked as the best of
        # gather / dense-masked / raw-Triton across dims — it wins on speed AND
        # memory as dim grows, and the margin widens with width.
        return jnp.sum(self.weight * x[self.indices], axis=-1) + self.bias


def sparse_count(count, dim: int) -> int:
    """Resolve a configured fan-in against an actual input width.

    Same semantics as `st.transformer.sparse_count`, so a `linear_fan_in` in a
    z1t yaml means exactly what it means in an st yaml: `None` is dense, and any
    value is clamped to [1, dim].
    """
    if count is None:
        return dim
    return max(1, min(count, dim))


def linear_factory(fan_in):
    """A Linear constructor with the signature (in, out, *, key).

    `fan_in` is the configured fan-in (`None` = dense) and is resolved against
    each linear's own `in_features`, so one config value does the right thing at
    both MLP projections even though they have different input widths. A fan-in
    that reaches `in_features` yields a plain dense `Linear` rather than a
    `SparseLinear` that gathers every row.
    """

    def make(i, o, *, key):
        k = sparse_count(fan_in, i)
        if k >= i:
            return eqx.nn.Linear(i, o, key=key)
        return SparseLinear(i, o, k=k, key=key)

    return make


class TanhLinear(eqx.Module):
    """f(x) = tanh(linear(x)) — affine map followed by a tanh.

    Puts a Linear/SparseLinear into the tanh-linear form f(x) = tanh(Wx + b)
    that the z1nn Z1 compiler ingests as a TanhLinearBlock. The call signature
    is forwarded unchanged, so parents that do `self.proj(x)` (or
    `jax.vmap(self.proj)(x)`) behave the same. Adds no parameters.
    """
    linear: eqx.Module

    def __call__(self, x, *args, **kwargs):
        return jnp.tanh(self.linear(x, *args, **kwargs))


class MLP(eqx.Module):
    proj1: eqx.Module
    proj2: eqx.Module
    act: callable

    def __init__(self, dim, mlp_ratio=4.0, *, key, linear=eqx.nn.Linear,
                 act=jax.nn.silu):
        k1, k2 = jax.random.split(key)
        hidden = int(dim * mlp_ratio)
        self.proj1 = linear(dim, hidden, key=k1)
        self.proj2 = linear(hidden, dim, key=k2)
        self.act = act

    def __call__(self, x):                 # x: (T, dim)
        x = jax.vmap(self.proj1)(x)        # (T, hidden) — Linear is per-token
        x = self.act(x)
        x = jax.vmap(self.proj2)(x)        # (T, dim)
        return x


class DyT(eqx.Module):
    """Dynamic Tanh — the model's only normalisation, statistics-free.

    DyT(x) = weight * tanh(alpha * x) + bias, over the last axis.

    `alpha` is a single learnable scalar that sets the input scale; the tanh
    supplies the saturation an RMS norm would otherwise get from dividing by
    the RMS. Nothing is reduced across the feature axis, so it's pure
    elementwise — which is what keeps the model in Z1-compilable form.

    From "Transformers without Normalization" (Liu et al., 2025).
    """
    alpha: Float[Array, ""]          # shared scalar input gain (learnable)
    weight: Float[Array, " dim"]     # per-channel scale  (gamma)
    bias: Float[Array, " dim"]       # per-channel shift  (beta)
    dim: int = eqx.field(static=True)

    def __init__(self, dim, alpha_init=0.5):
        self.dim = dim
        self.alpha = jnp.asarray(alpha_init, dtype=jnp.float32)
        self.weight = jnp.ones(dim)
        self.bias = jnp.zeros(dim)

    def __call__(self, x: Float[Array, "... dim"]) -> Float[Array, "... dim"]:
        return self.weight * jnp.tanh(self.alpha * x) + self.bias


class PE(eqx.Module):
    """Additive sinusoidal positional encoding. The table is a frozen buffer."""
    pe: jax.Array                          # (max_seq, dim)

    def __init__(self, dim, omega=1e4, max_seq=2048):
        assert dim % 2 == 0
        d = dim // 2
        freq = jnp.exp(-math.log(omega) * jnp.arange(d) / d)        # omega^(-i/d), (d,)
        pos = jnp.arange(max_seq)                                   # (max_seq,)
        angle = pos[:, None] * freq[None, :]                        # (max_seq, d)
        pe = jnp.stack([jnp.sin(angle), jnp.cos(angle)], axis=-1)   # (max_seq, d, 2)
        self.pe = pe.reshape(max_seq, dim)                          # [sin0, cos0, sin1, ...]

    def __call__(self, x, offset=0):       # x: (T, dim)
        T = x.shape[-2]
        return x + jax.lax.stop_gradient(self.pe[offset:offset + T])


# === AFT attention ===================================================
#
# The three attention-free-transformer formulations the model can be built
# with. All share the signature (T, dim) -> (T, dim), so `Block` is agnostic to
# which one it holds, and all take a `linear=` factory so they participate in
# the sparse/dense axis like every other module.
#
#   GatedAttention : AFT-full — a learned (max_seq, max_seq) position bias.
#                    Memory is O(T^2 · dim); toy sequence lengths only.
#   AFTRecurrence  : AFT-simple as a per-channel exponentially-decaying running
#                    context, via lax.scan. Cheapest — one decay scalar per
#                    channel — and the one that scales in T.
#   AFTConv        : AFT-conv (Eq. 6) — depthwise causal conv stencil + a causal
#                    running global pool (cumsum). Adds a local-window inductive
#                    bias on top of the global pool.
#
# All three are GENUINELY CAUSAL — output[t] depends only on input[:t+1]. A
# non-causal "attention" silently leaks future tokens during teacher forcing and
# gives fake-perfect training loss, so a regression here reads as an improvement
# in the loss curve. Check it by perturbing input position t and asserting every
# output before t is unchanged.


class GatedAttention(eqx.Module):
    qkv_proj: eqx.Module
    w: Float[Array, "max_seq max_seq"]
    dim: int = eqx.field(static=True)
    max_seq: int = eqx.field(static=True)

    def __init__(self, dim, max_seq=2048, *, key, linear=eqx.nn.Linear):
        self.dim = dim
        self.max_seq = max_seq
        self.qkv_proj = linear(dim, 3 * dim, key=key)
        self.w = jnp.zeros((max_seq, max_seq))

    def __call__(self, x: Float[Array, "T D"]) -> Float[Array, "T D"]:
        T, D = x.shape

        # Linear expects a single (D,) vector, so map it across the sequence.
        qkv = jax.vmap(self.qkv_proj)(x)          # (T, 3D)
        q, k, v = jnp.split(qkv, 3, axis=-1)      # each (T, D)
        q = jax.nn.sigmoid(q)

        w = self.w[:T, :T]                        # (T, T), indexed w[i, j]

        # Causal mask. `w` is indexed [key i, output j] — the TRANSPOSE of the
        # usual [query, key] convention — so the allowed half is i <= j, i.e.
        # triu, not tril. Masking with -inf here zeroes the corresponding k_exp
        # term, dropping it from both the numerator and the denominator sums.
        # Without this, num/den below sum over every i and output[j] sees the
        # whole sequence: future tokens leak under teacher forcing and training
        # loss looks fake-perfect while generation fails.
        w = jnp.where(jnp.triu(jnp.ones((T, T), dtype=bool)), w, -jnp.inf)

        # k_exp[i, j, d] = exp(k[i, d] + w[i, j])
        #   k -> (T, 1, D) broadcasts over the output index j
        #   w -> (T, T, 1) broadcasts over the feature index d
        k_exp = jnp.exp(k[:, None, :] + w[:, :, None])   # (T, T, D)

        # Sum over the key/value index i, leaving the output index j.
        num = jnp.sum(k_exp * v[:, None, :], axis=0)     # (T, D)
        den = jnp.sum(k_exp, axis=0)                     # (T, D)

        return q * num / den


def _dwconv1d_causal(x, kernel):           # x:(C,T)  kernel:(C,s)  -> (C,T)
    """Causal depthwise 1D conv: left-padded only, so output[t] depends on x[:t+1]."""
    C, T = x.shape
    s = kernel.shape[-1]
    k = kernel[:, None, :]                 # (C,1,s)
    lo, hi = s - 1, 0                      # causal: only past + current
    y = lax.conv_general_dilated(
        x[None], k, (1,), [(lo, hi)],
        dimension_numbers=("NCH", "OIH", "NCH"),
        feature_group_count=C)
    return y[0]


class AFTRecurrence(eqx.Module):
    """AFT-simple as a per-channel exponentially-decaying running context.

    Causal by construction (uses lax.scan). x:(T,dim) -> (T,dim).
    """
    qkv_proj: eqx.Module
    out_proj: eqx.Module
    decay: jax.Array                       # (dim,) per-channel log-decay
    dim: int = eqx.field(static=True)

    def __init__(self, dim, *, key, linear=eqx.nn.Linear):
        self.dim = dim
        # init: spread short..long memory across channels
        self.decay = jnp.linspace(-4.0, 2.0, dim)
        kp, ko = jax.random.split(key)
        self.qkv_proj = linear(dim, 3 * dim, key=kp)
        self.out_proj = linear(dim, dim, key=ko)

    def __call__(self, x):                                          # x: (T, dim)
        QKV = jax.vmap(self.qkv_proj)(x)
        Q, K, V = jnp.split(QKV, [self.dim, 2 * self.dim], axis=-1)  # (T, dim) each
        gamma = jnp.exp(-jax.nn.softplus(self.decay))                # (dim,) in (0, 1]
        eK = jnp.exp(jnp.tanh(K))                                    # bounded

        def step(carry, kv):
            S, Z = carry
            ek, v = kv
            S = gamma * S + ek * v
            Z = gamma * Z + ek
            return (S, Z), S / Z

        init = (jnp.zeros(self.dim), jnp.zeros(self.dim))
        _, ctx = lax.scan(step, init, (eK, V))                       # (T, dim)
        return jax.vmap(self.out_proj)(jnp.tanh(Q) * ctx)            # tanh query gate


class AFTConv(eqx.Module):
    """AFT-conv (Eq. 6, causal): depthwise conv1d stencil + cumsum global pool.

    Both the local conv and the global pool are CAUSAL (left-pad only / cumsum).
    A non-causal version would leak future tokens during teacher forcing and
    look perfect at training while failing autoregressive generation.
    """
    qkv_proj: eqx.Module
    out_proj: eqx.Module
    w_conv: jax.Array                      # (heads, ksize)
    dim: int = eqx.field(static=True)
    heads: int = eqx.field(static=True)
    ksize: int = eqx.field(static=True)

    def __init__(self, dim, *, heads=4, ksize=4, key, linear=eqx.nn.Linear):
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim, self.heads, self.ksize = dim, heads, ksize
        kp, ko = jax.random.split(key)
        self.out_proj = linear(dim, dim, key=ko)
        # Q:dim, V:dim, K:heads  (K tied to #heads, Eq. 6 of AFT paper)
        self.qkv_proj = linear(dim, 2 * dim + heads, key=kp)
        # exp(0)-1 = 0 → local stencil starts off; global pool dominates at init.
        self.w_conv = jnp.zeros((heads, ksize))

    def __call__(self, x):                                          # x: (T, dim)
        T = x.shape[0]
        h, hd = self.heads, self.dim // self.heads
        QKV = jax.vmap(self.qkv_proj)(x)                            # (T, 2*dim + h)
        Q, V, K = jnp.split(QKV, [self.dim, 2 * self.dim], axis=-1) # (T,dim),(T,dim),(T,h)
        Qh = Q.reshape(T, h, hd)
        Vh = V.reshape(T, h, hd)
        eK = jnp.exp(jnp.tanh(K))                                   # (T, h) bounded
        kernel = jnp.exp(self.w_conv) - 1.0                         # (h, s)

        eKV = eK[:, :, None] * Vh                                   # (T, h, hd)
        num_conv = _dwconv1d_causal(eKV.reshape(T, self.dim).T,
                                    jnp.repeat(kernel, hd, axis=0)).T.reshape(T, h, hd)
        den_conv = _dwconv1d_causal(eK.T, kernel).T                 # (T, h)
        num = num_conv + jnp.cumsum(eKV, axis=0)                    # (T, h, hd)
        den = den_conv + jnp.cumsum(eK, axis=0)                     # (T, h)
        ctx = num / (den[:, :, None] + 1e-6)                        # (T, h, hd)
        Y = (jnp.tanh(Qh) * ctx).reshape(T, self.dim)
        return jax.vmap(self.out_proj)(Y)


AFT_KINDS = ("full", "recurrence", "conv")


def attention_factory(kind, *, max_seq=2048, heads=4, ksize=4,
                      linear=eqx.nn.Linear):
    """An attention constructor with the signature (dim, *, key).

    Mirrors `linear_factory`: resolves the per-variant constructor arguments
    once, here, so `Block` can build attention without knowing which AFT
    formulation it is getting. `max_seq` sizes AFT-full's learned position bias;
    `heads`/`ksize` size AFT-conv's local stencil. Each is ignored by the
    variants that do not take it.
    """
    if kind == "full":
        return lambda dim, *, key: GatedAttention(dim, max_seq=max_seq, key=key,
                                                  linear=linear)
    if kind == "recurrence":
        return lambda dim, *, key: AFTRecurrence(dim, key=key, linear=linear)
    if kind == "conv":
        return lambda dim, *, key: AFTConv(dim, heads=heads, ksize=ksize, key=key,
                                           linear=linear)
    raise ValueError(f"unknown aft_kind: {kind!r} (expected one of {AFT_KINDS})")


class Block(eqx.Module):
    norm1: DyT
    attn: eqx.Module
    norm2: DyT
    mlp: MLP

    def __init__(self, dim, *, key, attn=attention_factory("conv"),
                 mlp_linear=eqx.nn.Linear, mlp_act=jax.nn.silu, dyt_alpha=0.5):
        ka, km = jax.random.split(key)
        self.norm1 = DyT(dim, alpha_init=dyt_alpha)
        self.attn = attn(dim, key=ka)
        self.norm2 = DyT(dim, alpha_init=dyt_alpha)
        self.mlp = MLP(dim, key=km, linear=mlp_linear, act=mlp_act)

    def __call__(self, x):                 # x: (T, dim)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@dataclass
class Config:
    """Model hyperparameters — the `model:` block of a run yaml.

    The first four fields are named to match `st.transformer.Config`, so the
    shared half of a z1t and an st `model:` block is literally the same keys.
    They diverge on where those two come from: st reads `vocab` and `sequence`
    straight out of the yaml, while `train.model_config` fills them in from the
    dataset, so a z1t config cannot disagree with the data it trains on.

    The rest is z1t's own axis:

    aft_kind        which AFT attention every block gets: full/recurrence/conv
    aft_heads       AFT-conv head count (Eq. 6 K-projection width)
    aft_ksize       AFT-conv local stencil width
    dyt_alpha       alpha init for DyT, the only norm
    linear_fan_in   shared sparse fan-in; None = dense (see `sparse_count`)
    attn_fan_in     per-site override of linear_fan_in for attention linears
    mlp_fan_in      per-site override of linear_fan_in for MLP linears
    tanh_linear     wrap every linear as tanh(Wx + b); the clf head is exempt
    tanh_mlp        ablate the MLP's SiLU to tanh (the RBM-native nonlinearity)
    remat           gradient-checkpoint each block (as in st)
    """

    vocab: int
    sequence: int
    n_layers: int
    n_embed: int
    aft_kind: str = "conv"
    aft_heads: int = 4
    aft_ksize: int = 4
    dyt_alpha: float = 0.5
    linear_fan_in: int | None = None
    attn_fan_in: int | None = None
    mlp_fan_in: int | None = None
    tanh_linear: bool = False
    tanh_mlp: bool = False
    remat: bool = False

    def fan_in(self, site: str) -> int | None:
        """Per-site fan-in: the site override when set, else the shared value."""
        override = getattr(self, f"{site}_fan_in")
        return override if override is not None else self.linear_fan_in


class Z1T(eqx.Module):
    embedding: eqx.nn.Embedding
    pe: PE
    blocks: list
    norm: DyT
    clf: eqx.Module
    sequence: int = eqx.field(static=True)
    remat: bool = eqx.field(static=True)

    def __init__(self, config: Config, key: Key[Array, ""]):
        keys = jax.random.split(key, config.n_layers + 2)
        dim = config.n_embed
        attn = attention_factory(
            config.aft_kind,
            max_seq=config.sequence,
            heads=config.aft_heads,
            ksize=config.aft_ksize,
            linear=linear_factory(config.fan_in("attn")),
        )
        self.embedding = eqx.nn.Embedding(config.vocab, dim, key=keys[0])
        self.pe = PE(dim, max_seq=config.sequence)
        self.blocks = [
            Block(
                dim,
                key=k,
                attn=attn,
                mlp_linear=linear_factory(config.fan_in("mlp")),
                mlp_act=jax.nn.tanh if config.tanh_mlp else jax.nn.silu,
                dyt_alpha=config.dyt_alpha,
            )
            for k in keys[1 : 1 + config.n_layers]
        ]
        self.norm = DyT(dim, alpha_init=config.dyt_alpha)
        self.clf = eqx.nn.Linear(dim, config.vocab, key=keys[-1])
        self.sequence = config.sequence
        self.remat = config.remat

    def __call__(self, tokens):            # tokens: (T,) int
        x = jax.vmap(self.embedding)(tokens)   # (T, dim)
        x = self.pe(x)                         # add positional encoding once

        def run(blk, h):
            return blk(h)

        apply = eqx.filter_checkpoint(run) if self.remat else run
        for block in self.blocks:
            x = apply(block, x)
        x = self.norm(x)
        logits = jax.vmap(self.clf)(x)         # (T, n_vocab)
        return logits


def active_params(model) -> int:
    """Parameters that actually participate in the forward pass.

    Mirrors `st.active_params`, but for z1t it always equals the raw parameter
    count: `SparseLinear` stores an `(out, k)` weight and gathers its inputs, so
    every stored weight is live. (st's `SparseLinear` keeps a dense weight and
    masks it, so there the two numbers differ.) Kept so run metadata carries the
    same field in both trees.
    """
    return sum(
        int(x.size) for x in jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))
    )
