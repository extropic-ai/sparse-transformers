"""Dataset loaders for the transformer track.

Four datasets, same interface. Each returns a `Dataset` namespace exposing:

    tokenizer  : a `z1t.tokenizer` instance (vocab_size / tokenize / decode)
    vocab_size : int, from the tokenizer
    get_batch(key, batch_size, max_seq, split="train") -> (inp, tgt)
        Both shaped (batch_size, max_seq), int32, with `tgt = inp shifted by 1`.

`shakespeare`, `wikitext` and `openwebtext` are loaded once from disk
(downloaded, and for the latter two cached in encoded form, on first call).
`addition` is generated on the fly from `key`; max digit count is configurable.

Design choices (locked):
- Addition format: zero-padded, fixed length per sample. At max_n_digits=N
  every row is exactly `3N + 4` chars: "x+y=sum\\n" with x,y left-padded
  to N digits and sum left-padded to N+1 digits. No reverse-target trick.
- Vocabulary for addition is the 13 chars "0123456789+=\\n".
- For every dataset, train and val differ only in the underlying RNG / data
  slice; `get_batch(..., split="val")` returns held-out windows.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int, Key

from z1t.tokenizer import CharTokenizer, GPT2Tokenizer


@dataclass
class Dataset:
    tokenizer: object
    get_batch: Callable[..., tuple[jnp.ndarray, jnp.ndarray]]
    # Optional fields for callers (model sequence sizing, naming, etc.)
    name: str = ""
    sample_len: int | None = None  # for addition; None for free-length corpora

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size


# ---------- tiny-Shakespeare ---------------------------------------------

_SHAKE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)


def make_shakespeare(
    data_dir: str | Path = "./data",
    val_frac: float = 0.1,
) -> Dataset:
    """Char-level tiny-Shakespeare. Downloads on first call."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "tiny_shakespeare.txt"
    if not path.exists():
        print(f"downloading {_SHAKE_URL} -> {path}")
        urllib.request.urlretrieve(_SHAKE_URL, path)

    text = path.read_text()
    tokenizer = CharTokenizer(sorted(set(text)))

    # Materialise the full corpus as a jax array once. ~1 MB for tiny-Shakespeare.
    data = jnp.asarray(tokenizer.tokenize(text))
    n_val = int(val_frac * data.shape[0])
    train_data = data[:-n_val]
    val_data = data[-n_val:]

    def get_batch(key, batch_size, max_seq, split="train"):
        d = train_data if split == "train" else val_data
        max_start = d.shape[0] - max_seq - 1
        # OWT train is ~9e9 tokens > int32; sample int64 starts on the host.
        rng = np.random.default_rng(int(jax.random.bits(key)))
        starts = rng.integers(0, max_start, size=batch_size)          # int64
        inp = np.stack([np.asarray(d[s: s + max_seq], dtype=np.int32) for s in starts])
        tgt = np.stack([np.asarray(d[s + 1: s + 1 + max_seq], dtype=np.int32) for s in starts])
        return jnp.asarray(inp), jnp.asarray(tgt)

    return Dataset(
        tokenizer=tokenizer,
        get_batch=get_batch,
        name="shakespeare",
    )


# ---------- WikiText-103 (char-level) ------------------------------------

# Canonical raw-text zip from WikiText's original author (Stephen Merity).
# Extracts to wikitext-103-raw/{wiki.train.raw,wiki.valid.raw,wiki.test.raw}.
_WIKITEXT103_URL = "https://wikitext.smerity.com/wikitext-103-raw-v1.zip"


def make_wikitext(
    data_dir: str | Path = "./data",
    val_frac: float = 0.1,            # ignored: WikiText ships an official split
    n_tokens: int | None = None,
) -> Dataset:
    """Char-level WikiText-103 — the large natural-language token-scaling axis.

    Downloads + extracts on first call, then caches the char-encoded corpus to
    ``<data_dir>/wikitext-103-char.npz`` (the ~500M-char encode is the slow part,
    so we only do it once). Uses the official train/valid split; ``val_frac`` is
    ignored. Vocab is the few-hundred distinct chars in train ∪ valid.

    Data-scaling knob (mirrors ``make_addition``): ``n_tokens > 0`` truncates the
    *train* split to that many characters, so you can sweep dataset size on real
    text. ``None`` / 0 uses the full corpus. The val split is always the full
    official valid set, so it measures held-out generalization either way.
    """
    import shutil, zipfile  # only needed on the first (download/extract) call

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / "wikitext-103-char.npz"

    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        train_data, val_data, chars = z["train"], z["val"], list(z["chars"])
    else:
        raw_dir = data_dir / "wikitext-103-raw"
        train_raw = raw_dir / "wiki.train.raw"
        if not train_raw.exists():
            zip_path = data_dir / "wikitext-103-raw-v1.zip"
            if not zip_path.exists():
                print(f"downloading {_WIKITEXT103_URL} -> {zip_path}")
                # The host 403s the default Python-urllib UA; send a browser one.
                req = urllib.request.Request(
                    _WIKITEXT103_URL, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as r, open(zip_path, "wb") as f:
                    shutil.copyfileobj(r, f)
            print(f"extracting {zip_path}")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(data_dir)
        train_text = train_raw.read_text(encoding="utf-8")
        val_text = (raw_dir / "wiki.valid.raw").read_text(encoding="utf-8")

        # Shared vocab over train ∪ valid; CharTokenizer encodes via a LUT.
        chars = sorted(set(train_text) | set(val_text))
        tok = CharTokenizer(chars)
        train_data, val_data = tok.tokenize(train_text), tok.tokenize(val_text)
        print(f"caching encoded corpus -> {cache}")
        np.savez(cache, train=train_data, val=val_data,
                 chars=np.array(chars, dtype=object))

    if n_tokens is not None and n_tokens > 0:
        train_data = train_data[: int(n_tokens)]
        name = f"wikitext-103-nt{int(n_tokens)}"
    else:
        name = "wikitext-103"

    tokenizer = CharTokenizer(chars)

    def get_batch(
        key: Key,
        batch_size: int,
        max_seq: int,
        split: str = "train",
    ) -> tuple[Int[Array, "B T"], Int[Array, "B T"]]:
        # Corpus stays on host (~2 GB); gather windows host-side, ship the batch.
        d = train_data if split == "train" else val_data
        max_start = d.shape[0] - max_seq - 1
        starts = np.asarray(jax.random.randint(key, (batch_size,), 0, max_start))
        inp = np.stack([d[s: s + max_seq] for s in starts])
        tgt = np.stack([d[s + 1: s + 1 + max_seq] for s in starts])
        return jnp.asarray(inp), jnp.asarray(tgt)

    return Dataset(
        tokenizer=tokenizer,
        get_batch=get_batch,
        name=name,
    )


# ---------- OpenWebText (GPT-2 BPE) --------------------------------------
#
# The dataset Karpathy's nanoGPT trains GPT-2 on. GPT-2 byte-level BPE
# (tiktoken, vocab 50257), tokenized once into uint16 train/val .bin memmaps —
# same on-disk format as nanoGPT's prepare.py. Token-level, not char-level.

_OPENWEBTEXT_HF = "Skylion007/openwebtext"


def prepare_openwebtext(
    data_dir: Path,
    val_frac: float = 0.0005,
    num_proc: int = 8,
    n_shards: int = 1024,
    smoke: bool = False,
    progress: bool = True,
) -> dict:
    """One-time: download OWT, GPT-2-tokenize, write {train,val}.bin (uint16).

    Faithful port of nanoGPT/data/openwebtext/prepare.py. Heavy: ~12 GB
    download + tokenizing ~9B tokens. Each doc is ``encode_ordinary`` + an EOT
    separator; tokens are streamed into a memmap in shards to bound memory.

    Parameters
    ----------
    data_dir
        Output dir; ``openwebtext-{train,val}.bin`` land here.
    val_frac
        Held-out fraction; nanoGPT uses ``0.0005`` (≈4.5 M val tokens).
    num_proc
        HF ``map`` workers. Bump on the cluster (e.g. 32, 64).
    n_shards
        How many shards the tokenised dataset is iterated over when filling the
        memmap. Bounds memory; does not affect the on-disk layout.
    smoke
        If ``True``, restrict the source dataset to the first 1024 documents.
        Produces real (tiny) bins in seconds — for CLI / cluster sanity checks.
    progress
        If ``True``, render a tqdm bar over the shard-write loop with
        tokens/s + ETA. The download and tokenisation phases already emit
        HF's own bars; this hook is the one for the long silent write phase.

    Returns
    -------
    dict
        ``{"train_tokens": int, "val_tokens": int, "train_path": Path,
        "val_path": Path}`` — handy for callers that want to log / report.
    """
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    print(f"loading {_OPENWEBTEXT_HF} via HF datasets (num_proc={num_proc})...")
    dataset = load_dataset(_OPENWEBTEXT_HF, num_proc=num_proc)
    if smoke:
        n = min(1024, len(dataset["train"]))
        print(f"smoke mode: restricting to first {n} documents")
        dataset["train"] = dataset["train"].select(range(n))
    split = dataset["train"].train_test_split(test_size=val_frac, seed=2357, shuffle=True)
    split["val"] = split.pop("test")

    def process(ex):
        ids = enc.encode_ordinary(ex["text"])
        ids.append(enc.eot_token)                       # doc separator
        return {"ids": ids, "len": len(ids)}

    tokenized = split.map(process, remove_columns=["text"],
                          desc="tokenizing OWT", num_proc=num_proc)

    stats: dict = {}
    for sp, dset in tokenized.items():
        arr_len = int(np.sum(dset["len"], dtype=np.uint64))
        path = data_dir / f"openwebtext-{sp}.bin"
        print(f"writing {arr_len:,} tokens ({arr_len * 2 / 1e9:.2f} GB uint16) -> {path}")
        arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(arr_len,))
        # Use n_shards no larger than the dataset (smoke mode can have <1024 docs).
        n_sh = max(1, min(n_shards, len(dset)))
        idx = 0
        bar = None
        if progress:
            try:
                from tqdm.auto import tqdm
                bar = tqdm(total=arr_len, unit="tok", unit_scale=True,
                           desc=f"writing {sp}.bin", dynamic_ncols=True)
            except ImportError:
                bar = None
        for s in range(n_sh):
            batch = dset.shard(num_shards=n_sh, index=s, contiguous=True).with_format("numpy")
            a = np.concatenate(batch["ids"])
            arr[idx: idx + len(a)] = a
            idx += len(a)
            if bar is not None:
                bar.update(len(a))
                bar.set_postfix(shard=f"{s + 1}/{n_sh}", refresh=False)
        if bar is not None:
            bar.close()
        arr.flush()
        stats[f"{sp}_tokens"] = arr_len
        stats[f"{sp}_path"] = path
    return stats


# Back-compat alias for any out-of-tree caller of the old private name.
_prepare_openwebtext = prepare_openwebtext


def make_openwebtext(
    data_dir: str | Path = "./data",
    val_frac: float = 0.0005,         # nanoGPT's OWT held-out fraction
    n_tokens: int | None = None,
    num_proc: int = 8,
) -> Dataset:
    """GPT-2-BPE OpenWebText — the nanoGPT GPT-2 corpus.

    Prepares {train,val}.bin (uint16 token memmaps) on first call, then memmaps
    them and samples random windows host-side (same interface as the other
    datasets). Vocab is GPT-2's 50257. Honors the --n-tokens data-scaling knob
    (truncates the train memmap); val is always the full held-out split.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_bin = data_dir / "openwebtext-train.bin"
    val_bin = data_dir / "openwebtext-val.bin"
    if not (train_bin.exists() and val_bin.exists()):
        print("preparing OpenWebText (one-time: ~12GB download + GPT-2 "
              "tokenization of ~9B tokens; this can take a while)...")
        prepare_openwebtext(data_dir, val_frac=val_frac, num_proc=num_proc,
                            progress=True)

    tokenizer = GPT2Tokenizer()
    train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_bin, dtype=np.uint16, mode="r")

    if n_tokens is not None and n_tokens > 0:
        train_data = train_data[: int(n_tokens)]
        name = f"openwebtext-nt{int(n_tokens)}"
    else:
        name = "openwebtext"

    def get_batch(key, batch_size, max_seq, split="train"):
        d = train_data if split == "train" else val_data
        max_start = d.shape[0] - max_seq - 1
        # OWT train is ~9e9 tokens > int32; sample int64 starts on the host.
        rng = np.random.default_rng(int(jax.random.bits(key)))
        starts = rng.integers(0, max_start, size=batch_size)          # int64
        inp = np.stack([np.asarray(d[s: s + max_seq], dtype=np.int32) for s in starts])
        tgt = np.stack([np.asarray(d[s + 1: s + 1 + max_seq], dtype=np.int32) for s in starts])
        return jnp.asarray(inp), jnp.asarray(tgt)

    return Dataset(
        tokenizer=tokenizer,
        get_batch=get_batch,
        name=name,
    )


# ---------- synthetic addition -------------------------------------------

# Ordered, not sorted: the table order fixes the token ids.
_ADD_TOKENIZER = CharTokenizer(_ADD_CHARS := "0123456789+=\n")
_ADD_STOI = _ADD_TOKENIZER.encoder


def _addition_sample_len(max_n_digits: int) -> int:
    # "<x:N>+<y:N>=<sum:N+1>\n" -> N + 1 + N + 1 + (N+1) + 1 = 3N + 4
    return 3 * max_n_digits + 4


def _format_addition_batch(
    xs: np.ndarray, ys: np.ndarray, max_n_digits: int
) -> np.ndarray:
    """Format a batch of (x, y) pairs as zero-padded "x+y=sum\\n" strings.

    Returns an int32 ndarray of shape (batch, sample_len).
    """
    N = max_n_digits
    L = _addition_sample_len(N)
    batch = xs.shape[0]
    out = np.empty((batch, L), dtype=np.int32)
    plus = _ADD_STOI["+"]
    eq = _ADD_STOI["="]
    nl = _ADD_STOI["\n"]
    for b in range(batch):
        s = f"{int(xs[b]):0{N}d}+{int(ys[b]):0{N}d}={int(xs[b]) + int(ys[b]):0{N + 1}d}\n"
        assert len(s) == L, (s, L)
        for i, c in enumerate(s):
            out[b, i] = _ADD_STOI[c]
        # sanity: positions of '+', '=', '\n' are deterministic
        assert out[b, N] == plus
        assert out[b, 2 * N + 1] == eq
        assert out[b, L - 1] == nl
    return out


def make_addition(
    max_n_digits: int = 5,
    n_tokens: int | None = None,
    pool_seed: int = 0,
) -> Dataset:
    """Synthetic char-level addition.

    Each sample is one zero-padded equation. `get_batch` returns a
    `(batch, sample_len-1)` int32 array (the input prefix) and the
    one-step-shifted target.

    Two data regimes, switched by `n_tokens` (the data-scaling knob):

    - `n_tokens` is None / <= 0  →  **unlimited stream** (default). Every
      train batch is freshly sampled from the full 10^N x 10^N pair space,
      so the model effectively never repeats data.
    - `n_tokens > 0`  →  **finite training pool**. A fixed set of
      `n_samples = n_tokens // sample_len` equations is pre-generated once
      (seeded by `pool_seed`, independent of the model/train RNG) and every
      train batch is drawn *with replacement* from that pool. This is the
      lever for data-scaling studies: hold the architecture fixed and sweep
      `n_tokens` to trace generalization vs. dataset size.

    In both regimes the **val** split is an independent held-out stream
    (fresh random pairs, never the train pool), so val always measures
    generalization. The (x, y) space (10^10 at N=5) makes train/val
    collisions negligible.
    """
    N = max_n_digits
    L = _addition_sample_len(N)
    upper = 10**N

    finite = n_tokens is not None and n_tokens > 0
    if finite:
        n_samples = max(1, int(n_tokens) // L)
        # Pre-generate the fixed pool once, host-side, from a dedicated RNG
        # so the training pool is reproducible and decoupled from model init.
        rng = np.random.default_rng(pool_seed)
        pool_x = rng.integers(0, upper, size=n_samples, dtype=np.int64)
        pool_y = rng.integers(0, upper, size=n_samples, dtype=np.int64)
        pool_seq = jnp.asarray(_format_addition_batch(pool_x, pool_y, N))  # (n_samples, L)
        name = f"addition-d{N}-nt{int(n_tokens)}-n{n_samples}"
    else:
        n_samples = None
        pool_seq = None
        name = f"addition-d{N}"

    def get_batch(
        key: Key,
        batch_size: int,
        max_seq: int | None = None,
        split: str = "train",
    ) -> tuple[Int[Array, "B T"], Int[Array, "B T"]]:
        # `max_seq` is allowed but ignored: every sample is exactly L chars.
        # `split` differentiates RNG so the val draw is a held-out stream.
        if finite and split == "train":
            # Draw indices into the fixed pool (with replacement).
            idx = jax.random.randint(key, (batch_size,), 0, n_samples)
            seq = pool_seq[idx]                                  # (B, L)
            return seq[:, :-1], seq[:, 1:]
        if split == "val":
            # Differentiate val RNG so the held-out draw is from a separate
            # stream — disjoint from the finite train pool by construction.
            key = jax.random.fold_in(key, 0xDA7A)
        kx, ky = jax.random.split(key)
        xs = jax.random.randint(kx, (batch_size,), 0, upper)
        ys = jax.random.randint(ky, (batch_size,), 0, upper)
        # Host-side format (we're outside the jit boundary anyway).
        xs_np = np.asarray(xs)
        ys_np = np.asarray(ys)
        seq_np = _format_addition_batch(xs_np, ys_np, N)
        seq = jnp.asarray(seq_np)  # (B, L)
        inp = seq[:, :-1]
        tgt = seq[:, 1:]
        return inp, tgt

    return Dataset(
        tokenizer=_ADD_TOKENIZER,
        get_batch=get_batch,
        name=name,
        sample_len=L,
    )


# ---------- quick self-check ---------------------------------------------

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    print("== addition (unlimited stream) ==")
    add = make_addition(max_n_digits=5)
    inp, tgt = add.get_batch(key, batch_size=4)
    print(f"name = {add.name}, vocab_size = {add.vocab_size}, sample_len = {add.sample_len}")
    print(f"inp shape  = {inp.shape}, tgt shape = {tgt.shape}")
    print("first row decoded:")
    print(repr(add.tokenizer.decode(inp[0])))
    print(repr(add.tokenizer.decode(tgt[0])))

    print("\n== addition (finite pool, n_tokens=2048) ==")
    add_small = make_addition(max_n_digits=3, n_tokens=2048, pool_seed=0)
    inp, tgt = add_small.get_batch(key, batch_size=4, split="train")
    vinp, vtgt = add_small.get_batch(key, batch_size=4, split="val")
    print(f"name = {add_small.name}, sample_len = {add_small.sample_len}")
    print(f"train inp shape = {inp.shape}, val inp shape = {vinp.shape}")
    print("first train row:", repr(add_small.tokenizer.decode(inp[0])))
    print("first val   row:", repr(add_small.tokenizer.decode(vinp[0])))

    print("\n== shakespeare ==")
    shake = make_shakespeare()
    inp, tgt = shake.get_batch(key, batch_size=2, max_seq=64)
    print(f"vocab_size = {shake.vocab_size}")
    print(f"inp shape  = {inp.shape}, tgt shape = {tgt.shape}")
    print("first row decoded:")
    print(repr(shake.tokenizer.decode(inp[0])))
