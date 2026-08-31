"""Tokenizers for the z1t datasets.

Same interface as `st.tokenizer.ByteTokenizer` — `vocab_size`, `encoder`,
`tokenize(text)`, `decode(tokens)` — so the two trees describe tokenization the
same way. z1t carries two of them, because its datasets are not all byte-level:

    CharTokenizer   a closed character vocabulary (shakespeare, wikitext, addition)
    GPT2Tokenizer   GPT-2 byte-level BPE via tiktoken (openwebtext)

A `CharTokenizer` is constructed from an ordered sequence of unique characters,
not from raw text: the caller decides the ordering, because it fixes the token
ids. The corpus datasets pass `sorted(set(text))`; addition passes its 13-char
table in its own order.
"""

from __future__ import annotations

import numpy as np


class CharTokenizer:
    """Character-level tokenizer over a fixed, ordered character vocabulary.

    Characters outside the vocabulary encode to id 0. That is inherent to a
    closed char vocab and matches what the WikiText loader already did; every
    dataset here builds its vocabulary from the corpus it will train on, so it
    only bites on text from somewhere else.
    """

    def __init__(self, chars):
        # dict.fromkeys dedups while preserving the caller's ordering
        self.chars = list(dict.fromkeys(chars))
        self.vocab_size = len(self.chars)
        self.encoder = {c: i for i, c in enumerate(self.chars)}
        self.decoder = {i: c for i, c in enumerate(self.chars)}
        # codepoint -> id lookup table, so tokenize() is one vectorised gather
        # rather than a Python loop (WikiText-103 is ~500M characters).
        self._lut = np.zeros(max(ord(c) for c in self.chars) + 1, dtype=np.int32)
        for c, i in self.encoder.items():
            self._lut[ord(c)] = i

    def tokenize(self, text: str) -> np.ndarray:
        codes = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
        codes = np.where(codes < len(self._lut), codes, 0)   # OOV -> id 0
        return self._lut[codes].astype(np.int32)

    def decode(self, tokens) -> str:
        return "".join(self.decoder[int(i)] for i in tokens)


class GPT2Tokenizer:
    """GPT-2 byte-level BPE (tiktoken), vocab 50257 — the nanoGPT tokenizer."""

    def __init__(self):
        import tiktoken  # only openwebtext needs it

        self._enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self._enc.n_vocab
        self.eot_token = self._enc.eot_token
        self.encoder = {"<EOT>": self._enc.eot_token}

    def tokenize(self, text: str) -> np.ndarray:
        return np.asarray(self._enc.encode_ordinary(text), dtype=np.int32)

    def decode(self, tokens) -> str:
        return self._enc.decode([int(i) for i in tokens])
