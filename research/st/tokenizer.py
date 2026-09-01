import numpy as np


class ByteTokenizer:
    def __init__(self):
        self.pad_id = 0
        self.vocab_size = 257
        self.encoder = {"<PAD>": 0}

    def tokenize(self, text: str):
        return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32) + 1

    def decode(self, tokens):
        tokens = np.asarray(tokens)
        tokens = tokens[tokens != self.pad_id]
        raw = np.clip(tokens - 1, 0, 255).astype(np.uint8).tobytes()
        return raw.decode("utf-8", errors="replace")
