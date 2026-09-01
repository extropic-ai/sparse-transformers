import os
from abc import abstractmethod

import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Int


class AbstractLoader:

    @abstractmethod
    def sample_train_batch(self, batch_size: int) -> Int[Array, "batch sequence"]: ...

    @abstractmethod
    def sample_val_batch(self, batch_size: int) -> Int[Array, "batch sequence"]: ...


class TextDataLoader(AbstractLoader):

    def __init__(self, train_path, val_path, sequence):
        mm = None if os.environ.get("ST_DATA_IN_RAM") else "r"
        self.train = np.load(train_path, mmap_mode=mm)
        self.val = np.load(val_path, mmap_mode=mm)
        self.sequence = sequence

    def _sample(self, tokens, batch_size) -> Int[Array, "batch sequence"]:
        starts = np.random.randint(
            0,
            len(tokens) - self.sequence,
            size=batch_size,
        )
        offsets = np.arange(self.sequence + 1)
        batch = tokens[starts[:, None] + offsets[None, :]]
        return jnp.asarray(batch.astype(np.int32))

    def sample_train_batch(self, batch_size) -> Int[Array, "batch sequence"]:
        return self._sample(self.train, batch_size)

    def sample_val_batch(self, batch_size) -> Int[Array, "batch sequence"]:
        return self._sample(self.val, batch_size)
