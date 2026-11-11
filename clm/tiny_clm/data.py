from __future__ import annotations

import numpy as np
import torch


class TokenStream:
    """Single-process, checkpointable permutation sampler; no prefetched workers."""

    def __init__(self, data, seed):
        if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] < 2:
            raise ValueError("expected nonempty token blocks [N,L], L>=2")
        self.data = data
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(len(data), generator=self.generator)
        self.position = 0
        self.epoch = 0

    def next(self, batch_size, device):
        indices = []
        while len(indices) < batch_size:
            if self.position == len(self.order):
                self.order = torch.randperm(len(self.data), generator=self.generator)
                self.position = 0
                self.epoch += 1
            count = min(batch_size-len(indices), len(self.order)-self.position)
            indices.extend(self.order[self.position:self.position+count].tolist())
            self.position += count
        return torch.tensor(np.asarray(self.data[indices]), dtype=torch.long, device=device)

    def state_dict(self):
        return {"rng": self.generator.get_state(), "order": self.order,
                "position": self.position, "epoch": self.epoch, "shape": tuple(self.data.shape)}

    def load_state_dict(self, state):
        if tuple(self.data.shape) != state["shape"]:
            raise ValueError("training data shape changed")
        self.generator.set_state(state["rng"])
        self.order, self.position, self.epoch = state["order"], state["position"], state["epoch"]


def load_blocks(path, vocab_size):
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if data.dtype.kind not in "iu" or data.ndim != 2 or not len(data) or data.shape[1] < 2:
        raise ValueError(f"invalid token blocks: {path}")
    if data.min() < 0 or data.max() >= vocab_size:
        raise ValueError("data contains out-of-vocabulary IDs")
    return data
