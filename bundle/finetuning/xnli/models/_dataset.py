"""Pretokenized .bin dataset + dynamic-padding collator."""

import numpy as np
import torch
from torch.utils.data import Dataset


class XNLIDataset(Dataset):
    """Reads a pretokenized .bin file.

    Binary layout:
      int32                              num_examples
      int32[num_examples * context_window] token ids (right-padded with pad_token_id)
      int64[num_examples]                 labels
    """

    def __init__(self, bin_path, context_window, pad_token_id: int = 0):
        super().__init__()
        self.pad_token_id = pad_token_id
        with open(bin_path, "rb") as f:
            num_examples = int(np.fromfile(f, dtype=np.int32, count=1)[0])
            self.data = (
                np.fromfile(f, dtype=np.int32, count=num_examples * context_window)
                .reshape(num_examples, context_window)
                .copy()
            )
            self.labels = np.fromfile(f, dtype=np.int64, count=num_examples).copy()
        print(f"Loaded {len(self.data):,} examples from {bin_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        length = int(np.count_nonzero(row != self.pad_token_id))
        length = max(length, 1)
        return (
            torch.from_numpy(row[:length].copy()).to(torch.int32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class DynamicPaddingCollator:
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_rows, labels = zip(*batch)
        max_len = max(int(row.size(0)) for row in input_rows)
        max_len = max(max_len, 1)

        input_ids = torch.full(
            (len(input_rows), max_len), self.pad_token_id, dtype=torch.int32
        )
        for i, row in enumerate(input_rows):
            input_ids[i, : row.size(0)] = row

        return input_ids, torch.stack(labels)
