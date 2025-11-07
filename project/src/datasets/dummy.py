from torch.utils.data import Dataset
from typing import Any

import torch


class DummyDataset(Dataset):
    """
    A dummy dataset for testing purposes.
    """
    def __init__(self, length: int = 1000):
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Any:
        return {
            "input": torch.arange(10).float() + idx,
            "target": torch.arange(10).float() + idx + 1,
        }

    @property
    def vocab_size(self) -> Any:
        return self.length + 10
