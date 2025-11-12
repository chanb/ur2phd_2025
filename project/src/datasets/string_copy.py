from torch.utils.data import Dataset
from typing import Any

import math
import numpy as np


class StringCopyDataset(Dataset):
    """
    A dataset representing the string copy task.
    Here, the goal is to repeat the input string.
    We assume binary strings, represented using 0s and 1s.
    """
    def __init__(
        self,
        max_question_len: int,
        split: str,
        train_val_ratio: float=0.8,
        seed: int = 0,
    ):
        assert 0.0 < train_val_ratio < 1.0
        assert split in ("train", "val")
        self.max_question_len = max_question_len
        self.split = split
        self.train_val_ratio = train_val_ratio
        self.num_instances = 2 ** self.max_question_len
        self.num_train = math.ceil(
            self.num_instances * self.train_val_ratio
        )
        self.num_val = self.num_instances - self.num_train

        # Generate instance indices for sampling from a specific set
        rng = np.random.RandomState(seed)
        self.instance_idxes = rng.permutation(self.num_instances)
        if split == "train":
            self.instance_idxes = self.instance_idxes[:self.num_train]
        elif split == "val":
            self.instance_idxes = self.instance_idxes[self.num_train:]
        else:
            raise ValueError("No split {}".format(split))

    def __len__(self) -> int:
        return len(self.instance_idxes)

    def __getitem__(self, idx: int) -> Any:
        """
        Expected format for supervised learning with next-token prediction:
        {
            "input": [<0>, <1>, <1>, <EQUAL>, <0>, <1>, <1>],
            "target": [<1>, <1>, <EQUAL>, <0>, <1>, <1>, <EOS>]
        }
        """

        instance = self.instance_idxes[idx]
        bin_repr = "{0:b}".format(instance)
        list_repr = [int(token) for token in bin_repr]
        list_repr = list_repr + [2] + list_repr
        eos_pads = [3] * (2 * (self.max_question_len + 1) - len(list_repr))
        list_repr = list_repr + eos_pads

        return {
            "input": list_repr[:-1],
            "target": list_repr[1:],
        }

    @property
    def vocab_size(self) -> Any:
        # <0>, <1>, <EQUAL>, <EOS>
        return 4
