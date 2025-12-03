from torch.utils.data import Dataset
from typing import Any

import math
import numpy as np
import torch


class SubtractionDataset(Dataset):
    """
    Simple N-bit subtraction dataset.
    Similar structure to string_copy.py.
    """

    def __init__(
        self,
        max_bits: int,
        split: str,
        train_val_ratio: float = 0.8,
        seed: int = 0,
    ):
        # Same idea as CopyDataset checks
        assert 0.0 < train_val_ratio < 1.0
        assert split in ("train", "val")

        self.max_bits = max_bits
        self.split = split
        self.train_val_ratio = train_val_ratio

        # Numbers range: 0 to (2^max_bits - 1)
        self.max_number = 2 ** self.max_bits

        # Create all (a, b) pairs where a >= b
        all_pairs = []
        for a in range(self.max_number):
            for b in range(self.max_number):
                if a >= b:
                    all_pairs.append((a, b))

        self.num_instances = len(all_pairs)

        # Train/val split, similar style to string_copy.py
        self.num_train = math.ceil(self.num_instances * self.train_val_ratio)
        self.num_val = self.num_instances - self.num_train

        rng = np.random.RandomState(seed)
        self.instance_idxes = rng.permutation(self.num_instances)

        if split == "train":
            self.instance_idxes = self.instance_idxes[:self.num_train]
        else:
            self.instance_idxes = self.instance_idxes[self.num_train:]

        self.all_pairs = all_pairs

        # Token IDs
        self.TOK_0 = 0
        self.TOK_1 = 1
        self.TOK_MINUS = 2
        self.TOK_EQUAL = 3
        self.TOK_EOS = 4

    def __len__(self) -> int:
        return len(self.instance_idxes)

    def number_to_bits(self, number: int):
        """
        Convert integer to list of bits padded to max_bits.
        Simple beginner method using bin().
        """
        bits = list(bin(number)[2:])   # remove '0b'
        bits = [int(b) for b in bits]

        # pad left to reach max_bits
        while len(bits) < self.max_bits:
            bits = [0] + bits

        return bits

    def __getitem__(self, idx: int) -> Any:
        """
        Build sequence:
            a_bits <MINUS> b_bits <EQUAL> result_bits <EOS>...
        Then make:
            input = seq[:-1]
            target = seq[1:]
        """

        instance = self.instance_idxes[idx]
        a, b = self.all_pairs[instance]
        r = a - b

        a_bits = self.number_to_bits(a)
        b_bits = self.number_to_bits(b)
        r_bits = self.number_to_bits(r)

        seq = (
            a_bits
            + [self.TOK_MINUS]
            + b_bits
            + [self.TOK_EQUAL]
            + r_bits
        )

        question_len = len(a_bits) + 1 + len(b_bits)  # up to and including EQUAL
        answer_len = len(r_bits) # result bits

        # Fixed length: 3*max_bits + 4 (simple padding rule)
        desired_len = 3 * self.max_bits + 4

        while len(seq) < desired_len:
            seq.append(self.TOK_EOS)

        input_tokens = seq[:-1]
        target_tokens = seq[1:]

        return {
            "input": torch.tensor(input_tokens),
            "target": torch.tensor(target_tokens),
            "question_len": torch.tensor(question_len, dtype=torch.long),
            "answer_len": torch.tensor(answer_len, dtype=torch.long),
        }

    @property
    def eos_token(self) -> Any:
        return self.TOK_EOS

    @property
    def vocab_size(self):
        return 5
