from torch.utils.data import Dataset
from typing import Any

import numpy as np
import torch

class AdditionDataset(Dataset):
    """
    Dataset for digit-wise addition designed specifically for length generalization.

    - Train on shorter numbers     (e.g., 1–3 digits)
    - Validate on longer numbers   (e.g., 4–6 digits)

    Each example is tokenized as:
        input:  digits(a) + [PLUS] + digits(b) + [EOS...]
        target: digits(a+b) + [EOS...]

    Tokens:
        0–9 : digit tokens
        10  : '+'
        11  : EOS
    """

    def __init__(
        self,
        train_max_digits: int,
        test_max_digits: int,
        split: str,
        num_samples: int,
        seed: int = 0,
    ):
        assert split in ("train", "val")

        self.train_max_digits = train_max_digits
        self.test_max_digits = test_max_digits
        self.split = split
        self.num_samples = num_samples
        self.rng = np.random.RandomState(seed)

        # Tokens
        self.PLUS = 10
        self.EOS = 11

        # Choose the number range based on split, different digit ranges to avoid leakage
        if split == "train":
            self.min_digits = 1
            self.max_digits = train_max_digits   # short lengths
        else:
            self.min_digits = 1
            self.max_digits = test_max_digits    # longer lengths

        # Maximum sequence lengths for padding
        self.max_input_len = self.max_digits * 2 + 3   # a + b + EOS + margins
        self.max_output_len = self.max_digits + 2      # result + EOS

    def __len__(self):
        return self.num_samples

    def _sample_number(self, num_digits):
        """Sample a number uniformly from the allowed digit size."""
        if num_digits == 1:
            low = 0
            high = 10
        else:
            low = 10 ** (num_digits - 1)
            high = 10 ** num_digits
        return self.rng.randint(low, high)

    def __getitem__(self, idx: int) -> Any:
        # Randomly choose the digit-length for this instance
        d = self.rng.randint(self.min_digits, self.max_digits + 1)

        # Sample two numbers of chosen digit-length
        a = self._sample_number(d)
        b = self._sample_number(d)
        s = a + b  # sum

        # Convert numbers to token lists
        a_digits = [int(x) for x in str(a)]
        b_digits = [int(x) for x in str(b)]
        s_digits = [int(x) for x in str(s)]

        # Construct input: a + b + EOS + padding
        input_tokens = (
            a_digits +
            [self.PLUS] +
            b_digits +
            [self.EOS]
        )

        # Construct target: sum + EOS + padding
        target_tokens = s_digits + [self.EOS]

        # Pad to fixed lengths
        input_tokens += [self.EOS] * (self.max_input_len - len(input_tokens))
        target_tokens += [self.EOS] * (self.max_output_len - len(target_tokens))

        # Shift like CopyDataset (next-token prediction)
        return {
            "input": torch.tensor(input_tokens[:-1]),
            "target": torch.tensor(input_tokens[1:]),
            "sum_tokens": torch.tensor(target_tokens),          # optional (for direct supervision)
            "digits_used": torch.tensor(d),
        }

    @property
    def eos_token(self) -> Any:
        return self.EOS

    @property
    def vocab_size(self):
        # digits 0–9 + '+' + EOS
        return 12


if __name__ == "__main__":
    ds = AdditionDataset(
        train_max_digits=3,
        test_max_digits=6,
        split="train",
        num_samples=5,
        seed=0,
    )

    print("Testing AdditionDataset...\n")

    for i in range(len(ds)):
        sample = ds[i]
        print(f"Sample {i}:")
        print("  input       :", sample["input"])
        print("  target      :", sample["target"])
        print("  sum_tokens  :", sample["sum_tokens"])
        print("  digits_used :", sample["digits_used"])
        print()
