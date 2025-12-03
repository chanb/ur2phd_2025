from torch.utils.data import Dataset
import numpy as np
from typing import Any

import torch


class MultiplicationDataset(Dataset):
    """
    Dataset for digit-wise multiplication designed for length generalization.

    credits to:
    https://github.com/chanb/ur2phd_2025/blob/tanvi-tasks/project/src/datasets/addition.py

    - Train on shorter numbers     (e.g., 1–3 digits)
    - Validate on longer numbers   (e.g., 4–6 digits)

    Tokens:
        0-9 : digit tokens
        10  : '*'
        11  : '='
        12  : EOS
    """
    def __init__(
        self,
        train_max_digits: int,
        test_max_digits: int,
        split: str,
        num_samples: int,
        seed: int = 0,
        eos_token: str = "<EOS>"
    ):
       
        assert split in ("train", "heldout"), "split must be 'train' or 'heldout'"
    
        self.split = split
        self.num_samples = num_samples
        self.rng = np.random.RandomState(seed)
        self.eos_token = eos_token

        # token mapping
        self.token_idx = {str(i): i for i in range(10)}
        self.token_idx.update({"*": 10, "=": 11, eos_token: 12})
    
        # choose digit range based on split
        if split == "train":
            self.max_digits = train_max_digits    # train on short lengths
        else:
            self.max_digits = test_max_digits     # train on longer lengths

        # max sequence lengths for padding
        self.input_len = self.max_digits * 2 + 4   # a * b 
        self.output_len = self.max_digits * 2 + 2 # product

    def __len__(self) -> int:
        # return the length of the dataset
        return self.num_samples
    
    def sample_integer(self, num_digits: int) -> int:
        """Sample a number uniformly with given digit length."""
        low = 10 ** (num_digits - 1)
        high = 10 ** num_digits - 1
        return self.rng.randint(low, high)

    def __getitem__(self, idx: int) -> Any:
        # sample a number uniformly with given digit length
        d = self.rng.randint(1, self.max_digits + 1)

        # sample two integers of chosen digit length then multiply them
        m = self.sample_integer(d)
        n = self.sample_integer(d)
        product = m * n

        # tokenize
        input_tokens = list(str(m)) + ["*"] + list(str(n)) + ["="] + [self.eos_token]
        output_tokens = list(str(product)) + [self.eos_token]

        # pad or truncate
        input_tokens = input_tokens[:self.input_len] + [self.eos_token] * (self.input_len - len(input_tokens))
        output_tokens = output_tokens[:self.output_len] + [self.eos_token] * (self.output_len - len(output_tokens))

        # convert all tokens to their corresponding indices using self.token_idx
        input_list = [self.token_idx[token] for token in input_tokens]
        output_list = [self.token_idx[token] for token in output_tokens]

        return {
            "input": torch.tensor(input_list, dtype=torch.long),
            "output":torch.tensor(output_list, dtype=torch.long),
            "digits_used": d
        }


    @property
    def vocab_size(self):
        return len(self.token_idx)
    

if __name__ == "__main__":
    
    d1 = MultiplicationDataset(
        train_max_digits=3,
        test_max_digits=6,
        split="train",
        num_samples=5,
        seed=0,
    )

    print("Testing MultiplicationDataset...\n")
    for i in range(len(d1)):
        sample = d1[i]
        print(f"Sample {i}:")
        print("input :", sample["input"].tolist())
        print("output:", sample["output"].tolist())
        print("digits_used:", sample["digits_used"])
        print()
