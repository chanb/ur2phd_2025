from torch.utils.data import Dataset
from typing import Any

import torch


class MultiplicationDataset(Dataset):
    """
    A dataset that generates integer multiplication problems
    """
    def __init__(self, a_max=999, b_max=99, eos_token="<EOS>", input_len=7, output_len=5):
        self.a_max = a_max  # integer upper limit for first multiplicand
        self.b_max = b_max  # integer upper limit for second multiplicand
        self.eos_token = eos_token      # EOS token for padding
        self.input_len = input_len      # length of input sequence
        self.output_len = output_len    # length of output sequence

        # generate pairs of multiplicands
        self.pairs = [(a, b) for a in range(0, a_max + 1) for b in range(0, b_max + 1)]
        self.length = len(self.pairs)

        # treat each character as a token
        # dictionary mapping string digits to integer values
        # tensor needs all elements to be type int
        self.token_idx = {str(i): i for i in range(10)}   # 10 tokens: digit 0-9
        self.token_idx["*"] = 10                             # 1 token: "*"
        self.token_idx["="] = 11                             # 1 token: "="
        self.token_idx[self.eos_token] = 12                  # 1 token: "<EOS>"

    def __len__(self) -> int:
        # return the length of the dataset
        return self.length

    def __getitem__(self, idx: int) -> Any:
        # this returns an integer multiplication problem, with the dictionary format
        a, b = self.pairs[idx]
        product = a * b

        # treat each character as a token
        input_tokens = list(str(a)) + ["*"] + list(str(b)) + ["="] + [self.eos_token]
        output_tokens = list(str(product)) + [self.eos_token]

        # if sequence is too long, truncate it
        input_tokens = input_tokens[:self.input_len]
        output_tokens = output_tokens[:self.output_len]

        # if sequence is too short, pad it with EOS tokens
        input_tokens += [self.eos_token] * (self.input_len - len(input_tokens))
        output_tokens += [self.eos_token] * (self.output_len - len(output_tokens))

        # convert all tokens to their corresponding indices using self.token_idx
        input_list = [self.token_idx[token] for token in input_tokens]
        output_list = [self.token_idx[token] for token in output_tokens]

        return {
            "input": torch.tensor(input_list, dtype=torch.long),
            "output":torch.tensor(output_list, dtype=torch.long)
        }


    @property
    def vocab_size(self):
        return len(self.token_idx)
    

if __name__ == "__main__":

    d1 = MultiplicationDataset(a_max=2, b_max=2, eos_token="<EOS>") # test  with small dataset
    for i in range(len(d1)):   
        sample = d1[i]
        input = sample["input"]
        output = sample["output"]
        print(f"Sample {i} ================")
        print(f"input: {input}")
        print(f"output: {output}", end="\n\n")
  
    d2 = MultiplicationDataset(a_max=9, b_max=9, eos_token="<EOS>") # test with larger dataset
    for i in range(len(d2)):   
        sample = d2[i]
        input = sample["input"]
        output = sample["output"]
        print(f"Sample {i} ================")
        print(f"input: {input}")
        print(f"output: {output}", end="\n\n")