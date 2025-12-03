import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.datasets.subtraction import SubtractionDataset

# create a small dataset with 3-bit numbers
ds = SubtractionDataset(max_bits=3, split="train", seed=0)

print("Dataset length:", len(ds))
sample = ds[0]

print("Sample 0 input:", sample["input"])
print("Sample 0 target:", sample["target"])
print("Vocab size:", ds.vocab_size)
print("Input length:", len(sample["input"]), " Target length:", len(sample["target"]))
