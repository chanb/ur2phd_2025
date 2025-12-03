import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.datasets.dummy import DummyDataset
from src.datasets.string_copy import CopyDataset
from src.datasets.addition import AdditionDataset
from src.datasets.subtraction import SubtractionDataset
