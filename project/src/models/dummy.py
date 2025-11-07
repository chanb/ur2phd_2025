from torch.nn import Module, Embedding


class DummyModel(Module):
    """
    A dummy model for testing purposes.
    """
    def __init__(self, vocab_size: int, *args, **kwargs,):
        super(DummyModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding = Embedding(vocab_size, 16)

    def forward(self, batch):
        return {
            "output": batch["input"],
        }
