from torch.nn import Module


class DummyModel(Module):
    """
    A dummy model for testing purposes.
    """
    def __init__(self):
        super(DummyModel, self).__init__()

    def forward(self, batch):
        return {
            "output": batch["input"],
        }
