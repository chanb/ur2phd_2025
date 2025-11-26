import torch


from torch import nn


class DecoderGRU(nn.Module):
    """
    An GRU decoder
    XXX: We use a gated recurrent unit (GRU) because it handles some vanishing gradient problems
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_gru_layers: int,
        dropout_p: float=0.1,
    ):
        super(DecoderGRU, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_gru_layers = num_gru_layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
        )
        self.hidden_1 = nn.Linear(hidden_size, hidden_size)
        self.hidden_2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, batch):
        """
        Predicts next tokens given input sequences, i.e., p(x_t | x_1, ..., x_{t-1}).

        Expects batch to include:
        - batch["input"]: a tensor of shape (batch_size, seq_len)
        """
        embedded = self.embedding(batch["input"])  # (batch_size, seq_len, hidden_size)
        if "state" in batch:
            rnn_out, rnn_state = self.rnn(embedded, batch["state"])
        else:
            rnn_out, rnn_state = self.rnn(embedded)  # (batch_size, seq_len, hidden_size)

        hidden = self.hidden_1(rnn_out)
        hidden = nn.functional.relu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.hidden_2(hidden)
        hidden = nn.functional.relu(hidden)
        hidden = self.dropout(hidden)
        output_logits = self.out(hidden)  # (batch_size, seq_len, vocab_size)

        return {
            "output": output_logits,
            "state": rnn_state,
        }

    def init_state(self, batch):
        """
        Initializes the hidden state of the GRU.

        Expects batch to include:
        - batch["input"]: a tensor of shape (batch_size, seq_len)
        """
        batch_size = batch["input"].size(0)
        return torch.zeros(
            self.num_gru_layers,
            batch_size,
            self.hidden_size,
            device=batch["input"].device,
        )
