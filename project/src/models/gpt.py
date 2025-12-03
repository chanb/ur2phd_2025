import torch

from torch import nn


class GPTBlock(nn.Module):
    """ A GPT-style block. """
    def __init__(self, embed_dim, num_heads, widening_factor, dropout_p=0.1):
        super(GPTBlock, self).__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout_p,
            batch_first=True,
        )
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.dense_1 = nn.Linear(embed_dim, embed_dim * widening_factor)
        self.dense_2 = nn.Linear(embed_dim * widening_factor, embed_dim)
        self.gelu = nn.GELU()

    def forward(self, input):
        normed_input = self.ln_1(input)
        mask = nn.Transformer.generate_square_subsequent_mask(
            sz=input.shape[1],
            device=input.device,
        )
        att_out, attn_weights = self.attention(
            normed_input,
            normed_input,
            normed_input,
            is_causal=True,
            attn_mask=mask,
            average_attn_weights=False,
        )
        input = att_out + input
        normed_input = self.gelu(self.dense_1(self.ln_2(input)))
        out = input + self.dense_2(normed_input)
        return out, attn_weights
    

class DecoderTransformer(nn.Module):
    """
    A transformer decoder.
    """
    def __init__(
        self,
        vocab_size: int,
        num_blocks,
        embed_dim,
        num_heads,
        pe,
        widening_factor=4,
        dropout_p=0.1,
    ):
        super(DecoderTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pe = pe
        gpt_blocks = []
        ln_layers = []
        for block_i in range(num_blocks):
            gpt_blocks.append(
                GPTBlock(
                    embed_dim,
                    num_heads,
                    widening_factor,
                    dropout_p,
                )
            )
            ln_layers.append(
                nn.LayerNorm(embed_dim)
            )
        self.gpt_blocks = nn.ModuleList(gpt_blocks)
        self.ln_layers = nn.ModuleList(ln_layers)
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, batch):
        """
        Predicts next tokens given input sequences, i.e., p(x_t | x_1, ..., x_{t-1}).

        Expects batch to include:
        - batch["input"]: a tensor of shape (batch_size, seq_len)
        """

        input = batch["input"]
        state = {}
        if "state" in batch:
            context = batch["state"]["context"]
            timestep_i = batch["state"]["timestep_i"]
            input = torch.cat((context, input), dim=-1)
            state = {
                "context": input,
                "timestep_i": timestep_i + 1,
            }

        out = self.embedding(input)
        out = self.pe(out)
        attn_weights = []
        for gpt_layer, ln_layer in zip(
            self.gpt_blocks,
            self.ln_layers,
        ):
            out, attn_w = gpt_layer(out)
            out = ln_layer(out)
            attn_weights.append(attn_w)
        out = self.out(out)
        return {
            "output": out,
            "state": state,
            "attn_weights": attn_weights,
        }

    def init_state(self, batch):
        """
        Initializes the hidden state of the GPT.
        Here a "hidden state" consists of the previously-seen context and the current timestep index.

        Expects batch to include:
        - batch["input"]: a tensor of shape (batch_size, seq_len)
        """
        batch_size = batch["input"].size(0)
        return {
            "context": torch.zeros(
                (batch_size, 0),
                device=batch["input"].device,
            ).long(),
            "timestep_i": 0
        }
