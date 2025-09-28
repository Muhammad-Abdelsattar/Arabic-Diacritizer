import torch
import torch.nn as nn

from ..model_factory import register_model


@register_model("bilstm")
class BiLSTMDiacritizer(nn.Module):
    """
    A BiLSTM architecture that accepts an additional "diacritic hint" input.

    - Input 1: character IDs (LongTensor)
    - Input 2: hint diacritic IDs (LongTensor)
    - Output: logits for diacritic classes (FloatTensor)
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
        **kwargs
    ):
        super().__init__()

        #  Standard embedding for characters
        self.char_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_idx
        )

        #  A second embedding layer for the diacritic hints.
        # It must have the same dimensions as the character embedding.
        self.hint_embedding = nn.Embedding(
            num_embeddings=num_classes, embedding_dim=embedding_dim, padding_idx=pad_idx
        )

        # The BiLSTM now processes the combined information.
        # Since we are adding the embeddings, the input_size remains embedding_dim.
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, hints, lengths=None):
        """
        The forward pass now accepts `x` (characters) and `hints`.
        """
        # (B, L, E)
        char_embedded = self.char_embedding(x)
        hint_embedded = self.hint_embedding(hints)

        # Combine the embeddings. Simple addition is a powerful and effective technique.
        embedded = char_embedded + hint_embedded

        if lengths is not None:
            lengths_sorted, idx_sort = torch.sort(lengths, descending=True)
            embedded = embedded.index_select(0, idx_sort)

            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
            )
            packed_out, _ = self.bilstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

            _, idx_unsort = torch.sort(idx_sort)
            lstm_out = lstm_out.index_select(0, idx_unsort)
        else:
            lstm_out, _ = self.bilstm(embedded)

        dropped = self.dropout(lstm_out)
        logits = self.fc(dropped)
        return logits
