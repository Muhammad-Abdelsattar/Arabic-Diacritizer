import torch
import torch.nn as nn

from ..model_factory import register_model


# apply the decorator to register this class under the name "bilstm"
@register_model("bilstm")
class BiLSTMDiacritizer(nn.Module):
    """
    BiLSTM architecture for character-level Arabic diacritization.

    - Input: character IDs (LongTensor, shape: [batch, seq_len])
    - Output: logits for diacritic classes (FloatTensor, shape: [batch, seq_len, num_classes])
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
    ):
        super().__init__()

        # Embedding for characters
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_idx
        )

        # BiLSTM Encoder
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        # Dropout on LSTM outputs
        self.dropout = nn.Dropout(dropout)

        # Final projection to diacritic classes
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths=None):
        """
        Args:
            x: LongTensor (batch_size, seq_len) - character IDs
            lengths: Optional LongTensor (batch_size,) - actual sequence lengths

        Returns:
            logits: FloatTensor (batch_size, seq_len, num_classes)
        """
        # (B, L, E)
        embedded = self.embedding(x)

        if lengths is not None:
            # Sort by length for packing (required by PyTorch LSTM)
            lengths_sorted, idx_sort = torch.sort(lengths, descending=True)
            embedded = embedded.index_select(0, idx_sort)

            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
            )
            packed_out, _ = self.bilstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

            # Restore to original order
            _, idx_unsort = torch.sort(idx_sort)
            lstm_out = lstm_out.index_select(0, idx_unsort)

        else:
            lstm_out, _ = self.bilstm(embedded)

        dropped = self.dropout(lstm_out)

        # (B, L, num_classes)
        logits = self.fc(dropped)
        return logits
