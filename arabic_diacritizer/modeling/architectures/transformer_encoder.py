import math
import torch
import torch.nn as nn
from ..model_factory import register_model


class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


@register_model("transformer_encoder")
class TransformerEncoderDiacritizer(nn.Module):
    """
    Transformer Encoder architecture for character-level Arabic diacritization.

    - Input: character IDs (LongTensor, shape: [batch, seq_len])
    - Output: logits for diacritic classes (FloatTensor, shape: [batch, seq_len, num_classes])
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.2,
        pad_idx: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.pad_idx = pad_idx

        # 1. Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.d_model = d_model

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 3. Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # 4. Final projection to diacritic classes
        self.fc = nn.Linear(d_model, num_classes)

    def _generate_padding_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Creates a boolean mask for padding tokens."""
        return src == self.pad_idx

    def forward(self, x: torch.Tensor, lengths=None):
        """
        Args:
            x: LongTensor (batch_size, seq_len) - character IDs
            lengths: Not used by the Transformer, but kept for API consistency.

        Returns:
            logits: FloatTensor (batch_size, seq_len, num_classes)
        """
        # (B, L) -> (B, L, D_MODEL)
        embedded = self.embedding(x) * math.sqrt(self.d_model)

        # The nn.TransformerEncoderLayer expects (L, B, E) if batch_first=False
        # but we use batch_first=True, so input is (B, L, E).
        # We add positional encoding after embedding.
        pos_encoded = self.pos_encoder(embedded.permute(1, 0, 2)).permute(1, 0, 2)

        # Create padding mask: (B, L)
        padding_mask = self._generate_padding_mask(x)

        # Pass through transformer encoder
        transformer_out = self.transformer_encoder(
            pos_encoded, src_key_padding_mask=padding_mask
        )

        # (B, L, D_MODEL) -> (B, L, num_classes)
        logits = self.fc(transformer_out)
        return logits
