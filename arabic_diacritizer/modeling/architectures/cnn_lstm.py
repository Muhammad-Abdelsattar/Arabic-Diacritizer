import torch
import torch.nn as nn
from typing import Optional

from ..model_factory import register_model


@register_model("cnn_bilstm")
class CNNBiLSTMDiacritizer(nn.Module):
    """
    A configurable CNN-BiLSTM architecture for character-level Arabic diacritization.

    - A stack of 1D CNNs learns local character n-gram features.
    - A BiLSTM models long-range sequential dependencies on these features.
    - A final Linear layer produces logits for a standard CrossEntropyLoss.
    """

    def __init__(
        self,
        # Essential Parameters (from factory)
        vocab_size: int,
        num_classes: int,
        pad_idx: int = 0,
        # Configurable Model Hyperparameters
        embedding_dim: int = 128,
        # CNN Layers
        cnn_num_layers: int = 3,
        cnn_hidden_dim: int = 128,
        cnn_kernel_size: int = 3,
        cnn_dropout: float = 0.25,
        # LSTM Layers
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.25,
        **kwargs
    ):
        super().__init__()
        self.pad_idx = pad_idx

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # Configurable CNN Block
        cnn_layers = []
        # The first CNN layer takes the embedding dimension as input
        input_channels = embedding_dim
        for _ in range(cnn_num_layers):
            cnn_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=input_channels,
                        out_channels=cnn_hidden_dim,
                        kernel_size=cnn_kernel_size,
                        padding="same",  # That's cirtical
                    ),
                    nn.BatchNorm1d(cnn_hidden_dim),
                    nn.GELU(),  # Or maybe relu ?
                    nn.Dropout(cnn_dropout),
                )
            )
            # Subsequent layers take the previous layer's output dimension as input
            input_channels = cnn_hidden_dim
        self.cnn_block = nn.Sequential(*cnn_layers)

        # BiLSTM Layer
        self.bilstm = nn.LSTM(
            # Input to LSTM is the feature vector from the final CNN layer
            input_size=cnn_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        # Projects the concatenated BiLSTM hidden states to the number of diacritic classes
        self.fc = nn.Linear(lstm_hidden_dim * 2, num_classes)

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Standard forward pass that returns logits.

        Args:
            x: LongTensor (batch_size, seq_len) - character IDs
            lengths: Optional LongTensor (batch_size,) - actual sequence lengths

        Returns:
            logits: FloatTensor (batch_size, seq_len, num_classes)
        """
        # (B, L) to (B, L, E_dim)
        embedded = self.embedding(x)

        # CNNs expect (B, Channels, L), so we permute the dimensions
        # (B, L, E_dim) to (B, E_dim, L)
        cnn_input = embedded.permute(0, 2, 1)

        # Pass through the stack of CNN layers
        cnn_output = self.cnn_block(cnn_input)  # (B, cnn_hidden_dim, L)

        # (B, cnn_hidden_dim, L) to (B, L, cnn_hidden_dim)
        lstm_input = cnn_output.permute(0, 2, 1)

        # Pack sequence for efficient LSTM processing (handles padding correctly)
        if lengths is not None:
            lengths_sorted, idx_sort = torch.sort(lengths, descending=True)
            lstm_input_sorted = lstm_input.index_select(0, idx_sort)

            packed = nn.utils.rnn.pack_padded_sequence(
                lstm_input_sorted,
                lengths_sorted.cpu(),
                batch_first=True,
                enforce_sorted=True,
            )
            packed_out, _ = self.bilstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

            # Restore to original batch order
            _, idx_unsort = torch.sort(idx_sort)
            lstm_out = lstm_out.index_select(0, idx_unsort)
        else:
            # Process without packing if lengths are not provided
            lstm_out, _ = self.bilstm(lstm_input)

        logits = self.fc(lstm_out)
        return logits
