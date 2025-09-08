import logging
from typing import Dict, Any
import torch.nn as nn

from .architectures.bilstm import BiLSTMDiacritizer

_LOGGER = logging.getLogger(__name__)

SUPPORTED_MODELS = {
    "bilstm": BiLSTMDiacritizer,
}


def build_model(
    config: Dict[str, Any],
    vocab_size: int,
    num_classes: int,
    pad_idx: int,
) -> nn.Module:
    """
    Builds a neural network model from a configuration.

    This factory function selects a model architecture based on the 'name' key
    in the configuration and initializes it with the provided parameters.

    Args:
        config: The architecture configuration dictionary. Must contain a 'name' key.
            Example: {"name": "bilstm", "embedding_dim": 128, ...}
        vocab_size: The total number of unique characters in the input vocabulary.
        num_classes: The total number of unique diacritic labels in the output.
        pad_idx: The index used for padding in the vocabulary.

    Returns:
        An instantiated PyTorch model (nn.Module).

    Raises:
        ValueError: If the model name specified in the config is not supported.
    """
    config = config.copy()
    model_name = config.pop("name", None)

    if not model_name:
        raise ValueError("Model configuration must include a 'name' key.")

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model name '{model_name}'. "
            f"Supported models are: {list(SUPPORTED_MODELS.keys())}"
        )

    model_class = SUPPORTED_MODELS[model_name]
    _LOGGER.info(f"Building '{model_name}' model architecture...")

    # Pass both static (from config) and dynamic (from data) parameters
    instance = model_class(
        vocab_size=vocab_size,
        num_classes=num_classes,
        pad_idx=pad_idx,
        **config,
    )
    _LOGGER.info(f"Successfully built '{model_name}' model.")
    return instance
