import logging
from typing import Dict, Any, Callable
import torch.nn as nn

_LOGGER = logging.getLogger(__name__)

# Create a registry (a simple dictionary) to hold model builders
MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str):
    """
    A decorator to register a new model architecture in the MODEL_REGISTRY.

    Args:
        name (str): The name to associate with the model class.
    """

    def wrapper(cls):
        if name in MODEL_REGISTRY:
            _LOGGER.warning(f"Model '{name}' is already registered. Overwriting.")
        MODEL_REGISTRY[name] = cls
        return cls

    return wrapper


def build_model(
    config: Dict[str, Any],
    vocab_size: int,
    num_classes: int,
    pad_idx: int,
) -> nn.Module:
    """
    Builds a neural network model from a configuration using the registry.

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

    # Look up the model class in the registry instead of a hardcoded dict.
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model name '{model_name}'. "
            f"Supported models are: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[model_name]
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
