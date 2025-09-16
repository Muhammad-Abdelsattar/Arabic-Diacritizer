import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn  # Explicitly import nn for clarity
from .data import CharTokenizer
from .data.preprocessing.constants import ARABIC_LETTERS

_LOGGER = logging.getLogger(__name__)


def export_for_inference(
    model: nn.Module,
    tokenizer: CharTokenizer,
    output_dir: str,
    dummy_input_length: int = 10,
    onnx_opset_version: int = 12,
    use_torch_dynamo: bool = False,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
):
    """
    Exports a trained PyTorch model and its tokenizer to a deployment-ready ONNX bundle
    and associated vocabulary artifacts.

    Args:
        model: The fully instantiated and trained torch.nn.Module object.
               It should be in evaluation mode (model.eval()).
        tokenizer: The CharTokenizer object associated with the model.
        output_dir: Path to the directory where the exported bundle will be saved.
        dummy_input_length: The sequence length for the dummy input used for ONNX tracing.
                            Should be representative but dynamic_axes will handle others.
        onnx_opset_version: The ONNX opset version to use for export.
        use_torch_dynamo: If True, uses torch.dynamo for ONNX export, potentially
                          optimizing the graph. Set to False for broader compatibility.
        input_names: List of names for the ONNX input nodes.
        output_names: List of names for the ONNX output nodes.
        dynamic_axes: Dictionary specifying dynamic axes for ONNX inputs/outputs.
                      If None, default dynamic axes for batch_size and sequence_length are used.
    """
    _LOGGER.info("Starting export process for the provided PyTorch model.")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.eval()  # Ensure model is in evaluation mode

    if not isinstance(tokenizer, CharTokenizer):
        raise TypeError(
            "The provided 'tokenizer' must be an instance of CharTokenizer."
        )

    _LOGGER.info("Successfully received model and tokenizer.")

    # Export the core model to ONNX with dynamic axes
    _export_core_model_to_onnx(
        model=model,
        output_path=output_path,
        dummy_input_length=dummy_input_length,
        opset_version=onnx_opset_version,
        use_dynamo=use_torch_dynamo,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    # Save the tokenizer vocabulary
    _save_vocabularies(tokenizer, output_path)

    _LOGGER.info(f"> Export successful. Bundle saved to: {output_path.resolve()}")


def _export_core_model_to_onnx(
    model: nn.Module,
    output_path: Path,
    dummy_input_length: int,
    opset_version: int,
    use_dynamo: bool,
    input_names: Optional[List[str]],
    output_names: Optional[List[str]],
    dynamic_axes: Optional[Dict[str, Dict[int, str]]],
):
    """Exports the nn.Module to a fully dynamic ONNX model."""
    onnx_path = output_path / "model.onnx"
    # Create dummy input. Using a device-agnostic tensor if possible, but
    # for tracing, it's often run on CPU.
    dummy_input = torch.randint(
        low=0, high=40, size=(1, dummy_input_length), dtype=torch.long
    )

    _LOGGER.info(f"> Exporting model to ONNX format at {onnx_path}...")

    # Default dynamic axes if not provided
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size", 1: "sequence_length"},
        }
    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        dynamo=use_dynamo,  # Configurable
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
    )
    _LOGGER.info("ONNX export completed with dynamic axes.")


def _save_vocabularies(tokenizer: CharTokenizer, output_path: Path):
    """Saves character-to-ID and ID-to-diacritic mappings."""
    vocab_data = {
        "char2id": tokenizer.char2id,
        "id2char": tokenizer.id2char,
        "diacritic2id": tokenizer.diacritic2id,
        "id2diacritic": tokenizer.id2diacritic,
        "pad_idx": tokenizer.char2id.get("<PAD>", 0),  # Explicitly save pad_idx
        "unk_idx": tokenizer.char2id.get("<UNK>", 1),  # Explicitly save unk_idx
        "arabic_letters": sorted(list(ARABIC_LETTERS)),
    }
    vocab_path = output_path / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    _LOGGER.info(f"Vocabularies saved to {vocab_path}")
