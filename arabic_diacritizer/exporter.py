import json
import logging
from pathlib import Path

import torch
from .modeling import ModelingOrchestrator
from .data import CharTokenizer

_LOGGER = logging.getLogger(__name__)


class Exporter:
    """
    Handles the conversion of a pre-built LightningModule into a
    deployment-ready ONNX bundle.
    """

    def export(self, lightning_module: ModelingOrchestrator, output_dir: str):
        """
        Exports the given model and its tokenizer to an ONNX bundle.

        Args:
            lightning_module: The fully instantiated and trained ModelingOrchestrator object.
            output_dir: Path to the directory where the exported bundle will be saved.
        """
        _LOGGER.info(f"Starting export process for the provided LightningModule.")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. Extract artifacts directly from the provided module
        core_model = lightning_module.model
        core_model.eval()

        tokenizer = getattr(lightning_module, "tokenizer", None)
        if not isinstance(tokenizer, CharTokenizer):
            raise AttributeError(
                "The provided LightningModule does not have a valid 'tokenizer' attribute. "
                "Ensure the tokenizer is attached before training."
            )
        _LOGGER.info("Successfully extracted model and tokenizer from the module.")

        # 2. Export the core model to ONNX with dynamic axes
        self._export_core_model_to_onnx(core_model, output_path)

        # 3. Save the tokenizer vocabulary
        self._save_vocabularies(tokenizer, output_path)

        _LOGGER.info(f"> Export successful. Bundle saved to: {output_path.resolve()}")

    def _export_core_model_to_onnx(self, model: torch.nn.Module, output_path: Path):
        """Exports the nn.Module to a fully dynamic ONNX model."""
        onnx_path = output_path / "model.onnx"
        dummy_input = torch.randint(low=0, high=100, size=(1, 10), dtype=torch.long)

        _LOGGER.info(f"> Exporting model to ONNX format at {onnx_path}...")
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            dynamo=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 1: "sequence_length"},
                "output": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=12,
        )
        _LOGGER.info("ONNX export completed with dynamic axes.")

    def _save_vocabularies(self, tokenizer: CharTokenizer, output_path: Path):
        """Saves character-to-ID and ID-to-diacritic mappings."""
        vocab_data = {
            "char2id": tokenizer.char2id,
            "id2diacritic": tokenizer.id2diacritic,
        }
        vocab_path = output_path / "vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        _LOGGER.info(f"Vocabularies saved to {vocab_path}")
