import json
import logging
from pathlib import Path

import torch
from .modeling import ModelingOrchestrator

_LOGGER = logging.getLogger(__name__)


class Exporter:
    """
    Handles the conversion of a self-contained PyTorch Lightning checkpoint
    into a deployment-ready ONNX bundle.
    """

    def export(self, ckpt_path: str, output_dir: str):
        """
        Exports the model to ONNX format and saves all necessary artifacts.

        Args:
            ckpt_path: Path to the .ckpt file of the trained model.
            output_dir: Path to the directory where the exported bundle will be saved.
        """
        _LOGGER.info(
            f"Starting model export from self-contained checkpoint: {ckpt_path}"
        )
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load the LightningModule
        _LOGGER.info("Loading model and artifacts from checkpoint...")
        lightning_module = ModelingOrchestrator.load_from_checkpoint(
            ckpt_path, map_location="cpu"
        )

        # Extract all required artifacts directly from the loaded module
        core_model = lightning_module.model
        core_model.eval()

        # Retrieve tokenizer and max_length from the checkpoint
        tokenizer = getattr(lightning_module, "tokenizer", None)

        if tokenizer is None:
            raise RuntimeError(
                "Could not find a 'tokenizer' attached to the LightningModule. Please retrain and save the checkpoint."
            )

        # Export the core model to ONNX with dynamic axes
        self._export_core_model_to_onnx(core_model, max_length, output_path)

        # Save tokenizer vocabularies
        self._save_vocabularies(tokenizer, output_path)


        _LOGGER.info(f"✅ Export successful. Bundle saved to: {output_path.resolve()}")

    def _export_core_model_to_onnx(
        self, model: torch.nn.Module, max_length: int, output_path: Path
    ):
        onnx_path = output_path / "model.onnx"
        dummy_input = torch.randint(
            low=0, high=100, size=(1, max_length), dtype=torch.long
        )

        _LOGGER.info(f"Exporting model to ONNX format at {onnx_path}...")
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
        _LOGGER.info("ONNX export completed.")

    def _save_vocabularies(self, tokenizer, output_path: Path):
        vocab_data = {
            "char2id": tokenizer.char2id,
            "id2diacritic": tokenizer.id2diacritic,
        }
        vocab_path = output_path / "vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        _LOGGER.info(f"Vocabularies saved to {vocab_path}")
