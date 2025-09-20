import json
from pathlib import Path
from typing import Union
import numpy as np

from arabic_diacritizer_common import CharTokenizer, TextSegmenter
from .predictor import OnnxPredictor
from .exceptions import ModelNotFound, InvalidInputError


class Diacritizer:
    def __init__(self, model_dir: Union[str, Path], use_gpu: bool = False):
        """
        Initializes the Diacritizer by loading the model and tokenizer.

        Args:
            model_dir: Path to the directory containing model.onnx, vocab.json, and config.json.
            use_gpu: If True, will attempt to use CUDA for inference. Defaults to False.
        """
        model_dir = Path(model_dir)
        onnx_path = model_dir / "model.onnx"
        vocab_path = model_dir / "vocab.json"
        config_path = model_dir / "config.json"

        if not all([onnx_path.exists(), vocab_path.exists()]):
            raise ModelNotFound(
                "Model directory must contain 'model.onnx', 'vocab.json'."
            )

        # load metadata
        # config = json.loads(config_path.read_text("utf-8"))
        # self.max_length = config.get("max_length", -1)
        self.max_length = -1

        self.predictor = OnnxPredictor(onnx_path, use_gpu)

        # load tokenizer from the common package
        vocab_data = json.loads(vocab_path.read_text(encoding="utf-8"))
        self.tokenizer = CharTokenizer(
            char2id=vocab_data["char2id"],
            diacritic2id=vocab_data["diacritic2id"],
        )
        self.segmenter = TextSegmenter()

    def _diacritize_chunk(self, chunk: str) -> str:
        """Helper to diacritize a single, short chunk of text."""
        # tokenize the input text chunk
        input_ids, _ = self.tokenizer.encode(chunk)
        if not input_ids:
            return ""

        input_tensor = np.array(input_ids, dtype=np.int64).reshape(1, -1)
        logits = self.predictor.predict(input_tensor)

        # decode the predictions
        predicted_diacritic_ids = np.argmax(logits, axis=-1)[0]
        return self.tokenizer.decode(input_ids, predicted_diacritic_ids.tolist())

    def diacritize(self, text: str) -> str:
        """
        Diacritizes a string of Arabic text, handling long inputs by segmentation.

        Args:
            text: The input text (without diacritics).

        Returns:
            The diacritized text.

        Raises:
            InvalidInputError: If the input text is not a valid string.
        """
        if not text.strip():
            return ""

        # segment text into chunks the model can handle
        segments = self.segmenter.segment_sentences(self.max_length, text)

        # diacritize each segment
        diacritized_segments = [self._diacritize_chunk(seg) for seg in segments]

        return " ".join(filter(None, diacritized_segments))
