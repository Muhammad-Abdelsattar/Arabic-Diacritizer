import json
from pathlib import Path
from typing import Union
import numpy as np

from arabic_diacritizer_common import CharTokenizer, TextSegmenter
from .predictor import OnnxPredictor
from .hub_manager import resolve_model_path, DEFAULT_HUB_REPO_ID


class Diacritizer:
    def __init__(
        self,
        model_identifier: str = None,
        size: str = "medium",
        revision: str = "main",
        force_sync: bool = False,
        use_gpu: bool = False,
    ):
        """
        Initializes the Diacritizer by loading the model and tokenizer.

        The model can be loaded from a local directory or downloaded automatically
        from the Hugging Face Hub.

        Args:
            model_identifier (str, optional): The identifier for the model.
                - Can be a path to a local directory.
                - Can be a repository ID on the Hugging Face Hub (e.g., "your-name/your-repo").
                If None, defaults to the official pre-trained model repository.
            size (str): The model size ('small', 'medium', 'large'). Defaults to "medium".
            revision (str): A specific model version from the Hub (tag, branch, or commit). Defaults to "main".
            force_sync (bool): If True, forces a re-download from the Hub. Defaults to False.
            use_gpu (bool): If True, attempts to use CUDA for inference. Defaults to False.
        """
        self.max_length = -1

        repo_to_resolve = model_identifier or DEFAULT_HUB_REPO_ID

        onnx_path, vocab_path = resolve_model_path(
            model_identifier=repo_to_resolve,
            size=size,
            revision=revision,
            force_sync=force_sync,
        )

        self.predictor = OnnxPredictor(onnx_path, use_gpu)

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
