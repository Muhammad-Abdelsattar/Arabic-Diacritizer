import json
from pathlib import Path
from typing import Union, List, Optional
import numpy as np
from arabic_diacritizer_common import (
    CharTokenizer,
    TextSegmenter,
    Postprocessor,
    DiacriticValidator,
    ARABIC_LETTERS_REGEX,
    TextCleaner,
    DIACRITIC_CHARS,
)
from .predictor import OnnxPredictor
from .hub_manager import resolve_model_path, DEFAULT_HUB_REPO_ID


class Diacritizer:
    def __init__(
        self,
        model_identifier: Optional[str] = None,
        architecture: str = "bilstm",
        size: str = "medium",
        revision: str = "main",
        force_sync: bool = False,
        use_gpu: bool = False,
    ):
        """
        Initializes the Diacritizer by loading the model and tokenizer.

        Args:
            model_identifier (str, optional): The identifier for the model. Can be a
                local path or a Hugging Face Hub repo ID. Defaults to the official repo.
            architecture (str): The model architecture ('bilstm', 'bigru', etc.).
                Defaults to "bilstm".
            size (str): The model size ('small', 'medium'). Defaults to "medium".
            revision (str): A specific model version from the Hub. Defaults to "main".
            force_sync (bool): If True, forces a re-download. Defaults to False.
            use_gpu (bool): If True, attempts to use CUDA. Defaults to False.
        """
        self.max_length = -1

        repo_to_resolve = model_identifier or DEFAULT_HUB_REPO_ID

        # Pass the new 'architecture' parameter to the resolver function
        onnx_path, vocab_path = resolve_model_path(
            model_identifier=repo_to_resolve,
            architecture=architecture,  # MODIFIED
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

    def _diacritize_sentence(self, text: str) -> str:
        """Helper to diacritize an arabic sentence string."""
        if not text.strip():
            return ""

        input_ids, diacritic_ids = self.tokenizer.encode(text)

        text_list = list(TextCleaner.remove_diacritics(text))

        original_len = len(input_ids)
        if original_len == 0:
            return ""

        input_chars = np.array(input_ids).astype(np.int64).reshape(1, -1)
        no_diacritic_id = self.tokenizer.diacritic2id.get("", 0)
        input_hints = np.full_like(
            input_chars, fill_value=no_diacritic_id, dtype=np.int64
        )
        # inference
        logits = self.predictor.predict(input_ids=input_chars, hints=input_hints)
        predicted_diac_ids = np.argmax(logits, axis=-1)

        # Decode the predictions
        return self.tokenizer.decode_inference(
            text_list, predicted_diac_ids[0].tolist()
        )

    def diacritize(
        self, text: Union[str, List[str]], postprocess: bool = True
    ) -> List[str]:
        """
        Diacritizes text while preserving non-Arabic characters and structure.

        This method dissects the input text into Arabic and non-Arabic segments.
        It processes only the Arabic segments and then reassembles the string,
        maintaining the original order and content of all non-Arabic parts.

        Any existing diacritics in the Arabic segments are stripped before
        being processed by the model to ensure a consistent output.

        Args:
            text (str): The input text.

        Returns:
            The diacritized string.
        """
        if not text:
            return ""

        if isinstance(text, str):
            # To handle a single string input gracefully
            text_or_list = [text]

        else:
            # To handle a list of strings input gracefully
            text_or_list = text

        if not isinstance(text_or_list, list):
            raise TypeError("Input must be a string or a list of strings.")

        diacritized_list = [self._diacritize_sentence(s) for s in text_or_list]

        if postprocess:
            for i, diacritized_sentence in enumerate(diacritized_list):
                diacritized_list[i] = Postprocessor.postprocess(diacritized_sentence)

        if isinstance(text, str):
            return diacritized_list[0]

        return diacritized_list
