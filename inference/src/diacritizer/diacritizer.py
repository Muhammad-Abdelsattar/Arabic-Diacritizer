import json
from pathlib import Path
from typing import Union, List
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

    def _diacritize_batch(self, arabic_texts: List[str]) -> List[str]:
        """Helper to diacritize a batch of clean, undiacritized Arabic strings."""
        if not arabic_texts:
            return []

        # Tokenize all text parts
        all_input_ids = [self.tokenizer.encode(text)[0] for text in arabic_texts]

        # Pad the batch to the length of the longest sequence
        max_len = max(len(ids) for ids in all_input_ids)
        padded_input_ids = np.zeros((len(all_input_ids), max_len), dtype=np.int64)
        for i, ids in enumerate(all_input_ids):
            padded_input_ids[i, : len(ids)] = ids

        # inference
        logits = self.predictor.predict(padded_input_ids)
        predicted_diac_ids = np.argmax(logits, axis=-1)

        # Decode the predictions
        diacritized_texts = []
        for i in range(len(all_input_ids)):
            # Decode only the original, unpadded length
            length = len(all_input_ids[i])
            decoded_text = self.tokenizer.decode(
                all_input_ids[i], predicted_diac_ids[i, :length].tolist()
            )
            diacritized_texts.append(decoded_text)

        return diacritized_texts

    def diacritize(self, text: str, postprocess: bool = True) -> str:
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

        # Remove all diacritics from the text
        text = TextCleaner.remove_diacritics(text)
        # Dissect the text into Arabic and non-Arabic segments.
        segments = ARABIC_LETTERS_REGEX.split(text)
        arabic_words = ARABIC_LETTERS_REGEX.findall(text)

        words_for_model = [TextCleaner.strip_diacritics(word) for word in arabic_words]

        # Run inference on the cleaned Arabic words.
        # The model is called only once with a batch of all Arabic words.
        if words_for_model:
            diacritized_words = self._diacritize_batch(words_for_model)
        else:
            diacritized_words = []

        # We interleave the original non-Arabic segments with the newly
        # diacritized Arabic words.
        result = []
        for i, segment in enumerate(segments):
            result.append(segment)
            if i < len(diacritized_words):
                result.append(diacritized_words[i])

        if postprocess:
            result = Postprocessor.postprocess("".join(result))

        return result
